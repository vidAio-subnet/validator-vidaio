"""Autoupdater trigger semantics — every rule in the service module docstring.

No fakes on the update path: the update command is a real subprocess (a tiny
python -c program writing evidence files), the http source is served by a real
ASGI app over httpx's ASGITransport (plus a MockTransport for the failure
shapes). `poll_once()` is driven directly, so every test is one deterministic
policy step — no loops, no sleeps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from vidaio.autoupdater import (
    TARGET_RUNTIME_DIGEST_ENV,
    TARGET_SOURCE_DIGEST_ENV,
    TARGET_STAGED_ROOT_ENV,
    TARGET_VERSION_ENV,
    Autoupdater,
    AutoupdaterConfig,
)
from vidaio.autoupdater.integrity import (
    RUNTIME_DIRS,
    source_digest,
    write_runtime_manifest,
)


def make_raw(tmp_path: Path, **autoupdater: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "version_file": str(tmp_path / "VERSION"),
        "state_file": str(tmp_path / "state.json"),
        "ci_pass_marker": str(tmp_path / "ci-pass"),
        "source_root": str(tmp_path),
        "runtime_manifest_file": str(tmp_path / "runtime-release-manifest.json"),
        "update_retry_delay_seconds": 0.0,
        "update_timeout_seconds": 30.0,
        "metrics_port": 0,
        **autoupdater,
    }
    return {
        "core": {"data_dir": str(tmp_path / "data"), "metrics_port": 0},
        "autoupdater": settings,
    }


def write_version(tmp_path: Path, version: str) -> None:
    # Minimal but complete shipped-runtime shape for manifest verification.
    for directory in RUNTIME_DIRS:
        (tmp_path / directory).mkdir(exist_ok=True)
    for filename, payload in (
        ("pyproject.toml", "[project]\nname='vidaio-test-artifact'\n"),
        ("uv.lock", "version = 1\n"),
    ):
        path = tmp_path / filename
        if not path.exists():
            path.write_text(payload, encoding="utf-8")
    (tmp_path / "VERSION").write_text(version + "\n", encoding="utf-8")


def write_marker(tmp_path: Path, version: str) -> None:
    """Bind full CI source plus the exact staged runtime artifact."""
    manifest = write_runtime_manifest(
        tmp_path / "runtime-release-manifest.json",
        source_root=tmp_path,
        runtime_root=tmp_path,
    )
    manifest_path = tmp_path / "runtime-release-manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "ci-pass").write_text(
        f"{version}\nsource-sha256 {manifest['source_sha256']}\n"
        f"runtime-sha256 {manifest['runtime_sha256']}\n"
        f"manifest-sha256 {manifest_sha}\n"
        "ci-pass 2026-08-21T00:00:00Z\n"
    )


#: A REAL update command: records the env-carried target version into out.txt.
def apply_cmd(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import os, sys, pathlib;"
            f" pathlib.Path(sys.argv[1]).write_text(os.environ['{TARGET_VERSION_ENV}'])"
        ),
        str(tmp_path / "out.txt"),
    ]


def identity_cmd(tmp_path: Path) -> list[str]:
    """Record every verified identity field passed to the argv activator."""
    names = (
        TARGET_VERSION_ENV,
        TARGET_SOURCE_DIGEST_ENV,
        TARGET_RUNTIME_DIGEST_ENV,
        TARGET_STAGED_ROOT_ENV,
    )
    return [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, sys;"
            " pathlib.Path(sys.argv[1]).write_text(json.dumps("
            f"{{name: os.environ[name] for name in {names!r}}}"
            "))"
        ),
        str(tmp_path / "identity.json"),
    ]


#: A REAL failing command: appends one attempt line, then exits 1 — unless a
#: "fixed" file exists, in which case it behaves like apply_cmd.
def flaky_cmd(tmp_path: Path) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import os, sys, pathlib;"
            " base = pathlib.Path(sys.argv[1]);"
            " (base / 'attempts.log').open('a').write('attempt\\n');"
            " ok = (base / 'fixed').exists();"
            f" ok and (base / 'out.txt').write_text(os.environ['{TARGET_VERSION_ENV}']);"
            " sys.exit(0 if ok else 1)"
        ),
        str(tmp_path),
    ]


# --- config -----------------------------------------------------------------------


def test_http_source_requires_a_url() -> None:
    with pytest.raises(ValidationError, match="version_url is empty"):
        AutoupdaterConfig(version_source="http")
    AutoupdaterConfig(version_source="http", version_url="http://updates/version")


# --- baseline: never on first sight ----------------------------------------------


async def test_first_sight_never_triggers(tmp_path: Path) -> None:
    """Fresh start, no state: the local VERSION is the baseline, nothing runs."""
    write_version(tmp_path, "0.2.0")
    svc = Autoupdater(make_raw(tmp_path, update_command=apply_cmd(tmp_path)))
    assert svc.last_applied == "0.2.0"  # baseline = current VERSION at start
    await svc.poll_once()
    assert not (tmp_path / "out.txt").exists()
    assert svc.pending_version is None
    assert svc.m_version_changes._value.get() == 0


async def test_first_observation_adopted_when_no_local_version(tmp_path: Path) -> None:
    """No state AND no version file: the first observed version becomes the
    baseline — persisted, never acted on."""
    app = FastAPI()

    @app.get("/version")
    async def version() -> PlainTextResponse:
        return PlainTextResponse("0.5.0\n")

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://updates.local"
    )
    svc = Autoupdater(
        make_raw(
            tmp_path,
            version_source="http",
            version_url="http://updates.local/version",
            update_command=apply_cmd(tmp_path),
        ),
        http_client=client,
    )
    assert svc.last_applied is None
    await svc.poll_once()
    assert svc.last_applied == "0.5.0"
    assert not (tmp_path / "out.txt").exists()
    assert json.loads((tmp_path / "state.json").read_text())["last_applied"] == "0.5.0"


# --- change detection: file + http (ASGI) -----------------------------------------


async def test_file_change_detected_report_only(tmp_path: Path) -> None:
    """update_command []: the change is announced ONCE, surfaced on /health, and
    nothing is ever executed or recorded as applied."""
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path))  # report-only default
    await svc.poll_once()
    assert svc.pending_version is None

    write_version(tmp_path, "0.2.0")
    await svc.poll_once()
    assert svc.pending_version == "0.2.0"
    assert svc.last_applied == "0.1.0"  # nothing applied
    assert svc.m_version_changes._value.get() == 1
    assert svc.m_pending._value.get() == 1
    ok, payload = svc.health.health_payload()
    assert ok is False and payload["checks"]["update_pending"] is False

    await svc.poll_once()  # announced once, counted once
    assert svc.m_version_changes._value.get() == 1
    assert not (tmp_path / "state.json").exists()


async def test_http_change_detected_via_asgi_app(tmp_path: Path) -> None:
    state = {"version": "0.1.0"}
    app = FastAPI()

    @app.get("/version")
    async def version() -> PlainTextResponse:
        return PlainTextResponse(state["version"] + "\n")

    write_version(tmp_path, "0.1.0")  # the RUNNING version (baseline)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://updates.local"
    )
    svc = Autoupdater(
        make_raw(
            tmp_path, version_source="http", version_url="http://updates.local/version"
        ),
        http_client=client,
    )
    await svc.poll_once()
    assert svc.pending_version is None

    state["version"] = "0.2.0"
    await svc.poll_once()
    assert svc.pending_version == "0.2.0"
    assert svc.m_version_changes._value.get() == 1


async def test_pending_update_withdrawn_when_source_reverts_to_baseline(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path))
    write_version(tmp_path, "0.2.0")
    await svc.poll_once()
    assert svc.pending_version == "0.2.0"
    write_version(tmp_path, "0.1.0")  # the owner withdrew the push
    await svc.poll_once()
    assert svc.pending_version is None
    assert svc.health.health_payload()[0] is True


async def test_lateral_change_is_treated_as_an_update(tmp_path: Path) -> None:
    """Equal numeric spine, different string: applied like an upgrade, by rule."""
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path))
    write_version(tmp_path, "0.1.0-hotfix1")
    await svc.poll_once()
    assert svc.pending_version == "0.1.0-hotfix1"
    assert svc.m_downgrades_refused._value.get() == 0


# --- source failures --------------------------------------------------------------


async def test_missing_version_file_is_a_poll_failure_not_a_crash(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path))
    (tmp_path / "VERSION").unlink()
    await svc.poll_once()
    assert svc.m_poll_failures._value.get() == 1
    ok, payload = svc.health.health_payload()
    assert ok is False and payload["checks"]["version_source"] is False
    write_version(tmp_path, "0.1.0")  # source recovers -> health recovers
    await svc.poll_once()
    assert svc.health.health_payload()[1]["checks"]["version_source"] is True


async def test_http_non_200_is_a_poll_failure(tmp_path: Path) -> None:
    responses = [httpx.Response(500), httpx.Response(200, text="0.1.0\n")]
    transport = httpx.MockTransport(lambda request: responses.pop(0))
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(
        make_raw(
            tmp_path, version_source="http", version_url="http://updates.local/version"
        ),
        http_client=httpx.AsyncClient(transport=transport),
    )
    await svc.poll_once()
    assert svc.m_poll_failures._value.get() == 1
    assert svc.health.health_payload()[1]["checks"]["version_source"] is False
    await svc.poll_once()
    assert svc.health.health_payload()[1]["checks"]["version_source"] is True


# --- the ci-pass gate -------------------------------------------------------------


def test_release_digest_includes_production_env_template(tmp_path: Path) -> None:
    write_version(tmp_path, "0.1.0")
    before = source_digest(tmp_path)
    (tmp_path / ".env.example").write_text(
        "VIDAIO__CHAIN__MODE=bittensor\n", encoding="utf-8"
    )
    assert source_digest(tmp_path) != before


async def test_ci_gate_refuses_without_marker_then_opens(tmp_path: Path) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path, update_command=apply_cmd(tmp_path)))
    write_version(tmp_path, "0.2.0")

    await svc.poll_once()  # no marker at all
    assert not (tmp_path / "out.txt").exists()
    assert svc.m_ci_gate_refusals._value.get() == 1
    assert svc.pending_version == "0.2.0"

    write_marker(tmp_path, "0.1.9")  # a marker for the WRONG version
    await svc.poll_once()
    assert not (tmp_path / "out.txt").exists()
    assert svc.m_ci_gate_refusals._value.get() == 2

    write_marker(tmp_path, "0.2.0")  # the full gate ran green on the new version
    await svc.poll_once()
    assert (tmp_path / "out.txt").read_text() == "0.2.0"  # env-carried target
    assert svc.last_applied == "0.2.0"
    assert svc.pending_version is None
    assert svc.m_updates_applied._value.get() == 1
    assert svc.health.health_payload()[0] is True


async def test_ci_gate_refuses_stale_marker_after_source_changes(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path, update_command=apply_cmd(tmp_path)))
    write_version(tmp_path, "0.2.0")
    write_marker(tmp_path, "0.2.0")

    # Same VERSION, different release bytes: the prior green marker is stale.
    (tmp_path / "pyproject.toml").write_text("[project]\nname='changed'\n")
    await svc.poll_once()
    assert not (tmp_path / "out.txt").exists()
    assert svc.pending_version == "0.2.0"


async def test_ci_gate_refuses_stale_marker_after_source_only_changes(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path, update_command=apply_cmd(tmp_path)))
    write_version(tmp_path, "0.2.0")
    (tmp_path / "deploy").mkdir()
    release_entrypoint = tmp_path / "deploy" / "release.py"
    release_entrypoint.write_text("REVIEWED = True\n", encoding="utf-8")
    write_marker(tmp_path, "0.2.0")

    # Runtime bytes are unchanged, but this is no longer the complete source tree
    # that passed CI. Source identity is an enforced gate, not marker decoration.
    release_entrypoint.write_text("REVIEWED = False\n", encoding="utf-8")
    await svc.poll_once()

    assert not (tmp_path / "out.txt").exists()
    assert svc.pending_version == "0.2.0"
    assert svc.m_ci_gate_refusals._value.get() == 1


async def test_ci_gate_treats_runtime_read_race_as_a_closed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vidaio.autoupdater.service as service_module

    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path, update_command=apply_cmd(tmp_path)))
    write_version(tmp_path, "0.2.0")

    def unreadable(*_args: object, **_kwargs: object) -> object:
        raise OSError("staged runtime disappeared during verification")

    monkeypatch.setattr(service_module, "verify_ci_release", unreadable)
    await svc.poll_once()

    assert svc.pending_version == "0.2.0"
    assert not (tmp_path / "out.txt").exists()
    assert svc.m_ci_gate_refusals._value.get() == 1


async def test_verified_identity_is_passed_to_the_argv_activator(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(make_raw(tmp_path, update_command=identity_cmd(tmp_path)))
    write_version(tmp_path, "0.2.0")
    write_marker(tmp_path, "0.2.0")

    await svc.poll_once()

    manifest = json.loads(
        (tmp_path / "runtime-release-manifest.json").read_text(encoding="utf-8")
    )
    identity = json.loads((tmp_path / "identity.json").read_text(encoding="utf-8"))
    assert identity == {
        TARGET_VERSION_ENV: "0.2.0",
        TARGET_SOURCE_DIGEST_ENV: manifest["source_sha256"],
        TARGET_RUNTIME_DIGEST_ENV: manifest["runtime_sha256"],
        TARGET_STAGED_ROOT_ENV: str(tmp_path.resolve()),
    }


async def test_update_runs_without_marker_when_gate_disabled(tmp_path: Path) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(
        make_raw(tmp_path, update_command=apply_cmd(tmp_path), require_ci_pass=False)
    )
    write_version(tmp_path, "0.2.0")
    await svc.poll_once()
    assert (tmp_path / "out.txt").read_text() == "0.2.0"
    assert json.loads((tmp_path / "state.json").read_text())["last_applied"] == "0.2.0"


# --- update failure: bounded retry, then persistently unhealthy -------------------


async def test_failure_retries_then_flags_update_failed_persistently(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    svc = Autoupdater(
        make_raw(
            tmp_path,
            update_command=flaky_cmd(tmp_path),
            require_ci_pass=False,
            update_retry_attempts=2,
        )
    )
    write_version(tmp_path, "0.2.0")
    await svc.poll_once()
    # the retry budget was spent — exactly two real subprocess runs
    assert (tmp_path / "attempts.log").read_text().count("attempt") == 2
    assert svc.m_updates_failed._value.get() == 1
    ok, payload = svc.health.health_payload()
    assert ok is False and payload["checks"]["update_failed"] is False
    assert svc.last_applied == "0.1.0"  # nothing was recorded as applied

    await svc.poll_once()  # same version: NOT retried, still unhealthy
    assert (tmp_path / "attempts.log").read_text().count("attempt") == 2
    assert svc.health.health_payload()[1]["checks"]["update_failed"] is False

    # a LATER version may trigger again; success clears the flag
    (tmp_path / "fixed").touch()
    write_version(tmp_path, "0.3.0")
    await svc.poll_once()
    assert svc.last_applied == "0.3.0"
    assert (tmp_path / "out.txt").read_text() == "0.3.0"
    assert svc.health.health_payload()[0] is True


# --- downgrade refusal ------------------------------------------------------------


async def test_downgrade_refused_with_critical_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_version(tmp_path, "0.2.0")
    svc = Autoupdater(
        make_raw(tmp_path, update_command=apply_cmd(tmp_path), require_ci_pass=False)
    )
    write_version(tmp_path, "0.1.9")
    with caplog.at_level(logging.CRITICAL, logger="autoupdater"):
        await svc.poll_once()
    assert not (tmp_path / "out.txt").exists()
    assert svc.m_downgrades_refused._value.get() == 1
    assert svc.pending_version is None
    assert any(
        "REFUSING the downgrade" in r.getMessage()
        for r in caplog.records
        if r.levelno == logging.CRITICAL
    )
    await svc.poll_once()  # refused once per version, not re-logged/re-counted
    assert svc.m_downgrades_refused._value.get() == 1


async def test_allow_downgrade_applies_the_rollback(tmp_path: Path) -> None:
    write_version(tmp_path, "0.2.0")
    svc = Autoupdater(
        make_raw(
            tmp_path,
            update_command=apply_cmd(tmp_path),
            require_ci_pass=False,
            allow_downgrade=True,
        )
    )
    write_version(tmp_path, "0.1.9")
    await svc.poll_once()
    assert (tmp_path / "out.txt").read_text() == "0.1.9"
    assert svc.last_applied == "0.1.9"


# --- restart safety ---------------------------------------------------------------


async def test_state_persists_across_restart_and_never_retriggers(
    tmp_path: Path,
) -> None:
    write_version(tmp_path, "0.1.0")
    raw = make_raw(tmp_path, update_command=apply_cmd(tmp_path), require_ci_pass=False)
    first = Autoupdater(raw)
    write_version(tmp_path, "0.2.0")
    await first.poll_once()
    assert first.last_applied == "0.2.0"
    (tmp_path / "out.txt").unlink()

    # a fresh process: the baseline comes from the STATE file, not first sight
    second = Autoupdater(raw)
    assert second.last_applied == "0.2.0"
    await second.poll_once()
    assert not (tmp_path / "out.txt").exists()  # the applied update did not re-run
    assert second.m_updates_applied._value.get() == 0
    assert second.m_version_changes._value.get() == 0
