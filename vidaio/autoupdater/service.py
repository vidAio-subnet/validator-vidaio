"""Autoupdater service — watches a version source, triggers the fleet's own updater.

the design spec §20 CI/CD row calls for a pipeline that "gates deploys"; this service is
the deploy half of that loop, local-first: it polls a version source (the repo
VERSION file now, an HTTP endpoint when the fleet is remote), and on a version
CHANGE it runs the deployment's own update command — `git pull && systemctl
restart`, a Modal redeploy script, whatever the deployment owns. The autoupdater
NEVER edits code itself; with no command configured it is a pure reporter.

TRIGGER SEMANTICS (each one tested in tests/autoupdater):

* **Never on first sight.** The baseline is established WITHOUT triggering: the
  persisted state file if one exists, else the local `version_file` (what this
  deployment is running), else the first successfully observed source version.
  Only a change AGAINST that baseline triggers anything.
* **The ordering rule (semver-ish, deliberately small).** Only the leading
  dotted numeric spine orders: ``version_key("v1.2.10-rc1") == (1, 2, 10)``,
  shorter spines are zero-padded (``1.2 == 1.2.0``), and ``0.1.10 > 0.1.9``
  numerically. A spine strictly above the baseline is an upgrade; strictly below
  is a DOWNGRADE (refused with a CRITICAL log unless `allow_downgrade`); an
  equal spine with a different full string (``0.1.0-rc1`` -> ``0.1.0``) is a
  LATERAL change, applied like an upgrade. Versions with no digits at all have
  an empty spine, so any change to or from them is lateral.
* **The ci-pass gate.** With `require_ci_pass` (the default), a configured
  update command still refuses to run unless the staged target's marker names
  the NEW version and binds its full-source digest, shipped-runtime digest and
  immutable manifest bytes. A stale marker cannot authorize drifted bytes.
* **Failure is loud and persistent.** The update command runs under a timeout
  with captured output and a bounded retry envelope; once the budget is spent,
  `update_failed` on /health goes false AND STAYS false, and that version is
  not retried (a later, different version may trigger again; a success clears
  the flag). `update_pending` is false on /health whenever a detected update
  has not been applied — including report-only mode, where "we are behind" is
  exactly the signal the mode exists to surface.
* **Restart-safe.** The applied version persists in a small JSON state file, so
  a restart right after an update does not re-trigger it. State is written only
  AFTER the command exits 0: an update command that restarts this very process
  should arrange the restart and exit 0 (systemd et al.), so the state lands
  first.

WHAT THIS SERVICE DOES NOT SHIP: champions. The model registry's
PromotionPipeline is the champion gate; see README.md in this package for the
separation and why the spec's conflation of the two is resisted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from prometheus_client import Counter, Gauge

from vidaio.autoupdater.config import AutoupdaterConfig
from vidaio.autoupdater.integrity import (
    VerifiedRuntimeManifest,
    verify_ci_release,
)
from vidaio.core import section
from vidaio.services.base import BaseService

#: Environment variable carrying the target version into the update command.
TARGET_VERSION_ENV = "VIDAIO_AUTOUPDATER_TARGET_VERSION"
#: Verified target identity and staging root passed to an argv-only activator.
TARGET_SOURCE_DIGEST_ENV = "VIDAIO_AUTOUPDATER_TARGET_SOURCE_SHA256"
TARGET_RUNTIME_DIGEST_ENV = "VIDAIO_AUTOUPDATER_TARGET_RUNTIME_SHA256"
TARGET_STAGED_ROOT_ENV = "VIDAIO_AUTOUPDATER_STAGED_ROOT"

_NUMERIC_SPINE = re.compile(r"\s*v?(\d+(?:\.\d+)*)")


class VersionSourceError(RuntimeError):
    """The version source could not produce a version string this poll."""


def version_key(version: str) -> tuple[int, ...]:
    """The leading dotted numeric spine of a version string, as an int tuple.

    ``"v1.2.10-rc1"`` -> ``(1, 2, 10)``; a string with no leading digits (after
    an optional ``v``) has the empty spine ``()``.
    """
    match = _NUMERIC_SPINE.match(version)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def compare_versions(candidate: str, baseline: str) -> int:
    """-1 / 0 / +1: candidate below / lateral-to / above baseline.

    THE ORDERING RULE (documented in the module docstring): only the numeric
    spines order, zero-padded to equal length; equal spines — including two
    empty ones — compare as 0 (lateral), never as a downgrade.
    """
    a, b = version_key(candidate), version_key(baseline)
    width = max(len(a), len(b))
    a += (0,) * (width - len(a))
    b += (0,) * (width - len(b))
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


@dataclass(frozen=True)
class _CommandResult:
    ok: bool
    detail: str
    stdout: str = ""
    stderr: str = ""


class Autoupdater(BaseService):
    name = "autoupdater"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cfg = section(raw_config, "autoupdater", AutoupdaterConfig)
        super().__init__(raw_config, metrics_port=self.cfg.metrics_port)
        #: injectable transport for the "http" source (ASGI/Mock in tests)
        self._http_client = http_client

        self._source_ok = True
        self._pending_version: str | None = None
        self._failed_version: str | None = None
        self._refused_downgrade: str | None = None
        self._ci_refusal_logged_for: str | None = None
        #: the baseline — see "never on first sight" in the module docstring
        self._last_applied: str | None = self._load_state() or self._local_version()

        self.health.register_check("version_source", lambda: self._source_ok)
        self.health.register_check("update_pending", lambda: self._pending_version is None)
        self.health.register_check("update_failed", lambda: self._failed_version is None)

        registry = self.health.registry
        self.m_polls = Counter(
            "vidaio_autoupdater_polls_total", "Version-source polls", registry=registry
        )
        self.m_poll_failures = Counter(
            "vidaio_autoupdater_poll_failures_total",
            "Polls on which the version source produced no version",
            registry=registry,
        )
        self.m_version_changes = Counter(
            "vidaio_autoupdater_version_changes_total",
            "Distinct version changes detected against the baseline",
            registry=registry,
        )
        self.m_updates_applied = Counter(
            "vidaio_autoupdater_updates_applied_total",
            "Update commands that ran and exited 0",
            registry=registry,
        )
        self.m_updates_failed = Counter(
            "vidaio_autoupdater_updates_failed_total",
            "Updates abandoned after the retry budget (persistent update_failed)",
            registry=registry,
        )
        self.m_downgrades_refused = Counter(
            "vidaio_autoupdater_downgrades_refused_total",
            "Version rollbacks refused (allow_downgrade not set)",
            registry=registry,
        )
        self.m_ci_gate_refusals = Counter(
            "vidaio_autoupdater_ci_gate_refusals_total",
            "Triggers refused because no ci-pass marker names the new version",
            registry=registry,
        )
        self.m_pending = Gauge(
            "vidaio_autoupdater_update_pending",
            "1 while a detected version change has not been applied",
            registry=registry,
        )

    # -- state -----------------------------------------------------------------

    @property
    def last_applied(self) -> str | None:
        return self._last_applied

    @property
    def pending_version(self) -> str | None:
        return self._pending_version

    def _load_state(self) -> str | None:
        try:
            state = json.loads(self.cfg.state_file.read_text(encoding="utf-8"))
        except OSError:
            return None
        except ValueError:
            self.log.warning(
                f"state file {self.cfg.state_file} is corrupt — falling back to the"
                " local version file for the baseline"
            )
            return None
        value = str(state.get("last_applied") or "").strip()
        return value or None

    def _persist_state(self, version: str) -> None:
        path = self.cfg.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "last_applied": version,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written state

    def _local_version(self) -> str | None:
        """The version THIS deployment is running (its version_file), if readable."""
        try:
            text = self.cfg.version_file.read_text(encoding="utf-8")
        except OSError:
            return None
        first = text.splitlines()[0].strip() if text.splitlines() else ""
        return first or None

    # -- the version source ----------------------------------------------------

    async def _read_version(self) -> str:
        if self.cfg.version_source == "file":
            try:
                text = self.cfg.version_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise VersionSourceError(
                    f"version file {self.cfg.version_file}: {type(exc).__name__}: {exc}"
                ) from exc
            first = text.splitlines()[0].strip() if text.splitlines() else ""
            if not first:
                raise VersionSourceError(f"version file {self.cfg.version_file} is empty")
            return first
        # http: GET returning the version string as its body (first line wins)
        url = self.cfg.version_url
        try:
            if self._http_client is not None:
                response = await self._http_client.get(
                    url, timeout=self.cfg.http_timeout_seconds
                )
            else:
                async with httpx.AsyncClient(
                    timeout=self.cfg.http_timeout_seconds
                ) as client:
                    response = await client.get(url)
        except httpx.HTTPError as exc:
            raise VersionSourceError(
                f"version url {url}: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code != 200:
            raise VersionSourceError(f"version url {url}: HTTP {response.status_code}")
        first = response.text.splitlines()[0].strip() if response.text.splitlines() else ""
        if not first:
            raise VersionSourceError(f"version url {url}: empty body")
        return first

    # -- one poll ----------------------------------------------------------------

    async def poll_once(self) -> None:
        """One full observe -> compare -> (maybe) trigger step. The whole policy."""
        self.m_polls.inc()
        try:
            observed = await self._read_version()
        except VersionSourceError as exc:
            self._source_ok = False
            self.m_poll_failures.inc()
            self.log.warning(f"version source unreadable: {exc}")
            return
        self._source_ok = True

        if self._last_applied is None:
            # No state file, no readable local version_file: the first observed
            # version IS the baseline. Adopted, persisted, never triggered on.
            self._last_applied = observed
            self._persist_state(observed)
            self.log.info(f"baseline adopted from first observation: {observed}")
            return

        if observed == self._last_applied:
            if self._pending_version is not None:
                # The source moved back to what we run (a withdrawn push).
                self.log.info(
                    f"pending update {self._pending_version} withdrawn — the source"
                    f" reports our own version {observed} again"
                )
                self._pending_version = None
                self.m_pending.set(0)
            return

        direction = compare_versions(observed, self._last_applied)
        if direction < 0 and not self.cfg.allow_downgrade:
            if self._refused_downgrade != observed:
                self._refused_downgrade = observed
                self.m_downgrades_refused.inc()
                self.log.critical(
                    f"version source moved BACKWARD: {self._last_applied} ->"
                    f" {observed}. REFUSING the downgrade — a rollback must be a"
                    " human decision (autoupdater.allow_downgrade to permit it)",
                    extra={
                        "fields": {
                            "current": self._last_applied,
                            "offered": observed,
                        }
                    },
                )
            return

        if self._pending_version != observed:
            self._pending_version = observed
            self._ci_refusal_logged_for = None
            self.m_version_changes.inc()
            self.m_pending.set(1)
            self.log.info(
                f"version change detected: {self._last_applied} -> {observed}"
                + (" (report-only: no update_command configured)"
                   if not self.cfg.update_command else ""),
                extra={"fields": {"current": self._last_applied, "target": observed}},
            )

        if not self.cfg.update_command:
            return  # report-only: pending stays visible on /health and /metrics

        if self._failed_version == observed:
            return  # already failed terminally for this version; stay unhealthy

        if self.cfg.require_ci_pass and self._verified_ci_pass(observed) is None:
            self.m_ci_gate_refusals.inc()
            if self._ci_refusal_logged_for != observed:
                self._ci_refusal_logged_for = observed
                self.log.warning(
                    f"update to {observed} refused: {self.cfg.ci_pass_marker} does"
                    f" not record a CI pass for that version — run the release CI gate (development tree)"
                    " (full, green) to write it"
                )
            return

        await self._apply_update(observed)

    def _verified_ci_pass(self, version: str) -> VerifiedRuntimeManifest | None:
        """Verify the staged target's marker, manifest and exact runtime bytes."""
        try:
            return verify_ci_release(
                self.cfg.resolved_ci_pass_marker,
                self.cfg.resolved_runtime_manifest_file,
                source_root=self.cfg.source_root,
                runtime_root=self.cfg.source_root,
                expected_version=version,
            )
        except (OSError, ValueError):
            # Permission changes and file replacement races can surface from a
            # runtime input after the verifier enumerates the tree. They are a
            # closed CI gate, not a reason to crash and skip future polls.
            return None

    def _ci_pass_satisfied(self, version: str) -> bool:
        """Compatibility/readability wrapper used by policy tests and callers."""
        return self._verified_ci_pass(version) is not None

    # -- the update command ------------------------------------------------------

    async def _apply_update(self, version: str) -> None:
        for attempt in range(1, self.cfg.update_retry_attempts + 1):
            # Re-verify immediately before every attempt. A staging tree changed
            # after the poll gate is no longer the artifact CI authorized.
            verified = (
                self._verified_ci_pass(version)
                if self.cfg.require_ci_pass
                else None
            )
            if self.cfg.require_ci_pass and verified is None:
                self.log.critical(
                    f"update to {version} refused before activation: staged release "
                    "identity changed after its CI gate"
                )
                return
            result = await asyncio.to_thread(self._run_command, version, verified)
            if result.ok:
                self._last_applied = version
                self._pending_version = None
                self._failed_version = None
                self._refused_downgrade = None
                self.m_pending.set(0)
                self._persist_state(version)
                self.m_updates_applied.inc()
                self.log.info(
                    f"update to {version} applied (attempt {attempt})",
                    extra={
                        "fields": {
                            "version": version,
                            "stdout_tail": result.stdout[-2000:],
                            "stderr_tail": result.stderr[-2000:],
                        }
                    },
                )
                return
            self.log.error(
                f"update command failed (attempt {attempt}/"
                f"{self.cfg.update_retry_attempts}): {result.detail}",
                extra={
                    "fields": {
                        "version": version,
                        "stdout_tail": result.stdout[-2000:],
                        "stderr_tail": result.stderr[-2000:],
                    }
                },
            )
            if attempt < self.cfg.update_retry_attempts:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self.stopping.wait(), timeout=self.cfg.update_retry_delay_seconds
                    )
                if self.stopping.is_set():
                    return  # shutting down mid-retry: still pending, not failed
        self._failed_version = version
        self.m_updates_failed.inc()
        self.log.critical(
            f"update to {version} FAILED after {self.cfg.update_retry_attempts}"
            " attempt(s) — update_failed is now flagged on /health and this version"
            " will not be retried (a later version, or a restart after a manual"
            " fix, clears it)"
        )

    def _run_command(
        self, version: str, verified: VerifiedRuntimeManifest | None
    ) -> _CommandResult:
        """One subprocess run of the deployment's update command (worker thread)."""
        env = {**os.environ, TARGET_VERSION_ENV: version}
        if verified is not None:
            env.update(
                {
                    TARGET_SOURCE_DIGEST_ENV: verified.source_sha256,
                    TARGET_RUNTIME_DIGEST_ENV: verified.runtime_sha256,
                    TARGET_STAGED_ROOT_ENV: str(self.cfg.source_root.resolve()),
                }
            )
        try:
            completed = subprocess.run(
                list(self.cfg.update_command),
                capture_output=True,
                text=True,
                timeout=self.cfg.update_timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return _CommandResult(
                ok=False,
                detail=f"timed out after {self.cfg.update_timeout_seconds}s",
                stdout=(exc.stdout or b"").decode(errors="replace")
                if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(exc.stderr or b"").decode(errors="replace")
                if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
            )
        except OSError as exc:
            return _CommandResult(ok=False, detail=f"{type(exc).__name__}: {exc}")
        if completed.returncode != 0:
            return _CommandResult(
                ok=False,
                detail=f"exit code {completed.returncode}",
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        return _CommandResult(
            ok=True, detail="ok", stdout=completed.stdout, stderr=completed.stderr
        )

    # -- lifecycle ---------------------------------------------------------------

    async def run(self) -> None:
        mode = "active" if self.cfg.update_command else "report-only"
        self.log.info(
            f"autoupdater started: baseline={self._last_applied!r},"
            f" source={self.cfg.version_source}, mode={mode},"
            f" require_ci_pass={self.cfg.require_ci_pass}"
        )
        while not self.stopping.is_set():
            await self.poll_once()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self.stopping.wait(), timeout=self.cfg.poll_seconds
                )
