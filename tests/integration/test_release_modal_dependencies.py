"""Offline release checks for the Modal competition runtime dependencies."""

from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

import pytest

from scripts import verify_release_dependencies as release_dependencies


ROOT = Path(__file__).resolve().parents[2]


def test_modal_is_exactly_pinned_locked_and_installed_in_release_and_test_images():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["modal"] == ["modal==1.5.4"]

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_modal = [
        package for package in lock["package"] if package["name"] == "modal"
    ]
    assert [package["version"] for package in locked_modal] == ["1.5.4"]

    dockerfile = " ".join((ROOT / "Dockerfile").read_text(encoding="utf-8").split())
    assert (
        "apt-get install -y --no-install-recommends ca-certificates git" in dockerfile
    )
    # Dependency-only runtime sync, source/runtime sync, and explicit test sync.
    assert dockerfile.count("--extra modal") == 3
    assert "git --version >/dev/null" in dockerfile


def test_locked_modal_sdk_has_every_offline_adapter_surface():
    pytest.importorskip("modal")
    assert importlib.metadata.version("modal") == "1.5.4"
    assert release_dependencies._verify_modal_contract() == {
        "create_only_signature_check": True,
        "filesystem_signature_check": True,
        "image_restore_signature_check": True,
        "process_stream_signature_check": True,
    }


def test_modal_contract_fails_closed_on_signature_drift(monkeypatch):
    modal = pytest.importorskip("modal")

    def incompatible_create(*, app, name):  # noqa: ARG001
        return None

    monkeypatch.setattr(modal.Sandbox, "create", staticmethod(incompatible_create))
    with pytest.raises(RuntimeError, match="(?:args|tags).*parameter"):
        release_dependencies._verify_modal_contract()


def test_modal_contract_fails_closed_when_exact_image_restore_drifts(monkeypatch):
    modal = pytest.importorskip("modal")

    def incompatible_from_id(*, provider_ref):  # noqa: ARG001
        return None

    monkeypatch.setattr(modal.Image, "from_id", staticmethod(incompatible_from_id))
    with pytest.raises(RuntimeError, match="image_id.*parameter"):
        release_dependencies._verify_modal_contract()


def test_release_git_executable_is_real_and_bounded():
    version = release_dependencies._verify_git_executable()
    assert version.startswith("git version ")
    assert len(version) <= 200


def test_release_git_executable_is_required(monkeypatch):
    monkeypatch.setattr(release_dependencies.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="missing the required git executable"):
        release_dependencies._verify_git_executable()


def test_release_calibration_gate_pins_upscaling_duration_floor():
    contract = release_dependencies._verify_launch_calibration_contract()
    assert contract["upscaling_min_clip_seconds"] == 10.0
    assert contract["max_eligibility_scan_assets"] == 96
    assert {
        contract["cpu_reference_crf"],
        contract["backend_default_crf"],
        contract["gpu_quality_crf"],
    } == {22}


def test_release_upscaling_smoke_media_pins_the_launch_floor() -> None:
    assert release_dependencies.CPU_UPSCALING_SMOKE_FPS == 4
    assert release_dependencies.CPU_UPSCALING_SMOKE_FRAME_COUNT == 40
    assert release_dependencies.CPU_UPSCALING_SMOKE_DURATION_SECONDS == 10.0
    assert (
        release_dependencies.CPU_UPSCALING_SMOKE_FRAME_COUNT
        / release_dependencies.CPU_UPSCALING_SMOKE_FPS
        == release_dependencies.CPU_UPSCALING_SMOKE_DURATION_SECONDS
    )
