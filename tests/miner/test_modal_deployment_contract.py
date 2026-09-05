"""Offline fail-closed checks for the fresh Modal inference deployment module."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

from httpx import ASGITransport, AsyncClient


MODULE = Path("deploy/modal/vidaio_next_gpu_miner.py")
README = Path("deploy/modal/README.md")
COMPETITION_README = Path("deploy/modal/COMPETITION.md")
CONFIRMATION = "CREATE_FRESH_VIDAIO_NEXT_MODAL_RESOURCES"


def _run_module(**overrides: str) -> subprocess.CompletedProcess[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "VIDAIO_NEXT_DEPLOYMENT_LABEL",
            "VIDAIO_NEXT_MODAL_SECRET_NAME",
            "VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION",
        }
    }
    env.update(overrides)
    return subprocess.run(
        [sys.executable, str(MODULE)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_deploy_module_has_no_implicit_remote_activation_or_reusable_name() -> None:
    result = _run_module()

    assert result.returncode != 0
    assert "deployment is disabled" in result.stderr


def test_deploy_module_requires_explicit_fresh_prefixed_app_and_secret() -> None:
    result = _run_module(
        VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION=CONFIRMATION,
        VIDAIO_NEXT_DEPLOYMENT_LABEL="existing-app",
        VIDAIO_NEXT_MODAL_SECRET_NAME="vidaio-next-auth-abcdef12",
    )

    assert result.returncode != 0
    assert "there is no reusable default" in result.stderr


def test_remote_environment_hint_cannot_bypass_local_creation_guard() -> None:
    result = _run_module(MODAL_IS_REMOTE="1")

    assert result.returncode != 0
    assert "deployment is disabled" in result.stderr


def test_source_uses_fresh_app_identity_and_never_resolves_other_resource_types() -> (
    None
):
    source = MODULE.read_text(encoding="utf-8")
    deployment_import_path = source.split("def gpu_miner_app", maxsplit=1)[0]

    assert "APP_NAME = DEPLOYMENT_LABEL" in source
    assert "VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION" in source
    # Importing vidaio.miner.gpu_worker first executes vidaio.miner.__init__, whose
    # configuration path imports yaml. The remote Image must carry that dependency;
    # the operator host's lock environment cannot mask an incomplete Function image.
    assert '"PyYAML==6.0.3"' in source
    assert "modal.App.from_name" not in source
    assert "modal.Volume.from_name" not in source
    assert "modal.Dict.from_name" not in source
    assert "modal.Queue.from_name" not in source
    # The clean, isolated `uvx --from modal` deployment client has the Modal SDK
    # but not Function-image dependencies. FastAPI/Starlette stay remote-only.
    assert "from fastapi" not in deployment_import_path
    assert "from starlette" not in deployment_import_path


def test_secret_provisioning_pins_modal_dotenv_parser() -> None:
    instructions = README.read_text(encoding="utf-8")

    assert "--with python-dotenv==1.2.3" in instructions
    secret_command = instructions.split("modal secret create", maxsplit=1)[0]
    assert secret_command.rfind("--with python-dotenv==1.2.3") > secret_command.rfind(
        "uvx --from modal==1.5.4"
    )


def test_runbooks_clean_only_the_exact_fresh_environment_without_inventory() -> None:
    inference = README.read_text(encoding="utf-8")
    competition = COMPETITION_README.read_text(encoding="utf-8")

    for instructions in (inference, competition):
        assert 'modal environment delete --yes "${MODAL_ENV}"' in instructions
        assert "uvx --from modal==1.5.4 modal environment list" not in instructions
        assert "uvx --from modal==1.5.4 modal app list" not in instructions
        assert "uvx --from modal==1.5.4 modal secret list" not in instructions
        assert (
            "--force"
            not in instructions.split("modal environment delete", maxsplit=1)[1]
        )

    assert "trap 'rm -f -- \"${SECRET_FILE}\"' EXIT HUP INT TERM" in inference


def test_remote_modal_reimport_needs_no_local_creation_environment(
    monkeypatch,
) -> None:
    """Modal's Function container re-import must not repeat the deploy operation."""

    class FakeImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def apt_install(self, *_args):
            return self

        def run_commands(self, *_commands):
            return self

        def uv_pip_install(self, *_args):
            return self

        def add_local_python_source(self, *_args):
            return self

    class FakeSecret:
        @staticmethod
        def from_name(_name):
            raise AssertionError("remote import must not resolve a named Secret")

        @staticmethod
        def from_dict(values):
            assert values == {}
            return object()

    class FakeApp:
        def __init__(self, _name):
            pass

        @staticmethod
        def function(**_kwargs):
            return lambda fn: fn

        @staticmethod
        def local_entrypoint():
            return lambda fn: fn

    def decorator(**_kwargs):
        return lambda fn: fn

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Secret=FakeSecret,
        asgi_app=decorator,
        concurrent=decorator,
        is_local=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    for name in (
        "VIDAIO_NEXT_DEPLOYMENT_LABEL",
        "VIDAIO_NEXT_MODAL_SECRET_NAME",
        "VIDAIO_NEXT_FRESH_CREATION_CONFIRMATION",
    ):
        monkeypatch.delenv(name, raising=False)

    namespace = runpy.run_path(str(MODULE))

    assert namespace["APP_NAME"] == "vidaio-next-remote-runtime"
    assert namespace["SECRET_NAME"] == "vidaio-next-remote-auth-placeholder"


def test_remote_asgi_routes_inject_request_instead_of_query_parameter(
    monkeypatch,
) -> None:
    """Future annotations must resolve for nested Modal ASGI route handlers."""

    class FakeImage:
        @classmethod
        def debian_slim(cls, **_kwargs):
            return cls()

        def apt_install(self, *_args):
            return self

        def run_commands(self, *_commands):
            return self

        def uv_pip_install(self, *_args):
            return self

        def add_local_python_source(self, *_args):
            return self

    class FakeSecret:
        @staticmethod
        def from_dict(values):
            assert values == {}
            return object()

    class FakeApp:
        def __init__(self, _name):
            pass

        @staticmethod
        def function(**_kwargs):
            return lambda fn: fn

        @staticmethod
        def local_entrypoint():
            return lambda fn: fn

    def decorator(**_kwargs):
        return lambda fn: fn

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        Image=FakeImage,
        Secret=FakeSecret,
        asgi_app=decorator,
        concurrent=decorator,
        is_local=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    monkeypatch.setenv(
        "VIDAIO_NEXT_DEPLOYMENT_LABEL", "vidaio-next-route-test-abcdef12"
    )
    monkeypatch.setenv("VIDAIO_NEXT_SOLUTION_VARIANT", "quality")
    monkeypatch.setenv("VIDAIO_NEXT_GPU_AUTH_TOKEN", "route-test-token")

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Fake L4")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _index: 0)

    namespace = runpy.run_path(str(MODULE))
    web = namespace["gpu_miner_app"]()
    headers = {"Authorization": "Bearer route-test-token"}

    async def exercise_routes() -> None:
        transport = ASGITransport(app=web)
        async with AsyncClient(
            transport=transport, base_url="http://modal.test"
        ) as client:
            health = await client.get("/healthz", headers=headers)
            assert health.status_code == 200
            assert health.json()["gpu_available"] is True

            metrics = await client.get("/metrics", headers=headers)
            assert metrics.status_code == 200
            assert "vidaio_next_modal_gpu_memory_allocated_bytes" in metrics.text

            process = await client.post("/process", headers=headers, content=b"")
            assert process.status_code == 400
            assert process.json() == {"detail": "missing GPU task metadata"}
            assert "query" not in process.text

    asyncio.run(exercise_routes())

    from vidaio.miner import gpu_worker
    from vidaio.miner.remote_gpu import (
        CPU_FALLBACK_DEVICE, GPU_ACCELERATED_HEADER, GPU_DEVICE_HEADER,
        GPU_METADATA_HEADER, GPU_PROTOCOL_HEADER, GPU_PROTOCOL_VERSION,
    )

    monkeypatch.setenv("VIDAIO_NEXT_GPU_ALLOW_CPU_FALLBACK", "true")
    observed = []

    def recovered(input_path, output_path, metadata, **kwargs):
        observed.append(kwargs)
        output_path.write_bytes(b"test-media")
        return gpu_worker.TransformResult(
            output_path, 10, 128, 96, CPU_FALLBACK_DEVICE, 0.0,
        )

    monkeypatch.setattr(gpu_worker, "transform_media", recovered)
    fallback_web = namespace["gpu_miner_app"]()

    async def exercise_recovery():
        payload = {
            "protocol": GPU_PROTOCOL_VERSION, "track": "upscaling",
            "solution_variant": "quality", "input_digest": hashlib.sha256(b"input").hexdigest(),
            "input_size": 5, "deadline_seconds": 30, "params": {"upscale_factor": 2},
        }
        bound_headers = {
            **headers, GPU_PROTOCOL_HEADER: GPU_PROTOCOL_VERSION,
            GPU_METADATA_HEADER: base64.urlsafe_b64encode(json.dumps(payload).encode()).decode(),
        }
        async with AsyncClient(transport=ASGITransport(app=fallback_web), base_url="http://modal.test") as client:
            response = await client.post("/process", headers=bound_headers, content=b"input")
        assert response.status_code == 200
        assert response.content == b"test-media"
        assert response.headers[GPU_ACCELERATED_HEADER] == "false"
        assert response.headers[GPU_DEVICE_HEADER] == CPU_FALLBACK_DEVICE
        assert observed[0]["allow_cpu_fallback"] is True

    asyncio.run(exercise_recovery())
