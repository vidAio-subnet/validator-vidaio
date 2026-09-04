"""Deploy-time serving seam for the active executable baseline.

The active baseline is a quality floor and may also serve inference traffic.
Actually running it needs a GPU host, so this module defines only the contract.
A deployment adapter loads the archived executable from
``BaselineRecord.artifact_ref()`` and serves the same wire protocol the inference
fleet speaks — from the gateway's point of view the baseline endpoint is one more
``POST /v1/task/artifact`` backend, so no gateway code changes at deploy time.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vidaio.registry.baseline import BaselineRecord
from vidaio.services.protocol import MinerTaskRequest, MinerTaskResponse


@runtime_checkable
class BaselineBackend(Protocol):
    """A running instance of the active executable baseline for one track.

    Implementations are deploy-time adapters (GPU); tests use in-process fakes.
    `process` must satisfy the same contract the gateway enforces on every
    backend: the response's output_digest is the sha256 of the bytes at
    output_path — the gateway re-verifies and fails the job on mismatch.
    """

    @property
    def baseline(self) -> BaselineRecord:
        """The registry row this backend is serving (identity + provenance)."""
        ...

    async def process(self, request: MinerTaskRequest) -> MinerTaskResponse:
        """Run one task through the active baseline executable."""
        ...


# Compatibility name for deployment adapters during the schema-v14 rollout.
# New code should implement/read ``baseline`` and use ``BaselineBackend``.
@runtime_checkable
class ChampionBackend(Protocol):
    """Compatibility protocol for adapters not yet renamed to ``baseline``."""

    @property
    def champion(self) -> BaselineRecord: ...

    async def process(self, request: MinerTaskRequest) -> MinerTaskResponse: ...
