"""Configuration for the schema-v14 executable-baseline registry service."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.store import ArtifactKind, ArtifactRef, backend_key
from vidaio.competition.interfaces import logical_build_identity
from vidaio.registry.baseline import GenesisBaseline, SUPPORTED_TRACKS

REGISTRY_HTTP_PORT = 8720
REGISTRY_METRICS_PORT = 9123

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class GenesisBaselineConfig(BaseModel):
    """One explicitly content-addressed public version-zero executable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_digest: str = ""
    artifact_bytes: int = 0
    image_digest: str = ""
    provenance_digest: str = ""
    provenance_bytes: int = 0
    repo_url: str = ""
    commit_sha: str = ""
    tree_sha: str = ""

    def problems(self, *, track: str) -> list[str]:
        prefix = f"registry.genesis_baselines[{track!r}]"
        problems: list[str] = []
        for field in ("artifact_digest", "image_digest", "provenance_digest"):
            if _SHA256.fullmatch(str(getattr(self, field))) is None:
                problems.append(f"{prefix}.{field} must be a lowercase sha256 digest")
        for field in ("artifact_bytes", "provenance_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                problems.append(f"{prefix}.{field} must be a positive integer")
        for field in ("commit_sha", "tree_sha"):
            if _GIT_SHA.fullmatch(str(getattr(self, field))) is None:
                problems.append(f"{prefix}.{field} must be a lowercase 40-hex git id")
        try:
            parsed = urlsplit(self.repo_url)
            _ = parsed.port
        except ValueError:
            parsed = None
        if (
            parsed is None
            or self.repo_url != self.repo_url.strip()
            or any(character.isspace() for character in self.repo_url)
            or parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/")
        ):
            problems.append(
                f"{prefix}.repo_url must be a credential-free https git URL"
            )
        elif (
            _GIT_SHA.fullmatch(self.commit_sha) is not None
            and _GIT_SHA.fullmatch(self.tree_sha) is not None
            and _SHA256.fullmatch(self.image_digest) is not None
        ):
            expected = logical_build_identity(
                repo_url=self.repo_url,
                commit_sha=self.commit_sha,
                tree_sha=self.tree_sha,
            )
            if self.image_digest != expected:
                problems.append(
                    f"{prefix}.image_digest must equal the stable logical build "
                    f"identity {expected} for repo_url/commit_sha/tree_sha"
                )
        return problems

    def seed(self, *, track: str) -> GenesisBaseline:
        problems = self.problems(track=track)
        if problems:
            raise ValueError("; ".join(problems))
        artifact_kind = ArtifactKind.SUBMISSION_ARCHIVE
        provenance_kind = ArtifactKind.MANIFEST
        return GenesisBaseline(
            track=track,
            artifact=ArtifactRef(
                digest=self.artifact_digest,
                kind=artifact_kind,
                byte_size=self.artifact_bytes,
                backend_key=backend_key(artifact_kind, self.artifact_digest),
            ),
            image_digest=self.image_digest,
            provenance=ArtifactRef(
                digest=self.provenance_digest,
                kind=provenance_kind,
                byte_size=self.provenance_bytes,
                backend_key=backend_key(provenance_kind, self.provenance_digest),
            ),
            repo_url=self.repo_url,
            commit_sha=self.commit_sha,
            tree_sha=self.tree_sha,
        )


class RegistryConfig(BaseModel):
    """The persistent registry API and its mandatory v0 archive identities."""

    model_config = ConfigDict(extra="forbid")

    db_path: Path = Path("./data/registry/registry.db")
    http_host: str = "0.0.0.0"
    http_port: int = Field(REGISTRY_HTTP_PORT, ge=1, le=65535)
    metrics_port: int = Field(REGISTRY_METRICS_PORT, ge=1, le=65535)
    automatic_promotion_enabled: bool = False
    allow_disabled_automatic_promotion_for_testnet: bool = False
    allow_disabled_automatic_promotion_for_mainnet: bool = False
    genesis_baselines: dict[str, GenesisBaselineConfig] = Field(default_factory=dict)

    def genesis_problems(self) -> list[str]:
        configured = set(self.genesis_baselines)
        expected = set(SUPPORTED_TRACKS)
        problems: list[str] = []
        if configured != expected:
            problems.append(
                "registry.genesis_baselines must configure exactly compression and "
                f"upscaling; got {sorted(configured)}"
            )
        for track in sorted(configured & expected):
            problems.extend(self.genesis_baselines[track].problems(track=track))
        return problems

    def genesis_seeds(self) -> tuple[GenesisBaseline, ...]:
        problems = self.genesis_problems()
        if problems:
            raise ValueError("; ".join(problems))
        return tuple(
            self.genesis_baselines[track].seed(track=track)
            for track in SUPPORTED_TRACKS
        )


def production_registry_problems(config: RegistryConfig) -> list[str]:
    """Static fail-closed checks for the independently deployed registry role."""

    problems = config.genesis_problems()
    if not config.db_path.is_absolute():
        problems.append("registry.db_path must be an absolute writable production path")
    if config.http_port != REGISTRY_HTTP_PORT:
        problems.append(f"registry.http_port must be {REGISTRY_HTTP_PORT}")
    if config.metrics_port != REGISTRY_METRICS_PORT:
        problems.append(f"registry.metrics_port must be {REGISTRY_METRICS_PORT}")
    if (
        config.automatic_promotion_enabled
        and (
            config.allow_disabled_automatic_promotion_for_testnet
            or config.allow_disabled_automatic_promotion_for_mainnet
        )
    ):
        problems.append(
            "registry.allow_disabled_automatic_promotion_for_testnet must be false "
            "when automatic promotion is enabled"
        )
    elif (
        not config.automatic_promotion_enabled
        and not config.allow_disabled_automatic_promotion_for_testnet
    ):
        problems.append(
            "registry.automatic_promotion_enabled must be true for production; "
            "testnet may explicitly set "
            "registry.allow_disabled_automatic_promotion_for_testnet=true while "
            "the verified chain watcher adapter remains unwired"
        )
    return problems


def production_registry_network_problems(
    config: RegistryConfig, *, network: str, netuid: int
) -> list[str]:
    """Require separate operator authorization for the SN85 genesis exception."""
    if config.allow_disabled_automatic_promotion_for_mainnet:
        if (
            network != "finney"
            or netuid != 85
            or config.automatic_promotion_enabled
            or not config.allow_disabled_automatic_promotion_for_testnet
        ):
            return [
                "registry.allow_disabled_automatic_promotion_for_mainnet requires "
                "finney SN85, disabled promotion and the explicit legacy exception"
            ]
        return []
    if config.allow_disabled_automatic_promotion_for_testnet and network != "test":
        return [
            "registry.allow_disabled_automatic_promotion_for_testnet is permitted "
            "only when chain.network is explicitly 'test' unless the SN85 "
            "mainnet genesis exception is separately authorized"
        ]
    return []


__all__ = [
    "REGISTRY_HTTP_PORT",
    "REGISTRY_METRICS_PORT",
    "GenesisBaselineConfig",
    "RegistryConfig",
    "production_registry_problems",
    "production_registry_network_problems",
]
