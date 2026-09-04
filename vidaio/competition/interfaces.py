"""Protocols for the later execution/scoring phases — spec: design spec §05 (interface only).

The Modal sandbox runner and the remote scoring worker are built in later phases;
this module pins down the seams the lifecycle engine will call so those phases can
be developed and faked independently. No implementation lives here.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


UPSCALE_FACTOR_SIDECAR_PREFIX = ".vidaio-next-upscale-factor-"
UPSCALE_TASK_SIDECAR_PREFIX = ".vidaio-next-upscale-task-"
SUPPORTED_UPSCALE_FACTORS = frozenset({2, 4})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_HEX = re.compile(r"[0-9a-f]{40}")

#: Versioned domain for the protocol-facing build identity.  This is deliberately
#: independent of a cloud provider's opaque Image object id: Modal creates a new
#: ``im-*`` id for every fresh build even when the pinned source is identical.
#: Provider ids remain separate, exact, append-only ownership evidence used for
#: restoration; this digest is the stable identity committed by manifests, score
#: bundles, epoch logs, and the executable-baseline registry.
BUILD_IDENTITY_SCHEME = "vidaio.competition.logical-build.v1"


def logical_build_identity(*, repo_url: str, commit_sha: str, tree_sha: str) -> str:
    """Return the stable logical image identity for one pinned source spec.

    The canonical preimage includes the exact source locator, both Git object ids,
    and a versioned domain.
    ``tree_sha`` binds the complete Docker build context, including the
    Dockerfile; ``commit_sha`` preserves the selected revision; ``repo_url``
    preserves the enrolled public source claim. Qualification may build from an
    already verified local checkout, but must assert this same canonical public
    locator. The result says *which pinned build input* executed. It does not claim
    that independent cloud builds have the same provider object: that exact
    ``im-*`` handle is recorded and verified separately.
    """

    if not isinstance(repo_url, str) or not repo_url or repo_url != repo_url.strip():
        raise ValueError("repo_url must be a non-empty, whitespace-trimmed string")
    if _GIT_SHA_HEX.fullmatch(commit_sha) is None:
        raise ValueError("commit_sha must be lowercase 40-hex")
    if _GIT_SHA_HEX.fullmatch(tree_sha) is None:
        raise ValueError("tree_sha must be lowercase 40-hex")
    canonical = json.dumps(
        {
            "commit_sha": commit_sha,
            "repo_url": repo_url,
            "scheme": BUILD_IDENTITY_SCHEME,
            "tree_sha": tree_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def upscale_factor_sidecar_name(input_sha256: str) -> str:
    """Return the hidden, per-input filename that publishes the committed factor.

    The name binds metadata to the miner-visible input digest and deliberately
    contains neither the private reference digest nor the evaluation index.
    """
    if _SHA256_HEX.fullmatch(input_sha256) is None:
        raise ValueError("input_sha256 must be lowercase sha256 hex")
    return f"{UPSCALE_FACTOR_SIDECAR_PREFIX}{input_sha256}"


def upscale_factor_sidecar_bytes(upscale_factor: int) -> bytes:
    """Canonical sidecar payload (exactly ``b\"2\\n\"`` or ``b\"4\\n\"``)."""
    if (
        type(upscale_factor) is not int
        or upscale_factor not in SUPPORTED_UPSCALE_FACTORS
    ):
        raise ValueError(
            f"upscale_factor must be an integer in {sorted(SUPPORTED_UPSCALE_FACTORS)}"
        )
    return f"{upscale_factor}\n".encode("ascii")


def upscale_task_sidecar_name(input_sha256: str) -> str:
    """Hidden filename for the digest-bound complete upscaling task contract."""
    if _SHA256_HEX.fullmatch(input_sha256) is None:
        raise ValueError("input_sha256 must be lowercase sha256 hex")
    return f"{UPSCALE_TASK_SIDECAR_PREFIX}{input_sha256}"


def upscale_task_sidecar_bytes(
    upscale_factor: int, target_width: int, target_height: int
) -> bytes:
    """Canonical factor + exact-output-geometry contract for contender code.

    A factor alone is insufficient because the challenge downscale includes a
    subpixel crop and even truncation: ``input_dimension * factor`` can differ
    from the pristine dimensions accepted by scoring. The one-line canonical
    JSON payload is committed before enrollment and names no holdout artifact.
    """
    # Reuse the factor validator and reject bool-as-int dimensions explicitly.
    upscale_factor_sidecar_bytes(upscale_factor)
    for field, value in (
        ("target_width", target_width),
        ("target_height", target_height),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    return (
        json.dumps(
            {
                "target_height": target_height,
                "target_width": target_width,
                "upscale_factor": upscale_factor,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("ascii")


@dataclass(frozen=True)
class ContenderSpec:
    """Pinned code identity of one contender (real or calibration)."""

    contender_id: int
    repo_url: str
    commit_sha: str
    tree_sha: str


@dataclass(frozen=True)
class BatchItem:
    """One miner-visible evaluation input, referenced by digest.

    For compression this is also the pristine reference.  For upscaling it is ONLY
    the low-resolution input; the distinct high-resolution reference deliberately
    does not exist on this runner-facing type and is never mounted into a contender
    sandbox. ``upscale_factor`` and the exact target dimensions are the only
    upscaling task metadata exposed to contender code: runners publish them as
    one hidden, digest-bound sidecar. They are committed before enrollment; no
    reference digest, bytes, path, or item index is written to that sidecar. Raw
    media bytes never transit the DB.
    """

    item_id: int
    item_index: int
    input_sha256: str
    input_bytes: int
    length_seconds: float | None = None
    upscale_factor: int | None = None
    target_width: int | None = None
    target_height: int | None = None


@dataclass(frozen=True)
class BatchOutput:
    """One produced output, again by digest + size only."""

    item_id: int
    output_sha256: str
    output_bytes: int
    wall_seconds: float | None = None


@dataclass(frozen=True)
class IsolationProbeReport:
    """Result of the pre-run isolation probe (spec §05): each field asserts one
    boundary the sandbox must hold before any miner code touches an input."""

    network_blocked: bool
    secrets_absent: bool
    reference_mounts_absent: bool
    index_leak_absent: bool
    details: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.network_blocked
            and self.secrets_absent
            and self.reference_mounts_absent
            and self.index_leak_absent
        )


@dataclass(frozen=True)
class ScorePacket:
    """One item's score as produced by the trusted remote scorer: the scorer's
    canonical ItemScore JSON, verbatim bytes. Persistence
    (repository.record_item_score) parses exactly these bytes, derives
    item_score/valid from the packet's own score/gate_passed and stores
    sha256(packet_bytes) as score_packet_digest — the packet is the single source
    of truth; no score field can be supplied out-of-band (spec §14 auditability)."""

    item_id: int  # evaluation_items.item_id this packet scores
    contender_id: int
    packet_bytes: bytes


@runtime_checkable
class SandboxRunner(Protocol):
    """Builds and runs untrusted miner code in a locked-down sandbox (spec §05).

    Contract the implementation MUST honor:
    - clone via read-only PAT and pin repository + commit + tree SHA; ``image_digest``
      is the stable, domain-separated :func:`logical_build_identity`. Provider image
      ids/handles are separate execution evidence and must never perturb it.
    - image size is UNATTESTED on Modal — the container_size_limit_gb manifest cap is
      acknowledged-but-unmeasured until the trusted-builder proof lands (spec §04).
    - sandbox: network blocked, secrets=[], no OIDC token, sealed inputs mounted
      read-only, per-hotkey output volume read-write.
    - isolation_probe() must run before any batch and assert no network, no reference
      mounts, no leaked index.json; a failing probe means the sandbox is never used.
    - upscaling batches expose one hidden factor+target-geometry sidecar per
      digest-named input; pristine reference material is never mounted or named.
    - one warm sandbox may be reused across batches but must be rolled over before the
      ~23h30m provider lifetime cap.
    """

    def build(self, contender: ContenderSpec) -> str:
        """Build and return the stable logical identity of the pinned source.

        Must be a killable subprocess with a bounded timeout; a failed/timed-out build
        marks the contender BUILD_FAILED, never the competition."""
        ...

    def run_batch(
        self, image_digest: str, items: Sequence[BatchItem], batch_index: int
    ) -> Sequence[BatchOutput]:
        """Run one evaluation batch inside the isolated sandbox; outputs land on the
        output volume and are reported here by digest + size only."""
        ...

    def isolation_probe(self, image_digest: str) -> IsolationProbeReport:
        """Assert the isolation boundary (no network, no secrets, no reference mounts,
        no leaked index) before the first batch runs."""
        ...


@runtime_checkable
class CompetitionScoringClient(Protocol):
    """Client for the trusted remote scoring worker (spec §05).

    Whoever controls the scoring endpoint controls the numbers — the endpoint must be
    validator-operated, and every ScorePacket must be independently recomputable from
    the audit store (inputs + outputs archived; §14). An in-process implementation is
    a test seam only, never production.
    """

    def score_item(
        self,
        competition_id: str,
        contender_id: int,
        item: BatchItem,
        output: BatchOutput,
    ) -> ScorePacket:
        """Score one output using repository-bound trusted artifacts.

        Compression normalizes reference=input; upscaling supplies the scorer a
        distinct pristine reference, miner input and manifest-committed factor.  The
        runner receives the low-resolution input plus its committed factor and
        exact target dimensions only.
        """
        ...
