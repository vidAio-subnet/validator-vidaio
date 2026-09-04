"""Per-item audit bundle: everything needed to independently recompute one score.

A bundle binds one challenge item's inputs, outputs, commitment, and recorded
score packet together under a single stable digest. Third parties fetch the
referenced artifacts by digest and rerun scoring; if anything in the bundle —
a ref, the timestamp, a version string — is altered after the fact, the
bundle_digest no longer matches what was published/anchored.

Lifecycle stages:
- PRE_REVEAL: the asset is still live. The reference original (sealed holdout)
  and the DAG reveal MUST be absent — publishing either mid-competition would
  leak the holdout / the private degradation seeds.
- POST_RETIREMENT: the "recomputable" stage. Everything must be present; only
  these bundles can pass full verification (recompute.verify_bundle).
- COMPETITION_SEALED: an upscaling competition bundle. It names the separately
  sealed pristine reference and carries the manifest-bound item preimage, but has
  no inference degradation DAG. Public recomputation succeeds only after the
  reference is released by the completion gate.

The score packet is the scoring module's ItemScore JSON, treated here as
opaque bytes + digest — the audit layer never interprets it beyond the
verification step's metric comparison.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind, ArtifactRef
from vidaio.challenge import ChallengeAnchor
from vidaio.services.artifact_auth import MinerArtifactReceipt


class LifecycleStage(StrEnum):
    PRE_REVEAL = "pre_reveal"
    COMPETITION_SEALED = "competition_sealed"
    POST_RETIREMENT = "post_retirement"


class CompetitionItemBinding(BaseModel):
    """Manifest-committed upscaling item preimage copied into a bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_index: int = Field(ge=0)
    input_sha256: str = Field(pattern=SHA256_HEX_PATTERN)
    reference_sha256: str = Field(pattern=SHA256_HEX_PATTERN)
    upscale_factor: Literal[2, 4]
    # NULL/NULL exists only for already-anchored v1 item commitments. New v2
    # commitments bind the exact output geometry required by scoring.
    target_width: int | None = Field(default=None, gt=0)
    target_height: int | None = Field(default=None, gt=0)
    item_commitment: str = Field(pattern=SHA256_HEX_PATTERN)

    @model_validator(mode="after")
    def _distinct_media(self) -> "CompetitionItemBinding":
        if self.reference_sha256 == self.input_sha256:
            raise ValueError(
                "upscaling competition binding requires distinct reference/input"
            )
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError(
                "upscaling competition target dimensions must appear together"
            )
        return self

    @model_serializer(mode="wrap")
    def _preserve_v1_canonical_shape(self, handler: Any) -> dict[str, Any]:
        """Do not rewrite already-published v1 bundle bytes on deserialization.

        Geometry is mandatory for new v2 commitments. Historical v1 bindings have
        NULL/NULL in memory; omitting the two new keys keeps their pre-upgrade bundle
        digest byte-identical instead of silently orphaning an anchored digest.
        """
        payload = handler(self)
        if self.target_width is None and self.target_height is None:
            payload.pop("target_width", None)
            payload.pop("target_height", None)
        return payload


class AuditBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge_id: str
    #: The evaluation item this bundle audits; the score packet's item_id must
    #: match it (recompute checks IDENTITY_MISMATCH otherwise).
    item_id: str
    #: The miner whose output this bundle audits. None only for bundles not
    #: attributed to a specific miner (e.g. calibration runs); when set, the
    #: score packet's miner_hotkey must match it.
    miner_hotkey: str | None = None
    #: sha256 over the DAG_REVEAL artifact bytes, committed before dispatch. The
    #: DAG_REVEAL artifact MUST be the challenge commitment preimage JSON
    #: (ChallengeCommitment.preimage_bytes() in vidaio.challenge.commitment) —
    #: NOT the raw DAG JSON, whose digest would never match the commitment.
    commitment_hash: str = Field(pattern=SHA256_HEX_PATTERN)
    #: Finalized-chain proof that ``commitment_hash`` existed before dispatch.
    #: Optional only for legacy/report bundles; production finalization requires it.
    challenge_anchor: ChallengeAnchor | None = None
    #: Miner-signed artifact-v2 response whose signed request embeds the exact
    #: challenge anchor. Together these establish result-after-anchor chronology.
    miner_receipt: MinerArtifactReceipt | None = None
    stage: LifecycleStage
    challenge_input: ArtifactRef
    miner_output: ArtifactRef
    manifest: ArtifactRef
    score_packet: ArtifactRef
    reference_original: ArtifactRef | None = None
    dag_reveal: ArtifactRef | None = None
    competition_item: CompetitionItemBinding | None = None
    #: Sandbox-runner image identity that produced this competition output. It is
    #: absent for inference bundles. Competition authority/auditor code requires
    #: every subject's full item matrix to carry one stable identity; the archived
    #: baseline identity must equal the pre-enrollment commitment.
    execution_image_digest: str | None = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )
    scorer_version: str
    #: e.g. {"vmaf": "3.0.0", "ffmpeg": "7.1"} — pins for recompute parity.
    backend_versions: dict[str, str] = Field(default_factory=dict)
    #: ISO-8601 UTC, supplied by the caller (never generated in here).
    created_at: str

    @model_validator(mode="after")
    def _validate_completeness(self) -> "AuditBundle":
        slots: list[tuple[str, ArtifactRef | None, ArtifactKind]] = [
            ("challenge_input", self.challenge_input, ArtifactKind.CHALLENGE_INPUT),
            ("miner_output", self.miner_output, ArtifactKind.MINER_OUTPUT),
            ("manifest", self.manifest, ArtifactKind.MANIFEST),
            ("score_packet", self.score_packet, ArtifactKind.SCORE_PACKET),
            ("reference_original", self.reference_original, ArtifactKind.REFERENCE_ORIGINAL),
            ("dag_reveal", self.dag_reveal, ArtifactKind.DAG_REVEAL),
        ]
        for name, ref, expected_kind in slots:
            if ref is not None and ref.kind is not expected_kind:
                raise ValueError(
                    f"bundle slot {name!r} holds a {ref.kind.value} ref, "
                    f"expected {expected_kind.value}"
                )
        if self.stage is LifecycleStage.PRE_REVEAL:
            if self.reference_original is not None or self.dag_reveal is not None:
                raise ValueError(
                    "pre-reveal bundle must not carry reference_original or dag_reveal: "
                    "the holdout and the DAG seeds are revealed only at asset retirement"
                )
        elif self.stage is LifecycleStage.COMPETITION_SEALED:
            if self.reference_original is None or self.competition_item is None:
                raise ValueError(
                    "competition-sealed bundle requires reference_original and "
                    "competition_item"
                )
            if self.dag_reveal is not None:
                raise ValueError("competition-sealed bundle has no inference DAG reveal")
            if self.challenge_input.digest != self.competition_item.input_sha256:
                raise ValueError(
                    "competition_item input digest differs from challenge_input"
                )
            if (
                self.reference_original.digest
                != self.competition_item.reference_sha256
            ):
                raise ValueError(
                    "competition_item reference digest differs from reference_original"
                )
        else:  # POST_RETIREMENT — recomputable inference stage: everything required
            missing = [
                name
                for name, ref in (
                    ("reference_original", self.reference_original),
                    ("dag_reveal", self.dag_reveal),
                )
                if ref is None
            ]
            if missing:
                raise ValueError(
                    "post-retirement bundle must be fully recomputable; missing: "
                    + ", ".join(missing)
                )
        return self

    def bundle_digest(self) -> str:
        """Stable sha256 over the canonical JSON form of the whole bundle."""
        return sha256_hex(canonical_json_bytes(self.model_dump(mode="json")))


def build_bundle(
    *,
    challenge_id: str,
    item_id: str,
    commitment_hash: str,
    stage: LifecycleStage,
    challenge_input: ArtifactRef,
    miner_output: ArtifactRef,
    manifest: ArtifactRef,
    score_packet: ArtifactRef,
    scorer_version: str,
    created_at: str,
    miner_hotkey: str | None = None,
    reference_original: ArtifactRef | None = None,
    dag_reveal: ArtifactRef | None = None,
    competition_item: CompetitionItemBinding | None = None,
    execution_image_digest: str | None = None,
    backend_versions: dict[str, str] | None = None,
    challenge_anchor: ChallengeAnchor | None = None,
    miner_receipt: MinerArtifactReceipt | None = None,
) -> AuditBundle:
    """Build a validated bundle; raises on any stage-completeness violation."""
    return AuditBundle(
        challenge_id=challenge_id,
        item_id=item_id,
        miner_hotkey=miner_hotkey,
        commitment_hash=commitment_hash,
        challenge_anchor=challenge_anchor,
        miner_receipt=miner_receipt,
        stage=stage,
        challenge_input=challenge_input,
        miner_output=miner_output,
        manifest=manifest,
        score_packet=score_packet,
        reference_original=reference_original,
        dag_reveal=dag_reveal,
        competition_item=competition_item,
        execution_image_digest=execution_image_digest,
        scorer_version=scorer_version,
        backend_versions=backend_versions or {},
        created_at=created_at,
    )
