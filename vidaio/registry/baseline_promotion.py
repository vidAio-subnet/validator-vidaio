"""Automatic current-epoch CROWN -> schema-v14 executable baseline promotion.

The caller supplies only a snapshot digest as a selector.  Winner identity,
competition, track, prior baseline, archive address and audit matrix all come from
``VerifiedCrownEpochSource`` after that adapter has verified the finalized chain
anchor and independently rederived the CROWN transition.

Promotion has two durable phases:

1. latch the verified CROWN, which blocks the next competition for that track;
2. verify archives, build/rerun the exact submission, publish provenance, then
   atomically supersede/insert/resolve.

A build/rerun failure intentionally leaves the latch pending and the prior serving
baseline untouched.  The payout window is not registry state and is never mutated
here.  Retrying the same (snapshot, competition, track) key is idempotent.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import re
import sqlite3
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes
from vidaio.audit.store import (
    SEALED_KINDS,
    ArtifactKind,
    ArtifactRef,
    AuditStore,
    IntegrityError,
)
from vidaio.epoch import EPOCH_LOG_SCHEMA_VERSION
from vidaio.registry import baseline
from vidaio.registry.baseline import (
    BaselineRecord,
    BaselineRegistryError,
    PendingPromotionError,
    PromotionLatch,
)
from vidaio.registry.crown_source import VerifiedCrownEpoch, VerifiedCrownEpochSource
from vidaio.registry.registry import iso, transaction
from vidaio.tokenomics.breakthrough import contender_margin, qualifies_for_crown
from vidaio.tokenomics.config import TokenomicsConfig

CROWN_EPOCH_SCHEMA_VERSION = EPOCH_LOG_SCHEMA_VERSION
CROWN = "CROWN"
_SCORE_TOLERANCE = 1e-12
_CROWN_POLICY = TokenomicsConfig()
_SHA256 = re.compile(SHA256_HEX_PATTERN)


class CrownPromotionError(BaselineRegistryError):
    """Base class for verified-CROWN promotion failures."""


class CrownEpochNotVerifiedError(CrownPromotionError):
    """No trusted source could prove the requested snapshot's finalized anchor."""


class ForeignEpochSchemaError(CrownPromotionError):
    """The verified source returned evidence from a foreign epoch schema."""


class NonCrownEpochError(CrownPromotionError):
    """A current-schema epoch did not transition the reward window to CROWN."""


class CrownProofMismatchError(CrownPromotionError):
    """The verified DTO is internally inconsistent with its selector/anchor/baseline."""


class CrownArchiveError(CrownPromotionError):
    """The exact winner submission or audit matrix is missing/corrupt."""


class BaselineBuildError(CrownPromotionError):
    """The exact crowned submission could not be built and rerun."""


class BaselineRerunResult(BaseModel):
    """Attested result of rebuilding and rerunning the exact crowned archive."""

    model_config = ConfigDict(frozen=True)

    submission_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    source_image_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    built_image_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    reproduced_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    receipt: ArtifactRef


@runtime_checkable
class BaselineBuildRunner(Protocol):
    """Trusted fresh-build adapter; failure is reported by raising."""

    def build_and_rerun(
        self,
        proof: VerifiedCrownEpoch,
        serving_baseline: BaselineRecord,
    ) -> BaselineRerunResult:
        """Build the archived submission and reproduce the crowned score.

        Implementations must use ``proof.winner_submission`` exactly, run the same
        committed item matrix in a fresh sandbox, archive a receipt, and raise on
        build, isolation, scoring or parity failure.
        """
        ...


class BaselinePromotionPipeline:
    """Latch, build/rerun, archive and atomically activate a verified CROWN."""

    def __init__(
        self,
        store: AuditStore,
        epochs: VerifiedCrownEpochSource,
        runner: BaselineBuildRunner,
    ) -> None:
        self._store = store
        self._epochs = epochs
        self._runner = runner

    def latch_verified_crown(
        self,
        conn: sqlite3.Connection,
        *,
        snapshot_digest: str,
        now: datetime,
    ) -> PromotionLatch:
        """Persist the next-competition interlock without building yet."""
        proof = self._verified_proof(snapshot_digest)
        promoted = _promoted_by_key(conn, proof)
        if promoted is not None:
            latch = baseline.promotion_latch_by_key(
                conn,
                snapshot_digest=proof.snapshot_digest,
                competition_id=proof.competition_id,
                track=proof.track,
            )
            if latch is None:
                raise CrownProofMismatchError(
                    "promoted baseline has no matching immutable promotion latch"
                )
            return latch
        return self._latch(conn, proof, now)

    def promote_verified_crown(
        self,
        conn: sqlite3.Connection,
        *,
        snapshot_digest: str,
        now: datetime,
    ) -> BaselineRecord:
        """Promote automatically; retrying the same CROWN returns the same row."""
        proof = self._verified_proof(snapshot_digest)
        existing = _promoted_by_key(conn, proof)
        if existing is not None:
            return existing

        latch = self._latch(conn, proof, now)
        if latch.status == "promoted":
            existing = _promoted_by_key(conn, proof)
            if existing is None:
                raise CrownProofMismatchError(
                    "promotion latch is resolved but its baseline row is absent"
                )
            return existing

        serving = baseline.current_baseline(conn, proof.track)
        self._require_compared_baseline(proof, serving)
        self._verify_crown_archives(proof)

        try:
            rerun = self._runner.build_and_rerun(proof, serving)
        except BaselineBuildError:
            raise
        except Exception as exc:
            raise BaselineBuildError(
                f"fresh build/rerun failed for crowned {proof.track} submission "
                f"{proof.winner_submission.digest}: {exc}"
            ) from exc
        self._verify_rerun(proof, rerun)
        self._verify_ref(rerun.receipt, what="baseline build/rerun receipt")

        provenance = self._archive_provenance(proof, rerun)
        # Publication is deliberately after successful build/rerun and before the
        # database handover.  A failed build exposes nothing and changes no serving
        # row; a later database crash is safely retryable from these content addresses.
        for ref, what in (
            (proof.winner_submission, "crowned winner submission"),
            (rerun.receipt, "baseline build/rerun receipt"),
            (provenance, "baseline promotion provenance"),
        ):
            self._publish(ref, what=what)

        return self._activate(conn, proof, rerun, provenance, now)

    # ---- trusted proof boundary -------------------------------------------------

    def _verified_proof(self, selector: str) -> VerifiedCrownEpoch:
        if _SHA256.fullmatch(selector) is None:
            raise CrownEpochNotVerifiedError(
                "snapshot selector must be lowercase sha256 hex"
            )
        proof = self._epochs.verified_crown(selector)
        if proof is None:
            raise CrownEpochNotVerifiedError(
                f"snapshot {selector} has no independently verified finalized "
                f"schema-v{CROWN_EPOCH_SCHEMA_VERSION} CROWN anchor"
            )
        if proof.schema_version != CROWN_EPOCH_SCHEMA_VERSION:
            raise ForeignEpochSchemaError(
                f"snapshot {selector} uses schema v{proof.schema_version}; "
                f"baseline promotion requires epoch schema "
                f"v{CROWN_EPOCH_SCHEMA_VERSION}"
            )
        if proof.reward_window_state != CROWN:
            raise NonCrownEpochError(
                f"snapshot {selector} reward state is {proof.reward_window_state!r}; "
                "only the machine-derived CROWN transition promotes"
            )
        if proof.snapshot_digest != selector:
            raise CrownProofMismatchError(
                f"verified source returned snapshot {proof.snapshot_digest}, "
                f"not requested {selector}"
            )
        # The trusted source verifies canonical log bytes against the on-chain
        # archive. Repeating the equality here makes accidental adapter drift fail
        # closed without introducing a meaningless `anchor_verified` flag.
        if proof.anchor_digest != proof.snapshot_digest:
            raise CrownProofMismatchError(
                f"anchor digest {proof.anchor_digest} does not bind canonical "
                f"snapshot {proof.snapshot_digest}"
            )
        if proof.track not in baseline.SUPPORTED_TRACKS:
            raise CrownProofMismatchError(
                f"CROWN names unsupported track {proof.track!r}"
            )
        # Use the exact same Decimal-backed representation and inclusive threshold
        # as the authority/weightsetter fold.  Recomputing this boundary with raw
        # binary floats makes an exact 0.100 -> 0.105 improvement appear slightly
        # below 5% and can strand an otherwise valid CROWN promotion.
        derived_margin = contender_margin(proof.baseline_score, proof.winner_score)
        if derived_margin is None:  # defensive: the DTO already requires baseline > 0
            raise CrownProofMismatchError("CROWN margin is not derivable")
        if not math.isclose(
            derived_margin,
            proof.winner_margin,
            rel_tol=0.0,
            abs_tol=_SCORE_TOLERANCE,
        ):
            raise CrownProofMismatchError(
                f"CROWN margin {proof.winner_margin} does not rederive from winner "
                f"score {proof.winner_score} and baseline score {proof.baseline_score}"
            )
        if not qualifies_for_crown(
            _CROWN_POLICY,
            proof.baseline_score,
            proof.winner_score,
        ):
            raise NonCrownEpochError(
                f"derived margin {derived_margin} is below canonical CROWN threshold "
                f"{_CROWN_POLICY.breakthrough_margin_floor}"
            )
        return proof

    # ---- durable latch ----------------------------------------------------------

    def _latch(
        self,
        conn: sqlite3.Connection,
        proof: VerifiedCrownEpoch,
        now: datetime,
    ) -> PromotionLatch:
        with transaction(conn):
            existing = baseline.promotion_latch_by_key(
                conn,
                snapshot_digest=proof.snapshot_digest,
                competition_id=proof.competition_id,
                track=proof.track,
            )
            if existing is not None:
                self._require_latch_matches(existing, proof)
                return existing
            other = baseline.pending_promotion(conn, proof.track)
            if other is not None:
                raise PendingPromotionError(
                    f"{proof.track} already has pending CROWN {other.snapshot_digest} "
                    f"from competition {other.competition_id!r}"
                )
            serving = baseline.current_baseline(conn, proof.track)
            self._require_compared_baseline(proof, serving)
            ts = iso(now)
            cur = conn.execute(
                """INSERT INTO baseline_promotion_latches
                   (track, snapshot_digest, competition_id, epoch_id, cycle,
                    anchor_block, anchor_digest, winner_uid, winner_hotkey,
                    compared_baseline_version, compared_baseline_digest,
                    status, latched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    proof.track,
                    proof.snapshot_digest,
                    proof.competition_id,
                    proof.epoch_id,
                    proof.cycle,
                    proof.anchor_block,
                    proof.anchor_digest,
                    proof.winner_uid,
                    proof.winner_hotkey,
                    proof.baseline_version,
                    proof.baseline_artifact_digest,
                    ts,
                ),
            )
            baseline._record_event(
                conn,
                proof.track,
                "crown_promotion_latched",
                None,
                now,
                snapshot_digest=proof.snapshot_digest,
                payload={
                    "anchor_block": proof.anchor_block,
                    "competition_id": proof.competition_id,
                    "cycle": proof.cycle,
                    "epoch_id": proof.epoch_id,
                    "latch_id": int(cur.lastrowid),
                    "winner_hotkey": proof.winner_hotkey,
                    "winner_uid": proof.winner_uid,
                },
            )
            row = conn.execute(
                "SELECT * FROM baseline_promotion_latches WHERE latch_id = ?",
                (int(cur.lastrowid),),
            ).fetchone()
        assert row is not None
        return PromotionLatch.from_row(row)

    @staticmethod
    def _require_latch_matches(
        latch: PromotionLatch, proof: VerifiedCrownEpoch
    ) -> None:
        expected = (
            proof.track,
            proof.snapshot_digest,
            proof.competition_id,
            proof.epoch_id,
            proof.cycle,
            proof.anchor_block,
            proof.anchor_digest,
            proof.winner_uid,
            proof.winner_hotkey,
            proof.baseline_version,
            proof.baseline_artifact_digest,
        )
        actual = (
            latch.track,
            latch.snapshot_digest,
            latch.competition_id,
            latch.epoch_id,
            latch.cycle,
            latch.anchor_block,
            latch.anchor_digest,
            latch.winner_uid,
            latch.winner_hotkey,
            latch.compared_baseline_version,
            latch.compared_baseline_digest,
        )
        if actual != expected:
            raise CrownProofMismatchError(
                "verified CROWN conflicts with its existing immutable promotion latch"
            )

    @staticmethod
    def _require_compared_baseline(
        proof: VerifiedCrownEpoch, serving: BaselineRecord | None
    ) -> None:
        if serving is None:
            raise CrownProofMismatchError(
                f"{proof.track} has no active genesis/serving baseline"
            )
        if (
            serving.version != proof.baseline_version
            or serving.artifact_digest != proof.baseline_artifact_digest
        ):
            raise CrownProofMismatchError(
                f"CROWN compared {proof.track} baseline v{proof.baseline_version} "
                f"({proof.baseline_artifact_digest}), but registry serves "
                f"v{serving.version} ({serving.artifact_digest})"
            )

    # ---- archive/build verification --------------------------------------------

    def _verify_crown_archives(self, proof: VerifiedCrownEpoch) -> None:
        self._verify_ref(proof.winner_submission, what="crowned winner submission")
        for item in proof.audit_items:
            self._verify_ref(
                item.score_packet,
                what=f"winner item {item.item_index} score packet",
                expected_kind=ArtifactKind.SCORE_PACKET,
            )
            self._verify_ref(
                item.audit_bundle,
                what=f"winner item {item.item_index} audit bundle",
                expected_kind=ArtifactKind.AUDIT_BUNDLE,
            )

    def _verify_ref(
        self,
        ref: ArtifactRef,
        *,
        what: str,
        expected_kind: ArtifactKind | None = None,
    ) -> None:
        if expected_kind is not None and ref.kind is not expected_kind:
            raise CrownArchiveError(
                f"{what} has kind {ref.kind.value}, expected {expected_kind.value}"
            )
        if ref.byte_size <= 0:
            raise CrownArchiveError(f"{what} is empty")
        digest = hashlib.sha256()
        size = 0
        try:
            with contextlib.closing(self._store.open_stream(ref)) as stream:
                while chunk := stream.read(1 << 20):
                    size += len(chunk)
                    if size > ref.byte_size:
                        raise IntegrityError(
                            f"{what} exceeds committed size {ref.byte_size}"
                        )
                    digest.update(chunk)
        except (FileNotFoundError, OSError, IntegrityError) as exc:
            raise CrownArchiveError(
                f"{what} is not verifiably archived: {exc}"
            ) from exc
        if size != ref.byte_size or digest.hexdigest() != ref.digest:
            raise CrownArchiveError(
                f"{what} bytes do not match its content-addressed reference"
            )

    @staticmethod
    def _verify_rerun(proof: VerifiedCrownEpoch, rerun: BaselineRerunResult) -> None:
        if rerun.submission_digest != proof.winner_submission.digest:
            raise BaselineBuildError(
                "build/rerun used a submission other than the crowned archive"
            )
        if rerun.source_image_digest != proof.winner_image_digest:
            raise BaselineBuildError(
                "build/rerun source image provenance differs from CROWN evidence"
            )
        if rerun.built_image_digest != rerun.source_image_digest:
            raise BaselineBuildError(
                "fresh build changed the stable logical image identity; provider "
                "object ids must be recorded separately and never replace the "
                "CROWN source identity"
            )
        if not math.isclose(
            rerun.reproduced_score,
            proof.winner_score,
            rel_tol=0.0,
            abs_tol=_SCORE_TOLERANCE,
        ):
            raise BaselineBuildError(
                f"fresh rerun reproduced {rerun.reproduced_score}, but anchored "
                f"CROWN score is {proof.winner_score}"
            )

    def _archive_provenance(
        self, proof: VerifiedCrownEpoch, rerun: BaselineRerunResult
    ) -> ArtifactRef:
        payload = {
            "proof": proof.canonical_provenance_obj(),
            "rerun": rerun.model_dump(mode="json"),
        }
        try:
            return self._store.put(canonical_json_bytes(payload), ArtifactKind.MANIFEST)
        except (OSError, IntegrityError) as exc:
            raise CrownArchiveError(
                f"could not archive baseline promotion provenance: {exc}"
            ) from exc

    def _publish(self, ref: ArtifactRef, *, what: str) -> None:
        if ref.kind not in SEALED_KINDS:
            self._verify_ref(ref, what=what)
            return
        try:
            self._store.release(ref)
            if not self._store.is_released(ref):
                raise IntegrityError("release marker/public copy is absent")
        except (FileNotFoundError, OSError, IntegrityError) as exc:
            raise CrownArchiveError(f"{what} could not be published: {exc}") from exc

    # ---- atomic handover --------------------------------------------------------

    def _activate(
        self,
        conn: sqlite3.Connection,
        proof: VerifiedCrownEpoch,
        rerun: BaselineRerunResult,
        provenance: ArtifactRef,
        now: datetime,
    ) -> BaselineRecord:
        with transaction(conn):
            existing = _promoted_by_key(conn, proof)
            if existing is not None:
                return existing
            latch = baseline.promotion_latch_by_key(
                conn,
                snapshot_digest=proof.snapshot_digest,
                competition_id=proof.competition_id,
                track=proof.track,
            )
            if latch is None or latch.status != "pending":
                raise CrownProofMismatchError(
                    "CROWN activation requires its unresolved immutable latch"
                )
            self._require_latch_matches(latch, proof)
            serving = baseline.current_baseline(conn, proof.track)
            self._require_compared_baseline(proof, serving)
            assert serving is not None
            version = baseline._next_version(conn, proof.track)
            ts = iso(now)
            conn.execute(
                "UPDATE baselines SET status = 'superseded', updated_at = ? "
                "WHERE baseline_id = ?",
                (ts, serving.baseline_id),
            )
            cur = conn.execute(
                """INSERT INTO baselines
                   (track, version, artifact_digest, artifact_kind, artifact_bytes,
                    image_digest, provenance_digest, provenance_kind,
                    provenance_bytes, repo_url, commit_sha, tree_sha, source_kind,
                    source_epoch_id,
                    source_snapshot_digest, source_anchor_block,
                    source_anchor_digest, source_competition_id, source_cycle,
                    winner_uid, winner_hotkey, winner_score, winner_margin,
                    compared_baseline_version, compared_baseline_score,
                    compared_baseline_digest, status,
                    activated_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'crown', ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (
                    proof.track,
                    version,
                    proof.winner_submission.digest,
                    proof.winner_submission.kind.value,
                    proof.winner_submission.byte_size,
                    rerun.built_image_digest,
                    provenance.digest,
                    provenance.kind.value,
                    provenance.byte_size,
                    proof.winner_repo_url,
                    proof.winner_commit_sha,
                    proof.winner_tree_sha,
                    proof.epoch_id,
                    proof.snapshot_digest,
                    proof.anchor_block,
                    proof.anchor_digest,
                    proof.competition_id,
                    proof.cycle,
                    proof.winner_uid,
                    proof.winner_hotkey,
                    proof.winner_score,
                    proof.winner_margin,
                    proof.baseline_version,
                    proof.baseline_score,
                    proof.baseline_artifact_digest,
                    ts,
                    ts,
                ),
            )
            promoted_id = int(cur.lastrowid)
            conn.execute(
                """UPDATE baseline_promotion_latches
                      SET status = 'promoted', promoted_baseline_id = ?, resolved_at = ?
                    WHERE latch_id = ? AND status = 'pending'""",
                (promoted_id, ts, latch.latch_id),
            )
            baseline._record_event(
                conn,
                proof.track,
                "baseline_promoted_from_crown",
                version,
                now,
                snapshot_digest=proof.snapshot_digest,
                payload={
                    "baseline_id": promoted_id,
                    "competition_id": proof.competition_id,
                    "previous_version": serving.version,
                    "provenance_digest": provenance.digest,
                    "winner_hotkey": proof.winner_hotkey,
                    "winner_uid": proof.winner_uid,
                },
            )
            baseline._record_event(
                conn,
                proof.track,
                "crown_promotion_resolved",
                version,
                now,
                snapshot_digest=proof.snapshot_digest,
                payload={"baseline_id": promoted_id, "latch_id": latch.latch_id},
            )
            result = baseline.baseline_version(conn, proof.track, version)
        assert result is not None
        return result


def _promoted_by_key(
    conn: sqlite3.Connection, proof: VerifiedCrownEpoch
) -> BaselineRecord | None:
    row = conn.execute(
        """SELECT * FROM baselines
            WHERE source_kind = 'crown'
              AND source_snapshot_digest = ?
              AND source_competition_id = ?
              AND track = ?""",
        proof.idempotence_key,
    ).fetchone()
    return BaselineRecord.from_row(row) if row is not None else None


__all__ = [
    "CROWN_EPOCH_SCHEMA_VERSION",
    "BaselineBuildError",
    "BaselineBuildRunner",
    "BaselinePromotionPipeline",
    "BaselineRerunResult",
    "CrownArchiveError",
    "CrownEpochNotVerifiedError",
    "CrownProofMismatchError",
    "CrownPromotionError",
    "ForeignEpochSchemaError",
    "NonCrownEpochError",
]
