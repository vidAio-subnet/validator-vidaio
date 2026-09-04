"""Completed competition database rows -> auditable epoch-log inputs.

This is the production bridge between the competition orchestrator and the epoch
authority.  It deliberately ignores ``final_rank`` and human-review eligibility:
economic ordering is derived from the exact, content-addressed score packets.  A
packet or bundle that is absent, malformed, or inconsistent makes the evidence
unpublishable rather than turning into an unauditable payout.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from vidaio.audit.commitments import (
    build_competition_commitment,
    load_competition_commitment,
    pin_git_sha,
    reward_parameter_digest,
)
from vidaio.audit.bundle import AuditBundle
from vidaio.audit.store import ArtifactKind, ArtifactRef, AuditStore, backend_key
from vidaio.authority.finalizer import ScoredItem
from vidaio.competition import repository as repo
from vidaio.competition.economic_result import (
    CompetitionDedupCandidate,
    competition_dedup_losers,
    derive_competition_result,
)
from vidaio.competition.orchestrator.persistence import (
    EVENT_COMMITMENT_ANCHORED,
    latest_verified_anchor_receipt,
)
from vidaio.competition.orchestrator.results import competition_cycle, completed_at
from vidaio.competition.states import Phase
from vidaio.epoch.log import (
    CompetitionAuditItem,
    CompetitionAuditSubject,
    CompetitionInput,
    MinerCensusEntry,
)
from vidaio.scoring.result import ItemScore
from vidaio.tokenomics.breakthrough import qualifies_for_crown
from vidaio.tokenomics.state import CompetitionResult
from vidaio.tokenomics.config import TokenomicsConfig

_MAX_SCORE_PACKET_BYTES = 4 * 1024 * 1024
_MAX_AUDIT_BUNDLE_BYTES = 4 * 1024 * 1024


class CompetitionEvidenceError(ValueError):
    """Persisted competition state cannot produce a complete auditable result."""


@dataclass(frozen=True, slots=True)
class CompetitionEpochEvidence:
    """Everything the finalizer needs for one score-derived competition result."""

    competition_input: CompetitionInput
    scored_items: tuple[ScoredItem, ...]
    packet_scores: Mapping[str, float]
    result: CompetitionResult


_SUBMISSION_ARCHIVED_EVENT = "contender_submission_archived"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _archived_submissions(
    conn: sqlite3.Connection, competition_id: str
) -> dict[int, ArtifactRef]:
    """Resolve each subject's immutable sealed source archive from append-only events."""
    rows = conn.execute(
        "SELECT payload_json FROM events WHERE competition_id = ? AND event_type = ? "
        "ORDER BY event_id",
        (competition_id, _SUBMISSION_ARCHIVED_EVENT),
    ).fetchall()
    archived: dict[int, ArtifactRef] = {}
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
            contender_id = int(payload["contender_id"])
            digest = str(payload["digest"])
            byte_size = int(payload["byte_size"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CompetitionEvidenceError(
                f"competition {competition_id!r} has a malformed submission-archive event"
            ) from exc
        if contender_id < 0 or byte_size <= 0 or _SHA256_RE.fullmatch(digest) is None:
            raise CompetitionEvidenceError(
                f"competition {competition_id!r} has an invalid submission archive "
                f"identity for contender {contender_id}"
            )
        ref = ArtifactRef(
            digest=digest,
            kind=ArtifactKind.SUBMISSION_ARCHIVE,
            byte_size=byte_size,
            backend_key=backend_key(ArtifactKind.SUBMISSION_ARCHIVE, digest),
        )
        prior = archived.get(contender_id)
        if prior is not None and prior != ref:
            raise CompetitionEvidenceError(
                f"competition {competition_id!r} records conflicting sealed source "
                f"archives for contender {contender_id}"
            )
        archived[contender_id] = ref
    return archived


def _verify_archived_ref(store: AuditStore, ref: ArtifactRef, *, what: str) -> None:
    """Stream-hash one potentially sealed archive through the configured store."""
    digest = hashlib.sha256()
    size = 0
    try:
        with contextlib.closing(store.open_stream(ref)) as stream:
            while chunk := stream.read(1 << 20):
                size += len(chunk)
                if size > ref.byte_size:
                    raise CompetitionEvidenceError(
                        f"{what} exceeds its committed byte size {ref.byte_size}"
                    )
                digest.update(chunk)
    except CompetitionEvidenceError:
        raise
    except Exception as exc:
        raise CompetitionEvidenceError(
            f"{what} is absent or unreadable from the audit store: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if size != ref.byte_size or digest.hexdigest() != ref.digest:
        raise CompetitionEvidenceError(
            f"{what} bytes do not match their content-addressed archive reference"
        )


def _persisted_anchor_receipt(
    conn: sqlite3.Connection,
    *,
    competition_id: str,
    commitment_root: str,
    expected_payload: bytes,
    enrollment_start: datetime,
) -> dict[str, object]:
    """Require the complete atomic pre-enrollment receipt persisted by anchoring.

    This validates the database hand-off before any receipt enters epoch evidence.
    Independent authority/auditor archive reads are a separate boundary; a
    self-consistent but incomplete local event is never enough to earn.
    """

    receipt = latest_verified_anchor_receipt(conn, competition_id)
    if receipt is None:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} has no independently finalized/archive-"
            "verified pre-enrollment anchor receipt"
        )
    required = {
        "root",
        "anchor_netuid",
        "payload_hex",
        "payload_digest",
        "anchor_block",
        "anchor_block_hash",
        "finalized_block",
        "archive_verified",
        "verified_at",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt is incomplete: "
            + ", ".join(missing)
        )
    if receipt["archive_verified"] is not True:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt is not archive-verified"
        )
    if receipt["root"] != commitment_root:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt root does not match the "
            "earning commitment root"
        )
    expected_hex = expected_payload.hex()
    if receipt["payload_hex"] != expected_hex:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt does not carry the exact "
            "root-bound raw payload"
        )
    if receipt["payload_digest"] != hashlib.sha256(expected_payload).hexdigest():
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt payload digest does not "
            "bind its exact raw payload"
        )

    integers: dict[str, int] = {}
    for name in ("anchor_netuid", "anchor_block", "finalized_block"):
        value = receipt[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CompetitionEvidenceError(
                f"competition {competition_id!r} anchor receipt {name} must be a "
                "non-negative integer"
            )
        integers[name] = value
    if integers["finalized_block"] < integers["anchor_block"]:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt claims finality before "
            "inclusion"
        )
    block_hash = receipt["anchor_block_hash"]
    if not isinstance(block_hash, str) or _SHA256_RE.fullmatch(block_hash) is None:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt has a non-canonical "
            "inclusion block hash"
        )
    try:
        verified_at = datetime.fromisoformat(str(receipt["verified_at"]))
    except ValueError as exc:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt verified_at is malformed"
        ) from exc
    if verified_at.tzinfo is None or verified_at.utcoffset() is None:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt verified_at is timezone-naive"
        )
    if verified_at >= enrollment_start:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt was not recorded before "
            "enrollment start"
        )

    anchor_event = conn.execute(
        "SELECT event_id FROM events WHERE competition_id = ? AND event_type = ? "
        "ORDER BY event_id DESC LIMIT 1",
        (competition_id, EVENT_COMMITMENT_ANCHORED),
    ).fetchone()
    enrollment_event = conn.execute(
        "SELECT event_id FROM events WHERE competition_id = ? AND event_type = "
        "'transition' AND to_phase = ? ORDER BY event_id ASC LIMIT 1",
        (competition_id, Phase.ENROLLING.value),
    ).fetchone()
    if anchor_event is None or enrollment_event is None:
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} lacks the receipt/enrollment lifecycle "
            "events required to prove pre-enrollment ordering"
        )
    if int(anchor_event["event_id"]) >= int(enrollment_event["event_id"]):
        raise CompetitionEvidenceError(
            f"competition {competition_id!r} anchor receipt does not precede its "
            "enrollment transition"
        )
    return receipt


def latest_completed_competition_id(
    conn: sqlite3.Connection, *, through_time: datetime | None = None
) -> str | None:
    """Latest terminal completion event visible at the cutoff.

    Event order, not competition creation/start order, is the immutable global
    result order.  ``event_id`` is also the deterministic tie-break when multiple
    transitions carry the same explicit completion timestamp.
    """
    rows = conn.execute(
        """SELECT e.competition_id, e.created_at
           FROM events AS e
           JOIN competitions AS c ON c.competition_id = e.competition_id
           WHERE c.status = ?
             AND e.from_phase = ?
             AND e.to_phase = ?
           ORDER BY e.event_id DESC""",
        (
            Phase.COMPLETED.value,
            Phase.AWAITING_END_TIME.value,
            Phase.COMPLETED.value,
        ),
    ).fetchall()
    for row in rows:
        competition_id = str(row["competition_id"])
        completion_time = repo.parse_ts(row["created_at"])
        if completion_time is None:
            raise CompetitionEvidenceError(
                f"competition {competition_id!r} has a malformed terminal completion time"
            )
        if through_time is None or completion_time <= through_time:
            return competition_id
    return None


def _performance_rows(
    conn: sqlite3.Connection, competition_id: str, contender_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT ph.*, ei.item_index, ei.challenge_id, ei.scoring_item_id,
                         ei.threshold_commitment, ei.input_sha256, ei.input_bytes,
                         ei.reference_sha256, ei.reference_bytes, ei.upscale_factor,
                         ei.target_width, ei.target_height, ei.item_commitment
           FROM performance_history ph
           JOIN evaluation_items ei ON ei.item_id = ph.item_id
                                  AND ei.competition_id = ph.competition_id
           WHERE ph.competition_id = ? AND ph.contender_id = ?
           ORDER BY ei.item_index""",
        (competition_id, contender_id),
    ).fetchall()


def _require_bundle(
    store: AuditStore,
    digest: str,
    *,
    row: sqlite3.Row,
    subject_id: str,
    expected_hotkey: str | None,
    manifest_digest: str,
    track: str,
) -> AuditBundle:
    """Require a canonical, identity-bound and presently resolvable audit bundle."""
    try:
        raw = store.get_digest_limited(
            ArtifactKind.AUDIT_BUNDLE,
            digest,
            max_bytes=_MAX_AUDIT_BUNDLE_BYTES,
        )
        bundle = AuditBundle.model_validate_json(raw)
    except Exception as exc:
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} references audit bundle {digest} "
            f"that is absent or malformed: {type(exc).__name__}: {exc}"
        ) from exc
    if bundle.bundle_digest() != digest:
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} audit bundle {digest} is not "
            "canonical for its content digest"
        )
    expected_identity = (
        str(row["challenge_id"]),
        str(row["scoring_item_id"]),
        expected_hotkey,
        str(row["score_packet_digest"]),
        manifest_digest,
        str(row["threshold_commitment"]),
    )
    actual_identity = (
        bundle.challenge_id,
        bundle.item_id,
        bundle.miner_hotkey,
        bundle.score_packet.digest,
        bundle.manifest.digest,
        bundle.commitment_hash,
    )
    if actual_identity != expected_identity:
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} audit bundle {digest} identity is "
            f"{actual_identity!r}, expected {expected_identity!r}"
        )
    if track == "upscaling":
        binding = bundle.competition_item
        expected_binding = (
            int(row["item_index"]),
            str(row["input_sha256"]),
            str(row["reference_sha256"]),
            int(row["upscale_factor"]),
            None if row["target_width"] is None else int(row["target_width"]),
            None if row["target_height"] is None else int(row["target_height"]),
            str(row["item_commitment"]),
        )
        actual_binding = (
            None
            if binding is None
            else (
                binding.item_index,
                binding.input_sha256,
                binding.reference_sha256,
                binding.upscale_factor,
                binding.target_width,
                binding.target_height,
                binding.item_commitment,
            )
        )
        if (
            bundle.stage.value != "competition_sealed"
            or actual_binding != expected_binding
        ):
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} audit bundle {digest} has "
                "no valid manifest-bound upscaling item preimage"
            )
        reference = bundle.reference_original
        if (
            reference is None
            or reference.digest != row["reference_sha256"]
            or reference.byte_size != row["reference_bytes"]
        ):
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} audit bundle {digest} does "
                "not bind the pristine reference artifact"
            )
        if not store.is_released(reference):
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} pristine reference "
                f"{reference.digest} is not publicly released"
            )
    for ref in (
        bundle.challenge_input,
        bundle.miner_output,
        bundle.manifest,
        bundle.score_packet,
        bundle.reference_original,
        bundle.dag_reveal,
    ):
        if ref is not None and not store.exists(ref):
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} audit bundle {digest} "
                f"references missing {ref.kind.value} artifact {ref.digest}"
            )
    return bundle


def _read_packet(
    store: AuditStore,
    row: sqlite3.Row,
    *,
    track: str,
    expected_hotkey: str | None,
    subject_id: str,
) -> ItemScore:
    digest = str(row["score_packet_digest"])
    try:
        payload = store.get_digest_limited(
            ArtifactKind.SCORE_PACKET,
            digest,
            max_bytes=_MAX_SCORE_PACKET_BYTES,
        )
        packet = ItemScore.model_validate_json(payload)
    except Exception as exc:
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} score packet {digest} is "
            f"unreadable or malformed: {type(exc).__name__}: {exc}"
        ) from exc

    expected_identity = (
        str(row["challenge_id"]),
        str(row["scoring_item_id"]),
        track,
        expected_hotkey,
    )
    actual_identity = (
        packet.challenge_id,
        packet.item_id,
        packet.track,
        packet.miner_hotkey,
    )
    if actual_identity != expected_identity:
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} packet {digest} identity is "
            f"{actual_identity!r}, expected {expected_identity!r}"
        )
    stored_score = float(row["item_score"])
    if not math.isclose(packet.score, stored_score, rel_tol=0.0, abs_tol=1e-12):
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} packet {digest} score "
            f"{packet.score} disagrees with its packet-bound DB value {stored_score}"
        )
    if bool(row["valid"]) != packet.gate_passed:
        raise CompetitionEvidenceError(
            f"competition subject {subject_id!r} packet {digest} gate outcome "
            "disagrees with its packet-bound DB value"
        )
    return packet


def build_competition_epoch_evidence(
    conn: sqlite3.Connection,
    *,
    census_by_hotkey: Mapping[str, MinerCensusEntry],
    store: AuditStore,
    tokenomics: TokenomicsConfig | None = None,
    competition_id: str | None = None,
    through_time: datetime | None = None,
    after_cycle: int | None = None,
) -> CompetitionEpochEvidence | None:
    """Build the latest completed, fully auditable economic competition input.

    Only machine-accepted (``BUILT``) contenders that remain registered in the
    close-block census can receive emissions.  Manual review flags and stored
    ranking columns are intentionally ignored.  Every included subject must cover
    the exact evaluation-item matrix; incomplete evidence fails closed.
    """
    if through_time is None:
        raise CompetitionEvidenceError(
            "competition economics require the timezone-aware epoch close-block time; "
            "a database or process-local clock is not an economic input"
        )
    if through_time.tzinfo is None or through_time.utcoffset() is None:
        raise CompetitionEvidenceError("epoch close-block time must be timezone-aware")
    if after_cycle is not None and (
        isinstance(after_cycle, bool)
        or not isinstance(after_cycle, int)
        or after_cycle < 0
    ):
        raise CompetitionEvidenceError("after_cycle must be a non-negative integer")
    selected = competition_id or latest_completed_competition_id(
        conn, through_time=through_time
    )
    if selected is None:
        return None
    malformed_census_keys = sorted(
        hotkey
        for hotkey, entry in census_by_hotkey.items()
        if hotkey != entry.hotkey
    )
    if malformed_census_keys:
        raise CompetitionEvidenceError(
            "close-block census mapping keys do not match their committed hotkeys: "
            f"{malformed_census_keys}"
        )
    selected_cycle = competition_cycle(conn, selected)
    if after_cycle is not None and selected_cycle <= after_cycle:
        return None
    competition = repo.get_competition(conn, selected)
    if competition is None or competition.status is not Phase.COMPLETED:
        raise CompetitionEvidenceError(
            f"competition {selected!r} is not a completed competition"
        )
    manifest = repo.get_manifest(conn, selected)
    if manifest.manifest_digest() != competition.manifest_digest:
        raise CompetitionEvidenceError(
            f"competition {selected!r} manifest bytes do not match persisted digest"
        )
    if competition.commitment_root is None:
        raise CompetitionEvidenceError(
            f"competition {selected!r} completed without a pre-enrollment commitment root"
        )
    try:
        commitment = load_competition_commitment(
            store, competition.commitment_root
        )
    except Exception as exc:
        raise CompetitionEvidenceError(
            f"competition {selected!r} commitment root is not publicly openable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    commitment_payload = build_competition_commitment(commitment)
    if commitment_payload.root != competition.commitment_root:
        raise CompetitionEvidenceError(
            f"competition {selected!r} commitment preimage does not re-open its "
            "persisted root"
        )
    anchor_receipt = _persisted_anchor_receipt(
        conn,
        competition_id=selected,
        commitment_root=competition.commitment_root,
        expected_payload=commitment_payload.payload,
        enrollment_start=manifest.start_time,
    )
    if manifest.baseline is None:
        raise CompetitionEvidenceError(
            f"competition {selected!r} has no non-earning archived baseline"
        )
    baseline_artifact = ArtifactRef(
        digest=manifest.baseline.artifact_digest,
        kind=ArtifactKind.SUBMISSION_ARCHIVE,
        byte_size=manifest.baseline.artifact_bytes,
        backend_key=backend_key(
            ArtifactKind.SUBMISSION_ARCHIVE, manifest.baseline.artifact_digest
        ),
    )
    baseline_provenance = ArtifactRef(
        digest=manifest.baseline.provenance_digest,
        kind=ArtifactKind.MANIFEST,
        byte_size=manifest.baseline.provenance_bytes,
        backend_key=backend_key(
            ArtifactKind.MANIFEST, manifest.baseline.provenance_digest
        ),
    )
    _verify_archived_ref(store, baseline_artifact, what="active baseline executable")
    _verify_archived_ref(store, baseline_provenance, what="active baseline provenance")
    expected_commitment = {
        "manifest_digest": competition.manifest_digest,
        "baseline_version": manifest.baseline.version,
        "baseline_artifact_digest": manifest.baseline.artifact_digest,
        "baseline_provenance_digest": manifest.baseline.provenance_digest,
        "baseline_tree_digest": pin_git_sha(manifest.baseline.tree_sha),
        "dataset_selection_seed_commitment": manifest.scoring_seed_commitment,
    }
    if tokenomics is not None:
        expected_commitment["reward_param_digest"] = reward_parameter_digest(
            tokenomics
        )
    for field, expected in expected_commitment.items():
        observed = getattr(commitment, field)
        if observed != expected:
            raise CompetitionEvidenceError(
                f"competition {selected!r} anchored {field} {observed} does not "
                f"match executed/persisted {expected}"
            )
    selected_completed_at = completed_at(conn, selected)
    if through_time is not None and selected_completed_at > through_time:
        raise CompetitionEvidenceError(
            f"competition {selected!r} completed at {selected_completed_at.isoformat()}, "
            f"after epoch cutoff {through_time.isoformat()}"
        )

    try:
        item_rows = repo.validate_evaluation_item_bindings(conn, selected)
    except repo.EvaluationItemBindingError as exc:
        raise CompetitionEvidenceError(
            f"competition {selected!r} evaluation item binding is invalid: {exc}"
        ) from exc
    if not item_rows:
        raise CompetitionEvidenceError(
            f"competition {selected!r} completed without evaluation items"
        )
    audit_items = tuple(
        CompetitionAuditItem(
            challenge_id=str(row["challenge_id"]),
            item_id=str(row["scoring_item_id"]),
            threshold_commitment=str(row["threshold_commitment"]),
            item_index=int(row["item_index"]),
            input_sha256=str(row["input_sha256"]),
            reference_sha256=str(row["reference_sha256"]),
            upscale_factor=(
                None
                if row["upscale_factor"] is None
                else int(row["upscale_factor"])
            ),
            target_width=(
                None if row["target_width"] is None else int(row["target_width"])
            ),
            target_height=(
                None if row["target_height"] is None else int(row["target_height"])
            ),
            item_commitment=(
                None
                if row["item_commitment"] is None
                else str(row["item_commitment"])
            ),
        )
        for row in item_rows
    )

    contenders = repo.list_contenders(conn, selected)
    baselines = [record for record in contenders if record.is_calibration]
    if len(baselines) != 1:
        raise CompetitionEvidenceError(
            f"competition {selected!r} needs exactly one archived baseline; found {len(baselines)}"
        )
    built_contenders = [
        record
        for record in contenders
        if not record.is_calibration and record.status == "BUILT"
    ]
    missing_census = sorted(
        str(record.hotkey or f"contender-id:{record.contender_id}")
        for record in built_contenders
        if not record.hotkey or record.hotkey not in census_by_hotkey
    )
    if missing_census:
        # Silently omitting a machine-accepted contender would let the authority
        # cherry-pick the ranked field after seeing its scores.  A close-block
        # identity mismatch is therefore an epoch HOLD, not a smaller podium.
        raise CompetitionEvidenceError(
            f"competition {selected!r} has BUILT contender(s) absent from the "
            f"close-block census: {missing_census}"
        )
    included = [baselines[0]]
    included.extend(
        sorted(
            built_contenders,
            key=lambda record: (record.hotkey or "", record.contender_id),
        )
    )
    if len(included) == 1:
        # A completed competition with no currently registered, machine-accepted
        # contender has no payable result.
        return None

    archived = _archived_submissions(conn, selected)
    included_ids = {record.contender_id for record in included}
    missing_archives = sorted(included_ids - set(archived))
    if missing_archives:
        raise CompetitionEvidenceError(
            f"competition {selected!r} has earning subject(s) without an exact sealed "
            f"source archive: {missing_archives}"
        )
    for record in included:
        _verify_archived_ref(
            store,
            archived[record.contender_id],
            what=f"competition subject {record.contender_id} sealed source archive",
        )
    baseline_archive = archived[baselines[0].contender_id]
    if baseline_archive != baseline_artifact:
        raise CompetitionEvidenceError(
            "competition baseline source archive does not exactly match the active "
            "registry artifact committed by the manifest"
        )

    subjects: list[CompetitionAuditSubject] = []
    scored_items: list[ScoredItem] = []
    packet_scores: dict[str, float] = {}
    dedup_candidates: list[CompetitionDedupCandidate] = []
    for record in included:
        role = "baseline" if record.is_calibration else "contender"
        hotkey = None if record.is_calibration else record.hotkey
        census = None if record.is_calibration else census_by_hotkey[hotkey or ""]
        uid = None if census is None else census.uid
        subject_id = "baseline" if record.is_calibration else f"contender:{hotkey}"
        rows = _performance_rows(conn, selected, record.contender_id)
        if len(rows) != len(item_rows):
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} has {len(rows)} packet row(s), "
                f"expected {len(item_rows)}"
            )
        observed_items = [int(row["item_id"]) for row in rows]
        expected_items = [int(row["item_id"]) for row in item_rows]
        if observed_items != expected_items:
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} does not cover the exact "
                "ordered evaluation-item matrix"
            )

        packet_digests: list[str] = []
        bundle_digests: list[str] = []
        output_digests: list[str] = []
        image_digests: set[str] = set()
        for row in rows:
            packet = _read_packet(
                store,
                row,
                track=manifest.track,
                expected_hotkey=hotkey,
                subject_id=subject_id,
            )
            packet_digest = str(row["score_packet_digest"])
            bundle_digest_raw = row["audit_bundle_digest"]
            if not bundle_digest_raw:
                raise CompetitionEvidenceError(
                    f"competition subject {subject_id!r} item {row['item_id']} has no "
                    "audit-bundle digest"
                )
            bundle_digest = str(bundle_digest_raw)
            bundle = _require_bundle(
                store,
                bundle_digest,
                row=row,
                subject_id=subject_id,
                expected_hotkey=hotkey,
                manifest_digest=competition.manifest_digest,
                track=manifest.track,
            )
            if packet.content_digest != bundle.miner_output.digest:
                raise CompetitionEvidenceError(
                    f"competition subject {subject_id!r} packet {packet_digest} "
                    f"content digest {packet.content_digest!r} does not match bundle "
                    f"miner output {bundle.miner_output.digest!r}"
                )
            if (
                bundle.execution_image_digest is None
                or record.image_digest is None
                or bundle.execution_image_digest != record.image_digest
            ):
                raise CompetitionEvidenceError(
                    f"competition subject {subject_id!r} bundle {bundle_digest} "
                    "does not bind the exact persisted execution image"
                )
            image_digests.add(bundle.execution_image_digest)
            output_digests.append(bundle.miner_output.digest)
            packet_digests.append(packet_digest)
            bundle_digests.append(bundle_digest)
            packet_scores[packet_digest] = packet.score
            scored_items.append(
                ScoredItem(
                    uid=0 if uid is None else uid,
                    hotkey="" if hotkey is None else hotkey,
                    challenge_id=str(row["challenge_id"]),
                    item_id=str(row["scoring_item_id"]),
                    bundle_digest=bundle_digest,
                    packet_digest=packet_digest,
                    committed_track=manifest.track,
                    source="competition",
                    baseline=record.is_calibration,
                    competition_subject=subject_id,
                )
            )
        if len(image_digests) != 1:
            raise CompetitionEvidenceError(
                f"competition subject {subject_id!r} used {len(image_digests)} "
                "execution image identities across its item matrix"
            )
        execution_image_digest = next(iter(image_digests))
        if (
            record.is_calibration
            and execution_image_digest != commitment.baseline_image_digest
        ):
            raise CompetitionEvidenceError(
                f"competition baseline executed image {execution_image_digest} does "
                f"not match anchored {commitment.baseline_image_digest}"
            )
        if census is not None:
            dedup_candidates.append(
                CompetitionDedupCandidate(
                    subject_id=subject_id,
                    uid=census.uid,
                    coldkey=census.coldkey,
                    ip=census.ip,
                    output_digests=tuple(output_digests),
                )
            )
        subjects.append(
            CompetitionAuditSubject(
                subject_id=subject_id,
                role=role,
                uid=uid,
                hotkey=hotkey,
                submission_archive_digest=(
                    None
                    if record.is_calibration
                    else archived[record.contender_id].digest
                ),
                submission_archive_bytes=(
                    None
                    if record.is_calibration
                    else archived[record.contender_id].byte_size
                ),
                execution_image_digest=execution_image_digest,
                repo_url=None if record.is_calibration else record.repo_url,
                commit_sha=None if record.is_calibration else record.commit_sha,
                tree_sha=None if record.is_calibration else record.tree_sha,
                packet_digests=tuple(packet_digests),
                audit_bundle_digests=tuple(bundle_digests),
            )
        )

    dedup_losers = competition_dedup_losers(tuple(dedup_candidates))
    subjects = [
        subject.model_copy(
            update={"dedup_excluded": subject.subject_id in dedup_losers}
        )
        for subject in subjects
    ]

    competition_input = CompetitionInput(
        competition_id=selected,
        track=manifest.track,
        cycle=selected_cycle,
        completed_at=selected_completed_at,
        applied_at=through_time,
        manifest_digest=competition.manifest_digest,
        commitment_root=competition.commitment_root,
        anchor_netuid=int(anchor_receipt["anchor_netuid"]),
        anchor_payload_hex=str(anchor_receipt["payload_hex"]),
        anchor_payload_digest=str(anchor_receipt["payload_digest"]),
        anchor_block=int(anchor_receipt["anchor_block"]),
        anchor_block_hash=str(anchor_receipt["anchor_block_hash"]),
        anchor_finalized_block=int(anchor_receipt["finalized_block"]),
        baseline_version=manifest.baseline.version,
        baseline_artifact_digest=manifest.baseline.artifact_digest,
        baseline_artifact_bytes=manifest.baseline.artifact_bytes,
        baseline_execution_image_digest=manifest.baseline.image_digest,
        baseline_provenance_digest=manifest.baseline.provenance_digest,
        baseline_provenance_bytes=manifest.baseline.provenance_bytes,
        items=audit_items,
        subjects=tuple(subjects),
    )
    result = derive_competition_result(competition_input, packet_scores)
    if (
        tokenomics is not None
        and result.contenders
        and qualifies_for_crown(
            tokenomics,
            result.baseline_score,
            result.contenders[0].score,
        )
    ):
        # CROWN is the disclosure boundary: the exact winning source archive must
        # become publicly readable before an epoch can commit the earning result.
        # Selection uses the same packet-derived result and inclusive Decimal gate
        # as the reward-window fold; no database ranking/review field participates.
        winner_hotkey = result.contenders[0].hotkey
        winner_record = next(
            (
                record
                for record in built_contenders
                if record.hotkey == winner_hotkey
            ),
            None,
        )
        if winner_record is None:
            raise CompetitionEvidenceError(
                f"CROWN winner {winner_hotkey!r} has no included contender archive"
            )
        winner_archive = archived[winner_record.contender_id]
        try:
            store.release(winner_archive)
        except Exception as exc:
            raise CompetitionEvidenceError(
                f"CROWN winner archive {winner_archive.digest} could not be released "
                f"for public audit: {type(exc).__name__}: {exc}"
            ) from exc
        if not store.is_released(winner_archive):
            raise CompetitionEvidenceError(
                f"CROWN winner archive {winner_archive.digest} has no content-valid "
                "release copy after explicit release"
            )
        # This privileged read proves the writer produced the exact plaintext.
        # EpochFinalizer separately reads it through make_public_store() (unsigned
        # in S3 mode) before it can write the earning epoch's `_FINALIZED` marker.
    return CompetitionEpochEvidence(
        competition_input=competition_input,
        scored_items=tuple(scored_items),
        packet_scores=packet_scores,
        result=result,
    )


__all__ = [
    "CompetitionEpochEvidence",
    "CompetitionEvidenceError",
    "build_competition_epoch_evidence",
    "latest_completed_competition_id",
]
