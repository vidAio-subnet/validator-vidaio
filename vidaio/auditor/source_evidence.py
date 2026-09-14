"""Independent verification of round-composition source-proximity verdicts.

A SOURCE_PROXIMITY zero packet is only as good as its roster. Three things are
checked, none of them trusting the authority:

1. **Shape** — the packet is the exact canonical decision the witness implies
   (identity, violation text, empty measurement fields), like a content zero.
2. **Roster values** — every roster member's residual pair equals the metrics of
   its archived ORIGINAL measured packet, that packet is gate-passed and positive
   under the committed scorer, and the medians/flags re-derive from the roster.
   The flagged member's own residuals are additionally recomputed from the
   released media by the caller (``RealScoreRecomputer.recompute_source_proximity``).
3. **Completeness** (epoch-log pass) — the roster equals the set of eligible
   measured packets the epoch actually folded for that challenge, so no member was
   dropped or invented to move the medians.
"""
from __future__ import annotations

import json
from typing import Any

from vidaio.audit.bundle import AuditBundle
from vidaio.audit.recompute import ScorePacketShape
from vidaio.audit.store import ArtifactKind
from vidaio.auditor.content_evidence import MAX_CONTENT_METADATA, ContentEvidenceUnavailable
from vidaio.scoring.source_proximity_evidence import (
    InvalidSourceEvidence,
    SourceProximityWitness,
    SourceRoundEvidence,
    derive_flags,
    derive_medians,
    is_source_proximity_identity,
    packet_residuals,
    rule_margins,
    source_proximity_identity,
    source_witness_from_packet,
    validate_member_packet,
    violation_detail,
)


def validate_source_packet(packet: Any, bundle: AuditBundle, witness: SourceProximityWitness) -> None:
    """The zero packet must be the exact canonical decision for its flagged member."""
    context = witness.round_evidence
    member = context.member(witness.member_uid)
    _vmaf_excess, chroma_excess = context.excess(witness.member_uid)
    expected_identity = source_proximity_identity(
        committed_scorer_version=context.committed_scorer_version,
        track=context.track, scoring_config_digest=context.scoring_config_digest, rule=context.rule,
    )
    expected_violation = [{"code": "SOURCE_PROXIMITY", "detail": violation_detail(context, witness.member_uid),
                           "limit": rule_margins(context.rule)[1], "measured": chroma_excess}]
    obj = packet.model_dump(mode="json") if hasattr(packet, "model_dump") else packet
    allowed_ids = {member.receipt.metadata.task_id,
                   f"{member.receipt.metadata.task_id}-c{context.commitment_anchor.dispatch_ordering_key}"}
    if (obj.get("item_id") not in allowed_ids or obj.get("challenge_id") != context.challenge_id
            or obj.get("track") != context.track or obj.get("miner_hotkey") != member.hotkey
            or obj.get("content_digest") != member.output.digest
            or obj.get("scorer_version") != expected_identity
            or obj.get("scoring_config_digest") != context.scoring_config_digest
            or obj.get("score") != 0.0 or obj.get("gate_passed") is not False
            or obj.get("violations") != expected_violation or obj.get("breakdown") is not None
            or obj.get("skips", []) or obj.get("backend_versions") != {}
            or obj.get("canonicalization_plan_digest") is not None
            or obj.get("pieapp_start_frame") is not None
            or any(obj.get(key) is not None for key in ("canonical_content_digest", "content_fingerprint", "encoded_size"))
            or obj.get("metrics") != {"source_proximity_witness": witness.to_json()}
            or bundle.miner_output != member.output or bundle.miner_receipt != member.receipt
            or bundle.challenge_anchor != context.commitment_anchor):
        raise InvalidSourceEvidence("source zero is not the exact canonical signed-member decision")


def verify_roster(context: SourceRoundEvidence, store: Any) -> None:
    """Every roster value must be the archived original packet's own published metric."""
    for member in context.roster:
        try:
            raw = store.get_limited(member.score_packet, MAX_CONTENT_METADATA)
        except (OSError, FileNotFoundError) as exc:
            raise ContentEvidenceUnavailable(f"source roster uid {member.uid}: {exc}") from exc
        original = ScorePacketShape.model_validate_json(raw)
        validate_member_packet(member, original, context)
    if ((context.vmaf_residual_median, context.chroma_residual_median) != derive_medians(context.roster)
            or context.flagged != derive_flags(context.roster, context.rule)):
        raise InvalidSourceEvidence("source round decision does not re-derive from its roster")


def audit_source_rounds(log: Any, store: Any) -> tuple[Any, ...]:
    """Epoch-log pass: each source round's roster must equal the epoch's eligible fold set.

    Eligibility is exactly what the authority applies: a gate-passed, positively
    scored, committed-scorer packet that publishes both residuals. For a uid whose
    finalized packet is a content share, the ORIGINAL measured packet (named by the
    content round roster) is the eligible one.
    """
    from vidaio.epoch import AuditFileKind
    from vidaio.auditor.report import ItemVerdict, ItemVerdictKind
    from vidaio.scoring.content_duplicate_evidence import (
        ContentShareWitness, content_witness_from_packet, is_content_duplicate_identity, parse_content_round,
    )

    verdicts: list[Any] = []
    if getattr(log, "schema_version", 0) < 17:
        return ()

    def verdict(challenge, item, kind, detail, digest=""):
        verdicts.append(ItemVerdict(source="source_proximity", challenge_id=challenge, item_id=item,
            bundle_digest=digest, packet_digest=digest, verdict=kind,
            code="SOURCE_EVIDENCE_MISMATCH" if kind is ItemVerdictKind.FAIL else
                 "SOURCE_EVIDENCE_UNAVAILABLE" if kind is ItemVerdictKind.SKIP else "", detail=detail))

    # Original measured packets named by content rounds (share losers keep no metrics).
    originals: dict[tuple[str, int], Any] = {}
    for entry in log.audit_manifest.content_rounds:
        try:
            context = parse_content_round(entry.round_json)
        except Exception:
            continue
        for member in context.roster:
            originals[(context.challenge_id, member.uid)] = member.score_packet

    finalized: dict[str, dict[int, dict]] = {}
    source_rounds: dict[str, tuple[SourceRoundEvidence, str]] = {}
    for uid, refs in log.audit_manifest.per_uid.items():
        for ref in refs:
            if ref.kind is not AuditFileKind.SCORE_PACKET or ref.source != "inference":
                continue
            try:
                obj = json.loads(store.get_digest_limited(ArtifactKind.SCORE_PACKET, ref.digest,
                                                         max_bytes=MAX_CONTENT_METADATA))
                if obj.get("track") != "compression":
                    continue
                if is_source_proximity_identity(obj.get("scorer_version")):
                    witness = source_witness_from_packet(obj)
                    if witness.member_uid != uid:
                        raise InvalidSourceEvidence("source zero packet is folded under a different uid")
                    known = source_rounds.get(ref.challenge_id)
                    if known is not None and known[0] != witness.round_evidence:
                        raise InvalidSourceEvidence("one challenge carries two different source rosters")
                    source_rounds[ref.challenge_id] = (witness.round_evidence, ref.digest)
                elif (is_content_duplicate_identity(obj.get("scorer_version"))
                        and obj.get("gate_passed") is not True):
                    # Content share loser: its eligible measurement is the archived original.
                    original_ref = originals.get((ref.challenge_id, uid))
                    if original_ref is None:
                        raise InvalidSourceEvidence("content loser without an original packet in the content index")
                    obj = json.loads(store.get_limited(original_ref, MAX_CONTENT_METADATA))
                finalized.setdefault(ref.challenge_id, {})[uid] = obj
            except (OSError, FileNotFoundError) as exc:
                verdict(ref.challenge_id, ref.item_id, ItemVerdictKind.SKIP, str(exc), ref.digest)
            except Exception as exc:
                verdict(ref.challenge_id, ref.item_id, ItemVerdictKind.FAIL, f"{type(exc).__name__}: {exc}", ref.digest)

    for challenge_id, (context, digest) in source_rounds.items():
        try:
            verify_roster(context, store)
            expected = set()
            for uid, obj in finalized.get(challenge_id, {}).items():
                if is_source_proximity_identity(obj.get("scorer_version")):
                    expected.add(uid)  # flagged: its original is in the roster by construction
                    continue
                score = obj.get("score")
                positive = isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0
                if (obj.get("gate_passed") is True and positive and packet_residuals(obj) is not None
                        and obj.get("scorer_version") == context.committed_scorer_version):
                    expected.add(uid)
            roster_uids = {member.uid for member in context.roster}
            if roster_uids != expected:
                raise InvalidSourceEvidence(
                    f"source roster uids {sorted(roster_uids)} differ from the epoch's eligible fold set {sorted(expected)}")
            for uid in context.flagged:
                if uid not in finalized.get(challenge_id, {}) or not is_source_proximity_identity(
                        finalized[challenge_id][uid].get("scorer_version")):
                    raise InvalidSourceEvidence(f"flagged uid {uid} was not folded as a source-proximity zero")
            verdict(challenge_id, challenge_id, ItemVerdictKind.PASS,
                    "source roster reproduced from archived originals and equals the eligible fold set", digest)
        except (OSError, FileNotFoundError, ContentEvidenceUnavailable) as exc:
            verdict(challenge_id, challenge_id, ItemVerdictKind.SKIP, str(exc), digest)
        except Exception as exc:
            verdict(challenge_id, challenge_id, ItemVerdictKind.FAIL, f"{type(exc).__name__}: {exc}", digest)
    return tuple(verdicts)
