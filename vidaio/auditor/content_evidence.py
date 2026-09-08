"""Independent archived-media verification of complete content duplicate components.

The anchored round index binds the full original roster. Each member is rerun as
an ordinary measured packet; its derived canonical evidence, signed receipt and
shared challenge determine the graph. Economic zeros never supply measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vidaio.audit.bundle import AuditBundle
from vidaio.audit.canonical import canonical_json_bytes
from vidaio.audit.recompute import ScorePacketShape, verify_bundle
from vidaio.audit.store import ArtifactKind, IntegrityError
from vidaio.scoring.content_duplicate_evidence import (
    ContentRoundEvidence, ContentDuplicateWitness, InvalidContentEvidence,
    content_duplicate_identity, derive_components, derive_edges, validate_member_packet,
)

MAX_CONTENT_METADATA = 16 * 1024 * 1024


class ContentEvidenceUnavailable(Exception):
    """An archived artifact or independent backend cannot be read/recomputed."""


@dataclass(frozen=True)
class ComponentResult:
    uids: tuple[int, ...]
    status: str  # verified | unavailable | invalid
    detail: str = ""


def measured_bundle(template: AuditBundle, member: Any, packet: ScorePacketShape) -> AuditBundle:
    """Bind a roster's original packet to its authenticated shared input/reveal."""
    obj = template.model_dump(mode="python")
    obj.update(item_id=packet.item_id, miner_hotkey=member.hotkey,
               miner_output=member.output, score_packet=member.score_packet,
               miner_receipt=member.receipt, scorer_version=packet.scorer_version,
               backend_versions=dict(packet.backend_versions))
    return AuditBundle.model_validate(obj)


def verify_round(
    context: ContentRoundEvidence, template: AuditBundle, store: Any,
    recomputer: Any, *, chronology: Callable[[AuditBundle], Any] | None = None,
    reveal_verifier: Any = None,
) -> tuple[ComponentResult, ...]:
    """Recompute every component; unavailable evidence cannot become a punitive zero."""
    if (template.challenge_id != context.challenge_id
            or template.challenge_anchor != context.commitment_anchor
            or template.challenge_input != context.challenge_input
            or template.reference_original != context.reference_original):
        raise InvalidContentEvidence("round template differs from committed challenge/media")
    results = []
    verified = []
    by_uid = {member.uid: member for member in context.roster}
    for component in context.components:
        unavailable, invalid = [], []
        for uid in component:
            member = by_uid[uid]
            try:
                raw = store.get_limited(member.score_packet, MAX_CONTENT_METADATA)
                packet = ScorePacketShape.model_validate_json(raw)
                validate_member_packet(member, packet, context)
                bundle = measured_bundle(template, member, packet)
                if chronology is not None:
                    checked = chronology(bundle)
                    if str(checked.kind) == "SKIP":
                        raise ContentEvidenceUnavailable(checked.detail)
                    if str(checked.kind) != "PASS":
                        raise InvalidContentEvidence(checked.detail)

                class RecordingRecomputer:
                    fresh = None
                    error = None

                    def recompute(self, b, artifacts):
                        try:
                            from vidaio.auditor.recomputer import RecomputeUnavailable
                            if any(kind not in artifacts for kind in (ArtifactKind.REFERENCE_ORIGINAL,
                                    ArtifactKind.CHALLENGE_INPUT, ArtifactKind.MINER_OUTPUT)):
                                raise RecomputeUnavailable("required original media artifact is unavailable")
                            self.fresh = recomputer.recompute(b, artifacts)
                            return self.fresh
                        except Exception as exc:
                            self.error = exc
                            raise

                recorder = RecordingRecomputer()
                report = verify_bundle(
                    bundle, store, recorder, expected_bundle_digest=bundle.bundle_digest(),
                    expected_miner_hotkey=member.hotkey, require_expected_miner=True,
                    reveal_verifier=reveal_verifier, strict=False,
                )
                # Merkle membership is supplied by the anchored complete-round index,
                # not the post-group packet tree; every other attempted check must pass.
                failures = report.failures()
                if failures:
                    from vidaio.auditor.recomputer import RecomputeUnavailable
                    backend_unavailable = isinstance(recorder.error, RecomputeUnavailable)
                    availability_failure = backend_unavailable or any(
                        check.code == "ARTIFACT_MISSING" for check in failures)
                    conclusive = [check for check in failures
                        if check.code != "ARTIFACT_MISSING"
                        and not (backend_unavailable and check.code == "RECOMPUTE_ERROR")]
                    if conclusive:
                        raise InvalidContentEvidence("; ".join(f"{c.code}: {c.reason}" for c in conclusive))
                    if availability_failure:
                        raise ContentEvidenceUnavailable("original member artifact/backend unavailable")
                    raise InvalidContentEvidence("; ".join(f"{c.code}: {c.reason}" for c in failures))
                if recorder.fresh is None:
                    raise ContentEvidenceUnavailable("original member was not independently measured")
                fresh = recorder.fresh
                # Rebuild descriptors from actual recomputation, rather than trusting
                # equality of two packet-controlled witness declarations.
                verified.append(member.model_copy(update={
                    "canonical_content_digest": fresh.canonical_content_digest,
                    "content_fingerprint": fresh.content_fingerprint,
                    "encoded_size": fresh.encoded_size,
                    "canonicalization_plan_digest": fresh.canonicalization_plan_digest,
                }))
            except (FileNotFoundError, OSError, ContentEvidenceUnavailable) as exc:
                unavailable.append(f"uid {uid}: {exc}")
            except Exception as exc:
                invalid.append(f"uid {uid}: {type(exc).__name__}: {exc}")
        status = "invalid" if invalid else "unavailable" if unavailable else "verified"
        results.append(ComponentResult(component, status, "; ".join(invalid + unavailable)))
    # All available members must reproduce their exact full graph, including edges
    # across claimed components. Partial availability is never a fabricated PASS.
    if len(verified) == len(context.roster):
        if derive_edges(verified) != context.edges or derive_components(verified) != context.components:
            raise InvalidContentEvidence("independent archived-media component graph differs")
    return tuple(results)


def validate_zero_packet(packet: Any, bundle: AuditBundle, witness: ContentDuplicateWitness) -> None:
    context = witness.round_evidence
    member = next(member for member in context.roster if member.uid == witness.loser_uid)
    expected_identity = content_duplicate_identity(
        committed_scorer_version=context.committed_scorer_version,
        track=context.track, scoring_config_digest=context.scoring_config_digest,
    )
    expected_violation = [{"code": "DUPLICATE_CONTENT", "detail":
        f"auditable same-content component; anchor-salted winner uid {witness.winner_uid}", "limit": None, "measured": None}]
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
            or obj.get("metrics") != {"content_duplicate_witness": witness.to_json()}
            or bundle.miner_output != member.output or bundle.miner_receipt != member.receipt
            or bundle.challenge_anchor != context.commitment_anchor):
        raise InvalidContentEvidence("content zero is not the exact canonical signed-loser decision")


def audit_manifest_rounds(service: Any, log: Any, store: Any, recomputer: Any, *, prior_log: Any = None) -> tuple[Any, ...]:
    """Check complete post-group economics in both own and beacon audit modes."""
    import json
    from vidaio.epoch import AuditFileKind
    from vidaio.auditor.report import ItemVerdict, ItemVerdictKind
    from vidaio.scoring.content_duplicate_evidence import (
        parse_content_round, content_witness_from_packet, is_content_duplicate_identity,
    )

    if log.schema_version < 17:
        return ()
    verdicts = []
    final = {}
    covered = set()
    domains = set()
    census = {entry.uid: entry.hotkey for entry in log.miner_census}
    prior_cursors = {}
    if (prior_log is not None and log.prior_log_digest is not None
            and prior_log.log_digest() == log.prior_log_digest):
        prior_cursors = prior_log.audit_manifest.fold_cursors

    def verdict(challenge, item, kind, detail, digest=""):
        verdicts.append(ItemVerdict(source="content_components", challenge_id=challenge,
            item_id=item, bundle_digest=digest, packet_digest=digest, verdict=kind,
            code="CONTENT_EVIDENCE_MISMATCH" if kind is ItemVerdictKind.FAIL else
                 "DUPLICATE_EVIDENCE_UNAVAILABLE" if kind is ItemVerdictKind.SKIP else "",
            detail=detail))

    for uid, refs in log.audit_manifest.per_uid.items():
        for ref in refs:
            if ref.kind is not AuditFileKind.SCORE_PACKET or ref.source != "inference":
                continue
            try:
                obj = json.loads(store.get_digest_limited(ArtifactKind.SCORE_PACKET, ref.digest,
                                                         max_bytes=MAX_CONTENT_METADATA))
                score = obj.get("score")
                if isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0 and obj.get("gate_passed") is not True:
                    raise InvalidContentEvidence("positive score claims a failed gate; cannot evade complete content index")
                key = (ref.challenge_id, uid, obj.get("cycle_sequence"))
                if key in final:
                    raise InvalidContentEvidence("duplicate finalized round/uid/ordering identity")
                final[key] = (obj, ref)
            except (OSError, FileNotFoundError) as exc:
                verdict(ref.challenge_id, ref.item_id, ItemVerdictKind.SKIP, str(exc), ref.digest)
            except Exception as exc:
                verdict(ref.challenge_id, ref.item_id, ItemVerdictKind.FAIL, str(exc), ref.digest)

    for entry in log.audit_manifest.content_rounds:
        try:
            context = parse_content_round(entry.round_json)
            domain = (context.challenge_id, context.track)
            if domain in domains:
                raise InvalidContentEvidence("duplicate/split content round domain")
            domains.add(domain)
            raw_template = store.get_limited(entry.template_bundle, MAX_CONTENT_METADATA)
            template = AuditBundle.model_validate_json(raw_template)
            if template.bundle_digest() != entry.template_bundle.digest:
                raise InvalidContentEvidence("content template canonical digest mismatch")
            results = verify_round(context, template, store, recomputer,
                chronology=lambda bundle: service._challenge_chronology(bundle, store),
                reveal_verifier=service._reveal_verifier)
            members = {member.uid: member for member in context.roster}
            ordering = context.commitment_anchor.dispatch_ordering_key
            for result in results:
                skipped = result.uids in context.skipped_components
                keys = {(context.challenge_id, uid, ordering) for uid in result.uids}
                covered.update(keys)
                if skipped:
                    if any(key in final for key in keys):
                        raise InvalidContentEvidence("skipped component still enters the economic fold")
                    # A conclusive archived-byte/signature mismatch remains a
                    # DISPUTED audit finding even when economics skipped the group.
                    # Current readability cannot disprove a past witness-write
                    # failure, so a declared non-punitive skip never fabricates PASS.
                    verdict(context.challenge_id, f"{context.item_id}:{result.uids}",
                            ItemVerdictKind.FAIL if result.status == "invalid" else ItemVerdictKind.SKIP,
                            result.detail or "component skipped without an economic fold; past evidence-write availability is not independently provable",
                            entry.round_digest)
                    continue
                if result.status != "verified":
                    verdict(context.challenge_id, f"{context.item_id}:{result.uids}",
                            ItemVerdictKind.FAIL if result.status == "invalid" else ItemVerdictKind.SKIP,
                            result.detail, entry.round_digest)
                    continue
                winner = context.winner(result.uids)
                for uid in result.uids:
                    member = members[uid]
                    key = (context.challenge_id, uid, ordering)
                    old_cursor = prior_cursors.get(uid)
                    must_fold = (census.get(uid) == member.hotkey
                                 and (old_cursor is None or old_cursor < ordering))
                    if not must_fold:
                        if key in final:
                            raise InvalidContentEvidence("old identity/already-folded content member re-enters current economics")
                        continue
                    if key not in final:
                        raise InvalidContentEvidence(f"eligible uid {uid} is absent from finalized component decisions")
                    obj, ref = final[key]
                    if uid == winner:
                        original = json.loads(store.get_limited(member.score_packet, MAX_CONTENT_METADATA))
                        expected = dict(original, item_id=f"{member.receipt.metadata.task_id}-c{ordering}",
                                        challenge_id=context.challenge_id, track=context.track,
                                        cycle_sequence=ordering, excluded=False)
                        if obj != expected:
                            raise InvalidContentEvidence(f"component winner uid {uid} is not its original measured packet")
                    else:
                        witness = content_witness_from_packet(obj)
                        if witness.round_evidence != context or witness.loser_uid != uid:
                            raise InvalidContentEvidence("zero witness differs from complete epoch round index")
                        # Bundle field substitution is entirely bound to the anchored
                        # original round roster; the final packet remains independently
                        # included in the epoch's packet Merkle tree.
                        zero_bundle = template.model_copy(update={
                            "miner_output": member.output, "miner_receipt": member.receipt,
                            "challenge_anchor": context.commitment_anchor,
                        })
                        validate_zero_packet(obj, zero_bundle, witness)
                        if obj.get("cycle_sequence") != ordering or obj.get("excluded") is not False:
                            raise InvalidContentEvidence("content zero fold ordering/exclusion differs")
                verdict(context.challenge_id, f"{context.item_id}:{result.uids}",
                        ItemVerdictKind.PASS, "archived component graph and all economic decisions reproduced", entry.round_digest)
        except (OSError, FileNotFoundError, ContentEvidenceUnavailable) as exc:
            verdict(entry.challenge_id, entry.item_id, ItemVerdictKind.SKIP, str(exc), entry.round_digest)
        except Exception as exc:
            verdict(entry.challenge_id, entry.item_id, ItemVerdictKind.FAIL,
                    f"{type(exc).__name__}: {exc}", entry.round_digest)

    for key, (obj, ref) in final.items():
        score = obj.get("score")
        positive = isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0
        eligible = (obj.get("gate_passed") is True and positive
                    and (bool(getattr(recomputer, "requires_content_evidence", False)) or obj.get("canonical_content_digest") is not None))
        if (eligible or is_content_duplicate_identity(obj.get("scorer_version"))) and key not in covered:
            verdict(ref.challenge_id, ref.item_id, ItemVerdictKind.FAIL,
                    "eligible/content-zero packet is omitted from the complete round index", ref.digest)
    return tuple(verdicts)
