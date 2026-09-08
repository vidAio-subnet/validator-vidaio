"""Build epoch content-roster inputs from committed ordinary-round evidence."""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from vidaio.audit import LifecycleStage, build_bundle, canonical_json_bytes
from vidaio.audit.store import ArtifactKind
from vidaio.authority.finalizer import AuditFileMissingError
from vidaio.epoch import ContentRoundInput
from vidaio.scoring.content_duplicate_evidence import parse_content_round


def build_content_round_inputs(
    rows: Iterable[Mapping[str, Any]], *, store: Any, commitment_source: Any,
    require_reference_release: Callable[..., None], now: Any,
    prior_close_block: int | None, close_block: int,
) -> tuple[ContentRoundInput, ...]:
    """Retain every new decision, including rounds with no economic fold.

    Rows come from the same sealed SQL capture as media and availability. The
    commit window is strict even when an old identity later returns; an expired
    roster cannot become payable again through a newly selected packet.
    """
    result = []
    for row in rows:
        commit_block = int(row["commit_block"])
        if (commit_block > close_block or
                (prior_close_block is not None and commit_block <= prior_close_block)):
            raise AuditFileMissingError("content round lies outside the captured commit window")
        context = parse_content_round(row["round_json"])
        if (context.digest() != row["round_digest"]
                or any(getattr(context, key) != row[key] for key in ("challenge_id", "item_id", "track"))):
            raise AuditFileMissingError("committed content round index differs from its canonical evidence")
        commitment = commitment_source.commitment(context.challenge_id)
        anchor = commitment_source.anchor(context.challenge_id)
        if (commitment is None or anchor != context.commitment_anchor
                or commitment.track != context.track
                or commitment.scorer_version != context.committed_scorer_version):
            raise AuditFileMissingError("content round does not bind its pre-dispatch commitment")
        # This is a shared-input metadata template, never an economic packet.
        # A declared skipped component may have unavailable member packet/media;
        # retaining those actual refs lets the auditor reproduce that SKIP while
        # independently checking other components. Ordinary final packet bundles
        # continue to use the stricter existing _persist_bundle path.
        for ref in (context.challenge_input, context.reference_original):
            if not store.exists(ref):
                raise AuditFileMissingError("content round shared input/reference is unavailable")
        require_reference_release(reference=context.reference_original,
            challenge_id=context.challenge_id, item_id=context.item_id)
        member = context.roster[0]
        reveal = store.put(commitment.preimage_bytes(), ArtifactKind.DAG_REVEAL)
        if reveal.digest != context.commitment_anchor.commitment_hash:
            raise AuditFileMissingError("content template reveal differs from finalized anchor")
        manifest = store.put(canonical_json_bytes({
            "schema_version": 1, "challenge_id": context.challenge_id,
            "content_round_digest": context.digest(),
            "challenge_input": context.challenge_input.model_dump(mode="json"),
            "reference_original": context.reference_original.model_dump(mode="json"),
        }), ArtifactKind.MANIFEST)
        bundle = build_bundle(
            challenge_id=context.challenge_id, item_id=member.receipt.metadata.task_id,
            miner_hotkey=member.hotkey, commitment_hash=reveal.digest,
            challenge_anchor=context.commitment_anchor, miner_receipt=member.receipt,
            stage=LifecycleStage.POST_RETIREMENT, challenge_input=context.challenge_input,
            miner_output=member.output, score_packet=member.score_packet, manifest=manifest,
            reference_original=context.reference_original, dag_reveal=reveal,
            scorer_version=context.committed_scorer_version, backend_versions={}, created_at=now.isoformat(),
        )
        template = store.put(canonical_json_bytes(bundle.model_dump(mode="json")), ArtifactKind.AUDIT_BUNDLE)
        result.append(ContentRoundInput(
            challenge_id=context.challenge_id, item_id=context.item_id, track=context.track,
            round_json=context.to_json(), round_digest=context.digest(),
            template_bundle=template,
        ))
    return tuple(sorted(result, key=lambda row: (row.challenge_id, row.item_id, row.track)))
