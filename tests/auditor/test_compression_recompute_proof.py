"""an internal review — PROVE report-mode COMPRESSION media-recompute CATCHES a substituted score.

The auditor's beacon-independent checks (earning re-fold, weight re-derivation, snapshot /
census / time-base bindings, strict merkle inclusion) prove a published vector STRUCTURALLY
follows from the committed evidence, but they do NOT authenticate that each committed SCORE is
the true score the media deserves. Only an INDEPENDENT re-run of the scoring engine over the
committed bytes does that — and the auditor only re-runs it when it SAMPLES media
(``media_sample_rate > 0``). COMPRESSION scoring is real ffmpeg/libvmaf (CPU-capable), so at a
positive compression rate the auditor RE-RUNS libvmaf on the committed media and a substituted
score cannot survive it.

This is the end-to-end substantiation of an internal review (see also
``the development-tree stack runner``'s ``run_auditor`` rationale): a compression epoch whose committed
packet's score has been substituted to a value that is *self-consistent* — the earning re-fold,
the weight vector and every structural binding all re-derive CLEAN — is NONETHELESS caught as
``SCORE_MISMATCH`` ⇒ DISPUTED once media is sampled at rate>0, while the SAME substitution is
CLEAN at rate 0 (the recompute is UNREACHABLE there — exactly the gap round-11 #1 flags). An
honest packet at the same rate>0 is CLEAN.

Real ffmpeg/libvmaf is used (skips cleanly, and only, if the binary is genuinely absent —
``requires_media_tools``); the local stack ships real libvmaf via static-ffmpeg, so this
normally runs a true end-to-end recompute rather than skipping.
"""

from __future__ import annotations

import json


from tests.auditor.conftest import (
    build_real_bundle,
    requires_media_tools,
    score_compression_item,
)
from tests.auditor.fakes import (
    BURN_UID,
    NOW,
    MetagraphAuditor,
    folded_miner,
    honest_log,
    scored_item,
)
from vidaio.audit.recompute import SCORE_MISMATCH
from vidaio.audit.store import LocalFsStore
from vidaio.auditor import (
    AuditorConfig,
    AuditStatus,
    InMemoryBundleSource,
    ItemVerdictKind,
    RealScoreRecomputer,
    SamplePolicy,
    persist_bundle,
)

pytestmark = requires_media_tools

# The single uid + identity the whole epoch is attributed to. The bundle's miner, the
# packet's miner, the ScoredItem hotkey (`hk{uid}` via scored_item) and the close-block
# metagraph identity (folded_miner -> `hk{uid}`) MUST all agree, or the item fails the
# identity binding before recompute is even reached.
_UID = 1
_CHALLENGE_ID = "c1"
_ITEM_ID = "i1"
_HOTKEY = f"hk{_UID}"


def _packet_with_earning_fields(item, *, tamper_to: float | None = None) -> bytes:
    """The committed score packet: the REAL ItemScore + the evidence-bound earning fields.

    The finalizer's ``_persist_score_packet`` writes exactly this shape — the real ItemScore
    (which the media recompute parses) MERGED with ``score`` / ``cycle_sequence`` / ``excluded``
    (which the earning re-fold re-reads). When ``tamper_to`` is set, the top-level ``score``,
    the ``final_score`` metric AND ``breakdown.final`` are all inflated together (mirrors
    ``tests/auditor/test_recomputer``): the packet stays INTERNALLY consistent (gates-first
    holds), so the misreport is invisible to every structural check and can ONLY be caught by an
    independent libvmaf recompute over the committed bytes.
    """
    payload = json.loads(item.to_json())
    if tamper_to is not None:
        payload["score"] = tamper_to
        payload["metrics"]["final_score"] = tamper_to
        if isinstance(payload.get("breakdown"), dict):
            payload["breakdown"]["final"] = tamper_to
    # The evidence-bound earning fields (extra="ignore" on the recompute side, re-read by the
    # earning fold) — a genesis single-cycle fold at ordering key 0.
    payload["cycle_sequence"] = 0
    payload["excluded"] = False
    return json.dumps(payload).encode("utf-8")


def _committed_epoch(store: LocalFsStore, clips, item, *, packet_bytes: bytes, cycle_score: float):
    """Commit ONE real compression item into a finalized-shaped epoch log.

    ``cycle_score`` is the earning value the miner's ``accumulate_score`` folds AND the value
    the manifest's ``EarningInput`` attests — set to the SAME (possibly substituted) value the
    committed packet carries, so the earning re-fold + weight re-derivation are self-consistent
    and the ONLY thing that can dispute the epoch is the media recompute.
    """
    bundle = build_real_bundle(
        store, clips, item,
        packet_bytes=packet_bytes,
        challenge_id=_CHALLENGE_ID, item_id=_ITEM_ID, miner_hotkey=_HOTKEY,
        committed_track="compression", dispatch_ordering_key=0,
    )
    persist_bundle(store, bundle)  # so StoredBundleSource / manifest can resolve it by digest

    from vidaio.authority import build_audit_manifest

    item_scored = scored_item(bundle, _UID, score=cycle_score, seq=0)
    manifest = build_audit_manifest([item_scored], store=store)  # merkle root + inclusion proof
    log = honest_log([folded_miner(_UID, cycle_score)], manifest)

    source = InMemoryBundleSource()
    source.add(bundle)
    return log, source


def _recomputer(worker_config, real_media_backends, scoring_config) -> RealScoreRecomputer:
    return RealScoreRecomputer(
        worker_config,
        real_media_backends,
        scoring_config=scoring_config,
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )


def _auditor(source) -> MetagraphAuditor:
    # MetagraphAuditor auto-wires the close-block metagraph from the log's honest identities
    # and pins block_time(close_block) == log.created_at, so the snapshot /
    # created_at bindings PASS and the ONLY variable is the media recompute verdict.
    # an internal review: under strict=True a MISSING reveal verifier is an INCONCLUSIVE strict
    # skip (no longer washed to PASS). Media-sampling fixtures wire a trivial reveal verifier
    # (the fake bundles' DAGs are not real build_dag outputs; the deep verifier has its own tests
    # in tests/audit/test_recompute.py) so the ONLY variable stays the media recompute verdict.
    return MetagraphAuditor(
        AuditorConfig(auditor_hotkey="hkAuditor", burn_uid=BURN_UID),
        source,
        reveal_verifier=lambda dag_bytes: True,
    )


# --- the proof ------------------------------------------------------------------------


def test_honest_compression_score_is_clean_at_rate_positive(
    tmp_path, clips, worker_config, real_media_backends, scoring_config
) -> None:
    """An HONEST compression packet, media-sampled at rate>0 and RE-RUN through real libvmaf,
    reproduces the committed score and the epoch is CLEAN."""
    store = LocalFsStore(tmp_path / "audit")
    item = score_compression_item(
        worker_config, real_media_backends, scoring_config, clips,
        challenge_id=_CHALLENGE_ID, item_id=_ITEM_ID, miner_hotkey=_HOTKEY,
    )
    assert item.gate_passed and item.score > 0.0  # a real, non-zero committed score
    log, source = _committed_epoch(
        store, clips, item,
        packet_bytes=_packet_with_earning_fields(item),
        cycle_score=item.score,
    )

    report = _auditor(source).audit_epoch(
        log, store, SamplePolicy(sample_rate=1.0, min_samples=1),
        _recomputer(worker_config, real_media_backends, scoring_config), NOW,
    )

    assert [v.verdict for v in report.item_verdicts] == [ItemVerdictKind.PASS]
    assert report.overall is AuditStatus.CLEAN


def test_substituted_compression_score_is_caught_at_rate_positive(
    tmp_path, clips, worker_config, real_media_backends, scoring_config
) -> None:
    """A SUBSTITUTED compression score — self-consistent across the packet, the earning fold
    and the weight vector — is CAUGHT by the media recompute (SCORE_MISMATCH ⇒ DISPUTED) once
    media is sampled at rate>0. This is the misreport structural checks alone cannot see."""
    store = LocalFsStore(tmp_path / "audit")
    item = score_compression_item(
        worker_config, real_media_backends, scoring_config, clips,
        challenge_id=_CHALLENGE_ID, item_id=_ITEM_ID, miner_hotkey=_HOTKEY,
    )
    substituted = 0.99
    assert abs(item.score - substituted) > 0.1  # the real libvmaf score is nowhere near 0.99
    log, source = _committed_epoch(
        store, clips, item,
        packet_bytes=_packet_with_earning_fields(item, tamper_to=substituted),
        cycle_score=substituted,  # earning state re-derives to the substituted value -> CLEAN there
    )

    report = _auditor(source).audit_epoch(
        log, store, SamplePolicy(sample_rate=1.0, min_samples=1),
        _recomputer(worker_config, real_media_backends, scoring_config), NOW,
    )

    verdict = report.item_verdicts[0]
    assert verdict.verdict is ItemVerdictKind.FAIL
    assert verdict.code == SCORE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_the_same_substitution_is_unreachable_at_rate_zero(
    tmp_path, clips, worker_config, real_media_backends, scoring_config
) -> None:
    """The SAME substituted epoch is CLEAN at rate 0 — proving the media recompute (not any
    structural check) is what catches it, and that report mode running at rate 0 was the gap
    an internal review flags. No media is sampled, so libvmaf is never re-run."""
    store = LocalFsStore(tmp_path / "audit")
    item = score_compression_item(
        worker_config, real_media_backends, scoring_config, clips,
        challenge_id=_CHALLENGE_ID, item_id=_ITEM_ID, miner_hotkey=_HOTKEY,
    )
    substituted = 0.99
    log, source = _committed_epoch(
        store, clips, item,
        packet_bytes=_packet_with_earning_fields(item, tamper_to=substituted),
        cycle_score=substituted,
    )

    report = _auditor(source).audit_epoch(
        log, store, SamplePolicy(sample_rate=0.0, min_samples=0),
        _recomputer(worker_config, real_media_backends, scoring_config), NOW,
    )

    assert report.item_verdicts == ()  # nothing sampled -> recompute never reached
    assert SCORE_MISMATCH not in {v.code for v in report.item_verdicts}
    assert report.overall is AuditStatus.CLEAN
