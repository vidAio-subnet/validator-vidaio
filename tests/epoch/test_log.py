"""EpochLog model: determinism (the convergence crux) + the audit/weight invariants."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from vidaio.audit.canonical import sha256_hex
from vidaio.audit.commitments import COMMITMENT_DOMAIN
from vidaio.epoch import (
    EPOCH_LOG_SCHEMA_VERSION,
    AuditFileKind,
    AuditFileRef,
    AuditManifest,
    CompetitionAuditSubject,
    CompetitionInput,
    EpochLog,
    EpochLogInvalid,
    MinerCensusEntry,
    weight_vector_digest,
)
from vidaio.tokenomics import MinerSnapshot, quantize_u16

NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)


def _miner(uid: int, score: float, track: str = "compression") -> MinerSnapshot:
    return MinerSnapshot(
        uid=uid,
        hotkey=f"hk{uid}",
        coldkey=f"ck{uid}",
        ip=f"10.0.0.{uid}",
        track=track,
        accumulate_score=score,
    )


def _packet_ref(uid: int) -> AuditFileRef:
    return AuditFileRef(
        kind=AuditFileKind.SCORE_PACKET,
        digest=sha256_hex(f"packet-{uid}".encode()),
        challenge_id="c1",
        item_id=f"i{uid}",
        source="inference",
        committed_track="compression",  # REQUIRED on every SCORE_PACKET ref (#9)
    )


def _bundle_ref(uid: int) -> AuditFileRef:
    return AuditFileRef(
        kind=AuditFileKind.AUDIT_BUNDLE,
        digest=sha256_hex(f"bundle-{uid}".encode()),
        challenge_id="c1",
        item_id=f"i{uid}",
        source="inference",
    )


def _log(
    *,
    weight_shares: dict[int, float],
    miners: list[MinerSnapshot],
    manifest: AuditManifest,
    weight_u16: dict[int, int] | None = None,
    wv_digest: str | None = None,
    burn_uid: int | None = None,
    miner_census: tuple[MinerCensusEntry, ...] | None = None,
) -> EpochLog:
    u16 = weight_u16 if weight_u16 is not None else quantize_u16(weight_shares)
    resolved_census = (
        miner_census
        if miner_census is not None
        else tuple(MinerCensusEntry.from_miner(m) for m in miners)
    )
    cursors = dict(manifest.fold_cursors)
    for entry in resolved_census:
        cursors.setdefault(entry.uid, None)
    manifest = manifest.model_copy(update={"fold_cursors": cursors})
    return EpochLog(
        epoch_id=41822,
        close_block=15057191,
        scorer_version="scoring-1.0.0+abc123def456",
        created_at=NOW,
        burn_uid=burn_uid,
        miners=tuple(miners),
        miner_census=resolved_census,
        # This helper defaults to an inference-only epoch unless callers override it.
        weight_shares=weight_shares,
        weight_u16=u16,
        weight_vector_digest=wv_digest if wv_digest is not None else weight_vector_digest(u16),
        audit_manifest=manifest,
    )


def _valid_log() -> EpochLog:
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    shares = {1: 0.8, 2: 0.2}
    manifest = AuditManifest(
        per_uid={1: (_bundle_ref(1), _packet_ref(1)), 2: (_bundle_ref(2), _packet_ref(2))}
    )
    return _log(weight_shares=shares, miners=miners, manifest=manifest)


# --------------------------------------------------------------------------------------
# Determinism — the convergence-critical property.
# --------------------------------------------------------------------------------------


def test_same_state_byte_identical() -> None:
    a = _valid_log()
    b = _valid_log()
    assert a.to_json() == b.to_json()
    assert a.log_digest() == b.log_digest()


def test_determinism_independent_of_input_order() -> None:
    """Different dict/list input orders MUST yield byte-identical bytes+digest.

    This is what lets two validators on different machines converge without any
    coordination: the log normalizes every collection to a canonical order.
    """
    miners_a = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    shares_a = {1: 0.8, 2: 0.2}
    man_a = AuditManifest(
        per_uid={1: (_bundle_ref(1), _packet_ref(1)), 2: (_bundle_ref(2), _packet_ref(2))}
    )
    # reversed miner order, reversed weight-dict insertion order, reversed ref order
    miners_b = [_miner(2, 0.8, track="upscaling"), _miner(1, 0.9)]
    shares_b = {2: 0.2, 1: 0.8}
    man_b = AuditManifest(
        per_uid={2: (_packet_ref(2), _bundle_ref(2)), 1: (_packet_ref(1), _bundle_ref(1))}
    )
    a = _log(weight_shares=shares_a, miners=miners_a, manifest=man_a)
    b = _log(weight_shares=shares_b, miners=miners_b, manifest=man_b)
    assert a.to_json() == b.to_json()
    assert a.log_digest() == b.log_digest()


def test_from_json_to_json_roundtrip() -> None:
    log = _valid_log()
    data = log.to_json()
    back = EpochLog.from_json(data)
    assert back.to_json() == data
    assert back.log_digest() == log.log_digest()
    assert back.weight_u16 == log.weight_u16
    assert back.miners == log.miners
    assert back.miner_census == log.miner_census
    assert back.audit_manifest == log.audit_manifest


@pytest.mark.parametrize(
    "missing_path",
    ("miner_census", "audit_manifest.fold_cursors"),
)
def test_from_json_requires_cumulative_canonical_fields(missing_path: str) -> None:
    import json

    data = json.loads(_valid_log().to_json())
    if missing_path == "miner_census":
        del data["miner_census"]
    else:
        del data["audit_manifest"]["fold_cursors"]

    with pytest.raises(EpochLogInvalid, match="missing required canonical field"):
        EpochLog.from_json(json.dumps(data).encode())


def test_legacy_payload_without_v11_fields_hits_schema_fence_not_key_error() -> None:
    import json

    data = json.loads(_valid_log().to_json())
    data["schema_version"] = 10
    del data["miner_census"]
    del data["audit_manifest"]["fold_cursors"]

    with pytest.raises(EpochLogInvalid, match="schema_version 10"):
        EpochLog.from_json(json.dumps(data).encode())


def test_registered_census_is_canonical_and_independent_of_economic_order() -> None:
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    manifest = AuditManifest(
        per_uid={1: (_bundle_ref(1), _packet_ref(1)), 2: (_bundle_ref(2), _packet_ref(2))}
    )
    census = tuple(MinerCensusEntry.from_miner(m) for m in miners)
    a = _log(
        weight_shares={1: 0.8, 2: 0.2},
        miners=miners,
        miner_census=census,
        manifest=manifest,
    )
    b = _log(
        weight_shares={2: 0.2, 1: 0.8},
        miners=list(reversed(miners)),
        miner_census=tuple(reversed(census)),
        manifest=manifest,
    )
    assert a.to_json() == b.to_json()


def test_census_only_registration_needs_no_track_or_economic_row() -> None:
    census_only = MinerCensusEntry(
        uid=7, hotkey="hk7", coldkey="ck7", ip="10.0.0.7"
    )
    log = _log(
        weight_shares={999: 1.0},
        miners=[],
        miner_census=(census_only,),
        manifest=AuditManifest(),
        burn_uid=999,
    )
    assert log.miners == ()
    assert log.miner_census == (census_only,)


def test_duplicate_registered_census_uid_is_rejected() -> None:
    miner = _miner(1, 0.9)
    entry = MinerCensusEntry.from_miner(miner)
    with pytest.raises(EpochLogInvalid, match="duplicate miner_census uid"):
        _log(
            weight_shares={1: 1.0},
            miners=[miner],
            miner_census=(entry, entry),
            manifest=AuditManifest(per_uid={1: (_bundle_ref(1), _packet_ref(1))}),
        )


def test_economic_miner_must_match_registered_census_identity() -> None:
    miner = _miner(1, 0.9)
    manifest = AuditManifest(per_uid={1: (_bundle_ref(1), _packet_ref(1))})
    with pytest.raises(EpochLogInvalid, match="absent from miner_census"):
        _log(
            weight_shares={1: 1.0}, miners=[miner], miner_census=(), manifest=manifest
        )
    with pytest.raises(EpochLogInvalid, match="does not match miner_census identity"):
        _log(
            weight_shares={1: 1.0},
            miners=[miner],
            miner_census=(
                MinerCensusEntry(uid=1, hotkey="impostor", coldkey="ck1", ip="10.0.0.1"),
            ),
            manifest=manifest,
        )


def test_naive_created_at_is_rejected_at_construction() -> None:
    """an internal review: a timezone-NAIVE created_at is rejected at the construction
    boundary (it would crash the auditor's aware close-block-time subtraction)."""
    shares = {1: 1.0}
    u16 = quantize_u16(shares)
    with pytest.raises(EpochLogInvalid, match="timezone-NAIVE"):
        EpochLog(
            epoch_id=41822,
            close_block=15057191,
            scorer_version="scoring-1.0.0+abc123def456",
            created_at=datetime(2026, 8, 21, 12, 0, 0),  # NAIVE (no tzinfo)
            miners=(_miner(1, 0.9),),
            miner_census=(MinerCensusEntry.from_miner(_miner(1, 0.9)),),
            weight_shares=shares,
            weight_u16=u16,
            weight_vector_digest=weight_vector_digest(u16),
            audit_manifest=AuditManifest(
                per_uid={1: (_bundle_ref(1), _packet_ref(1))},
                fold_cursors={1: None},
            ),
        )


def test_from_json_rejects_naive_created_at() -> None:
    """an internal review: canonical bytes carrying an offset-NAIVE created_at are rejected at
    parse (EpochLogInvalid), never round-tripped into a log that later crashes the audit."""
    import json

    data = json.loads(_valid_log().to_json())
    data["created_at"] = "2026-08-21T12:00:00"  # NO offset -> datetime.fromisoformat is naive
    with pytest.raises(EpochLogInvalid, match="timezone-NAIVE"):
        EpochLog.from_json(json.dumps(data).encode())


def _packet_digest(uid: int) -> str:
    return sha256_hex(f"packet-{uid}".encode())


def _v3_manifest() -> AuditManifest:
    from vidaio.epoch import CycleScore, EarningInput

    return AuditManifest(
        per_uid={1: (_bundle_ref(1), _packet_ref(1))},
        earning_inputs={
            1: EarningInput(
                prior_accumulate_score=0.2,
                cycle_scores=(
                    CycleScore(packet_digest=_packet_digest(1), ordering_key=0, score=0.5),
                    CycleScore(packet_digest=_packet_digest(1), ordering_key=1, score=0.6),
                ),
            )
        },
        fold_cursors={1: 1},
    )


def test_v3_fields_survive_roundtrip_and_are_order_independent() -> None:
    """#1/#9: earning_inputs (evidence-bound cycle scores), prior_log_digest, and
    committed_track are part of the canonical bytes and survive a JSON round-trip."""
    prior_digest = sha256_hex(b"prior-epoch-log")
    shares = {1: 1.0}
    a = EpochLog(
        epoch_id=100, close_block=360_000, scorer_version="s+1", created_at=NOW,
        prior_log_digest=prior_digest, burn_uid=None, miners=(_miner(1, 0.9),),
        miner_census=(MinerCensusEntry.from_miner(_miner(1, 0.9)),),
        weight_shares=shares,
        weight_u16=quantize_u16(shares), weight_vector_digest=weight_vector_digest(quantize_u16(shares)),
        audit_manifest=_v3_manifest(),
    )
    back = EpochLog.from_json(a.to_json())
    assert back.to_json() == a.to_json()
    assert back.log_digest() == a.log_digest()
    ei = back.audit_manifest.earning_for(1)
    assert ei is not None and ei.prior_accumulate_score == 0.2
    assert ei.folded_scores() == (0.5, 0.6)  # committed ordering_key order preserved
    assert [c.ordering_key for c in ei.cycle_scores] == [0, 1]
    assert back.audit_manifest.fold_cursors == {1: 1}
    assert back.prior_log_digest == prior_digest
    packet_ref = next(r for r in back.audit_manifest.refs_for(1) if r.committed_track is not None)
    assert packet_ref.committed_track == "compression"
    assert back.audit_manifest == a.audit_manifest


def test_schema_version_default() -> None:
    assert _valid_log().schema_version == EPOCH_LOG_SCHEMA_VERSION


def test_score_packet_ref_requires_committed_track() -> None:
    """#9: a SCORE_PACKET ref MUST pin a committed track (no packet-track fallback)."""
    with pytest.raises(EpochLogInvalid, match="committed_track"):
        AuditFileRef(
            kind=AuditFileKind.SCORE_PACKET,
            digest=sha256_hex(b"p"),
            challenge_id="c1",
            item_id="i1",
            source="inference",
        )
    # an AUDIT_BUNDLE ref carries no track and is fine without one
    AuditFileRef(
        kind=AuditFileKind.AUDIT_BUNDLE, digest=sha256_hex(b"b"),
        challenge_id="c1", item_id="i1",
    )


def test_duplicate_snapshot_uids_are_rejected() -> None:
    """an internal review: two snapshot rows for one uid let unaudited rows influence the
    weight vector (build_weight_vector folds every row) while the auditor's uid->snapshot
    map keeps only the last — refuse the log at construction."""
    miners = [_miner(1, 0.9), _miner(1, 0.8, track="upscaling")]  # uid 1 twice
    shares = {1: 1.0}
    manifest = AuditManifest(per_uid={1: (_bundle_ref(1), _packet_ref(1))})
    with pytest.raises(EpochLogInvalid, match="duplicate miner snapshot uid"):
        _log(weight_shares=shares, miners=miners, manifest=manifest)


def test_audit_file_ref_rejects_unknown_source() -> None:
    """an internal review: `source` outside the known media set would dodge the all-SKIP
    media-coverage floor — refuse it at the model level (a spoofed source is tampering)."""
    with pytest.raises(EpochLogInvalid, match="not a known media source"):
        AuditFileRef(
            kind=AuditFileKind.AUDIT_BUNDLE,
            digest=sha256_hex(b"b"),
            challenge_id="c1",
            item_id="i1",
            source="totally-made-up",
        )
    # the two legitimate media sources still construct fine
    for src in ("competition", "inference"):
        AuditFileRef(
            kind=AuditFileKind.AUDIT_BUNDLE, digest=sha256_hex(b"b"),
            challenge_id="c1", item_id="i1", source=src,
        )


def test_earning_input_rejects_non_ascending_ordering_keys() -> None:
    """The evidence-bound fold order is strictly ascending — a reordered sequence is
    not even representable in a valid log."""
    from vidaio.epoch import CycleScore, EarningInput

    with pytest.raises(EpochLogInvalid, match="ascending"):
        EarningInput(
            cycle_scores=(
                CycleScore(packet_digest=sha256_hex(b"a"), ordering_key=2, score=0.1),
                CycleScore(packet_digest=sha256_hex(b"b"), ordering_key=1, score=0.9),
            )
        )


# --------------------------------------------------------------------------------------
# Invariants — a log that violates any of these is REJECTED.
# --------------------------------------------------------------------------------------


def test_u16_must_equal_quantize_of_float() -> None:
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    shares = {1: 0.8, 2: 0.2}
    good = quantize_u16(shares)
    bad = dict(good)
    bad[1] += 1  # perturb one unit -> no longer quantize_u16(shares)
    manifest = AuditManifest(per_uid={1: (_packet_ref(1),), 2: (_packet_ref(2),)})
    with pytest.raises(EpochLogInvalid, match="quantize_u16"):
        _log(weight_shares=shares, miners=miners, manifest=manifest, weight_u16=bad)


def test_partial_float_vector_cannot_omit_canonical_sink() -> None:
    """A partial vector must not become a full donated u16 vector by normalization."""
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    shares = {1: 0.64, 2: 0.16}  # IDLE earners with the fixed 20% sink omitted
    manifest = AuditManifest(
        per_uid={1: (_packet_ref(1),), 2: (_packet_ref(2),)}
    )
    assert sum(quantize_u16(shares).values()) == 65535  # the dangerous normalization

    with pytest.raises(EpochLogInvalid, match="complete fixed emission vector"):
        _log(weight_shares=shares, miners=miners, manifest=manifest)


def test_weight_vector_digest_must_bind_u16() -> None:
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    shares = {1: 0.8, 2: 0.2}
    manifest = AuditManifest(per_uid={1: (_packet_ref(1),), 2: (_packet_ref(2),)})
    with pytest.raises(EpochLogInvalid, match="weight_vector_digest"):
        _log(weight_shares=shares, miners=miners, manifest=manifest, wv_digest="0" * 64)


def test_nonzero_weight_without_manifest_entry_or_carried_accumulator_is_rejected() -> None:
    """A nonzero-weight uid with NO manifest entry AND a non-positive accumulate_score has no
    audit backing at all (its weight is not derived from a carried accumulator) — rejected.

    an internal review: a nonzero-weight uid with no refs IS allowed as a pure carry-forward, but
    ONLY when its snapshot carries a positive accumulator (backed by the prior epoch's fold, which
    the auditor chains). A zero/negative accumulator has no such backing."""
    miners = [_miner(1, 0.9), _miner(2, 0.0, track="upscaling")]  # uid 2 accumulator not positive
    shares = {1: 0.8, 2: 0.2}
    manifest = AuditManifest(per_uid={1: (_packet_ref(1),)})  # uid 2 missing
    with pytest.raises(EpochLogInvalid, match="uid 2 has nonzero weight"):
        _log(weight_shares=shares, miners=miners, manifest=manifest)


def test_nonzero_weight_carry_forward_without_manifest_entry_is_allowed() -> None:
    """an internal review: a nonzero-weight uid with NO manifest entry, NO earning input, and a
    POSITIVE carried accumulator is a valid pure CARRY-FORWARD (an idle prior earner). The schema
    defers its cross-epoch backing to the auditor's `_carry_forward_verdict`; construction is not
    refused (the pre-round-20 schema rejected every carry-forward, so idle epochs could not be
    represented)."""
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]  # uid 2 carries 0.8 forward
    shares = {1: 0.8, 2: 0.2}
    manifest = AuditManifest(per_uid={1: (_packet_ref(1),)})  # uid 2 has no current evidence
    log = _log(weight_shares=shares, miners=miners, manifest=manifest)
    assert log.weight_shares[2] == 0.2
    assert not log.audit_manifest.refs_for(2)


def test_zero_weight_uid_needs_no_manifest_entry() -> None:
    # uid 3 is a known miner with zero weight -> no manifest entry required.
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling"), _miner(3, 0.0)]
    shares = {1: 0.8, 2: 0.2, 3: 0.0}
    manifest = AuditManifest(
        per_uid={1: (_packet_ref(1),), 2: (_packet_ref(2),)}
    )
    log = _log(weight_shares=shares, miners=miners, manifest=manifest)
    assert log.weight_shares[3] == 0.0


# --------------------------------------------------------------------------------------
# Empty-epoch burn vector.
# --------------------------------------------------------------------------------------


def test_empty_epoch_burn_vector_needs_no_manifest() -> None:
    log = _log(
        weight_shares={7: 1.0},
        miners=[],
        manifest=AuditManifest(),  # possibly-empty manifest
        burn_uid=7,
    )
    assert log.weight_u16 == {7: 65535}
    assert log.burn_uid == 7
    assert log.audit_manifest.per_uid == {}
    # roundtrips like any other log
    assert EpochLog.from_json(log.to_json()).log_digest() == log.log_digest()


def test_burn_uid_set_but_not_burn_vector_is_rejected() -> None:
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    shares = {1: 0.8, 2: 0.2}
    manifest = AuditManifest(per_uid={1: (_packet_ref(1),), 2: (_packet_ref(2),)})
    with pytest.raises(EpochLogInvalid, match="burn_uid .* no positive withheld share"):
        _log(weight_shares=shares, miners=miners, manifest=manifest, burn_uid=99)


def test_evidenced_or_seated_burn_uid_is_refused() -> None:
    """an internal review: the RESERVED empty-epoch burn uid must NOT double as a census/evidence
    identity. The auditor exempts burn_uid from the snapshot/identity/dedup/track binding AND from
    the earning fold, so a log that seats burn_uid in `miners` (with another miner's evidence + a
    self-attested hotkey) and publishes {burn_uid:1.0} would ride free ⇒ CLEAN. Refuse any overlap
    between burn_uid and the census / earning / manifest evidence at the construction boundary."""
    burn_uid = 7
    shares = {burn_uid: 1.0}
    # (a) burn_uid ALSO seated as a census miner
    with pytest.raises(EpochLogInvalid, match="seated as a census miner"):
        _log(
            weight_shares=shares,
            miners=[_miner(burn_uid, 0.0)],
            manifest=AuditManifest(),
            burn_uid=burn_uid,
        )
    # (b) burn_uid carries earning/manifest evidence
    with pytest.raises(EpochLogInvalid, match="earning/manifest evidence"):
        _log(
            weight_shares=shares,
            miners=[],
            manifest=AuditManifest(
                per_uid={burn_uid: (_bundle_ref(burn_uid), _packet_ref(burn_uid))}
            ),
            burn_uid=burn_uid,
        )


def test_out_of_protocol_track_is_refused() -> None:
    """an internal review: every committed / log TRACK must be a MEMBER of the protocol track set.
    `_require_committed_track_on_packets` only requires a non-null committed track, and tokenomics
    SILENTLY drops a miner whose track is absent from `track_weights` — so committing evidence
    under an out-of-protocol track (e.g. "unknown") substitutes a burn while every declaration
    self-agrees. Refuse it at the construction boundary, for both the snapshot track and the
    committed-track on an audit ref."""
    # (a) a MinerSnapshot on an out-of-protocol track
    with pytest.raises(EpochLogInvalid, match="NOT a protocol track"):
        _log(
            weight_shares={1: 1.0},
            miners=[_miner(1, 0.9, track="unknown")],
            manifest=AuditManifest(per_uid={1: (_bundle_ref(1), _packet_ref(1))}),
        )
    # (b) a SCORE_PACKET ref carrying an out-of-protocol committed_track
    bad_ref = AuditFileRef(
        kind=AuditFileKind.SCORE_PACKET,
        digest=sha256_hex(b"packet-1"),
        challenge_id="c1", item_id="i1", source="inference",
        committed_track="unknown",
    )
    with pytest.raises(EpochLogInvalid, match="NOT a protocol track"):
        _log(
            weight_shares={1: 1.0},
            miners=[_miner(1, 0.9)],
            manifest=AuditManifest(per_uid={1: (_bundle_ref(1), bad_ref)}),
        )


def test_from_json_refuses_burn_uid_seated_as_evidence_identity() -> None:
    """an internal review (companion): the same refusal fires on UNTRUSTED BYTES — a tampered burn
    log that seats the reserved burn uid in `miners` is rejected at `from_json`, so a log that
    bypassed the finalizer cannot be round-tripped into an auditable object."""
    import json

    burn = _log(
        weight_shares={7: 1.0}, miners=[], manifest=AuditManifest(), burn_uid=7
    )
    data = json.loads(burn.to_json())
    # Inject the reserved burn uid (7) as a zero-weight census miner — the burn hole.
    data["miners"] = [
        {
            "uid": 7, "hotkey": "hk7", "coldkey": "ck7", "ip": "10.0.0.7",
            "track": "compression", "accumulate_score": 0.0, "excluded": False,
        }
    ]
    data["miner_census"] = [
        {"uid": 7, "hotkey": "hk7", "coldkey": "ck7", "ip": "10.0.0.7"}
    ]
    data["audit_manifest"]["fold_cursors"] = {"7": None}
    with pytest.raises(EpochLogInvalid, match="seated as a census miner"):
        EpochLog.from_json(json.dumps(data).encode())


def test_foreign_epoch_log_schema_is_refused() -> None:
    """an internal review: `schema_version` is ENFORCED. Current-shape bytes LABELLED a FOREIGN
    schema (14, 15, or 17) is refused (EpochLogInvalid), so validators on a different code version
    cannot converge on a foreign-schema log; only the code's own EPOCH_LOG_SCHEMA_VERSION (16)
    is accepted."""
    import json

    valid = json.loads(_valid_log().to_json())
    assert valid["schema_version"] == EPOCH_LOG_SCHEMA_VERSION == 16
    for bad in (14, 15, 17):
        data = dict(valid, schema_version=bad)
        with pytest.raises(EpochLogInvalid, match="schema_version"):
            EpochLog.from_json(json.dumps(data).encode())
    # the exact code schema (16) still parses cleanly
    assert EpochLog.from_json(json.dumps(valid).encode()).schema_version == 16


def _v14_competition_input() -> CompetitionInput:
    from vidaio.epoch import CompetitionAuditItem

    commitment_root = "2" * 64
    anchor_payload = (
        f"{COMMITMENT_DOMAIN}:competition:{commitment_root}".encode("ascii")
    )
    return CompetitionInput(
        competition_id="competition-v14",
        track="compression",
        cycle=1,
        completed_at=NOW,
        applied_at=NOW,
        manifest_digest="1" * 64,
        commitment_root=commitment_root,
        anchor_netuid=85,
        anchor_payload_hex=anchor_payload.hex(),
        anchor_payload_digest=sha256_hex(anchor_payload),
        anchor_block=10,
        anchor_block_hash="f" * 64,
        anchor_finalized_block=11,
        baseline_version=3,
        baseline_artifact_digest="3" * 64,
        baseline_artifact_bytes=1024,
        baseline_execution_image_digest="4" * 64,
        baseline_provenance_digest="5" * 64,
        baseline_provenance_bytes=256,
        items=(
            CompetitionAuditItem(
                challenge_id="challenge-1",
                item_id="item-1",
                threshold_commitment="6" * 64,
            ),
        ),
        subjects=(
            CompetitionAuditSubject(
                subject_id="baseline",
                role="baseline",
                execution_image_digest="4" * 64,
                packet_digests=("7" * 64,),
                audit_bundle_digests=("8" * 64,),
            ),
            CompetitionAuditSubject(
                subject_id="contender:hk1",
                role="contender",
                uid=1,
                hotkey="hk1",
                submission_archive_digest="9" * 64,
                submission_archive_bytes=4096,
                execution_image_digest="a" * 64,
                repo_url="https://example.invalid/miner.git",
                commit_sha="b" * 40,
                tree_sha="c" * 40,
                packet_digests=("d" * 64,),
                audit_bundle_digests=("e" * 64,),
            ),
        ),
    )


def test_schema_v14_competition_input_commits_release_provenance() -> None:
    value = _v14_competition_input()
    obj = value._canonical_obj()

    assert value.aggregation_version == "mean_item_score.v2"
    assert obj["anchor_netuid"] == 85
    assert bytes.fromhex(obj["anchor_payload_hex"]).decode("ascii").endswith(
        value.commitment_root
    )
    assert obj["anchor_payload_digest"] == sha256_hex(
        bytes.fromhex(obj["anchor_payload_hex"])
    )
    assert obj["anchor_block"] == 10
    assert obj["anchor_block_hash"] == "f" * 64
    assert obj["anchor_finalized_block"] == 11
    assert obj["baseline_version"] == 3
    assert obj["baseline_artifact_bytes"] == 1024
    assert obj["baseline_execution_image_digest"] == "4" * 64
    assert obj["baseline_provenance_bytes"] == 256
    contender = obj["subjects"][1]
    assert contender["submission_archive_bytes"] == 4096
    assert contender["execution_image_digest"] == "a" * 64
    assert contender["commit_sha"] == "b" * 40
    assert contender["tree_sha"] == "c" * 40


def test_schema_v14_rejects_legacy_subject_roles_and_unsealed_contender() -> None:
    valid = _v14_competition_input()
    with pytest.raises(ValidationError):
        valid.subjects[0].model_copy(update={"role": "legacy_calibration"}).model_validate(
            {**valid.subjects[0].model_dump(), "role": "legacy_calibration"}
        )
    with pytest.raises(EpochLogInvalid, match="sealed release identity"):
        CompetitionAuditSubject(
            subject_id="contender:unsealed",
            role="contender",
            uid=2,
            hotkey="hk2",
            execution_image_digest="a" * 64,
            packet_digests=("f" * 64,),
            audit_bundle_digests=("0" * 64,),
        )


def test_schema_v14_rejects_retired_top_level_fields() -> None:
    import json

    data = json.loads(_valid_log().to_json())
    data["current_cycle"] = 0
    with pytest.raises(EpochLogInvalid, match="retired canonical field"):
        EpochLog.from_json(json.dumps(data).encode())


# --------------------------------------------------------------------------------------
# v16 outage-gap declaration (P1.5)
# --------------------------------------------------------------------------------------


def _gap_log(gap_epochs: tuple[int, ...], prior_digest: str | None) -> EpochLog:
    miners = [_miner(1, 0.9), _miner(2, 0.8, track="upscaling")]
    manifest = AuditManifest(
        per_uid={1: (_bundle_ref(1), _packet_ref(1)), 2: (_bundle_ref(2), _packet_ref(2))}
    )
    resolved_census = tuple(MinerCensusEntry.from_miner(m) for m in miners)
    cursors = dict(manifest.fold_cursors)
    for entry in resolved_census:
        cursors.setdefault(entry.uid, None)
    manifest = manifest.model_copy(update={"fold_cursors": cursors})
    shares = {1: 0.8, 2: 0.2}
    u16 = quantize_u16(shares)
    return EpochLog(
        epoch_id=41822,
        close_block=15057191,
        scorer_version="scoring-1.0.0+abc123def456",
        created_at=NOW,
        prior_log_digest=prior_digest,
        gap_epochs=gap_epochs,
        miners=tuple(miners),
        miner_census=resolved_census,
        weight_shares=shares,
        weight_u16=u16,
        weight_vector_digest=weight_vector_digest(u16),
        audit_manifest=manifest,
    )


def test_gap_epochs_roundtrip_and_prior_epoch_id() -> None:
    prior = "ab" * 32
    log = _gap_log((41819, 41820, 41821), prior)
    assert log.prior_epoch_id == 41818
    parsed = EpochLog.from_json(log.to_json())
    assert parsed.gap_epochs == (41819, 41820, 41821)
    assert parsed.prior_epoch_id == 41818
    # no gap: predecessor is the literal epoch_id - 1
    plain = _gap_log((), prior)
    assert plain.prior_epoch_id == 41821
    # genesis: no predecessor at all
    assert _gap_log((), None).prior_epoch_id is None


def test_gap_epochs_change_the_digest() -> None:
    prior = "ab" * 32
    assert (
        _gap_log((41821,), prior).log_digest() != _gap_log((), prior).log_digest()
    )


def test_gap_on_genesis_is_refused() -> None:
    with pytest.raises(EpochLogInvalid, match="genesis"):
        _gap_log((41820, 41821), None)


def test_gap_must_be_the_full_contiguous_range_below_epoch_id() -> None:
    prior = "ab" * 32
    # hole in the range
    with pytest.raises(EpochLogInvalid, match="contiguous"):
        _gap_log((41819, 41821), prior)
    # range not ending at epoch_id - 1
    with pytest.raises(EpochLogInvalid, match="contiguous"):
        _gap_log((41818, 41819), prior)
    # reordered
    with pytest.raises(EpochLogInvalid, match="contiguous"):
        _gap_log((41821, 41820), prior)


def test_gap_declaration_survives_from_json_reconstruction() -> None:
    import json

    log = _gap_log((41820, 41821), "cd" * 32)
    obj = json.loads(log.to_json())
    # tampering the declared gap breaks the digest chain (different canonical bytes)
    obj["gap_epochs"] = [41821]
    tampered = EpochLog.from_json(json.dumps(obj).encode())
    assert tampered.log_digest() != log.log_digest()
