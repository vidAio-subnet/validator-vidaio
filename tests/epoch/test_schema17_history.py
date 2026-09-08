"""Current-schema fencing and authenticated history preserve the anchored v16 bytes."""
from dataclasses import replace
from datetime import datetime, timezone
import json

import pytest

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.epoch import AuditManifest, EpochLog, EpochLogInvalid, MinerCensusEntry, weight_vector_digest
from vidaio.tokenomics.state import MinerSnapshot


def current_log():
    miner = MinerSnapshot(uid=1, hotkey="h", coldkey="c", ip="10.0.0.1", track="compression", accumulate_score=0.0, alpha_stake=4.5)
    return EpochLog(epoch_id=42, close_block=100, scorer_version="s", created_at=datetime(2026,9,6,tzinfo=timezone.utc), burn_uid=0, miners=(miner,), miner_census=(MinerCensusEntry.from_miner(miner),), weight_shares={0:1.0}, weight_u16={0:65535}, weight_vector_digest=weight_vector_digest({0:65535}), audit_manifest=AuditManifest(fold_cursors={1:None}))


def legacy_bytes():
    obj=json.loads(current_log().to_json())
    obj["schema_version"]=16
    obj["audit_manifest"].pop("content_rounds")
    obj["audit_manifest"].pop("round_commits")
    obj["audit_manifest"].pop("round_commit_cursor")
    obj.pop("round_membership")
    obj.pop("payout_min_alpha_stake")
    for item in obj["miners"]+obj["miner_census"]:
        item.pop("alpha_stake")
    return canonical_json_bytes(obj)


def test_history_preserves_schema_and_canonical_digest():
    raw=legacy_bytes()
    old=EpochLog.from_history_json(raw,expected_digest=sha256_hex(raw),expected_epoch_id=42)
    assert old.schema_version==16
    assert old.to_json()==raw
    assert old.log_digest()==sha256_hex(raw)
    assert old.miners[0].alpha_stake==0.0
    newer=current_log().model_copy(update={"epoch_id":43,"prior_log_digest":old.log_digest()})
    assert EpochLog.from_json(newer.to_json()).prior_log_digest==sha256_hex(raw)
    with pytest.raises(EpochLogInvalid,match="schema_version"):
        EpochLog.from_json(raw)


@pytest.mark.parametrize("mutation",["wrong_digest","wrong_epoch","relabel","noncanonical","unknown_schema","legacy_new_fields"])
def test_history_refuses_unbound_or_rewritten_input(mutation):
    raw=legacy_bytes(); digest=sha256_hex(raw); epoch=42
    if mutation=="wrong_digest": digest="f"*64
    elif mutation=="wrong_epoch": epoch=41
    else:
        obj=json.loads(raw)
        if mutation=="relabel": obj["schema_version"]=17
        elif mutation=="unknown_schema": obj["schema_version"]=15
        elif mutation=="legacy_new_fields": obj["miners"][0]["alpha_stake"]=10.0
        raw=json.dumps(obj).encode() if mutation=="noncanonical" else canonical_json_bytes(obj)
        digest=sha256_hex(raw)
    with pytest.raises(EpochLogInvalid):
        EpochLog.from_history_json(raw,expected_digest=digest,expected_epoch_id=epoch)


@pytest.mark.parametrize("location",["miners","miner_census"])
def test_current_requires_exact_close_block_alpha(location):
    obj=json.loads(current_log().to_json())
    del obj[location][0]["alpha_stake"]
    with pytest.raises(EpochLogInvalid,match="alpha_stake"):
        EpochLog.from_json(canonical_json_bytes(obj))


def test_current_canonical_alpha_roundtrip_and_no_implicit_history():
    log=current_log()
    assert EpochLog.from_json(log.to_json())==log
    assert json.loads(log.to_json())["miner_census"][0]["alpha_stake"]==4.5
    assert EpochLog.from_history_json(log.to_json(),expected_digest=log.log_digest(),expected_epoch_id=42)==log


def test_current_census_snapshot_alpha_mismatch_rejected():
    obj=json.loads(current_log().to_json())
    obj['miners'][0]['alpha_stake']=9.0
    with pytest.raises(EpochLogInvalid,match='match miner_census identity'):
        EpochLog.from_json(canonical_json_bytes(obj))


@pytest.mark.parametrize('schema',[17.0,'17',True])
def test_current_schema_requires_integer_type(schema):
    obj=json.loads(current_log().to_json());obj['schema_version']=schema
    with pytest.raises(EpochLogInvalid,match='schema_version'):
        EpochLog.from_json(canonical_json_bytes(obj))


def test_current_archives_the_applied_payout_floor_and_history_implies_zero():
    """D-025: v17 logs carry the exact floor they were built with; v16 bytes are untouched."""
    log = current_log().model_copy(update={"payout_min_alpha_stake": 50.0})
    obj = json.loads(log.to_json())
    assert obj["payout_min_alpha_stake"] == 50.0
    assert EpochLog.from_json(log.to_json()).payout_min_alpha_stake == 50.0
    assert EpochLog.from_json(current_log().to_json()).payout_min_alpha_stake == 0.0

    stripped = dict(obj)
    stripped.pop("payout_min_alpha_stake")
    with pytest.raises(EpochLogInvalid, match="canonical field|payout_min_alpha_stake"):
        EpochLog.from_json(canonical_json_bytes(stripped))
    for bad in (True, "50", None):
        broken = dict(obj)
        broken["payout_min_alpha_stake"] = bad
        with pytest.raises(EpochLogInvalid, match="payout_min_alpha_stake"):
            EpochLog.from_json(canonical_json_bytes(broken))

    raw = legacy_bytes()
    old = EpochLog.from_history_json(raw, expected_digest=sha256_hex(raw), expected_epoch_id=42)
    assert old.payout_min_alpha_stake == 0.0
    assert old.to_json() == raw  # authenticated v16 bytes never gain the field
    legacy_with_floor = json.loads(raw)
    legacy_with_floor["payout_min_alpha_stake"] = 0.0
    tampered = canonical_json_bytes(legacy_with_floor)
    with pytest.raises(EpochLogInvalid):
        EpochLog.from_history_json(tampered, expected_digest=sha256_hex(tampered), expected_epoch_id=42)
