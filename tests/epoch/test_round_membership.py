"""The unpublished v17 commit rule is canonical; v16 history stays exact."""
import json

import pytest
from pydantic import ValidationError

from tests.epoch.test_schema17_history import current_log, legacy_bytes
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.epoch import AuditManifest, EpochLog, EpochLogInvalid, RoundCommitInput


def entry(challenge="c1", ordering_key=1, commit_block=90, round_id="round-a"):
    return RoundCommitInput(
        round_id=round_id, challenge_id=challenge, commit_block=commit_block,
        anchor_block=80, ordering_key=ordering_key,
    )


def test_round_index_is_canonical_and_binds_the_log_digest():
    first, second = entry(), entry("c2", 2)
    a = AuditManifest(round_commits=(first, second), round_commit_cursor=2, fold_cursors={1: None})
    b = AuditManifest(round_commits=(second, first), round_commit_cursor=2, fold_cursors={1: None})
    log = current_log().model_copy(update={"audit_manifest": a})
    reordered = current_log().model_copy(update={"audit_manifest": b})
    assert log.to_json() == reordered.to_json()
    assert EpochLog.from_json(log.to_json()).audit_manifest.round_commits == (first, second)
    changed = current_log().model_copy(update={"audit_manifest": AuditManifest(
        round_commits=(entry(commit_block=91), entry("c2", 2, commit_block=91)),
        round_commit_cursor=2, fold_cursors={1: None},
    )})
    assert changed.log_digest() != log.log_digest()


@pytest.mark.parametrize("field", ["round_membership", "round_commits", "round_commit_cursor"])
def test_current_bytes_cannot_omit_membership_contract(field):
    obj = json.loads(current_log().to_json())
    target = obj if field == "round_membership" else obj["audit_manifest"]
    target.pop(field)
    with pytest.raises(EpochLogInvalid, match="round_membership|round_commits"):
        EpochLog.from_json(canonical_json_bytes(obj))


@pytest.mark.parametrize("rule", [None, "assignment/1", "commit/2", True])
def test_current_bytes_cannot_choose_a_legacy_or_unknown_rule(rule):
    obj = json.loads(current_log().to_json())
    obj["round_membership"] = rule
    with pytest.raises(EpochLogInvalid, match="round_membership"):
        EpochLog.from_json(canonical_json_bytes(obj))


@pytest.mark.parametrize("rows,cursor", [
    ((entry(), entry("c2", 1)), 1),
    ((entry(), entry("c1", 2)), 2),
    ((entry(), entry("c2", 2, commit_block=91)), 2),
    ((entry(),), None),
])
def test_round_index_rejects_split_commit_duplicate_challenge_and_missing_cursor(rows, cursor):
    with pytest.raises((EpochLogInvalid, ValidationError)):
        AuditManifest(round_commits=rows, round_commit_cursor=cursor)


@pytest.mark.parametrize("field", ["commit_block", "anchor_block", "ordering_key"])
@pytest.mark.parametrize("value", [True, "90", 90.0])
def test_membership_heights_and_keys_are_strict_integers(field, value):
    obj = entry().model_dump()
    obj[field] = value
    with pytest.raises(ValidationError):
        RoundCommitInput(**obj)


def test_authenticated_v16_has_assignment_rule_without_new_canonical_bytes():
    raw = legacy_bytes()
    history = EpochLog.from_history_json(raw, expected_digest=sha256_hex(raw), expected_epoch_id=42)
    assert history.round_membership == "assignment/1"
    assert history.audit_manifest.round_commits == ()
    assert history.audit_manifest.round_commit_cursor is None
    assert history.to_json() == raw
    obj = json.loads(raw)
    assert "round_membership" not in obj
    assert "round_commits" not in obj["audit_manifest"]
    assert "round_commit_cursor" not in obj["audit_manifest"]
    obj["round_membership"] = "commit/1"
    tampered = canonical_json_bytes(obj)
    with pytest.raises(EpochLogInvalid):
        EpochLog.from_history_json(tampered, expected_digest=sha256_hex(tampered), expected_epoch_id=42)


def test_zero_historical_cursor_is_representable_but_dispatch_keys_stay_positive():
    manifest = AuditManifest(round_commit_cursor=0, fold_cursors={1: None})
    log = current_log().model_copy(update={"audit_manifest": manifest})
    assert EpochLog.from_json(log.to_json()).audit_manifest.round_commit_cursor == 0
    with pytest.raises(ValidationError):
        entry(ordering_key=0)
    with pytest.raises(ValidationError):
        AuditManifest(round_commit_cursor=-1)
