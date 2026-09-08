"""Historical submission diagnostics cannot enter the current weight boundary."""
import json
import pytest
from tests.epoch.test_schema17_history import legacy_bytes, current_log
from vidaio.audit.canonical import sha256_hex
from vidaio.authority.api import AnchorPointer, EpochPointer
from vidaio.weightsetter.shared_snapshot import (
    parse_authority_history_submission, parse_authority_submission, SnapshotDigestMismatch,
)


def pointer(raw):
    obj=json.loads(raw);digest=sha256_hex(raw)
    return EpochPointer(epoch_id=obj['epoch_id'],close_block=obj['close_block'],
        snapshot_key=f"finalized/epoch={obj['epoch_id']}/log.json",snapshot_digest=digest,
        weight_vector_digest=obj['weight_vector_digest'],
        anchor=AnchorPointer(txid='test-authenticated-anchor',digest=digest,block=123))


def test_v16_history_keeps_packet_commitment_without_entering_current_weights():
    raw=legacy_bytes();p=pointer(raw)
    view=parse_authority_history_submission(raw,p)
    assert view.packet_digests==()
    assert view.miner_census_hotkeys=={1:'h'}
    with pytest.raises(SnapshotDigestMismatch,match='schema_version'):
        parse_authority_submission(raw,p)


def test_history_requires_exact_authenticated_bytes():
    raw=legacy_bytes();p=pointer(raw)
    with pytest.raises(SnapshotDigestMismatch,match='historical submission bytes'):
        parse_authority_history_submission(raw+b' ',p)


def test_current_submission_and_history_share_current_shape():
    raw=current_log().to_json();p=pointer(raw)
    assert parse_authority_submission(raw,p)==parse_authority_history_submission(raw,p)
