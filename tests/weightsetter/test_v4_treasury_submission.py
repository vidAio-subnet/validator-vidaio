"""Tokenomics v4 logs name the licensing treasury at the top level; the weight-setter
submission boundary (also the own-auditor's submission identity) must accept them."""

from __future__ import annotations

import json

import pytest

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.authority.api import AnchorPointer, EpochPointer
from vidaio.weightsetter.shared_snapshot import (
    SnapshotDigestMismatch,
    parse_authority_history_submission,
    parse_authority_submission,
)

from tests.auditor.test_licensing_treasury import _gap_log, _v4


def _pointer(raw: bytes) -> EpochPointer:
    obj = json.loads(raw)
    digest = sha256_hex(raw)
    return EpochPointer(
        epoch_id=obj["epoch_id"], close_block=obj["close_block"],
        snapshot_key=f"finalized/epoch={obj['epoch_id']}/log.json", snapshot_digest=digest,
        weight_vector_digest=obj["weight_vector_digest"],
        anchor=AnchorPointer(txid="test-authenticated-anchor", digest=digest, block=123),
    )


def test_a_v4_log_naming_the_treasury_is_a_valid_submission() -> None:
    raw = _gap_log(_v4()).to_json()
    assert b'"treasury_hotkey"' in raw
    view = parse_authority_submission(raw, _pointer(raw))
    assert view == parse_authority_history_submission(raw, _pointer(raw))


@pytest.mark.parametrize("bad", ("", "   ", 7))
def test_an_empty_or_non_string_treasury_is_refused(bad) -> None:
    obj = json.loads(_gap_log(_v4()).to_json())
    obj["treasury_hotkey"] = bad
    raw = canonical_json_bytes(obj)
    with pytest.raises(SnapshotDigestMismatch, match="treasury_hotkey"):
        parse_authority_submission(raw, _pointer(raw))


def test_any_other_extra_top_level_field_is_still_refused() -> None:
    obj = json.loads(_gap_log(_v4()).to_json())
    obj["unexpected"] = 1
    raw = canonical_json_bytes(obj)
    with pytest.raises(SnapshotDigestMismatch, match="exact schema"):
        parse_authority_submission(raw, _pointer(raw))
