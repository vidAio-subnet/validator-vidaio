import json

import pytest
from pydantic import ValidationError

from vidaio.scoring.result import ItemScore

BASE = {"item_id": "i", "challenge_id": "c", "track": "compression", "score": 0, "gate_passed": False}
EVIDENCE = {"canonical_content_digest": "a" * 64, "content_fingerprint": ["f" * 16] * 32, "encoded_size": 321}


def test_old_packet_json_bytes_are_unchanged():
    original = '{"item_id":"i","challenge_id":"c","track":"compression","miner_hotkey":null,"content_digest":null,"score":0.0,"gate_passed":false,"violations":[],"skips":[],"breakdown":null,"metrics":{},"scorer_version":null,"backend_versions":{},"canonicalization_plan_digest":null,"pieapp_start_frame":null,"scoring_config_digest":null}'
    score = ItemScore(**BASE)
    assert score.to_json() == original
    assert ItemScore.from_json(original).to_json() == original
    assert not set(EVIDENCE) & score.model_dump().keys()


def test_new_evidence_roundtrips_without_replacing_encoded_digest():
    packet = ItemScore(**BASE, **EVIDENCE, content_digest="b" * 64)
    assert isinstance(packet.content_fingerprint, tuple)
    body = json.loads(packet.to_json())
    assert body["content_digest"] == "b" * 64
    for key, value in EVIDENCE.items():
        assert body[key] == value
    assert ItemScore.from_json(packet.to_json()) == packet


@pytest.mark.parametrize("key,value", [
    ("canonical_content_digest", None), ("canonical_content_digest", "A" * 64),
    ("canonical_content_digest", "a" * 63), ("content_fingerprint", None),
    ("content_fingerprint", ["a" * 16] * 31), ("content_fingerprint", ["a" * 16] * 33),
    ("content_fingerprint", ["A" * 16] * 32), ("content_fingerprint", ["a" * 15] * 32),
    ("encoded_size", None), ("encoded_size", True), ("encoded_size", 0),
    ("encoded_size", -1), ("encoded_size", 321.0), ("encoded_size", "321"),
])
def test_invalid_or_partial_evidence_is_not_an_item_score(key, value):
    changed = EVIDENCE | {key: value}
    with pytest.raises(ValidationError):
        ItemScore(**BASE, **changed)
