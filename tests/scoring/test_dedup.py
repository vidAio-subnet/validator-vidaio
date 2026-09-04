"""Cross-miner economic dedup is exact-only and deterministic."""

import random

from vidaio.scoring import (
    DedupEntry,
    ReasonCode,
    dedup_responses,
)


def entries() -> list[DedupEntry]:
    return [
        DedupEntry(
            key="miner-a", content_digest="d1", perceptual_hash="ff00", order_key="001"
        ),
        # exact digest replay of miner-a's output
        DedupEntry(
            key="miner-b", content_digest="d1", perceptual_hash="00ff", order_key="002"
        ),
        # different digest but perceptually near-identical (hamming distance 1)
        DedupEntry(
            key="miner-c", content_digest="d3", perceptual_hash="ff01", order_key="003"
        ),
        # genuinely distinct (hamming distance 12 from miner-a's hash)
        DedupEntry(
            key="miner-d", content_digest="d4", perceptual_hash="00aa", order_key="004"
        ),
    ]


def test_exact_duplicate_zeroed_but_near_match_kept() -> None:
    verdicts = dedup_responses(entries())
    assert verdicts["miner-a"].kept
    assert not verdicts["miner-b"].kept
    assert verdicts["miner-b"].duplicate_of == "miner-a"
    assert verdicts["miner-b"].reason == ReasonCode.REPLAY_DUPLICATE
    assert verdicts["miner-c"].kept
    assert verdicts["miner-d"].kept


def test_deterministic_regardless_of_arrival_order() -> None:
    baseline = dedup_responses(entries())
    rng = random.Random(7)  # test-only shuffle; scoring logic itself has no randomness
    for _ in range(5):
        shuffled = entries()
        rng.shuffle(shuffled)
        assert dedup_responses(shuffled) == baseline


def test_order_key_decides_who_is_kept() -> None:
    flipped = [
        e.model_copy(update={"order_key": {"miner-a": "009"}.get(e.key, e.order_key)})
        for e in entries()
    ]
    verdicts = dedup_responses(flipped)
    # miner-b now precedes miner-a, so b is kept and a is the replay
    assert verdicts["miner-b"].kept
    assert not verdicts["miner-a"].kept
    assert verdicts["miner-a"].duplicate_of == "miner-b"


def test_phash_values_are_non_economic() -> None:
    verdicts = dedup_responses(entries())
    assert not verdicts["miner-b"].kept  # exact match still caught
    assert verdicts["miner-c"].kept  # distance-one pHash stays non-economic
