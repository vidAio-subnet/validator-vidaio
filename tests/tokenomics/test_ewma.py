from __future__ import annotations

import pytest

from vidaio.tokenomics import EXCLUDED_SCORE, TokenomicsConfig, accumulate, is_excluded


def test_basic_accumulation_with_config_decay(cfg: TokenomicsConfig) -> None:
    assert cfg.ewma_decay == 0.75
    assert accumulate(0.8, 0.4, cfg.ewma_decay) == pytest.approx(0.75 * 0.8 + 0.25 * 0.4)


def test_accumulation_from_zero(cfg: TokenomicsConfig) -> None:
    assert accumulate(0.0, 1.0, cfg.ewma_decay) == pytest.approx(0.25)


def test_sentinel_score_excludes(cfg: TokenomicsConfig) -> None:
    assert accumulate(0.9, EXCLUDED_SCORE, cfg.ewma_decay) == EXCLUDED_SCORE
    assert is_excluded(accumulate(0.9, -1.0, cfg.ewma_decay))


def test_sentinel_latches_until_genuine_score(cfg: TokenomicsConfig) -> None:
    state = accumulate(0.9, EXCLUDED_SCORE, cfg.ewma_decay)
    state = accumulate(state, EXCLUDED_SCORE, cfg.ewma_decay)
    assert state == EXCLUDED_SCORE


def test_reentry_after_exclusion_starts_from_zero(cfg: TokenomicsConfig) -> None:
    state = accumulate(0.9, EXCLUDED_SCORE, cfg.ewma_decay)
    # Excluded history is not carried back in — re-entry is (1 - decay) * score.
    assert accumulate(state, 0.8, cfg.ewma_decay) == pytest.approx(0.25 * 0.8)


def test_invalid_decay_rejected() -> None:
    with pytest.raises(ValueError):
        accumulate(0.5, 0.5, 0.0)
    with pytest.raises(ValueError):
        accumulate(0.5, 0.5, 1.0)


def test_negative_non_sentinel_score_rejected(cfg: TokenomicsConfig) -> None:
    with pytest.raises(ValueError):
        accumulate(0.5, -0.5, cfg.ewma_decay)


def test_is_excluded_only_matches_sentinel() -> None:
    assert is_excluded(-1.0)
    assert not is_excluded(0.0)
    assert not is_excluded(0.99)
