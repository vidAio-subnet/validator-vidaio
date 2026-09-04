from __future__ import annotations

import random

import pytest

from vidaio.tokenomics import (
    EXCLUDED_SCORE,
    eligible_for_ranking,
    inference_shares,
    track_shares,
)


class TestEligibility:
    def test_excluded_flag_and_sentinel_and_zero_score_removed(self, mk_miner) -> None:
        miners = [
            mk_miner(1, score=0.9, excluded=True),
            mk_miner(2, score=EXCLUDED_SCORE),
            mk_miner(3, score=0.0),
            mk_miner(4, score=0.4),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [4]

    def test_ip_dedup_lowest_uid_wins(self, mk_miner) -> None:
        miners = [
            mk_miner(7, score=0.9, ip="1.1.1.1"),
            mk_miner(3, score=0.1, ip="1.1.1.1"),
            mk_miner(5, score=0.5),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [3, 5]

    def test_unspecified_ipv4_does_not_collapse_non_serving_miners(self, mk_miner) -> None:
        miners = [
            mk_miner(2, ip="0.0.0.0", coldkey="ck-2"),
            mk_miner(1, ip="0.0.0.0", coldkey="ck-1"),
            mk_miner(3, ip="0.0.0.0", coldkey="ck-3"),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [1, 2, 3]

    def test_unspecified_ipv6_spellings_do_not_collide(self, mk_miner) -> None:
        miners = [
            mk_miner(1, ip="::", coldkey="ck-1"),
            mk_miner(2, ip="0:0:0:0:0:0:0:0", coldkey="ck-2"),
            mk_miner(3, ip="::ffff:0.0.0.0", coldkey="ck-3"),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [1, 2, 3]

    def test_unspecified_ip_still_enforces_coldkey_dedup(self, mk_miner) -> None:
        miners = [
            mk_miner(1, ip="0.0.0.0", coldkey="shared"),
            mk_miner(2, ip="0.0.0.0", coldkey="shared"),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [1]

    def test_equivalent_ipv6_spellings_share_one_dedup_slot(self, mk_miner) -> None:
        miners = [
            mk_miner(1, ip="2001:db8::1", coldkey="ck-1"),
            mk_miner(2, ip="2001:0db8:0:0:0:0:0:1", coldkey="ck-2"),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [1]

    def test_ipv4_mapped_ipv6_shares_ipv4_dedup_slot(self, mk_miner) -> None:
        miners = [
            mk_miner(1, ip="192.0.2.8", coldkey="ck-1"),
            mk_miner(2, ip="::ffff:192.0.2.8", coldkey="ck-2"),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [1]

    def test_coldkey_dedup_lowest_uid_wins(self, mk_miner) -> None:
        miners = [
            mk_miner(9, score=0.9, coldkey="ckX"),
            mk_miner(2, score=0.2, coldkey="ckX"),
        ]
        assert [m.uid for m in eligible_for_ranking(miners)] == [2]

    def test_dedup_is_order_independent(self, mk_miner) -> None:
        miners = [
            mk_miner(1, ip="a", coldkey="c1"),
            mk_miner(2, ip="a", coldkey="c2"),
            mk_miner(3, ip="b", coldkey="c2"),
            mk_miner(4, ip="b", coldkey="c4"),
        ]
        expected = [m.uid for m in eligible_for_ranking(miners)]
        rng = random.Random(85)
        for _ in range(20):
            shuffled = miners[:]
            rng.shuffle(shuffled)
            assert [m.uid for m in eligible_for_ranking(shuffled)] == expected

    def test_ineligible_miner_does_not_shadow_a_dedup_slot(self, mk_miner) -> None:
        # uid 1 shares the IP but has no genuine score — uid 2 keeps the slot.
        miners = [mk_miner(1, score=0.0, ip="a"), mk_miner(2, score=0.9, ip="a")]
        assert [m.uid for m in eligible_for_ranking(miners)] == [2]

    def test_absolute_payout_floor_is_inclusive_and_precedes_dedup(
        self, mk_miner
    ) -> None:
        miners = [
            mk_miner(1, score=0.099999, ip="shared"),
            mk_miner(2, score=0.10, ip="shared"),
        ]
        eligible = eligible_for_ranking(miners, minimum_payout_score=0.10)
        assert [m.uid for m in eligible] == [2]


class TestTrackShares:
    def test_top_n_graded_and_rank_six_zero(self, cfg, mk_miner) -> None:
        miners = [mk_miner(u, score=1.0 - u / 100) for u in range(1, 8)]
        shares = track_shares(cfg, miners)
        assert set(shares) == {1, 2, 3, 4, 5}
        assert shares == pytest.approx(
            {1: 5 / 15, 2: 4 / 15, 3: 3 / 15, 4: 2 / 15, 5: 1 / 15}
        )
        assert shares[1] == pytest.approx(5 * shares[5])
        assert 6 not in shares and 7 not in shares

    def test_score_ties_break_by_uid(self, cfg, mk_miner) -> None:
        miners = [mk_miner(u, score=0.5) for u in (9, 8, 7, 6, 5, 4)]
        shares = track_shares(cfg, miners)
        assert set(shares) == {4, 5, 6, 7, 8, 9} - {9}

    def test_fewer_than_n_renormalizes(self, cfg, mk_miner) -> None:
        shares = track_shares(cfg, [mk_miner(1), mk_miner(2)])
        assert shares[1] == pytest.approx(5 / 9)
        assert shares[2] == pytest.approx(4 / 9)

    # (test_retention_reshapes_owner_vs_sellers REMOVED with the retention multiplier for v1
    # — retention removed — owner decision; an internal review — rank
    # weight no longer depends on any windowed retention input.)

    def test_empty_track(self, cfg) -> None:
        assert track_shares(cfg, []) == {}


class TestInferenceShares:
    def test_cross_track_split(self, cfg, mk_miner) -> None:
        miners = [
            mk_miner(1, track="compression"),
            mk_miner(2, track="upscaling"),
        ]
        shares = inference_shares(cfg, miners)
        assert shares[1] == pytest.approx(0.8)
        assert shares[2] == pytest.approx(0.2)
        assert sum(shares.values()) == pytest.approx(1.0)

    def test_empty_track_weight_is_not_reallocated(self, cfg, mk_miner) -> None:
        shares = inference_shares(cfg, [mk_miner(1), mk_miner(2)])
        assert sum(shares.values()) == pytest.approx(0.8)
        assert shares[1] == pytest.approx(0.8 * 5 / 9)
        assert shares[2] == pytest.approx(0.8 * 4 / 9)

    def test_unknown_track_takes_nothing(self, cfg, mk_miner) -> None:
        shares = inference_shares(cfg, [mk_miner(1), mk_miner(2, track="karaoke")])
        assert 2 not in shares
        assert shares[1] == pytest.approx(0.8)

    def test_below_floor_track_receives_nothing(self, cfg, mk_miner) -> None:
        shares = inference_shares(
            cfg,
            [
                mk_miner(1, score=0.9),
                mk_miner(2, score=0.099, track="upscaling"),
            ],
        )
        assert shares == {1: pytest.approx(0.8)}

    def test_no_eligible_miners(self, cfg, mk_miner) -> None:
        assert inference_shares(cfg, [mk_miner(1, excluded=True)]) == {}
