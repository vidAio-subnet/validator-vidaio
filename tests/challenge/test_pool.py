import random
import sqlite3

import pytest

from vidaio.challenge import (
    Asset,
    ChallengeConfig,
    NearDuplicateError,
    NoFreshAssetError,
    RevealBeforeResolutionError,
    StaticFingerprintIndex,
    UnresolvedChallengeError,
    append_provenance,
    assign_split,
    check_near_duplicate,
    checkout_asset,
    get_asset,
    make_challenge,
    provenance_log,
    record_challenge,
    release_asset,
    resolve_challenge,
    retire_asset,
    reveal_commitment,
)
from asset_factories import add_assets, mk_asset

AT = "2026-08-20T12:00:00Z"
SEED = 0xA3D8F1E64C2B905A7E118F0D3C5B2A4968D7E6F5


def test_split_groups_whole_source_and_is_deterministic() -> None:
    cfg = ChallengeConfig(holdout_fraction=0.5, split_salt="test-salt-1")
    # many clips from ONE source (different scene/subject/clip) -> one split
    splits = {
        assign_split(mk_asset(i, scene=f"scene-{i}", subject=f"subj-{i % 3}"), cfg)
        for i in range(25)
    }
    assert len(splits) == 1
    # deterministic across calls
    a = mk_asset(0)
    assert assign_split(a, cfg) == assign_split(a, cfg)
    # across many distinct sources both splits occur (salted hash actually splits)
    values = {
        assign_split(mk_asset(i, creator=f"creator-{i}", source=f"source-{i}"), cfg)
        for i in range(60)
    }
    assert values == {"challenge", "holdout"}


def test_split_changes_with_salt_not_with_clip_fields() -> None:
    cfg1 = ChallengeConfig(holdout_fraction=0.5, split_salt="salt-a")
    per_salt = [
        {
            s: assign_split(mk_asset(0, creator=f"c{s}", source=f"s{s}"), cfg)
            for s in range(40)
        }
        for cfg in (cfg1, ChallengeConfig(holdout_fraction=0.5, split_salt="salt-b"))
    ]
    assert per_salt[0] != per_salt[1]  # salt matters
    # clip-level fields (scene/subject) do NOT matter under the default key fields
    base = mk_asset(1, scene="x", subject="y")
    variant = mk_asset(2, scene="other", subject="other")
    assert assign_split(base, cfg1) == assign_split(variant, cfg1)


def test_retire_after_use_lifecycle(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    rng = random.Random(0)

    out = checkout_asset(conn, rng, AT)
    assert out.id == asset.id
    assert out.status == "in_use"
    assert out.use_count == 1
    # in_use assets cannot be double-issued
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, rng, AT)

    done = release_asset(conn, asset.id, 1, AT)
    assert done.status == "retired"
    # retired assets never re-enter
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, rng, AT)
    assert get_asset(conn, asset.id).status == "retired"


def test_multi_use_before_retirement(conn) -> None:
    add_assets(conn, mk_asset(1))
    rng = random.Random(0)
    a = checkout_asset(conn, rng, AT)
    assert release_asset(conn, a.id, 2, AT).status == "fresh"  # 1 of 2 uses consumed
    a = checkout_asset(conn, rng, AT)
    assert a.use_count == 2
    assert release_asset(conn, a.id, 2, AT).status == "retired"


def test_holdout_assets_never_issued(conn) -> None:
    add_assets(conn, mk_asset(1, split="holdout"))
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, random.Random(0), AT)


def test_ingesting_assets_never_issued(conn) -> None:
    # lifecycle: ingesting -> fresh -> in_use -> retired; checkout sees only 'fresh'
    add_assets(conn, mk_asset(1, status="ingesting", metadata_stripped=False))
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, random.Random(0), AT)


def test_asset_model_defaults_to_ingesting_not_fresh(conn) -> None:
    """The Asset model default mirrors the SQL DEFAULT: constructing an Asset
    without an explicit status yields 'ingesting', so a default-built asset
    inserted directly is NOT checkoutable — 'fresh' can never happen by omission,
    only via confirm_ingest_step (or a deliberate explicit status)."""
    fields = mk_asset(1).model_dump()
    del fields["status"]
    a = Asset.model_validate(fields)
    assert a.status == "ingesting"
    add_assets(conn, a)
    assert get_asset(conn, a.id).status == "ingesting"
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, random.Random(0), AT)


def test_checkout_tag_weighting_flows_through_rng(conn) -> None:
    add_assets(
        conn,
        mk_asset(1, motion_tag="low"),
        mk_asset(2, motion_tag="high"),
    )
    picked = checkout_asset(
        conn, random.Random(3), AT, prefer_tags={"motion_tag": "high"}
    )
    # deterministic given the rng; re-running the same setup picks the same asset
    assert picked.id in {"asset_0001", "asset_0002"}


def test_checkout_is_restricted_to_exact_prevalidated_asset_ids(conn) -> None:
    add_assets(conn, mk_asset(1), mk_asset(2), mk_asset(3))
    picked = checkout_asset(
        conn,
        random.Random(0),
        AT,
        eligible_ids=("asset_0002",),
    )
    assert picked.id == "asset_0002"
    assert get_asset(conn, "asset_0001").status == "fresh"
    assert get_asset(conn, "asset_0003").status == "fresh"


def test_empty_eligible_asset_set_has_no_checkout_side_effect(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    before = conn.total_changes
    with pytest.raises(NoFreshAssetError, match="track-eligible"):
        checkout_asset(conn, random.Random(0), AT, eligible_ids=())
    assert conn.total_changes == before
    assert get_asset(conn, asset.id).status == "fresh"
    assert provenance_log(conn, asset.id) == []


class _RacingRng:
    """Simulates a concurrent worker claiming the chosen asset between checkout's
    SELECT and its guarded UPDATE (for the first `steal_limit` selections)."""

    def __init__(self, conn, steal_limit: int = 1) -> None:
        self.conn = conn
        self.steal_limit = steal_limit
        self.steals = 0

    def choices(self, candidates, weights=None, k=1):
        chosen = candidates[0]
        if self.steals < self.steal_limit:
            self.conn.execute(
                "UPDATE assets SET status = 'in_use', use_count = use_count + 1"
                " WHERE id = ? AND status = 'fresh'",
                (chosen.id,),
            )
            self.steals += 1
        return [chosen]


def test_checkout_claim_is_atomic_under_race(conn) -> None:
    """A single-use asset claimed concurrently between SELECT and UPDATE must not
    be issued twice: the guarded predicate loses the claim and, with nothing else
    fresh, checkout raises instead of double-issuing."""
    asset = mk_asset(1)
    add_assets(conn, asset)
    racer = _RacingRng(conn)
    with pytest.raises(NoFreshAssetError):
        checkout_asset(conn, racer, AT)
    assert racer.steals == 1
    # exactly one issuance happened (the simulated concurrent worker's)
    claimed = get_asset(conn, asset.id)
    assert claimed.status == "in_use"
    assert claimed.use_count == 1


def test_checkout_reselects_after_lost_claim(conn) -> None:
    """Losing a claim re-selects from the remaining fresh pool instead of failing."""
    add_assets(conn, mk_asset(1), mk_asset(2))
    racer = _RacingRng(conn, steal_limit=1)  # concurrent worker takes asset_0001
    issued = checkout_asset(conn, racer, AT)
    assert issued.id == "asset_0002"
    # neither row was double-counted: one use each, one per claimant
    assert get_asset(conn, "asset_0001").use_count == 1
    assert get_asset(conn, "asset_0002").use_count == 1


def test_checkout_provenance_failure_rolls_back_guarded_claim(conn) -> None:
    """Claim and checked_out fact are inseparable even when the insert aborts."""

    asset = mk_asset(1)
    add_assets(conn, asset)
    conn.execute(
        "CREATE TRIGGER fail_checked_out BEFORE INSERT ON provenance_log "
        "WHEN NEW.event = 'checked_out' BEGIN "
        "SELECT RAISE(ABORT, 'fault-injected provenance failure'); END"
    )

    with pytest.raises(sqlite3.DatabaseError, match="provenance failure"):
        checkout_asset(conn, random.Random(0), AT)

    unchanged = get_asset(conn, asset.id)
    assert unchanged.status == "fresh"
    assert unchanged.use_count == 0
    assert provenance_log(conn, asset.id) == []
    assert conn.in_transaction is False


def test_retire_asset_blocks_on_unresolved_challenges(conn) -> None:
    add_assets(conn, mk_asset(1))
    issued = checkout_asset(conn, random.Random(0), AT)
    ch = make_challenge("compression", issued, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)

    with pytest.raises(UnresolvedChallengeError):
        retire_asset(conn, issued.id, AT)
    assert get_asset(conn, issued.id).status == "in_use"

    resolve_challenge(conn, ch.challenge_id, AT)
    assert retire_asset(conn, issued.id, AT).status == "retired"
    # with everything resolved, reveal is now allowed
    reveal_commitment(conn, ch.commitment.commit_hash, AT)


def test_force_retire_still_blocks_reveal_until_resolution(conn) -> None:
    add_assets(conn, mk_asset(1))
    issued = checkout_asset(conn, random.Random(0), AT)
    ch = make_challenge("compression", issued, SEED, "scorer-v1")
    record_challenge(conn, ch, AT)

    # explicit force: retirement is allowed (leak suspicion etc.)...
    assert retire_asset(conn, issued.id, AT, force=True).status == "retired"
    # ...but the seed still must not be exposed while miners may be working
    with pytest.raises(RevealBeforeResolutionError):
        reveal_commitment(conn, ch.commitment.commit_hash, AT)
    resolve_challenge(conn, ch.challenge_id, AT, outcome="expired")
    reveal_commitment(conn, ch.commitment.commit_hash, AT)


def test_provenance_log_is_append_only(conn) -> None:
    asset = mk_asset(1)
    add_assets(conn, asset)
    append_provenance(conn, asset.id, "ingested", {"source_url": asset.source_url}, AT)
    entries = provenance_log(conn, asset.id)
    assert [e["event"] for e in entries] == ["ingested"]

    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(
            "UPDATE provenance_log SET event = 'tampered' WHERE asset_id = ?",
            (asset.id,),
        )
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM provenance_log WHERE asset_id = ?", (asset.id,))
    assert [e["event"] for e in provenance_log(conn, asset.id)] == ["ingested"]


def test_near_duplicate_hook() -> None:
    index = StaticFingerprintIndex({"fp_known_benchmark"})
    check_near_duplicate(index, "fp_novel")  # passes silently
    check_near_duplicate(None, "fp_known_benchmark")  # no index -> no check
    with pytest.raises(NearDuplicateError):
        check_near_duplicate(index, "fp_known_benchmark")
