"""Pre-commitment anchoring gates enrollment: SCHEDULED -> ENROLLING
requires an anchored commitment_root, and the anchored event precedes the transition."""

import logging

import pytest

from vidaio.competition import IllegalTransition, Phase
from vidaio.competition import repository as repo

from support import COMMITMENT_ROOT, START, T0, Driver, build_manifest


def test_tick_without_commitment_stays_scheduled_with_reason(
    driver: Driver, caplog: pytest.LogCaptureFixture
) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    with caplog.at_level(logging.INFO, logger="vidaio.competition.engine"):
        assert driver.engine.tick(driver.conn, START) == []
    assert driver.phase(cid) is Phase.SCHEDULED
    assert any("pre-commitment not anchored" in r.message for r in caplog.records)


def test_tick_with_commitment_opens_enrollment(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    assert driver.anchor(cid) is True
    comp = repo.get_competition(driver.conn, cid)
    assert comp is not None and comp.commitment_root == COMMITMENT_ROOT

    assert driver.engine.tick(driver.conn, START) == [(cid, Phase.SCHEDULED, Phase.ENROLLING)]
    assert driver.phase(cid) is Phase.ENROLLING

    # Event ordering: the anchored event strictly precedes the enrolling transition.
    types = [e["event_type"] for e in driver.events(cid)]
    assert types.index("commitment_anchored") < types.index("transition")
    anchored = next(e for e in driver.events(cid) if e["event_type"] == "commitment_anchored")
    assert COMMITMENT_ROOT in (anchored["payload_json"] or "")


def test_anchor_is_idempotent_but_never_rebindable(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    assert driver.anchor(cid) is True
    assert driver.anchor(cid) is False  # same root: no-op, no second event
    assert (
        sum(1 for e in driver.events(cid) if e["event_type"] == "commitment_anchored") == 1
    )
    with pytest.raises(ValueError, match="already anchored"):
        driver.engine.mark_commitment_anchored(driver.conn, cid, "d" * 64, T0)


@pytest.mark.parametrize("bad_root", ["", "abc", "C" * 64, "g" * 64, "a" * 63])
def test_malformed_root_rejected(driver: Driver, bad_root: str) -> None:
    manifest = build_manifest()
    driver.engine.create_competition(driver.conn, manifest, T0)
    with pytest.raises(ValueError, match="sha256"):
        driver.engine.mark_commitment_anchored(
            driver.conn, manifest.competition_id, bad_root, T0
        )


def test_anchor_only_while_scheduled(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    driver.engine.fail(driver.conn, cid, T0, "aborted before anchoring")
    with pytest.raises(IllegalTransition, match="SCHEDULED"):
        driver.anchor(cid)


def test_direct_apply_backstop_requires_commitment(driver: Driver) -> None:
    manifest = build_manifest()
    cid = manifest.competition_id
    driver.engine.create_competition(driver.conn, manifest, T0)
    with pytest.raises(IllegalTransition, match="not anchored"):
        driver.engine._apply(driver.conn, cid, Phase.ENROLLING, START)
    assert driver.phase(cid) is Phase.SCHEDULED
