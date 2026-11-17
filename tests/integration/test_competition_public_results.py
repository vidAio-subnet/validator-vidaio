"""The public results table of a COMPLETED competition matches the economic result the
epoch evidence derives (speed term and failed-item limit included), with per-item
scores and failure codes."""

from __future__ import annotations

from pathlib import Path

import pytest

import tests.integration.test_upscaling_generative_competition as generative
from vidaio.competition.economic_result import speed_adjusted_score, speed_score
from vidaio.competition.orchestrator import public_results
from vidaio.competition.orchestrator.results import ResultNotReady


def _run_capturing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs):
    """Run the generative-upscaling fixture and keep the database and store it built."""
    captured: dict[str, object] = {}
    real = generative.build_competition_epoch_evidence

    def capture(conn, **evidence_kwargs):
        captured["conn"] = conn
        captured["store"] = evidence_kwargs["store"]
        return real(conn, **evidence_kwargs)

    monkeypatch.setattr(generative, "build_competition_epoch_evidence", capture)
    evidence, log, expected = generative._run(tmp_path, **kwargs)
    conn, store = captured["conn"], captured["store"]
    plan = public_results.collect(conn, generative.COMPETITION_ID)
    table = public_results.assemble(
        plan, public_results.read_failure_codes(store, plan.packets_to_read)
    )
    return evidence, expected, table


def test_results_match_the_speed_adjusted_economic_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evidence, expected, table = _run_capturing(
        monkeypatch, tmp_path, items=3, rules=generative.SPEED_RULES,
        resources=generative.RESOURCES, batch_walls=(200.0, 250.0, 150.0),
    )
    (row,) = table["results"]
    (paid,) = evidence.result.contenders
    subject = evidence.competition_input.subjects[0]
    assert row["hotkey"] == paid.hotkey == generative.HOTKEY
    assert row["rank"] == 1
    assert row["final_score"] == pytest.approx(paid.score, abs=1e-12)
    assert row["run_seconds"] == subject.run_seconds == 600.0
    quality = sum(p.score for p, _, _ in expected.values()) / 3
    speed = speed_score(600.0, inserted_frames=120, fast_spf=0.5, slow_spf=20.0)
    assert row["quality"] == pytest.approx(quality)
    assert row["speed"] == pytest.approx(speed)
    assert row["final_score"] == pytest.approx(speed_adjusted_score(quality, speed, weight=0.3))
    assert row["final_score"] != pytest.approx(quality)  # not the review score
    assert "items" not in row  # no per-clip scores in public (owner rule 2026-11-05)
    assert row["failed_items"] == 0 and row["failure_codes"] == {}
    assert table["results_baseline"] is None
    assert table["results_speed_term"] == {
        "weight": 0.3, "fast_spf": 0.5, "slow_spf": 20.0, "inserted_frames": 120,
    }


def test_results_apply_the_failed_item_limit_and_name_the_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rules = {**generative.RULES, "max_failed_item_fraction": 0.10}
    evidence, expected, table = _run_capturing(
        monkeypatch, tmp_path, items=10, failed=2, rules=rules, full=False
    )
    assert evidence is None  # nobody reached the podium bar: no economic result to apply
    (row,) = table["results"]
    mean = sum(p.score for p, _, _ in expected.values()) / 10
    assert row["quality"] == pytest.approx(mean)
    assert row["failed_fraction"] == pytest.approx(0.2) and row["failed_items"] == 2
    assert row["final_score"] == 0.0  # ranked with 0 above the anchored limit
    assert row["failure_codes"] == {"LOWRES_INCONSISTENT": 2}  # counts, not which items
    assert "items" not in row


def test_results_within_the_limit_equal_the_paid_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rules = {**generative.RULES, "max_failed_item_fraction": 0.10}
    evidence, _expected, table = _run_capturing(
        monkeypatch, tmp_path, items=10, failed=1, rules=rules, full=False
    )
    (row,) = table["results"]
    assert row["final_score"] == pytest.approx(evidence.result.contenders[0].score, abs=1e-12)
    assert row["failed_fraction"] == pytest.approx(0.1)


def test_no_results_before_completion(tmp_path: Path) -> None:
    from vidaio.competition import CompetitionManifest, migrate
    from vidaio.competition import repository as repo
    from vidaio.core import connect

    conn = connect(":memory:")
    migrate(conn)
    manifest = CompetitionManifest.model_validate(
        generative._manifest_data([generative._commitment(0, "2" * 64, "1" * 64)])
    )
    repo.insert_competition(conn, manifest, generative.T0)
    with pytest.raises(ResultNotReady):
        public_results.collect(conn, generative.COMPETITION_ID)
