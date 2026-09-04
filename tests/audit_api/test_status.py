"""GET /audit/status, /audit/feed, /audit/epochs — the aggregate honesty surface.

The status endpoint is the investigation/alerting surface: the aggregate across auditors, not
one opinion. CLEAN only when >=1 report and none disputed; DISPUTED the moment any
auditor reports a FAIL; UNAUDITED when none has reported. Manual remediation only;
status never gates weight-setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))

from audit_api_support import (
    AuditApi,
    earning_skip_item,
    fail_item,
    make_report,
    skip_item,
)

from vidaio.auditor.report import AuditStatus

POST = "/audit/report"


def _body(report) -> dict:
    return report.model_dump(mode="json")


async def _submit(client: httpx.AsyncClient, report) -> None:
    resp = await client.post(POST, json=_body(report))
    assert resp.status_code in (200, 201), resp.text


# -- earning-SKIP INCONCLUSIVE consistency (store/aggregate pass earning_verdicts) --


async def test_earning_skip_persists_and_aggregates_inconclusive(
    client: httpx.AsyncClient,
) -> None:
    """A report whose only anomaly is an UNVERIFIABLE earning re-derivation (SKIP) is
    INCONCLUSIVE — and the store's persisted ``overall`` + the /audit/status aggregate
    must AGREE with the report's own derived verdict, not wash it to CLEAN.

    Regression for the an internal review: ``_recomputed_overall`` /
    ``_effective_verdict`` now pass ``report.earning_verdicts`` as the third channel, so
    the earning-SKIP INCONCLUSIVE the report itself derives is exactly what is stored and
    aggregated (a PASS media item alone would otherwise roll up CLEAN).
    """
    report = make_report(
        auditor_hotkey="hk-earn",
        epoch_id=77,
        item_verdicts=(skip_item(),),  # a media SKIP would already be INCONCLUSIVE...
        earning_verdicts=(earning_skip_item(),),  # ...but the earning SKIP is the point
    )
    assert report.overall is AuditStatus.INCONCLUSIVE  # the report derives it itself
    await _submit(client, report)
    status = (await client.get("/audit/status", params={"epoch_id": 77})).json()
    assert status["verdict"] == "INCONCLUSIVE"
    assert status["disputed"] == 0


async def test_earning_skip_alone_is_inconclusive_not_clean(
    client: httpx.AsyncClient,
) -> None:
    """With every media item PASSing, an earning SKIP must STILL roll up INCONCLUSIVE.

    This is the case the missing third arg silently washed to CLEAN: no media SKIP to
    trip the media-coverage floor, only the earning channel carries the anomaly.
    """
    from audit_api_support import pass_item

    report = make_report(
        auditor_hotkey="hk-earn2",
        epoch_id=78,
        item_verdicts=(pass_item(),),
        earning_verdicts=(earning_skip_item(),),
        inference_n=1,
    )
    assert report.overall is AuditStatus.INCONCLUSIVE
    await _submit(client, report)
    status = (await client.get("/audit/status", params={"epoch_id": 78})).json()
    assert status["verdict"] == "INCONCLUSIVE"
    assert status["disputed"] == 0


# -- verdict aggregation across auditors -------------------------------------------


async def test_unaudited_when_no_reports(client: httpx.AsyncClient) -> None:
    status = (await client.get("/audit/status", params={"epoch_id": 42})).json()
    assert status["verdict"] == "UNAUDITED"
    assert status["auditors_reporting"] == 0
    assert status["clean"] == 0 and status["disputed"] == 0
    assert status["disputed_items"] == []


async def test_clean_when_all_auditors_clean(client: httpx.AsyncClient) -> None:
    for hk in ("hk-a", "hk-b", "hk-c"):
        await _submit(client, make_report(auditor_hotkey=hk, epoch_id=50, snapshot_digest="d" * 64))
    status = (await client.get("/audit/status", params={"epoch_id": 50})).json()
    assert status["verdict"] == "CLEAN"
    assert status["auditors_reporting"] == 3
    assert status["clean"] == 3 and status["disputed"] == 0
    # all three audited the same bytes
    assert status["snapshot_digest"] == "d" * 64


async def test_disputed_when_one_auditor_disputes(client: httpx.AsyncClient) -> None:
    await _submit(client, make_report(auditor_hotkey="hk-a", epoch_id=51))
    await _submit(client, make_report(auditor_hotkey="hk-b", epoch_id=51))
    # one auditor finds a provable fault
    await _submit(
        client,
        make_report(
            auditor_hotkey="hk-c",
            epoch_id=51,
            item_verdicts=(fail_item(code="SCORE_MISMATCH"),),
        ),
    )
    status = (await client.get("/audit/status", params={"epoch_id": 51})).json()
    # a single provable FAIL flips the epoch — an honest majority cannot out-vote it
    assert status["verdict"] == "DISPUTED"
    assert status["auditors_reporting"] == 3
    assert status["clean"] == 2 and status["disputed"] == 1
    assert status["reason_counts"] == {"SCORE_MISMATCH": 1}
    disputed = status["disputed_items"]
    assert len(disputed) == 1
    assert disputed[0]["code"] == "SCORE_MISMATCH"
    assert disputed[0]["auditor_hotkey"] == "hk-c"


async def test_self_reported_clean_with_fail_item_aggregates_disputed(
    client: httpx.AsyncClient,
) -> None:
    """One provable fault ⇒ DISPUTED is RECOMPUTED at aggregation, never trusted.

    A report that self-reports overall=CLEAN while carrying a FAIL item must NOT let
    the epoch show CLEAN — the aggregator recomputes the verdict from item_verdicts.
    """
    tampered = make_report(
        auditor_hotkey="hk-liar",
        epoch_id=55,
        item_verdicts=(fail_item(code="SCORE_MISMATCH"),),
        overall=AuditStatus.CLEAN,  # invalid self-report
    )
    resp = await client.post(POST, json=tampered.model_dump(mode="json"))
    assert resp.status_code == 201  # validly signed, so persisted

    status = (await client.get("/audit/status", params={"epoch_id": 55})).json()
    assert status["verdict"] == "DISPUTED"  # recomputed, not the CLEAN it claimed
    assert status["clean"] == 0 and status["disputed"] == 1
    assert status["disputed_items"][0]["code"] == "SCORE_MISMATCH"


async def test_inconclusive_when_all_reports_inconclusive(
    client: httpx.AsyncClient,
) -> None:
    """Reports exist but every one is all-SKIP => INCONCLUSIVE, distinct from CLEAN (#8).

    Nothing was recomputed for the epoch, so it is neither proven clean nor disputed —
    a needs-attention state, never washed to CLEAN.
    """
    for hk in ("hk-a", "hk-b"):
        await _submit(
            client,
            make_report(auditor_hotkey=hk, epoch_id=56, inference_n=1,
                        item_verdicts=(skip_item(),)),
        )
    status = (await client.get("/audit/status", params={"epoch_id": 56})).json()
    assert status["verdict"] == "INCONCLUSIVE"  # NOT CLEAN, NOT DISPUTED
    assert status["auditors_reporting"] == 2
    assert status["clean"] == 0 and status["disputed"] == 0
    assert status["inconclusive"] == 2


async def test_inconclusive_does_not_bury_a_dispute(client: httpx.AsyncClient) -> None:
    """A DISPUTED report still flips an epoch that also has INCONCLUSIVE reports."""
    await _submit(
        client,
        make_report(auditor_hotkey="hk-a", epoch_id=57, inference_n=1,
                    item_verdicts=(skip_item(),)),
    )
    await _submit(
        client,
        make_report(auditor_hotkey="hk-b", epoch_id=57,
                    item_verdicts=(fail_item(code="SCORE_MISMATCH"),)),
    )
    status = (await client.get("/audit/status", params={"epoch_id": 57})).json()
    assert status["verdict"] == "DISPUTED"  # a provable fault dominates INCONCLUSIVE
    assert status["disputed"] == 1 and status["inconclusive"] == 1


async def test_weight_derivation_dispute_surfaces(client: httpx.AsyncClient) -> None:
    await _submit(client, make_report(auditor_hotkey="hk-w", epoch_id=52, weight_fail=True))
    status = (await client.get("/audit/status", params={"epoch_id": 52})).json()
    assert status["verdict"] == "DISPUTED"
    assert status["reason_counts"] == {"WEIGHT_DERIVATION_MISMATCH": 1}
    assert status["disputed_items"][0]["source"] == "weight"


async def test_reason_counts_union_across_reports(client: httpx.AsyncClient) -> None:
    await _submit(
        client,
        make_report(
            auditor_hotkey="hk-a",
            epoch_id=53,
            item_verdicts=(fail_item("i1", code="SCORE_MISMATCH"),),
        ),
    )
    await _submit(
        client,
        make_report(
            auditor_hotkey="hk-b",
            epoch_id=53,
            item_verdicts=(
                fail_item("i2", code="SCORE_MISMATCH"),
                fail_item("i3", code="MERKLE_EXCLUSION"),
            ),
        ),
    )
    status = (await client.get("/audit/status", params={"epoch_id": 53})).json()
    assert status["reason_counts"] == {"MERKLE_EXCLUSION": 1, "SCORE_MISMATCH": 2}
    assert len(status["disputed_items"]) == 3


async def test_divergent_snapshot_digests_reported(client: httpx.AsyncClient) -> None:
    """Two auditors that audited DIFFERENT bytes for one epoch — surfaced, not hidden."""
    await _submit(client, make_report(auditor_hotkey="hk-a", epoch_id=54, snapshot_digest="a" * 64))
    await _submit(client, make_report(auditor_hotkey="hk-b", epoch_id=54, snapshot_digest="b" * 64))
    status = (await client.get("/audit/status", params={"epoch_id": 54})).json()
    assert status["snapshot_digest"] is None  # no single agreed digest
    assert status["snapshot_digests"] == ["a" * 64, "b" * 64]


# -- feed + epochs -----------------------------------------------------------------


async def test_feed_is_newest_first_with_failures(client: httpx.AsyncClient) -> None:
    await _submit(client, make_report(auditor_hotkey="hk-a", epoch_id=60))
    await _submit(
        client,
        make_report(
            auditor_hotkey="hk-b", epoch_id=61, item_verdicts=(fail_item(code="REVEAL_INVALID"),)
        ),
    )
    feed = (await client.get("/audit/feed", params={"limit": 10})).json()
    assert feed["limit"] == 10
    assert [r["epoch_id"] for r in feed["reports"]] == [61, 60]  # newest received first
    disputed = feed["reports"][0]
    assert disputed["overall"] == "DISPUTED"
    assert disputed["failures"][0]["code"] == "REVEAL_INVALID"
    assert feed["disputed_epochs"] == 1


async def test_feed_limit_is_clamped(client: httpx.AsyncClient) -> None:
    for i in range(4):
        await _submit(client, make_report(auditor_hotkey=f"hk-{i}", epoch_id=70 + i))
    feed = (await client.get("/audit/feed", params={"limit": 2})).json()
    assert len(feed["reports"]) == 2


async def test_epochs_rollup(client: httpx.AsyncClient) -> None:
    await _submit(client, make_report(auditor_hotkey="hk-a", epoch_id=80))
    await _submit(client, make_report(auditor_hotkey="hk-b", epoch_id=80))
    await _submit(
        client,
        make_report(auditor_hotkey="hk-a", epoch_id=81, item_verdicts=(fail_item(),)),
    )
    epochs = (await client.get("/audit/epochs")).json()["epochs"]
    assert [e["epoch_id"] for e in epochs] == [81, 80]  # newest epoch first
    by_id = {e["epoch_id"]: e for e in epochs}
    assert by_id[80]["verdict"] == "CLEAN" and by_id[80]["auditors_reporting"] == 2
    assert by_id[81]["verdict"] == "DISPUTED"


async def test_disputed_epochs_gauge_tracks_disputes(
    api: AuditApi, client: httpx.AsyncClient
) -> None:
    assert api.metric("vidaio_audit_disputed_epochs") == 0.0
    await _submit(client, make_report(auditor_hotkey="hk-a", epoch_id=90))
    assert api.metric("vidaio_audit_disputed_epochs") == 0.0
    await _submit(
        client, make_report(auditor_hotkey="hk-b", epoch_id=91, item_verdicts=(fail_item(),))
    )
    assert api.metric("vidaio_audit_disputed_epochs") == 1.0
