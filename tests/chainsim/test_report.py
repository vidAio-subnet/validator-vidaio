"""Report content: ranked vector table, history deltas, decoded anchors, files."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from chainsim_support import bearer, register

from vidaio.chainsim.report import (
    build_report,
    decode_anchor_payload,
    render_markdown,
    write_report,
)
from vidaio.chainsim.service import AnchorRequest, RegisterRequest, WeightsRequest


@pytest.fixture
def populated(make_sim):
    """A sim with two vectors from one validator, three neurons, one anchor."""
    sim = make_sim(tempo=2)
    sim.register(RegisterRequest(hotkey="val", role="validator", alpha_stake=1000.0))
    sim.register(RegisterRequest(hotkey="hk1", role="miner"))
    sim.register(RegisterRequest(hotkey="hk2", role="miner"))
    assert sim.submit_weights(WeightsRequest(hotkey="val", vector={1: 0.25, 2: 0.75}))[
        "success"
    ]
    sim.advance(4)
    assert sim.submit_weights(WeightsRequest(hotkey="val", vector={1: 0.6, 2: 0.4}))[
        "success"
    ]
    payload = b"vidaio.commitment.v1:publication:" + b"ab" * 32
    sim.anchor(AnchorRequest(payload_hex=payload.hex(), hotkey="val"))
    return sim


def test_report_json_contents(populated):
    report = build_report(populated.state())

    assert report["kind"] == "vidaio.chainsim.report.v1"
    assert [(n["uid"], n["role"]) for n in report["neurons"]] == [
        (0, "validator"),
        (1, "miner"),
        (2, "miner"),
    ]

    # latest vector, ranked with hotkeys and share %
    latest = report["latest_vector"]
    assert latest["set_by"] == "val" and latest["block"] == 5
    assert latest["ranked"] == [
        {"rank": 1, "uid": 1, "hotkey": "hk1", "weight": 0.6, "share_pct": 60.0},
        {"rank": 2, "uid": 2, "hotkey": "hk2", "weight": 0.4, "share_pct": 40.0},
    ]

    # per-validator history with blocks and per-uid deltas
    history = report["weight_history"]["val"]
    assert [entry["block"] for entry in history] == [1, 5]
    assert history[0]["delta"] == {"1": 0.25, "2": 0.75}
    assert history[1]["delta"] == {"1": pytest.approx(0.35), "2": pytest.approx(-0.35)}

    # anchors carry the decoded domain-tagged payload
    anchor = report["anchors"][0]
    assert anchor["decoded"] == {
        "domain": "vidaio.commitment.v1",
        "kind": "publication",
        "root": "ab" * 32,
    }

    # emission credited per uid is present (blocks 2..5 under the 1:3 vector)
    by_uid = {n["uid"]: n["emission_credited"] for n in report["neurons"]}
    assert by_uid[1] == pytest.approx(1.0) and by_uid[2] == pytest.approx(3.0)


def test_markdown_report_golden_fragments(populated):
    report = build_report(populated.state())
    md = render_markdown(report)

    assert "# Chain-sim run report" in md
    assert "## Latest weight vector" in md
    # the ranked table rows, hotkey + share included
    assert "| rank | uid | hotkey | weight | share % |" in md
    assert "| 1 | 1 | hk1 | 0.600000 | 60.00% |" in md
    assert "| 2 | 2 | hk2 | 0.400000 | 40.00% |" in md
    # anchor row with txid + decoded kind/root
    txid = report["anchors"][0]["txid"]
    assert f"| {txid} | 5 | vidaio.commitment.v1 | publication | {'ab' * 32} |" in md
    # history and emission sections exist
    assert "### Validator `val`" in md
    assert "## Emission" in md


def test_write_report_persists_timestamped_json_and_md(populated, tmp_path):
    stamp = datetime(2026, 8, 20, 15, 30, 0, tzinfo=timezone.utc)
    json_path, md_path = write_report(populated.state(), tmp_path / "out", now=stamp)

    assert json_path.name == "report-20260820T153000Z.json"
    assert md_path.name == "report-20260820T153000Z.md"
    on_disk = json.loads(json_path.read_text())
    assert on_disk["latest_vector"]["ranked"][0]["uid"] == 1
    assert md_path.read_text() == render_markdown(build_report(populated.state()))


async def test_report_endpoints(sim, client):
    _uid, token = await register(client, "val", role="validator")
    await client.post(
        "/weights",
        json={"hotkey": "val", "vector": {"0": 1.0}, "version_key": 0},
        headers=bearer(token),
    )
    body = (await client.get("/report")).json()  # reads stay open
    assert body == build_report(sim.state())

    # /report/write puts files on disk: any registered identity may, nobody else
    assert (await client.post("/report/write")).status_code == 401
    assert (
        await client.post("/report/write", headers=bearer("not-a-token"))
    ).status_code == 403
    written = (await client.post("/report/write", headers=bearer(token))).json()
    report_on_disk = json.loads(open(written["json_path"]).read())
    assert report_on_disk["latest_vector"]["ranked"][0]["uid"] == 0
    assert "## Latest weight vector" in open(written["md_path"]).read()


def test_decode_anchor_payload_shapes():
    tagged = ("vidaio.commitment.v1:publication:" + "cd" * 32).encode().hex()
    assert decode_anchor_payload(tagged) == {
        "domain": "vidaio.commitment.v1",
        "kind": "publication",
        "root": "cd" * 32,
    }
    assert decode_anchor_payload(b"hello world".hex()) == {"text": "hello world"}
    assert decode_anchor_payload(b"\x00\x01\xff".hex()) is None
    assert decode_anchor_payload("not-hex") is None
