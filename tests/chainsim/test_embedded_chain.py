"""EmbeddedReportingChain: InMemoryChain semantics + a JSONL paper trail."""

from __future__ import annotations

import hashlib
import json

from vidaio.chain import (
    ChainAdapter,
    EmbeddedReportingChain,
    InMemoryChain,
    SubmittedWeightsReader,
)


def journal_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_embedded_chain_satisfies_the_protocol(tmp_path):
    chain = EmbeddedReportingChain(journal_path=tmp_path / "journal.jsonl")
    assert isinstance(chain, ChainAdapter)


async def test_the_in_process_adapters_can_read_their_own_vector_back(tmp_path):
    """Both fakes implement the optional weight read — otherwise a weight-setter
    running against them could never publish anything."""
    for chain in (
        InMemoryChain(),
        EmbeddedReportingChain(journal_path=tmp_path / "journal.jsonl"),
    ):
        assert isinstance(chain, SubmittedWeightsReader)
        assert chain.submitted_weights("anybody") is None  # positive "none yet"

        await chain.set_weights({1: 0.75, 2: 0.25}, version_key=3)

        reported = chain.submitted_weights("anybody")
        assert reported is not None
        assert reported.weights == {1: 0.75, 2: 0.25}
        assert reported.block == 1


async def test_the_in_process_adapters_report_the_submitted_u16(tmp_path):
    """Round-4 #3: a successful set_weights reports the EXACT u16 that lands on the
    chain's grid — never the pre-quantization float intent — so the weight-setter
    publishes/anchors chain state byte-for-byte even when the two are scale-equivalent."""
    from vidaio.tokenomics.quantize import max_normalize_u16, quantize_u16

    intent = {1: 0.4, 2: 0.6}
    for chain in (
        InMemoryChain(),
        EmbeddedReportingChain(journal_path=tmp_path / "journal.jsonl"),
    ):
        result = await chain.set_weights(intent, version_key=3)
        assert result.success
        assert result.submitted == max_normalize_u16(quantize_u16(intent))
        assert result.submitted != intent  # the u16 grid, NOT the float intent
        assert max(result.submitted.values()) == 65535
        # the fake still keeps the raw floats in its own read-back store
        assert chain.weight_calls[-1][1] == intent


async def test_every_weight_call_and_anchor_is_journaled(tmp_path):
    path = tmp_path / "reports" / "journal.jsonl"  # parent dir is created on demand
    chain = EmbeddedReportingChain(tempo=10, journal_path=path)

    accepted = await chain.set_weights({1: 0.75, 2: 0.25}, version_key=3)
    rejected = await chain.set_weights({1: 1.0}, version_key=3)  # tempo-gated
    payload = b"vidaio.commitment.v1:publication:" + b"e" * 64
    txid = await chain.anchor_commitment(payload)

    assert accepted.success and not rejected.success
    # InMemoryChain semantics intact: only the accepted call is in weight_calls
    assert chain.weight_calls == [(1, {1: 0.75, 2: 0.25})]
    assert chain.anchored == [payload]
    assert txid == "0x" + hashlib.sha256(payload).hexdigest()[:16]

    # the journal shows what was ATTEMPTED — accepted, rejected, and the anchor
    entries = journal_lines(path)
    assert [e["kind"] for e in entries] == ["set_weights", "set_weights", "anchor"]
    assert entries[0] == {
        "kind": "set_weights",
        "block": 1,
        "success": True,
        "message": "",
        "version_key": 3,
        "weights": {"1": 0.75, "2": 0.25},
    }
    assert entries[1]["success"] is False and "tempo" in entries[1]["message"]
    assert entries[2] == {
        "kind": "anchor",
        "block": 1,
        "txid": txid,
        "payload_hex": payload.hex(),
    }


async def test_journal_appends_across_instances(tmp_path):
    """Restart-safe paper trail: a new chain over the same file appends, not truncates."""
    path = tmp_path / "journal.jsonl"
    first = EmbeddedReportingChain(journal_path=path)
    await first.set_weights({0: 1.0}, version_key=0)

    second = EmbeddedReportingChain(journal_path=path)
    await second.set_weights({0: 0.5}, version_key=1)

    assert [e["version_key"] for e in journal_lines(path)] == [0, 1]
