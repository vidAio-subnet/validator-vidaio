"""Independent finalized-chain/miner-signature chronology proofs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tests.auditor.fakes import make_fake_bundle
from vidaio.audit import (
    ArtifactKind,
    LifecycleStage,
    LocalFsStore,
    build_bundle,
    sha256_hex,
)
from vidaio.auditor.chronology import ChronologyKind, verify_challenge_chronology
from vidaio.chain import InMemoryChain
from vidaio.challenge import (
    ChallengeAnchor,
    ChallengeCommitment,
    challenge_anchor_payload,
)
from tests.legacy_validator_zero import forged_validator_zero_packet
from vidaio.scoring.config import ScoringConfig
from vidaio.services.artifact_auth import MinerArtifactReceipt
from vidaio.services.protocol import MinerArtifactTaskRequest


def _receipt_verifier(receipt: MinerArtifactReceipt) -> bool:
    return (
        receipt.response_signature == hashlib.sha512(receipt.signed_bytes()).hexdigest()
    )


async def _measured_fixture(tmp_path: Path):  # type: ignore[no-untyped-def]
    store = LocalFsStore(tmp_path / "store")
    bundle = make_fake_bundle(
        store,
        challenge_id="chal-1",
        item_id="chal-1:4-c7",
        miner_hotkey="miner-4",
        dispatch_ordering_key=7,
    )
    chain = InMemoryChain()
    await chain.anchor_commitment(
        challenge_anchor_payload(
            netuid=85,
            dispatch_ordering_key=7,
            commitment_hash=bundle.commitment_hash,
        )
    )
    anchor = ChallengeAnchor(
        netuid=85,
        dispatch_ordering_key=7,
        commitment_hash=bundle.commitment_hash,
        block=1,
        block_hash=chain.block_hash(1),
        txid="0xanchor",
    )
    metadata = MinerArtifactTaskRequest(
        task_id="chal-1:4",
        track="compression",
        input_digest=bundle.challenge_input.digest,
        params={},
        commitment_anchor=anchor,
        deadline_seconds=30,
    )
    unsigned = MinerArtifactReceipt(
        version="2",
        validator_hotkey="validator-1",
        miner_hotkey="miner-4",
        timestamp=1_000,
        nonce="01" * 16,
        input_size=bundle.challenge_input.byte_size,
        metadata=metadata,
        output_digest=bundle.miner_output.digest,
        output_size=bundle.miner_output.byte_size,
        processing_seconds="0.25",
        response_signature="0" * 128,
    )
    receipt = unsigned.model_copy(
        update={
            "response_signature": hashlib.sha512(unsigned.signed_bytes()).hexdigest()
        }
    )
    return (
        bundle.model_copy(
            update={"challenge_anchor": anchor, "miner_receipt": receipt}
        ),
        store,
        chain,
    )


@pytest.mark.asyncio
async def test_measured_result_passes_only_with_exact_finalized_anchor_and_receipt(
    tmp_path: Path,
) -> None:
    bundle, store, chain = await _measured_fixture(tmp_path)
    result = verify_challenge_chronology(
        bundle,
        store,
        chain,
        require_anchor=True,
        expected_netuid=85,
        scoring=ScoringConfig(),
        receipt_verifier=_receipt_verifier,
    )
    assert result.kind is ChronologyKind.PASS

    tampered = bundle.model_copy(
        update={
            "challenge_anchor": bundle.challenge_anchor.model_copy(  # type: ignore[union-attr]
                update={"block_hash": "f" * 64}
            )
        }
    )
    result = verify_challenge_chronology(
        tampered,
        store,
        chain,
        require_anchor=True,
        expected_netuid=85,
        scoring=ScoringConfig(),
        receipt_verifier=_receipt_verifier,
    )
    assert result.kind is ChronologyKind.FAIL
    assert "block hash" in result.detail


@pytest.mark.asyncio
async def test_auditor_detects_a_same_block_external_slot_overwrite(
    tmp_path: Path,
) -> None:
    bundle, store, chain = await _measured_fixture(tmp_path)
    await chain.anchor_commitment(
        challenge_anchor_payload(
            netuid=85,
            dispatch_ordering_key=8,
            commitment_hash="8" * 64,
        )
    )

    result = verify_challenge_chronology(
        bundle,
        store,
        chain,
        require_anchor=True,
        expected_netuid=85,
        scoring=ScoringConfig(),
        receipt_verifier=_receipt_verifier,
    )
    assert result.kind is ChronologyKind.FAIL
    assert "archive state" in result.detail


@pytest.mark.asyncio
async def test_measured_packet_without_miner_receipt_is_inconclusive(
    tmp_path: Path,
) -> None:
    bundle, store, chain = await _measured_fixture(tmp_path)
    result = verify_challenge_chronology(
        bundle.model_copy(update={"miner_receipt": None}),
        store,
        chain,
        require_anchor=True,
        expected_netuid=85,
        scoring=ScoringConfig(),
        receipt_verifier=_receipt_verifier,
    )
    assert result.kind is ChronologyKind.SKIP
    assert "no miner-signed" in result.detail


@pytest.mark.asyncio
async def test_validator_zero_is_rejected_as_unauditable_economic_evidence(
    tmp_path: Path,
) -> None:
    store = LocalFsStore(tmp_path / "zero-store")
    scoring = ScoringConfig()
    challenge_id = "chal-zero"
    item_id = "chal-zero:9-c11"
    miner = "miner-9"
    committed_scorer = "scoring-1.0.0+abc123def456"
    reveal = ChallengeCommitment.preimage_payload(
        "asset-zero",
        sha256_hex(b"dag-zero"),
        99,
        committed_scorer,
        "compression",
        11,
    )
    commitment_hash = sha256_hex(reveal)
    packet = forged_validator_zero_packet(
        item_id=item_id,
        challenge_id=challenge_id,
        track="compression",
        miner_hotkey=miner,
        committed_scorer_version=committed_scorer,
        failure_reason="timeout",
        config=scoring,
    )
    chain = InMemoryChain()
    await chain.anchor_commitment(
        challenge_anchor_payload(
            netuid=85,
            dispatch_ordering_key=11,
            commitment_hash=commitment_hash,
        )
    )
    anchor = ChallengeAnchor(
        netuid=85,
        dispatch_ordering_key=11,
        commitment_hash=commitment_hash,
        block=1,
        block_hash=chain.block_hash(1),
    )
    bundle = build_bundle(
        challenge_id=challenge_id,
        item_id=item_id,
        miner_hotkey=miner,
        commitment_hash=commitment_hash,
        challenge_anchor=anchor,
        stage=LifecycleStage.POST_RETIREMENT,
        challenge_input=store.put(b"input", ArtifactKind.CHALLENGE_INPUT),
        miner_output=store.put(b"", ArtifactKind.MINER_OUTPUT),
        manifest=store.put(b"{}", ArtifactKind.MANIFEST),
        score_packet=store.put(packet.to_json().encode(), ArtifactKind.SCORE_PACKET),
        reference_original=store.put(b"reference", ArtifactKind.REFERENCE_ORIGINAL),
        dag_reveal=store.put(reveal, ArtifactKind.DAG_REVEAL),
        scorer_version=packet.scorer_version,
        created_at="2026-08-23T00:00:00Z",
    )

    result = verify_challenge_chronology(
        bundle,
        store,
        chain,
        require_anchor=True,
        expected_netuid=85,
        scoring=scoring,
        receipt_verifier=_receipt_verifier,
    )
    assert result.kind is ChronologyKind.FAIL
    assert "not launch-valid" in result.detail

    mismatched = bundle.model_copy(update={"item_id": "another-item"})
    result = verify_challenge_chronology(
        mismatched,
        store,
        chain,
        require_anchor=True,
        expected_netuid=85,
        scoring=scoring,
        receipt_verifier=_receipt_verifier,
    )
    assert result.kind is ChronologyKind.FAIL
    assert "not launch-valid" in result.detail
