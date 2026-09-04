"""CPU auditor verification of schema-v14 availability-zero earning evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vidaio.audit.canonical import sha256_hex
from vidaio.audit.recompute import IDENTITY_MISMATCH
from vidaio.audit.store import LocalFsStore
from vidaio.auditor import (
    Auditor,
    AuditorConfig,
    AuditStatus,
    CENSUS_MISMATCH,
    EARNING_PACKET_REPLAY,
    EARNING_STATE_MISMATCH,
    EARNING_STATE_UNVERIFIED,
    FOLD_CURSOR_MISMATCH,
    InMemoryBundleSource,
    ItemVerdictKind,
    SamplePolicy,
)
from vidaio.authority.finalizer import EpochFinalizer, build_audit_manifest
from vidaio.challenge import ChallengeAnchor
from vidaio.chain import ChainNeuron
from vidaio.epoch import (
    AuditManifest,
    AvailabilityInput,
    CycleScore,
    EarningInput,
    EpochLog,
)
from vidaio.services.artifact_auth import (
    ArtifactClientAuth,
    CallableHotkeySigner,
    ValidatorArtifactRequestReceipt,
)
from vidaio.services.protocol import (
    MINER_REQUEST_SIGNATURE_HEADER,
    MinerArtifactTaskRequest,
)
from vidaio.tokenomics import MinerSnapshot, TokenomicsConfig
from vidaio.validator.availability import (
    AvailabilityFailureReason,
    AvailabilityObservation,
    DispatchAttempt,
    build_availability_observation,
)

NOW = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)
CLOSE_BLOCK = 200
BURN_UID = 999
KEY = b"availability-auditor-key"
NO_SAMPLE = SamplePolicy(sample_rate=0.0, min_samples=0)


def _sign(hotkey: str, payload: bytes) -> str:
    assert hotkey == "validator"
    return hashlib.sha512(KEY + b"\x00" + payload).hexdigest()


def _verify(hotkey: str, payload: bytes, signature: str) -> bool:
    return hotkey == "validator" and signature == _sign(hotkey, payload)


def _observation(
    *,
    uid: int = 7,
    miner_hotkey: str = "miner",
    ordering_key: int = 9,
) -> AvailabilityObservation:
    challenge_id = f"challenge-{ordering_key}"
    item_id = f"{challenge_id}:{uid}"
    signer = CallableHotkeySigner(
        "validator", lambda payload: _sign("validator", payload)
    )
    auth = ArtifactClientAuth(
        signer,
        verify_fn=_verify,
        clock=lambda: 1_777_000_000,
        nonce_factory=lambda: f"{ordering_key:032x}",
    )
    metadata = MinerArtifactTaskRequest(
        task_id=item_id,
        track="compression",
        input_digest=sha256_hex(b"availability-input"),
        params={"round": 1},
        deadline_seconds=30.0,
        commitment_anchor=ChallengeAnchor(
            netuid=85,
            dispatch_ordering_key=ordering_key,
            commitment_hash=sha256_hex(f"commitment-{ordering_key}".encode()),
            block=100,
            block_hash=sha256_hex(b"availability-anchor-block"),
        ),
    )
    claims, headers = auth.sign_request(
        metadata,
        input_size=123,
        intended_miner_hotkey=miner_hotkey,
    )
    receipt = ValidatorArtifactRequestReceipt(
        version=claims.version,
        validator_hotkey=claims.validator_hotkey,
        miner_hotkey=claims.miner_hotkey,
        timestamp=claims.timestamp,
        nonce=claims.nonce,
        input_size=claims.input_size,
        metadata=metadata,
        request_signature=headers[MINER_REQUEST_SIGNATURE_HEADER],
    )
    return build_availability_observation(
        attempt=DispatchAttempt(
            uid=uid,
            miner_hotkey=miner_hotkey,
            endpoint="http://203.0.113.7:8091",
            challenge_id=challenge_id,
            item_id=item_id,
            track="compression",
            request=receipt,
        ),
        reason=AvailabilityFailureReason.TIMEOUT,
        signer=signer,
    )


def _input(observation: AvailabilityObservation) -> AvailabilityInput:
    attempt = observation.attempt
    anchor = attempt.request.metadata.commitment_anchor
    assert anchor is not None
    return AvailabilityInput(
        uid=attempt.uid,
        hotkey=attempt.miner_hotkey,
        challenge_id=attempt.challenge_id,
        item_id=attempt.item_id,
        track=attempt.track,
        ordering_key=anchor.dispatch_ordering_key,
        observation_json=observation.canonical_bytes().decode("utf-8"),
        observation_digest=observation.digest(),
    )


@dataclass
class _ArchiveChain:
    observation: AvailabilityObservation
    archive_available: bool = True

    def _neuron(self) -> ChainNeuron:
        attempt = self.observation.attempt
        return ChainNeuron(
            uid=attempt.uid,
            hotkey=attempt.miner_hotkey,
            coldkey="coldkey-7",
            ip="203.0.113.7",
            alpha_stake=0.0,
            emission=0.0,
        )

    def neurons_at(self, close_block: int) -> list[ChainNeuron]:
        assert close_block == CLOSE_BLOCK
        return [self._neuron()]

    def neurons(self) -> list[ChainNeuron]:
        return [self._neuron()]

    def block_time(self, close_block: int) -> datetime:
        assert close_block == CLOSE_BLOCK
        return NOW

    def finalized_block(self) -> int:
        return CLOSE_BLOCK

    def block_hash(self, block: int) -> str | None:
        anchor = self.observation.attempt.request.metadata.commitment_anchor
        assert anchor is not None and block == anchor.block
        return anchor.block_hash

    def read_anchor_at(
        self,
        *,
        netuid: int,
        epoch_id: int,
        domain: str,
        block_number: int,
    ) -> str | None:
        if not self.archive_available:
            raise OSError("archive endpoint unavailable")
        anchor = self.observation.attempt.request.metadata.commitment_anchor
        assert anchor is not None
        assert (netuid, epoch_id, block_number) == (
            anchor.netuid,
            anchor.dispatch_ordering_key,
            anchor.block,
        )
        assert domain
        return anchor.commitment_hash


def _build_log(
    observation: AvailabilityObservation,
) -> tuple[EpochLog, AvailabilityInput]:
    evidence = _input(observation)
    manifest = build_audit_manifest(
        [],
        availability_evidence=[evidence],
        availability_verify_fn=_verify,
        prior_fold_cursors={},
    )
    miner = MinerSnapshot(
        uid=observation.attempt.uid,
        hotkey=observation.attempt.miner_hotkey,
        coldkey="coldkey-7",
        ip="203.0.113.7",
        track="compression",
        accumulate_score=0.0,
    )
    log = EpochFinalizer(
        TokenomicsConfig(), scorer_version="scorer+availability"
    ).build_log(
        epoch_id=12,
        close_block=CLOSE_BLOCK,
        snapshots=(miner,),
        burn_uid=BURN_UID,
        audit_manifest=manifest,
        now=NOW,
        prior_fold_cursors={},
    )
    return log, evidence


def _audit(
    log: EpochLog,
    store: LocalFsStore,
    observation: AvailabilityObservation,
    *,
    archive_available: bool = True,
    prior_log: EpochLog | None = None,
    is_genesis: bool = True,
):
    return Auditor(
        AuditorConfig(
            auditor_hotkey="auditor",
            tokenomics=TokenomicsConfig(),
            burn_uid=BURN_UID,
        ),
        InMemoryBundleSource(),
        chain=_ArchiveChain(observation, archive_available=archive_available),
        availability_verify_fn=_verify,
    ).audit_epoch(
        log,
        store,
        NO_SAMPLE,
        None,
        NOW,
        prior_log=prior_log,
        is_genesis=is_genesis,
    )


def _earning_verdict(report, uid: int = 7):  # type: ignore[no-untyped-def]
    return next(
        verdict
        for verdict in report.earning_verdicts
        if verdict.item_id == f"uid:{uid}"
    )


def _replace_availability(
    log: EpochLog,
    evidence: AvailabilityInput,
    *,
    cycle: CycleScore | None = None,
    fold_cursors: dict[int, int] | None = None,
) -> EpochLog:
    current = log.audit_manifest.earning_for(evidence.uid)
    assert current is not None
    replacement_cycle = cycle or current.cycle_scores[0]
    earning = EarningInput(
        prior_accumulate_score=current.prior_accumulate_score,
        cycle_scores=(replacement_cycle,),
    )
    manifest = log.audit_manifest.model_copy(
        update={
            "earning_inputs": {evidence.uid: earning},
            "availability_inputs": (evidence,),
            "fold_cursors": (
                log.audit_manifest.fold_cursors
                if fold_cursors is None
                else fold_cursors
            ),
        }
    )
    return log.model_copy(update={"audit_manifest": manifest})


def test_cpu_auditor_accepts_archive_bound_availability_only_zero(
    tmp_path: Path,
) -> None:
    observation = _observation()
    log, _ = _build_log(observation)

    report = _audit(log, LocalFsStore(tmp_path / "store"), observation)

    assert _earning_verdict(report).verdict is ItemVerdictKind.PASS
    assert report.overall is AuditStatus.CLEAN


def test_auditor_revalidates_canonical_availability_input(tmp_path: Path) -> None:
    observation = _observation()
    log, evidence = _build_log(observation)
    malformed = evidence.model_copy(
        update={"observation_json": evidence.observation_json + " "}
    )
    tampered = _replace_availability(log, malformed)

    report = _audit(tampered, LocalFsStore(tmp_path / "store"), observation)

    verdict = _earning_verdict(report)
    assert verdict.verdict is ItemVerdictKind.FAIL
    assert verdict.code == EARNING_STATE_MISMATCH
    assert "malformed/non-canonical" in verdict.detail
    assert report.overall is AuditStatus.DISPUTED


def test_auditor_rejects_invalid_availability_signature(tmp_path: Path) -> None:
    observation = _observation()
    log, _ = _build_log(observation)
    forged_observation = observation.model_copy(
        update={"observation_signature": "00" * 64}
    )
    forged = _input(forged_observation)
    tampered = _replace_availability(
        log,
        forged,
        cycle=CycleScore(
            packet_digest=forged.observation_digest,
            ordering_key=forged.ordering_key,
            score=0.0,
        ),
    )

    report = _audit(tampered, LocalFsStore(tmp_path / "store"), observation)

    verdict = _earning_verdict(report)
    assert verdict.verdict is ItemVerdictKind.FAIL
    assert verdict.code == IDENTITY_MISMATCH
    assert "signature is invalid" in verdict.detail


def test_auditor_binds_availability_identity_to_close_block_census(
    tmp_path: Path,
) -> None:
    observation = _observation()
    log, _ = _build_log(observation)
    relabelled = log.miner_census[0].model_copy(update={"hotkey": "other-miner"})
    tampered = log.model_copy(update={"miner_census": (relabelled,)})

    report = _audit(tampered, LocalFsStore(tmp_path / "store"), observation)

    verdict = _earning_verdict(report)
    assert verdict.verdict is ItemVerdictKind.FAIL
    assert verdict.code == CENSUS_MISMATCH
    assert "close-block miner_census identity" in verdict.detail


@pytest.mark.parametrize(
    ("ordering_key", "score"),
    [(10, 0.0), (9, 0.25)],
)
def test_auditor_binds_committed_order_and_exact_zero_in_shared_fold(
    tmp_path: Path, ordering_key: int, score: float
) -> None:
    observation = _observation()
    log, evidence = _build_log(observation)
    tampered = _replace_availability(
        log,
        evidence,
        cycle=CycleScore(
            packet_digest=evidence.observation_digest,
            ordering_key=ordering_key,
            score=score,
        ),
        fold_cursors={7: max(9, ordering_key)},
    )

    report = _audit(tampered, LocalFsStore(tmp_path / "store"), observation)

    verdict = _earning_verdict(report)
    assert verdict.verdict is ItemVerdictKind.FAIL
    assert verdict.code == EARNING_STATE_MISMATCH
    assert report.overall is AuditStatus.DISPUTED


def test_auditor_binds_availability_fold_cursor(tmp_path: Path) -> None:
    observation = _observation()
    log, evidence = _build_log(observation)
    tampered = _replace_availability(log, evidence, fold_cursors={7: 8})

    report = _audit(tampered, LocalFsStore(tmp_path / "store"), observation)

    watermark = next(
        verdict
        for verdict in report.earning_verdicts
        if verdict.item_id == "fold-cursor:7"
    )
    assert watermark.verdict is ItemVerdictKind.FAIL
    assert watermark.code == FOLD_CURSOR_MISMATCH


def test_auditor_rejects_cross_epoch_availability_replay(tmp_path: Path) -> None:
    observation = _observation(ordering_key=9)
    log, _ = _build_log(observation)
    prior = EpochFinalizer(
        TokenomicsConfig(), scorer_version="scorer+availability"
    ).build_log(
        epoch_id=11,
        close_block=CLOSE_BLOCK - 1,
        snapshots=(),
        burn_uid=BURN_UID,
        audit_manifest=AuditManifest(fold_cursors={7: 9}),
        now=NOW,
    )
    replay = log.model_copy(
        update={"prior_log_digest": prior.log_digest(), "epoch_id": 12}
    )

    report = _audit(
        replay,
        LocalFsStore(tmp_path / "store"),
        observation,
        prior_log=prior,
        is_genesis=False,
    )

    verdict = _earning_verdict(report)
    assert verdict.verdict is ItemVerdictKind.FAIL
    assert verdict.code == EARNING_PACKET_REPLAY


def test_auditor_holds_when_availability_anchor_archive_is_unavailable(
    tmp_path: Path,
) -> None:
    observation = _observation()
    log, _ = _build_log(observation)

    report = _audit(
        log,
        LocalFsStore(tmp_path / "store"),
        observation,
        archive_available=False,
    )

    verdict = _earning_verdict(report)
    assert verdict.verdict is ItemVerdictKind.SKIP
    assert verdict.code == EARNING_STATE_UNVERIFIED
    assert "archive/finality read unavailable" in verdict.detail
    assert report.overall is AuditStatus.INCONCLUSIVE
