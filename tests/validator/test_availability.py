"""Evidence-backed availability folds: signatures, bindings and closed taxonomy."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from vidaio.challenge import ChallengeAnchor
from vidaio.services.artifact_auth import (
    ArtifactClientAuth,
    CallableHotkeySigner,
    ValidatorArtifactRequestReceipt,
)
from vidaio.services.protocol import (
    MINER_REQUEST_SIGNATURE_HEADER,
    MinerArtifactTaskRequest,
)
from vidaio.validator.availability import (
    AvailabilityFailureReason,
    AvailabilityObservation,
    DispatchAttempt,
    build_availability_observation,
    verify_availability_observation,
)

KEYS = {"validator": b"validator-key", "miner": b"miner-key"}


def _sign(hotkey: str, payload: bytes) -> str:
    return hashlib.sha512(KEYS[hotkey] + b"\x00" + payload).hexdigest()


def _verify(hotkey: str, payload: bytes, signature: str) -> bool:
    return hotkey in KEYS and signature == _sign(hotkey, payload)


def _attempt(*, anchored: bool = True) -> tuple[DispatchAttempt, CallableHotkeySigner]:
    signer = CallableHotkeySigner(
        "validator", lambda payload: _sign("validator", payload)
    )
    auth = ArtifactClientAuth(
        signer,
        verify_fn=_verify,
        clock=lambda: 1234,
        nonce_factory=lambda: "01" * 16,
    )
    metadata = MinerArtifactTaskRequest(
        task_id="challenge-1:7",
        track="compression",
        input_digest=hashlib.sha256(b"input").hexdigest(),
        params={"round": 1},
        deadline_seconds=120.0,
        commitment_anchor=(
            ChallengeAnchor(
                netuid=85,
                dispatch_ordering_key=9,
                commitment_hash="ab" * 32,
                block=100,
                block_hash="cd" * 32,
            )
            if anchored
            else None
        ),
    )
    claims, headers = auth.sign_request(
        metadata, input_size=5, intended_miner_hotkey="miner"
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
    return (
        DispatchAttempt(
            uid=7,
            miner_hotkey="miner",
            endpoint="http://203.0.113.7:8091",
            challenge_id="challenge-1",
            item_id="challenge-1:7",
            track="compression",
            request=receipt,
        ),
        signer,
    )


def test_timeout_observation_is_cpu_verifiable_and_content_addressed() -> None:
    attempt, signer = _attempt()
    observation = build_availability_observation(
        attempt=attempt,
        reason=AvailabilityFailureReason.TIMEOUT,
        signer=signer,
    )

    assert observation.score == 0.0
    assert verify_availability_observation(observation, verify_fn=_verify)
    assert len(observation.digest()) == 64
    assert observation.as_row()["observation_digest"] == observation.digest()


def test_post_signature_reason_change_is_rejected() -> None:
    attempt, signer = _attempt()
    observation = build_availability_observation(
        attempt=attempt,
        reason=AvailabilityFailureReason.TIMEOUT,
        signer=signer,
    )
    tampered = observation.model_copy(
        update={"reason": AvailabilityFailureReason.TRANSPORT_ERROR}
    )
    assert not verify_availability_observation(tampered, verify_fn=_verify)


@pytest.mark.parametrize("signature", ("ab" * 63, "ab" * 65, "AB" * 64))
def test_observation_signature_requires_exact_lowercase_sr25519_shape(
    signature: str,
) -> None:
    attempt, signer = _attempt()
    body = build_availability_observation(
        attempt=attempt,
        reason=AvailabilityFailureReason.TIMEOUT,
        signer=signer,
    ).model_dump(mode="json")
    body["observation_signature"] = signature
    with pytest.raises(ValidationError):
        AvailabilityObservation.model_validate(body)


def test_unanchored_request_cannot_become_an_availability_fold() -> None:
    with pytest.raises(ValueError, match="finalized challenge anchor"):
        _attempt(anchored=False)


def test_reason_taxonomy_and_zero_are_closed() -> None:
    attempt, signer = _attempt()
    valid = build_availability_observation(
        attempt=attempt,
        reason=AvailabilityFailureReason.PROTOCOL_ERROR,
        signer=signer,
    )
    body = valid.model_dump(mode="json")
    body["reason"] = "scorer_failure"
    with pytest.raises(ValidationError):
        AvailabilityObservation.model_validate(body)
    body = valid.model_dump(mode="json")
    body["score"] = 0.5
    with pytest.raises(ValidationError):
        AvailabilityObservation.model_validate(body)
