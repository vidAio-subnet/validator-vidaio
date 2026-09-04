"""Canonical, request-bound evidence for miner-attributable availability zeros.

An availability zero is not a media score. It records that the authority offered an
anchored, deadline-bounded artifact-v2 task to the chain-attributed endpoint and then
observed a protocol-enumerated miner-side failure. The validator signs both the exact
request receipt and the resulting observation. Auditors can deterministically verify
the identities, signatures, anchor, deadline, endpoint binding and zero fold on CPU.

This does not make a negative network observation Byzantine-proof: the authority could
withhold a response and claim timeout. That explicit trust boundary is preferable to an
invisible EWMA freeze, and the immutable observation gives miners/auditors a concrete
fact to dispute. Scorer, audit-store, chain and challenge-service failures are not in
the reason enum and therefore cannot become economic zeros through this model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.services.artifact_auth import (
    ArtifactHotkeySigner,
    MinerArtifactReceipt,
    ValidatorArtifactRequestReceipt,
    bittensor_hotkey_verify,
    verify_validator_artifact_request_receipt,
)

AVAILABILITY_OBSERVATION_DOMAIN = b"vidaio:availability-observation:v1\x00"


class AvailabilityFailureReason(StrEnum):
    """The complete launch set of miner-attributable zero reasons."""

    TIMEOUT = "timeout"
    TRANSPORT_ERROR = "transport_error"
    RESTART_FENCE_EXHAUSTED = "restart_fence_exhausted"
    UNREACHABLE_ENDPOINT = "unreachable_endpoint"
    PROTOCOL_ERROR = "protocol_error"
    TASK_ID_MISMATCH = "task_id_mismatch"
    OUTPUT_DIGEST_MISMATCH = "output_digest_mismatch"
    RECEIPT_INVALID = "receipt_invalid"


class DispatchAttempt(BaseModel):
    """The exact signed task offer plus its chain-attributed network target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uid: int = Field(ge=0)
    miner_hotkey: str
    endpoint: str = Field(min_length=1, max_length=512)
    challenge_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    track: str = Field(min_length=1, max_length=32)
    request: ValidatorArtifactRequestReceipt

    @model_validator(mode="after")
    def _bind_request(self) -> "DispatchAttempt":
        metadata = self.request.metadata
        if self.request.miner_hotkey != self.miner_hotkey:
            raise ValueError("request intended miner does not match dispatch miner")
        if metadata.task_id != self.item_id:
            raise ValueError("signed request task_id does not match dispatch item_id")
        if not self.item_id.startswith(f"{self.challenge_id}:"):
            raise ValueError("dispatch item_id is not bound to challenge_id")
        if metadata.track != self.track:
            raise ValueError("signed request track does not match dispatch track")
        if metadata.commitment_anchor is None:
            raise ValueError("availability folds require a finalized challenge anchor")
        if metadata.deadline_seconds <= 0:
            raise ValueError("availability dispatch deadline must be positive")
        if any(ch.isspace() for ch in self.endpoint):
            raise ValueError("dispatch endpoint must not contain whitespace")
        return self


class AvailabilityObservation(BaseModel):
    """One validator-signed, deterministic economic zero observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    attempt: DispatchAttempt
    reason: AvailabilityFailureReason
    score: Literal[0.0] = 0.0
    returned_task_id: str | None = Field(default=None, max_length=128)
    observed_output_digest: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    miner_receipt: MinerArtifactReceipt | None = None
    observation_signature: str = Field(pattern=r"^[0-9a-f]{128}$")

    @model_validator(mode="after")
    def _reason_shape(self) -> "AvailabilityObservation":
        if self.reason is AvailabilityFailureReason.TASK_ID_MISMATCH:
            if (
                not self.returned_task_id
                or self.returned_task_id == self.attempt.item_id
            ):
                raise ValueError(
                    "task-id mismatch must record the different returned id"
                )
        elif self.returned_task_id is not None:
            raise ValueError("returned_task_id is only valid for task_id_mismatch")

        if self.reason is AvailabilityFailureReason.OUTPUT_DIGEST_MISMATCH:
            if self.observed_output_digest is None:
                raise ValueError(
                    "digest mismatch must record the observed output digest"
                )
        elif self.observed_output_digest is not None:
            raise ValueError(
                "observed_output_digest is only valid for output_digest_mismatch"
            )

        if self.miner_receipt is not None:
            request = self.miner_receipt.request_receipt()
            if request is None or request != self.attempt.request:
                raise ValueError(
                    "miner receipt does not carry the exact dispatch request"
                )
            if self.miner_receipt.miner_hotkey != self.attempt.miner_hotkey:
                raise ValueError(
                    "miner receipt identity does not match dispatch target"
                )
        return self

    def unsigned_obj(self) -> dict[str, object]:
        payload = self.model_dump(mode="json")
        payload.pop("observation_signature")
        return payload

    def signed_bytes(self) -> bytes:
        return AVAILABILITY_OBSERVATION_DOMAIN + canonical_json_bytes(
            self.unsigned_obj()
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))

    def digest(self) -> str:
        return sha256_hex(self.canonical_bytes())

    def as_row(self) -> dict[str, object]:
        """SQLite row shape consumed atomically by ``commit_round``."""
        return {
            "uid": self.attempt.uid,
            "item_id": self.attempt.item_id,
            "challenge_id": self.attempt.challenge_id,
            "track": self.attempt.track,
            "miner_hotkey": self.attempt.miner_hotkey,
            "endpoint": self.attempt.endpoint,
            "reason": self.reason.value,
            "score": 0.0,
            "observation_digest": self.digest(),
            "observation_json": self.canonical_bytes().decode("utf-8"),
        }


def build_availability_observation(
    *,
    attempt: DispatchAttempt,
    reason: AvailabilityFailureReason,
    signer: ArtifactHotkeySigner,
    returned_task_id: str | None = None,
    observed_output_digest: str | None = None,
    miner_receipt: MinerArtifactReceipt | None = None,
) -> AvailabilityObservation:
    """Build and sign the canonical observation under the request validator."""
    unsigned = AvailabilityObservation(
        attempt=attempt,
        reason=reason,
        returned_task_id=returned_task_id,
        observed_output_digest=observed_output_digest,
        miner_receipt=miner_receipt,
        observation_signature="aa" * 64,  # exact-shape placeholder, replaced below
    )
    if signer.hotkey != attempt.request.validator_hotkey:
        raise ValueError("observation signer is not the request-signing validator")
    signature = signer.sign(unsigned.signed_bytes())
    return unsigned.model_copy(update={"observation_signature": signature})


def verify_availability_observation(
    observation: AvailabilityObservation,
    *,
    verify_fn: Callable[[str, bytes, str], bool] = bittensor_hotkey_verify,
) -> bool:
    """CPU-only cryptographic/structural verification of an availability zero."""
    if not verify_validator_artifact_request_receipt(
        observation.attempt.request, verify_fn=verify_fn
    ):
        return False
    try:
        if not verify_fn(
            observation.attempt.request.validator_hotkey,
            observation.signed_bytes(),
            observation.observation_signature,
        ):
            return False
        if observation.miner_receipt is not None and not verify_fn(
            observation.miner_receipt.miner_hotkey,
            observation.miner_receipt.signed_bytes(),
            observation.miner_receipt.response_signature,
        ):
            return False
    except Exception:
        return False
    return True


__all__ = [
    "AVAILABILITY_OBSERVATION_DOMAIN",
    "AvailabilityFailureReason",
    "AvailabilityObservation",
    "DispatchAttempt",
    "build_availability_observation",
    "verify_availability_observation",
]
