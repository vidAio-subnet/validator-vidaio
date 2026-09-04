"""Test-only constructor for packets the launch protocol explicitly refuses."""

from __future__ import annotations

import hashlib

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.gates import ReasonCode, ValidityViolation
from vidaio.scoring.result import ItemScore, compose_item_score, config_digest

_REASONS = {
    "timeout": ReasonCode.MINER_TIMEOUT,
    "miner_error": ReasonCode.MINER_TRANSPORT_ERROR,
    "task_id_mismatch": ReasonCode.MINER_TASK_ID_MISMATCH,
    "digest_mismatch": ReasonCode.MINER_OUTPUT_DIGEST_MISMATCH,
    "duplicate": ReasonCode.REPLAY_DUPLICATE,
}


def forged_validator_zero_packet(
    *,
    item_id: str,
    challenge_id: str,
    track: str,
    miner_hotkey: str,
    committed_scorer_version: str,
    failure_reason: str,
    config: ScoringConfig,
) -> ItemScore:
    """Build the retired empty-output convention for rejection regressions."""
    identity_digest = sha256_hex(
        canonical_json_bytes(
            {
                "committed_scorer_version": committed_scorer_version,
                "convention": "validator-zero/1",
                "scoring_config_digest": config_digest(config),
                "track": track,
            }
        )
    )
    return compose_item_score(
        item_id=item_id,
        challenge_id=challenge_id,
        track=track,
        gate_passed=False,
        violations=[
            ValidityViolation(
                code=_REASONS[failure_reason],
                detail=f"validator observed miner failure: {failure_reason}",
            )
        ],
        breakdown=None,
        config=config,
        miner_hotkey=miner_hotkey,
        content_digest=hashlib.sha256(b"").hexdigest(),
        metrics={},
        backend_versions={},
        scorer_version=f"validator-zero/1+{identity_digest[:12]}",
    )


__all__ = ["forged_validator_zero_packet"]
