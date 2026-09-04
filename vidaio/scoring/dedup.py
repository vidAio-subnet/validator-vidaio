"""Byte-exact cross-miner response dedup — replay/collusion detection.

Only equal, independently verified SHA-256 content digests carry an economic
duplicate verdict. Perceptual similarity is non-economic telemetry: honest
restorations of the same scene are expected to look similar.

Determinism: entries are sorted by ``(order_key, key)`` before clustering, so the same
set of responses always yields the same verdicts regardless of arrival order. Launch
inference supplies the independently recomputable ``anchor_hash_hotkey/1`` rank as
``order_key``; ``key`` (the response/miner id) only breaks a theoretical rank tie.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from vidaio.scoring.gates import ReasonCode


class DedupEntry(BaseModel):
    """One miner response as seen by the dedup pass."""

    model_config = {"frozen": True}

    key: str
    content_digest: str
    #: Optional observability fact. It never changes an economic verdict.
    perceptual_hash: str | None = None
    #: Deterministic precedence — earlier sorts first and is kept on a duplicate match.
    order_key: str = ""


class DedupVerdict(BaseModel):
    model_config = {"frozen": True}

    key: str
    kept: bool
    duplicate_of: str | None = None
    reason: ReasonCode | None = None
    detail: str = ""


def dedup_responses(entries: Sequence[DedupEntry]) -> dict[str, DedupVerdict]:
    """Keep the first deterministic entry per exact content digest."""
    ordered = sorted(entries, key=lambda e: (e.order_key, e.key))
    kept: list[DedupEntry] = []
    verdicts: dict[str, DedupVerdict] = {}

    for entry in ordered:
        match: DedupEntry | None = None
        detail = ""
        for prior in kept:
            if entry.content_digest == prior.content_digest:
                match = prior
                detail = "exact content digest match"
                break
        if match is None:
            kept.append(entry)
            verdicts[entry.key] = DedupVerdict(key=entry.key, kept=True)
        else:
            verdicts[entry.key] = DedupVerdict(
                key=entry.key,
                kept=False,
                duplicate_of=match.key,
                reason=ReasonCode.REPLAY_DUPLICATE,
                detail=detail,
            )
    return verdicts
