"""Frozen, I/O-free inputs and state for tokenomics v2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EmissionState(str, Enum):
    IDLE = "IDLE"
    PODIUM = "PODIUM"
    CROWN = "CROWN"


@dataclass(frozen=True, slots=True)
class MinerSnapshot:
    uid: int
    hotkey: str
    coldkey: str
    ip: str
    track: str
    accumulate_score: float
    excluded: bool = False


@dataclass(frozen=True, slots=True)
class ContenderResult:
    """A ranked contender; margin is always derived, never accepted as input."""

    hotkey: str
    uid: int
    score: float

    def __post_init__(self) -> None:
        if not self.hotkey:
            raise ValueError("contender hotkey must be non-empty")
        if isinstance(self.uid, bool) or not isinstance(self.uid, int) or self.uid < 0:
            raise ValueError("contender uid must be a non-negative integer")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("contender score must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class CompetitionResult:
    """An audit-complete result ready for economic application.

    ``applied_at`` is the finalized chain/epoch timestamp that first commits the
    result, never a database-local completion clock. Baseline provenance remains
    mandatory when ``baseline_score`` is None so a failed rerun is diagnosable.
    """

    competition_id: str
    track: str
    cycle: int
    applied_at: datetime
    contenders: Sequence[ContenderResult]
    baseline_score: float | None
    baseline_version: int
    baseline_artifact_digest: str

    def __post_init__(self) -> None:
        if not self.competition_id:
            raise ValueError("competition_id must be non-empty")
        if not self.track:
            raise ValueError("competition track must be non-empty")
        if (
            isinstance(self.cycle, bool)
            or not isinstance(self.cycle, int)
            or self.cycle < 1
        ):
            raise ValueError("competition cycle must be an integer >= 1")
        if self.applied_at.tzinfo is None or self.applied_at.utcoffset() is None:
            raise ValueError("competition applied_at must be timezone-aware")
        if (
            isinstance(self.baseline_version, bool)
            or not isinstance(self.baseline_version, int)
            or self.baseline_version < 0
        ):
            raise ValueError("baseline_version must be a non-negative integer")
        if not _SHA256_RE.fullmatch(self.baseline_artifact_digest):
            raise ValueError(
                "baseline_artifact_digest must be a lowercase sha256 hex digest"
            )
        if self.baseline_score is not None and (
            not math.isfinite(self.baseline_score)
            or not 0.0 <= self.baseline_score <= 1.0
        ):
            raise ValueError("baseline_score must be None or finite and in [0, 1]")
        contenders = tuple(self.contenders)
        if len({c.hotkey for c in contenders}) != len(contenders):
            raise ValueError("competition contenders must have unique hotkeys")
        if len({c.uid for c in contenders}) != len(contenders):
            raise ValueError("competition contenders must have unique uids")
        # Ranking is a derivation of independently recomputed scores, never caller
        # insertion order. The same tie-break is used by competition evidence.
        object.__setattr__(
            self,
            "contenders",
            tuple(
                sorted(
                    contenders,
                    key=lambda contender: (
                        -contender.score,
                        contender.hotkey,
                        contender.uid,
                    ),
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RewardWindowState:
    """Latest successfully applied global competition window.

    PODIUM/CROWN provenance remains after expiry; callers observe IDLE outside the
    half-open interval. Serving-champion persistence is intentionally separate.
    """

    kind: EmissionState = EmissionState.IDLE
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    podium_hotkeys: tuple[str, ...] = ()
    winner_hotkey: str | None = None
    winner_uid: int | None = None
    winner_score: float | None = None
    winner_margin: float | None = None
    baseline_score: float | None = None
    baseline_version: int | None = None
    baseline_artifact_digest: str | None = None
    source_competition_id: str | None = None
    source_track: str | None = None
    source_cycle: int | None = None
    last_applied_cycle: int | None = None

    def __post_init__(self) -> None:
        try:
            kind = EmissionState(self.kind)
        except ValueError as exc:
            raise ValueError(f"unknown emission state {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        if self.kind is EmissionState.IDLE:
            populated = (
                self.starts_at,
                self.ends_at,
                self.winner_hotkey,
                self.winner_uid,
                self.winner_score,
                self.winner_margin,
                self.baseline_score,
                self.baseline_version,
                self.baseline_artifact_digest,
                self.source_competition_id,
                self.source_track,
                self.source_cycle,
                self.last_applied_cycle,
            )
            if any(v is not None for v in populated) or self.podium_hotkeys:
                raise ValueError(
                    "an IDLE state cannot carry competition-window provenance"
                )
            return

        required = (
            self.starts_at,
            self.ends_at,
            self.winner_hotkey,
            self.winner_uid,
            self.winner_score,
            self.winner_margin,
            self.baseline_score,
            self.baseline_version,
            self.baseline_artifact_digest,
            self.source_competition_id,
            self.source_track,
            self.source_cycle,
            self.last_applied_cycle,
        )
        if any(v is None for v in required):
            raise ValueError("a reward window must carry complete source provenance")
        assert self.starts_at is not None and self.ends_at is not None
        if self.starts_at.tzinfo is None or self.starts_at.utcoffset() is None:
            raise ValueError("reward-window starts_at must be timezone-aware")
        if self.ends_at.tzinfo is None or self.ends_at.utcoffset() is None:
            raise ValueError("reward-window ends_at must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("reward-window ends_at must be after starts_at")
        if not self.podium_hotkeys or len(self.podium_hotkeys) > 3:
            raise ValueError("a reward window requires one to three podium hotkeys")
        if len(set(self.podium_hotkeys)) != len(self.podium_hotkeys):
            raise ValueError("reward-window podium hotkeys must be unique")
        if self.podium_hotkeys[0] != self.winner_hotkey:
            raise ValueError("reward-window winner must be podium rank one")
        if self.source_cycle != self.last_applied_cycle:
            raise ValueError("source_cycle and last_applied_cycle must match")
        if (
            isinstance(self.source_cycle, bool)
            or not isinstance(self.source_cycle, int)
            or self.source_cycle < 1
        ):
            raise ValueError("reward-window source cycle must be an integer >= 1")
        if (
            isinstance(self.winner_uid, bool)
            or not isinstance(self.winner_uid, int)
            or self.winner_uid < 0
        ):
            raise ValueError("reward-window winner uid must be a non-negative integer")
        if (
            isinstance(self.baseline_version, bool)
            or not isinstance(self.baseline_version, int)
            or self.baseline_version < 0
        ):
            raise ValueError(
                "reward-window baseline version must be a non-negative integer"
            )
        if not self.source_competition_id or not self.source_track:
            raise ValueError(
                "reward-window source identity and track must be non-empty"
            )
        for name in ("winner_score", "baseline_score"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"reward-window {name} must be finite and in [0, 1]")
        if (
            not isinstance(self.winner_margin, (int, float))
            or isinstance(self.winner_margin, bool)
            or not math.isfinite(float(self.winner_margin))
        ):
            raise ValueError("reward-window winner margin must be finite")
        if not _SHA256_RE.fullmatch(str(self.baseline_artifact_digest)):
            raise ValueError(
                "reward-window baseline digest must be lowercase sha256 hex"
            )


@dataclass(frozen=True, slots=True)
class EmissionShares:
    inference: float
    competition: float
    burn: float
