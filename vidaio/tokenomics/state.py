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
    alpha_stake: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha_stake, bool)
            or not math.isfinite(self.alpha_stake)
            or self.alpha_stake < 0.0
        ):
            raise ValueError("alpha_stake must be finite and >= 0")


def _optional_unit(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or None")
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


#: Upper bound on paid places per competition (the per-rank splits below).
MAX_PODIUM_PLACES = 5


def _validate_split(name: str, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of per-rank shares")
    shares = tuple(value)
    if not 1 <= len(shares) <= MAX_PODIUM_PLACES:
        raise ValueError(f"{name} must name 1 to {MAX_PODIUM_PLACES} places")
    for share in shares:
        if (
            isinstance(share, bool)
            or not isinstance(share, (int, float))
            or not math.isfinite(float(share))
            or not 0.0 <= float(share) <= 1.0
        ):
            raise ValueError(f"{name} shares must be finite and in [0, 1]")
    if abs(sum(float(x) for x in shares) - 1.0) > 1e-9:
        raise ValueError(f"{name} shares must sum to 1.0")
    return tuple(float(x) for x in shares)


@dataclass(frozen=True, slots=True)
class CompetitionRules:
    """Per-competition result rules, fixed in the manifest anchored before enrollment.

    ``crown_margin`` is the inclusive relative improvement over the rerun baseline that
    opens a CROWN window. ``crown_min_score`` is an additional absolute score the winner
    must reach to crown (how an operator raises the bar from one competition to the
    next without any mutable baseline state). ``podium_min_margin`` /
    ``podium_min_score`` are the inclusive conditions a contender must meet to hold a
    paid podium rank at all. ``None`` means "no such condition".

    The optional PAYOUT POLICY fields override the protocol defaults
    (``TokenomicsConfig``) for THIS competition's reward window and are anchored with
    the manifest: ``crown_competition_share`` / ``podium_competition_share`` are the
    fractions of all miner emissions paid to the competition while its window is
    active (the remainder is burned), ``crown_split`` / ``podium_split`` are the
    per-rank fractions of that pot (1 to ``MAX_PODIUM_PLACES`` places, summing to 1),
    ``redistribute_empty_places`` hands the shares of unfilled places to the filled
    ones in proportion, and ``no_qualifier_closes_window`` makes a result with no
    qualifying contender close the running window (burn until the next result)
    instead of leaving it untouched.
    """

    crown_margin: float
    crown_min_score: float | None = None
    podium_min_margin: float | None = None
    podium_min_score: float | None = None
    crown_competition_share: float | None = None
    podium_competition_share: float | None = None
    crown_split: tuple[float, ...] | None = None
    podium_split: tuple[float, ...] | None = None
    redistribute_empty_places: bool | None = None
    no_qualifier_closes_window: bool | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.crown_margin, bool)
            or not isinstance(self.crown_margin, (int, float))
            or not math.isfinite(float(self.crown_margin))
            or not 0.0 < float(self.crown_margin) <= 10.0
        ):
            raise ValueError("crown_margin must be finite and in (0, 10]")
        _optional_unit("crown_min_score", self.crown_min_score)
        _optional_unit("podium_min_score", self.podium_min_score)
        if self.podium_min_margin is not None and (
            isinstance(self.podium_min_margin, bool)
            or not isinstance(self.podium_min_margin, (int, float))
            or not math.isfinite(float(self.podium_min_margin))
            or not -1.0 <= float(self.podium_min_margin) <= 10.0
        ):
            raise ValueError("podium_min_margin must be finite and in [-1, 10]")
        if (
            self.podium_min_margin is not None
            and float(self.podium_min_margin) > float(self.crown_margin)
        ):
            raise ValueError("podium_min_margin cannot exceed crown_margin")
        _optional_unit("crown_competition_share", self.crown_competition_share)
        _optional_unit("podium_competition_share", self.podium_competition_share)
        object.__setattr__(self, "crown_split", _validate_split("crown_split", self.crown_split))
        object.__setattr__(self, "podium_split", _validate_split("podium_split", self.podium_split))
        for name in ("redistribute_empty_places", "no_qualifier_closes_window"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None")


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

    A competition anchored WITHOUT an executable baseline carries ``None`` for all
    three baseline fields (see :attr:`has_baseline`); its crown and podium are
    decided by the anchored absolute score bars alone.
    """

    competition_id: str
    track: str
    cycle: int
    applied_at: datetime
    contenders: Sequence[ContenderResult]
    baseline_score: float | None
    baseline_version: int | None
    baseline_artifact_digest: str | None
    #: Manifest-anchored per-competition rules; ``None`` = the protocol defaults.
    rules: CompetitionRules | None = None

    def __post_init__(self) -> None:
        if self.rules is not None and not isinstance(self.rules, CompetitionRules):
            raise ValueError("competition rules must be a CompetitionRules value")
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
        if self.baseline_version is None or self.baseline_artifact_digest is None:
            if (
                self.baseline_version is not None
                or self.baseline_artifact_digest is not None
                or self.baseline_score is not None
            ):
                raise ValueError(
                    "a result without a baseline carries no baseline version, "
                    "artifact or score"
                )
        else:
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

    @property
    def has_baseline(self) -> bool:
        return self.baseline_version is not None


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
    #: Resolved payout policy of THIS window (schema v18): the fraction of all miner
    #: emissions paid to the competition and the per-hotkey fractions of that pot,
    #: aligned with ``podium_hotkeys``. Empty on windows folded before v18; those
    #: fall back to the live protocol defaults.
    competition_share: float | None = None
    place_shares: tuple[float, ...] = ()
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
            if self.competition_share is not None or self.place_shares:
                raise ValueError("an IDLE state cannot carry a payout policy")
            return

        required = (
            self.starts_at,
            self.ends_at,
            self.winner_hotkey,
            self.winner_uid,
            self.winner_score,
            self.winner_margin,
            self.source_competition_id,
            self.source_track,
            self.source_cycle,
            self.last_applied_cycle,
        )
        if any(v is None for v in required):
            raise ValueError("a reward window must carry complete source provenance")
        baseline = (self.baseline_score, self.baseline_version, self.baseline_artifact_digest)
        # a result anchored without a baseline carries none of the three
        has_baseline = not all(v is None for v in baseline)
        if has_baseline and any(v is None for v in baseline):
            raise ValueError("a reward window must carry complete source provenance")
        assert self.starts_at is not None and self.ends_at is not None
        if self.starts_at.tzinfo is None or self.starts_at.utcoffset() is None:
            raise ValueError("reward-window starts_at must be timezone-aware")
        if self.ends_at.tzinfo is None or self.ends_at.utcoffset() is None:
            raise ValueError("reward-window ends_at must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("reward-window ends_at must be after starts_at")
        if not self.podium_hotkeys or len(self.podium_hotkeys) > MAX_PODIUM_PLACES:
            raise ValueError(
                f"a reward window requires one to {MAX_PODIUM_PLACES} podium hotkeys"
            )
        _optional_unit("competition_share", self.competition_share)
        if self.place_shares:
            if len(self.place_shares) != len(self.podium_hotkeys):
                raise ValueError("place_shares must align with podium_hotkeys")
            for share in self.place_shares:
                _optional_unit("place_shares", share)
            if sum(float(x) for x in self.place_shares) > 1.0 + 1e-9:
                raise ValueError("place_shares cannot exceed the competition pot")
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
        if has_baseline and (
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
            if name == "baseline_score" and not has_baseline:
                continue
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
        if has_baseline and not _SHA256_RE.fullmatch(str(self.baseline_artifact_digest)):
            raise ValueError(
                "reward-window baseline digest must be lowercase sha256 hex"
            )


@dataclass(frozen=True, slots=True)
class EmissionShares:
    inference: float
    competition: float
    burn: float
