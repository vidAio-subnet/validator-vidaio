"""Content pool: assets, leakage-controlled splits, provenance log, retire-after-use.

Leakage controls (spec design spec §18):
  * Splits are computed per SOURCE GROUP (creator/source/... fields), never per clip:
    every asset whose split-key fields match lands in the same split, deterministically,
    via sha256(source_key || split_salt). Holdout assets are never issued as challenges.
  * Each asset records whether the metadata-strip step is part of its ingest pipeline.
  * A near-duplicate hook checks perceptual fingerprints against known public
    benchmark/training-corpus fingerprints before an asset enters the pool.

Status lifecycle: ingesting -> fresh -> in_use -> (fresh ... ->) retired.
Assets enter the pool as 'ingesting' (register_asset) and become 'fresh' only when
every planned ingest step — fetch, transcode (incl. metadata strip), segment — is
confirmed via confirm_ingest_step. `checkout_asset` issues only 'fresh'
challenge-split assets (weighted by tags on request) and marks them in_use;
`release_asset` retires them once `use_count` reaches `retire_after_uses`.
Retired assets never re-enter circulation.

All timestamps are passed in by the caller (no clock reads inside logic); all
randomness comes from an injected `random.Random`.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vidaio.challenge.config import ChallengeConfig
from vidaio.challenge.dag import canonical_json_dumps

Split = Literal["challenge", "holdout"]
AssetStatus = Literal["ingesting", "fresh", "in_use", "retired"]

_PREFERRED_TAG_WEIGHT = 4  # weight multiplier for assets matching all preferred tags


class NoFreshAssetError(LookupError):
    """Raised when the challenge split has no fresh asset to issue."""


class NearDuplicateError(RuntimeError):
    """Raised when an asset's perceptual fingerprint matches a known public corpus."""


class UnresolvedChallengeError(RuntimeError):
    """Raised when force-retiring an asset that still has dispatched challenges
    without an explicit force=True."""


class Asset(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    content_digest: str  # sha256 of the pristine reference bytes
    perceptual_fingerprint: str  # backend-injected (e.g. videohash); opaque here
    source_url: str
    license_basis: str
    ingest_date: str
    creator: str
    source: str
    subject: str = ""
    scene: str = ""
    resolution_tag: str  # difficulty dial: e.g. 1080p / 4k
    motion_tag: str  # e.g. low / medium / high
    content_type_tag: str  # e.g. sports / animation / talking-head
    metadata_stripped: bool = False
    split: Split
    # Default mirrors the SQL DEFAULT and the ingest lifecycle: an Asset built
    # without an explicit status is 'ingesting' — not checkoutable — so no insert
    # path can create a fresh (issuable) asset by omission; 'fresh' must be stated
    # deliberately, and the production path only reaches it via confirm_ingest_step.
    status: AssetStatus = "ingesting"
    use_count: int = Field(0, ge=0)


class FingerprintIndex(Protocol):
    """Near-duplicate lookup against known public benchmark/corpus fingerprints."""

    def is_near_duplicate(self, fingerprint: str) -> bool: ...


class StaticFingerprintIndex:
    """Exact-match test fake for the FingerprintIndex protocol.

    Production challenge ingest injects the CPU-pHash Hamming-distance index from
    :mod:`vidaio.challenge_service.fingerprint`.
    """

    def __init__(self, known: set[str] | frozenset[str]) -> None:
        self._known = frozenset(known)

    def is_near_duplicate(self, fingerprint: str) -> bool:
        return fingerprint in self._known


# --- splits ---------------------------------------------------------------------------


def _field(obj: Asset | Mapping[str, Any], name: str) -> str:
    if isinstance(obj, Mapping):
        return str(obj.get(name, ""))
    return str(getattr(obj, name))


def source_key(obj: Asset | Mapping[str, Any], cfg: ChallengeConfig) -> str:
    """Grouping identity for split assignment, from cfg.split_key_fields."""
    return "\x1f".join(_field(obj, f) for f in cfg.split_key_fields)


def assign_split(obj: Asset | Mapping[str, Any], cfg: ChallengeConfig) -> Split:
    """Deterministic split by source group: sha256(source_key || split_salt).

    Never keyed on clip identity — all assets sharing the source key land together.
    """
    key = source_key(obj, cfg) + "\x1f" + cfg.split_salt
    frac = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return "holdout" if frac < cfg.holdout_fraction else "challenge"


# --- persistence ----------------------------------------------------------------------

_ASSET_COLUMNS = (
    "id",
    "content_digest",
    "perceptual_fingerprint",
    "source_url",
    "license_basis",
    "ingest_date",
    "creator",
    "source",
    "subject",
    "scene",
    "resolution_tag",
    "motion_tag",
    "content_type_tag",
    "metadata_stripped",
    "split",
    "status",
    "use_count",
)


def _row_to_asset(row: sqlite3.Row) -> Asset:
    data = {c: row[c] for c in _ASSET_COLUMNS}
    data["metadata_stripped"] = bool(data["metadata_stripped"])
    return Asset.model_validate(data)


def add_asset(conn: sqlite3.Connection, asset: Asset) -> None:
    cols = ", ".join(_ASSET_COLUMNS)
    marks = ", ".join("?" for _ in _ASSET_COLUMNS)
    values = tuple(
        int(v) if isinstance(v, bool) else v
        for v in (getattr(asset, c) for c in _ASSET_COLUMNS)
    )
    conn.execute(f"INSERT INTO assets ({cols}) VALUES ({marks})", values)


def get_asset(conn: sqlite3.Connection, asset_id: str) -> Asset:
    row = conn.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown asset {asset_id}")
    return _row_to_asset(row)


# --- provenance log (append-only; enforced by DB triggers) ----------------------------


def append_provenance(
    conn: sqlite3.Connection,
    asset_id: str,
    event: str,
    detail: Mapping[str, Any],
    recorded_at: str,
) -> None:
    conn.execute(
        "INSERT INTO provenance_log (asset_id, event, detail, recorded_at)"
        " VALUES (?, ?, ?, ?)",
        (asset_id, event, canonical_json_dumps(dict(detail)), recorded_at),
    )


def provenance_log(conn: sqlite3.Connection, asset_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT seq, event, detail, recorded_at FROM provenance_log"
        " WHERE asset_id = ? ORDER BY seq",
        (asset_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# --- checkout / retire-after-use ------------------------------------------------------


def checkout_asset(
    conn: sqlite3.Connection,
    rng: Any,
    checked_out_at: str,
    *,
    prefer_tags: Mapping[str, str] | None = None,
    eligible_ids: Sequence[str] | None = None,
) -> Asset:
    """Issue one fresh challenge-split asset and mark it in_use (+1 use).

    Only status='fresh' rows are ever considered, so an asset whose ingest steps
    are not all confirmed (status='ingesting') can never be issued — an
    unconfirmed/unstripped asset is structurally unavailable here, as are
    holdout-split assets. When `prefer_tags` is given
    (attribute -> value, e.g. {"motion_tag": "high"}), assets matching ALL preferred
    tags are weighted 4:1 over the rest; selection still flows through `rng` only.
    When ``eligible_ids`` is supplied, checkout is restricted to that exact set.
    Challenge production uses this after validating each asset's immutable segment
    manifest, so an asset without a track-eligible clip cannot win a pool race.

    The claim and its ``checked_out`` provenance fact are one SQLite transaction:
    a guarded UPDATE (`WHERE status = 'fresh'`) either wins the row and records the
    matching fact, or both effects roll back. A concurrent loser re-selects from
    the remaining fresh pool; a provenance failure can never strand ``in_use``.
    """
    if conn.in_transaction:
        raise RuntimeError("asset checkout requires no caller transaction")
    restricted_ids = None if eligible_ids is None else frozenset(eligible_ids)
    if restricted_ids is not None and not restricted_ids:
        raise NoFreshAssetError(
            "no fresh asset with a track-eligible immutable segment"
        )
    while True:
        rows = conn.execute(
            "SELECT * FROM assets WHERE status = 'fresh' AND split = 'challenge' "
            "ORDER BY id"
        ).fetchall()
        if restricted_ids is not None:
            # Filter in Python: a production corpus can exceed SQLite's bounded
            # parameter count, so an unbounded ``IN (?, ...)`` is not safe here.
            rows = [row for row in rows if str(row["id"]) in restricted_ids]
        if not rows:
            suffix = (
                " with a track-eligible immutable segment"
                if restricted_ids is not None
                else ""
            )
            raise NoFreshAssetError(
                f"no fresh asset available in the challenge split{suffix}"
            )
        candidates = [_row_to_asset(r) for r in rows]
        if prefer_tags:
            weights = [
                _PREFERRED_TAG_WEIGHT
                if all(getattr(a, k) == v for k, v in prefer_tags.items())
                else 1
                for a in candidates
            ]
        else:
            weights = [1] * len(candidates)
        chosen = rng.choices(candidates, weights=weights, k=1)[0]
        conn.execute("BEGIN IMMEDIATE")
        try:
            claimed = conn.execute(
                "UPDATE assets SET status = 'in_use', use_count = use_count + 1"
                " WHERE id = ? AND status = 'fresh'",
                (chosen.id,),
            )
            if claimed.rowcount != 1:
                conn.execute("ROLLBACK")
                # Lost the claim race to a concurrent checkout; the candidate list
                # was stale, so rebuild it outside the write transaction.
                continue
            issued = get_asset(conn, chosen.id)
            append_provenance(
                conn,
                chosen.id,
                "checked_out",
                {"use": issued.use_count},
                checked_out_at,
            )
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return issued


def release_asset(
    conn: sqlite3.Connection,
    asset_id: str,
    retire_after_uses: int,
    released_at: str,
) -> Asset:
    """Finish an issued use: retire once use_count >= retire_after_uses, else re-pool."""
    asset = get_asset(conn, asset_id)
    if asset.status != "in_use":
        raise ValueError(f"asset {asset_id} is {asset.status}, not in_use")
    new_status = "retired" if asset.use_count >= retire_after_uses else "fresh"
    conn.execute("UPDATE assets SET status = ? WHERE id = ?", (new_status, asset_id))
    append_provenance(
        conn,
        asset_id,
        "retired" if new_status == "retired" else "released",
        {"use_count": asset.use_count},
        released_at,
    )
    return get_asset(conn, asset_id)


def retire_asset(
    conn: sqlite3.Connection, asset_id: str, retired_at: str, *, force: bool = False
) -> Asset:
    """Force-retire an asset (admin path: leak suspicion, license withdrawal, ...).

    While challenges on the asset are still dispatched, retirement requires an
    explicit force=True — and even then reveal_commitment stays blocked until every
    challenge resolves or expires.
    """
    get_asset(conn, asset_id)  # existence check
    unresolved = conn.execute(
        "SELECT COUNT(*) AS n FROM challenges WHERE asset_id = ? AND status = 'dispatched'",
        (asset_id,),
    ).fetchone()["n"]
    if unresolved and not force:
        raise UnresolvedChallengeError(
            f"asset {asset_id} has {unresolved} dispatched challenge(s); pass force=True"
            " to retire anyway (reveal stays blocked until they resolve)"
        )
    conn.execute("UPDATE assets SET status = 'retired' WHERE id = ?", (asset_id,))
    append_provenance(
        conn,
        asset_id,
        "retired",
        {"forced": True, "unresolved_challenges": unresolved},
        retired_at,
    )
    return get_asset(conn, asset_id)


def check_near_duplicate(index: FingerprintIndex | None, fingerprint: str) -> None:
    """Raise NearDuplicateError if the fingerprint matches a known public corpus."""
    if index is not None and index.is_near_duplicate(fingerprint):
        raise NearDuplicateError(
            "perceptual fingerprint matches a known public benchmark/corpus item"
        )
