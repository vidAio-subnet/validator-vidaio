"""Miner registry (SQLite, own migrations).

Function-based persistence over a caller-owned connection (core.connect /
core.apply_migrations), mirroring the challenge pool style. Concerns:

1. Registry sync from ChainAdapter neuron snapshots — a hotkey change on a uid
   is a NEW miner: the EWMA accumulator is purged and the warrant track is reset to
   NULL (must be re-probed) (spec §01 "refresh_miner_manager · purge hotkey
   changes"; no phantom history carries over).

2. Track classification — the fixed TaskWarrant. A miner's track exists ONLY as
   an explicit warrant probe result recorded via `record_track`; a probe timeout,
   a missing record or a garbage value leaves track=NULL and the round loop SKIPS
   the miner with a structured log + metric. This deliberately replaces the old
   validator.py:844-849 behaviour where any of those cases was silently bucketed
   as upscaling and a real compression miner got mis-scored (design spec §07 confirmed bug).

   (The block-driven retention-window bookkeeping — `observe_retention` /
   `latest_full_window` / the `retention_windows` table — was REMOVED with the
   retention multiplier for v1 — retention removed — owner decision; an internal review. It fed the removed MinerSnapshot windowed
   fields, which no longer affect weight.)

4. Round atomicity + evidence. `begin_round`/`commit_round` make
   ONE round's OBSERVABLE state a single BEGIN IMMEDIATE transaction stamped into
   a round ledger, so the independently-running weight-setter can never read a
   half-applied round and a partial round stays detectable (committed_at IS NULL)
   after a crash. `inflight_challenges` tracks every challenge the round fetched
   so it is resolved even after a crash — WITH the identity it was fetched under
, because resolve is ownership-enforced and the process that
   recovers may no longer be configured as the process that fetched.

   Round-2 an internal review: "observable state" is EVERYTHING a reader can see — not just
   the EWMA folds. The registry sync (hotkey-change purges, new miners, ip/coldkey
   updates) and the warrant-probe track writes used to commit in their own earlier
   transaction, so a reader could observe a hotkey reset from a round that then
   crashed and never applied its scores. They are now
   STAGED in memory for the duration of the round (`RegistryUpdate`, applied by
   `commit_round`) and the round loop reads the staged view through
   `planned_tracks`. The only thing a round writes before its commit is the
   `rounds` marker row itself, which carries no miner state and which every
   evidence reader ignores until committed_at is stamped.

5. The durable scorer pin: `load_scorer_pin` /
   `record_scorer_pin` / `clear_scorer_pin` over the single-row `scorer_pin`
   table, so the scorer identity a validator's accumulators were built under
   survives a restart and a silent worker swap is refused rather than merged.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from vidaio.chain.adapter import ChainNeuron
from vidaio.core.db import connect
from vidaio.scoring.config import TRACK_COMPRESSION, TRACK_UPSCALING
from vidaio.tokenomics import MinerSnapshot, accumulate

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

KNOWN_TRACKS = (TRACK_COMPRESSION, TRACK_UPSCALING)

#: The only outcomes the challenge service's /challenge/{id}/resolve accepts.
RESOLVE_OUTCOMES = ("resolved", "expired")


def utc_now_iso() -> str:
    """Canonical tz-aware UTC stamp ('+00:00'), so ISO strings sort as instants."""
    return datetime.now(timezone.utc).isoformat()


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One explicit BEGIN IMMEDIATE ... COMMIT (connections are autocommit).

    The write lock is taken up-front so a concurrent reader either sees the whole
    batch or none of it; any exception rolls the whole batch back.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def connection_factory(conn: sqlite3.Connection) -> Callable[[], sqlite3.Connection] | None:
    """A factory that opens a NEW connection to the same database file, or None.

    sqlite3 connections must not be shared across threads, so health checks
    served from the HealthServer's thread need their own handle.
    Returns None for ':memory:' databases, which have no reopenable file.
    """
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main" and row[2]:
            path = str(row[2])
            return lambda: connect(path)
    return None


def normalize_track(value: object) -> str | None:
    """Map an arbitrary probe/DB value onto a known track, else None.

    Anything that is not exactly a known track string is UNKNOWN — garbage never
    becomes a default bucket.
    """
    return value if isinstance(value, str) and value in KNOWN_TRACKS else None


def sync_neurons(conn: sqlite3.Connection, neurons: Sequence[ChainNeuron], block: int) -> list[int]:
    """Upsert the (caller-filtered) miner neurons into the registry.

    Returns the uids whose hotkey changed — those rows were purged (score reset,
    track cleared) and re-seeded as new miners.
    """
    purged: list[int] = []
    for n in neurons:
        row = conn.execute("SELECT hotkey FROM miners WHERE uid = ?", (n.uid,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO miners (uid, hotkey, coldkey, ip, track, accumulate_score,"
                " first_seen_block) VALUES (?, ?, ?, ?, NULL, 0.0, ?)",
                (n.uid, n.hotkey, n.coldkey, n.ip, block),
            )
        elif row["hotkey"] != n.hotkey:
            purged.append(n.uid)
            conn.execute(
                "UPDATE miners SET hotkey = ?, coldkey = ?, ip = ?, track = NULL,"
                " accumulate_score = 0.0, first_seen_block = ? WHERE uid = ?",
                (n.hotkey, n.coldkey, n.ip, block, n.uid),
            )
        else:
            conn.execute(
                "UPDATE miners SET coldkey = ?, ip = ? WHERE uid = ?",
                (n.coldkey, n.ip, n.uid),
            )
    return purged


def get_miner(conn: sqlite3.Connection, uid: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM miners WHERE uid = ?", (uid,)).fetchone()


def record_track(conn: sqlite3.Connection, uid: int, track: str) -> None:
    """Record an explicit warrant probe result. Only known tracks are recordable —
    there is deliberately no way to write a default or a garbage value."""
    if track not in KNOWN_TRACKS:
        raise ValueError(f"unknown track {track!r}; known: {KNOWN_TRACKS}")
    conn.execute("UPDATE miners SET track = ? WHERE uid = ?", (track, uid))


def track_of(conn: sqlite3.Connection, uid: int) -> str | None:
    """The recorded track for uid, normalized: missing row / NULL / garbage → None."""
    row = get_miner(conn, uid)
    return normalize_track(row["track"]) if row is not None else None


def planned_tracks(
    conn: sqlite3.Connection, neurons: Sequence[ChainNeuron]
) -> dict[int, str | None]:
    """The tracks a round would see AFTER its (staged, uncommitted) registry sync.

    review #9 round 2: the sync no longer runs before the round, so `track_of`
    alone would answer with pre-sync state. This applies the sync's own rule
    WITHOUT writing anything: a uid with no row, or one whose hotkey has changed
    (`sync_neurons` will purge it), is a NEW miner whose track is unknown and must
    be re-probed; everyone else keeps their recorded (normalized) track.
    """
    planned: dict[int, str | None] = {}
    for n in neurons:
        row = get_miner(conn, n.uid)
        if row is None or row["hotkey"] != n.hotkey:
            planned[n.uid] = None
            continue
        planned[n.uid] = normalize_track(row["track"])
    return planned


def apply_scores(conn: sqlite3.Connection, scores: Mapping[int, float], decay: float) -> None:
    """EWMA-fold one round of cycle scores into the accumulators.

    Uses vidaio.tokenomics.accumulate verbatim, so the -1 exclusion sentinel
    latches and a genuine score after exclusion restarts from 0.0 — the validator
    has no private EWMA math to drift from the weight composition.
    """
    for uid, score in scores.items():
        row = get_miner(conn, uid)
        if row is None:
            continue
        new = accumulate(row["accumulate_score"], score, decay)
        conn.execute("UPDATE miners SET accumulate_score = ? WHERE uid = ?", (new, uid))


# -- round ledger + atomic round application ------------------------


class RoundLedgerError(RuntimeError):
    """A round was committed twice, or committed without ever being started."""


@dataclass(frozen=True)
class RegistryUpdate:
    """One round's STAGED registry effects, applied only by `commit_round`.

    Round-2 an internal review: these writes (hotkey-change purges, new miner rows,
    warrant-probe tracks) used to commit in their own transaction before scoring, so
    a weight-setter could read a hotkey reset belonging to a round that then died.
    Staged here for the round's duration instead; the round loop reads the same view
    through `planned_tracks`.
    """

    #: The (already filtered/deduped) neurons this round ran over.
    neurons: tuple[ChainNeuron, ...]
    block: int
    #: uid -> track, from THIS round's warrant probes (only known tracks).
    tracks: Mapping[int, str] = field(default_factory=dict)


def begin_round(conn: sqlite3.Connection, round_id: str, block: int, started_at: str) -> None:
    """Open a round in the ledger. Its committed_at stays NULL until commit_round.

    This marker row is the ONLY thing a round writes before it commits, and it
    deliberately carries no miner-visible state: readers (ScorePacketEvidence,
    the weight-setter) ignore every effect of a round whose committed_at is NULL,
    which is what makes a crashed round both invisible and detectable.
    """
    conn.execute(
        "INSERT INTO rounds (round_id, started_at, block, committed_at)"
        " VALUES (?, ?, ?, NULL)",
        (round_id, started_at, block),
    )


def commit_round(
    conn: sqlite3.Connection,
    round_id: str,
    *,
    scores: Mapping[int, float],
    decay: float,
    packets: Iterable[Mapping[str, object]] = (),
    availability_observations: Iterable[Mapping[str, object]] = (),
    committed_at: str,
    registry: RegistryUpdate | None = None,
) -> list[int]:
    """Apply ONE round's whole observable state atomically. Returns purged uids.

    Registry sync + warrant tracks + EWMA folds + score/availability evidence + the ledger stamp
    land in a single BEGIN IMMEDIATE transaction, so a crash or a concurrent
    weight-setter read can never observe ANY of a round's effects without all of them
. A round whose transaction never committed leaves committed_at NULL and
    is ignored by evidence readers.

    `registry` is optional so callers that only fold scores (tests, tools) keep
    working; the round loop always passes it.
    """
    with transaction(conn):
        purged: list[int] = []
        if registry is not None:
            purged = sync_neurons(conn, registry.neurons, registry.block)
            for uid, track in registry.tracks.items():
                record_track(conn, uid, track)
        for uid, score in scores.items():
            row = get_miner(conn, uid)
            if row is None:
                continue
            conn.execute(
                "UPDATE miners SET accumulate_score = ? WHERE uid = ?",
                (accumulate(row["accumulate_score"], score, decay), uid),
            )
        for packet in packets:
            conn.execute(
                "INSERT OR REPLACE INTO score_packets (round_id, uid, item_id,"
                " challenge_id, track, miner_hotkey, content_digest, packet_digest,"
                " packet_json, scorer_version, score, audit_ref, challenge_input_ref,"
                " miner_output_ref, reference_original_ref, miner_receipt_json,"
                " created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    round_id,
                    packet["uid"],
                    packet["item_id"],
                    packet["challenge_id"],
                    packet["track"],
                    packet["miner_hotkey"],
                    packet["content_digest"],
                    packet["packet_digest"],
                    packet["packet_json"],
                    packet["scorer_version"],
                    packet["score"],
                    packet.get("audit_ref"),
                    packet.get("challenge_input_ref"),
                    packet.get("miner_output_ref"),
                    packet.get("reference_original_ref"),
                    packet.get("miner_receipt_json"),
                    committed_at,
                ),
            )
        for observation in availability_observations:
            if float(observation.get("score", 0.0)) != 0.0:
                raise RoundLedgerError(
                    "availability evidence may only back an economic zero"
                )
            conn.execute(
                "INSERT OR REPLACE INTO availability_folds (round_id, uid, item_id,"
                " challenge_id, track, miner_hotkey, endpoint, reason, score,"
                " observation_digest, observation_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?, ?)",
                (
                    round_id,
                    observation["uid"],
                    observation["item_id"],
                    observation["challenge_id"],
                    observation["track"],
                    observation["miner_hotkey"],
                    observation["endpoint"],
                    observation["reason"],
                    observation["observation_digest"],
                    observation["observation_json"],
                    committed_at,
                ),
            )
        # Immutable post-fold state for historical epoch finalization. Record
        # exactly this round's staged registry view, including unknown-track rows,
        # in the same transaction as its scores and packet evidence.
        if registry is not None:
            for neuron in registry.neurons:
                state = get_miner(conn, neuron.uid)
                if state is None or state["hotkey"] != neuron.hotkey:
                    raise RoundLedgerError(
                        f"round {round_id!r} registry state for uid {neuron.uid} "
                        "was not applied before its historical snapshot"
                    )
                conn.execute(
                    "INSERT INTO miner_state_history (round_id, block, uid, hotkey,"
                    " coldkey, ip, track, accumulate_score, committed_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        round_id,
                        registry.block,
                        neuron.uid,
                        state["hotkey"],
                        state["coldkey"],
                        state["ip"],
                        state["track"],
                        state["accumulate_score"],
                        committed_at,
                    ),
                )
        cur = conn.execute(
            "UPDATE rounds SET committed_at = ? WHERE round_id = ? AND committed_at IS NULL",
            (committed_at, round_id),
        )
        if cur.rowcount != 1:
            raise RoundLedgerError(
                f"round {round_id!r} is not an open ledger row (already committed,"
                " or begin_round was never called) — refusing to apply its scores"
            )
    return purged


def uncommitted_rounds(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Rounds that started but never committed — the detectable partial rounds."""
    return conn.execute(
        "SELECT * FROM rounds WHERE committed_at IS NULL ORDER BY started_at"
    ).fetchall()


# -- in-flight challenge tracking ----------------------------------


def record_inflight_challenge(
    conn: sqlite3.Connection,
    *,
    challenge_id: str,
    round_id: str,
    track: str,
    fetched_at: str,
    owner: str = "",
) -> None:
    """Persist a fetched-but-unresolved challenge so a crash cannot strand it.

    `owner` is the identity this validator ACTUALLY fetched under (an internal review
    #5) — empty when it fetched anonymously (no `validator.identity`, or a
    challenge client predating the owner contract). The recovery pass resolves
    with this recorded value rather than whatever identity the process happens to
    be configured with when it restarts, because the challenge service enforces
    ownership on resolve and a rotated identity would be 403'd forever.
    """
    conn.execute(
        "INSERT OR REPLACE INTO inflight_challenges (challenge_id, round_id, track,"
        " outcome, fetched_at, owner) VALUES (?, ?, ?, 'expired', ?, ?)",
        (challenge_id, round_id, track, fetched_at, owner),
    )


def set_inflight_outcome(conn: sqlite3.Connection, challenge_id: str, outcome: str) -> None:
    """Record what this challenge should be resolved AS ('resolved' | 'expired')."""
    if outcome not in RESOLVE_OUTCOMES:
        raise ValueError(f"outcome must be one of {RESOLVE_OUTCOMES}, not {outcome!r}")
    conn.execute(
        "UPDATE inflight_challenges SET outcome = ? WHERE challenge_id = ?",
        (outcome, challenge_id),
    )


def inflight_challenges(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """The LIVE drain/recovery selection: parked rows are excluded.

    Round-4 an internal review: a row parked by a genuine ownership 403 must not be
    re-selected by every round and every restart — the refusal is permanent, so
    retrying it only rings the same alarm forever. Parked rows stay in the table
    (see `parked_challenges`) but are no longer this function's business.
    """
    return conn.execute(
        "SELECT * FROM inflight_challenges WHERE parked_at IS NULL"
        " ORDER BY fetched_at, challenge_id"
    ).fetchall()


def parked_challenges(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Rows parked by a permanent ownership refusal — an OPERATOR's worklist.

    Each one is the only record that a service-side asset is stranded `in_use`
    with its commitment unrevealed. They are surfaced (gauge + startup log) but
    never retried until explicitly unparked.
    """
    return conn.execute(
        "SELECT * FROM inflight_challenges WHERE parked_at IS NOT NULL"
        " ORDER BY parked_at, challenge_id"
    ).fetchall()


def park_inflight_challenge(
    conn: sqlite3.Connection, challenge_id: str, *, parked_at: str, reason: str
) -> None:
    """Take a row out of the drain selection, durably, with the refusal on record."""
    conn.execute(
        "UPDATE inflight_challenges SET parked_at = ?, park_reason = ?"
        " WHERE challenge_id = ?",
        (parked_at, reason, challenge_id),
    )


def unpark_inflight_challenges(conn: sqlite3.Connection) -> list[str]:
    """Return every parked row to the normal drain. Returns the unparked ids.

    The operator's explicit way out (config `validator.unpark_challenges`, or the
    validator's `unpark_challenges()` admin method) after the service-side
    ownership state has been fixed — or accepted. A row whose refusal still
    stands is simply parked again on its next 403.
    """
    ids = [str(r["challenge_id"]) for r in parked_challenges(conn)]
    conn.execute(
        "UPDATE inflight_challenges SET parked_at = NULL, park_reason = ''"
        " WHERE parked_at IS NOT NULL"
    )
    return ids


def clear_inflight_challenge(conn: sqlite3.Connection, challenge_id: str) -> None:
    conn.execute("DELETE FROM inflight_challenges WHERE challenge_id = ?", (challenge_id,))


# -- the durable scorer pin ------------------------------


def load_scorer_pin(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """The pinned scorer identity this validator's accumulators were built under."""
    return conn.execute("SELECT * FROM scorer_pin WHERE id = 1").fetchone()


def record_scorer_pin(
    conn: sqlite3.Connection, scorer_version: str, *, pinned_at: str, source: str
) -> None:
    """Take the pin. Refuses to overwrite a DIFFERENT existing pin.

    Overwriting is the whole failure mode this exists to stop: two scorers'
    packets folded into one accumulator. `clear_scorer_pin` is the only way out,
    and it is reachable only through an explicit operator acknowledgement.
    """
    identity = (scorer_version or "").strip()
    if not identity:
        raise ValueError("refusing to pin an empty scorer identity")
    row = load_scorer_pin(conn)
    if row is not None:
        if str(row["scorer_version"]) != identity:
            raise ValueError(
                f"scorer pin {row['scorer_version']!r} cannot be overwritten with"
                f" {identity!r} — clear it explicitly instead"
            )
        return
    conn.execute(
        "INSERT INTO scorer_pin (id, scorer_version, pinned_at, source)"
        " VALUES (1, ?, ?, ?)",
        (identity, pinned_at, source),
    )


def clear_scorer_pin(conn: sqlite3.Connection) -> str | None:
    """Drop the pin (operator acknowledgement). Returns what was cleared, if any."""
    row = load_scorer_pin(conn)
    conn.execute("DELETE FROM scorer_pin WHERE id = 1")
    return str(row["scorer_version"]) if row is not None else None


# (observe_retention / latest_full_window — the block-driven retention-window fold and
# lookup — were REMOVED with the retention multiplier for v1 — retention removed — owner
# decision; an internal review.)


def snapshot(
    conn: sqlite3.Connection, chain_neurons: Sequence[ChainNeuron], now: datetime
) -> list[MinerSnapshot]:
    """Map registry state onto tokenomics MinerSnapshots.

    Only miners with a recorded (known) track appear — an unknown-track miner is
    not a member of any pool and MUST NOT be defaulted into one; the round loop
    accounts for the skip separately. The -1 exclusion sentinel passes through
    `accumulate_score` untouched (rank_curve gates on it). `now` is unused today;
    it is kept for parity with build_weight_vector(now) so callers thread one
    clock through the whole composition.
    """
    del now  # reserved: parity with build_weight_vector's clock threading
    snapshots: list[MinerSnapshot] = []
    for n in sorted(chain_neurons, key=lambda x: x.uid):
        row = get_miner(conn, n.uid)
        if row is None or row["hotkey"] != n.hotkey:
            continue
        track = normalize_track(row["track"])
        if track is None:
            continue
        # (The retention-window fields — emission_window / alpha_stake_delta_window /
        # has_full_retention_window — were REMOVED from MinerSnapshot with the retention
        # multiplier for v1 — retention removed — owner decision; an internal review
        # / round-9 #4 — so the window is no longer read into the snapshot.)
        snapshots.append(
            MinerSnapshot(
                uid=n.uid,
                hotkey=n.hotkey,
                coldkey=n.coldkey,
                ip=n.ip,
                track=track,
                accumulate_score=row["accumulate_score"],
            )
        )
    return snapshots


def snapshot_at(
    conn: sqlite3.Connection,
    chain_neurons: Sequence[ChainNeuron],
    close_block: int,
    now: datetime,
) -> list[MinerSnapshot]:
    """Return miner state at the newest committed round not after ``close_block``.

    ``chain_neurons`` must itself be a close-block metagraph. A state row is
    accepted only when its uid and hotkey still match that view, preventing a
    recycled uid's previous owner from carrying into a new identity. This never
    falls back to the mutable live ``miners`` table: doing so would leak future
    rounds into a catch-up epoch.
    """
    del now  # retained for API parity with snapshot()
    if close_block < 0:
        raise ValueError(f"close_block must be non-negative, got {close_block}")

    snapshots: list[MinerSnapshot] = []
    for neuron in sorted(chain_neurons, key=lambda value: value.uid):
        row = conn.execute(
            "SELECT h.* FROM miner_state_history h"
            " JOIN rounds r ON r.round_id = h.round_id"
            " WHERE h.uid = ? AND h.block <= ? AND r.committed_at IS NOT NULL"
            " ORDER BY h.block DESC, h.committed_at DESC, h.round_id DESC LIMIT 1",
            (neuron.uid, close_block),
        ).fetchone()
        if row is None or str(row["hotkey"]) != neuron.hotkey:
            continue
        track = normalize_track(row["track"])
        if track is None:
            continue
        snapshots.append(
            MinerSnapshot(
                uid=neuron.uid,
                hotkey=neuron.hotkey,
                coldkey=neuron.coldkey,
                ip=neuron.ip,
                track=track,
                accumulate_score=float(row["accumulate_score"]),
            )
        )
    return snapshots
