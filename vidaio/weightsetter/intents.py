"""Weight-submission intent ledger — durability across a non-idempotent write.

review service-review #10. `ChainAdapter.set_weights` is a chain WRITE with no
idempotency key: if the extrinsic lands but the response is lost, a retry is
tempo-rejected and, before this module existed, the whole attempt was recorded as
a failure even though the chain weights had changed. Worse, publication only
started AFTER a successful submit, so a crash in between left an accepted vector
permanently unaudited — and the "pending anchors are re-drivable" note in the
publication path had nothing that actually re-drove them.

The fix is an intent record written BEFORE the first `set_weights` call:

    record_intent(...)                      -> 'pending'
    mark_accepted(..., resolution=...)      -> 'accepted'   (publication owed)
    attach_commitment(...)                  ->  ledger id pinned to the intent
    mark_published(...)                     -> 'published'
    mark_abandoned(..., resolution=...)     -> 'abandoned'

Everything downstream (the stored WEIGHT_VECTOR artifact, the PublicationRecord,
the anchor) is driven FROM the row, so `WeightSetter.reconcile()` can finish any
half-done attempt on the next startup or the next loop iteration.

`resolution` is deliberately explicit about HOW an acceptance was established
(`chain_accepted`, `chain_confirmed`, `tempo_after_ambiguous`), so an inferred
reconciliation is never silently indistinguishable from a directly-observed one.

Round-2 an internal review: `abandoned` is a claim that the chain does NOT hold this
vector, and it is terminal — the vector is never published, so if the claim is
wrong an accepted weight set stays unaudited forever. It may therefore only be
reached from a POSITIVE denial (`mark_abandoned`); an intent whose fate is
unknown stays `pending` indefinitely and is re-checked by every reconciliation
pass (`note_check`). "We could not find out" is not evidence of absence.

Round-3 an internal review: an intent is only ever settled as `accepted` against ITS OWN
vector. `quantize_weights` / `weights_match` / `vector_fingerprint` (below) put a
chain-reported vector and this intent's vector on the same u16 grid so the
question "did THIS vector land?" can actually be answered — the old test ("our
last_update advanced past the attempt block") answered a different question and
published vectors that had never landed.

Round-2 review new-6: `packets_frozen_at` is the instant an intent's packet-digest
set was captured. It — not `settled_at` — is the lower bound of the NEXT
publication's evidence window, so packets created while an accepted intent waits
for a failed anchor still belong to a publication.

Schema lives in vidaio/weightsetter/migrations/0002_weight_intents.sql through
0007_publication_snapshot_epoch.sql, applied by crown_store.migrate() alongside
the crown tables. In a shared authority intent, JSON ``null`` packet digests mean
"post-submit resolution still owed", never "the epoch had no packets"; the durable
snapshot epoch id/digest make that resolution exact and crash-recoverable.
Migration 0010 adds durable publication retry reservations and reveal-wait log claims.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Mapping, Sequence

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.tokenomics.quantize import max_normalize_u16

STATE_PENDING = "pending"
STATE_ACCEPTED = "accepted"
STATE_PUBLISHED = "published"
STATE_ABANDONED = "abandoned"

#: States a reconciliation pass must still act on.
UNSETTLED_STATES = (STATE_PENDING, STATE_ACCEPTED)


def vector_json(weights: Mapping[int, float]) -> str:
    """Canonical JSON of a weight vector: uid keys as strings, sorted."""
    return canonical_json_bytes(
        {str(uid): weights[uid] for uid in sorted(weights)}
    ).decode("utf-8")


# -- recognising OUR vector when it comes back off the chain --
#
# `vector_digest` above is the digest of the EXACT floats we submitted; it can
# never match a vector read back from a chain that stores u16 integers. The
# comparison below is the one that can: both sides are put on the chain's own
# grid first, so "is this my vector?" survives the round trip.

#: The chain's weight grid: bittensor MAX-normalizes a submitted vector onto
#: u16 (largest weight -> 65535).
WEIGHT_QUANTIZATION_SCALE = 65535

#: Tolerated per-uid difference, in u16 steps, when matching a chain-reported
#: vector against ours. Requantizing a vector that was quantized once is exact
#: in principle, but the chain may report its u16s renormalized to floats (by
#: sum, by max, by anything positive), and one rounding step must not turn OUR
#: OWN vector into somebody else's — a mismatch is evidence used to DENY.
WEIGHT_MATCH_TOLERANCE_STEPS = 1


def quantize_weights(weights: Mapping[int, float]) -> dict[int, int]:
    """Put a weight vector on the pinned SDK's emitted u16 grid.

    Non-positive entries are dropped (they carry no emission and the chain does
    not record them), the rest are max-normalized so the largest weight is
    exactly WEIGHT_QUANTIZATION_SCALE. An adapter may report raw u16s,
    sum-normalized floats or the untouched submission and all three compare equal
    under :data:`WEIGHT_MATCH_TOLERANCE_STEPS`. Exact canonical bytes can differ by
    one step under positive rescaling because Bittensor first casts inputs to
    binary32; that is why semantic comparison is tolerant. Entries that round to
    0 are dropped, matching a chain that cannot express a smaller weight.
    """
    return max_normalize_u16({int(uid): float(w) for uid, w in weights.items()})


def vector_fingerprint(weights: Mapping[int, float]) -> str:
    """Identity of a vector AS THE CHAIN WOULD HOLD IT (quantized digest).

    An audit/log LABEL only — it names a vector compactly in CRITICAL abandon
    records and diagnostics. It preserves the exact pinned-SDK binary32 boundary,
    so mathematically proportional float inputs can have distinct fingerprints
    when their emitted u16 bytes differ by one step. It is deliberately NOT used
    to assess ambiguity:
    ambiguity ("could another intent be the author of the chain's vector?") is
    decided by `weights_match`, the SAME tolerance-based equivalence that ties a
    chain report to an intent's own vector. Round-4 an internal review: using exact
    fingerprint equality for twin detection while the match tolerates one u16
    step let a tolerance-near LATER vector land, match an EARLIER intent, and
    escape twin detection — confirming and publishing a vector that never
    landed. One question, one relation: wherever "is this vector that vector?"
    is asked, `weights_match` answers it.
    """
    body = canonical_json_bytes(
        {str(uid): q for uid, q in sorted(quantize_weights(weights).items())}
    )
    return sha256_hex(body)


def weights_match(
    chain_weights: Mapping[int, float], intent_weights: Mapping[int, float]
) -> bool:
    """Is the vector the chain reports OUR vector?

    Same uid set, and every uid within WEIGHT_MATCH_TOLERANCE_STEPS on the u16
    grid. Two EMPTY vectors never match: an empty chain report means "no
    positive weights recorded", which is not evidence that our (never empty)
    vector landed.

    THE one equivalence relation: every place that asks "is
    this vector that vector?" — matching a chain report to an intent AND
    deciding whether another intent could equally be the report's author — must
    use this function, tolerance included. A stricter rule anywhere ambiguity is
    assessed turns a tolerance-near neighbour into a false confirmation.
    """
    reported = quantize_weights(chain_weights)
    ours = quantize_weights(intent_weights)
    if not reported or not ours or reported.keys() != ours.keys():
        return False
    return all(
        abs(reported[uid] - ours[uid]) <= WEIGHT_MATCH_TOLERANCE_STEPS for uid in ours
    )


def load_vector(row: sqlite3.Row) -> dict[int, float]:
    return {int(uid): float(w) for uid, w in json.loads(row["vector_json"]).items()}


def load_packet_digests(row: sqlite3.Row) -> list[str]:
    payload = json.loads(row["packet_digests_json"])
    if payload is None:
        raise ValueError(
            "weight intent publication inputs are unresolved; refusing to publish "
            "the empty-packet sentinel"
        )
    if not isinstance(payload, list):
        raise ValueError("weight intent packet_digests_json is not a JSON array")
    return [str(d) for d in payload]


def packet_digests_resolved(row: sqlite3.Row) -> bool:
    """Whether publication packet leaves were durably captured before submission."""
    return json.loads(row["packet_digests_json"]) is not None


def attach_packet_digests(
    conn: sqlite3.Connection, intent_id: int, *, packet_digests: Sequence[str]
) -> None:
    """Resolve a post-submit publication obligation without ever inventing empty.

    ``null`` means the pre-submit best-effort capture failed but the authority vector
    was still submitted. A later publication worker may fill it exactly once. Repeats
    with the same list are idempotent; a different list is a hard conflict.
    """
    encoded = json.dumps(sorted(str(d) for d in packet_digests))
    row = get_intent(conn, intent_id)
    current = json.loads(row["packet_digests_json"])
    if current is not None:
        if json.dumps(sorted(str(d) for d in current)) != encoded:
            raise ValueError(
                f"weight intent {intent_id} packet-digest resolution conflicts with "
                "its durable publication inputs"
            )
        return
    conn.execute(
        "UPDATE weight_intents SET packet_digests_json = ? "
        "WHERE id = ? AND packet_digests_json = 'null'",
        (encoded, intent_id),
    )


def record_intent(
    conn: sqlite3.Connection,
    *,
    created_at: str,
    attempt_block: int,
    version_key: int,
    weights: Mapping[int, float],
    packet_digests: Sequence[str] | None,
    packets_frozen_at: str | None = None,
    snapshot_digest: str | None = None,
    snapshot_epoch_id: int | None = None,
) -> int:
    """Persist WHAT is about to be submitted, BEFORE submitting it. Returns its id.

    `packets_frozen_at` is when `packet_digests` was captured;
    it defaults to `created_at`, which is the same instant on the normal path.
    `snapshot_digest` and `snapshot_epoch_id` freeze the verified shared EpochLog
    identity in the SAME pre-write row as the vector. If the best-effort packet-ref
    copy failed, ``packet_digests=None`` persists JSON ``null`` so crash recovery
    must resolve that exact epoch rather than publish an empty-set sentinel or a
    later provider snapshot.
    """
    body = vector_json(weights)
    cur = conn.execute(
        "INSERT INTO weight_intents (created_at, attempt_block, version_key,"
        " vector_json, vector_digest, packet_digests_json, state, packets_frozen_at,"
        " snapshot_digest, snapshot_epoch_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            created_at,
            attempt_block,
            version_key,
            body,
            sha256_hex(body.encode("utf-8")),
            (
                "null"
                if packet_digests is None
                else json.dumps(sorted(str(d) for d in packet_digests))
            ),
            STATE_PENDING,
            packets_frozen_at or created_at,
            snapshot_digest,
            snapshot_epoch_id,
        ),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def mark_accepted(
    conn: sqlite3.Connection, intent_id: int, *, accepted_block: int, resolution: str
) -> None:
    """The chain holds this vector. Publication is now owed and re-drivable."""
    conn.execute(
        "UPDATE weight_intents SET state = ?, accepted_block = ?, resolution = ?"
        " WHERE id = ?",
        (STATE_ACCEPTED, accepted_block, resolution, intent_id),
    )


def accept_with_vector(
    conn: sqlite3.Connection,
    intent_id: int,
    *,
    accepted_block: int,
    resolution: str,
    weights: Mapping[int, float] | None,
) -> None:
    """Mark ACCEPTED and rewrite the stored vector to what LANDED — ATOMICALLY.

    review round-5 #4. The connection is autocommit (isolation_level=None, see
    vidaio/core/db.py:24), so calling mark_accepted() and reconcile_vector() in
    sequence is TWO separate commits. A crash BETWEEN them leaves an ACCEPTED
    intent still carrying its pre-quantization FLOAT vector, and startup
    reconciliation then publishes/anchors that float verbatim — a vector the chain
    never held (even when it is merely scale-equivalent). Acceptance and the
    submitted-vector rewrite must therefore land together or not at all.

    When `weights` is falsy (NO submitted vector was reported — the block-only /
    recovery-bookkeeping case), there is nothing to rewrite and this degrades to a
    plain `mark_accepted`: the stored vector is left EXACTLY as-is, and the caller
    must have already established (via `weights_match`) that the chain holds this
    intent's own vector. Never substitute a vector here.

    `BEGIN IMMEDIATE ... COMMIT` (with ROLLBACK on any exception) mirrors the
    transactional migration runner in vidaio/core/db.py apply_migrations — the one
    place in this codebase that already batches multiple statements atomically over
    the autocommit connection.
    """
    if not weights:
        mark_accepted(
            conn, intent_id, accepted_block=accepted_block, resolution=resolution
        )
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "UPDATE weight_intents SET state = ?, accepted_block = ?, resolution = ?"
            " WHERE id = ?",
            (STATE_ACCEPTED, accepted_block, resolution, intent_id),
        )
        reconcile_vector(conn, intent_id, weights=weights)
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def reconcile_vector(
    conn: sqlite3.Connection, intent_id: int, *, weights: Mapping[int, float]
) -> None:
    """Rewrite an ACCEPTED intent's stored vector to what ACTUALLY landed on chain.

    Round-3 an internal review / round-4 #3. The chain adapter reports the EXACT max-grid u16
    vector emitted by the pinned SDK (`SetWeightsResult.submitted`) — the chain stores
    those u16 pairs. The live adapter rejects target churn before writing and therefore
    preserves the authority uid set; this rewrite still captures the SDK's max-grid
    representation. Publication and anchoring
    read THIS durable row, so it must carry the SUBMITTED u16 vector; otherwise the
    anchored, auditable document would describe the pre-quantization float intent — a
    vector the chain never held, even when it is merely scale-equivalent. Re-derives
    `vector_json` + `vector_digest` from the submitted weights so every downstream
    read (publication, `_chain_evidence`, twin detection) sees chain state.

    Called after EVERY chain acceptance that reported a vector — the direct path
    (the scale-equivalent common case) AND,
    since an internal review, the RECOVERY path when the confirming read carried the
    exact reported vector. Both go through `accept_with_vector` so the state change
    and this rewrite share one commit; a recovery confirmation that could NOT read a
    vector back (block bookkeeping only) skips the rewrite and leaves the intent's
    OWN vector, which the match already proved the chain holds.

    Round-6 an internal review: this is the SINGLE choke point at which the durable row is
    reconciled to the accepted vector, so it is where the ONE canonical representation
    is imposed. `SubmittedWeights.weights` / `SetWeightsResult.submitted` are
    permitted to be raw u16, sum-normalized floats, or the untouched submission
    (vidaio/chain/adapter.py:73), and the two paths read from DIFFERENT surfaces:
    the direct path passes `SetWeightsResult.submitted` (u16), the recovery path
    passes the `submitted_weights()` readback (which InMemoryChain returns as the raw
    floats, and the live SDK returns as the runtime u16). Persisting either verbatim
    let the SAME accepted write yield DIFFERENT `vector_json`/`vector_digest` bytes
    depending on which path ran — breaking "anchored == chain state, deterministically"
    and validator convergence. `max_normalize_u16` is the deterministic canonicaliser:
    it mirrors Bittensor 10.5's last emit conversion, so it maps EVERY permitted phrasing of
    the same vector — raw u16, sum-normalized float, any positive rescaling — onto the
    identical grid, and is idempotent on a vector already on the grid. Applying it here
    makes the persisted/published/anchored bytes byte-identical on BOTH paths,
    adapter-agnostic (the adapter's own return representation is never touched). The
    stored `vector_json`/`vector_digest` are therefore always derived from the
    canonical max-u16 runtime grid (uid keys, sorted, integer values), regardless of caller.
    """
    body = vector_json(
        max_normalize_u16({int(uid): float(w) for uid, w in weights.items()})
    )
    conn.execute(
        "UPDATE weight_intents SET vector_json = ?, vector_digest = ? WHERE id = ?",
        (body, sha256_hex(body.encode("utf-8")), intent_id),
    )


def attach_commitment(
    conn: sqlite3.Connection, intent_id: int, commitment_id: int
) -> None:
    """Pin the CommitmentLedger row so a failed anchor is re-driven, not duplicated."""
    conn.execute(
        "UPDATE weight_intents SET commitment_id = ? WHERE id = ?",
        (commitment_id, intent_id),
    )


def mark_published(
    conn: sqlite3.Connection, intent_id: int, *, at: str, resolution: str | None = None
) -> None:
    conn.execute(
        "UPDATE weight_intents SET state = ?, settled_at = ?,"
        " resolution = COALESCE(?, resolution) WHERE id = ?",
        (STATE_PUBLISHED, at, resolution, intent_id),
    )


def mark_abandoned(
    conn: sqlite3.Connection, intent_id: int, *, at: str, resolution: str
) -> None:
    """This attempt PROVABLY did not change the chain.

    Terminal: an abandoned intent is never published. It may therefore only be
    reached from a positive denial — a synchronous chain rejection, or a fresh
    post-write snapshot that positively does not carry our weights. An intent
    whose fate is merely unknown stays pending (`note_check`) forever instead.
    """
    conn.execute(
        "UPDATE weight_intents SET state = ?, settled_at = ?, resolution = ? WHERE id = ?",
        (STATE_ABANDONED, at, resolution, intent_id),
    )


def note_check(
    conn: sqlite3.Connection, intent_id: int, *, at: str, verdict: str
) -> None:
    """Record what the chain said about a still-PENDING intent, without settling it.

    The audit trail behind "this vector's fate is still unknown": how recently we
    asked and what came back. State is deliberately untouched.
    """
    conn.execute(
        "UPDATE weight_intents SET last_checked_at = ?, last_check = ? WHERE id = ?",
        (at, verdict, intent_id),
    )


def note_commit_reveal_pending(conn: sqlite3.Connection, intent_id: int) -> None:
    """Remember a CR obligation without claiming an active vector or acceptance.

    Also used for legacy NULL-cause rows after a positive CR-mode read; that mode
    alone cannot prove their commitment landed, so only vector proof settles them.
    """
    conn.execute(
        "UPDATE weight_intents SET resolution = 'commit_reveal_pending' "
        "WHERE id = ? AND state = ?",
        (intent_id, STATE_PENDING),
    )


def get_intent(conn: sqlite3.Connection, intent_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM weight_intents WHERE id = ?", (intent_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown weight intent {intent_id}")
    return row


def latest_snapshot_epoch_id(conn: sqlite3.Connection) -> int | None:
    """Highest authenticated authority epoch ever admitted to an intent.

    Every shared intent is written only after the latest pointer has passed the
    provider's digest/anchor/boundary checks.  Keeping the maximum across *all*
    states (including a synchronously rejected or still-pending chain write)
    therefore forms a durable non-regression floor: a transiently stale authority
    and RPC view cannot make this validator submit an older, still-valid vector
    after restart.  Equality is intentionally allowed for normal same-epoch
    retries/resubmissions.
    """
    row = conn.execute(
        "SELECT MAX(snapshot_epoch_id) AS epoch_id FROM weight_intents "
        "WHERE snapshot_epoch_id IS NOT NULL"
    ).fetchone()
    if row is None or row["epoch_id"] is None:
        return None
    epoch_id = int(row["epoch_id"])
    if epoch_id < 1:
        raise ValueError(
            f"weight_intents contains an invalid snapshot_epoch_id floor {epoch_id}"
        )
    return epoch_id


def unsettled_intents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Intents a reconciliation pass must finish, oldest first."""
    marks = ",".join("?" * len(UNSETTLED_STATES))
    return conn.execute(
        f"SELECT * FROM weight_intents WHERE state IN ({marks}) ORDER BY id",
        UNSETTLED_STATES,
    ).fetchall()


def pending_reveals(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Known finalized CR commitments that still need exact-vector confirmation."""
    return conn.execute(
        "SELECT * FROM weight_intents WHERE state = ? "
        "AND resolution = 'commit_reveal_pending' ORDER BY id",
        (STATE_PENDING,),
    ).fetchall()


def claim_reveal_wait_log(conn: sqlite3.Connection, intent_id: int, *, at: str) -> bool:
    """Claim the one reveal-wait log durably, including across restarts."""
    return conn.execute(
        "UPDATE weight_intents SET reveal_wait_logged_at = ? "
        "WHERE id = ? AND reveal_wait_logged_at IS NULL",
        (at, intent_id),
    ).rowcount == 1


def publication_retry_at(conn: sqlite3.Connection, *, interval: float) -> float:
    """UTC epoch seconds when both backoff and the rolling budget permit a start."""
    rows = conn.execute(
        "SELECT started_at, retry_after FROM weight_publication_attempts "
        "ORDER BY id DESC LIMIT 3"
    ).fetchall()
    if not rows:
        return 0.0
    return max(
        float(rows[0]["retry_after"]),
        float(rows[2]["started_at"]) + interval if len(rows) == 3 else 0.0,
    )


def publication_retry_delay(*, failures: int, base_delay: float, interval: float) -> float:
    return min(interval, base_delay * 2 ** min(failures - 1, 16))


def reserve_publication_attempt(
    conn: sqlite3.Connection, intent_id: int, *, now: float, interval: float,
    base_delay: float, timeout: float,
) -> int | None:
    """Reserve before launch atomically; an interrupted start still spends budget."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        if now < publication_retry_at(conn, interval=interval):
            conn.execute("COMMIT")
            return None
        latest = conn.execute(
            "SELECT failure_count, succeeded FROM weight_publication_attempts "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        failures = (
            int(latest["failure_count"]) + 1
            if latest is not None and latest["succeeded"] != 1 else 1
        )
        delay = publication_retry_delay(
            failures=failures, base_delay=base_delay, interval=interval
        )
        cursor = conn.execute(
            "INSERT INTO weight_publication_attempts "
            "(intent_id, started_at, retry_after, failure_count) VALUES (?, ?, ?, ?)",
            (intent_id, now, now + timeout + delay, failures),
        )
        attempt_id = int(cursor.lastrowid)
        conn.execute("COMMIT")
        return attempt_id
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def finish_publication_attempt(
    conn: sqlite3.Connection, attempt_id: int, *, now: float, succeeded: bool,
    base_delay: float, interval: float,
) -> None:
    row = conn.execute(
        "SELECT failure_count FROM weight_publication_attempts WHERE id = ?",
        (attempt_id,),
    ).fetchone()
    delay = publication_retry_delay(
        failures=int(row["failure_count"]), base_delay=base_delay, interval=interval
    )
    conn.execute(
        "UPDATE weight_publication_attempts SET finished_at = ?, succeeded = ?, "
        "retry_after = ? WHERE id = ?",
        (now, int(succeeded), 0.0 if succeeded else now + delay, attempt_id),
    )


def other_intents(
    conn: sqlite3.Connection, *, exclude_id: int | None, max_attempt_block: int | None
) -> list[sqlite3.Row]:
    """Every OTHER intent that could have authored a vector recorded on chain.

    "Could have authored" = attempted no later than the block at which the chain
    says its current vector was recorded (all of them when that block is
    unknown). Used to spot an intent carrying an IDENTICAL vector, which makes
    "whose write is this?" unanswerable — see WeightSetter._identical_twin_exists.

    Every state is considered, including settled ones: an intent that already
    published its vector is precisely the kind of author that must not be allowed
    to confirm a second, earlier intent for free.
    """
    sql = "SELECT * FROM weight_intents WHERE 1 = 1"
    args: list[object] = []
    if exclude_id is not None:
        sql += " AND id != ?"
        args.append(exclude_id)
    if max_attempt_block is not None:
        sql += " AND attempt_block <= ?"
        args.append(max_attempt_block)
    return conn.execute(sql + " ORDER BY id", args).fetchall()


def publication_watermark(conn: sqlite3.Connection) -> str | None:
    """The lower bound of the NEXT publication's evidence window.

    The latest instant at which an ALREADY-PUBLISHED intent froze its packet list
    — not when it settled. A publication commits to the packets it captured; the
    next one must therefore start where that capture ended, or every packet
    created while a slow/failed anchor was retried belongs to no publication at
    all. `settled_at` is the pre-fix fallback for rows written before the column
    existed (a slightly LATER bound, i.e. the old behaviour, only for them).

    Overlap between consecutive windows is harmless (a digest appearing in two
    merkle sets proves the same packet twice); a gap is not, which is why the
    earlier of the two candidate bounds is the safe one.
    """
    row = conn.execute(
        "SELECT COALESCE(packets_frozen_at, settled_at) AS watermark"
        " FROM weight_intents WHERE state = ?"
        " AND COALESCE(packets_frozen_at, settled_at) IS NOT NULL"
        " ORDER BY watermark DESC LIMIT 1",
        (STATE_PUBLISHED,),
    ).fetchone()
    return str(row["watermark"]) if row is not None else None


def intents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every intent ever recorded, oldest first (audit/inspection)."""
    return conn.execute("SELECT * FROM weight_intents ORDER BY id").fetchall()
