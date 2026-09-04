"""On-chain commitment payloads, merkle proofs, and the append-only local ledger.

Implements the design spec §15 review fix "Anchor commitments on chain": today the
manifest/baseline/dataset/param digests live only in SQLite — publicly auditable
but not on-chain-auditable. Here we build the exact payload bytes to anchor:

- BEFORE enrollment: a competition commitment over {manifest digest, baseline tree
  digest, baseline image digest, dataset-selection seed commitment, reward-param
  digest}. The chain payload is a single domain-tagged sha256 root over the
  canonical JSON (well under 128 bytes); the JSON itself is kept off-chain as
  a store artifact so the root is always openable.
- AFTER evaluation: a publication record over {merkle root of all per-item
  score-packet digests, weight-vector digest} — with inclusion proofs so any
  third party can check that a given score packet fed the published weights,
  and that no packet was injected outside the committed set.

Chain submission itself is a later-phase adapter; this module produces the
payload bytes and records every commitment in an append-only SQLite ledger
(UPDATE/DELETE are blocked by triggers; status changes are appended events).

Merkle construction (documented so third parties can reimplement it):
- leaves = the sha256 digests (32 raw bytes each), sorted ascending as bytes,
  duplicates kept — the root is independent of input ordering;
- leaf hash = sha256(0x00 || leaf); inner node = sha256(0x01 || left || right)
  (domain separation prevents leaf/node second-preimage splices);
- odd node at any level is promoted unchanged to the next level.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN, canonical_json_bytes, sha256_hex
from vidaio.core.db import apply_migrations, connect
from vidaio.tokenomics.breakthrough import PODIUM_SPLIT
from vidaio.tokenomics.config import TokenomicsConfig

if TYPE_CHECKING:
    from vidaio.audit.store import AuditStore

#: Versioned domain tag — bump on any change to the canonical-JSON contract.
COMMITMENT_DOMAIN = "vidaio.commitment.v2"

#: Versioned pre-enrollment economic-policy document. This is distinct from the
#: outer commitment domain because changing an otherwise hard-coded reward rule
#: must change ``reward_param_digest`` even when TokenomicsConfig is unchanged.
REWARD_POLICY_DOMAIN = "vidaio.reward-policy.v2"

#: The only commitment kinds the ledger accepts (mirrored by the SQL CHECK).
ALLOWED_COMMITMENT_KINDS = frozenset({"competition", "publication"})

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_HexField = Field(pattern=SHA256_HEX_PATTERN)
MAX_COMPETITION_COMMITMENT_BYTES = 16 * 1024


class LedgerIntegrityError(ValueError):
    """A caller-supplied CommitmentPayload is internally inconsistent."""


def pin_git_sha(sha: str) -> str:
    """THE canonical adapter from a git object id (sha1 or sha256 repo, any case)
    to the sha256 hex digest this module's _HexField commitments require.

    Defined once here so every caller pinning e.g. ArchivedBaseline.tree_sha produces
    the same commitment bytes: sha256 over the ascii of the lowercased hex string.
    """
    normalized = sha.strip().lower()
    if not normalized or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"not a hex git object id: {sha!r}")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


class CompetitionCommitment(BaseModel):
    """Pre-enrollment commitment: pins the whole competition before anyone enrolls."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_digest: str = _HexField
    baseline_version: int = Field(ge=0)
    baseline_artifact_digest: str = _HexField
    baseline_provenance_digest: str = _HexField
    baseline_tree_digest: str = _HexField
    baseline_image_digest: str = _HexField
    dataset_selection_seed_commitment: str = _HexField
    reward_param_digest: str = _HexField


class PublicationRecord(BaseModel):
    """Post-evaluation publication: pins packets, weights, and their epoch inputs.

    ``snapshot_digest`` is nullable only for the legacy/local report path, which has
    no shared epoch snapshot. Production shared publications require and persist it
    before submitting weights, so the record identifies the exact anchored EpochLog
    from which both the packet set and vector were derived.
    """

    model_config = ConfigDict(frozen=True)

    score_packet_merkle_root: str = _HexField
    weight_vector_digest: str = _HexField
    snapshot_digest: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)


class CommitmentPayload(BaseModel):
    """What goes on chain (payload) + what goes in the store (canonical_json)."""

    model_config = ConfigDict(frozen=True)

    kind: str  # "competition" | "publication"
    root: str = _HexField  # sha256 over canonical_json
    payload: bytes  # domain-tagged bytes to anchor on chain (<= 128 bytes)
    canonical_json: bytes  # keep as a store artifact (kind=manifest)


def _build_payload(kind: str, doc: dict[str, Any]) -> CommitmentPayload:
    canonical = canonical_json_bytes(doc)
    root = sha256_hex(canonical)
    payload = f"{COMMITMENT_DOMAIN}:{kind}:{root}".encode("ascii")
    assert len(payload) <= 128, "chain payload must stay <= 128 bytes"
    return CommitmentPayload(kind=kind, root=root, payload=payload, canonical_json=canonical)


def build_competition_commitment(commitment: CompetitionCommitment) -> CommitmentPayload:
    """Payload to anchor on chain BEFORE enrollment opens."""
    return _build_payload("competition", commitment.model_dump(mode="json"))


def reward_parameter_digest(config: TokenomicsConfig) -> str:
    """SHA-256 of the exact canonical tokenomics policy used for emissions."""
    policy = {
        "domain": REWARD_POLICY_DOMAIN,
        "tokenomics": config.model_dump(mode="json"),
        "competition_podium_split": list(PODIUM_SPLIT),
    }
    return sha256_hex(canonical_json_bytes(policy))


def load_competition_commitment(
    store: "AuditStore", root: str
) -> CompetitionCommitment:
    """Open, bound-read, canonicalize, and verify a commitment root preimage."""
    from vidaio.audit.store import ArtifactKind

    raw = store.get_digest_limited(
        ArtifactKind.MANIFEST,
        root,
        max_bytes=MAX_COMPETITION_COMMITMENT_BYTES,
    )
    commitment = CompetitionCommitment.model_validate_json(raw)
    if canonical_json_bytes(commitment.model_dump(mode="json")) != raw:
        raise LedgerIntegrityError(
            "competition commitment preimage is not canonical JSON"
        )
    if build_competition_commitment(commitment).root != root:
        raise LedgerIntegrityError(
            "competition commitment preimage does not open the supplied root"
        )
    return commitment


def build_publication_record(record: PublicationRecord) -> CommitmentPayload:
    """Payload to anchor/publish AFTER evaluation, alongside signed packets."""
    return _build_payload("publication", record.model_dump(mode="json"))


# ---- merkle tree over score-packet digests -----------------------------------


def _leaf_hash(leaf: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + leaf).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _sorted_leaves(leaf_digests: Iterable[str]) -> list[bytes]:
    leaves = sorted(bytes.fromhex(d) for d in leaf_digests)
    if not leaves:
        raise ValueError("merkle tree requires at least one leaf")
    return leaves


def _levels(leaves: list[bytes]) -> list[list[bytes]]:
    levels = [[_leaf_hash(x) for x in leaves]]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt = [_node_hash(cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
        if len(cur) % 2:
            nxt.append(cur[-1])  # odd node promoted unchanged
        levels.append(nxt)
    return levels


def merkle_root(leaf_digests: Iterable[str]) -> str:
    """Root (hex) over the sorted leaf digests. Order-independent by design."""
    return _levels(_sorted_leaves(leaf_digests))[-1][0].hex()


def merkle_proof(leaf_digests: Iterable[str], target_digest: str) -> list[tuple[str, str]]:
    """Inclusion proof for target_digest: [(sibling_hex, "left"|"right"), ...].

    The side names where the SIBLING sits relative to the running hash.
    """
    leaves = _sorted_leaves(leaf_digests)
    try:
        idx = leaves.index(bytes.fromhex(target_digest))
    except ValueError:
        raise ValueError(f"digest {target_digest} is not a leaf of this tree") from None
    proof: list[tuple[str, str]] = []
    for level in _levels(leaves)[:-1]:
        if idx == len(level) - 1 and len(level) % 2:
            idx //= 2  # promoted odd node: no sibling at this level
            continue
        sibling = idx ^ 1
        proof.append((level[sibling].hex(), "left" if sibling < idx else "right"))
        idx //= 2
    return proof


def verify_merkle_proof(
    leaf_digest: str, proof: Sequence[tuple[str, str]], root: str
) -> bool:
    """Check that leaf_digest is included under root via proof."""
    h = _leaf_hash(bytes.fromhex(leaf_digest))
    for sibling_hex, side in proof:
        sibling = bytes.fromhex(sibling_hex)
        if side == "left":
            h = _node_hash(sibling, h)
        elif side == "right":
            h = _node_hash(h, sibling)
        else:
            return False
    return h.hex() == root


# ---- append-only local ledger ------------------------------------------------


class CommitmentStatus(StrEnum):
    PENDING_CHAIN = "pending_chain"
    ANCHORED = "anchored"
    PUBLISHED = "published"


_STATUS_ORDER = {
    CommitmentStatus.PENDING_CHAIN: 0,
    CommitmentStatus.ANCHORED: 1,
    CommitmentStatus.PUBLISHED: 2,
}


def _utc_instant(value: str, field: str) -> datetime:
    """Parse a caller-supplied ISO-8601 timestamp as an INSTANT.

    The timestamp MUST be timezone-aware (an explicit UTC offset or 'Z'):
    naive timestamps are ambiguous instants and are rejected outright, and
    comparing raw strings would let an offset disguise a backdate (e.g.
    '09:00+05:00' is 04:00Z, EARLIER than '08:00+00:00' despite sorting
    later as a string). Returns the datetime converted to UTC.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field} {value!r} is not a parseable ISO-8601 timestamp"
        ) from None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(
            f"{field} {value!r} is timezone-naive — ledger timestamps must"
            " carry an explicit UTC offset (e.g. 2026-08-20T12:00:00+00:00)"
        )
    return parsed.astimezone(timezone.utc)


def _stored_instant(value: str) -> datetime:
    """Parse a timestamp read back FROM the ledger as a UTC instant.

    Rows written by this module are always canonical UTC ('+00:00'). A naive
    value can only appear via direct SQL that slipped past the triggers; it
    is interpreted as UTC — exactly how SQLite's own date functions (and
    therefore the in-database triggers) treat it — so both layers agree.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class CommitmentLedger:
    """Append-only record of every commitment ever produced.

    Rows are immutable (SQLite triggers abort UPDATE/DELETE); a commitment's
    status advances only forward and only ONE step at a time (pending_chain ->
    anchored -> published; jumping pending_chain -> published is rejected: a
    commitment cannot claim publication without having been anchored), and
    each advance is a NEW appended event row — history is never rewritten.
    Status timestamps are monotonic AS INSTANTS: every event's recorded_at
    must be >= the ledger row's created_at and >= the previous event's
    recorded_at, compared after normalizing to UTC — an ISO-8601 offset
    cannot disguise a backdate. Caller-supplied timestamps must be
    timezone-aware ISO-8601 (naive values are rejected) and are normalized
    to canonical UTC ('+00:00') before being stored, so the persisted
    strings and the instants they denote always agree. All of these
    invariants are enforced in-database too (triggers compare via
    julianday(), which parses '+HH:MM' offsets), not just here.
    `record()` re-derives the payload's internal relationships (root over
    canonical JSON, domain-tagged bytes, allowed kind) before insert, and
    writes the ledger row plus its initial pending_chain status in ONE
    transaction (BEGIN IMMEDIATE) — a crash between the two can never leave a
    commitment without status history. All timestamps are supplied by the
    caller.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, db_path: str | Path) -> "CommitmentLedger":
        conn = connect(db_path)
        apply_migrations(conn, MIGRATIONS_DIR)
        return cls(conn)

    @staticmethod
    def _validate_payload(payload: CommitmentPayload) -> None:
        """Recompute the payload's internal relationships; raise if tampered."""
        if payload.kind not in ALLOWED_COMMITMENT_KINDS:
            raise LedgerIntegrityError(
                f"commitment kind {payload.kind!r} is not one of "
                f"{sorted(ALLOWED_COMMITMENT_KINDS)}"
            )
        expected_root = sha256_hex(payload.canonical_json)
        if payload.root != expected_root:
            raise LedgerIntegrityError(
                f"payload root {payload.root} != sha256(canonical_json) {expected_root}"
                " — root does not commit to the supplied document"
            )
        expected_bytes = f"{COMMITMENT_DOMAIN}:{payload.kind}:{payload.root}".encode("ascii")
        if payload.payload != expected_bytes:
            raise LedgerIntegrityError(
                "payload bytes do not match the domain-tagged form"
                f" {expected_bytes!r} for kind={payload.kind!r}"
            )

    def record(self, payload: CommitmentPayload, created_at: str) -> int:
        """Append a commitment (initial status pending_chain). Returns its id.

        Raises LedgerIntegrityError when the payload's kind, root, and bytes
        are not mutually consistent — the ledger never trusts the caller.
        `created_at` must be a timezone-aware ISO-8601 timestamp; it is
        normalized to canonical UTC ('+00:00') before being stored.
        The ledger row and its initial status row are written atomically in
        one BEGIN IMMEDIATE transaction: either both exist or neither does.
        """
        self._validate_payload(payload)
        created_at = _utc_instant(created_at, "created_at").isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                "INSERT INTO commitment_ledger"
                " (kind, root_digest, payload, canonical_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    payload.kind,
                    payload.root,
                    payload.payload,
                    payload.canonical_json.decode("utf-8"),
                    created_at,
                ),
            )
            commitment_id = int(cur.lastrowid)  # type: ignore[arg-type]
            self._conn.execute(
                "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
                " VALUES (?, ?, ?)",
                (commitment_id, CommitmentStatus.PENDING_CHAIN.value, created_at),
            )
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")
        return commitment_id

    def advance(self, commitment_id: int, status: CommitmentStatus, at: str) -> None:
        """Append a forward-only, single-step, time-monotonic status event.

        pending_chain -> anchored -> published, one step per call: skipping
        anchored is rejected. `at` must be a timezone-aware ISO-8601
        timestamp whose INSTANT is >= the previous event's and >= the ledger
        row's created_at (comparison is in UTC — an offset cannot disguise a
        backdate); it is stored normalized to canonical UTC ('+00:00').
        """
        at_instant = _utc_instant(at, "status timestamp")
        row = self._conn.execute(
            "SELECT status, recorded_at FROM commitment_ledger_status"
            " WHERE commitment_id = ? ORDER BY id DESC LIMIT 1",
            (commitment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown commitment id {commitment_id}")
        current = CommitmentStatus(row["status"])
        if _STATUS_ORDER[status] <= _STATUS_ORDER[current]:
            raise ValueError(
                f"commitment {commitment_id} status can only advance forward "
                f"({current.value} -> {status.value} rejected)"
            )
        if _STATUS_ORDER[status] != _STATUS_ORDER[current] + 1:
            raise ValueError(
                f"commitment {commitment_id} status cannot skip a stage "
                f"({current.value} -> {status.value} rejected; statuses advance"
                " one step at a time)"
            )
        if at_instant < _stored_instant(row["recorded_at"]):
            raise ValueError(
                f"commitment {commitment_id} status timestamp {at!r} precedes"
                f" the previous status timestamp {row['recorded_at']!r} —"
                " timestamps must be monotonically non-decreasing as instants"
            )
        created_row = self._conn.execute(
            "SELECT created_at FROM commitment_ledger WHERE id = ?",
            (commitment_id,),
        ).fetchone()
        if created_row is not None and at_instant < _stored_instant(
            created_row["created_at"]
        ):
            raise ValueError(
                f"commitment {commitment_id} status timestamp {at!r} precedes"
                f" the ledger row's created_at {created_row['created_at']!r} —"
                " timestamps must be monotonically non-decreasing as instants"
            )
        self._conn.execute(
            "INSERT INTO commitment_ledger_status (commitment_id, status, recorded_at)"
            " VALUES (?, ?, ?)",
            (commitment_id, status.value, at_instant.isoformat()),
        )

    def current_status(self, commitment_id: int) -> CommitmentStatus:
        row = self._conn.execute(
            "SELECT status FROM commitment_ledger_status WHERE commitment_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (commitment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown commitment id {commitment_id}")
        return CommitmentStatus(row["status"])

    def history(self, commitment_id: int) -> list[tuple[str, str]]:
        """[(status, recorded_at), ...] oldest first."""
        return [
            (row["status"], row["recorded_at"])
            for row in self._conn.execute(
                "SELECT status, recorded_at FROM commitment_ledger_status"
                " WHERE commitment_id = ? ORDER BY id",
                (commitment_id,),
            )
        ]

    def get(self, commitment_id: int) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT id, kind, root_digest, payload, canonical_json, created_at"
            " FROM commitment_ledger WHERE id = ?",
            (commitment_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown commitment id {commitment_id}")
        return dict(row)
