"""Commit-reveal fairness for challenges (spec design spec §18, "Validator fairness").

The validator commits sha256({asset_id, dag_digest, seed, scorer_version}) BEFORE
dispatching a challenge, and may reveal the preimage only after the clean asset is
retired AND every challenge on that asset is resolved or expired — so a validator
can never cherry-pick a favorable corruption after seeing miner outputs, never
expose the seed while miners are still working, and auditors can verify the
corruption was fixed up front without the live challenge ever leaking.

Structural ordering: the `challenges` table has a NOT NULL foreign key to
`challenge_commitments`, so a challenge row cannot exist before its commitment row;
a BEFORE INSERT trigger additionally forces the challenge's (asset_id, dag_digest)
to equal the commitment's, and commit_hash is UNIQUE on challenges (one challenge
per commitment — see migrations/0002_challenge_binding.sql).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from pydantic import BaseModel, ConfigDict, Field

from vidaio.challenge.dag import (
    DAG_VERSION,
    TRACK_RULES,
    DegradationDag,
    build_dag,
    canonical_json_dumps,
    dag_rng_from_seed,
)


class RevealBeforeRetireError(RuntimeError):
    """Raised when a reveal is attempted while the clean asset is not yet retired."""


class RevealBeforeResolutionError(RuntimeError):
    """Raised when a reveal is attempted while a challenge on the asset is still
    dispatched — asset retirement alone is not sufficient to reveal."""


#: Domain-separated payload stored in the Bittensor Commitments pallet before a
#: challenge may leave the challenge service.  The pallet has one mutable slot per
#: account, so auditors verify old receipts against archive state at ``block``.
CHALLENGE_ANCHOR_DOMAIN = "vidaio.challenge.anchor.v1"


class ChallengeAnchor(BaseModel):
    """Finalized-chain receipt for one pre-dispatch challenge commitment.

    ``dispatch_ordering_key`` is the payload identifier (the generic chain anchor
    readers call it ``epoch_id``); it is already unique, durable and committed in
    the reveal preimage.  ``txid`` is operational provenance.  Security comes from
    the exact archive state at ``block`` containing ``commitment_hash`` under the
    configured authority hotkey.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    netuid: int = Field(ge=0)
    dispatch_ordering_key: int = Field(ge=1)
    commitment_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    block: int = Field(ge=0)
    #: Finalized consensus block hash. Unlike a predictable height, this cannot
    #: be placed in a miner signature before the anchor's block exists.
    block_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    txid: str | None = None

    def payload(self) -> bytes:
        payload = (
            f"{CHALLENGE_ANCHOR_DOMAIN}:{self.netuid}:"
            f"{self.dispatch_ordering_key}:{self.commitment_hash}"
        ).encode("ascii")
        if len(payload) > 128:
            raise ValueError(f"challenge anchor payload is {len(payload)} bytes (> 128)")
        return payload


def challenge_anchor_payload(
    *, netuid: int, dispatch_ordering_key: int, commitment_hash: str
) -> bytes:
    """Canonical <=128-byte chain payload for a challenge commitment."""
    return ChallengeAnchor(
        netuid=netuid,
        dispatch_ordering_key=dispatch_ordering_key,
        commitment_hash=commitment_hash,
        block=0,
    ).payload()


def record_commitment_anchor(
    conn: sqlite3.Connection,
    *,
    commit_hash: str,
    netuid: int,
    dispatch_ordering_key: int,
    block: int,
    block_hash: str | None,
    txid: str | None,
    anchored_at: str,
) -> ChallengeAnchor:
    """Append the finalized external receipt for a prepared commitment.

    The separate table is append-only and trigger-bound to the commitment's own
    ordering key.  A second identical observation is idempotent; a conflicting
    receipt is rejected rather than rewriting history.
    """
    proof = ChallengeAnchor(
        netuid=netuid,
        dispatch_ordering_key=dispatch_ordering_key,
        commitment_hash=commit_hash,
        block=block,
        block_hash=block_hash,
        txid=txid,
    )
    existing = conn.execute(
        "SELECT netuid, dispatch_ordering_key, anchor_block, anchor_block_hash,"
        " anchor_txid"
        " FROM challenge_commitment_anchors WHERE commit_hash = ?",
        (commit_hash,),
    ).fetchone()
    if existing is not None:
        current = ChallengeAnchor(
            netuid=int(existing["netuid"]),
            dispatch_ordering_key=int(existing["dispatch_ordering_key"]),
            commitment_hash=commit_hash,
            block=int(existing["anchor_block"]),
            block_hash=existing["anchor_block_hash"],
            txid=existing["anchor_txid"],
        )
        if current != proof:
            raise sqlite3.IntegrityError(
                f"challenge commitment {commit_hash} already has a different chain anchor"
            )
        return current
    conn.execute(
        "INSERT INTO challenge_commitment_anchors"
        " (commit_hash, netuid, dispatch_ordering_key, anchor_block,"
        " anchor_block_hash, anchor_txid, anchored_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            commit_hash,
            netuid,
            dispatch_ordering_key,
            block,
            block_hash,
            txid,
            anchored_at,
        ),
    )
    return proof


class ChallengeCommitment(BaseModel):
    model_config = ConfigDict(frozen=True)

    clean_asset_id: str
    dag_digest: str
    seed: int
    scorer_version: str
    #: The challenge's scoring TRACK ("compression"|"upscaling"), bound PRE-DISPATCH
    #:. Committing the track before any score exists is what
    #: lets the auditor prove the SCORE_PACKET ref's `committed_track` was not substituted
    #: at finalization to dodge recompute-ability (#9), because the track is fixed in the
    #: anchored commitment, outside the authority's finalization-time control.
    track: str
    #: The challenge's DISPATCH ORDERING KEY — a monotonic, pre-committed sequence (the
    #: dispatch counter/timestamp/block) that fixes this challenge's position in a miner's
    #: EWMA earning fold BEFORE any score exists. Because EWMA
    #: is order-dependent and this key is committed pre-scoring and independently anchored,
    #: a misreporting authority can no longer reorder scores and stamp matching sequences at
    #: finalization: the auditor re-reads this committed key and rejects any fold order
    #: that is not the challenge-committed dispatch order.
    dispatch_ordering_key: int
    commit_hash: str

    @staticmethod
    def compute_hash(
        clean_asset_id: str,
        dag_digest: str,
        seed: int,
        scorer_version: str,
        track: str,
        dispatch_ordering_key: int,
    ) -> str:
        return hashlib.sha256(
            ChallengeCommitment.preimage_payload(
                clean_asset_id, dag_digest, seed, scorer_version, track, dispatch_ordering_key
            )
        ).hexdigest()

    @staticmethod
    def preimage_payload(
        clean_asset_id: str,
        dag_digest: str,
        seed: int,
        scorer_version: str,
        track: str,
        dispatch_ordering_key: int,
    ) -> bytes:
        """The canonical commit preimage bytes: sha256 of these IS the commit hash.

        This is also the exact byte content of the audit store's DAG_REVEAL artifact —
        publishing anything else (e.g. the raw DAG JSON) would never match the
        commitment during audit recompute (see vidaio.audit.bundle.AuditBundle). The
        `track` and `dispatch_ordering_key` are bound here PRE-DISPATCH so the auditor
        reads the fold ORDER and the TRACK from the anchored commitment, never from the
        authority's finalization-time score packet.
        """
        return canonical_json_dumps(
            {
                "asset_id": clean_asset_id,
                "dag_digest": dag_digest,
                "dispatch_ordering_key": dispatch_ordering_key,
                "scorer_version": scorer_version,
                "seed": seed,
                "track": track,
            }
        ).encode()

    def preimage_bytes(self) -> bytes:
        return self.preimage_payload(
            self.clean_asset_id,
            self.dag_digest,
            self.seed,
            self.scorer_version,
            self.track,
            self.dispatch_ordering_key,
        )

    @staticmethod
    def committed_dispatch_from_preimage(preimage_bytes: bytes) -> tuple[str, int] | None:
        """Read the committed `(track, dispatch_ordering_key)` out of a DAG_REVEAL preimage.

        The auditor calls this on the item's fetched DAG_REVEAL bytes (whose sha256 IS the
        anchored commit hash) to recover the CHALLENGE-COMMITTED fold order + track without
        trusting the finalization-time score packet. Returns None when the bytes do not
        carry a well-formed track + integral ordering key (an un-auditable/legacy reveal).
        """
        try:
            doc = json.loads(preimage_bytes)
        except (ValueError, TypeError):
            return None
        if not isinstance(doc, dict):
            return None
        track = doc.get("track")
        key = doc.get("dispatch_ordering_key")
        if not isinstance(track, str) or track == "":
            return None
        if not isinstance(key, int) or isinstance(key, bool):
            return None
        return (track, key)

    @classmethod
    def create(
        cls,
        clean_asset_id: str,
        dag: DegradationDag,
        seed: int,
        scorer_version: str,
        track: str,
        dispatch_ordering_key: int = 0,
    ) -> "ChallengeCommitment":
        dag_digest = dag.canonical_digest()
        return cls(
            clean_asset_id=clean_asset_id,
            dag_digest=dag_digest,
            seed=seed,
            scorer_version=scorer_version,
            track=track,
            dispatch_ordering_key=dispatch_ordering_key,
            commit_hash=cls.compute_hash(
                clean_asset_id, dag_digest, seed, scorer_version, track, dispatch_ordering_key
            ),
        )


class RevealedCommitment(BaseModel):
    model_config = ConfigDict(frozen=True)

    clean_asset_id: str
    dag_digest: str
    seed: int
    scorer_version: str
    #: The pre-dispatch committed track + fold-order key,
    #: carried through the reveal so verify_reveal recomputes the SAME commit hash the
    #: preimage bytes bind and the auditor reads the committed order/track back.
    track: str
    dispatch_ordering_key: int
    commit_hash: str
    revealed_at: str


def record_commitment(
    conn: sqlite3.Connection, commitment: ChallengeCommitment, committed_at: str
) -> None:
    """Persist a commitment. Must happen before the challenge row (FK-enforced)."""
    conn.execute(
        "INSERT INTO challenge_commitments"
        " (commit_hash, clean_asset_id, dag_digest, seed, scorer_version, track,"
        "  dispatch_ordering_key, committed_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            commitment.commit_hash,
            commitment.clean_asset_id,
            commitment.dag_digest,
            str(commitment.seed),
            commitment.scorer_version,
            commitment.track,
            commitment.dispatch_ordering_key,
            committed_at,
        ),
    )


def reveal_commitment(
    conn: sqlite3.Connection, commit_hash: str, revealed_at: str
) -> RevealedCommitment:
    """Reveal a commitment's preimage.

    Allowed only once the clean asset is retired AND no challenge referencing that
    asset is still dispatched — a (force-)retired asset with miners still working
    its challenge must not have its seed/DAG exposed.
    """
    row = conn.execute(
        "SELECT c.*, a.status AS asset_status"
        " FROM challenge_commitments c JOIN assets a ON a.id = c.clean_asset_id"
        " WHERE c.commit_hash = ?",
        (commit_hash,),
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown commitment {commit_hash}")
    if row["asset_status"] != "retired":
        raise RevealBeforeRetireError(
            f"asset {row['clean_asset_id']} is {row['asset_status']}, not retired;"
            " reveal is forbidden while the challenge item is live"
        )
    unresolved = conn.execute(
        "SELECT COUNT(*) AS n FROM challenges WHERE asset_id = ? AND status = 'dispatched'",
        (row["clean_asset_id"],),
    ).fetchone()["n"]
    if unresolved:
        raise RevealBeforeResolutionError(
            f"{unresolved} challenge(s) on asset {row['clean_asset_id']} still dispatched;"
            " reveal is forbidden while miners may still be working"
        )
    if row["revealed_at"] is None:
        conn.execute(
            "UPDATE challenge_commitments SET revealed_at = ? WHERE commit_hash = ?",
            (revealed_at, commit_hash),
        )
        at = revealed_at
    else:
        at = row["revealed_at"]  # idempotent: keep the first reveal timestamp
    return RevealedCommitment(
        clean_asset_id=row["clean_asset_id"],
        dag_digest=row["dag_digest"],
        seed=int(row["seed"]),
        scorer_version=row["scorer_version"],
        track=row["track"],
        dispatch_ordering_key=int(row["dispatch_ordering_key"]),
        commit_hash=row["commit_hash"],
        revealed_at=at,
    )


def verify_reveal(revealed: RevealedCommitment) -> bool:
    """Recompute the commit hash from the revealed preimage and compare."""
    return (
        ChallengeCommitment.compute_hash(
            revealed.clean_asset_id,
            revealed.dag_digest,
            revealed.seed,
            revealed.scorer_version,
            revealed.track,
            revealed.dispatch_ordering_key,
        )
        == revealed.commit_hash
    )


def verify_reveal_deep(revealed: RevealedCommitment, dag_version: int = DAG_VERSION) -> bool:
    """Deep reveal check: the committed DAG must actually generate from the seed.

    Beyond the hash check, rebuild the DAG from the revealed seed via the sanctioned
    derived-key path (dag_rng_from_seed + build_dag) for each known track and require
    a rebuilt canonical digest equal to the revealed dag_digest. This is the check the
    audit layer injects to prove the dispatched corruption was seed-determined rather
    than hand-picked and merely hashed into the commitment.
    """
    if not verify_reveal(revealed):
        return False
    # The track is now committed pre-dispatch: rebuild the
    # DAG for the COMMITTED track specifically. A commitment whose seed does not
    # regenerate the committed DAG for its own committed track is hand-picked, not
    # seed-determined. Unknown committed track => cannot regenerate => fail closed.
    rule = TRACK_RULES.get(revealed.track)
    if rule is None:
        return False
    dag = build_dag(revealed.track, dag_rng_from_seed(revealed.seed), dag_version=dag_version)
    return dag.canonical_digest() == revealed.dag_digest


def deep_reveal_verifier(dag_bytes: bytes) -> bool:
    """The reveal verifier the AUDIT layer injects (``Auditor.reveal_verifier``).

    Parses a DAG_REVEAL artifact's bytes — the challenge commitment's canonical preimage JSON
    (``{asset_id, dag_digest, seed, scorer_version, track, dispatch_ordering_key}``) — back into
    a :class:`RevealedCommitment` and runs :func:`verify_reveal_deep`: the preimage must hash to
    the committed commit hash AND the committed DAG must genuinely REGENERATE from the revealed
    seed for its committed track. Anything unparseable / non-canonical fails CLOSED (returns
    False; a raise is itself a REVEAL_INVALID finding in ``verify_bundle``). This is the SAME deep
    check the integration/e2e harnesses inject; it lives here so the production report-mode auditor
    loop can wire a REAL reveal verifier instead of running with none: a
    ``verify_bundle`` at ``media_sample_rate > 0`` with no reveal verifier records a STRICT SKIP
    that must NOT wash to PASS.
    """
    doc = json.loads(dag_bytes)
    revealed = RevealedCommitment(
        clean_asset_id=doc["asset_id"],
        dag_digest=doc["dag_digest"],
        seed=doc["seed"],
        scorer_version=doc["scorer_version"],
        track=doc["track"],
        dispatch_ordering_key=doc["dispatch_ordering_key"],
        # The canonical preimage bytes hash to the commit hash by construction; any
        # non-canonical re-serialization changes the hash and fails verify_reveal.
        commit_hash=hashlib.sha256(dag_bytes).hexdigest(),
        revealed_at="",
    )
    return verify_reveal_deep(revealed)
