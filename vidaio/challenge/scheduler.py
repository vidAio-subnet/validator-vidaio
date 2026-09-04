"""Per-cycle challenge production + the content-ingestion admin's backend contract.

`make_challenge` turns (track, checked-out asset, private seed) into a Challenge:
a private DAG + commitment, and a miner-facing dispatch payload that carries ONLY
the degraded-input reference and the public task type. The payload is structurally
checked against leaks (seed, DAG params, clean-asset identity, ground-truth digest)
before the challenge is returned — never trust convention where an assert is cheap.

`register_asset` is ingest-lite: the backend half of the "just find & upload videos"
admin (spec design spec §18). It performs the pure parts — near-duplicate check, split
assignment, provenance log entries, pool insert — and emits command PLANS for the
side-effectful parts (fetch, pristine transcode + metadata strip, segmentation).
Plans are recorded as *_planned provenance only and the asset enters the pool as
'ingesting'; `confirm_ingest_step` records the completion facts after the executor
ran them, and only when ALL planned steps are confirmed does the asset flip to
'fresh' (the sole checkoutable status — an unconfirmed/unstripped asset can never
be issued). Nothing here touches the network or spawns processes.
"""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from vidaio.challenge.commitment import ChallengeCommitment, record_commitment
from vidaio.challenge.config import ChallengeConfig
from vidaio.challenge.dag import (
    DAG_VERSION,
    DegradationDag,
    build_dag,
    dag_rng_from_seed,
    seed_to_bytes,
)
from vidaio.challenge.pool import (
    Asset,
    FingerprintIndex,
    add_asset,
    append_provenance,
    assign_split,
    check_near_duplicate,
    get_asset,
)

# Minimum private-seed entropy accepted by make_challenge. Anything smaller (dates,
# counters, timestamps) is brute-forceable from public dispatch material.
MIN_SEED_BITS = 128


class PayloadLeakError(RuntimeError):
    """Raised when a dispatch payload would leak private challenge material."""


class WeakSeedError(ValueError):
    """Raised when a private seed carries too little entropy to stay private."""


class ChallengeIntegrityError(RuntimeError):
    """Raised when a Challenge object is internally inconsistent: its DAG does not
    hash to the dag_digest its commitment binds (or asset/commit identity
    disagrees). Such an object must never be persisted."""


_ALLOWED_PAYLOAD_KEYS = {"challenge_id", "task_type", "input_ref"}

# Secrets shorter than this are not substring-probed: a short raw value (e.g. a
# single-digit int) collides with legitimate payload bytes constantly and proves
# nothing. Real secrets here (seeds >= 128 bits, sha256 digests, asset ids, DAG
# JSON) are always comfortably above it; entropy itself is enforced by
# make_challenge's WeakSeedError, not by this structural probe.
_MIN_LEAK_PROBE_LEN = 16


class DispatchPayload(BaseModel):
    """Exactly what the miner sees. Nothing else may ever be added casually —
    every new field must survive `_assert_payload_clean`."""

    model_config = ConfigDict(frozen=True)

    challenge_id: str
    task_type: str  # public by design: routing is legitimate
    input_ref: str  # where the degraded input will be served from; derived from challenge_id only


class Challenge(BaseModel):
    model_config = ConfigDict(frozen=True)

    challenge_id: str
    track: str
    asset_id: str  # validator-private
    dag: DegradationDag  # validator-private
    commitment: ChallengeCommitment
    dispatch: DispatchPayload


def _assert_payload_clean(challenge: Challenge, asset: Asset) -> None:
    payload = challenge.dispatch.model_dump(mode="json")
    extra = set(payload) - _ALLOWED_PAYLOAD_KEYS
    if extra:
        raise PayloadLeakError(
            f"dispatch payload has unexpected fields: {sorted(extra)}"
        )
    text = challenge.dispatch.model_dump_json()
    forbidden = {
        "seed": str(challenge.commitment.seed),
        "asset_id": asset.id,
        "clean_content_digest": asset.content_digest,
        "source_url": asset.source_url,
        "dag_digest": challenge.commitment.dag_digest,
        "dag_json": challenge.dag.canonical_json(),
    }
    for label, value in forbidden.items():
        if value and len(value) >= _MIN_LEAK_PROBE_LEN and value in text:
            raise PayloadLeakError(f"dispatch payload leaks {label}")


def make_challenge(
    track: str,
    asset: Asset,
    private_seed: int,
    scorer_version: str,
    *,
    dag_version: int = DAG_VERSION,
    min_seed_bits: int = MIN_SEED_BITS,
    dispatch_ordering_key: int = 0,
) -> Challenge:
    """Produce one challenge item. Deterministic in (track, asset, private_seed).

    `private_seed` MUST come from a CSPRNG at the call site (secrets.randbits(256)
    or better); seeds below `min_seed_bits` are rejected with WeakSeedError.

    `dispatch_ordering_key` is the monotonic, pre-committed sequence (the dispatch
    counter/block) that fixes this challenge's position in a miner's EWMA earning fold
    BEFORE any score exists. It is bound into the commitment
    here — pre-dispatch — so the fold order is outside the authority's finalization-time
    control; the caller (the dispatcher) supplies a strictly increasing value.

    Derivations (fixed forever within a dag_version):
      challenge_id: uuid4 from sha256(b"challenge-id" || seed_bytes || asset_id)
      DAG rng:      MT seeded from sha256(b"dag" || seed_bytes)  (dag_rng_from_seed)
    The bare seed never seeds the Mersenne Twister and no MT output is ever public,
    so the public challenge_id cannot be brute-forced back to the private DAG stream.
    """
    if private_seed.bit_length() < min_seed_bits:
        raise WeakSeedError(
            f"private_seed has only {private_seed.bit_length()} bits;"
            f" >= {min_seed_bits} required — draw seeds from a CSPRNG (secrets.randbits)"
        )
    seed_bytes = seed_to_bytes(private_seed)
    challenge_id = str(
        uuid.UUID(
            bytes=hashlib.sha256(
                b"challenge-id" + seed_bytes + asset.id.encode()
            ).digest()[:16],
            version=4,
        )
    )
    dag = build_dag(track, dag_rng_from_seed(private_seed), dag_version=dag_version)
    commitment = ChallengeCommitment.create(
        asset.id, dag, private_seed, scorer_version, track, dispatch_ordering_key
    )
    challenge = Challenge(
        challenge_id=challenge_id,
        track=track,
        asset_id=asset.id,
        dag=dag,
        commitment=commitment,
        dispatch=DispatchPayload(
            challenge_id=challenge_id,
            task_type=track,
            input_ref=f"challenges/{challenge_id}/input.mp4",
        ),
    )
    _assert_payload_clean(challenge, asset)
    return challenge


def _verify_challenge_integrity(challenge: Challenge) -> None:
    """Require the Challenge to be internally consistent BEFORE anything persists:
    sha256-canonical-digest(challenge.dag) == commitment.dag_digest (recomputed from
    the DAG object, never trusted from the field), the commitment binds this
    challenge's asset, and the commit hash actually hashes its own preimage."""
    recomputed = challenge.dag.canonical_digest()
    if recomputed != challenge.commitment.dag_digest:
        raise ChallengeIntegrityError(
            f"challenge {challenge.challenge_id}: DAG canonical digest {recomputed}"
            f" != committed dag_digest {challenge.commitment.dag_digest}"
        )
    if challenge.asset_id != challenge.commitment.clean_asset_id:
        raise ChallengeIntegrityError(
            f"challenge {challenge.challenge_id}: asset_id {challenge.asset_id}"
            f" != commitment's clean_asset_id {challenge.commitment.clean_asset_id}"
        )
    if challenge.track != challenge.commitment.track:
        raise ChallengeIntegrityError(
            f"challenge {challenge.challenge_id}: track {challenge.track!r}"
            f" != commitment's committed track {challenge.commitment.track!r}"
        )
    expected_hash = ChallengeCommitment.compute_hash(
        challenge.commitment.clean_asset_id,
        challenge.commitment.dag_digest,
        challenge.commitment.seed,
        challenge.commitment.scorer_version,
        challenge.commitment.track,
        challenge.commitment.dispatch_ordering_key,
    )
    if expected_hash != challenge.commitment.commit_hash:
        raise ChallengeIntegrityError(
            f"challenge {challenge.challenge_id}: commit_hash does not hash the"
            " commitment's own preimage"
        )


def record_challenge(
    conn: sqlite3.Connection, challenge: Challenge, created_at: str
) -> None:
    """Persist commitment first (commit-before-dispatch), then the challenge row.

    The Challenge object is integrity-checked first (ChallengeIntegrityError on any
    DAG/digest/commitment mismatch — an inconsistent object never persists, not even
    its commitment row). The row starts in status 'dispatched'; asset_id/dag_digest
    are trigger-checked against the commitment, commit_hash is UNIQUE (no commitment
    reuse), and identity columns are frozen by a BEFORE UPDATE trigger.
    """
    _verify_challenge_integrity(challenge)
    record_commitment(conn, challenge.commitment, created_at)
    conn.execute(
        "INSERT INTO challenges"
        " (challenge_id, track, asset_id, commit_hash, dag_digest, dag_json,"
        " dag_version, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'dispatched', ?)",
        (
            challenge.challenge_id,
            challenge.track,
            challenge.asset_id,
            challenge.commitment.commit_hash,
            challenge.commitment.dag_digest,
            challenge.dag.canonical_json(),
            challenge.dag.dag_version,
            created_at,
        ),
    )


def resolve_challenge(
    conn: sqlite3.Connection,
    challenge_id: str,
    resolved_at: str,
    *,
    outcome: str = "resolved",
) -> None:
    """Terminate a dispatched challenge: scoring finished -> 'resolved', deadline
    passed without completion -> 'expired'. Revealing the underlying commitment is
    blocked until every challenge on the asset reaches a terminal status."""
    if outcome not in ("resolved", "expired"):
        raise ValueError(f"outcome must be 'resolved' or 'expired', not {outcome!r}")
    cur = conn.execute(
        "UPDATE challenges SET status = ?, resolved_at = ?"
        " WHERE challenge_id = ? AND status = 'dispatched'",
        (outcome, resolved_at, challenge_id),
    )
    if cur.rowcount == 1:
        return
    row = conn.execute(
        "SELECT status FROM challenges WHERE challenge_id = ?", (challenge_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown challenge {challenge_id}")
    raise ValueError(f"challenge {challenge_id} is already {row['status']}")


# --- ingest-lite: the content-ingestion admin's backend contract ----------------------


class IngestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    asset: Asset
    fetch_plan: list[str]  # argv: bring the raw source local
    transcode_plan: list[str]  # argv: pristine FFV1 reference + metadata strip
    segment_plan: list[str]  # argv: cut the reference into challenge clips


def _build_plans(
    source_url: str, work_dir: str, cfg: ChallengeConfig
) -> tuple[list[str], list[str], list[str]]:
    raw = f"{work_dir}/raw.bin"
    if "youtube.com" in source_url or "youtu.be" in source_url:
        fetch = [
            "yt-dlp",
            "--no-part",
            "--output",
            f"{work_dir}/raw.%(ext)s",
            source_url,
        ]
        raw = f"{work_dir}/raw.mp4"
    else:
        fetch = ["curl", "-L", "--fail", "-o", raw, source_url]
    pristine = f"{work_dir}/pristine.mkv"
    transcode = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        raw,
        "-map_metadata",
        "-1",  # leakage control: strip filenames/IDs/codec tags
        "-map_chapters",
        "-1",
        "-c:v",
        "ffv1",
        "-an",
        pristine,
    ]
    segment = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        pristine,
        "-f",
        "segment",
        "-segment_time",
        str(cfg.max_clip_seconds),
        "-reset_timestamps",
        "1",
        "-c",
        "copy",
        f"{work_dir}/clip_%04d.mkv",
    ]
    return fetch, transcode, segment


def register_asset(
    conn: sqlite3.Connection,
    cfg: ChallengeConfig,
    *,
    source_url: str,
    license_basis: str,
    creator: str,
    source: str,
    subject: str = "",
    scene: str = "",
    content_digest: str,
    perceptual_fingerprint: str,
    resolution_tag: str,
    motion_tag: str,
    content_type_tag: str,
    ingested_at: str,
    duplicate_index: FingerprintIndex | None = None,
    work_dir: str = "./ingest",
    extra_provenance: Mapping[str, Any] | None = None,
) -> IngestResult:
    """Register one pristine asset into the rotating challenge pool.

    Pure contract: digest and perceptual fingerprint are computed by the (injected)
    media backend and passed in; this function decides admission (near-duplicate
    check), assigns the source-group split, writes the pool row + provenance log,
    and returns the command plans the executor must run.

    The asset is inserted as 'ingesting' with ALL planned steps recorded
    (fetch_planned, transcode_planned, segment_planned); it becomes 'fresh' — and
    thereby checkoutable — only when confirm_ingest_step has confirmed every one.
    """
    check_near_duplicate(duplicate_index, perceptual_fingerprint)

    fields = {"creator": creator, "source": source, "subject": subject, "scene": scene}
    split = assign_split(fields, cfg)
    asset = Asset(
        id=f"asset_{content_digest[:16]}",
        content_digest=content_digest,
        perceptual_fingerprint=perceptual_fingerprint,
        source_url=source_url,
        license_basis=license_basis,
        ingest_date=ingested_at,
        creator=creator,
        source=source,
        subject=subject,
        scene=scene,
        resolution_tag=resolution_tag,
        motion_tag=motion_tag,
        content_type_tag=content_type_tag,
        # The transcode PLAN includes the strip step, but nothing has run yet:
        # the flag flips only via confirm_ingest_step("transcode") after execution.
        metadata_stripped=False,
        split=split,
        # Not issuable yet: flips to 'fresh' only when confirm_ingest_step has
        # confirmed every planned step (fetch, transcode, segment).
        status="ingesting",
        use_count=0,
    )
    fetch, transcode, segment = _build_plans(source_url, work_dir, cfg)
    add_asset(conn, asset)
    append_provenance(
        conn,
        asset.id,
        "ingested",
        {
            "source_url": source_url,
            "license_basis": license_basis,
            **(dict(extra_provenance) if extra_provenance else {}),
        },
        ingested_at,
    )
    append_provenance(
        conn, asset.id, "fetch_planned", {"via": "fetch_plan"}, ingested_at
    )
    append_provenance(
        conn,
        asset.id,
        "transcode_planned",
        {"via": "transcode_plan", "metadata_stripped": False},
        ingested_at,
    )
    append_provenance(
        conn, asset.id, "segment_planned", {"via": "segment_plan"}, ingested_at
    )
    append_provenance(
        conn,
        asset.id,
        "fingerprinted",
        {"perceptual_fingerprint": perceptual_fingerprint},
        ingested_at,
    )
    append_provenance(conn, asset.id, "split_assigned", {"split": split}, ingested_at)
    return IngestResult(
        asset=asset, fetch_plan=fetch, transcode_plan=transcode, segment_plan=segment
    )


_INGEST_STEPS = ("fetch", "transcode", "segment")


def confirm_ingest_step(
    conn: sqlite3.Connection,
    asset_id: str,
    step: str,
    confirmed_at: str,
    *,
    detail: Mapping[str, Any] | None = None,
) -> None:
    """Record that the executor actually completed an ingest plan step.

    register_asset records PLANS only; completion facts — including the
    metadata_stripped flag, which the transcode plan's strip step implies — enter
    the record exclusively here, after the caller ran the plan and verified it.
    Each step confirms at most once. When the LAST of the three steps lands, the
    asset flips ingesting -> fresh (guarded UPDATE) and becomes checkoutable;
    until then it stays 'ingesting' and checkout_asset can never issue it.

    The whole confirm — duplicate check, completion append, metadata_stripped
    flip, final status flip — runs in ONE BEGIN IMMEDIATE transaction: either
    every fact of this confirmation persists or none does. A crash between the
    event append and the status flip is therefore unobservable (the append rolls
    back with it), and a retry after any failure starts from a clean slate.
    Duplicates are also rejected inside SQLite itself (partial UNIQUE index on
    provenance confirmation events), so a concurrent confirm that slips past the
    Python read still fails at the SQL layer, inside the same transaction.
    """
    if step not in _INGEST_STEPS:
        raise ValueError(f"unknown ingest step {step!r}; known: {_INGEST_STEPS}")
    completed_events = tuple(f"{s}_completed" for s in _INGEST_STEPS)
    conn.execute("BEGIN IMMEDIATE")
    try:
        get_asset(conn, asset_id)  # existence check
        marks = ", ".join("?" for _ in completed_events)
        already = {
            row["event"]
            for row in conn.execute(
                f"SELECT DISTINCT event FROM provenance_log"
                f" WHERE asset_id = ? AND event IN ({marks})",
                (asset_id, *completed_events),
            )
        }
        if f"{step}_completed" in already:
            raise ValueError(
                f"ingest step {step!r} already confirmed for asset {asset_id}"
            )
        try:
            append_provenance(
                conn, asset_id, f"{step}_completed", dict(detail or {}), confirmed_at
            )
        except sqlite3.IntegrityError as exc:
            # The idx_provenance_ingest_confirm_once UNIQUE index caught a
            # duplicate the read above could not see (e.g. a confirm committed by
            # another connection after our SELECT). Same contract, SQL-enforced.
            raise ValueError(
                f"ingest step {step!r} already confirmed for asset {asset_id}"
            ) from exc
        if step == "transcode":
            conn.execute(
                "UPDATE assets SET metadata_stripped = 1 WHERE id = ?", (asset_id,)
            )
            append_provenance(
                conn,
                asset_id,
                "metadata_stripped",
                {"via": "transcode_plan"},
                confirmed_at,
            )
        if already | {f"{step}_completed"} == set(completed_events):
            # Last confirmation: the asset is fully ingested (fetched, transcoded +
            # stripped, segmented) and may now be issued. Guarded so only an
            # 'ingesting' row flips — never a resurrection of in_use/retired.
            flipped = conn.execute(
                "UPDATE assets SET status = 'fresh' WHERE id = ? AND status = 'ingesting'",
                (asset_id,),
            )
            if flipped.rowcount == 1:
                append_provenance(
                    conn,
                    asset_id,
                    "ingest_confirmed",
                    {"steps": list(_INGEST_STEPS)},
                    confirmed_at,
                )
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
