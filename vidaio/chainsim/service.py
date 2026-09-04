"""ChainSim — the simulated chain every service talks to in report mode.

This is the DEFAULT runtime chain (the project design record rule 8): competitions
register contenders here, the weight-setter submits vectors here (same tempo
semantics as vidaio.chain.InMemoryChain), publications anchor here — and the
outcome is a report (JSON + markdown, vidaio/chainsim/report.py) of scores and
weight vectors instead of a real chain push. Real-chain mode is a separate,
explicit opt-in behind the same ChainAdapter Protocol (vidaio.chain.factory).

Simulation model (deliberately simple; every simplification documented here):

- Blocks — lazy wall-clock production, no background task:
  block = 1 + floor(max(0, now - start_time) / block_seconds) + advance_offset.
  `start_time` and `advance_offset` persist in SQLite, so a restart resumes the
  block clock. POST /advance bumps the offset for deterministic tests (inject a
  frozen `now` and drive blocks entirely via /advance).

- Registration — uids assigned sequentially from 0. Re-registering a known
  hotkey is idempotent: same uid, fields updated (alpha_stake only when the
  request carries one). A NEW hotkey taking over an EXISTING uid slot (real
  chain deregistration/recycling) is NOT modeled — every new hotkey gets a
  fresh uid; model hotkey churn by registering a new neuron.

- Emission — each block mints `emission_per_block`, distributed proportionally
  to the LAST recorded weight vector: a vector recorded at block B directs the
  emission of blocks strictly AFTER B, until the next recorded vector takes
  over. Blocks with no prior vector, and vector share pointing at unregistered
  uids, are undistributed ("burned" — reported in /state). Crediting is lazy
  but purely a function of (call history, credited watermark, current block),
  so WHEN settlement runs never changes the outcome. GET /neurons reports
  ChainNeuron.emission as the CURRENT per-block rate under the latest vector
  (vidaio.validator.miner_manager treats emission as a rate at observation
  time); cumulative credited emission lives in /state and the report.

- Weights — tempo-gated per validator hotkey exactly like InMemoryChain:
  a call fails with "tempo gate: too soon" while block <= last_accepted + tempo;
  rejected calls are not recorded. Multi-validator stake-weighted consensus is
  NOT modeled: the last recorded vector (any hotkey) IS the emission director.
  Every accepted vector is kept, and `GET /weights/{hotkey}` reads the latest
  one back (vector + the block it was recorded at, or a null vector when that
  hotkey has never set weights) — the evidence a validator needs to prove that
  ITS OWN vector landed before publishing it.

- Alpha stake — static: whatever registration (or re-registration) set. There
  are no stake dynamics; move stake by re-registering with a new alpha_stake
  (this is the lever that feeds alpha_stake_delta into retention windows).

Authorization — the sim's stand-in for the real chain's signature model:

  On the real chain a mutation is an extrinsic SIGNED by a hotkey's keypair, and
  the runtime authorizes it by (a) who signed it and (b) what that neuron is
  permitted to do (only registered validators can set weights; only the sudo/
  owner key can touch chain parameters). The sim has no keypairs, so it uses a
  bearer token as the signature analogue:

    token  ~  a signature by that hotkey     (proves "I am this neuron")
    operator token  ~  the node/sudo key     (proves "I run this chain")

  - Every identity gets a token at registration: the server generates one (or
    adopts a client-supplied `auth_token`, so a local fleet can share one
    dev secret from config) and returns it ONCE in the /register response. Only
    its SHA-256 is stored (`neurons.token_sha256`) — the sim can verify a token
    but cannot hand it back out.
  - Participant mutations (/weights, /anchor) require
    `Authorization: Bearer <token>` matching the hotkey NAMED IN THE BODY, so a
    caller can only act as itself. /weights additionally requires that hotkey's
    role to be "validator" (a miner submitting weights gets 403) — the runtime's
    validator-permit check.
  - Operator powers (/advance, /reset) require the OPERATOR token: producing
    blocks and destroying history are node powers, not participant powers.
    /report/write accepts either (any registered identity, or the operator).
  - Re-registration of a claimed hotkey requires that hotkey's token — 409 with
    no credential, 403 with a wrong one — so nobody can take over an identity by
    re-registering it (the real chain's "you don't have the key" equivalent).
  - Reads (/healthz, /neurons, /weights/{hotkey}, /state, /report) stay OPEN:
    the sim's whole point is an observable local chain, and real chain state is
    public too (a hotkey's weights are a plain storage query on the real chain).

  The REAL adapter will sign extrinsics with a wallet instead; no service code
  changes — vidaio.chain.HttpChainAdapter carries the token exactly where the
  bittensor adapter will carry the keypair.

Health — two checks, both of which must be able to say NO:

  sim_db    opens its OWN short-lived connection per check (health runs on the
            HealthServer thread, and a check that shares the service connection
            reports on the wrong thing — see _db_ok).
  http_api  False once the uvicorn task exits without a stop being requested.
            A bind failure otherwise leaves a "healthy" process with no API.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import secrets
import sqlite3
from collections import defaultdict
from pathlib import Path
from time import time as wall_time
from typing import Any, Callable, Literal

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from prometheus_client import Counter
from pydantic import BaseModel, Field, field_validator

from vidaio.core import apply_migrations, section
from vidaio.chain.adapter import synthetic_block_hash
from vidaio.chainsim.config import ChainSimConfig
from vidaio.chainsim.report import build_report, write_report
from vidaio.services.base import BaseService

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

#: Filename under `chainsim.report_dir` holding a self-generated operator token.
OPERATOR_TOKEN_FILE = "operator-token.txt"

#: Typed authorization failures (the `error` field of every 401/403 detail).
AUTH_MISSING = "auth_token_missing"
AUTH_INVALID = "auth_token_invalid"
AUTH_UNKNOWN_HOTKEY = "auth_unknown_hotkey"
AUTH_WRONG_ROLE = "auth_wrong_role"
AUTH_HOTKEY_CLAIMED = "auth_hotkey_claimed"


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_matches(presented: str, stored_hash: str) -> bool:
    """Constant-time compare of sha256(presented) against a stored hash."""
    if not stored_hash:
        return False  # an unclaimed identity is never authenticated
    return hmac.compare_digest(_hash_token(presented), stored_hash)


def _bearer(authorization: str | None) -> str | None:
    """Extract `Authorization: Bearer <token>`; None if absent or malformed."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def _auth_error(status: int, error: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": error, "message": message})


def _connect(path: str | Path) -> sqlite3.Connection:
    """core.connect settings + check_same_thread=False.

    FastAPI/uvicorn may execute handlers (and the health check runs on the
    HealthServer thread) off the constructing thread; access is effectively
    serialized by the single event loop, but the connection must not refuse
    other threads outright.
    """
    p = Path(path)
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ---- wire models ---------------------------------------------------------------


class RegisterRequest(BaseModel):
    hotkey: str = Field(min_length=1)
    coldkey: str = ""
    ip: str = ""
    role: Literal["miner", "validator"] = "miner"
    #: omit to keep the existing stake (0.0 for a brand-new registration)
    alpha_stake: float | None = Field(default=None, ge=0)
    #: Optional client-chosen secret for this hotkey. A NEW hotkey adopts it (so
    #: a local fleet can share one configured dev secret — chain.auth_token);
    #: an ALREADY CLAIMED hotkey must present its existing token here or in the
    #: Authorization header. Omit it and the server generates one, returned once.
    auth_token: str | None = Field(default=None, min_length=8)


class AdvanceRequest(BaseModel):
    blocks: int = Field(ge=0)  # blocks only move forward (InMemoryChain parity)


class WeightsRequest(BaseModel):
    hotkey: str = Field(min_length=1)
    vector: dict[int, float]
    version_key: int = 0

    @field_validator("vector")
    @classmethod
    def _finite_non_negative(cls, v: dict[int, float]) -> dict[int, float]:
        for uid, w in v.items():
            if not math.isfinite(w) or w < 0:
                raise ValueError(f"weight for uid {uid} must be finite and >= 0, got {w}")
        return v


class AnchorRequest(BaseModel):
    payload_hex: str = Field(min_length=0)
    #: optional attribution (HttpChainAdapter sends its validator hotkey)
    hotkey: str | None = None


# ---- the service ---------------------------------------------------------------


class ChainSim(BaseService):
    name = "chainsim"

    def __init__(
        self,
        raw_config: dict[str, Any],
        *,
        metrics_port: int | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        cfg = section(raw_config, "chainsim", ChainSimConfig)
        super().__init__(
            raw_config,
            metrics_port=metrics_port if metrics_port is not None else cfg.metrics_port,
        )
        self.config = cfg
        self._now = now or wall_time
        self._conn = _connect(cfg.db_path)
        apply_migrations(self._conn, MIGRATIONS_DIR)
        self._init_meta()
        self._init_operator_token()
        #: Flipped when the HTTP API exits without being asked to (bind failure,
        #: crashed server task) — a simulator with no API is not healthy.
        self._http_api_ok = True
        self.health.register_check("sim_db", self._db_ok)
        self.health.register_check("http_api", lambda: self._http_api_ok)

        reg = self.health.registry
        self._m_registrations = Counter(
            "vidaio_chainsim_registrations_total",
            "Registrations by role and outcome (new/existing)",
            ["role", "outcome"],
            registry=reg,
        )
        self._m_weight_calls = Counter(
            "vidaio_chainsim_weight_calls_total",
            "set_weights calls by result",
            ["result"],
            registry=reg,
        )
        self._m_anchors = Counter(
            "vidaio_chainsim_anchors_total", "Anchored commitment payloads", registry=reg
        )
        self._m_auth_failures = Counter(
            "vidaio_chainsim_auth_failures_total",
            "Rejected mutations by typed authorization error",
            ["error"],
            registry=reg,
        )
        self.app = self._build_app()

    # -- meta / block clock ----------------------------------------------------

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else None

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _init_meta(self) -> None:
        if self._meta_get("start_time") is None:
            self._meta_set("start_time", repr(float(self._now())))
        for key, default in (("advance_offset", "0"), ("emission_credited_until", "0")):
            if self._meta_get(key) is None:
                self._meta_set(key, default)

    # -- authorization (see the module docstring for the chain mapping) ----------

    def _init_operator_token(self) -> None:
        """Establish the operator credential for /advance, /reset, /report/write.

        `chainsim.operator_token` in config wins and is (re)applied on every
        start — that is also the recovery path if a generated token is lost.
        Otherwise a token is generated ONCE per sim database, written to
        `<report_dir>/operator-token.txt` (owner-only) and logged once; restarts
        keep the stored hash, so the file stays valid.
        """
        configured = self.config.operator_token.strip()
        if configured:
            self._meta_set("operator_token_sha256", _hash_token(configured))
            self._meta_set("operator_token_source", "config")
            return
        if self._meta_get("operator_token_sha256") is not None:
            return
        token = secrets.token_urlsafe(32)
        self._meta_set("operator_token_sha256", _hash_token(token))
        self._meta_set("operator_token_source", "generated")
        path = Path(self.config.report_dir) / OPERATOR_TOKEN_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token + "\n", encoding="utf-8")
            path.chmod(0o600)
        except OSError as exc:
            self.log.warning(
                "could not write the operator token file — use the logged value"
                " or set chainsim.operator_token",
                extra={"path": str(path), "error": f"{type(exc).__name__}: {exc}"},
            )
        # Logged ONCE per sim database (local simulator credential, not a chain key).
        self.log.warning(
            "generated a chainsim operator token for /advance, /reset and"
            " /report/write — store it; set chainsim.operator_token to pin your own",
            extra={"operator_token": token, "path": str(path)},
        )

    def _neuron_row(self, hotkey: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT uid, hotkey, role, token_sha256 FROM neurons WHERE hotkey = ?", (hotkey,)
        ).fetchone()

    def _reject(self, status: int, error: str, message: str) -> HTTPException:
        self._m_auth_failures.labels(error).inc()
        # NB: "message" is reserved by logging.LogRecord — the detail goes as "reason".
        self.log.warning("mutation rejected", extra={"error": error, "reason": message})
        return _auth_error(status, error, message)

    def _authorize_hotkey(
        self, authorization: str | None, hotkey: str, *, require_role: str | None = None
    ) -> sqlite3.Row:
        """Bearer token must be the one issued to `hotkey`. Returns its neuron row.

        401 when no credential is presented, 403 when it does not belong to that
        hotkey (or the hotkey is unknown/unclaimed, or holds the wrong role).
        """
        token = _bearer(authorization)
        if token is None:
            raise self._reject(
                401,
                AUTH_MISSING,
                "this endpoint requires 'Authorization: Bearer <token>' — the token"
                " issued to this hotkey at registration",
            )
        row = self._neuron_row(hotkey)
        if row is None:
            raise self._reject(
                403, AUTH_UNKNOWN_HOTKEY, f"hotkey {hotkey!r} is not registered"
            )
        if not _token_matches(token, row["token_sha256"]):
            raise self._reject(
                403, AUTH_INVALID, f"token does not belong to hotkey {hotkey!r}"
            )
        if require_role is not None and row["role"] != require_role:
            raise self._reject(
                403,
                AUTH_WRONG_ROLE,
                f"hotkey {hotkey!r} has role {row['role']!r}; this endpoint requires"
                f" role {require_role!r}",
            )
        return row

    def _is_operator(self, token: str) -> bool:
        stored = self._meta_get("operator_token_sha256") or ""
        return _token_matches(token, stored)

    def _authorize_operator(self, authorization: str | None) -> None:
        """Operator powers: block production and history destruction."""
        token = _bearer(authorization)
        if token is None:
            raise self._reject(
                401,
                AUTH_MISSING,
                "operator endpoint — present 'Authorization: Bearer <operator token>'"
                f" (chainsim.operator_token, or {OPERATOR_TOKEN_FILE} in the report dir)",
            )
        if not self._is_operator(token):
            raise self._reject(403, AUTH_INVALID, "not the operator token")

    def _authorize_any_identity(self, authorization: str | None) -> None:
        """Operator OR any registered identity (report artifacts are shared output)."""
        token = _bearer(authorization)
        if token is None:
            raise self._reject(
                401, AUTH_MISSING, "this endpoint requires 'Authorization: Bearer <token>'"
            )
        if self._is_operator(token):
            return
        presented = _hash_token(token)
        row = self._conn.execute(
            "SELECT 1 FROM neurons WHERE token_sha256 = ? AND token_sha256 != ''",
            (presented,),
        ).fetchone()
        if row is None:
            raise self._reject(
                403, AUTH_INVALID, "token is neither the operator token nor a registered identity"
            )

    def current_block(self) -> int:
        start = float(self._meta_get("start_time"))  # type: ignore[arg-type]
        elapsed = max(0.0, float(self._now()) - start)
        offset = int(self._meta_get("advance_offset"))  # type: ignore[arg-type]
        return 1 + int(elapsed / self.config.block_seconds) + offset

    def block_time(self, block_number: int) -> float | None:
        """The wall-clock UTC epoch-seconds a block was produced.

        A PRODUCED block's time is a PURE function of its number:
        `start_time + (block_number - 1) * block_seconds`. Returns None for a block not
        yet produced (`b > current_block()`) — its time is not knowable. The auditor
        binds `EpochLog.created_at` to `block_time(close_block)`, so a backdated created_at
        (keeping an expired PODIUM/CROWN reward window active) is caught.

        an internal review: `advance_offset` is DELIBERATELY absent from the time VALUE. It
        governs ONLY the produced/None gate (via `current_block`), never the value — so a
        block that is already produced keeps the SAME time when the clock later advances.
        The earlier `- advance_offset` term applied every /advance RETROACTIVELY to past
        blocks, shifting an honest historical `created_at` out of tolerance so a later
        /advance turned a CLEAN epoch DISPUTED. /advance now mints FUTURE-dated blocks
        (fast-forwarding time), never rewrites past ones. Deterministic given the persisted
        `start_time`, so the finalizer and a later auditor read an identical value.
        """
        if block_number > self.current_block():
            return None
        start = float(self._meta_get("start_time"))  # type: ignore[arg-type]
        return start + (block_number - 1) * self.config.block_seconds

    # -- emission settlement -----------------------------------------------------

    def _settle_emission(self) -> int:
        """Credit emission for every block up to the current one. Returns the block.

        Deterministic regardless of when it runs: block b is governed by the
        latest recorded vector with record-block < b (calls are recorded at the
        then-current block and blocks only move forward, so history behind the
        credited watermark can never change).
        """
        current = self.current_block()
        credited = int(self._meta_get("emission_credited_until"))  # type: ignore[arg-type]
        if current <= credited or self.config.emission_per_block <= 0:
            if current > credited:
                self._meta_set("emission_credited_until", str(current))
            return current
        calls = self._conn.execute(
            "SELECT block, vector_json FROM weight_calls ORDER BY seq"
        ).fetchall()
        per_uid: dict[int, float] = defaultdict(float)
        for i, row in enumerate(calls):
            seg_start = row["block"] + 1  # a vector governs blocks strictly after its own
            seg_end = calls[i + 1]["block"] if i + 1 < len(calls) else current
            lo, hi = max(seg_start, credited + 1), min(seg_end, current)
            if hi < lo:
                continue
            vector = {int(k): float(v) for k, v in json.loads(row["vector_json"]).items()}
            total = sum(w for w in vector.values() if w > 0)
            if total <= 0:
                continue
            pot = (hi - lo + 1) * self.config.emission_per_block
            for uid, w in vector.items():
                if w > 0:
                    per_uid[uid] += pot * (w / total)
        for uid, amount in per_uid.items():
            # unregistered uids silently earn nothing (their share is burned)
            self._conn.execute(
                "UPDATE neurons SET emission_credited = emission_credited + ? WHERE uid = ?",
                (amount, uid),
            )
        self._meta_set("emission_credited_until", str(current))
        return current

    # -- views -------------------------------------------------------------------

    def _latest_vector(self) -> tuple[dict[int, float], str, int] | None:
        row = self._conn.execute(
            "SELECT hotkey, block, vector_json FROM weight_calls ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        vector = {int(k): float(v) for k, v in json.loads(row["vector_json"]).items()}
        return vector, row["hotkey"], row["block"]

    def latest_weights(self, hotkey: str) -> dict[str, Any]:
        """The vector this hotkey CURRENTLY has recorded, or a positive "none".

        The read behind `GET /weights/{hotkey}` and
        `vidaio.chain.SubmittedWeightsReader`. The sim already stores every
        accepted vector; this exposes the latest one per hotkey so a caller can
        prove that ITS OWN vector landed rather than inferring it from the fact
        that some write happened.

        `vector` is None — not `{}` — when the hotkey has no weight record at
        all: an empty vector and a missing one are different chain states, and
        only the missing one is evidence that a write did not land.
        """
        row = self._conn.execute(
            "SELECT block, version_key, vector_json FROM weight_calls"
            " WHERE hotkey = ? ORDER BY seq DESC LIMIT 1",
            (hotkey,),
        ).fetchone()
        if row is None:
            return {
                "hotkey": hotkey,
                "vector": None,
                "block": None,
                "version_key": None,
                "current_block": self.current_block(),
            }
        return {
            "hotkey": hotkey,
            "vector": json.loads(row["vector_json"]),
            "block": row["block"],
            "version_key": row["version_key"],
            "current_block": self.current_block(),
        }

    def _neuron_dicts(self) -> list[dict[str, Any]]:
        latest = self._latest_vector()
        vector = latest[0] if latest is not None else {}
        total = sum(w for w in vector.values() if w > 0)
        last_updates = {
            row["hotkey"]: row["b"]
            for row in self._conn.execute(
                "SELECT hotkey, MAX(block) AS b FROM weight_calls GROUP BY hotkey"
            )
        }
        out: list[dict[str, Any]] = []
        for row in self._conn.execute("SELECT * FROM neurons ORDER BY uid"):
            share = vector.get(row["uid"], 0.0) / total if total > 0 else 0.0
            out.append(
                {
                    "uid": row["uid"],
                    "hotkey": row["hotkey"],
                    "coldkey": row["coldkey"],
                    "ip": row["ip"],
                    "role": row["role"],
                    "is_validator": row["role"] == "validator",
                    "alpha_stake": row["alpha_stake"],
                    #: CURRENT per-block emission rate (see module docstring)
                    "emission": share * self.config.emission_per_block,
                    "emission_credited": row["emission_credited"],
                    "registered_block": row["registered_block"],
                    "last_update": last_updates.get(row["hotkey"], row["registered_block"]),
                }
            )
        return out

    def state(self) -> dict[str, Any]:
        """Full sim state (also the input contract of chainsim.report.build_report)."""
        block = self._settle_emission()
        neurons = self._neuron_dicts()
        weight_calls = [
            {
                "seq": row["seq"],
                "hotkey": row["hotkey"],
                "block": row["block"],
                "version_key": row["version_key"],
                "vector": json.loads(row["vector_json"]),
                "created_at": row["created_at"],
            }
            for row in self._conn.execute("SELECT * FROM weight_calls ORDER BY seq")
        ]
        anchors = [
            {
                "seq": row["seq"],
                "txid": row["txid"],
                "payload_hex": row["payload_hex"],
                "hotkey": row["hotkey"],
                "block": row["block"],
                "created_at": row["created_at"],
            }
            for row in self._conn.execute("SELECT * FROM anchors ORDER BY seq")
        ]
        credited_until = int(self._meta_get("emission_credited_until"))  # type: ignore[arg-type]
        distributed = sum(n["emission_credited"] for n in neurons)
        minted = credited_until * self.config.emission_per_block
        return {
            "block": block,
            "config": {
                "tempo": self.config.tempo,
                "block_seconds": self.config.block_seconds,
                "emission_per_block": self.config.emission_per_block,
            },
            "neurons": neurons,
            "weight_calls": weight_calls,
            "anchors": anchors,
            "emission": {
                "credited_until_block": credited_until,
                "per_block": self.config.emission_per_block,
                "minted": minted,
                "distributed": distributed,
                "undistributed": minted - distributed,
            },
        }

    # -- mutations ----------------------------------------------------------------

    def register(self, req: RegisterRequest) -> dict[str, Any]:
        """Register (or update) a neuron. TRUSTED entry point — no auth check.

        In-process callers (tests, harnesses) are inside the trust boundary; HTTP
        callers reach this only through the /register handler, which enforces
        token ownership of an already-claimed hotkey first.

        `auth_token` in the result is the identity's bearer credential, returned
        ONCE — at the registration that claims the hotkey. Later registrations of
        a claimed hotkey return None (only the hash is stored; it cannot be
        re-read). Callers that lose it must reset the sim or claim a new hotkey.
        """
        block = self.current_block()
        row = self._neuron_row(req.hotkey)
        if row is not None:
            sets = "coldkey = ?, ip = ?, role = ?"
            args: list[Any] = [req.coldkey, req.ip, req.role]
            if req.alpha_stake is not None:
                sets += ", alpha_stake = ?"
                args.append(req.alpha_stake)
            issued: str | None = None
            if not row["token_sha256"]:
                # Unclaimed: a row migrated from a pre-auth sim database. The
                # first registrar claims it (see migrations/0002_auth.sql).
                issued = req.auth_token or secrets.token_urlsafe(24)
                sets += ", token_sha256 = ?"
                args.append(_hash_token(issued))
            self._conn.execute(
                f"UPDATE neurons SET {sets} WHERE uid = ?", (*args, row["uid"])
            )
            self._m_registrations.labels(req.role, "existing").inc()
            return {
                "uid": row["uid"],
                "hotkey": req.hotkey,
                "new": False,
                "block": block,
                "auth_token": issued,
            }
        uid = self._conn.execute(
            "SELECT COALESCE(MAX(uid), -1) + 1 AS next FROM neurons"
        ).fetchone()["next"]
        token = req.auth_token or secrets.token_urlsafe(24)
        self._conn.execute(
            "INSERT INTO neurons (uid, hotkey, coldkey, ip, role, alpha_stake,"
            " registered_block, token_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uid,
                req.hotkey,
                req.coldkey,
                req.ip,
                req.role,
                req.alpha_stake or 0.0,
                block,
                _hash_token(token),
            ),
        )
        self._m_registrations.labels(req.role, "new").inc()
        return {
            "uid": uid,
            "hotkey": req.hotkey,
            "new": True,
            "block": block,
            "auth_token": token,
        }

    def submit_weights(self, req: WeightsRequest) -> dict[str, Any]:
        """Record a weight vector (tempo-gated). TRUSTED entry point — no auth.

        Chain RULES live here (registered? tempo?); chain PERMISSIONS live in the
        /weights handler (own the hotkey, hold the validator role), exactly as a
        runtime separates extrinsic validity from signature/permit checks.
        """
        block = self._settle_emission()
        known = self._conn.execute(
            "SELECT 1 FROM neurons WHERE hotkey = ?", (req.hotkey,)
        ).fetchone()
        if known is None:
            self._m_weight_calls.labels("rejected").inc()
            return {
                "success": False,
                "block": block,
                "message": f"hotkey {req.hotkey!r} is not registered",
            }
        last = self._conn.execute(
            "SELECT MAX(block) AS b FROM weight_calls WHERE hotkey = ?", (req.hotkey,)
        ).fetchone()["b"]
        if last is not None and block <= last + self.config.tempo:
            self._m_weight_calls.labels("tempo_rejected").inc()
            return {"success": False, "block": block, "message": "tempo gate: too soon"}
        self._conn.execute(
            "INSERT INTO weight_calls (hotkey, block, version_key, vector_json)"
            " VALUES (?, ?, ?, ?)",
            (
                req.hotkey,
                block,
                req.version_key,
                json.dumps({str(uid): w for uid, w in sorted(req.vector.items())}),
            ),
        )
        self._m_weight_calls.labels("accepted").inc()
        self.log.info(
            "weights recorded",
            extra={"hotkey": req.hotkey, "block": block, "uids": len(req.vector)},
        )
        return {"success": True, "block": block, "message": ""}

    def anchor(self, req: AnchorRequest) -> dict[str, Any]:
        """Record a commitment payload. TRUSTED entry point — the /anchor handler
        proves the caller owns `req.hotkey` (or is the operator) before calling."""
        try:
            payload = bytes.fromhex(req.payload_hex)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"payload_hex is not hex: {exc}")
        if len(payload) > 128:
            raise HTTPException(status_code=400, detail="chain payload must be <= 128 bytes")
        block = self.current_block()
        txid = "0x" + hashlib.sha256(payload).hexdigest()[:16]  # InMemoryChain parity
        self._conn.execute(
            "INSERT INTO anchors (txid, payload_hex, hotkey, block) VALUES (?, ?, ?, ?)",
            (txid, payload.hex(), req.hotkey, block),
        )
        self._m_anchors.inc()
        return {"txid": txid, "block": block}

    def advance(self, blocks: int) -> int:
        """Move the block clock forward. TRUSTED entry point — over HTTP this is
        an OPERATOR power (see the /advance handler)."""
        offset = int(self._meta_get("advance_offset"))  # type: ignore[arg-type]
        self._meta_set("advance_offset", str(offset + blocks))
        return self.current_block()

    def reset(self) -> int:
        """Wipe sim history and rewind the clock. Returns the new current block.

        Identities go with it (their tokens too — a reset sim re-issues on
        registration); the OPERATOR credential survives, it belongs to the node,
        not to the run.
        """
        for table in ("neurons", "weight_calls", "anchors"):
            self._conn.execute(f"DELETE FROM {table}")
        self._meta_set("start_time", repr(float(self._now())))
        self._meta_set("advance_offset", "0")
        self._meta_set("emission_credited_until", "0")
        return self.current_block()

    # -- app ----------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        """The HTTP surface. READS are open; every MUTATION is authorized here
        (module docstring: token ~ hotkey signature, operator token ~ node key)."""
        app = FastAPI(title="vidaio chain simulator", docs_url=None, redoc_url=None)

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {"service": self.name, "status": "ok", "block": self.current_block()}

        @app.post("/register")
        async def register(
            req: RegisterRequest, authorization: str | None = Header(default=None)
        ) -> dict[str, Any]:
            presented = _bearer(authorization) or req.auth_token
            row = self._neuron_row(req.hotkey)
            if row is not None and row["token_sha256"]:
                # Identity takeover guard: only the holder may re-register.
                if presented is None:
                    raise self._reject(
                        409,
                        AUTH_HOTKEY_CLAIMED,
                        f"hotkey {req.hotkey!r} is already registered — re-registration"
                        " requires its token",
                    )
                if not _token_matches(presented, row["token_sha256"]):
                    raise self._reject(
                        403, AUTH_INVALID, f"token does not belong to hotkey {req.hotkey!r}"
                    )
            if presented is not None and presented != req.auth_token:
                req = req.model_copy(update={"auth_token": presented})
            return self.register(req)

        @app.post("/advance")
        async def advance(
            req: AdvanceRequest, authorization: str | None = Header(default=None)
        ) -> dict[str, Any]:
            self._authorize_operator(authorization)  # block production is a node power
            return {"block": self.advance(req.blocks)}

        @app.get("/neurons")
        async def neurons() -> dict[str, Any]:
            block = self._settle_emission()
            return {"block": block, "neurons": self._neuron_dicts()}

        @app.post("/weights")
        async def weights(
            req: WeightsRequest, authorization: str | None = Header(default=None)
        ) -> dict[str, Any]:
            # Own the hotkey AND hold the validator permit for it.
            self._authorize_hotkey(authorization, req.hotkey, require_role="validator")
            return self.submit_weights(req)

        @app.get("/weights/{hotkey}")
        async def submitted_weights(hotkey: str) -> dict[str, Any]:
            # Chain state is PUBLIC, like /neurons and /state: anyone may read
            # which vector a hotkey has recorded (on a real chain it is a
            # storage query). Writing one still needs the hotkey's token.
            return self.latest_weights(hotkey)

        @app.post("/anchor")
        async def anchor(
            req: AnchorRequest, authorization: str | None = Header(default=None)
        ) -> dict[str, Any]:
            if req.hotkey is None:
                # Unattributed anchors are an operator action, not a participant one.
                self._authorize_operator(authorization)
            else:
                self._authorize_hotkey(authorization, req.hotkey)
            return self.anchor(req)

        @app.get("/block_hash/{block_number}")
        async def block_hash(block_number: int) -> dict[str, Any]:
            # an internal review: the un-grindable sampling beacon is block_hash(close_block
            # + K). The sim owns the block clock, so it is authoritative about whether a
            # block has been produced: `hash` is the deterministic `synthetic_block_hash`
            # for a PRODUCED block (n <= current) and null for a not-yet-produced one (the
            # beacon is then not finalized, so the auditor HOLDS). Chain state is PUBLIC —
            # a block hash is a plain read on a real chain — so this stays open like
            # /state. The derivation MATCHES InMemoryChain byte-for-byte.
            current = self.current_block()
            produced = block_number <= current
            return {
                "block": block_number,
                "current_block": current,
                "hash": synthetic_block_hash(block_number) if produced else None,
            }

        @app.get("/block_time/{block_number}")
        async def block_time(block_number: int) -> dict[str, Any]:
            # an internal review: the wall-clock time a block was produced (UTC epoch seconds),
            # so the auditor can bind EpochLog.created_at to the epoch's close_block time. Chain
            # state is PUBLIC (a block timestamp is a plain read on a real chain), so this stays
            # open like /state and /block_hash. Null for a not-yet-produced block.
            current = self.current_block()
            return {
                "block": block_number,
                "current_block": current,
                "time": self.block_time(block_number) if block_number <= current else None,
            }

        @app.get("/state")
        async def state() -> dict[str, Any]:
            return self.state()

        @app.get("/report")
        async def report() -> dict[str, Any]:
            return build_report(self.state())

        @app.post("/report/write")
        async def report_write(
            authorization: str | None = Header(default=None),
        ) -> dict[str, Any]:
            self._authorize_any_identity(authorization)  # writes files; not a read
            json_path, md_path = write_report(self.state(), self.config.report_dir)
            return {"json_path": str(json_path), "md_path": str(md_path)}

        @app.post("/reset")
        async def reset(authorization: str | None = Header(default=None)) -> dict[str, Any]:
            if not self.config.enable_reset:
                raise HTTPException(status_code=403, detail="reset is disabled by config")
            self._authorize_operator(authorization)  # destroying history is a node power
            return {"block": self.reset()}

        return app

    # -- lifecycle ------------------------------------------------------------------

    def _db_ok(self) -> bool:
        """Health probe for the sim database — on a connection OF ITS OWN.

        Health checks run on the HealthServer THREAD. Reaching for `self._conn`
        there tests the wrong thing twice over: it shares a connection (and its
        transaction state) with whatever the event loop is doing, and it can pass
        while the database FILE is gone, because sqlite keeps serving a deleted
        inode. A short-lived connection that reads a real table each time answers
        the question the check is actually asking. An in-memory sim has no file to
        reopen, so it necessarily falls back to the service connection.
        """
        path = str(self.config.db_path)
        try:
            if path == ":memory:":
                self._conn.execute("SELECT 1 FROM meta LIMIT 1").fetchone()
                return True
            probe = _connect(path)
            try:
                probe.execute("SELECT 1 FROM meta LIMIT 1").fetchone()
                return True
            finally:
                probe.close()
        except Exception:
            return False

    def _on_api_exit(self, api: asyncio.Task[Any]) -> None:
        """The uvicorn task ended without a stop being requested.

        Usually a bind failure (uvicorn raises SystemExit after logging it). Left
        unmonitored it produces the worst outcome available: a process reporting
        "ok" on /health with nothing answering on its API port at all. The exit is
        FATAL (non-zero) so a supervisor restarts it — exit 0 means "deliberate
        stop" and would leave the whole stack without a chain.
        """
        self._http_api_ok = False
        error: BaseException | None = None
        try:
            error = api.exception()
        except (asyncio.CancelledError, asyncio.InvalidStateError):
            error = None
        detail = "" if error is None else f"{type(error).__name__}: {error}"
        self.fail_fatal(
            "chainsim HTTP API exited unexpectedly — the simulator has no API"
            f" (port={self.config.port} error={detail})"
        )

    async def run(self) -> None:
        server = uvicorn.Server(
            uvicorn.Config(
                self.app, host="0.0.0.0", port=self.config.port, log_level="warning"
            )
        )

        # SystemExit (uvicorn's bind-failure exit) is a BaseException: awaited bare
        # it would tear the event loop down instead of being reported.
        async def _serve() -> None:
            try:
                await server.serve()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise RuntimeError(f"uvicorn exited: {type(exc).__name__}: {exc}") from exc

        api = asyncio.create_task(_serve(), name="chainsim-http")
        stop = asyncio.create_task(self.stopping.wait(), name="chainsim-stop")
        try:
            await asyncio.wait({api, stop}, return_when=asyncio.FIRST_COMPLETED)
            if api.done() and not self.stopping.is_set():
                self._on_api_exit(api)
        finally:
            server.should_exit = True
            stop.cancel()
            await asyncio.gather(api, return_exceptions=True)
            self.close()

    def close(self) -> None:
        self._conn.close()
