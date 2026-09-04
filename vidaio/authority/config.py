"""Scoring Authority configuration (config section: `authority`).

The Scoring Authority API is a THIN POINTER index over the object store: it serves
epoch pointers (object key + digests + on-chain anchor), never the epoch-log bytes
(the project design record §3.1, build-wave 4). This config carries the HTTP/metrics
ports, the bearer token that gates validators/operators, the SQLite epoch-index path,
and the epoch/anchor parameters (netuid, blocks_per_epoch, burn uid).

Credentials are never stored here. The object-store backend and the chain mode are
selected by the SHARED `audit:` / `chain:` sections (this service reuses
`make_store` / `make_chain_adapter`), so a report-mode overlay drives the exact same
service code as production — only the store/adapter implementations swap.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class AuthorityConfig(BaseModel):
    """Schema for the `authority:` section of config."""

    model_config = ConfigDict(extra="forbid")

    # -- HTTP API + metrics (see vidaio.services.protocol port table) ------------
    #: The pointer API bind host/port (scoring-authority http 8700).
    http_host: str = "0.0.0.0"
    http_port: int = 8700
    #: Prometheus metrics port (scoring-authority 9111).
    metrics_port: int = 9111

    #: Bearer token gating EVERY epoch-pointer route (validators/operators carry
    #: it). None = OPEN — only appropriate on a loopback/dev bind; PRODUCTION
    #: DEPLOYMENTS MUST SET IT. `/healthz` is always open (liveness).
    api_token: str | None = None

    # -- epoch index (this service's own SQLite, own migrations) -----------------
    #: Append-only index of which epochs are finalized (epoch_id -> pointer + anchor).
    db_path: Path = Path("./data/authority.db")

    # -- epoch / anchor parameters ----------------------------------------------
    #: The subnet this authority scores (VidAIO = 85); stamped on the anchor.
    netuid: int = 85
    #: Blocks per epoch; `close_block(E) = (E+1)*blocks_per_epoch - 1`. Read from
    #: the chain at startup in production (never hardcode) — this is the fallback.
    blocks_per_epoch: int = Field(default=360, gt=0)
    #: Report/local fallback for the uid a genuinely-empty epoch burns 100% to.
    #: Production resolves the subnet owner's current uid from chain state and
    #: fails closed when it cannot; ``None`` prevents a stale configured uid from
    #: silently receiving emission. Report overlays explicitly pin uid 0.
    burn_uid: int | None = Field(default=None, ge=0)

    #: The scorer identity (`<name>+<digest12>`) the finalizer stamps into each
    #: epoch log. Supplied by the central scorer; empty only in tests/dev.
    scorer_version: str = ""

    # -- outage-gap recovery (P1.5, epoch schema v16) -----------------------------
    #: When a PERSISTED spine resumes after an outage, epochs whose un-grindable
    #: anchor windows (`close + K`) already elapsed can never be finalized; the
    #: finalizer SKIPS them and declares the contiguous range as `gap_epochs` in
    #: the next anchorable epoch's log (signed + anchored, so the outage is an
    #: auditable on-chain fact, not a wedged spine). Gaps up to this many epochs
    #: self-heal automatically; a LARGER gap fails loudly until the operator
    #: acknowledges it via `gap_ack_through_epoch` (an outage that long deserves
    #: a human looking at it before earnings resume). Never applies to a fresh
    #: launch — genesis keeps the strict preflight floor rule.
    max_auto_gap_epochs: int = Field(default=48, ge=1)
    #: Operator acknowledgment for an oversized gap: the highest runtime epoch id
    #: the operator confirms may be declared as a gap. Set via
    #: `VIDAIO__AUTHORITY__GAP_ACK_THROUGH_EPOCH`, remove after recovery.
    gap_ack_through_epoch: int | None = Field(default=None, ge=1)
