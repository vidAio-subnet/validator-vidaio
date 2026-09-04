"""Chain-adapter selection — mode is config-only; both modes drive the SAME code.

the project design record rules 8 & 9 (owner, 2026-08-21): the PRODUCTION default is the
REAL chain — config/default.yaml ships `chain.mode: bittensor`, so validators sync
the metagraph, set weights and anchor on the real chain. `chain.mode: report` (the
chainsim / embedded-journal path) stays available and is the default ONLY in
test/dev/local overlays (tests, local-stack, compose, e2e) so the whole system can
still run end-to-end WITHOUT a chain. Only the `ChainAdapter` implementation behind
the Protocol swaps — never service code.

NOTE on the model default below: it is deliberately kept `report` (not `bittensor`)
so a bare `make_chain_adapter({})` with NO yaml — a testing convenience — stays
chainless. Production never passes an empty dict: it loads config/default.yaml,
whose `chain.mode: bittensor` is the real default. The real adapter is built (and
its bittensor deps lazily imported) only when `mode: bittensor` is actually set.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit import NotConfiguredError
from vidaio.chain.adapter import ChainAdapter
from vidaio.chain.client import EmbeddedReportingChain, HttpChainAdapter
from vidaio.core import section

if TYPE_CHECKING:
    from vidaio.chain.bittensor_adapter import BittensorAdapterConfig


BITTENSOR_PRODUCTION_SEAMS = (
    "anchor_commitment",
    "block_hash",
    "block_time",
    "commitment_capacity",
    "commitment_rate_limit",
    "read_commitment_record",
    "commit_reveal_enabled",
    "current_block",
    "epoch_close_block",
    "finalized_block",
    "get_burn_uid",
    "latest_closed_epoch",
    "neurons",
    "neurons_at",
    "read_anchor",
    "read_anchor_at",
    "read_anchor_block",
    "refresh",
    "sign",
    "set_weights",
    "submitted_weights",
    "tempo",
    "weight_commit_pending",
)

BITTENSOR_READ_ONLY_SEAMS = (
    "block_hash",
    "block_time",
    "commitment_capacity",
    "read_commitment_record",
    "current_block",
    "epoch_close_block",
    "finalized_block",
    "get_burn_uid",
    "latest_closed_epoch",
    "neurons",
    "neurons_at",
    "read_anchor",
    "read_anchor_at",
    "read_anchor_block",
    "refresh",
    "submitted_weights",
    "tempo",
)


def assert_bittensor_adapter_contract(adapter: object) -> None:
    """Fail before construction when a real-chain adapter loses a required seam.

    This lives in the library factory—not only a launcher script—so every shipped
    entrypoint receives the same production guard. The checks are structural;
    live RPC capability is exercised by the deployment preflight/testnet ladder.
    """
    missing = [
        name
        for name in BITTENSOR_PRODUCTION_SEAMS
        if not callable(getattr(adapter, name, None))
    ]
    if missing:
        raise NotConfiguredError(
            "bittensor adapter is missing release-required seam(s): "
            + ", ".join(missing)
        )


def assert_bittensor_read_only_adapter_contract(adapter: object) -> None:
    """Require every bounded chain-read seam without requiring a signer seam."""
    missing = [
        name
        for name in BITTENSOR_READ_ONLY_SEAMS
        if not callable(getattr(adapter, name, None))
    ]
    if missing:
        raise NotConfiguredError(
            "read-only bittensor adapter is missing release-required seam(s): "
            + ", ".join(missing)
        )


class ChainConfig(BaseModel):
    """Schema for the `chain:` section of config."""

    model_config = ConfigDict(extra="forbid")

    #: "report" (chainless report/sim path — the DEFAULT in test/dev overlays and
    #: the model default here) or "bittensor" (the REAL chain — what
    #: config/default.yaml ships as the production default).
    mode: Literal["report", "bittensor"] = "report"

    #: The chainsim base URL, or the sentinel "embedded" for an in-process
    #: EmbeddedReportingChain (single-process harness runs, no HTTP sim).
    chainsim_url: str = "http://127.0.0.1:8400"

    #: Identifies this process's validator on the sim (register + weights + anchors).
    validator_hotkey: str = "local-validator"

    #: The Scoring Authority's ss58 whose commitment carries the epoch-log anchors — read
    #: back as the independent third verification leg (#3, the project design record §4/§5).
    #: Empty falls back to `validator_hotkey` (a self-anchoring single-node deployment). Used
    #: in BOTH modes: bittensor reads only this account's Commitments-pallet
    #: entry, and report mode now FILTERS the sim's globally-recorded anchors to those written
    #: by this account — so a non-authority participant cannot REPLACE the effective anchor.
    anchor_hotkey: str = ""

    #: One coherent POSIX lock file shared by every process that writes this
    #: wallet's mutable Commitments-pallet slot. Production writer roles require
    #: an absolute path and hold this lane through finalized archive read-back.
    anchor_writer_lock_path: Path | None = None
    anchor_writer_lock_timeout_seconds: float = Field(default=30.0, gt=0, le=60)

    #: Bearer credential proving this process owns `validator_hotkey` on the sim —
    #: the report-mode stand-in for the wallet that will sign real extrinsics
    #: (vidaio/chainsim/service.py, authorization section). Every mutation carries
    #: it; without one the sim answers 401 and no weights or anchors land.
    #: Leave empty when the process registers itself through
    #: HttpChainAdapter.register(), which captures the token the sim issues;
    #: set it (VIDAIO__CHAIN__AUTH_TOKEN) when another process did the
    #: registering, or to claim the hotkey with a secret you chose.
    #: Ignored by the "embedded" in-process chain, which has no trust boundary.
    auth_token: str = ""

    #: Report artifacts (embedded-chain journal; chainsim report/write output).
    report_dir: Path = Path("./data/chain-reports")

    # -- bittensor mode (mode: bittensor) --------------------------------------
    # These are consumed ONLY when mode == "bittensor". They mirror the proven
    # env sets; secrets are NEVER stored here — only the NAME of the env var
    # that holds the hotkey seed.

    #: Named network ("finney" prod, "test" testnet, "archive" for historical
    #: reads). An explicit `endpoint` wss:// URL below BEATS it in production.
    network: str = "finney"
    #: The subnet this validator writes weights on (VidAIO = 85).
    netuid: int = 85
    #: Explicit wss:// endpoint; when set it overrides `network`.
    endpoint: str = ""
    #: Ordered secondary archive-capable endpoints. Empty at initial launch;
    #: operators can add a second endpoint through YAML/env without changing code.
    fallback_endpoints: list[str] = Field(default_factory=list)
    #: On-disk btcli wallet (name+hotkey; path optional). Takes precedence over
    #: the seed env when both are present.
    wallet_name: str = ""
    wallet_hotkey: str = ""
    wallet_path: str = ""
    #: NAME of the env var holding the hotkey seed/mnemonic (never the secret
    #: value). The pod loads it at startup and crashes if absent when
    #: mode: bittensor (fail-fast on identity).
    hotkey_seed_env: str = "VIDAIO_HOTKEY_SEED"
    #: Fleet convergence fence; bump with the epoch-log schema. Report/test
    #: overlays may explicitly select 0 when no live SDK submission occurs.
    version_key: int = Field(default=16, ge=0)
    #: Per-attempt connect timeout and short-RPC timeout (daemon-thread bounded).
    connect_timeout_seconds: float = Field(default=30.0, gt=0)
    rpc_timeout_seconds: float = Field(default=30.0, gt=0)
    #: Bounded finalized-state observations after a non-CR SDK success. The
    #: immediate read plus four one-block waits covers ordinary archive lag
    #: without issuing a second extrinsic.
    weight_readback_attempts: int = Field(default=5, ge=1)
    weight_readback_delay_seconds: float = Field(default=12.0, ge=0)
    #: Metagraph snapshot TTL — refresh() skips the RPC inside this window.
    metagraph_ttl_seconds: float = Field(default=120.0, gt=0)
    #: Reconnect the socket only after this many CONSECUTIVE raised failures.
    reconnect_after_consecutive_failures: int = Field(default=3, ge=1)


def _bittensor_adapter_config(
    config: ChainConfig, *, read_only: bool = False
) -> "BittensorAdapterConfig":
    # Imported lazily with the real adapter so chainless/report imports never pull
    # in the optional SDK dependency.
    from vidaio.chain.bittensor_adapter import BittensorAdapterConfig

    return BittensorAdapterConfig(
        # A read-only service has no "own" validator. Do not even carry signer or
        # wallet configuration into its adapter object; only public chain locators
        # and bounded/reconnect settings cross this boundary.
        validator_hotkey="" if read_only else config.validator_hotkey,
        anchor_hotkey=config.anchor_hotkey,
        anchor_writer_lock_path=(None if read_only else config.anchor_writer_lock_path),
        anchor_writer_lock_timeout_seconds=(config.anchor_writer_lock_timeout_seconds),
        network=config.network,
        netuid=config.netuid,
        endpoint=config.endpoint,
        fallback_endpoints=tuple(config.fallback_endpoints),
        wallet_name="" if read_only else config.wallet_name,
        wallet_hotkey="" if read_only else config.wallet_hotkey,
        wallet_path="" if read_only else config.wallet_path,
        hotkey_seed_env="" if read_only else config.hotkey_seed_env,
        version_key=config.version_key,
        connect_timeout_seconds=config.connect_timeout_seconds,
        rpc_timeout_seconds=config.rpc_timeout_seconds,
        weight_readback_attempts=config.weight_readback_attempts,
        weight_readback_delay_seconds=config.weight_readback_delay_seconds,
        metagraph_ttl_seconds=config.metagraph_ttl_seconds,
        reconnect_after_consecutive_failures=(
            config.reconnect_after_consecutive_failures
        ),
        read_only=read_only,
    )


def make_chain_adapter(raw_config: dict[str, Any]) -> ChainAdapter:
    """Build the configured ChainAdapter from the `chain:` config section.

    mode: bittensor builds the REAL adapter (lazily importing the optional
    `.[chain]` bittensor deps inside `bittensor_adapter`); a missing dep or wallet
    fails fast with a clear message (NotConfiguredError points at the extra).
    """
    config = section(raw_config, "chain", ChainConfig)
    if config.mode == "bittensor":
        # Imported HERE, not at module top, so importing vidaio.chain never drags
        # in bittensor and the chainless test suite runs without it.
        from vidaio.chain.bittensor_adapter import BittensorChainAdapter

        assert_bittensor_adapter_contract(BittensorChainAdapter)
        return BittensorChainAdapter(_bittensor_adapter_config(config))
    if config.chainsim_url == "embedded":
        return EmbeddedReportingChain(
            journal_path=config.report_dir / "embedded-chain.jsonl"
        )
    return HttpChainAdapter(
        config.chainsim_url,
        validator_hotkey=config.validator_hotkey,
        # an internal review: bind report-mode anchor READS to the Scoring Authority account,
        # exactly like bittensor mode — the sim records anchors from ANY registered hotkey, so
        # the reader must honor ONLY the authority's (empty falls back to validator_hotkey, the
        # self-anchoring single-node case). Without this a non-authority participant could write
        # a competing anchor and REPLACE the effective one.
        anchor_hotkey=config.anchor_hotkey or None,
        auth_token=config.auth_token or None,
    )


def make_read_only_chain_adapter(raw_config: dict[str, Any]) -> ChainAdapter:
    """Build the wallet-free chain reader used by the Audit Results API.

    Bittensor mode constructs a strict read-only adapter: it never loads seed or
    on-disk wallet material and locally rejects every sign/write attempt while
    retaining the normal bounded/reconnecting read path. Report mode keeps the
    existing simulator adapter for local development; no real signing key exists
    in that mode.
    """
    config = section(raw_config, "chain", ChainConfig)
    if config.mode == "bittensor":
        from vidaio.chain.bittensor_adapter import BittensorReadOnlyChainAdapter

        assert_bittensor_read_only_adapter_contract(BittensorReadOnlyChainAdapter)
        return BittensorReadOnlyChainAdapter(
            _bittensor_adapter_config(config, read_only=True)
        )
    return make_chain_adapter(raw_config)
