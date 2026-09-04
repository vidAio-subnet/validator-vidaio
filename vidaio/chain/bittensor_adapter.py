"""The REAL bittensor ChainAdapter.

Built from prior production-subnet experience. This module is the
ONLY place in vidaio that touches `bittensor` / `async-substrate-interface`, and
it touches them LAZILY: nothing at import time drags the ~150 MB SDK tree in, so
`import vidaio.chain` (and the whole test suite) runs without the optional
`.[chain]` dependency group installed. mode selection is the project design record rule 8
(`chain.mode: bittensor` = production default; `report` = tests only).

Two halves, split on the one line that matters for testability:

* the ADAPTER LOGIC — u16 quantization, metagraph -> ChainNeuron mapping,
  set_weights result classification, freshness/cache, submitted-weights readback
  + commit-reveal awareness, hotkey->uid reconciliation, and the
  one-socket/reconnect-after-3-raised-failures discipline — lives in
  `BittensorChainAdapter` and is FULLY UNIT-TESTED against a fake transport
  (tests/chain/); it never imports bittensor.
* the TRANSPORT — the thin seam (`_SubtensorTransport`) that actually constructs
  `bt.Subtensor` / a `SubstrateInterface` and makes RPCs — is the only part that
  imports bittensor, and the only part not unit-tested (validated on testnet).

The convergence phase keeps two explicit representations: ``quantize_u16`` builds
the byte-identical authority/EpochLog sum-grid, while ``max_normalize_u16`` mirrors
the pinned SDK's final conversion to the byte-identical runtime max-grid. Both are
dependency-free and differentially tested against bittensor 10.5.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from vidaio.audit import NotConfiguredError
from vidaio.chain.adapter import (
    ChainCommitmentRecord,
    ChainNeuron,
    ChainStateUnavailable,
    CommitmentCapacity,
    EpochBoundary,
    SetWeightsResult,
    SubmittedWeights,
    parse_anchor_digest,
)
from vidaio.chain.anchor_writer import anchor_writer_lock
from vidaio.core import get_logger

#: The canonical deterministic quantizer now lives in the dependency-free shared
#: home vidaio/tokenomics/quantize.py (the project design record wave 1 —
#: CONSOLIDATE). It is re-exported here so this module's historical import site
#: (`from vidaio.chain.bittensor_adapter import quantize_u16, U16_MAX`) and the
#: `vidaio.chain` package re-export keep working; the CORE math is single-sourced.
#: U16_MAX must equal vidaio.weightsetter.intents.WEIGHT_QUANTIZATION_SCALE — the
#: weight-setter puts BOTH sides on this same grid to compare (§d submitted_weights).
from vidaio.tokenomics.quantize import U16_MAX, max_normalize_u16, quantize_u16

__all__ = [
    "BittensorAdapterConfig",
    "BittensorChainAdapter",
    "BittensorHotkeySigner",
    "BittensorReadOnlyChainAdapter",
    "CommitmentCapacity",
    "EpochScheduleView",
    "MetagraphView",
    "ReadOnlyChainError",
    "U16_MAX",
    "max_normalize_u16",
    "quantize_u16",
]

#: The SDK pins the real adapter is built against (mirrored in pyproject's
#: `[project.optional-dependencies] chain`). Co-pinned on purpose: the
#: scalecodec/cyscale split makes `bittensor` and `async-substrate-interface`
#: break on import if their versions drift.
_INSTALL_HINT = (
    "the real bittensor chain adapter needs the optional 'chain' dependency group"
    " — install it with:  uv pip install -e '.[chain]'"
    " (bittensor==10.5.0 + async-substrate-interface==2.2.1, co-pinned)"
)


class ReadOnlyChainError(PermissionError):
    """A signing or mutation was attempted through a wallet-free chain reader."""


# --------------------------------------------------------------------------------------
# The transport seam — the ONLY interface the real substrate/SDK calls hide behind.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetagraphView:
    """A normalized, SDK-free snapshot of one `subtensor.metagraph(netuid)` read.

    The real transport builds this from a bittensor metagraph object; the fake
    transport builds it directly. Keeping the metagraph -> ChainNeuron MAPPING in
    the adapter (not the transport) is what makes that mapping unit-testable.
    """

    block: int
    hotkeys: list[str]
    coldkeys: list[str]
    axon_ips: list[str]
    alpha_stake: list[float]
    emission: list[float]
    validator_permit: list[bool]
    last_update: list[int]
    registration_block: list[int] = field(default_factory=list)
    axon_ports: list[int | None] = field(default_factory=list)
    incentive: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class EpochScheduleView:
    """SDK-free, block-pinned view of Bittensor's stateful epoch scheduler."""

    block: int
    last_epoch_block: int
    pending_epoch_at: int
    subnet_epoch_index: int
    tempo: int
    blocks_since_last_step: int


@dataclass(frozen=True)
class _CommitmentUsageView:
    """Raw block-pinned ``MaxSpace`` + optional ``UsedSpaceOf`` tracker."""

    block: int
    max_space: int
    usage_epoch: int | None
    used_space: int


@runtime_checkable
class _SubtensorTransport(Protocol):
    """Every actual chain call, behind one thin interface.

    A *raised* exception means TRANSPORT trouble (counts toward reconnect); a
    `(False, message)` return from `set_weights` means the chain answered over a
    HEALTHY socket. Implementations serialize their
    own socket; the adapter serializes writes with a thread-level socket mutex on
    top so a caller-abandoned submit thread cannot be run over.
    """

    def current_block(self) -> int: ...

    def finalized_block(self) -> int:
        """Latest GRANDPA-finalized block, never the best/current head."""
        ...

    def epoch_schedule(self, netuid: int, block_number: int) -> EpochScheduleView:
        """All epoch scheduler fields pinned to one archive block."""
        ...

    def epoch_index(self, netuid: int, block_number: int) -> int:
        """``SubnetEpochIndex`` at an exact archive block (binary-search seam)."""
        ...

    def metagraph(
        self, netuid: int, block_number: int | None = None
    ) -> MetagraphView: ...

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[int], version_key: int
    ) -> tuple[bool, str, bool]:
        """Return ``(accepted, message, commit_reveal_enabled_at_submit)``."""
        ...

    def commit_reveal_enabled(self, netuid: int) -> bool: ...

    def query_weights(self, netuid: int, uid: int) -> list[tuple[int, int]]: ...

    def query_last_update(self, netuid: int, uid: int) -> int: ...

    def submitted_weights_at_finalized_head(
        self, netuid: int, hotkey: str
    ) -> SubmittedWeights | None:
        """Read one validator's Uids/Weights/LastUpdate at one finalized hash.

        ``None`` is the positive, block-pinned answer "registered but no weights";
        an unregistered hotkey or any unreadable/malformed storage must raise.
        """
        ...

    def pending_timelocked_commit(self, netuid: int, hotkey: str) -> bool:
        """True while a v10 timelocked (CRv4) weight commit for `hotkey` pends reveal.

        Bittensor v10 keys commits by ``(netuid_index, epoch)``.  This seam must
        inspect every live epoch bucket: the pinned SDK convenience getter uses
        ``page_size=1`` and then reads only ``result.records[0]``, so it can miss
        our commit across an epoch transition.  The legacy per-hotkey
        ``WeightCommits(netuid, hotkey)`` storage shape is not CRv4 state.
        """
        ...

    def uid_for_hotkey(self, hotkey: str, netuid: int) -> int | None: ...

    def subnet_owner_hotkey(self, netuid: int) -> str | None: ...

    def weights_rate_limit(self, netuid: int) -> int: ...

    def commitment_rate_limit(self) -> int: ...

    def commitment_usage(
        self, *, netuid: int, ss58: str, block_number: int
    ) -> _CommitmentUsageView:
        """Raw Commitments-pallet capacity state at one exact block."""
        ...

    def tempo(self, netuid: int) -> int: ...

    def block_time(self, block_number: int) -> datetime | None: ...

    def signer_hotkey(self) -> str: ...

    def sign_hotkey(self, payload: bytes) -> bytes: ...

    def set_commitment(self, *, netuid: int, payload: str) -> str:
        """Anchor the <=128-byte ascii `payload` on the Commitments pallet.

        v10.5.0 `set_commitment(data: str)` calls ``data.encode()`` itself, so the
        payload crosses this seam as a STR — passing bytes raises before submission
        and NO anchor is ever published. The adapter decodes the
        ascii anchor bytes to a str at its boundary so the transport always hands
        the SDK a str.
        """
        ...

    def get_commitment(
        self, *, netuid: int, ss58: str, block_number: int | None = None
    ) -> bytes | None:
        """The raw commitment payload `ss58` currently has on `netuid` (or None).

        Reads the Commitments pallet back for the anchor's account (#3, the third
        verification leg). Returns None when the account holds no commitment; raises
        on transport trouble. UNPROVEN like `set_commitment` — validate on testnet.
        """
        ...

    def get_commitment_block(
        self, *, netuid: int, ss58: str, block_number: int | None = None
    ) -> int | None:
        """The INCLUSION BLOCK of `ss58`'s current commitment on `netuid` (or None).

        The Commitments pallet stores a per-account record carrying the block the
        commitment was set at; this returns that block. It is the un-grindable half of
        the sampling beacon — the authority does not choose which
        block its anchor extrinsic lands in. None when the account holds no commitment;
        raises on transport trouble. UNPROVEN — validate on testnet.
        """
        ...

    def get_block_hash(self, block_number: int) -> str | None:
        """The chain block HASH for `block_number` (hex; the substrate block hash).

        The round-6 sampling beacon is `block_hash(close_block + K)` — the hash of a
        FUTURE FINALIZED block, chain-determined and un-grindable at log-build time
. Returns None when the chain has not produced `block_number`
        yet (the beacon is not finalized); raises on transport trouble. UNPROVEN —
        validate on testnet.
        """
        ...

    def close(self) -> None: ...


@dataclass
class _TransportGeneration:
    """One transport plus the mutex owned by calls already using that transport."""

    sequence: int
    transport: _SubtensorTransport
    lock: Any = field(default_factory=threading.RLock, repr=False)


# --------------------------------------------------------------------------------------
# Adapter configuration.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BittensorAdapterConfig:
    """Everything the real adapter/transport needs — mirrors the proven env sets."""

    #: Own validator hotkey ss58 (the identity that writes weights; the vector is
    #: read back under it — SubmittedWeightsReader). REQUIRED.
    validator_hotkey: str = ""
    #: The ss58 whose on-chain COMMITMENT carries the epoch-log anchors (the Scoring
    #: Authority's account, the project design record §4). `read_anchor` reads THIS
    #: account's commitment back as the independent third verification leg (#3).
    #: Empty falls back to `validator_hotkey` (a self-anchoring single-node deployment).
    anchor_hotkey: str = ""
    #: Cross-process lane for this wallet's single mutable commitment slot.
    anchor_writer_lock_path: Path | None = None
    anchor_writer_lock_timeout_seconds: float = 30.0
    network: str = "finney"  # "finney" | "test" | "archive"
    netuid: int = 85
    #: Explicit wss:// endpoint. When set it BEATS `network` (prod pins a URL).
    endpoint: str = ""
    #: Ordered additional archive-capable endpoints. The pinned SDK receives
    #: these as fallback endpoints; an empty tuple preserves one-endpoint launch
    #: config while allowing a second public archive RPC without a code change.
    fallback_endpoints: tuple[str, ...] = ()
    #: On-disk btcli wallet (takes precedence over the seed env when both exist).
    wallet_name: str = ""
    wallet_hotkey: str = ""
    wallet_path: str = ""
    #: Name of the env var holding the hotkey seed/mnemonic (never the value).
    hotkey_seed_env: str = "VIDAIO_HOTKEY_SEED"
    #: Explicit convergence fence. Keep synchronized with EPOCH_LOG_SCHEMA_VERSION;
    #: zero is reserved for dependency-free report/test overlays.
    version_key: int = 16
    connect_timeout_seconds: float = 30.0
    rpc_timeout_seconds: float = 30.0
    #: A successful non-CR SDK response is only the submitter's claim.  Observe
    #: finalized Uids/Weights/LastUpdate repeatedly so ordinary archive/finality
    #: read lag does not turn a landed write into an ambiguous retry.  Five
    #: observations spaced by one expected block mirrors a proven production validator.
    weight_readback_attempts: int = 5
    weight_readback_delay_seconds: float = 12.0
    metagraph_ttl_seconds: float = 120.0
    reconnect_after_consecutive_failures: int = 3
    #: Internal construction mode for public/read-only services. In this mode no
    #: wallet or seed is loaded, validator identity lookups are skipped, and every
    #: signing/mutation method raises :class:`ReadOnlyChainError` locally.
    read_only: bool = False

    def __post_init__(self) -> None:
        """Reject launch-breaking adapter configuration before opening a socket."""
        if not self.read_only and not self.validator_hotkey.strip():
            raise ValueError("validator_hotkey is required for bittensor mode")
        if self.netuid < 0:
            raise ValueError(f"netuid must be non-negative, got {self.netuid}")
        if self.version_key < 0:
            raise ValueError(
                f"version_key must be non-negative, got {self.version_key}"
            )
        if not 0 < self.anchor_writer_lock_timeout_seconds <= 60:
            raise ValueError("anchor_writer_lock_timeout_seconds must be in (0, 60]")
        if not self.read_only and bool(self.wallet_name.strip()) != bool(
            self.wallet_hotkey.strip()
        ):
            raise ValueError(
                "wallet_name and wallet_hotkey must either both be set or both be empty"
            )
        if not self.read_only and not self.hotkey_seed_env.strip():
            raise ValueError("hotkey_seed_env must not be empty")
        for name in (
            "connect_timeout_seconds",
            "rpc_timeout_seconds",
            "metagraph_ttl_seconds",
        ):
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.weight_readback_attempts < 1:
            raise ValueError("weight_readback_attempts must be at least 1")
        if self.weight_readback_delay_seconds < 0:
            raise ValueError("weight_readback_delay_seconds must be non-negative")
        if self.reconnect_after_consecutive_failures < 1:
            raise ValueError("reconnect_after_consecutive_failures must be at least 1")
        normalized_fallbacks = tuple(value.strip() for value in self.fallback_endpoints)
        if any(not value for value in normalized_fallbacks):
            raise ValueError("fallback_endpoints must not contain empty values")
        if len(set(normalized_fallbacks)) != len(normalized_fallbacks):
            raise ValueError("fallback_endpoints must not contain duplicates")
        if self.endpoint.strip() and self.endpoint.strip() in normalized_fallbacks:
            raise ValueError("fallback_endpoints must not repeat the primary endpoint")
        object.__setattr__(self, "fallback_endpoints", normalized_fallbacks)


# --------------------------------------------------------------------------------------
# The adapter.
# --------------------------------------------------------------------------------------


class BittensorChainAdapter:
    """ChainAdapter + SubmittedWeightsReader over one long-lived Subtensor socket.

    Construction fails fast: with no injected transport it opens the real socket
    NOW (which loads the wallet and connects), so a pod that forgot its seed or
    cannot reach the endpoint crashes at startup rather than idling forever
    without submitting. Tests inject a fake transport
    and never touch bittensor.
    """

    def __init__(
        self,
        config: BittensorAdapterConfig,
        *,
        transport: _SubtensorTransport | None = None,
        connect_transport: Callable[[], _SubtensorTransport] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._clock = clock
        self._log = get_logger("chain.bittensor")
        #: Coroutine-level write serialization (the sync SDK is not
        #: concurrency-safe on one socket). In the NORMAL case only one submit
        #: coroutine dispatches a worker thread at a time.
        self._write_lock = asyncio.Lock()
        # Transport lifecycle state is separate from each generation's call lock.
        # A timed-out synchronous SDK worker may own its generation lock forever;
        # reconnect must still be able to atomically install a fresh generation.
        self._transport_state_lock = threading.RLock()
        self._main_reconnect_lock = threading.Lock()
        self._anchor_reconnect_lock = threading.Lock()
        self._retirement_lock = threading.Lock()
        self._unreaped_generations: dict[str, threading.Thread] = {}
        self._retired_generation_sequences: set[int] = set()
        self._next_transport_sequence = 0

        # How the socket is (re)built. Tests pass a fake factory; production
        # defaults to the real, lazily-imported one. When only a transport is
        # given, reconnect returns that same object.
        if connect_transport is not None:
            self._connect_transport = connect_transport
            self._anchor_transport_is_shared = False
        elif transport is not None:
            self._connect_transport = lambda: transport
            # Injected unit-test transports are intentionally single objects.
            self._anchor_transport_is_shared = True
        else:
            self._connect_transport = lambda: _connect_real_transport(config)
            self._anchor_transport_is_shared = False

        self._transport: _SubtensorTransport = (
            transport if transport is not None else self._connect_transport()
        )
        self._validate_transport_identity(self._transport)
        self._main_generation = self._new_generation(self._transport)
        # Historical private aliases retained for diagnostics/tests. They always
        # point at the current generation and are swapped together under
        # ``_transport_state_lock``.
        self._socket_lock = self._main_generation.lock
        self._anchor_transport: _SubtensorTransport | None = (
            self._transport if self._anchor_transport_is_shared else None
        )
        self._anchor_generation: _TransportGeneration | None = (
            self._main_generation if self._anchor_transport_is_shared else None
        )
        self._anchor_socket_lock = (
            self._main_generation.lock
            if self._anchor_transport_is_shared
            else threading.RLock()
        )
        self._anchor_condemned = False
        self._anchor_consecutive_failures = 0

        # Reconnect discipline: reconnect ONLY after
        # N consecutive RAISED failures, or when the socket is condemned (a fired
        # set_weights timeout — reuse after an abandoned wait is what leaked and
        # OOMed the pod). A clean set_weights resets the counter; a clean read
        # does NOT (a deliberate asymmetry — only a submit proves the write path
        # healthy).
        self._consecutive_failures = 0
        self._condemned = False

        # Cached snapshot (reads are snapshots; refresh() owns the fetch).
        self._block = 0
        self._neurons: list[ChainNeuron] = []
        self._own_uid: int | None = None
        self._own_last_update: int | None = None
        self._weights_rate_limit = 0
        self._last_successful_refresh: float | None = None
        self._last_refresh_error: str | None = None
        # Archive-proven runtime epoch boundaries are immutable once finalized.
        # Cache them by epoch id so the 15-second finalizer/auditor loops do not
        # repeat an O(log(chain height)) historical index search every pass.
        self._epoch_boundaries: dict[int, EpochBoundary] = {}

    def _validate_transport_identity(self, transport: _SubtensorTransport) -> None:
        """Prove writes and readback use one hotkey, including after reconnect."""
        if self._config.read_only:
            # A wallet-free transport deliberately has no signer identity. Reads
            # remain bound to the configured subnet and explicit read accounts.
            return
        # A configured readback identity that is not the key actually signing writes
        # makes every accepted vector look absent and strands intents forever.  The
        # real transport always exposes its signer; small injected test transports may
        # omit the optional introspection seam.
        signer_hotkey = getattr(transport, "signer_hotkey", None)
        if callable(signer_hotkey):
            try:
                actual = str(signer_hotkey())
            except Exception:
                try:
                    transport.close()
                finally:
                    raise RuntimeError(
                        "loaded wallet hotkey identity could not be read"
                    )
            if actual != self._config.validator_hotkey:
                try:
                    transport.close()
                finally:
                    raise RuntimeError(
                        "configured validator_hotkey does not match the loaded wallet "
                        f"hotkey ({self._config.validator_hotkey!r} != {actual!r}); "
                        "refusing to write under one identity and read back another"
                    )

    # -- identity / health surface -------------------------------------------------

    @property
    def validator_hotkey(self) -> str:
        return self._config.validator_hotkey

    @property
    def own_uid(self) -> int | None:
        return self._own_uid

    def blocks_since_last_update(self) -> int | None:
        """The primary 'am I actually weight-setting' gauge.

        current_block - own last_update, or None until both are known.
        """
        if self._own_last_update is None:
            return None
        return max(0, self._block - self._own_last_update)

    @property
    def last_successful_refresh(self) -> float | None:
        return self._last_successful_refresh

    @property
    def last_refresh_error(self) -> str | None:
        return self._last_refresh_error

    # -- socket discipline ---------------------------------------------------------

    def _new_generation(self, transport: _SubtensorTransport) -> _TransportGeneration:
        with self._transport_state_lock:
            sequence = self._next_transport_sequence
            self._next_transport_sequence += 1
        return _TransportGeneration(sequence=sequence, transport=transport)

    @property
    def unreaped_transport_generations(self) -> tuple[str, ...]:
        """Background retirements still waiting for an abandoned SDK call/close."""
        with self._retirement_lock:
            return tuple(sorted(self._unreaped_generations))

    def _retire_generation(self, lane: str, generation: _TransportGeneration) -> None:
        """Close an old generation without ever delaying installation of its heir.

        Real SDK ``close()`` takes the transport's internal mutex. A daemon RPC
        which exceeded our timeout can retain that mutex indefinitely, so closing
        inline recreates the timeout -> reconnect deadlock. Real transports and
        any busy injected generation are retired on a daemon thread and exposed
        through ``unreaped_transport_generations`` for operator diagnostics.
        """
        with self._retirement_lock:
            if generation.sequence in self._retired_generation_sequences:
                return
            self._retired_generation_sequences.add(generation.sequence)

        if not isinstance(generation.transport, _RealSubtensorTransport):
            acquired = generation.lock.acquire(blocking=False)
            if acquired:
                try:
                    generation.transport.close()
                except Exception:  # noqa: BLE001 - dead transport cleanup
                    pass
                finally:
                    generation.lock.release()
                return

        key = f"{lane}:{generation.sequence}"

        def _retire() -> None:
            try:
                with generation.lock:
                    generation.transport.close()
            except Exception:  # noqa: BLE001 - retirement never affects live calls
                pass
            finally:
                with self._retirement_lock:
                    self._unreaped_generations.pop(key, None)

        thread = threading.Thread(
            target=_retire,
            name=f"bt-retire-{lane}-{generation.sequence}",
            daemon=True,
        )
        with self._retirement_lock:
            self._unreaped_generations[key] = thread
            count = len(self._unreaped_generations)
        thread.start()
        self._log.warning(
            "retiring an old subtensor generation in the background",
            extra={"generation": key, "unreaped_generations": count},
        )

    def _main_reconnect_due(self) -> bool:
        return self._condemned or (
            self._consecutive_failures
            >= self._config.reconnect_after_consecutive_failures
        )

    def _reconnect(self, *, force: bool = True) -> None:
        """Atomically install a fresh socket; retire the old one asynchronously.

        Crucially, this method never acquires the old generation's call lock and
        never calls the old real transport's ``close()`` inline. A synchronous SDK
        worker abandoned after timeout can therefore remain wedged without freezing
        all subsequent reads and submissions.
        """
        with self._main_reconnect_lock:
            with self._transport_state_lock:
                if not force and not self._main_reconnect_due():
                    return
                failures = self._consecutive_failures
                condemned = self._condemned
            self._log.warning(
                "reconnecting the subtensor socket",
                extra={
                    "consecutive_failures": failures,
                    "condemned": condemned,
                },
            )
            replacement = self._connect_transport()
            self._validate_transport_identity(replacement)
            replacement_generation = self._new_generation(replacement)
            redundant = False
            old_generation: _TransportGeneration | None = None
            with self._transport_state_lock:
                if not force and not self._main_reconnect_due():
                    redundant = True
                else:
                    old_generation = self._main_generation
                    self._main_generation = replacement_generation
                    self._transport = replacement
                    self._socket_lock = replacement_generation.lock
                    if self._anchor_transport_is_shared:
                        self._anchor_generation = replacement_generation
                        self._anchor_transport = replacement
                        self._anchor_socket_lock = replacement_generation.lock
                    self._consecutive_failures = 0
                    self._condemned = False

            if redundant:
                self._retire_generation("redundant-main", replacement_generation)
                return
            assert old_generation is not None
            if old_generation.transport is not replacement:
                self._retire_generation("main", old_generation)
            else:
                # Only an injected singleton factory can do this. It cannot provide
                # timeout isolation, but must not be closed as its own replacement.
                self._log.warning(
                    "transport factory reused the condemned transport; reconnect "
                    "cannot isolate an abandoned SDK generation"
                )

    def _ensure_main_transport(self) -> None:
        with self._transport_state_lock:
            due = self._main_reconnect_due()
        if due:
            self._reconnect(force=False)

    @contextmanager
    def _transport_call(self):
        """Yield a serialized current transport, escaping a condemned busy lock."""
        while True:
            self._ensure_main_transport()
            with self._transport_state_lock:
                generation = self._main_generation

            # Poll instead of waiting forever: cancellation may condemn the
            # generation after we selected it but while another worker owns it.
            acquired = False
            while not acquired:
                acquired = generation.lock.acquire(timeout=0.05)
                if acquired:
                    break
                with self._transport_state_lock:
                    stale_or_due = (
                        generation is not self._main_generation
                        or self._main_reconnect_due()
                    )
                if stale_or_due:
                    break
            if not acquired:
                continue

            with self._transport_state_lock:
                stale_or_due = (
                    generation is not self._main_generation
                    or self._main_reconnect_due()
                )
            if stale_or_due:
                generation.lock.release()
                continue
            try:
                yield generation.transport
            finally:
                generation.lock.release()
            return

    def _transport_for_call(self) -> _SubtensorTransport:
        """Compatibility accessor; new call sites must use ``_transport_call``."""
        self._ensure_main_transport()
        with self._transport_state_lock:
            return self._main_generation.transport

    def _anchor_reconnect_due(self) -> bool:
        return (
            self._anchor_generation is None
            or self._anchor_condemned
            or (
                self._anchor_consecutive_failures
                >= self._config.reconnect_after_consecutive_failures
            )
        )

    def _reconnect_anchor(self) -> None:
        if self._anchor_transport_is_shared:
            self._ensure_main_transport()
            return
        with self._anchor_reconnect_lock:
            with self._transport_state_lock:
                if not self._anchor_reconnect_due():
                    return
            replacement = self._connect_transport()
            self._validate_transport_identity(replacement)
            replacement_generation = self._new_generation(replacement)
            old_generation: _TransportGeneration | None = None
            redundant = False
            with self._transport_state_lock:
                if not self._anchor_reconnect_due():
                    redundant = True
                else:
                    old_generation = self._anchor_generation
                    if replacement is self._main_generation.transport:
                        self._anchor_transport_is_shared = True
                        self._anchor_generation = self._main_generation
                        self._anchor_transport = self._main_generation.transport
                        self._anchor_socket_lock = self._main_generation.lock
                    else:
                        self._anchor_generation = replacement_generation
                        self._anchor_transport = replacement
                        self._anchor_socket_lock = replacement_generation.lock
                    self._anchor_condemned = False
                    self._anchor_consecutive_failures = 0

            if redundant:
                self._retire_generation("redundant-anchor", replacement_generation)
                return
            if (
                old_generation is not None
                and old_generation.transport is not replacement
                and old_generation is not self._main_generation
            ):
                self._retire_generation("anchor", old_generation)

    @contextmanager
    def _anchor_transport_call(self):
        """Yield the dedicated current commitment generation without wedging."""
        while True:
            if self._anchor_transport_is_shared:
                with self._transport_call() as transport:
                    yield transport
                return
            self._reconnect_anchor()
            with self._transport_state_lock:
                generation = self._anchor_generation
            assert generation is not None

            acquired = False
            while not acquired:
                acquired = generation.lock.acquire(timeout=0.05)
                if acquired:
                    break
                with self._transport_state_lock:
                    stale_or_due = (
                        generation is not self._anchor_generation
                        or self._anchor_reconnect_due()
                    )
                if stale_or_due:
                    break
            if not acquired:
                continue
            with self._transport_state_lock:
                stale_or_due = (
                    generation is not self._anchor_generation
                    or self._anchor_reconnect_due()
                )
            if stale_or_due:
                generation.lock.release()
                continue
            try:
                yield generation.transport
            finally:
                generation.lock.release()
            return

    def _anchor_transport_for_call(self) -> _SubtensorTransport:
        """Compatibility accessor; anchor calls use ``_anchor_transport_call``."""
        if self._anchor_transport_is_shared:
            return self._transport_for_call()
        self._reconnect_anchor()
        assert self._anchor_transport is not None
        return self._anchor_transport

    def _note_raise(self) -> None:
        self._consecutive_failures += 1

    def _note_clean_submit(self) -> None:
        self._consecutive_failures = 0

    def _note_anchor_raise(self) -> None:
        if self._anchor_transport_is_shared:
            self._note_raise()
        else:
            self._anchor_consecutive_failures += 1

    def _note_anchor_clean_submit(self) -> None:
        if self._anchor_transport_is_shared:
            self._note_clean_submit()
        else:
            self._anchor_consecutive_failures = 0

    # -- reads (cached snapshot) ---------------------------------------------------

    def current_block(self) -> int:
        """Last observed head; 0 until the first successful refresh."""
        return self._block

    def finalized_block(self) -> int:
        """Return the latest GRANDPA-finalized height from the live socket.

        ``current_block()`` is only the last observed best head and may be
        reverted.  Epoch close snapshots and sampling beacons must instead be
        gated by consensus finality, so this read never falls back to the cached
        head on an RPC failure.
        """
        try:
            with self._transport_call() as transport:
                value = transport.finalized_block()
            if isinstance(value, bool):
                raise TypeError("boolean finalized height")
            finalized = int(value)
            if finalized < 0:
                raise ValueError(f"negative finalized height {finalized}")
            return finalized
        except Exception as exc:  # noqa: BLE001 - unknown finality must HOLD
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read the GRANDPA-finalized block: {type(exc).__name__}: {exc}"
            ) from exc

    def _require_bound_netuid(self, netuid: int) -> None:
        if netuid != self._config.netuid:
            raise ValueError(
                f"adapter is bound to subnet {self._config.netuid}, not {netuid}"
            )

    def _epoch_index_at(self, netuid: int, block_number: int) -> int:
        with self._transport_call() as transport:
            value = transport.epoch_index(netuid, block_number)
        if isinstance(value, bool):
            raise TypeError("boolean SubnetEpochIndex")
        index = int(value)
        if index < 0:
            raise ValueError(f"negative SubnetEpochIndex {index}")
        return index

    def _epoch_schedule_at(self, netuid: int, block_number: int) -> EpochScheduleView:
        with self._transport_call() as transport:
            state = transport.epoch_schedule(netuid, block_number)
        if not isinstance(state, EpochScheduleView):
            raise TypeError(
                "epoch_schedule returned "
                f"{type(state).__name__}, expected EpochScheduleView"
            )
        if state.block != block_number:
            raise ValueError(
                f"epoch schedule was pinned to block {state.block}, not {block_number}"
            )
        if (
            min(
                state.last_epoch_block,
                state.pending_epoch_at,
                state.subnet_epoch_index,
                state.tempo,
                state.blocks_since_last_step,
            )
            < 0
        ):
            raise ValueError(f"epoch schedule contains a negative field: {state!r}")
        if state.last_epoch_block > block_number:
            raise ValueError(
                f"LastEpochBlock {state.last_epoch_block} is after state block "
                f"{block_number}"
            )
        return state

    def _find_epoch_close_block(
        self,
        *,
        netuid: int,
        epoch_id: int,
        finalized: int,
        finalized_index: int | None = None,
    ) -> int | None:
        cached = self._epoch_boundaries.get(epoch_id)
        if cached is not None:
            if cached.close_block > finalized:
                raise ValueError(
                    f"cached epoch {epoch_id} close {cached.close_block} is after "
                    f"finalized block {finalized}"
                )
            return cached.close_block

        if finalized < 2:
            return None
        high_index = (
            self._epoch_index_at(netuid, finalized)
            if finalized_index is None
            else finalized_index
        )
        if high_index < epoch_id:
            return None  # positive: this runtime epoch has not finalized yet

        # Find a RECENT readable block below the target before binary-searching
        # the first block whose monotonic runtime counter is >= ``epoch_id``.
        #
        # SubnetEpochIndex was introduced by a runtime migration.  Querying block
        # 1 is therefore not a valid archive test: a perfectly healthy archive
        # node cannot decode storage which did not exist under that block's old
        # metadata.  Search backwards exponentially from the known-readable
        # finalized head instead.  If that crosses the migration, locate the
        # first readable post-migration block and accept it only when its seeded
        # counter is still BELOW the requested transition.  Asking for the seeded
        # counter itself remains unverifiable and fails closed because there is no
        # exact E-1 -> E predecessor transition.
        high = finalized
        high_index_for_search = high_index
        distance = 1
        while True:
            probe = max(1, finalized - distance)
            if probe >= high:
                raise ValueError(
                    f"epoch {epoch_id} has no earlier block from which to prove "
                    "its runtime counter transition"
                )
            try:
                probe_index = self._epoch_index_at(netuid, probe)
            except Exception as unavailable:  # noqa: BLE001 - migration prefix
                # Storage availability is a prefix boundary for this runtime
                # item.  Bisect that boundary without treating the unavailable
                # pre-migration state as an epoch-index value.
                unreadable = probe
                readable = high
                readable_index = high_index_for_search
                last_unavailable = unavailable
                while unreadable + 1 < readable:
                    middle = (unreadable + readable) // 2
                    try:
                        middle_index = self._epoch_index_at(netuid, middle)
                    except Exception as exc:  # noqa: BLE001 - same prefix search
                        unreadable = middle
                        last_unavailable = exc
                    else:
                        readable = middle
                        readable_index = middle_index
                if readable_index >= epoch_id:
                    raise ValueError(
                        f"epoch {epoch_id} has no archive-readable predecessor: "
                        f"the first readable SubnetEpochIndex at block {readable} "
                        f"is {readable_index}; the preceding runtime state is "
                        f"unavailable ({type(last_unavailable).__name__}: "
                        f"{last_unavailable}). Refusing to treat a migration-seeded "
                        "counter as an epoch close"
                    ) from unavailable
                low = readable
                break

            if probe_index < epoch_id:
                low = probe
                break
            high = probe
            high_index_for_search = probe_index
            if probe == 1:
                raise ValueError(
                    f"epoch {epoch_id} has no archive-readable predecessor below "
                    f"SubnetEpochIndex {probe_index}; refusing to synthesize a close"
                )
            distance *= 2

        # A one-step transition is verified below. This is intentionally based
        # on historical state rather than tempo arithmetic: tempo can change,
        # an owner can trigger early, and the runtime can defer a due epoch.
        while low + 1 < high:
            middle = (low + high) // 2
            middle_index = self._epoch_index_at(netuid, middle)
            if middle_index >= epoch_id:
                high = middle
            else:
                low = middle
        candidate = high

        if candidate < 2:
            raise ValueError(
                f"epoch {epoch_id}'s predecessor block cannot be archive-verified "
                f"(candidate close {candidate})"
            )

        before = self._epoch_schedule_at(netuid, candidate - 1)
        at_close = self._epoch_schedule_at(netuid, candidate)
        expected_before = epoch_id - 1
        if (
            before.subnet_epoch_index != expected_before
            or at_close.subnet_epoch_index != epoch_id
            or at_close.last_epoch_block != candidate
        ):
            raise ValueError(
                f"block {candidate} is not an exact runtime epoch-{epoch_id} close: "
                f"index {before.subnet_epoch_index}->{at_close.subnet_epoch_index} "
                f"(expected {expected_before}->{epoch_id}), "
                f"LastEpochBlock={at_close.last_epoch_block}. This can indicate a "
                "tempo-cycle reset, a migration counter jump, or inconsistent "
                "archive state; refusing to synthesize a boundary"
            )

        boundary = EpochBoundary(epoch_id=epoch_id, close_block=candidate)
        self._epoch_boundaries[epoch_id] = boundary
        return candidate

    def epoch_close_block(self, *, netuid: int, epoch_id: int) -> int | None:
        """Archive-prove the exact close block for one runtime epoch index.

        ``None`` means the requested index is still in the future relative to
        the finalized head.  Missing/pruned history, a migration counter jump,
        or anything other than the exact ``E-1 -> E`` transition is UNKNOWN and
        raises so callers HOLD rather than accept a synthetic close.
        """
        self._require_bound_netuid(netuid)
        if isinstance(epoch_id, bool) or epoch_id < 1:
            raise ValueError(f"epoch_id must be a positive integer, got {epoch_id!r}")
        try:
            finalized = self.finalized_block()
            return self._find_epoch_close_block(
                netuid=netuid, epoch_id=epoch_id, finalized=finalized
            )
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - unverifiable history must HOLD
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot derive subnet {netuid} epoch {epoch_id}'s close from "
                f"historical SubnetEpochIndex transitions: {type(exc).__name__}: {exc}"
            ) from exc

    def latest_closed_epoch(self, *, netuid: int) -> EpochBoundary | None:
        """Latest archive-proven epoch transition at the finalized head."""
        self._require_bound_netuid(netuid)
        try:
            finalized = self.finalized_block()
            if finalized < 2:
                return None
            latest_index = self._epoch_index_at(netuid, finalized)
            if latest_index < 1:
                return None
            close_block = self._find_epoch_close_block(
                netuid=netuid,
                epoch_id=latest_index,
                finalized=finalized,
                finalized_index=latest_index,
            )
            if close_block is None:  # impossible after latest_index >= 1
                raise ValueError(
                    f"runtime index {latest_index} has no finalized close block"
                )
            return EpochBoundary(epoch_id=latest_index, close_block=close_block)
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - unknown schedule must HOLD
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot derive subnet {netuid}'s latest finalized epoch: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def neurons(self) -> list[ChainNeuron]:
        """The cached metagraph snapshot.

        Raises ChainStateUnavailable if refresh() has NEVER succeeded — an empty
        list would be indistinguishable from a genuinely empty subnet, which is
        exactly how a startup race becomes a 'successful' empty round with
        silently omitted weights (adapter.py freshness contract).
        """
        if self._last_successful_refresh is None:
            raise ChainStateUnavailable(
                "no chain snapshot yet"
                + (
                    f" (last refresh error: {self._last_refresh_error})"
                    if self._last_refresh_error
                    else " (refresh() has not been called)"
                )
            )
        return list(self._neurons)

    def snapshot_age(self, now: float) -> float | None:
        if self._last_successful_refresh is None:
            return None
        return max(0.0, now - self._last_successful_refresh)

    def has_fresh_snapshot(self, now: float, max_age_seconds: float) -> bool:
        age = self.snapshot_age(now)
        return age is not None and age <= max_age_seconds

    def refresh(self) -> None:
        """Pull one metagraph read into the cached snapshot. NEVER raises.

        Adapter-owned throttling (TTL). On failure the previous snapshot is KEPT
        (a transient RPC failure must never zero the neuron list — risk 8) and the
        failure is reported through last_refresh_error / has_fresh_snapshot. Also
        refreshes the cached head and own-uid LastUpdate so set_weights can
        pre-gate.
        """
        now = self._clock()
        if (
            self._last_successful_refresh is not None
            and now - self._last_successful_refresh < self._config.metagraph_ttl_seconds
        ):
            return  # inside the TTL — cheap snapshot read, no RPC

        try:
            # Reads share the exact socket used by the blocking write path.  Hold
            # the mutex across the whole coherent snapshot so refresh cannot run on
            # a socket whose caller-abandoned set_weights worker is still alive.
            with self._transport_call() as transport:
                view = transport.metagraph(self._config.netuid)
                neurons = self._map_metagraph(view)
                block = transport.current_block()
                if self._config.read_only:
                    # Registration-only consumers need the live metagraph, not a
                    # fictitious "own" uid. Avoid identity-specific reads entirely.
                    own_uid = None
                    own_last_update = None
                    rate_limit = 0
                else:
                    own_uid = transport.uid_for_hotkey(
                        self._config.validator_hotkey, self._config.netuid
                    )
                    own_last_update = (
                        transport.query_last_update(self._config.netuid, own_uid)
                        if own_uid is not None
                        else None
                    )
                    # weights_rate_limit is effectively static; read it best-effort here
                    # so the pre-gate has it (0 = pre-gate disabled; the chain re-checks).
                    try:
                        rate_limit = transport.weights_rate_limit(self._config.netuid)
                    except Exception:  # noqa: BLE001 - optional pre-gate input only
                        rate_limit = self._weights_rate_limit
        except Exception as exc:  # noqa: BLE001 - refresh() MUST NOT raise
            self._note_raise()
            self._last_refresh_error = f"{type(exc).__name__}: {exc}"
            self._log.warning(
                "metagraph refresh failed — keeping cached snapshot",
                extra={
                    "error": self._last_refresh_error,
                    "ever_refreshed": self._last_successful_refresh is not None,
                    "consecutive_failures": self._consecutive_failures,
                },
            )
            return

        self._neurons = neurons
        self._block = block
        self._own_uid = own_uid
        self._own_last_update = own_last_update
        self._weights_rate_limit = rate_limit
        self._last_successful_refresh = self._clock()
        self._last_refresh_error = None

    def _map_metagraph(self, view: MetagraphView) -> list[ChainNeuron]:
        """metagraph arrays -> list[ChainNeuron].

        Miner HTTP discovery preserves both ``metagraph.axons[uid].ip`` and its
        advertised ``port``. A neuron that does not serve an axon normally has
        ``0.0.0.0``/port 0; dispatch retains it for accounting/dedup semantics,
        while the HTTP peer-address policy refuses to dial that undialable pair.
        """
        neurons: list[ChainNeuron] = []
        n = len(view.hotkeys)
        for uid in range(n):
            neurons.append(
                ChainNeuron(
                    uid=uid,
                    hotkey=view.hotkeys[uid],
                    coldkey=view.coldkeys[uid],
                    ip=view.axon_ips[uid] if uid < len(view.axon_ips) else "0.0.0.0",
                    alpha_stake=float(view.alpha_stake[uid]),
                    emission=float(view.emission[uid]),
                    is_validator=bool(view.validator_permit[uid]),
                    last_update=int(view.last_update[uid]),
                    registration_block=(
                        int(view.registration_block[uid])
                        if uid < len(view.registration_block)
                        else 0
                    ),
                    incentive=(
                        float(view.incentive[uid]) if uid < len(view.incentive) else 0.0
                    ),
                    axon_port=(
                        view.axon_ports[uid] if uid < len(view.axon_ports) else None
                    ),
                )
            )
        return neurons

    def neurons_at(self, block_number: int) -> list[ChainNeuron]:
        """Read a metagraph pinned to ``block_number`` without mutating the cache.

        Epoch finalization must bind registration, stake and validator state to the
        epoch-close block.  A current-head snapshot relabelled as historical is not a
        historical read, so any pruned/archive/RPC failure raises and the caller HOLDs.
        """
        if block_number < 0:
            raise ValueError(f"block_number must be non-negative, got {block_number}")
        try:
            with self._transport_call() as transport:
                view = transport.metagraph(self._config.netuid, block_number)
                if int(view.block) != block_number:
                    raise ValueError(
                        f"transport returned metagraph block {view.block}, not the "
                        f"requested historical block {block_number}"
                    )
            return self._map_metagraph(view)
        except Exception as exc:  # noqa: BLE001 - failed historical read is UNKNOWN
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read metagraph for subnet {self._config.netuid} at block "
                f"{block_number}: {type(exc).__name__}: {exc}"
            ) from exc

    # -- submitted-weights readback (SubmittedWeightsReader) -----------------------

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        """The vector the latest FINALIZED chain state records for `hotkey` (§d).

        Commit-reveal AWARE: while a v10 timelocked commit is pending for this
        hotkey the `Weights` storage still holds the PREVIOUS vector, so a literal
        read would look like 'a different vector, recorded before my attempt' —
        the one positive DENIAL, which would bury an intent whose commit is merely
        waiting for its reveal window. So a pending commit RAISES
        (ChainStateUnavailable == UNKNOWN, which HOLDS the intent) rather than
        answering (§d commit-reveal caveat).

        - pending commit for this hotkey -> raise (UNKNOWN);
        - Weights empty AND hotkey registered -> None (positive 'no weights');
        - otherwise -> SubmittedWeights(raw u16, block=LastUpdate) — Uids,
          Weights and LastUpdate are pinned to one GRANDPA-finalized hash; u16 is
          RAW, NOT renormalized (the weight-setter puts both sides on the grid);
        - anything unreadable (RPC failure, unresolvable uid, decode error) ->
          raise ChainStateUnavailable. NEVER None for a failed read: None denies,
          and a denied intent is abandoned unpublished.
        """
        try:
            # This is most commonly called immediately after an ambiguous write.
            # Serialize the pending probe + uid/vector/LastUpdate reads against the
            # write worker, otherwise readback can reuse the socket while an
            # asyncio-cancelled to_thread submit is still subscribed on it.
            with self._transport_call() as transport:
                netuid = self._config.netuid

                if transport.pending_timelocked_commit(netuid, hotkey):
                    raise ChainStateUnavailable(
                        f"a weight commit for {hotkey!r} is pending reveal;"
                        " on-chain Weights still hold the pre-commit vector — UNKNOWN,"
                        " not DENIED (commit-reveal caveat)"
                    )

                # Production reads Uids + Weights + LastUpdate at ONE GRANDPA-
                # finalized hash.  Mixing three best-head reads can manufacture a
                # vector/block pairing that never coexisted, and trusting the SDK's
                # submit verdict without this storage proof recreates a production
                # false-success incident (Aug 2026).  The fallback keeps deliberately-small
                # third-party/injected transports source-compatible; the real
                # transport always implements the finalized reader and its startup
                # contract is pinned below.
                finalized_reader = getattr(
                    transport, "submitted_weights_at_finalized_head", None
                )
                if callable(finalized_reader):
                    report = finalized_reader(netuid, hotkey)
                    if report is not None and not isinstance(report, SubmittedWeights):
                        raise TypeError(
                            "submitted_weights_at_finalized_head returned "
                            f"{type(report).__name__}, expected SubmittedWeights or None"
                        )
                    return report

                uid = transport.uid_for_hotkey(hotkey, netuid)
                if uid is None:
                    raise ChainStateUnavailable(
                        f"hotkey {hotkey!r} is not registered on subnet {netuid}"
                    )

                raw = transport.query_weights(netuid, uid)
                last = transport.query_last_update(netuid, uid)
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed read is UNKNOWN, never None
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read on-chain weights for {hotkey!r}:"
                f" {type(exc).__name__}: {exc}"
            ) from exc

        if not raw:
            # Registered, but no positive weight recorded — the one positive
            # answer that can deny an intent on its own.
            return None
        return SubmittedWeights(
            weights={int(u): float(w) for u, w in raw}, block=int(last)
        )

    def commit_reveal_enabled(self) -> bool:
        """Whether the bound subnet currently uses CRv4 weight commits.

        This is a live, fail-closed read. A caller deciding whether a successful
        SDK response means "active vector" or merely "commit accepted" must never
        guess from stale configuration.
        """
        try:
            with self._transport_call() as transport:
                enabled = transport.commit_reveal_enabled(self._config.netuid)
            if not isinstance(enabled, bool):
                raise TypeError(
                    f"commit_reveal_enabled returned {type(enabled).__name__}, not bool"
                )
            return enabled
        except Exception as exc:  # noqa: BLE001 - unknown mode must HOLD
            self._note_raise()
            raise ChainStateUnavailable(
                "cannot read commit-reveal mode for subnet "
                f"{self._config.netuid}: {type(exc).__name__}: {exc}"
            ) from exc

    def weight_commit_pending(self, hotkey: str) -> bool:
        """Whether ``hotkey`` has a timelocked CRv4 commit awaiting reveal."""
        try:
            with self._transport_call() as transport:
                return bool(
                    transport.pending_timelocked_commit(self._config.netuid, hotkey)
                )
        except Exception as exc:  # noqa: BLE001 - false would permit a second write
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read pending weight commit for {hotkey!r} on subnet "
                f"{self._config.netuid}: {type(exc).__name__}: {exc}"
            ) from exc

    def commitment_rate_limit(self) -> int:
        """Return the runtime's generic transaction-rate limit in blocks.

        This is retained only as generic chain diagnostics.  The current
        Commitments pallet does *not* enforce this value; callers deciding whether
        an anchor can land must use :meth:`commitment_capacity` instead.
        """
        try:
            with self._transport_call() as transport:
                raw = transport.commitment_rate_limit()
            if isinstance(raw, bool):
                raise TypeError("boolean transaction rate limit")
            value = int(raw)
            if value < 0:
                raise ValueError(f"negative transaction rate limit {value}")
            return value
        except Exception as exc:  # noqa: BLE001 - unreadable limit is UNKNOWN
            self._note_raise()
            raise ChainStateUnavailable(
                "cannot read the commitment transaction-rate limit: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def commitment_capacity(self, netuid: int, hotkey: str) -> CommitmentCapacity:
        """Read the exact per-epoch Commitments-pallet byte budget.

        ``MaxSpace``, ``UsedSpaceOf(netuid, hotkey)``, and the subnet epoch are
        pinned to the same best-head block.  An absent usage row means the account
        has spent zero bytes; a row from an older epoch is retained as diagnostics
        but has effective usage zero, matching the pallet's next-write reset.

        No runtime default is duplicated here: an unavailable or malformed storage
        read raises :class:`ChainStateUnavailable` so callers HOLD instead of
        assuming capacity.
        """
        self._require_bound_netuid(netuid)
        if not isinstance(hotkey, str) or not hotkey.strip():
            raise ValueError("commitment-capacity hotkey must be non-empty")
        try:
            # Keep all three reads on one transport generation and pin the latter
            # two to the exact head returned by the first.
            with self._transport_call() as transport:
                raw_block = transport.current_block()
                if isinstance(raw_block, bool):
                    raise TypeError("boolean current block")
                block = int(raw_block)
                if block < 1:
                    raise ValueError(f"invalid current block {block}")

                raw_epoch = transport.epoch_index(netuid, block)
                if isinstance(raw_epoch, bool):
                    raise TypeError("boolean subnet epoch index")
                current_epoch = int(raw_epoch)
                if current_epoch < 0:
                    raise ValueError(f"negative subnet epoch index {current_epoch}")

                usage = transport.commitment_usage(
                    netuid=netuid, ss58=hotkey, block_number=block
                )
            if not isinstance(usage, _CommitmentUsageView):
                raise TypeError(
                    "commitment_usage returned "
                    f"{type(usage).__name__}, expected _CommitmentUsageView"
                )
            if usage.block != block:
                raise ValueError(
                    f"commitment usage was pinned to block {usage.block}, not {block}"
                )
            effective_used = (
                usage.used_space if usage.usage_epoch == current_epoch else 0
            )
            return CommitmentCapacity(
                netuid=netuid,
                hotkey=hotkey,
                block=block,
                current_epoch=current_epoch,
                usage_epoch=usage.usage_epoch,
                max_space=usage.max_space,
                reported_used_space=usage.used_space,
                used_space=effective_used,
            )
        except Exception as exc:  # noqa: BLE001 - unreadable budget is UNKNOWN
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read Commitments capacity for {hotkey!r} on subnet "
                f"{netuid}: {type(exc).__name__}: {exc}"
            ) from exc

    # -- writes --------------------------------------------------------------------

    async def _wait_for_exact_non_cr_weight_proof(
        self,
        *,
        expected: dict[int, int],
        after_block: int,
    ) -> int:
        """Poll finalized storage after an SDK non-CR success claim.

        ``wait_for_finalization=True`` waits on the submit socket, but the
        independent archive read used for proof can still trail it briefly.  A
        single immediate miss caused a landed write to enter the ambiguous retry
        path in the first live testnet epoch.  Observe the exact finalized
        Uids/Weights/LastUpdate tuple for a bounded number of block-spaced reads;
        no second extrinsic is emitted anywhere in this method.

        Every observation retains the strict proof contract: the runtime bytes
        must equal ``expected`` and LastUpdate must be strictly newer than the
        pre-submit metagraph block.  Exhaustion raises the last proof error, so
        the caller preserves UNKNOWN/condemn/reconciliation semantics rather than
        trusting the SDK claim.
        """
        last_error: Exception | None = None
        for observation in range(1, self._config.weight_readback_attempts + 1):
            if observation > 1 and self._config.weight_readback_delay_seconds > 0:
                await asyncio.sleep(self._config.weight_readback_delay_seconds)
            try:
                report = await asyncio.to_thread(
                    self.submitted_weights, self._config.validator_hotkey
                )
                return _require_exact_finalized_weight_proof(
                    report,
                    expected=expected,
                    after_block=after_block,
                )
            except Exception as exc:  # noqa: BLE001 - every miss remains unproven
                last_error = exc
                if observation < self._config.weight_readback_attempts:
                    self._log.info(
                        "non-CR weight write is not visible in exact finalized "
                        "storage yet; waiting before the next proof observation",
                        extra={
                            "observation": observation,
                            "observations": self._config.weight_readback_attempts,
                            "delay_seconds": (
                                self._config.weight_readback_delay_seconds
                            ),
                            "proof_error": f"{type(exc).__name__}: {exc}",
                        },
                    )
        assert last_error is not None  # attempts is startup-validated as >= 1
        raise last_error

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        """Submit a weight vector (§d).

        Quantizes to VIDAIO's u16 sum grid, verifies the complete vector against a
        fresh metagraph on the (uid, hotkey) BINDING, pre-gates on the
        weights_rate_limit window, then submits under the socket mutex in a worker
        thread. A deregistered/recycled target rejects the whole attempt before any
        write: silently dropping it would renormalize its fixed share into survivors
        and mutate the authenticated authority vector.

        `hotkeys` is the intended uid -> hotkey binding the vector was scored
        against; when supplied, a uid whose CURRENT metagraph hotkey no longer
        matches makes the exact vector temporarily unsafe and the attempt fails
        cleanly, so neither the new occupant nor the surviving targets receive a
        rewritten allocation. Callers SHOULD pass it (weightsetter companion
        change); without it, verification falls back to uid-liveness only.

        THE INCLUSION/FINALIZATION WAIT IS INTENTIONALLY LONG AND MUST NOT BE
        TIMEOUT-BOUNDED BY THE CALLER: `asyncio.to_thread` cannot be
        cancelled, so a caller `with_timeout` that fires does NOT stop the worker
        thread — it keeps the SDK's per-submit block subscription open (the leak
        that OOMed a prior production validator's pod). The socket mutex is
        held for the whole wait so a cancelled attempt's thread cannot be run over.
        On cancellation we CONDEMN that transport generation; the next call
        atomically installs a fresh socket and retires the old generation in the
        background without waiting for its worker or ``close()``. Callers must still
        NOT wrap this coroutine in with_timeout because the abandoned extrinsic's
        chain fate remains ambiguous and must be reconciled from readback.
        """
        if self._config.read_only:
            raise ReadOnlyChainError(
                "wallet-free bittensor reader cannot submit weights"
            )

        positive_uids = {
            int(uid) for uid, weight in weights.items() if float(weight) > 0.0
        }
        if hotkeys is not None:
            missing_bindings = sorted(positive_uids.difference(hotkeys))
            if missing_bindings:
                # A partial binding map quietly falls back to uid liveness and can
                # pay a recycled uid's new owner. Once a caller opts into binding
                # safety, it must bind the complete positive vector (including the
                # burn target); reject locally before any RPC write.
                return SetWeightsResult(
                    success=False,
                    block=self._block,
                    message=(
                        "incomplete uid/hotkey binding map for positive targets: "
                        f"missing uids {missing_bindings}"
                    ),
                )

        if not positive_uids:
            return SetWeightsResult(
                success=False,
                block=self._block,
                message="no positive weights to submit after quantization/reconciliation",
            )

        async with self._write_lock:
            # Fetch immediately before reconciliation/write. The cached metagraph
            # can be up to metagraph_ttl_seconds old; using it here re-opens the uid
            # recycle hole that the binding check exists to close. A failed fresh
            # read is a clean no-write result, never permission to use stale state.
            try:
                live_view = await asyncio.to_thread(self._fresh_metagraph_for_submit)
            except Exception as exc:  # noqa: BLE001 - unavailable identity => no write
                self._note_raise()
                return SetWeightsResult(
                    success=False,
                    block=self._block,
                    message=(
                        "fresh metagraph unavailable; refusing to submit weights: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            # Verify BEFORE quantizing. Target churn cannot be repaired locally:
            # dropping/requantizing a target would donate its share to survivors and
            # cease to be the authenticated authority vector. The next authority
            # epoch can bind the hotkey to its new uid or route an absent payee to
            # the canonical sink.
            try:
                verified = self._reconcile_targets(weights, hotkeys, view=live_view)
            except ValueError as exc:
                return SetWeightsResult(
                    success=False,
                    block=self._block,
                    message=(
                        "fresh metagraph does not preserve the authority vector's "
                        f"uid/hotkey targets; refusing any rewritten submission: {exc}"
                    ),
                )
            u16 = quantize_u16(verified)
            if not u16:
                return SetWeightsResult(
                    success=False,
                    block=self._block,
                    message=(
                        "no positive weights to submit after "
                        "quantization/reconciliation"
                    ),
                )
            uids = sorted(u16)
            vals = [u16[uid] for uid in uids]
            # This is what bittensor==10.5.0's
            # convert_weights_and_uids_for_emit will place in the extrinsic.
            emitted_u16 = max_normalize_u16(u16)

            # Pre-gate: don't fire a doomed commit inside the rate-limit window.
            # The message MUST contain "tempo" — weightsetter/service.py:_is_tempo
            # string-matches it as a normal reschedule rather than a failure.
            blocks_since = self.blocks_since_last_update()
            if (
                self._weights_rate_limit > 0
                and blocks_since is not None
                and blocks_since <= self._weights_rate_limit
            ):
                return SetWeightsResult(
                    success=False,
                    block=self._block,
                    message=(
                        f"tempo gate: rate-limit window open"
                        f" ({blocks_since}/{self._weights_rate_limit} blocks since last update)"
                    ),
                )

            accepted_block: int | None = None
            try:
                success, message, commit_reveal = await asyncio.to_thread(
                    self._submit, uids, vals, version_key
                )
                if not isinstance(commit_reveal, bool):
                    raise TypeError(
                        "set_weights transport did not return a boolean "
                        "commit-reveal mode"
                    )
                if success and not commit_reveal:
                    # A live production false-success incident established the boundary:
                    # even a correctly-parsed SDK verdict is still the submitter's
                    # claim.  Non-CR has no timelocked pending state to defer through,
                    # so prove the exact emitted bytes in runtime storage before this
                    # method can return success.  The production reader pins Uids,
                    # Weights and LastUpdate to one GRANDPA-finalized hash.
                    accepted_block = await self._wait_for_exact_non_cr_weight_proof(
                        expected=emitted_u16,
                        after_block=int(live_view.block),
                    )
            except (asyncio.CancelledError, TimeoutError):
                # The caller's with_timeout fired (or the loop cancelled us). The
                # worker thread is abandoned WITH ITS BLOCK SUBSCRIPTION STILL ON
                # THE SOCKET — condemn so the next call reconnects rather than
                # reuse it (the leak that OOMed a prior production pod).
                self._condemned = True
                self._log.error(
                    "set_weights was cancelled/timed out; the socket is CONDEMNED"
                    " and will be reconnected before the next call"
                )
                raise
            except Exception as exc:  # noqa: BLE001 - a RAISE is transport trouble
                self._note_raise()
                # The extrinsic may already have been gossiped before the socket
                # died.  Returning success=False would classify that UNKNOWN fate as
                # an explicit chain rejection and let the intent be abandoned.  Raise
                # OSError so WeightSetter enters its ambiguity/readback branch, and
                # force that readback onto a fresh socket.
                self._condemned = True
                raise OSError(
                    f"set_weights transport failure: {type(exc).__name__}: {exc}"
                ) from exc

        # A non-CR success carries the exact block from its finalized LastUpdate
        # proof. Other outcomes retain the bounded diagnostic head read.
        block = (
            accepted_block
            if accepted_block is not None
            else self._read_block_best_effort()
        )
        if success:
            self._note_clean_submit()  # ONLY a clean submit resets the counter
            if commit_reveal:
                # Bittensor 10.5's set_weights() has only finalized the encrypted
                # CRv4 commitment at this point. ``Weights`` still contains the
                # previous vector until drand-driven reveal execution, so treating
                # this as success would publish bytes the chain does not yet hold.
                return SetWeightsResult(
                    success=False,
                    block=block,
                    message=(
                        "commit-reveal commitment finalized; awaiting automatic "
                        "reveal and exact vector readback"
                    ),
                    submitted=dict(emitted_u16),
                    pending_reveal=True,
                )
            # Report the exact-target SDK-EMITTED max-grid vector that went to
            # chain. Target churn was already refused before the transport call, so
            # the uid set is unchanged; callers still publish the emitted max-grid
            # bytes rather than the pre-SDK sum-grid representation.
            return SetWeightsResult(
                success=True,
                block=block,
                message=message,
                submitted=dict(emitted_u16),
            )
        # (False, message): the chain answered over a HEALTHY socket — neither
        # increment nor reset the reconnect counter. Map any rate-limit rejection
        # text into a "tempo"-containing message so the weight-setter reschedules.
        return SetWeightsResult(
            success=False, block=block, message=_normalize_reject_message(message)
        )

    def _submit(
        self, uids: list[int], vals: list[int], version_key: int
    ) -> tuple[bool, str, bool]:
        # Hold the socket mutex across the WHOLE extrinsic + inclusion/finalization
        # wait. If the caller's coroutine is cancelled mid-wait,
        # asyncio.to_thread does NOT stop THIS thread — it keeps the mutex until the
        # SDK wait truly returns. That mutex belongs to this generation: retirement
        # cannot close the socket under the worker, while a later submission can use
        # the atomically installed replacement generation without blocking here.
        with self._transport_call() as transport:
            return transport.set_weights(
                netuid=self._config.netuid,
                uids=uids,
                weights=vals,
                # 0 -> omit -> SDK default version_as_int; the transport treats 0 as
                # "SDK default". version_key from config OR the caller's arg — prefer
                # the caller's explicit value when it is > 0.
                version_key=version_key or self._config.version_key,
            )

    def _fresh_metagraph_for_submit(self) -> MetagraphView:
        """Read the live uid/hotkey bindings used by exactly one weight write."""
        with self._transport_call() as transport:
            return transport.metagraph(self._config.netuid)

    def _reconcile_targets(
        self,
        weights: dict[int, float],
        hotkeys: dict[int, str] | None = None,
        *,
        view: MetagraphView,
    ) -> dict[int, float]:
        """Require every positive target's (uid, hotkey) binding to still hold.

        A uid can be recycled to a DIFFERENT hotkey between scoring and submission
        (deregistration + re-registration). Checking only uid liveness would pay
        the new occupant the old miner's weight: uid 17 is still "live"
        after A -> B recycling, so A's weight lands on B. So when the caller supplies
        the intended per-uid `hotkeys`, a target survives only if the CURRENT
        metagraph binds that uid to the SAME hotkey. A uid absent from the snapshot,
        or now bound to a different hotkey, rejects the complete attempt before
        quantization. Re-normalizing survivors would silently donate the missing
        fixed share and violate authority-vector convergence.

        Without intended hotkeys, fall back to uid-liveness only. When a map is
        supplied, :meth:`set_weights` has already proven it covers every positive
        target, so every positive target is checked on the complete binding.
        """
        current = {uid: hotkey for uid, hotkey in enumerate(view.hotkeys)}
        kept: dict[int, float] = {}
        orphans: list[int] = []
        recycled: list[int] = []
        for uid, w in weights.items():
            if float(w) <= 0.0:
                continue
            if uid not in current:
                orphans.append(uid)
            elif hotkeys is not None and hotkeys[uid] != current[uid]:
                # Every positive target is present in ``hotkeys`` by the local
                # completeness gate above. A mismatch is therefore unambiguously a
                # recycled uid rather than an absent/partial caller claim.
                recycled.append(uid)
            else:
                kept[uid] = w
        if orphans or recycled:
            self._log.error(
                "refusing to mutate the authority vector after target churn",
                extra={
                    "orphan_uids": sorted(orphans),
                    "recycled_uids": sorted(recycled),
                },
            )
            raise ValueError(
                f"orphan_uids={sorted(orphans)}, recycled_uids={sorted(recycled)}"
            )
        return kept

    def _read_block_best_effort(self) -> int:
        """A bounded head read for a result's `block`; cached head on failure."""
        try:
            with self._transport_call() as transport:
                self._block = transport.current_block()
        except Exception:  # noqa: BLE001 - best effort; keep the cached head
            pass
        return self._block

    async def anchor_commitment(self, payload: bytes) -> str:
        """Anchor <=128 payload bytes on the Commitments pallet; returns a tx id.

        The current pallet does not use ``SubtensorModule.TxRateLimit``. It charges
        ``max(100, len(payload))`` against the signer's mutable ``MaxSpace`` budget,
        tracked by ``UsedSpaceOf(netuid, signer)`` and reset on the subnet epoch.
        Callers reserve/check that exact budget through ``commitment_capacity``.

        Commitment waits use a dedicated socket and lock; the process/cross-process
        writer lane stays held until the non-cancellable SDK worker exits. A failed
        write remains retryable, but it is NOT harmless: an epoch anchor that misses
        ``close_block + K`` can never satisfy the sampling-beacon audit and
        permanently HOLDs that epoch. Receipt/hash behavior remains a mandatory
        live-testnet proof because there is no Commitments-pallet production
        precedent to lean on.
        """
        if self._config.read_only:
            raise ReadOnlyChainError(
                "wallet-free bittensor reader cannot anchor commitments"
            )
        if len(payload) > 128:
            raise ValueError("chain payload must be <= 128 bytes")
        async with anchor_writer_lock(
            self._config.anchor_writer_lock_path,
            timeout_seconds=self._config.anchor_writer_lock_timeout_seconds,
        ):
            async with self._write_lock:
                worker = asyncio.create_task(asyncio.to_thread(self._anchor, payload))
                try:
                    # The synchronous SDK cannot cancel an extrinsic after dispatch.
                    # Shield its worker and, if our caller disappears, keep holding
                    # the cross-process lane until the finalization wait ends. Releasing
                    # it while the abandoned thread is still writing would recreate the
                    # exact same-block overwrite race this lane prevents.
                    return await asyncio.shield(worker)
                except asyncio.CancelledError:
                    self._anchor_condemned = True
                    try:
                        await worker
                    except Exception:  # noqa: BLE001 - preserve caller cancellation
                        pass
                    raise
                except TimeoutError:
                    self._anchor_condemned = True
                    raise
                except Exception as exc:  # noqa: BLE001
                    self._note_anchor_raise()
                    raise OSError(
                        "anchor_commitment transport failure: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

    def _anchor(self, payload: bytes) -> str:
        # v10.5.0 set_commitment takes a *str* and does data.encode() itself; handing
        # it bytes raises before the extrinsic is ever submitted, so no anchor is
        # published and every fail-closed shared validator HOLDS forever (review
        # critical). The anchor payload is ascii
        # (`vidaio.epoch.anchor.v1:<netuid>:<epoch_id>:<log_digest>`), so decode it to
        # a str at THIS boundary and always give the transport a str. A payload that
        # is not ascii is caught here (UnicodeDecodeError -> the OSError transport-
        # failure path) rather than mid-extrinsic. The read side (`read_anchor`)
        # decodes the same ascii bytes back, so str in -> str out -> the same digest.
        text = payload.decode("ascii")
        # A commitment wait has its own socket/mutex, so a cancelled non-cancellable
        # SDK worker cannot strand the read socket. The process-level _write_lock and
        # cross-process anchor writer lock still prevent concurrent extrinsics and
        # mutable-slot overwrites.
        with self._anchor_transport_call() as transport:
            txid = transport.set_commitment(netuid=self._config.netuid, payload=text)
        self._note_anchor_clean_submit()
        return str(txid)

    # -- anchor read (EpochAnchorReadable) -----------------------------------------

    def read_anchor(self, *, netuid: int, epoch_id: int, domain: str) -> str | None:
        """Read the epoch-log anchor back off chain — the third verification leg (#3).

        Reads the Commitments pallet for the AUTHORITY's account (`anchor_hotkey`,
        falling back to our own hotkey for a self-anchoring node) and parses the
        domain-tagged payload. The pallet has ONE slot per ``(netuid, account)``:
        a non-matching current payload therefore cannot prove a historical epoch was
        never anchored (it may simply have been overwritten). That case RAISES and
        HOLDs explicitly; only an actually-empty slot returns None. Historical/backfill
        verification requires an archive/indexed design and remains a testnet gate.
        """
        self._require_bound_netuid(netuid)
        account = self._config.anchor_hotkey or self._config.validator_hotkey
        try:
            payload = self._read_commitment(netuid, account)
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed read is UNKNOWN, never None
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read the on-chain anchor commitment for {account!r} on"
                f" subnet {netuid}: {type(exc).__name__}: {exc}"
            ) from exc
        if payload is None:
            return None
        digest = parse_anchor_digest(
            [payload], netuid=netuid, epoch_id=epoch_id, domain=domain
        )
        if digest is None:
            raise ChainStateUnavailable(
                f"the Commitments pallet has a single current slot for {account!r} on "
                f"subnet {netuid}; its current payload is not {domain!r} epoch "
                f"{epoch_id}, so absence of that HISTORICAL anchor cannot be proven. "
                "HOLDING until an archive/indexed historical-anchor reader is wired"
            )
        return digest

    def read_anchor_at(
        self,
        *,
        netuid: int,
        epoch_id: int,
        domain: str,
        block_number: int,
    ) -> str | None:
        """Read an anchor from archive state at its claimed inclusion block.

        ``CommitmentOf`` is a single mutable slot. Querying it at head cannot
        verify an older pointer after the next epoch overwrites that slot, so this
        method pins both the payload and its metadata to ``block_number``. A digest
        is returned only when the record itself says it was included at that exact
        block. A later historical state that merely still contains an older record
        is a definitive non-match (``None``), not proof for a false pointer.
        """
        self._require_bound_netuid(netuid)
        if block_number < 0:
            raise ValueError(f"block_number must be non-negative, got {block_number}")
        account = self._config.anchor_hotkey or self._config.validator_hotkey
        try:
            with self._transport_call() as transport:
                payload = transport.get_commitment(
                    netuid=netuid, ss58=account, block_number=block_number
                )
                if payload is None:
                    return None
                digest = parse_anchor_digest(
                    [payload], netuid=netuid, epoch_id=epoch_id, domain=domain
                )
                if digest is None:
                    # Unlike a head read, this exact archive state positively proves
                    # the requested epoch/domain payload was not in the slot here.
                    return None
                inclusion_block = transport.get_commitment_block(
                    netuid=netuid, ss58=account, block_number=block_number
                )
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - pruned/unreadable archive is UNKNOWN
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read the on-chain anchor for {account!r} on subnet {netuid} "
                f"at block {block_number}: {type(exc).__name__}: {exc}"
            ) from exc

        if inclusion_block is None:
            raise ChainStateUnavailable(
                f"the archive commitment for {account!r} on subnet {netuid} at "
                f"block {block_number} contains the requested anchor but exposes no "
                "inclusion block"
            )
        if int(inclusion_block) != block_number:
            # The slot at a later block can still carry an older commitment. It does
            # not authenticate that later block as the pointer's inclusion block.
            return None
        return digest

    def _read_commitment(
        self, netuid: int, ss58: str, *, block_number: int | None = None
    ) -> bytes | None:
        with self._transport_call() as transport:
            return transport.get_commitment(
                netuid=netuid, ss58=ss58, block_number=block_number
            )

    def read_commitment_record(
        self, *, netuid: int, block_number: int | None = None
    ) -> ChainCommitmentRecord | None:
        """Read the authority account's raw commitment record at head/archive.

        Payload and stored inclusion height are queried while one adapter
        transport generation is held.  A historical call intentionally returns
        the record's *original* inclusion height; receipt verification requires
        that value to equal the requested archive block, preventing a later state
        that merely still carries an older value from masquerading as inclusion.
        """

        self._require_bound_netuid(netuid)
        if block_number is not None and (
            isinstance(block_number, bool) or block_number < 0
        ):
            raise ValueError("block_number must be a non-negative integer")
        account = self._config.anchor_hotkey or self._config.validator_hotkey
        try:
            with self._transport_call() as transport:
                payload = transport.get_commitment(
                    netuid=netuid, ss58=account, block_number=block_number
                )
                if payload is None:
                    return None
                included_at = transport.get_commitment_block(
                    netuid=netuid, ss58=account, block_number=block_number
                )
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # unreadable/pruned state is UNKNOWN, never empty
            self._note_raise()
            where = "head" if block_number is None else f"block {block_number}"
            raise ChainStateUnavailable(
                f"cannot read raw commitment record for {account!r} on subnet "
                f"{netuid} at {where}: {type(exc).__name__}: {exc}"
            ) from exc
        if included_at is None:
            raise ChainStateUnavailable(
                f"commitment record for {account!r} on subnet {netuid} has payload "
                "bytes but no readable inclusion block"
            )
        if not isinstance(payload, bytes):
            raise ChainStateUnavailable(
                f"commitment record for {account!r} decoded as "
                f"{type(payload).__name__}, expected bytes"
            )
        return ChainCommitmentRecord(payload=payload, block=int(included_at))

    # -- anchor inclusion-block read (EpochAnchorBlockReadable) --------------------

    def read_anchor_block(
        self, *, netuid: int, epoch_id: int, domain: str
    ) -> int | None:
        """The INCLUSION BLOCK of `(netuid, epoch_id)`'s anchor.

        Reads the anchor account's commitment (confirming it is THIS epoch's anchor via
        the domain-tagged prefix) and returns the block that commitment was set at — the
        block the anchor extrinsic landed in. The auditor uses it to confirm the item set
        was committed BEFORE the round-6 beacon block (`close_block + K`) could be known
        (`anchor_block <= close_block + K`); an anchor committed AFTER that is a grind
        risk.

        Returns None only when the account's single commitment slot is empty. A
        non-matching current payload is UNKNOWN, not positive absence: a newer payload
        may have overwritten this epoch. It therefore raises/HOLDs until historical
        reads are backed by an archive/index rather than silently rejecting an honest
        old epoch.
        """
        self._require_bound_netuid(netuid)
        account = self._config.anchor_hotkey or self._config.validator_hotkey
        try:
            # Keep payload + inclusion-block reads together so this process cannot
            # overwrite the single slot between the two calls and pair one anchor's
            # bytes with another anchor's block.
            with self._transport_call() as transport:
                payload = transport.get_commitment(
                    netuid=netuid, ss58=account, block_number=None
                )
                if payload is None:
                    return None
                if (
                    parse_anchor_digest(
                        [payload], netuid=netuid, epoch_id=epoch_id, domain=domain
                    )
                    is None
                ):
                    raise ChainStateUnavailable(
                        f"the Commitments pallet's single current slot for {account!r} "
                        f"on subnet {netuid} no longer contains {domain!r} epoch "
                        f"{epoch_id}; its historical inclusion block is unknowable from "
                        "head state, so the auditor HOLDs"
                    )
                block = transport.get_commitment_block(
                    netuid=netuid, ss58=account, block_number=None
                )
            if block is None:
                # The commitment IS this epoch's anchor but we cannot resolve its
                # inclusion block — UNKNOWN, so HOLD (never a substituted None).
                raise ChainStateUnavailable(
                    f"the anchor commitment for {account!r} on subnet {netuid} exists"
                    " but its inclusion block could not be read"
                )
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed read is UNKNOWN, never None
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read the anchor inclusion block for {account!r} on"
                f" subnet {netuid}: {type(exc).__name__}: {exc}"
            ) from exc
        return int(block)

    # -- block-hash read (BlockHashReadable) ---------------------------------------

    def block_hash(self, block_number: int) -> str | None:
        """The substrate block HASH at `block_number`, else None.

        The round-6 beacon is `block_hash(close_block + K)` — the hash of a FUTURE
        FINALIZED block. The authority cannot precompute the hash of a block that has not
        been produced, and the beacon block is fixed by the epoch's `close_block`, so
        re-anchoring the same payload later cannot reroll it. Returns the real substrate
        hash (0x stripped, lowercased) for a produced block, None when the chain has not
        produced `block_number` yet (the beacon is not finalized, so the auditor HOLDS),
        and RAISES `ChainStateUnavailable` on a read/transport failure. UNPROVEN —
        validate on testnet (wave 8); holds the socket mutex like the sibling reads.
        """
        try:
            raw = self._read_block_hash(block_number)
        except Exception as exc:  # noqa: BLE001 - a failed read is UNKNOWN, never None
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read block_hash({block_number}) on subnet"
                f" {self._config.netuid}: {type(exc).__name__}: {exc}"
            ) from exc
        if not raw:
            # A None/empty hash means the block is not produced yet — HOLD, retry later.
            return None
        normalized = str(raw)
        if normalized.startswith("0x"):
            normalized = normalized[2:]
        return normalized.lower()

    def _read_commitment_block(
        self, netuid: int, ss58: str, *, block_number: int | None = None
    ) -> int | None:
        with self._transport_call() as transport:
            return transport.get_commitment_block(
                netuid=netuid, ss58=ss58, block_number=block_number
            )

    def _read_block_hash(self, block_number: int) -> str | None:
        with self._transport_call() as transport:
            return transport.get_block_hash(block_number)

    def block_time(self, block_number: int) -> datetime | None:
        """UTC timestamp recorded by ``Timestamp.Now`` at ``block_number``.

        The epoch log's ``created_at`` is an economic input, so an unavailable or
        pruned timestamp is UNKNOWN (raise/HOLD), never substituted with wall time.
        A future block is the one positive ``None`` answer.
        """
        if block_number < 0:
            raise ValueError(f"block_number must be non-negative, got {block_number}")
        try:
            with self._transport_call() as transport:
                value = transport.block_time(block_number)
        except Exception as exc:  # noqa: BLE001 - unreadable block time is UNKNOWN
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read block_time({block_number}) on subnet "
                f"{self._config.netuid}: {type(exc).__name__}: {exc}"
            ) from exc
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ChainStateUnavailable(
                f"block_time({block_number}) returned {type(value).__name__}, expected datetime"
            )
        if value.tzinfo is None or value.utcoffset() is None:
            raise ChainStateUnavailable(
                f"block_time({block_number}) returned a timezone-naive datetime"
            )
        return value.astimezone(timezone.utc)

    def tempo(self, netuid: int | None = None) -> int:
        """The subnet's live epoch length in blocks, read from chain (#14).

        The epoch driver derives `blocks_per_epoch` from THIS rather than a hardcoded
        constant. In pinned bittensor 10.5.0 the runtime schedule is
        ``last_epoch_block + tempo``: tempo is already the period and must not be
        incremented. Raises on an unreadable chain.
        """
        nid = self._config.netuid if netuid is None else netuid
        try:
            with self._transport_call() as transport:
                raw_tempo = int(transport.tempo(nid))
                if raw_tempo <= 0:
                    raise ValueError(f"non-positive on-chain tempo {raw_tempo}")
                return raw_tempo
        except Exception as exc:  # noqa: BLE001
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot read tempo for subnet {nid}: {type(exc).__name__}: {exc}"
            ) from exc

    def get_burn_uid(self) -> int:
        """Current uid of the subnet-owner hotkey, resolved entirely from chain.

        ``SubnetOwnerHotkey`` is mutable governance state and uids can be recycled,
        so neither uid 0 nor a deploy-time uid is safe.  Read the owner hotkey and
        resolve that hotkey on this subnet in the same socket-serialized operation.
        Any missing/unreadable leg is UNKNOWN and fails closed.
        """
        netuid = self._config.netuid
        try:
            with self._transport_call() as transport:
                owner_hotkey = transport.subnet_owner_hotkey(netuid)
                if not owner_hotkey:
                    raise ChainStateUnavailable(
                        f"subnet {netuid} has no readable subnet owner hotkey "
                        "(SubnetOwnerHotkey)"
                    )
                uid = transport.uid_for_hotkey(owner_hotkey, netuid)
        except ChainStateUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - an unreadable identity is UNKNOWN
            self._note_raise()
            raise ChainStateUnavailable(
                f"cannot resolve subnet {netuid}'s owner uid: {type(exc).__name__}: {exc}"
            ) from exc
        if uid is None:
            raise ChainStateUnavailable(
                f"subnet-owner hotkey {owner_hotkey!r} is not registered on subnet {netuid}"
            )
        resolved = int(uid)
        if resolved < 0:
            raise ChainStateUnavailable(
                f"subnet-owner hotkey {owner_hotkey!r} resolved to invalid uid {resolved}"
            )
        return resolved

    def sign(self, payload: bytes) -> str:
        """Sign canonical bytes with the loaded validator hotkey (hex output).

        This satisfies the auditor ``ReportSigner`` seam without exposing seed
        material. Verification already uses ``bittensor.Keypair(ss58_address=...)``
        in ``vidaio.audit_api.verify.HotkeySignatureVerifier``.
        """
        if self._config.read_only:
            raise ReadOnlyChainError("wallet-free bittensor reader cannot sign")
        if not isinstance(payload, bytes):
            raise TypeError(f"payload must be bytes, got {type(payload).__name__}")
        signer = getattr(self._transport, "sign_hotkey", None)
        if not callable(signer):
            raise NotConfiguredError(
                "the bittensor transport does not expose hotkey signing; refusing to "
                "emit an unsigned auditor report"
            )
        signature = signer(payload)
        if not isinstance(signature, (bytes, bytearray)) or len(signature) != 64:
            size = len(signature) if isinstance(signature, (bytes, bytearray)) else None
            raise RuntimeError(
                "the loaded hotkey must return an exact 64-byte sr25519/ed25519 "
                f"signature (got {size!r} bytes)"
            )
        return bytes(signature).hex()

    def close(self) -> None:
        """Retire current transports without waiting on a wedged SDK generation."""
        with self._transport_state_lock:
            generations = [("close-main", self._main_generation)]
            if (
                self._anchor_generation is not None
                and self._anchor_generation is not self._main_generation
            ):
                generations.append(("close-anchor", self._anchor_generation))
        for lane, generation in generations:
            self._retire_generation(lane, generation)


class BittensorReadOnlyChainAdapter(BittensorChainAdapter):
    """Wallet-free Bittensor reader for public services.

    It retains the parent adapter's bounded RPCs, metagraph cache, failure
    accounting, and reconnect discipline. Construction requires an explicitly
    read-only config; the parent and real transport both reject every write/sign
    path as defense in depth.
    """

    def __init__(
        self,
        config: BittensorAdapterConfig,
        *,
        transport: _SubtensorTransport | None = None,
        connect_transport: Callable[[], _SubtensorTransport] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not config.read_only:
            raise ValueError("BittensorReadOnlyChainAdapter requires read_only=True")
        super().__init__(
            config,
            transport=transport,
            connect_transport=connect_transport,
            clock=clock,
        )


class BittensorHotkeySigner:
    """Narrow ``ReportSigner`` view over a wallet-backed chain adapter."""

    def __init__(self, chain: BittensorChainAdapter) -> None:
        self._chain = chain

    def sign(self, payload: bytes) -> str:
        return self._chain.sign(payload)


# --------------------------------------------------------------------------------------
# Reject-message normalization (§d step 2 + risk 5).
# --------------------------------------------------------------------------------------

#: Chain rejection texts that mean 'you set weights too recently' — mapped to a
#: "tempo"-containing message so weightsetter/service.py:_is_tempo reschedules
#: rather than counting a chain failure.
_RATE_LIMIT_MARKERS = ("settingweightstoofast", "too fast", "rate limit", "ratelimit")


def _normalize_reject_message(message: str) -> str:
    low = message.lower()
    if "tempo" in low:
        return message
    if any(marker in low for marker in _RATE_LIMIT_MARKERS):
        return f"tempo gate: chain rate-limit rejection ({message})"
    return message


def _require_exact_finalized_weight_proof(
    report: SubmittedWeights | None,
    *,
    expected: dict[int, int],
    after_block: int,
) -> int:
    """Validate a non-CR acceptance against exact finalized runtime storage.

    A successful SDK response is a claim, not the durable state itself.  This
    deliberately does *not* use scale-tolerant reconciliation: ``expected`` is
    already the byte-for-byte max-grid emitted by the pinned SDK, and the real
    reader returns raw ``Weights`` from one finalized hash.  Any absent, stale,
    fractional, out-of-range, or differing value leaves the write ambiguous.

    Returns the exact ``LastUpdate`` block carried by the proven snapshot.
    """
    if report is None:
        raise RuntimeError(
            "SDK claimed non-CR success but finalized storage has no weight record"
        )
    block = report.block
    if isinstance(block, bool) or not isinstance(block, int):
        raise TypeError("finalized weight proof has no integer LastUpdate block")
    # ``after_block`` was observed before the extrinsic was submitted. A write
    # cannot be included in that already-observed block, so equality proves only
    # that an identical vector pre-existed; it cannot prove this attempt landed.
    if block <= after_block:
        raise RuntimeError(
            "SDK claimed non-CR success but finalized LastUpdate did not advance "
            f"past the submission snapshot ({block} <= {after_block})"
        )

    observed: dict[int, int] = {}
    for raw_uid, raw_weight in report.weights.items():
        if isinstance(raw_uid, bool):
            raise TypeError("finalized Weights contains a boolean uid")
        uid = int(raw_uid)
        if uid != raw_uid:
            raise TypeError(f"finalized Weights uid {raw_uid!r} is not an integer")
        if isinstance(raw_weight, bool):
            raise TypeError(f"finalized Weights[{uid}] is boolean")
        weight = float(raw_weight)
        if (
            not math.isfinite(weight)
            or not weight.is_integer()
            or not 0 < weight <= U16_MAX
        ):
            raise ValueError(
                f"finalized Weights[{uid}] is not a positive u16: {raw_weight!r}"
            )
        if uid in observed:
            raise ValueError(f"finalized Weights repeats uid {uid}")
        observed[uid] = int(weight)

    canonical_expected = {int(uid): int(weight) for uid, weight in expected.items()}
    if observed != canonical_expected:
        raise RuntimeError(
            "SDK claimed non-CR success but finalized Weights differs from the "
            f"emitted max-grid (observed={observed}, expected={canonical_expected})"
        )
    return block


# --------------------------------------------------------------------------------------
# The REAL transport — the ONLY code that imports bittensor. Not unit-tested.
# --------------------------------------------------------------------------------------


def _run_with_timeout(fn: Callable[[], Any], seconds: float, name: str) -> Any:
    """Bound a short, blocking RPC with a daemon worker thread.

    A wedged socket otherwise hangs the loop — observed in production after a
    successful submit. A fired timeout ABANDONS (does not cancel) the
    worker; the caller condemns the socket. This is used for SHORT RPCs ONLY —
    NEVER to bound the set_weights inclusion/finalization wait (the leak that
    OOMed the pod).
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller thread
            box["error"] = exc

    thread = threading.Thread(target=_run, name=f"bt-rpc-{name}", daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"subtensor RPC {name!r} did not return in {seconds}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _connect_real_transport(config: BittensorAdapterConfig) -> _SubtensorTransport:
    """Build the real, bittensor-backed transport. Lazy import; fail fast.

    Raises NotConfiguredError (pointing at the '.[chain]' extra) if the optional
    deps are not installed. Writable adapters also fail clearly when the
    wallet/hotkey is absent; read-only adapters never inspect wallet configuration
    or seed environment variables.
    """
    try:
        import bittensor as bt  # noqa: F401 - imported for its side of the seam
    except ImportError as exc:
        raise NotConfiguredError(_INSTALL_HINT) from exc

    return _RealSubtensorTransport(config)


class _RealSubtensorTransport:
    """The real seam: one long-lived sync `bt.Subtensor` and raw storage.

    This is the ONLY class that imports/uses bittensor. It is intentionally thin —
    all decision logic lives in BittensorChainAdapter — and is validated on
    testnet rather than by unit tests. Every short RPC is bounded by a daemon
    thread; set_weights is NOT (see _run_with_timeout). Writable construction
    loads one hotkey wallet. Read-only construction deliberately leaves both
    `_hotkey` and `_wallet` unset and rejects all signing/extrinsic methods.
    """

    def __init__(self, config: BittensorAdapterConfig) -> None:
        import bittensor as bt

        self._bt = bt
        self._config = config
        self._log = get_logger("chain.bittensor.transport")
        # Acquired by the ACTUAL SDK-calling thread (including daemon workers used
        # for bounded reads), not merely by its caller. If a short RPC times out,
        # that abandoned worker keeps this lock until it really exits, preventing a
        # second call or close from running over the same websocket.
        self._sdk_lock = threading.RLock()
        if config.read_only:
            self._hotkey = None
            self._wallet = None
        else:
            self._hotkey = self._load_hotkey(config)
            self._wallet = _KeypairWallet(self._hotkey)

        network = config.endpoint or config.network
        # Retrying, timeout-bounded connect: a public endpoint's HTTP 429
        # otherwise produced 139 restarts in 10h.
        self._subtensor = self._connect_with_retry(network)
        self._substrate = self._subtensor.substrate
        self._validate_sdk_contract()

    # -- wallet load (fail fast) ---------------------------------------------------

    def _load_hotkey(self, config: BittensorAdapterConfig) -> Any:
        import os

        bt = self._bt
        # On-disk btcli wallet takes precedence when configured.
        if config.wallet_name and config.wallet_hotkey:
            wallet = bt.Wallet(
                name=config.wallet_name,
                hotkey=config.wallet_hotkey,
                path=config.wallet_path or None,
            )
            return wallet.hotkey
        seed = os.environ.get(config.hotkey_seed_env, "").strip()
        if not seed:
            raise RuntimeError(
                "no wallet configured: set an on-disk wallet"
                " (chain.wallet_name/wallet_hotkey) or the hotkey seed env"
                f" {config.hotkey_seed_env!r} — refusing to start without a hotkey"
            )
        if seed.startswith("0x"):
            return bt.Keypair.create_from_seed(seed)
        return bt.Keypair.create_from_mnemonic(seed)

    def _connect_with_retry(self, network: str) -> Any:
        bt = self._bt
        last: BaseException | None = None
        delay = 1.0
        for attempt in range(1, 9):  # 8 attempts, exp 1->60s backoff
            try:
                return _run_with_timeout(
                    lambda: bt.Subtensor(
                        network=network,
                        fallback_endpoints=(
                            list(self._config.fallback_endpoints) or None
                        ),
                        # Bittensor 10.5 keeps a distinct pool for historical
                        # queries. Supplying fallbacks only to the ordinary pool
                        # silently left archive reads pinned to the primary RPC.
                        archive_endpoints=(
                            list(self._config.fallback_endpoints) or None
                        ),
                    ),
                    self._config.connect_timeout_seconds,
                    "connect",
                )
            except BaseException as exc:  # noqa: BLE001
                last = exc
                self._log.warning(
                    "subtensor connect failed; backing off",
                    extra={"attempt": attempt, "error": str(exc)},
                )
                time.sleep(min(60.0, delay))
                delay *= 2
        raise RuntimeError(f"could not connect subtensor after 8 attempts: {last}")

    def _validate_sdk_contract(self) -> None:
        """Fail at startup if the installed SDK cannot honor our safety kwargs.

        In particular, silently omitting ``mev_protection=False`` would let a
        process-level environment variable reroute writes through MEV Shield, and
        missing historical/block parameters would turn pinned reads into head reads.
        """

        required: list[tuple[str, Any, tuple[str, ...]]] = []
        if not self._config.read_only:
            required.extend(
                (
                    (
                        "Subtensor.set_weights",
                        getattr(self._subtensor, "set_weights", None),
                        (
                            "netuid",
                            "uids",
                            "weights",
                            "commit_reveal_version",
                            "max_attempts",
                            "version_key",
                            "mev_protection",
                            "raise_error",
                            "wait_for_inclusion",
                            "wait_for_finalization",
                            "wait_for_revealed_execution",
                        ),
                    ),
                    (
                        "Subtensor.commit_reveal_enabled",
                        getattr(self._subtensor, "commit_reveal_enabled", None),
                        ("netuid", "block"),
                    ),
                    (
                        "Subtensor.set_commitment",
                        getattr(self._subtensor, "set_commitment", None),
                        (
                            "netuid",
                            "data",
                            "mev_protection",
                            "raise_error",
                            "wait_for_inclusion",
                            "wait_for_finalization",
                            "wait_for_revealed_execution",
                        ),
                    ),
                )
            )
        required.extend(
            (
                (
                    "Subtensor.metagraph",
                    getattr(self._subtensor, "metagraph", None),
                    ("netuid", "block"),
                ),
                (
                    "Subtensor.tx_rate_limit",
                    getattr(self._subtensor, "tx_rate_limit", None),
                    ("block",),
                ),
                (
                    "Subtensor.get_timestamp",
                    getattr(self._subtensor, "get_timestamp", None),
                    ("block",),
                ),
                (
                    "Subtensor.get_commitment_metadata",
                    getattr(self._subtensor, "get_commitment_metadata", None),
                    ("netuid", "hotkey_ss58", "block"),
                ),
                (
                    "Subtensor.get_epoch_schedule_state",
                    getattr(self._subtensor, "get_epoch_schedule_state", None),
                    ("netuid", "block"),
                ),
                (
                    "Subtensor.get_subnet_epoch_index",
                    getattr(self._subtensor, "get_subnet_epoch_index", None),
                    ("netuid", "block"),
                ),
            )
        )
        problems: list[str] = []
        for label, method, params in required:
            if not callable(method):
                problems.append(f"{label} is missing")
                continue
            try:
                names = set(inspect.signature(method).parameters)
            except (TypeError, ValueError) as exc:
                problems.append(f"cannot inspect {label}: {exc}")
                continue
            missing = [name for name in params if name not in names]
            if missing:
                problems.append(f"{label} lacks {', '.join(missing)}")
        for label, method, params in (
            (
                "SubstrateInterface.get_chain_finalised_head",
                getattr(self._substrate, "get_chain_finalised_head", None),
                (),
            ),
            (
                "SubstrateInterface.get_chain_head",
                getattr(self._substrate, "get_chain_head", None),
                (),
            ),
            (
                "SubstrateInterface.get_block_number",
                getattr(self._substrate, "get_block_number", None),
                ("block_hash",),
            ),
            (
                "SubstrateInterface.get_block_hash",
                getattr(self._substrate, "get_block_hash", None),
                # async-substrate-interface 2.2.1 names this positional height
                # ``block_id`` (not substrate-interface's historical
                # ``block_number`` spelling).  The release dependency probe
                # enforces the same pinned contract before deployment.
                ("block_id",),
            ),
            (
                "SubstrateInterface.query",
                getattr(self._substrate, "query", None),
                ("module", "storage_function", "params", "block_hash"),
            ),
            (
                "SubstrateInterface.query_map",
                getattr(self._substrate, "query_map", None),
                (
                    "module",
                    "storage_function",
                    "params",
                    "block_hash",
                    "page_size",
                    "ignore_decoding_errors",
                ),
            ),
        ):
            if not callable(method):
                problems.append(f"{label} is missing")
                continue
            try:
                names = set(inspect.signature(method).parameters)
            except (TypeError, ValueError) as exc:
                problems.append(f"cannot inspect {label}: {exc}")
                continue
            missing = [name for name in params if name not in names]
            if missing:
                problems.append(f"{label} lacks {', '.join(missing)}")
        if problems:
            raise RuntimeError(
                "installed bittensor SDK is incompatible with vidaio's fail-closed "
                "chain contract: " + "; ".join(problems) + ". " + _INSTALL_HINT
            )

    def _rpc(self, fn: Callable[[], Any], name: str) -> Any:
        # Unit tests construct this private transport without __init__ to exercise
        # SDK-shape logic without importing bittensor. Production always sets it in
        # __init__; the fallback keeps that test seam honest.
        if not hasattr(self, "_sdk_lock"):
            self._sdk_lock = threading.RLock()

        def _serialized() -> Any:
            with self._sdk_lock:
                return fn()

        return _run_with_timeout(_serialized, self._config.rpc_timeout_seconds, name)

    # -- transport surface ---------------------------------------------------------

    def current_block(self) -> int:
        return int(self._rpc(self._subtensor.get_current_block, "current_block"))

    def finalized_block(self) -> int:
        """Resolve GRANDPA's finalized hash to its exact block number."""

        def _read() -> int:
            block_hash = self._substrate.get_chain_finalised_head()
            if not block_hash:
                raise RuntimeError("chain_getFinalizedHead returned an empty hash")
            block_number = self._substrate.get_block_number(block_hash=block_hash)
            if block_number is None:
                raise RuntimeError(
                    f"get_block_number could not resolve finalized hash {block_hash}"
                )
            return int(block_number)

        return int(self._rpc(_read, "finalized_block"))

    def _require_archive_block(self, block_number: int) -> None:
        """Prove a historical height resolves before an SDK query can fall back to head."""
        if block_number < 1:
            raise ValueError(
                f"historical epoch reads require block >= 1, got {block_number}"
            )
        if self._substrate.get_block_hash(block_number) is None:
            raise LookupError(
                f"block {block_number} is unavailable on this endpoint; stateful epoch "
                "boundary verification requires archive state"
            )

    def epoch_index(self, netuid: int, block_number: int) -> int:
        """Read ``SubnetEpochIndex`` at an exact archive block."""

        def _read() -> int:
            self._require_archive_block(block_number)
            return int(
                self._subtensor.get_subnet_epoch_index(
                    netuid=netuid, block=block_number
                )
            )

        return int(self._rpc(_read, f"epoch_index_at_{block_number}"))

    def epoch_schedule(self, netuid: int, block_number: int) -> EpochScheduleView:
        """Normalize the SDK's block-pinned stateful epoch schedule snapshot."""

        def _read() -> Any:
            self._require_archive_block(block_number)
            return self._subtensor.get_epoch_schedule_state(
                netuid=netuid, block=block_number
            )

        state = self._rpc(_read, f"epoch_schedule_at_{block_number}")
        return EpochScheduleView(
            block=int(getattr(state, "current_block")),
            last_epoch_block=int(getattr(state, "last_epoch_block")),
            pending_epoch_at=int(getattr(state, "pending_epoch_at")),
            subnet_epoch_index=int(getattr(state, "subnet_epoch_index")),
            tempo=int(getattr(state, "tempo")),
            blocks_since_last_step=int(getattr(state, "blocks_since_last_step")),
        )

    def metagraph(self, netuid: int, block_number: int | None = None) -> MetagraphView:
        mg = self._rpc(
            lambda: self._subtensor.metagraph(netuid, lite=False, block=block_number),
            "metagraph" if block_number is None else f"metagraph_at_{block_number}",
        )

        def _ip(uid: int) -> str:
            try:
                return str(mg.axons[uid].ip)
            except Exception:  # noqa: BLE001 - non-serving neuron -> 0.0.0.0
                return "0.0.0.0"

        def _port(uid: int) -> int | None:
            try:
                return int(mg.axons[uid].port)
            except Exception:  # noqa: BLE001 - old/non-serving SDK view
                return None

        n = int(mg.n)
        return MetagraphView(
            block=int(mg.block.item()) if hasattr(mg.block, "item") else int(mg.block),
            hotkeys=[str(h) for h in mg.hotkeys],
            coldkeys=[str(c) for c in mg.coldkeys],
            axon_ips=[_ip(uid) for uid in range(n)],
            alpha_stake=[float(mg.alpha_stake[uid]) for uid in range(n)],
            emission=[float(mg.emission[uid]) for uid in range(n)],
            validator_permit=[bool(mg.validator_permit[uid]) for uid in range(n)],
            last_update=[int(mg.last_update[uid]) for uid in range(n)],
            registration_block=[int(mg.block_at_registration[uid]) for uid in range(n)],
            axon_ports=[_port(uid) for uid in range(n)],
            # Older SDK views/fakes may not carry incentive; the feed treats a
            # missing column as zeros rather than failing the whole snapshot.
            incentive=(
                [float(mg.incentive[uid]) for uid in range(n)]
                if getattr(mg, "incentive", None) is not None
                else []
            ),
        )

    def set_weights(
        self, *, netuid: int, uids: list[int], weights: list[int], version_key: int
    ) -> tuple[bool, str, bool]:
        if self._config.read_only:
            raise ReadOnlyChainError(
                "wallet-free bittensor transport cannot submit weights"
            )
        # NEVER wrapped in _run_with_timeout — the inclusion/finalization wait must
        # not be abandoned mid-flight (risk 1). raise_error stays default False so
        # chain answers come back as (False, message); mev_protection=False so a
        # stray BT_MEV_PROTECTION env var can't reroute the extrinsic (§b#7).
        kwargs: dict[str, Any] = dict(
            wallet=self._wallet,
            netuid=netuid,
            uids=uids,
            weights=weights,
            # Pin the release protocol explicitly. Letting an SDK-default drift
            # choose a different commit representation would make the pending-
            # state/readback contract below invalid without a code change.
            commit_reveal_version=4,
            # One durable intent owns retry/reconciliation. The SDK default of
            # five immediate attempts can fire repeated doomed CRv4 commits or
            # blur an ambiguous transport outcome before our ledger can inspect
            # chain state (a timelocked-commit pre-gate proven in production).
            max_attempts=1,
            mev_protection=False,
            raise_error=False,
            wait_for_inclusion=True,
            wait_for_finalization=True,
            wait_for_revealed_execution=True,
        )
        if version_key:  # 0 -> omit -> SDK default version_as_int
            kwargs["version_key"] = version_key
        with self._sdk_lock:
            # Probe under the SAME SDK serialization lane immediately before the
            # write. In CRv4 mode a successful response means only that the
            # timelocked commitment finalized; it does not mean ``Weights`` changed.
            commit_reveal = self._subtensor.commit_reveal_enabled(netuid=netuid)
            if not isinstance(commit_reveal, bool):
                raise RuntimeError(
                    "Subtensor.commit_reveal_enabled returned "
                    f"{type(commit_reveal).__name__}, not bool"
                )
            raw_response = self._subtensor.set_weights(**kwargs)
            outcome = _parse_chain_result(raw_response)
        if outcome.transport_error is not None:
            raise outcome.transport_error
        message = outcome.message
        if (
            not outcome.success
            and not message
            and hasattr(raw_response, "success")
            and getattr(raw_response, "error", None) is None
        ):
            # Pinned bittensor 10.5 initializes ExtrinsicResponse(False), then
            # skips its set_weights loop when the SDK's own
            # blocks-since-last-update precheck says the rate-limit window is
            # closed. It returns that empty object without making an extrinsic.
            # Keep this narrow to the v10 response shape and this set_weights
            # boundary; unrelated empty responses remain unclassified.
            message = (
                "tempo gate: pinned SDK rate-limit precheck made no set_weights attempt"
            )
        return outcome.success, message, commit_reveal

    def commit_reveal_enabled(self, netuid: int) -> bool:
        value = self._rpc(
            lambda: self._subtensor.commit_reveal_enabled(netuid=netuid),
            "commit_reveal_enabled",
        )
        if not isinstance(value, bool):
            raise RuntimeError(
                "Subtensor.commit_reveal_enabled returned "
                f"{type(value).__name__}, not bool"
            )
        return value

    def query_weights(self, netuid: int, uid: int) -> list[tuple[int, int]]:
        raw = self._rpc(
            lambda: self._substrate.query("SubtensorModule", "Weights", [netuid, uid]),
            "query_weights",
        )
        value = getattr(raw, "value", raw) or []
        return [(int(u), int(w)) for u, w in value]

    def query_last_update(self, netuid: int, uid: int) -> int:
        raw = self._rpc(
            lambda: self._substrate.query("SubtensorModule", "LastUpdate", [netuid]),
            "query_last_update",
        )
        value = getattr(raw, "value", raw)
        return int(value[uid])

    def submitted_weights_at_finalized_head(
        self, netuid: int, hotkey: str
    ) -> SubmittedWeights | None:
        """Read Uids, Weights and LastUpdate from one GRANDPA-finalized state.

        This is the storage proof behind every production weight confirmation.
        Resolving the uid at head and the vector at another head can pair states
        that never coexisted (especially across deregistration/recycling), so all
        three keys use the exact same finalized block hash.
        """

        def _read() -> SubmittedWeights | None:
            block_hash = self._substrate.get_chain_finalised_head()
            if not isinstance(block_hash, str) or not block_hash.startswith("0x"):
                raise RuntimeError(
                    "chain_getFinalizedHead returned an invalid hash while proving "
                    "submitted weights"
                )
            block_number = self._substrate.get_block_number(block_hash=block_hash)
            if block_number is None:
                raise RuntimeError(
                    f"cannot resolve finalized weight-proof hash {block_hash}"
                )

            uid_raw = self._substrate.query(
                module="SubtensorModule",
                storage_function="Uids",
                params=[netuid, hotkey],
                block_hash=block_hash,
            )
            uid_value = getattr(uid_raw, "value", uid_raw)
            if uid_value is None:
                raise LookupError(
                    f"hotkey {hotkey!r} is not registered on subnet {netuid} at "
                    f"finalized block {int(block_number)}"
                )
            if isinstance(uid_value, bool):
                raise TypeError("SubtensorModule.Uids returned a boolean uid")
            uid = int(uid_value)
            if uid < 0:
                raise ValueError(f"SubtensorModule.Uids returned negative uid {uid}")

            weights_raw = self._substrate.query(
                module="SubtensorModule",
                storage_function="Weights",
                params=[netuid, uid],
                block_hash=block_hash,
            )
            weights_value = getattr(weights_raw, "value", weights_raw)
            if weights_value is None:
                raise TypeError(
                    "SubtensorModule.Weights returned None, not a proven empty list"
                )
            try:
                pairs = list(weights_value)
            except TypeError as exc:
                raise TypeError(
                    "SubtensorModule.Weights did not decode to an iterable"
                ) from exc

            last_raw = self._substrate.query(
                module="SubtensorModule",
                storage_function="LastUpdate",
                params=[netuid],
                block_hash=block_hash,
            )
            last_value = getattr(last_raw, "value", last_raw)
            if last_value is None:
                raise TypeError("SubtensorModule.LastUpdate returned None")
            try:
                last_update = int(last_value[uid])
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise TypeError(
                    f"SubtensorModule.LastUpdate has no integer entry for uid {uid}"
                ) from exc
            finalized_number = int(block_number)
            if last_update < 0 or last_update > finalized_number:
                raise ValueError(
                    "SubtensorModule.LastUpdate lies outside the finalized snapshot: "
                    f"{last_update} not in [0, {finalized_number}]"
                )

            if not pairs:
                return None
            decoded: dict[int, float] = {}
            for entry in pairs:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise TypeError(
                        "SubtensorModule.Weights yielded a malformed uid/weight pair"
                    )
                target_uid, weight = entry
                if isinstance(target_uid, bool) or isinstance(weight, bool):
                    raise TypeError(
                        "SubtensorModule.Weights yielded a boolean uid/weight"
                    )
                target = int(target_uid)
                value = int(weight)
                if target != target_uid or value != weight:
                    raise TypeError(
                        "SubtensorModule.Weights yielded a non-integer uid/weight "
                        f"pair {entry!r}"
                    )
                if target in decoded:
                    raise ValueError(
                        f"SubtensorModule.Weights repeats target uid {target}"
                    )
                if target < 0 or not 0 < value <= U16_MAX:
                    raise ValueError(
                        "SubtensorModule.Weights yielded an out-of-range pair "
                        f"({target}, {value})"
                    )
                decoded[target] = float(value)
            return SubmittedWeights(weights=decoded, block=last_update)

        return self._rpc(_read, "submitted_weights_at_finalized_head")

    def pending_timelocked_commit(self, netuid: int, hotkey: str) -> bool:
        """Scan every CRv4 epoch bucket at one pinned head for ``hotkey``.

        ``bt.Subtensor.get_timelocked_weight_commits`` in the pinned 10.5.0 SDK
        calls ``query_map(..., page_size=1)`` but then inspects only
        ``result.records[0]``.  ``records`` is merely the first page; the query-map
        contract explicitly requires iterating the result to exhaust pagination.
        At an epoch transition that shortcut can inspect a different bucket and
        falsely report that our finalized commit is gone, exposing the previous
        active ``Weights`` vector as a positive denial.

        Query the current runtime's exact storage shape instead:
        ``TimelockedWeightCommits[netuid_index][epoch]``.  Main mechanism id zero
        has ``netuid_index == netuid`` under both Bittensor 10.5 and current
        Subtensor.  The best-head hash is captured once and reused for every page,
        so a rollover cannot change the map while it is being scanned.  Any
        malformed key/value or paging failure raises; UNKNOWN must HOLD, never be
        flattened into a false ``False``.
        """

        def _read() -> bool:
            head_hash = self._substrate.get_chain_head()
            if not isinstance(head_hash, str) or not head_hash.startswith("0x"):
                raise RuntimeError(
                    "get_chain_head returned an invalid hash; pending commit state "
                    "is UNKNOWN"
                )
            keypair = getattr(self, "_hotkey", None)
            hotkey_account_id = None
            if (
                keypair is not None
                and str(getattr(keypair, "ss58_address", hotkey)) == hotkey
            ):
                hotkey_account_id = _account_id_bytes_from_keypair(keypair)
            if hotkey_account_id is None:
                converter = getattr(
                    getattr(getattr(self, "_bt", None), "utils", None),
                    "ss58_address_to_bytes",
                    None,
                )
                if callable(converter):
                    try:
                        candidate = bytes(converter(hotkey))
                    except Exception:  # noqa: BLE001 - decoded string rows may suffice
                        candidate = b""
                    if len(candidate) == 32:
                        hotkey_account_id = candidate

            rows = self._substrate.query_map(
                module="SubtensorModule",
                storage_function="TimelockedWeightCommits",
                params=[int(netuid)],  # mechid=0 => storage index is the netuid
                block_hash=head_hash,
                page_size=100,
                ignore_decoding_errors=False,
            )
            if rows is None or not hasattr(rows, "__iter__"):
                raise TypeError(
                    "TimelockedWeightCommits query_map returned a non-iterable result"
                )
            for row in rows:  # iteration (not .records) exhausts every SDK page
                if not isinstance(row, (tuple, list)) or len(row) != 2:
                    raise TypeError(
                        "TimelockedWeightCommits query_map yielded a malformed row"
                    )
                epoch_raw, bucket_raw = row
                epoch_value = getattr(epoch_raw, "value", epoch_raw)
                if isinstance(epoch_value, bool):
                    raise TypeError("TimelockedWeightCommits epoch key is boolean")
                try:
                    epoch = int(epoch_value)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        "TimelockedWeightCommits epoch key is not an integer"
                    ) from exc
                if epoch < 0:
                    raise ValueError(
                        f"TimelockedWeightCommits epoch key is negative: {epoch}"
                    )
                bucket = getattr(bucket_raw, "value", bucket_raw)
                if _hotkey_in_timelocked_commits(
                    bucket,
                    hotkey,
                    epoch=epoch,
                    account_id=hotkey_account_id,
                ):
                    return True
            return False

        return bool(self._rpc(_read, "timelocked_commits_all_epochs"))

    def uid_for_hotkey(self, hotkey: str, netuid: int) -> int | None:
        uid = self._rpc(
            lambda: self._subtensor.get_uid_for_hotkey_on_subnet(hotkey, netuid),
            "uid_for_hotkey",
        )
        return None if uid is None else int(uid)

    def subnet_owner_hotkey(self, netuid: int) -> str | None:
        raw = self._rpc(
            lambda: self._subtensor.query_subtensor(
                "SubnetOwnerHotkey", params=[netuid]
            ),
            "subnet_owner_hotkey",
        )
        value = getattr(raw, "value", raw)
        if value is None:
            return None
        owner = str(value).strip()
        return owner or None

    def weights_rate_limit(self, netuid: int) -> int:
        return int(
            self._rpc(
                lambda: self._subtensor.weights_rate_limit(netuid),
                "weights_rate_limit",
            )
            or 0
        )

    def commitment_rate_limit(self) -> int:
        value = self._rpc(self._subtensor.tx_rate_limit, "tx_rate_limit")
        if value is None:
            raise RuntimeError("Subtensor.tx_rate_limit returned None")
        return int(value)

    def commitment_usage(
        self, *, netuid: int, ss58: str, block_number: int
    ) -> _CommitmentUsageView:
        """Read ``Commitments.MaxSpace`` and ``UsedSpaceOf`` at one block hash.

        ``MaxSpace`` is mutable root-governed storage, so the historical default
        (3100) is never copied into application code.  ``UsedSpaceOf`` is an
        OptionQuery: only a decoded ``None`` means no usage; every unknown shape
        raises and lets the adapter fail closed.
        """

        def _nonnegative_int(value: Any, label: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"Commitments.{label} returned {type(value).__name__}, not int"
                )
            if value < 0:
                raise ValueError(f"Commitments.{label} returned negative {value}")
            return value

        def _read() -> tuple[Any, Any]:
            block_hash = self._substrate.get_block_hash(block_number)
            if not block_hash:
                raise LookupError(
                    f"block {block_number} is unavailable on this endpoint; "
                    "historical Commitments capacity requires archive state"
                )
            max_space = self._substrate.query(
                module="Commitments",
                storage_function="MaxSpace",
                params=[],
                block_hash=block_hash,
            )
            usage = self._substrate.query(
                module="Commitments",
                storage_function="UsedSpaceOf",
                params=[netuid, ss58],
                block_hash=block_hash,
            )
            return max_space, usage

        max_raw, usage_raw = self._rpc(_read, f"commitment_capacity_at_{block_number}")
        max_value = getattr(max_raw, "value", max_raw)
        max_space = _nonnegative_int(max_value, "MaxSpace")

        value = getattr(usage_raw, "value", usage_raw)
        if value is None:
            usage_epoch = None
            used_space = 0
        else:
            missing = object()
            if isinstance(value, dict):
                epoch_value = value.get("last_epoch", missing)
                used_value = value.get("used_space", missing)
            else:
                epoch_value = getattr(value, "last_epoch", missing)
                used_value = getattr(value, "used_space", missing)
            if epoch_value is missing or used_value is missing:
                raise TypeError(
                    "Commitments.UsedSpaceOf must decode to None or a UsageTracker "
                    "with last_epoch and used_space"
                )
            usage_epoch = _nonnegative_int(epoch_value, "UsedSpaceOf.last_epoch")
            used_space = _nonnegative_int(used_value, "UsedSpaceOf.used_space")

        return _CommitmentUsageView(
            block=block_number,
            max_space=max_space,
            usage_epoch=usage_epoch,
            used_space=used_space,
        )

    def tempo(self, netuid: int) -> int:
        return int(self._rpc(lambda: self._subtensor.tempo(netuid), "tempo"))

    def set_commitment(self, *, netuid: int, payload: str) -> str:
        if self._config.read_only:
            raise ReadOnlyChainError(
                "wallet-free bittensor transport cannot anchor commitments"
            )
        # UNPROVEN (see adapter.anchor_commitment). Not timeout-bounded — same
        # write discipline as set_weights. v10.5.0 set_commitment(data: str) calls
        # data.encode() itself, so `data` MUST be a str — the adapter decodes the
        # ascii anchor bytes to a str before this seam (review critical; passing bytes
        # raised before submission and no anchor was ever published).
        with self._sdk_lock:
            result = self._subtensor.set_commitment(
                wallet=self._wallet,
                netuid=netuid,
                data=payload,
                mev_protection=False,
                raise_error=False,
                wait_for_inclusion=True,
                wait_for_finalization=True,
                wait_for_revealed_execution=True,
            )
        outcome = _parse_chain_result(result)
        if outcome.transport_error is not None:
            raise outcome.transport_error
        if not outcome.success:
            raise OSError(f"set_commitment rejected: {outcome.message}")
        # The txid is the extrinsic RECEIPT hash, NOT the response message.
        return outcome.receipt_hash or outcome.message or "committed"

    def get_commitment(
        self, *, netuid: int, ss58: str, block_number: int | None = None
    ) -> bytes | None:
        # UNPROVEN (see set_commitment). Reads the account's committed data back — the
        # v10.5.0's get_commitment takes a UID, not an ss58_address. The correct
        # account-keyed API is get_commitment_metadata(netuid, hotkey_ss58); using
        # the former with ss58_address= raised TypeError on every live read.
        raw = self._commitment_metadata(
            netuid=netuid,
            ss58=ss58,
            block_number=block_number,
            rpc_name="get_commitment_metadata",
        )
        if raw is None or raw == "" or raw == b"":
            return None
        if isinstance(raw, dict):
            from bittensor.core.chain_data.utils import decode_metadata

            decoded = decode_metadata(raw)
            return None if not decoded else decoded.encode("utf-8")
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            # The authority commits ascii payload bytes; if the SDK hands back hex
            # (0x-prefixed) decode it, else treat the string as the ascii payload.
            text = raw
            if text.startswith("0x"):
                try:
                    return bytes.fromhex(text[2:])
                except ValueError:
                    return text.encode("utf-8", "ignore")
            return text.encode("utf-8", "ignore")
        raise TypeError(
            f"get_commitment_metadata returned unsupported {type(raw).__name__} shape"
        )

    def get_commitment_block(
        self, *, netuid: int, ss58: str, block_number: int | None = None
    ) -> int | None:
        # UNPROVEN. The Commitments pallet stores a per-account
        # Registration record whose `block` field is the inclusion block the commitment
        # was set at. Raw-query it (the SDK's get_commitment returns only the decoded
        # DATA, not the block). Liberal about the returned shape: a mapping with a
        # `block` key, or an object exposing `.block`. Timeout-bounded like every read.
        raw = self._commitment_metadata(
            netuid=netuid,
            ss58=ss58,
            block_number=block_number,
            rpc_name="commitment_metadata_block",
        )
        value = getattr(raw, "value", raw)
        if not value:
            return None
        block: Any = None
        if isinstance(value, dict):
            block = value.get("block")
        else:
            block = getattr(value, "block", None)
        return None if block is None else int(block)

    def _commitment_metadata(
        self,
        *,
        netuid: int,
        ss58: str,
        block_number: int | None,
        rpc_name: str,
    ) -> Any:
        """Read CommitmentOf, refusing the SDK's pruned-block head fallback."""

        def _read() -> Any:
            if block_number is not None:
                # v10.5 `determine_block_hash(block)` returns None when a pruned
                # endpoint cannot resolve the height; substrate.query interprets
                # block_hash=None as HEAD. Prove the hash exists first so archive
                # absence is UNKNOWN rather than a silently mislabeled head read.
                if self._substrate.get_block_hash(block_number) is None:
                    raise LookupError(
                        f"block {block_number} is unavailable on this endpoint; "
                        "historical CommitmentOf requires archive state"
                    )
            return self._subtensor.get_commitment_metadata(
                netuid=netuid, hotkey_ss58=ss58, block=block_number
            )

        name = rpc_name if block_number is None else f"{rpc_name}_at_{block_number}"
        return self._rpc(_read, name)

    def get_block_hash(self, block_number: int) -> str | None:
        # UNPROVEN. block_hash(close_block + K) is the un-grindable
        # beacon source. A block not yet produced comes back None from substrate (do NOT
        # stringify it — that would substitute a "None" hash); the adapter treats None as
        # "not finalized yet -> HOLD". Timeout-bounded like every short read.
        raw = self._rpc(
            lambda: self._substrate.get_block_hash(block_number),
            "get_block_hash",
        )
        return None if raw is None else str(raw)

    def block_time(self, block_number: int) -> datetime | None:
        def _read() -> datetime | None:
            # get_timestamp alone may report a confusing decode error for a future
            # height. A missing block hash is the positive "not produced yet" case.
            if self._substrate.get_block_hash(block_number) is None:
                return None
            return self._subtensor.get_timestamp(block=block_number)

        value = self._rpc(_read, "block_time")
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(
                f"Subtensor.get_timestamp returned {type(value).__name__}, expected datetime"
            )
        return value

    def signer_hotkey(self) -> str:
        if self._config.read_only:
            raise ReadOnlyChainError(
                "wallet-free bittensor transport has no signer identity"
            )
        value = getattr(self._hotkey, "ss58_address", None)
        if not value:
            raise RuntimeError("loaded bittensor hotkey exposes no ss58_address")
        return str(value)

    def sign_hotkey(self, payload: bytes) -> bytes:
        if self._config.read_only:
            raise ReadOnlyChainError("wallet-free bittensor transport cannot sign")
        signature = self._hotkey.sign(payload)
        if not isinstance(signature, (bytes, bytearray)):
            raise TypeError(
                f"hotkey.sign returned {type(signature).__name__}, expected bytes"
            )
        return bytes(signature)

    def close(self) -> None:
        try:
            with self._sdk_lock:
                self._subtensor.close()
        except Exception:  # noqa: BLE001
            pass


class _KeypairWallet:
    """A minimal wallet shim exposing `.hotkey` + a no-op `unlock_hotkey()`.

    Production validators hold a bare Keypair (coldkey never touched at runtime);
    `set_weights` only needs a `.hotkey` and an `unlock_hotkey()` no-op.
    """

    def __init__(self, hotkey: Any) -> None:
        self.hotkey = hotkey

    def unlock_hotkey(self) -> Any:
        return self.hotkey


@dataclass(frozen=True)
class _ChainOutcome:
    """A normalized write result: did it land, why, and its extrinsic receipt hash."""

    success: bool
    message: str
    receipt_hash: str | None = None
    #: bittensor v10 catches socket/RPC exceptions and returns them on
    #: ``ExtrinsicResponse.error`` when ``raise_error=False``. Preserve that
    #: boundary so the adapter can RAISE UNKNOWN while ordinary dispatch errors
    #: (whose ``error`` is a decoded mapping) remain explicit rejections.
    transport_error: BaseException | None = None


def _extract_receipt_hash(receipt: Any) -> str | None:
    """The extrinsic hash off a v10 ExtrinsicReceipt (duck-typed), else None.

    Tries the `.extrinsic_hash` / `.hash` attributes, then a
    `get_extrinsic_hash()`-equivalent method — whatever the pinned SDK exposes.
    """
    if receipt is None:
        return None
    for attr in ("extrinsic_hash", "hash"):
        value = getattr(receipt, attr, None)
        if value:
            return str(value)
    getter = getattr(receipt, "get_extrinsic_hash", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001 - a receipt that cannot hash yields no txid
            value = None
        if value:
            return str(value)
    return None


_PUBLIC_DISPATCH_ERROR_KEYS = frozenset(
    {"code", "error", "message", "module", "name", "type"}
)


def _safe_dispatch_error_text(raw_error: object) -> str:
    """Render only bounded public fields from a decoded SDK dispatch error.

    Bittensor may put ``SettingWeightsTooFast`` in ``ExtrinsicResponse.error``
    while leaving ``message`` empty.  Do not call ``repr``/``str`` on arbitrary
    objects and do not serialize the complete mapping: it may grow without bound
    or contain caller context.  Only the small public dispatch vocabulary above
    is traversed, with deterministic ordering and hard depth/count/length caps.
    """
    if not isinstance(raw_error, Mapping):
        return ""

    fragments: list[str] = []

    def scalar(value: object) -> str | None:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if not isinstance(value, str):
            return None
        compact = " ".join(value.split())
        # Dispatch identifiers/messages need letters, digits and modest public
        # punctuation only. Control characters and opaque encoded blobs are not
        # useful operator diagnostics.
        clean = "".join(
            char for char in compact if char.isalnum() or char in " ._:/-[]()"
        )
        return clean[:96] or None

    def visit(value: object, *, path: str, depth: int) -> None:
        if depth > 3 or len(fragments) >= 8:
            return
        direct = scalar(value)
        if direct is not None:
            fragments.append(f"{path}={direct}" if path else direct)
            return
        if isinstance(value, Mapping):
            try:
                public_items = sorted(
                    (
                        (key, child)
                        for key, child in value.items()
                        if isinstance(key, str)
                        and key.lower() in _PUBLIC_DISPATCH_ERROR_KEYS
                    ),
                    key=lambda item: item[0],
                )
            except Exception:  # noqa: BLE001 - unreadable mapping is omitted
                return
            for key, child in public_items:
                child_path = f"{path}.{key}" if path else key
                visit(child, path=child_path, depth=depth + 1)
                if len(fragments) >= 8:
                    return
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value[:4]):
                child_path = f"{path}[{index}]" if path else f"[{index}]"
                visit(child, path=child_path, depth=depth + 1)
                if len(fragments) >= 8:
                    return

    visit(raw_error, path="", depth=0)
    return "; ".join(fragments)[:256]


def _parse_chain_result(result: Any) -> _ChainOutcome:
    """Classify an SDK write result WITHOUT ever inferring an implicit success.

    Order:
      1. an ExtrinsicResponse-shaped object (duck-typed: has ``.success``) — the v10
         contract: read ``.success`` / ``.message`` / ``.extrinsic_receipt``.
         ``bool(obj)`` is DELIBERATELY NOT consulted: an ExtrinsicResponse can be
         truthy for a REJECTION, so `bool(response)` published weights that never
         landed;
      2. the legacy ``(success: bool, message: str)`` tuple;
      3. a bare ``bool``;
      4. anything else -> RAISE. An unrecognized shape is a FAILURE, never an
         implicit success (the adapter maps the raise to a transport failure).
    """
    if hasattr(result, "success"):
        raw_error = getattr(result, "error", None)
        message = str(getattr(result, "message", "") or "")
        dispatch_error = _safe_dispatch_error_text(raw_error)
        if dispatch_error:
            rendered = f"dispatch_error[{dispatch_error}]"
            if not message:
                message = rendered
        return _ChainOutcome(
            success=bool(result.success),
            message=message,
            receipt_hash=_extract_receipt_hash(
                getattr(result, "extrinsic_receipt", None)
            ),
            transport_error=(
                raw_error
                if isinstance(raw_error, BaseException)
                and _is_ambiguous_transport_error(raw_error)
                else None
            ),
        )
    if isinstance(result, tuple) and len(result) == 2:
        success, message = result
        return _ChainOutcome(success=bool(success), message=str(message or ""))
    if isinstance(result, bool):
        return _ChainOutcome(success=result, message="")
    raise TypeError(
        f"unrecognized set_weights/commitment result {type(result).__name__!r};"
        " refusing to treat it as an implicit success (bittensor v10 returns an"
        " ExtrinsicResponse — .success / .message / .extrinsic_receipt)"
    )


def _is_ambiguous_transport_error(error: BaseException) -> bool:
    """Whether an SDK-caught exception leaves extrinsic inclusion ambiguous.

    Bittensor 10.5 returns decoded dispatch failures in ``response.error`` as a
    mapping, but catches websocket/RPC failures and stores the actual exception.
    Some definite local/transaction errors are exceptions too, so do not flatten
    every ``BaseException`` into UNKNOWN: preserve explicit rejection semantics.
    Unknown exception classes deliberately bias to ambiguity/readback.
    """
    if isinstance(error, (TypeError, ValueError, AssertionError)):
        return False  # definite local validation/encoding failure, before gossip

    mro_names = {base.__name__ for base in type(error).__mro__}
    if "ChainTransactionError" in mro_names:
        return False
    if "ChainConnectionError" in mro_names:
        return True

    message = str(error).lower()
    explicit_pool_markers = (
        "already imported",
        "priority is too low",
        "invalid transaction",
        "transaction is outdated",
        "transaction is stale",
        "future transaction",
    )
    if any(marker in message for marker in explicit_pool_markers):
        return False
    return True


def _account_id_bytes_from_keypair(keypair: Any) -> bytes | None:
    """Extract one 32-byte AccountId from a bittensor keypair-like object."""
    public_key = getattr(keypair, "public_key", None)
    if isinstance(public_key, (bytes, bytearray)):
        raw = bytes(public_key)
        return raw if len(raw) == 32 else None
    if isinstance(public_key, str):
        candidate = public_key.removeprefix("0x")
        try:
            raw = bytes.fromhex(candidate)
        except ValueError:
            return None
        return raw if len(raw) == 32 else None
    return None


def _storage_account_id_bytes(value: Any) -> bytes | None:
    """Decode AccountId shapes emitted for TimelockedWeightCommits by ASI 2.2.1.

    Live production evidence shows this storage can decode an AccountId32 as
    ``((b0, ..., b31),)`` rather than an SS58 string.  Accept bytes, a flat
    32-integer vector, and arbitrarily wrapped one-element tuple/list/ScaleValue
    forms.  ``None`` means this was not a recognizable raw AccountId shape.
    """
    raw = getattr(value, "value", value)
    if isinstance(raw, (bytes, bytearray)):
        account = bytes(raw)
        return account if len(account) == 32 else None
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return _storage_account_id_bytes(raw[0])
        if len(raw) == 32 and all(
            isinstance(part, int) and not isinstance(part, bool) for part in raw
        ):
            try:
                return bytes(raw)
            except ValueError:
                return None
    return None


def _hotkey_in_timelocked_commits(
    commits: Any,
    hotkey: str,
    *,
    epoch: int | None = None,
    account_id: bytes | None = None,
) -> bool:
    """Is `hotkey` present in one decoded CRv4 commitment collection?

    The raw storage value is a flat list of ``(account_id, commit_block,
    ciphertext, reveal_round)`` records. Account ids may be decoded as SS58 strings
    or nested 32-byte integer tuples; both are compared when ``account_id`` is
    available. A small set of older/mocked mapping shapes is accepted, but an
    unknown shape RAISES: treating undecodable state as an empty list would turn a
    pending commit into a false DENIAL and abandon a possibly-live intent.

    * a mapping keyed by hotkey -> a direct lookup;
    * a mapping keyed by EPOCH (int) -> select the relevant epoch's commits, then
      scan those;
    * a flat iterable of hotkeys / (hotkey, ...) records / objects with a `.hotkey`.
    """
    if commits is None:
        raise TypeError("timelocked commitment result is None, not a proven empty list")
    if not commits:
        return False
    if isinstance(commits, dict):
        # Hotkey-keyed: our hotkey is a direct key. (Epochs are ints, so a str
        # hotkey can never collide with an epoch key.)
        if hotkey in commits:
            return bool(commits[hotkey])
        # A mapping whose keys are non-numeric strings is hotkey-keyed. The
        # requested hotkey's absence is therefore a decoded, positive False; its
        # values are commit details, not buckets to scan as hotkeys.
        if commits and all(
            isinstance(key, str) and not key.isdecimal() for key in commits
        ):
            return False
        # Epoch-keyed: pick the epoch we asked about; a different epoch is not
        # ours-pending-now. Without a resolvable epoch, scan every bucket.
        if epoch is not None:
            bucket = commits.get(epoch, commits.get(str(epoch)))
            return _iter_has_hotkey(bucket, hotkey, account_id=account_id)
        return any(
            _iter_has_hotkey(bucket, hotkey, account_id=account_id)
            for bucket in commits.values()
        )
    if not isinstance(commits, (list, tuple, set, frozenset)):
        raise TypeError(
            f"unsupported timelocked commitment result {type(commits).__name__}"
        )
    return _iter_has_hotkey(commits, hotkey, account_id=account_id)


def _iter_has_hotkey(
    entries: Any, hotkey: str, *, account_id: bytes | None = None
) -> bool:
    """Scan a decoded flat commit list; malformed records fail closed."""
    if not entries:
        return False
    if isinstance(entries, str):
        return entries == hotkey
    if not isinstance(entries, (list, tuple, set, frozenset)):
        raise TypeError(
            f"unsupported timelocked commitment bucket {type(entries).__name__}"
        )
    for entry in entries:
        candidate: Any
        if isinstance(entry, str):
            candidate = entry
        elif isinstance(entry, (tuple, list)):
            if not entry:
                raise TypeError("empty timelocked commitment record")
            candidate = entry[0]
        elif hasattr(entry, "ss58"):
            candidate = getattr(entry, "ss58")
        elif hasattr(entry, "hotkey"):
            candidate = getattr(entry, "hotkey")
        else:
            raise TypeError(
                f"unsupported timelocked commitment record {type(entry).__name__}"
            )
        if candidate == hotkey or str(candidate) == hotkey:
            return True
        candidate_account_id = _storage_account_id_bytes(candidate)
        if candidate_account_id is not None:
            if account_id is None:
                raise TypeError(
                    "TimelockedWeightCommits returned a raw AccountId but the "
                    "validator hotkey could not be decoded for comparison"
                )
            if candidate_account_id == account_id:
                return True
            continue
        # A different decoded SS58 string is a valid non-match. Every other
        # unrecognized AccountId shape is UNKNOWN and must fail closed.
        if isinstance(candidate, str):
            continue
        raise TypeError(
            "unsupported TimelockedWeightCommits AccountId shape "
            f"{type(candidate).__name__}"
        )
    return False
