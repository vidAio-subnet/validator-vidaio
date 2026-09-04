"""Report-mode ChainAdapter implementations (the project design record rule 8).

- HttpChainAdapter: the ChainAdapter Protocol over a running chainsim
  (vidaio.chainsim.service.ChainSim) — what services use in report mode when
  running as separate processes. `refresh()` pulls GET /neurons (block +
  neuron list) into the cached snapshot; writes POST /weights and /anchor.
  Transport failures on writes surface as OSError/TimeoutError so callers'
  resilience envelopes (vidaio.core.retry_async over TimeoutError/OSError)
  retry them; a refresh failure keeps the previous cached snapshot (reads are
  snapshots by contract — a flaky sim must not crash a service loop) BUT is
  never silent: `last_refresh_error`, `snapshot_age()` and `has_fresh_snapshot()`
  expose it, and `neurons()` on an adapter that has NEVER refreshed raises
  ChainStateUnavailable instead of returning an empty subnet.

  `submitted_weights(hotkey)` (the optional SubmittedWeightsReader surface) is
  a LIVE `GET /weights/{hotkey}` — the vector the sim records for that hotkey,
  or None when it records none. It is what proves a specific weight write
  landed; EmbeddedReportingChain inherits InMemoryChain's implementation.

  Mutations carry the identity's bearer token (`auth_token`) — the sim's
  stand-in for a signed extrinsic; the real bittensor adapter will carry a
  keypair in exactly this spot. `register()` claims the hotkey and captures the
  token the sim issues, so a process that registers itself needs no config.

- EmbeddedReportingChain: InMemoryChain that additionally journals every
  set_weights call (accepted AND rejected, with its outcome) and every anchor
  to a JSONL file — for single-process harness runs without the HTTP sim.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx

from vidaio.chain.adapter import (
    ChainCommitmentRecord,
    ChainNeuron,
    ChainStateUnavailable,
    InMemoryChain,
    SetWeightsResult,
    SubmittedWeights,
    earliest_reanchor_block,
    parse_anchor_digest,
)
from vidaio.core import get_logger
from vidaio.tokenomics.quantize import max_normalize_u16, quantize_u16


class HttpChainAdapter:
    """ChainAdapter over a chainsim URL. The validator hotkey identifies the caller."""

    def __init__(
        self,
        base_url: str,
        *,
        validator_hotkey: str,
        anchor_hotkey: str | None = None,
        auth_token: str | None = None,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._hotkey = validator_hotkey
        # an internal review: the Scoring Authority's identity whose anchors are HONORED. The
        # anchor READ path (read_anchor / read_anchor_block) considers ONLY anchors written by
        # this account, mirroring the production BittensorChainAdapter (which reads the
        # Commitments pallet for `anchor_hotkey or validator_hotkey`). Without this the sim's
        # `/anchor` accepts ANY registered hotkey and the reader honored the LAST matching
        # payload regardless of who wrote it — so any participant could REPLACE the effective
        # anchor (forcing global REFUSE/HOLD or swapping in a competing digest). Empty falls
        # back to `validator_hotkey` (a self-anchoring single-node deployment — the same
        # fallback the real adapter uses).
        self._anchor_authority = (anchor_hotkey or "").strip() or validator_hotkey
        self._auth_token = auth_token or None
        self._timeout = timeout_seconds
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._async_client = async_client or httpx.AsyncClient(timeout=timeout_seconds)
        self._clock = clock
        self._log = get_logger("chain.http")
        self._block = 0
        self._neurons: list[ChainNeuron] = []
        self._last_successful_refresh: float | None = None
        self._last_refresh_error: str | None = None

    def _url(self, path: str) -> str:
        return self._base_url + path

    # -- identity (token ~ signature; see vidaio/chainsim/service.py) ---------------

    @property
    def validator_hotkey(self) -> str:
        return self._hotkey

    @property
    def anchor_authority_hotkey(self) -> str:
        """The Scoring Authority identity whose anchors the read path honors."""
        return self._anchor_authority

    @property
    def auth_token(self) -> str | None:
        """The bearer credential mutations are signed with (None = unauthenticated)."""
        return self._auth_token

    def set_auth_token(self, token: str | None) -> None:
        self._auth_token = token or None

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._auth_token}"} if self._auth_token else {}

    # -- reads (cached snapshot) ---------------------------------------------------

    def current_block(self) -> int:
        """Last observed block; 0 until the first successful refresh."""
        return self._block

    def neurons(self) -> list[ChainNeuron]:
        """The cached neuron snapshot.

        Raises ChainStateUnavailable if no refresh has EVER succeeded: an empty
        list would be indistinguishable from a genuinely empty subnet, and that
        ambiguity is exactly how a startup race turns into "successful" empty
        rounds and silently omitted weights.
        """
        if self._last_successful_refresh is None:
            raise ChainStateUnavailable(
                f"no chain snapshot from {self._base_url} yet"
                + (
                    f" (last refresh error: {self._last_refresh_error})"
                    if self._last_refresh_error
                    else " (refresh() has not been called)"
                )
            )
        return list(self._neurons)

    def get_burn_uid(self) -> int:
        """Resolve the report chain's configured subnet-owner identity to its uid.

        Chainless report mode has no SubnetOwnerHotkey storage item.  Its equivalent
        trusted identity is ``anchor_hotkey`` (the account whose authority anchors are
        accepted).  Resolve that hotkey against the last successfully refreshed
        chainsim neuron registry; never assume that the first registered uid is zero.
        """
        neurons = self.neurons()  # raises when no authoritative snapshot exists
        matches = [n.uid for n in neurons if n.hotkey == self._anchor_authority]
        if len(matches) != 1:
            raise ChainStateUnavailable(
                "cannot resolve the report-mode subnet-owner burn uid: "
                f"anchor authority {self._anchor_authority!r} has {len(matches)} "
                "registered uid matches"
            )
        uid = int(matches[0])
        if uid < 0:
            raise ChainStateUnavailable(
                f"report chain returned a negative burn uid for {self._anchor_authority!r}"
            )
        return uid

    def submitted_weights(self, hotkey: str) -> SubmittedWeights | None:
        """GET /weights/{hotkey}: the vector the sim currently records for it.

        A LIVE read, not a snapshot read — the question ("did my write land?")
        is about the chain right now, and the sim answers it in one call. The
        contract of SubmittedWeightsReader is strict about the two outcomes:

        - `{"vector": null}` -> None: the sim POSITIVELY holds no weights for
          this hotkey;
        - anything unreadable (transport failure, non-200, malformed body) ->
          ChainStateUnavailable. It must never be flattened into None, because
          None denies a weight intent and a denied intent is eventually
          abandoned unpublished.
        """
        url = self._url("/weights/" + quote(hotkey, safe=""))
        try:
            response = self._client.get(url, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            vector = data["vector"]
            if vector is None:
                return None
            weights = {int(uid): float(w) for uid, w in vector.items()}
            block = data.get("block")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read the weights {self._base_url} records for {hotkey!r}:"
                f" {type(exc).__name__}: {exc}"
            ) from exc
        return SubmittedWeights(
            weights=weights, block=None if block is None else int(block)
        )

    # -- anchor read (EpochAnchorReadable) -----------------------------------------

    def read_anchor(self, *, netuid: int, epoch_id: int, domain: str) -> str | None:
        """The anchored digest for `(netuid, epoch_id)`, read from the sim's `/state`.

        A LIVE read of what the sim actually recorded (the third verification leg,
        genuinely exercised in report mode — never a None-skip). Parses the
        domain-tagged commitment payloads out of `state.anchors[*].payload_hex`. A
        transport failure / non-200 / malformed body RAISES `ChainStateUnavailable`
        (an unreadable chain must HOLD, never be read as "no anchor"); a clean read
        with no matching anchor returns None (a substituted pointer is then caught).

        an internal review: only anchors written by the Scoring Authority
        (`self._anchor_authority`) are considered — a competing anchor from any other
        registered hotkey is IGNORED, so a non-authority participant cannot REPLACE the
        effective anchor. This binds report-mode anchors to the authority account exactly
        like the production adapter (which reads only that account's commitment).
        """
        try:
            response = self._client.get(self._url("/state"), timeout=self._timeout)
            response.raise_for_status()
            anchors = response.json()["anchors"]
            payloads = [
                bytes.fromhex(a["payload_hex"])
                for a in anchors
                if a.get("hotkey") == self._anchor_authority
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read anchors from {self._base_url} /state:"
                f" {type(exc).__name__}: {exc}"
            ) from exc
        return parse_anchor_digest(
            payloads, netuid=netuid, epoch_id=epoch_id, domain=domain
        )

    # -- anchor inclusion-block read (EpochAnchorBlockReadable) --------------------

    def read_anchor_block(
        self, *, netuid: int, epoch_id: int, domain: str
    ) -> int | None:
        """The INCLUSION BLOCK of `(netuid, epoch_id)`'s anchor.

        A LIVE read of the sim's `/state`: returns the EARLIEST inclusion `block` among anchors
        whose `payload_hex` EQUALS the COMMITTED anchor (the last-matching payload `read_anchor`
        selects). The auditor uses it to confirm the item set was committed BEFORE the beacon block
        could be known (`anchor_block <= close_block + K`); an idempotent recovery RE-ANCHOR of the
        SAME payload (a crash between the chain write and the index update) records a DUPLICATE
        identical anchor at a later block, and choosing the earliest keeps that from being mis-read
        as a grind. A transport failure / non-200 / malformed body RAISES
        `ChainStateUnavailable` (an unreadable chain must HOLD, never read as "no anchor"); a clean
        read with no matching anchor returns None (a positive "no anchor yet").

        an internal review: as in `read_anchor`, only anchors written by the Scoring
        Authority (`self._anchor_authority`) are considered — a non-authority hotkey's
        anchor cannot move (or substitute) the inclusion block the beacon grind check reads.
        """
        try:
            response = self._client.get(self._url("/state"), timeout=self._timeout)
            response.raise_for_status()
            anchors = response.json()["anchors"]
            entries = [
                (bytes.fromhex(a["payload_hex"]), int(a["block"]))
                for a in anchors
                if a.get("hotkey") == self._anchor_authority
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read the anchor inclusion block from {self._base_url} /state:"
                f" {type(exc).__name__}: {exc}"
            ) from exc
        return earliest_reanchor_block(
            entries, netuid=netuid, epoch_id=epoch_id, domain=domain
        )

    def read_anchor_at(
        self,
        *,
        netuid: int,
        epoch_id: int,
        domain: str,
        block_number: int,
    ) -> str | None:
        """Read report-chain one-slot state at an exact inclusion block.

        The simulator journals all writes, so collapse same-authority writes in
        the requested block to their LAST value before parsing it. This mirrors
        substrate's one storage value per block and exposes same-block overwrite
        bugs instead of letting the append-only journal hide them.
        """
        try:
            response = self._client.get(self._url("/state"), timeout=self._timeout)
            response.raise_for_status()
            anchors = response.json()["anchors"]
            payloads = [
                bytes.fromhex(a["payload_hex"])
                for a in anchors
                if a.get("hotkey") == self._anchor_authority
                and int(a["block"]) == block_number
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read historical anchor block {block_number} from "
                f"{self._base_url} /state: {type(exc).__name__}: {exc}"
            ) from exc
        if not payloads:
            return None
        return parse_anchor_digest(
            [payloads[-1]], netuid=netuid, epoch_id=epoch_id, domain=domain
        )

    def read_commitment_record(
        self, *, netuid: int, block_number: int | None = None
    ) -> ChainCommitmentRecord | None:
        """Read the authority's raw commitment slot from report-chain history.

        The simulator exposes an append-only journal, while substrate exposes
        one value as of a requested block.  Selecting the last authority write at
        or before that block makes the report adapter model the latter exactly.
        """

        del netuid  # ChainSim is one configured subnet, as are its other reads
        if block_number is not None and (
            isinstance(block_number, bool) or block_number < 0
        ):
            raise ValueError("block_number must be a non-negative integer")
        try:
            response = self._client.get(self._url("/state"), timeout=self._timeout)
            response.raise_for_status()
            anchors = response.json()["anchors"]
            entries = [
                (bytes.fromhex(a["payload_hex"]), int(a["block"]))
                for a in anchors
                if a.get("hotkey") == self._anchor_authority
                and (block_number is None or int(a["block"]) <= block_number)
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read the raw commitment record from {self._base_url} /state: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not entries:
            return None
        payload, included_at = entries[-1]
        return ChainCommitmentRecord(payload=payload, block=included_at)

    def finalized_block(self) -> int:
        """The simulator's current deterministic block is final immediately."""
        try:
            response = self._client.get(self._url("/state"), timeout=self._timeout)
            response.raise_for_status()
            block = int(response.json()["block"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read finalized block from {self._base_url} /state: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._block = block
        return block

    # -- block-hash read (BlockHashReadable) ---------------------------------------

    def block_hash(self, block_number: int) -> str | None:
        """The chainsim block HASH at `block_number`, else None.

        A LIVE read of the sim's `/block_hash/{n}` endpoint: the sim owns the block clock,
        so it is authoritative about whether block `n` has been produced. Returns the
        64-hex hash for a produced block (the sim derives it with the SAME
        `synthetic_block_hash` as `InMemoryChain`, so report-mode and in-memory agree
        byte-for-byte), and None when `n` is not yet produced (`n` > the sim's current
        block) — the beacon block is then simply not finalized, so the auditor HOLDS. A
        transport failure / non-200 / malformed body RAISES `ChainStateUnavailable` (an
        unreadable chain HOLDS, never a substituted None for a produced block).
        """
        try:
            response = self._client.get(
                self._url(f"/block_hash/{block_number}"), timeout=self._timeout
            )
            response.raise_for_status()
            value = response.json()["hash"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read block_hash({block_number}) from {self._base_url}:"
                f" {type(exc).__name__}: {exc}"
            ) from exc
        return None if value is None else str(value)

    def block_time(self, block_number: int) -> datetime | None:
        """The chainsim block's wall-clock UTC time at `block_number`, else None (round-9 #6).

        A LIVE read of the sim's `/block_time/{n}` endpoint (the sim owns the block clock). The
        auditor binds `EpochLog.created_at` to the epoch's close_block time with this. Returns a
        timezone-aware UTC datetime for a produced block, None when `n` is not yet produced. A
        transport failure / non-200 / malformed body RAISES `ChainStateUnavailable` (an
        unreadable chain HOLDs, never a substituted time).
        """
        try:
            response = self._client.get(
                self._url(f"/block_time/{block_number}"), timeout=self._timeout
            )
            response.raise_for_status()
            value = response.json()["time"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read block_time({block_number}) from {self._base_url}:"
                f" {type(exc).__name__}: {exc}"
            ) from exc
        if value is None:
            return None
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    def tempo(self) -> int:
        """The sim's live tempo (blocks per epoch), read from `/state.config.tempo`.

        Report-mode analogue of the real adapter's `subtensor.tempo(netuid)`: the
        epoch driver derives `blocks_per_epoch` from the LIVE chain rather than a
        hardcoded constant (#14). Raises on an unreadable/ malformed state.
        """
        try:
            response = self._client.get(self._url("/state"), timeout=self._timeout)
            response.raise_for_status()
            return int(response.json()["config"]["tempo"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ChainStateUnavailable(
                f"cannot read tempo from {self._base_url} /state:"
                f" {type(exc).__name__}: {exc}"
            ) from exc

    # -- freshness -----------------------------------------------------------------

    @property
    def last_successful_refresh(self) -> float | None:
        """Wall-clock (epoch seconds) stamp of the last successful refresh, or None."""
        return self._last_successful_refresh

    @property
    def last_refresh_error(self) -> str | None:
        """The failure that ended the last refresh attempt; None if it succeeded."""
        return self._last_refresh_error

    def snapshot_age(self, now: float) -> float | None:
        """Seconds since the last successful refresh; None if never refreshed."""
        if self._last_successful_refresh is None:
            return None
        return max(0.0, now - self._last_successful_refresh)

    def has_fresh_snapshot(self, now: float, max_age_seconds: float) -> bool:
        """False when never refreshed OR the snapshot is older than max_age_seconds.

        `now` is wall-clock epoch seconds (`time.time()`), the Protocol's clock
        family and this adapter's default `clock`.
        """
        age = self.snapshot_age(now)
        return age is not None and age <= max_age_seconds

    def refresh(self) -> None:
        """Pull GET /neurons into the cached snapshot. Never raises (by contract).

        Both halves — fetch AND decode — are guarded: a malformed payload must
        not leave the snapshot half-applied or crash the caller's loop. Success
        clears `last_refresh_error` and stamps `last_successful_refresh`.
        """
        try:
            response = self._client.get(self._url("/neurons"), timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
            block = int(data["block"])
            neurons = [
                ChainNeuron(
                    uid=int(n["uid"]),
                    hotkey=str(n["hotkey"]),
                    coldkey=str(n["coldkey"]),
                    ip=str(n["ip"]),
                    alpha_stake=float(n["alpha_stake"]),
                    emission=float(n["emission"]),
                    is_validator=bool(n["is_validator"]),
                    last_update=int(n["last_update"]),
                    # an internal review: the block the neuron registered at, so the auditor
                    # can cross-check committed window evidence + re-derive the full-window flag.
                    registration_block=int(n.get("registered_block", 0)),
                )
                for n in data["neurons"]
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            self._last_refresh_error = f"{type(exc).__name__}: {exc}"
            self._log.warning(
                "chainsim refresh failed — keeping cached snapshot",
                extra={
                    "url": self._base_url,
                    "error": self._last_refresh_error,
                    "ever_refreshed": self._last_successful_refresh is not None,
                },
            )
            return
        self._block = block
        self._neurons = neurons
        self._last_successful_refresh = self._clock()
        self._last_refresh_error = None

    # -- writes --------------------------------------------------------------------

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        del hotkeys  # the chainsim reconciles bindings server-side; not forwarded
        body = {
            "hotkey": self._hotkey,
            "vector": {str(uid): float(w) for uid, w in weights.items()},
            "version_key": version_key,
        }
        try:
            response = await self._async_client.post(
                self._url("/weights"), json=body, headers=self._auth_headers()
            )
        except httpx.TransportError as exc:
            raise OSError(f"chainsim unreachable: {exc}") from exc
        if response.status_code in (401, 403):
            # Not retryable: this identity may not set weights (no/wrong token,
            # or it is not a registered validator). Report it as a failed set —
            # the weight-setter's failure path already surfaces the message.
            self._log.error(
                "chainsim rejected set_weights authorization",
                extra={
                    "hotkey": self._hotkey,
                    "status": response.status_code,
                    "detail": response.text,
                    "has_token": self._auth_token is not None,
                },
            )
        if response.status_code >= 400:
            return SetWeightsResult(
                success=False,
                block=self._block,
                message=f"HTTP {response.status_code}: {response.text}",
            )
        data = response.json()
        success = bool(data["success"])
        return SetWeightsResult(
            success=success,
            block=int(data["block"]),
            message=str(data.get("message", "")),
            # Report the EXACT u16 vector that landed on the chain's grid (review
            # round-4 #3), so the caller publishes/anchors chain state rather than
            # its pre-quantization float intent — even when the two are merely
            # scale-equivalent. The sim records the floats we sent (its own store),
            # but the pinned SDK max-normalizes VIDAIO's sum-grid immediately
            # before emission. Empty on a failed/rejected submit (nothing landed).
            submitted=(
                max_normalize_u16(quantize_u16(dict(weights))) if success else {}
            ),
        )

    async def anchor_commitment(self, payload: bytes) -> str:
        if len(payload) > 128:
            raise ValueError("chain payload must be <= 128 bytes")
        body = {"payload_hex": payload.hex(), "hotkey": self._hotkey}
        try:
            response = await self._async_client.post(
                self._url("/anchor"), json=body, headers=self._auth_headers()
            )
        except httpx.TransportError as exc:
            raise OSError(f"chainsim unreachable: {exc}") from exc
        if response.status_code in (401, 403):
            # PermissionError IS an OSError, so callers' retry envelopes still
            # handle it — but the message says "credential", not "chain down".
            raise PermissionError(
                f"chainsim rejected the anchor for hotkey {self._hotkey!r}:"
                f" HTTP {response.status_code}: {response.text}"
            )
        if response.status_code >= 400:
            # retryable by the caller's OSError envelope; the sim is authoritative
            raise OSError(f"chainsim anchor failed: HTTP {response.status_code}: {response.text}")
        return str(response.json()["txid"])

    # -- convenience ---------------------------------------------------------------

    def register(
        self,
        *,
        coldkey: str = "local",
        ip: str = "127.0.0.1",
        role: str = "validator",
        alpha_stake: float | None = None,
    ) -> int:
        """Register this adapter's hotkey on the sim and CAPTURE its token. Returns uid.

        - With a configured `auth_token`, the hotkey is claimed with it (and a
          re-registration re-proves ownership) — that is how separately started
          processes share one identity from config.
        - Without one, the sim generates the token and it is captured here, so a
          process that registers itself is authorized for the rest of its life.
        Raises httpx.HTTPStatusError on 409/403 — the hotkey belongs to somebody
        else's token; do NOT silently continue unauthenticated.
        """
        body: dict[str, Any] = {
            "hotkey": self._hotkey,
            "coldkey": coldkey,
            "ip": ip,
            "role": role,
        }
        if alpha_stake is not None:
            body["alpha_stake"] = alpha_stake
        if self._auth_token:
            body["auth_token"] = self._auth_token
        response = self._client.post(
            self._url("/register"),
            json=body,
            timeout=self._timeout,
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        data = response.json()
        issued = data.get("auth_token")
        if issued:
            self._auth_token = str(issued)  # returned once, at the claiming call
        return int(data["uid"])

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._async_client.aclose()
        self._client.close()


@dataclass
class EmbeddedReportingChain(InMemoryChain):
    """InMemoryChain that persists every weight call + anchor to a JSONL journal.

    For single-process harness runs without the HTTP sim: the journal is the
    report-mode paper trail (one JSON object per line, append-only). Rejected
    set_weights calls are journaled too, with their outcome — the journal shows
    what was ATTEMPTED, InMemoryChain.weight_calls keeps only what was accepted.
    """

    journal_path: Path = field(default=Path("./data/chain-reports/embedded-chain.jsonl"))

    def _journal(self, entry: dict[str, Any]) -> None:
        path = Path(self.journal_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    async def set_weights(
        self,
        weights: dict[int, float],
        *,
        version_key: int,
        hotkeys: dict[int, str] | None = None,
    ) -> SetWeightsResult:
        result = await super().set_weights(
            weights, version_key=version_key, hotkeys=hotkeys
        )
        self._journal(
            {
                "kind": "set_weights",
                "block": result.block,
                "success": result.success,
                "message": result.message,
                "version_key": version_key,
                "weights": {str(uid): float(w) for uid, w in sorted(weights.items())},
            }
        )
        return result

    async def anchor_commitment(self, payload: bytes) -> str:
        txid = await super().anchor_commitment(payload)
        self._journal(
            {
                "kind": "anchor",
                "block": self.current_block(),
                "txid": txid,
                "payload_hex": payload.hex(),
            }
        )
        return txid
