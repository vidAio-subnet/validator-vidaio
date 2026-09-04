"""Builders for the Scoring Authority suite — real store, real chain fake, real index.

A `ScoringAuthority` wired with a `LocalFsStore` (the shared object store), an
`InMemoryChain` (records anchors), and a fresh `EpochIndex` — no boto3, no
bittensor. `seed_epoch` runs the real `finalize_and_anchor` so every test reads
pointers that only the real producer could have written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from vidaio.audit.canonical import sha256_hex
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.authority import EpochIndex, ScoringAuthority, ScoredItem, build_audit_manifest
from vidaio.authority.finalizer import EpochFinalizer
from vidaio.chain.adapter import InMemoryChain
from vidaio.epoch import AuditManifest
from vidaio.tokenomics import MinerSnapshot, TokenomicsConfig
from vidaio.tokenomics.ewma import accumulate


NOW = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
SCORER = "scoring-1.0.0+abc123def456"
DECAY = TokenomicsConfig().ewma_decay

#: The single-cycle per-uid score both builders share so a seeded epoch's earning inputs
#: fold EXACTLY to the miner's accumulate_score (the finalizer now requires this, #1). A
#: caller-passed ``make_miner`` score is treated as this per-cycle score.
_CYCLE_SCORE = 0.8


def make_miner(uid: int, score: float = _CYCLE_SCORE, track: str = "compression") -> MinerSnapshot:
    # accumulate_score is the genesis fold of ONE cycle at ``score`` — so a matching
    # ``make_item(uid)`` earning input re-derives it and the finalizer accepts the log.
    return MinerSnapshot(
        uid=uid,
        hotkey=f"hk{uid}",
        coldkey=f"ck{uid}",
        ip=f"10.0.0.{uid}",
        track=track,
        accumulate_score=accumulate(0.0, _CYCLE_SCORE, DECAY),
    )


def make_item(uid: int, store: LocalFsStore) -> ScoredItem:
    packet = f"packet-bytes-{uid}".encode()
    store.put(packet, ArtifactKind.SCORE_PACKET)  # a REAL stored audit file
    # A REAL stored, resolvable bundle object (#8): the finalizer probes it exists.
    bundle_ref = store.put(f"bundle-bytes-{uid}".encode(), ArtifactKind.AUDIT_BUNDLE)
    return ScoredItem(
        uid=uid,
        hotkey=f"hk{uid}",
        challenge_id="c1",
        item_id=f"i{uid}",
        bundle_digest=bundle_ref.digest,
        packet_digest=sha256_hex(packet),
        committed_track="compression",  # REQUIRED (#9)
        score=_CYCLE_SCORE,
        cycle_sequence=0,
    )


class Authority:
    """The service under test + the concrete backends it was wired with."""

    def __init__(self, tmp_path: Path, *, api_token: str | None = None, burn_uid: int = 0) -> None:
        self.store = LocalFsStore(tmp_path / "audit")
        self.chain = InMemoryChain()
        self.index = EpochIndex.open(tmp_path / "authority.db")
        raw_config = {
            "core": {"metrics_port": 0},
            "authority": {
                "http_host": "127.0.0.1",
                "http_port": 0,
                "metrics_port": 0,
                "api_token": api_token,
                "netuid": 85,
                "burn_uid": burn_uid,
                "scorer_version": SCORER,
            },
        }
        self.service = ScoringAuthority(
            raw_config,
            metrics_port=0,
            store=self.store,
            chain=self.chain,
            index=self.index,
            finalizer=EpochFinalizer(TokenomicsConfig(), scorer_version=SCORER),
            now=lambda: NOW,
        )

    async def seed_epoch(
        self,
        *,
        epoch_id: int,
        close_block: int,
        miners: list[MinerSnapshot],
        items: list[ScoredItem] | None = None,
    ):
        manifest = (
            build_audit_manifest(
                items,
                store=self.store,
                # an internal review: committed window evidence per nonzero uid, consistent
                # with make_miner's windowed fields + the close_block (the finalizer requires it).
            )
            if items is not None
            else AuditManifest()
        )
        return await self.service.finalize_and_anchor(
            epoch_id=epoch_id,
            close_block=close_block,
            snapshots=miners,
            audit_manifest=manifest,
        )

    def close(self) -> None:
        self.index.close()
