"""Chain-simulator configuration — schema for the `chainsim:` section of config.

Defaults mirror the real chain where a real analogue exists (tempo 100 blocks,
like the weight-setter's InMemoryChain gate); everything else is chosen for
fast, deterministic local runs (1 s blocks, unit emission).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ChainSimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: HTTP API port (service port map: vidaio/services/protocol.py).
    port: int = 8400

    #: Metrics/health port.
    metrics_port: int = 9108

    #: SQLite state — restart-safe: blocks, uids, weights, anchors, emission all resume.
    db_path: Path = Path("./data/chainsim.db")

    #: Wall-clock seconds per block (blocks are computed lazily — no background task).
    block_seconds: float = Field(default=1.0, gt=0)

    #: Tempo gate: a validator's set_weights fails while block <= last + tempo
    #: (identical semantics to vidaio.chain.InMemoryChain).
    tempo: int = Field(default=100, ge=0)

    #: Emission minted per block, distributed proportionally to the last recorded
    #: weight vector (see vidaio/chainsim/service.py for the exact model).
    emission_per_block: float = Field(default=1.0, ge=0)

    #: Allow POST /reset (test/dev convenience). Disable for long-lived sims.
    enable_reset: bool = True

    #: Operator credential for the node-level endpoints (POST /advance, /reset,
    #: /report/write) — the sim's analogue of the node/sudo key. Leave empty and
    #: the sim generates one at first start, writes it to
    #: `<report_dir>/operator-token.txt` (owner-only) and logs it once; set it
    #: here (or via VIDAIO__CHAINSIM__OPERATOR_TOKEN) to pin your own, which also
    #: recovers access if a generated token is lost.
    operator_token: str = ""

    #: Where POST /report/write drops report-<ts>.json / report-<ts>.md.
    report_dir: Path = Path("./data/chain-reports")
