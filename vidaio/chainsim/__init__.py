"""Chain simulator — the DEFAULT "report mode" chain (the project design record rule 8).

The full stack (competitions, weight-setter, inference miners + gateway) runs
end-to-end against this HTTP simulator instead of bittensor, producing reports
of scores and weight vectors instead of chain pushes. Configuration section:
`chainsim` (see config.py); services select it via the `chain` section
(vidaio.chain.factory).
"""

from vidaio.chainsim.config import ChainSimConfig
from vidaio.chainsim.report import build_report, decode_anchor_payload, render_markdown, write_report
from vidaio.chainsim.service import ChainSim

__all__ = [
    "ChainSim",
    "ChainSimConfig",
    "build_report",
    "decode_anchor_payload",
    "render_markdown",
    "write_report",
]
