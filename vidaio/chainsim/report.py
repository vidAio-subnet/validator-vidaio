"""Run reports — the report-mode replacement for a real chain push.

`build_report` turns a full sim state (ChainSim.state() / GET /state) into a
self-contained JSON document: registered neurons with roles, the weight-vector
history per validator (blocks + per-uid deltas), the latest vector as a ranked
share table, every anchored commitment with its decoded domain-tagged payload,
and the emission credited per uid. `render_markdown` renders the same document
as a human-readable report; `write_report` persists both, timestamped.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: vidaio.audit anchors are ascii "domain:kind:sha256root" (commitments.py).
_DOMAIN_TAGGED = re.compile(
    r"^(?P<domain>[A-Za-z0-9_.-]+):(?P<kind>[A-Za-z0-9_-]+):(?P<root>[0-9a-f]{64})$"
)


def decode_anchor_payload(payload_hex: str) -> dict[str, Any] | None:
    """Best-effort decode of anchored bytes.

    Returns {"domain", "kind", "root"} for vidaio.audit domain-tagged payloads,
    {"text": ...} for other printable ascii, None for opaque bytes/bad hex.
    """
    try:
        raw = bytes.fromhex(payload_hex)
        text = raw.decode("ascii")
    except (ValueError, UnicodeDecodeError):
        return None
    m = _DOMAIN_TAGGED.match(text)
    if m is not None:
        return {"domain": m.group("domain"), "kind": m.group("kind"), "root": m.group("root")}
    return {"text": text} if text.isprintable() else None


def build_report(state: dict[str, Any]) -> dict[str, Any]:
    """The report JSON over one sim state (see module docstring for contents)."""
    neurons = state["neurons"]
    hotkey_by_uid = {n["uid"]: n["hotkey"] for n in neurons}

    # weight history per validator hotkey, each entry with per-uid deltas
    history: dict[str, list[dict[str, Any]]] = {}
    previous: dict[str, dict[int, float]] = {}
    for call in state["weight_calls"]:
        vector = {int(uid): float(w) for uid, w in call["vector"].items()}
        prev = previous.get(call["hotkey"], {})
        delta = {
            str(uid): round(vector.get(uid, 0.0) - prev.get(uid, 0.0), 12)
            for uid in sorted(set(vector) | set(prev))
            if vector.get(uid, 0.0) != prev.get(uid, 0.0)
        }
        history.setdefault(call["hotkey"], []).append(
            {
                "block": call["block"],
                "version_key": call["version_key"],
                "vector": {str(uid): vector[uid] for uid in sorted(vector)},
                "delta": delta,
            }
        )
        previous[call["hotkey"]] = vector

    # latest vector overall, ranked by weight with share %
    latest: dict[str, Any] | None = None
    if state["weight_calls"]:
        last_call = state["weight_calls"][-1]
        vector = {int(uid): float(w) for uid, w in last_call["vector"].items()}
        total = sum(w for w in vector.values() if w > 0)
        ranked = [
            {
                "rank": rank,
                "uid": uid,
                "hotkey": hotkey_by_uid.get(uid, "<unregistered>"),
                "weight": weight,
                "share_pct": round(100.0 * weight / total, 4) if total > 0 else 0.0,
            }
            for rank, (uid, weight) in enumerate(
                sorted(vector.items(), key=lambda kv: (-kv[1], kv[0])), start=1
            )
        ]
        latest = {
            "block": last_call["block"],
            "set_by": last_call["hotkey"],
            "version_key": last_call["version_key"],
            "total_weight": total,
            "ranked": ranked,
        }

    anchors = [
        {
            "txid": a["txid"],
            "block": a["block"],
            "hotkey": a.get("hotkey"),
            "payload_hex": a["payload_hex"],
            "decoded": decode_anchor_payload(a["payload_hex"]),
        }
        for a in state["anchors"]
    ]

    return {
        "kind": "vidaio.chainsim.report.v1",
        "block": state["block"],
        "sim_config": state["config"],
        "neurons": [
            {
                "uid": n["uid"],
                "hotkey": n["hotkey"],
                "role": n["role"],
                "alpha_stake": n["alpha_stake"],
                "emission_rate": n["emission"],
                "emission_credited": n["emission_credited"],
                "registered_block": n["registered_block"],
                "last_update": n["last_update"],
            }
            for n in neurons
        ],
        "weight_history": history,
        "latest_vector": latest,
        "anchors": anchors,
        "emission": state["emission"],
    }


# ---- markdown rendering ---------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Chain-sim run report",
        "",
        f"Block {report['block']} — tempo {report['sim_config']['tempo']},"
        f" emission/block {report['sim_config']['emission_per_block']}",
        "",
        "## Neurons",
        "",
        *_table(
            ["uid", "hotkey", "role", "alpha stake", "emission/block", "emission credited"],
            [
                [
                    str(n["uid"]),
                    n["hotkey"],
                    n["role"],
                    f"{n['alpha_stake']:.4f}",
                    f"{n['emission_rate']:.6f}",
                    f"{n['emission_credited']:.6f}",
                ]
                for n in report["neurons"]
            ],
        ),
        "",
        "## Latest weight vector",
        "",
    ]
    latest = report["latest_vector"]
    if latest is None:
        lines.append("No weight vector has been recorded.")
    else:
        lines += [
            f"Set by `{latest['set_by']}` at block {latest['block']}"
            f" (version_key {latest['version_key']}).",
            "",
            *_table(
                ["rank", "uid", "hotkey", "weight", "share %"],
                [
                    [
                        str(r["rank"]),
                        str(r["uid"]),
                        r["hotkey"],
                        f"{r['weight']:.6f}",
                        f"{r['share_pct']:.2f}%",
                    ]
                    for r in latest["ranked"]
                ],
            ),
        ]
    lines += ["", "## Weight history", ""]
    if not report["weight_history"]:
        lines.append("No weight calls recorded.")
    for hotkey, entries in report["weight_history"].items():
        lines += [
            f"### Validator `{hotkey}`",
            "",
            *_table(
                ["block", "vector", "delta vs previous"],
                [
                    [
                        str(e["block"]),
                        ", ".join(f"{uid}: {w:.4f}" for uid, w in e["vector"].items()),
                        ", ".join(f"{uid}: {d:+.4f}" for uid, d in e["delta"].items()) or "—",
                    ]
                    for e in entries
                ],
            ),
            "",
        ]
    lines += ["## Anchors", ""]
    if not report["anchors"]:
        lines.append("No commitments anchored.")
    else:
        rows = []
        for a in report["anchors"]:
            decoded = a["decoded"] or {}
            rows.append(
                [
                    a["txid"],
                    str(a["block"]),
                    decoded.get("domain", "—"),
                    decoded.get("kind", decoded.get("text", "—")),
                    decoded.get("root", a["payload_hex"][:32] + "…"),
                ]
            )
        lines += _table(["txid", "block", "domain", "kind", "root"], rows)
    em = report["emission"]
    lines += [
        "",
        "## Emission",
        "",
        f"Minted through block {em['credited_until_block']}: {em['minted']:.6f}"
        f" — distributed {em['distributed']:.6f}, undistributed {em['undistributed']:.6f}.",
        "",
    ]
    return "\n".join(lines)


def write_report(
    state: dict[str, Any], directory: str | Path, *, now: datetime | None = None
) -> tuple[Path, Path]:
    """Persist report-<ts>.json + report-<ts>.md under `directory`; returns the paths."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = build_report(state)
    json_path = directory / f"report-{stamp}.json"
    md_path = directory / f"report-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
