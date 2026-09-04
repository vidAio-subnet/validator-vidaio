from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from vidaio.competition import CompetitionManifest, LifecycleEngine
from vidaio.competition import repository as repo

# Fake clock reference points — all logic takes `now` explicitly.
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
START = T0 + timedelta(hours=1)
ENROLL_DEADLINE = T0 + timedelta(hours=2)
FINALIZATION = T0 + timedelta(hours=3)
# end_time sits past the default 24h human-review window opened at SCORES_AT, so the
# AWAITING_END_TIME -> COMPLETED guard (max of end_time and review deadline) is
# governed by end_time in the default walk.
END = T0 + timedelta(hours=48)
SCORES_AT = FINALIZATION + timedelta(hours=1)

SEED_COMMITMENT = "a" * 64
COMMITMENT_ROOT = "c0" * 32
BACKUP_REF = "audit://backups/comp/submissions/" + "b" * 16
BASELINE = {
    "version": 0,
    "artifact_digest": "1" * 64,
    "artifact_bytes": 1024,
    "image_digest": "2" * 64,
    "provenance_digest": "3" * 64,
    "provenance_bytes": 512,
    "repo_url": "https://github.com/vidaio/reference-baseline",
    "commit_sha": "b" * 40,
    "tree_sha": "c" * 40,
}


def build_manifest(
    competition_id: str = "comp-01", *, baseline: dict[str, Any] | None = None, **overrides: Any
) -> CompetitionManifest:
    data: dict[str, Any] = {
        "competition_id": competition_id,
        "track": "compression",
        "start_time": START,
        "enrollment_deadline": ENROLL_DEADLINE,
        "finalization_time": FINALIZATION,
        "end_time": END,
        "minimum_alpha_stake": 500.0,
        "scoring_factors": {"quality": 0.6, "cost_efficiency": 0.0, "length_coverage": 0.4},
        "vmaf_threshold": 90.0,
        "sealed_vmaf_variants": [85.0, 89.0, 93.0],
        "allowed_gpus": ["L4", "L40S", "RTX6000"],
        "evaluation_batch_size": {"min": 1, "max": 5},
        "scoring_seed_commitment": SEED_COMMITMENT,
        "container_size_limit_gb": 25.0,
        "scoring_version": "v1.0.0",
        "baseline": baseline,
    }
    data.update(overrides)
    return CompetitionManifest.model_validate(data)


def packet_bytes(
    *,
    challenge_id: str,
    scoring_item_id: str,
    miner_hotkey: str | None,
    score: float,
    gate_passed: bool = True,
    vmaf: float | None = 92.0,
    compression_rate: float | None = 0.5,
    cost: float | None = 1.0,
    length_seconds: float | None = 10.0,
) -> bytes:
    """The scoring module's ItemScore JSON, as record_item_score expects it."""
    metrics: dict[str, float] = {}
    for key, value in (
        ("vmaf", vmaf),
        ("compression_rate", compression_rate),
        ("cost", cost),
        ("length_seconds", length_seconds),
    ):
        if value is not None:
            metrics[key] = value
    return json.dumps(
        {
            "item_id": scoring_item_id,
            "challenge_id": challenge_id,
            "track": "compression",
            "miner_hotkey": miner_hotkey,
            "score": score,
            "gate_passed": gate_passed,
            "violations": [],
            "metrics": metrics,
            "scorer_version": "v1.0.0",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class Driver:
    """Walks a competition through its lifecycle with an explicit fake clock."""

    def __init__(self, conn: sqlite3.Connection, engine: LifecycleEngine) -> None:
        self.conn = conn
        self.engine = engine

    def phase(self, competition_id: str):
        comp = repo.get_competition(self.conn, competition_id)
        assert comp is not None
        return comp.status

    def events(self, competition_id: str) -> list[sqlite3.Row]:
        return repo.list_events(self.conn, competition_id)

    def anchor(self, competition_id: str, now: datetime = T0) -> bool:
        """Anchor the pre-commitment (required before enrollment can open)."""
        return self.engine.mark_commitment_anchored(
            self.conn, competition_id, COMMITMENT_ROOT, now
        )

    def enroll(
        self,
        competition_id: str,
        hotkey: str,
        stake: float = 1000.0,
        now: datetime = START + timedelta(minutes=5),
    ) -> int:
        return repo.enroll_contender(
            self.conn,
            competition_id,
            hotkey=hotkey,
            repo_url=f"https://github.com/miners/{hotkey}",
            commit_sha="d" * 40,
            tree_sha="e" * 40,
            stake=stake,
            now=now,
        )

    def accept_all(self, competition_id: str, now: datetime = FINALIZATION) -> None:
        for c in repo.list_contenders(self.conn, competition_id):
            if c.status == "ENROLLED":
                repo.set_contender_status(self.conn, c.contender_id, "ACCEPTED", now)

    def seed_items(
        self, competition_id: str, n: int = 2, length_seconds: float = 10.0
    ) -> list[int]:
        now = FINALIZATION + timedelta(minutes=5)
        return [
            repo.add_evaluation_item(
                self.conn,
                competition_id,
                item_index=i,
                input_sha256=hashlib.sha256(
                    f"{competition_id}:{i}".encode("utf-8")
                ).hexdigest(),
                input_bytes=1_000_000,
                threshold_commitment="f" * 64,
                challenge_id=f"chal-{competition_id}",
                length_seconds=length_seconds,
                now=now,
            )
            for i in range(n)
        ]

    def item_identity(self, item_id: int) -> tuple[str, str]:
        row = self.conn.execute(
            "SELECT challenge_id, scoring_item_id FROM evaluation_items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        assert row is not None
        return row["challenge_id"], row["scoring_item_id"]

    def make_packet(
        self,
        contender_id: int,
        item_id: int,
        score: float,
        *,
        gate_passed: bool = True,
        cost: float | None = 1.0,
        vmaf: float | None = 92.0,
        length_seconds: float | None = 10.0,
    ) -> bytes:
        contender = repo.get_contender(self.conn, contender_id)
        assert contender is not None
        challenge_id, scoring_item_id = self.item_identity(item_id)
        return packet_bytes(
            challenge_id=challenge_id,
            scoring_item_id=scoring_item_id,
            miner_hotkey=contender.hotkey,
            score=score,
            gate_passed=gate_passed,
            vmaf=vmaf,
            cost=cost,
            length_seconds=length_seconds,
        )

    def score_contender(
        self,
        competition_id: str,
        contender_id: int,
        item_ids: list[int],
        item_score: float,
        cost: float = 1.0,
        vmaf: float = 92.0,
        gate_passed: bool = True,
        length_seconds: float | None = 10.0,
    ) -> None:
        now = FINALIZATION + timedelta(minutes=30)
        for item_id in item_ids:
            repo.record_item_score(
                self.conn,
                competition_id,
                contender_id=contender_id,
                item_id=item_id,
                packet_bytes=self.make_packet(
                    contender_id,
                    item_id,
                    item_score,
                    gate_passed=gate_passed,
                    cost=cost,
                    vmaf=vmaf,
                    length_seconds=length_seconds,
                ),
                now=now,
            )

    def run_to_evaluating(
        self, manifest: CompetitionManifest, hotkeys: list[str]
    ) -> tuple[str, dict[str, int], list[int]]:
        """SCHEDULED -> ... -> EVALUATING with accepted+built contenders and seeded
        evaluation items. Returns (competition_id, hotkey -> contender_id, item_ids)."""
        cid = manifest.competition_id
        self.engine.create_competition(self.conn, manifest, T0)
        self.anchor(cid)
        self.engine.tick(self.conn, START)
        ids = {hk: self.enroll(cid, hk) for hk in hotkeys}
        self.engine.tick(self.conn, FINALIZATION)
        self.accept_all(cid)
        t = FINALIZATION + timedelta(minutes=1)
        self.engine.mark_submissions_backed_up(self.conn, cid, BACKUP_REF, t)
        self.engine.mark_validation_complete(self.conn, cid, t + timedelta(minutes=1))
        contenders = repo.list_contenders(self.conn, cid)
        for c in contenders:
            repo.set_contender_image_digest(self.conn, c.contender_id, "1" * 64, t)
        self.engine.mark_builds_complete(
            self.conn, cid, len(contenders), t + timedelta(minutes=2)
        )
        item_ids = self.seed_items(cid)
        return cid, ids, item_ids

    def link_audit_bundles(self, competition_id: str) -> int:
        """Link every still-unlinked performance row (baseline calibration rows included)
        to a deterministic fake audit-bundle digest via the real repository API —
        the audit runner's job, minus the audit store. Returns the rows linked."""
        rows = self.conn.execute(
            "SELECT performance_id FROM performance_history"
            " WHERE competition_id = ? AND audit_bundle_digest IS NULL",
            (competition_id,),
        ).fetchall()
        for row in rows:
            digest = hashlib.sha256(
                f"fake-audit-bundle:{competition_id}:{row['performance_id']}".encode()
            ).hexdigest()
            repo.set_audit_bundle_digest(self.conn, row["performance_id"], digest)
        return len(rows)

    def run_to_awaiting(
        self,
        manifest: CompetitionManifest,
        contender_scores: dict[str, float],
        *,
        baseline_score: float | None = None,
        costs: dict[str, float] | None = None,
        link_audit: bool = True,
    ) -> tuple[str, dict[str, int]]:
        """SCHEDULED -> ... -> AWAITING_END_TIME with persisted per-item scores.

        contender_scores: hotkey -> per-item score; baseline_score scores the calibration
        contender when the manifest declares a baseline. link_audit=True (default) also
        links every score row to a fake audit bundle so the audit-linkage completion
        gate (CompetitionConfig.require_audit_linkage) lets the competition COMPLETE;
        pass False to leave linkage gaps open.
        """
        cid, ids, item_ids = self.run_to_evaluating(manifest, list(contender_scores))
        t = FINALIZATION + timedelta(minutes=1)
        self.engine.mark_evaluation_complete(self.conn, cid, t + timedelta(minutes=3))
        for hk, score in contender_scores.items():
            self.score_contender(
                cid, ids[hk], item_ids, score, cost=(costs or {}).get(hk, 1.0)
            )
        if baseline_score is not None:
            baseline_row = next(c for c in repo.list_contenders(self.conn, cid) if c.is_calibration)
            self.score_contender(cid, baseline_row.contender_id, item_ids, baseline_score)
        self.engine.mark_scores_persisted(self.conn, cid, SCORES_AT)
        if link_audit:
            self.link_audit_bundles(cid)
        return cid, ids
