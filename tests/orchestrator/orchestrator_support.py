"""Shared fixtures/fakes for the orchestrator test suite.

Fake clock discipline matches tests/competition/support.py: all logic takes `now`
explicitly; tests drive Orchestrator.step(now) with these reference points.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from vidaio.chain.adapter import InMemoryChain
from vidaio.audit import ArtifactKind
from vidaio.competition import CompetitionManifest
from vidaio.competition import repository as repo
from vidaio.competition.interfaces import (
    BatchItem,
    BatchOutput,
    IsolationProbeReport,
    ScorePacket,
    logical_build_identity,
)
from vidaio.competition.runners.errors import (
    BatchExecutionError,
    BuildError,
    ContenderBuildError,
)
from vidaio.scoring.compression import score_compression
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.gates import ReasonCode, ValidityViolation
from vidaio.scoring.result import compose_item_score
from vidaio.services.protocol import ScorerIdentityUnavailable

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
START = T0 + timedelta(hours=1)
ENROLL_DEADLINE = T0 + timedelta(hours=2)
FINALIZATION = T0 + timedelta(hours=3)
END = T0 + timedelta(hours=48)
M = timedelta(minutes=1)

DOCKER = shutil.which("docker") or "/usr/local/bin/docker"

COMMITMENT_ROOT = "c0" * 32
SEED_COMMITMENT = "a" * 64
VMAF_FAKE = 92.0
VMAF_THRESHOLD_ITEM = 85.0

BASELINE_URL = "local://reference-baseline"
_BASELINE_TREE_SHA = "0b" * 20
BASELINE = {
    "version": 0,
    "artifact_digest": "1" * 64,
    "artifact_bytes": 1024,
    "image_digest": logical_build_identity(
        repo_url=BASELINE_URL,
        commit_sha="b" * 40,
        tree_sha=_BASELINE_TREE_SHA,
    ),
    "provenance_digest": "3" * 64,
    "provenance_bytes": 512,
    "repo_url": BASELINE_URL,
    "commit_sha": "b" * 40,
    "tree_sha": _BASELINE_TREE_SHA,
}


def materialize_baseline(orch, repos: dict[str, Path]) -> dict[str, Any]:
    """Return a manifest baseline backed by the exact archived fixture tree.

    Schema v14 treats the active baseline executable and its provenance as
    content-addressed economic evidence.  The old all-``1``/all-``3`` fixture
    digests described no bytes and therefore could exercise orchestration but
    could never produce an auditable result.  Build the same deterministic source
    archive that finalization will record, persist both required objects, and bind
    the manifest to their real identities.
    """

    from vidaio.competition.runners import safeio

    archive = safeio.deterministic_tarball(
        repos[BASELINE_URL], max_bytes=orch.cfg.submission_backup_max_bytes
    )
    artifact = orch.store.put(archive, ArtifactKind.SUBMISSION_ARCHIVE)
    provenance_bytes = json.dumps(
        {
            "artifact_digest": artifact.digest,
            "commit_sha": BASELINE["commit_sha"],
            "repo_url": BASELINE["repo_url"],
            "schema": "vidaio-baseline-provenance/1",
            "tree_sha": BASELINE["tree_sha"],
            "version": BASELINE["version"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    provenance = orch.store.put(provenance_bytes, ArtifactKind.MANIFEST)
    return {
        **BASELINE,
        "artifact_digest": artifact.digest,
        "artifact_bytes": artifact.byte_size,
        "provenance_digest": provenance.digest,
        "provenance_bytes": provenance.byte_size,
    }


#: Distinct pinned identities per fixture contender (fake tree shas).
CONTENDER_SHAS = {
    "hk-a": ("1a" * 20, "2a" * 20),
    "hk-b": ("1b" * 20, "2b" * 20),
    "hk-mal": ("1c" * 20, "2c" * 20),
    # untrusted fixtures for the review service-review regression tests
    "hk-sym": ("1d" * 20, "2d" * 20),  # emits a symlink as its output (#3)
    "hk-flood": ("1e" * 20, "2e" * 20),  # fills /output (#13)
    "hk-exit": ("1f" * 20, "2f" * 20),  # exits 1 (#14)
    "hk-silent": ("11" * 20, "22" * 20),  # exits 0, writes nothing (#14)
    "hk-lie": ("13" * 20, "23" * 20),  # fake wget that always "fails" (#12)
    "hk-link": ("14" * 20, "24" * 20),  # repo tree containing a symlink (#3)
    "hk-logflood": ("15" * 20, "25" * 20),  # floods stdout then exits fast (#13 r2)
}


def repo_url(hotkey: str) -> str:
    return f"local://{hotkey}"


def build_manifest(
    competition_id: str = "comp-orch",
    *,
    baseline: dict[str, Any] | None = None,
    **overrides: Any,
) -> CompetitionManifest:
    data: dict[str, Any] = {
        "competition_id": competition_id,
        "track": "compression",
        "start_time": START,
        "enrollment_deadline": ENROLL_DEADLINE,
        "finalization_time": FINALIZATION,
        "end_time": END,
        "minimum_alpha_stake": 500.0,
        "scoring_factors": {
            "quality": 0.6,
            "cost_efficiency": 0.0,
            "length_coverage": 0.4,
        },
        "vmaf_threshold": 90.0,
        "sealed_vmaf_variants": [85.0, 89.0, 93.0],
        "allowed_gpus": ["L4"],
        "evaluation_batch_size": {"min": 1, "max": 2},
        "scoring_seed_commitment": SEED_COMMITMENT,
        "container_size_limit_gb": 25.0,
        "scoring_version": "v1.0.0",
        "baseline": baseline,
    }
    data.update(overrides)
    return CompetitionManifest.model_validate(data)


def enroll(orch, competition_id: str, hotkey: str, now: datetime | None = None) -> int:
    commit_sha, tree_sha = CONTENDER_SHAS[hotkey]
    return orch.enroll_contender(
        competition_id,
        hotkey=hotkey,
        repo_url=repo_url(hotkey),
        commit_sha=commit_sha,
        tree_sha=tree_sha,
        stake=1000.0,
        now=now or (START + timedelta(minutes=5)),
    )


def seed_items(orch, competition_id: str, tmp_dir: Path, n: int = 3) -> list[int]:
    """Seed n sealed inputs (4 KiB each, deterministic bytes) via the orchestrator."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    item_ids = []
    now = FINALIZATION + timedelta(minutes=1)
    for i in range(n):
        src = tmp_dir / f"input-{i}.bin"
        src.write_bytes(bytes([i % 251]) * 4096)
        item_ids.append(
            orch.add_evaluation_item(
                competition_id,
                input_path=src,
                item_index=i,
                threshold_commitment=hashlib.sha256(
                    f"threshold:{i}".encode()
                ).hexdigest(),
                length_seconds=10.0,
                now=now,
            )
        )
    return item_ids


def phase(orch, competition_id: str):
    comp = repo.get_competition(orch.conn, competition_id)
    assert comp is not None
    return comp.status


def events_of(orch, competition_id: str, event_type: str) -> list[sqlite3.Row]:
    return [
        e
        for e in repo.list_events(orch.conn, competition_id)
        if e["event_type"] == event_type
    ]


# ---- shared wiring helpers -----------------------------------------------------


def client_conn(db_path: Path) -> sqlite3.Connection:
    """A read connection for scoring clients: the orchestrator invokes score_item
    from asyncio.to_thread worker threads, so the client's conn must allow
    cross-thread use (calls are serialized — one scoring call at a time)."""
    conn = sqlite3.connect(
        str(db_path), timeout=30, isolation_level=None, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def make_raw_config(tmp_path: Path, **orchestrator_overrides: Any) -> dict[str, Any]:
    orchestrator = {
        "work_dir": str(tmp_path / "work"),
        "tick_seconds": 0.1,
        "retry_base_delay_seconds": 0.01,
        "retry_max_delay_seconds": 0.02,
        "anchor_receipt_timeout_seconds": 0.05,
        "anchor_receipt_poll_seconds": 0.001,
        "build_retry_attempts": 2,
        "batch_retry_attempts": 2,
        "scoring_retry_attempts": 3,
        "max_batch_requeues": 3,
        "metrics_port": 0,
    }
    orchestrator.update(orchestrator_overrides)
    return {
        "core": {"data_dir": str(tmp_path / "data"), "db_filename": "orch.db"},
        "competition": {},  # require_audit_linkage stays True (production default)
        "audit": {
            "local_root": str(tmp_path / "audit"),
            "allow_plaintext_holdout": True,  # test env: no Envelope
        },
        "orchestrator": orchestrator,
    }


# ---- fixture solution repos (see the /app/run.sh contract) ---------------------

DOCKERFILE = "FROM alpine:3.20\nCOPY run.sh /app/run.sh\n"

HONEST_RUN_SH = """\
#!/bin/sh
set -e
in="$1"; out="$2"
for f in "$in"/*; do
  [ -f "$f" ] || continue
  head -c {head} "$f" > "$out/$(basename "$f")"
done
"""

UNTRUSTED_RUN_SH = """\
#!/bin/sh
in="$1"; out="$2"
for f in "$in"/*; do
  [ -f "$f" ] || continue
  if wget -q -T 2 -t 1 -O - http://example.com/ > "$out/$(basename "$f")" 2>/dev/null; then
    :
  else
    printf 'NO-NETWORK' > "$out/$(basename "$f")"
  fi
done
( echo hack > "$in/hack" ) 2>/dev/null || true
( echo hack > /evaluation-inputs/hack ) 2>/dev/null || true
( echo hack > /hack ) 2>/dev/null || true
exit 0
"""

#: Untrusted: emits the expected output NAME as a symlink to a host file. Following
#: it would make the validator hash and archive its own /etc/passwd.
SYMLINK_RUN_SH = """\
#!/bin/sh
in="$1"; out="$2"
for f in "$in"/*; do
  [ -f "$f" ] || continue
  ln -s /etc/passwd "$out/$(basename "$f")"
done
exit 0
"""

#: Untrusted: fills /output until the host watchdog kills it.
FLOOD_RUN_SH = """\
#!/bin/sh
in="$1"; out="$2"
for f in "$in"/*; do
  [ -f "$f" ] || continue
  dd if=/dev/zero of="$out/$(basename "$f")" bs=65536 count=4096 2>/dev/null
done
exit 0
"""

#: Untrusted: floods STDOUT and exits immediately. The old
#: watchdog only measured logs while `poll()` was None and only re-checked /output
#: after exit, so a writer that finished between two polls escaped the cap
#: entirely. ~1 MiB against a 64 KiB cap, produced in well under one poll interval.
LOG_FLOOD_RUN_SH = """\
#!/bin/sh
i=0
while [ $i -lt 16 ]; do
  dd if=/dev/zero bs=65536 count=1 2>/dev/null | tr '\\000' 'x'
  i=$((i+1))
done
exit 0
"""

#: Untrusted: the plain `exit 1` that used to halt the entire competition (#14).
EXIT_ONE_RUN_SH = """\
#!/bin/sh
echo "this solution refuses to work" >&2
exit 1
"""

#: Untrusted: exits 0 and produces NOTHING. The empty file used to be handed to
#: real ffmpeg, whose 502 halted the competition (#14).
NO_OUTPUT_RUN_SH = """\
#!/bin/sh
exit 0
"""

#: Untrusted: a FAKE wget that always "fails", so the in-container probe reports
#: NETWORK_ATTEMPT=0 ("blocked") no matter what the sandbox really allows. Only a
#: host-observed verdict can catch a runner regression against this image (#12).
LYING_PROBE_DOCKERFILE = (
    "FROM alpine:3.20\n"
    "COPY run.sh /app/run.sh\n"
    "RUN rm -f /usr/bin/wget /bin/wget \\\n"
    "    && printf '#!/bin/sh\\nexit 1\\n' > /usr/bin/wget \\\n"
    "    && chmod +x /usr/bin/wget\n"
)


def write_solution(path: Path, run_sh: str, dockerfile: str = DOCKERFILE) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "Dockerfile").write_text(dockerfile)
    (path / "run.sh").write_text(run_sh)
    return path


# ---- repo providers ------------------------------------------------------------


class FlakyRepoProvider:
    """LocalRepoProvider whose checkout fails for chosen repo_urls.

    Models the an internal review scenario: a TRANSIENT checkout failure during
    FINALIZING_SUBMISSIONS. `fail_times` counts down per repo_url, so a provider
    can be made to fail once and then behave — which is precisely how a contender
    used to end up eligible with no archived submission.
    """

    def __init__(
        self,
        mapping: dict[str, Path],
        *,
        fail_for: dict[str, int] | None = None,
        error: type[Exception] | None = None,
    ) -> None:
        from vidaio.competition.runners import LocalRepoProvider
        from vidaio.competition.runners.errors import CheckoutError

        self._inner = LocalRepoProvider(mapping)
        #: repo_url -> remaining failures
        self.fail_for = dict(fail_for or {})
        self.error = error or CheckoutError
        self.calls: list[str] = []
        self.releases: list[Path] = []

    def checkout(self, repo_url: str, commit_sha: str) -> Path:
        self.calls.append(repo_url)
        remaining = self.fail_for.get(repo_url, 0)
        if remaining > 0:
            self.fail_for[repo_url] = remaining - 1
            raise self.error(f"transient checkout failure for {repo_url}")
        return self._inner.checkout(repo_url, commit_sha)

    def release(self, checkout: str | Path) -> None:
        self.releases.append(Path(checkout))
        self._inner.release(checkout)


# ---- lifecycle drivers ---------------------------------------------------------


async def start_and_enroll(orch, manifest, hotkeys):
    from vidaio.competition.states import Phase

    cid = manifest.competition_id
    orch.create_competition(manifest, T0)
    orch.anchor_commitment(cid, COMMITMENT_ROOT, T0)
    await orch.step(START)
    assert phase(orch, cid) is Phase.ENROLLING
    for hk in hotkeys:
        enroll(orch, cid, hk)
    return cid


async def drive_to_completion(orch, cid, tmp_path, *, seed=True):
    if seed:
        seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)  # ENROLLING -> FINALIZING -> (backup) VALIDATING
    await orch.step(FINALIZATION + 2 * M)  # VALIDATING -> BUILDING
    await orch.step(FINALIZATION + 3 * M)  # BUILDING -> EVALUATING
    await orch.step(FINALIZATION + 10 * M)  # EVALUATING -> SCORING
    await orch.step(FINALIZATION + 15 * M)  # SCORING -> AWAITING_END_TIME
    await orch.step(END + M)  # review window over, end_time reached


# ---- fakes ---------------------------------------------------------------------


class FakeRunner:
    """All-fake SandboxRunner honoring the content-addressed output-pool contract.
    Each image's fake output size differs by contender so compression scores (and
    the ranking) are distinct and deterministic."""

    def __init__(self, outputs_dir: Path) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.build_calls: list[int] = []
        self.batch_calls: list[tuple[str, int]] = []
        self.probe_calls: list[str] = []
        #: repo_urls whose OWN Dockerfile fails to build (CONTENDER fault)
        self.fail_build_for: set[str] = set()
        #: repo_urls whose build fails for an INFRA reason (our docker is broken)
        self.infra_fail_build_for: set[str] = set()
        self.fail_probe_for: set[str] = set()  # tree_shas whose probe must fail
        #: tree_shas whose probe cannot be RUN at all (INFRA — nothing attested)
        self.probe_unavailable_for: set[str] = set()
        self.batch_fail_times = 0  # fail this many run_batch calls, then succeed

    @staticmethod
    def digest_for(tree_sha: str) -> str:
        if tree_sha == BASELINE["tree_sha"]:
            return str(BASELINE["image_digest"])
        return hashlib.sha256(f"img:{tree_sha}".encode()).hexdigest()

    def build(self, contender) -> str:
        self.build_calls.append(contender.contender_id)
        if contender.repo_url in self.infra_fail_build_for:
            raise BuildError(
                f"docker daemon unusable while building {contender.repo_url}"
            )
        if contender.repo_url in self.fail_build_for:
            raise ContenderBuildError(f"forced build failure for {contender.repo_url}")
        return self.digest_for(contender.tree_sha)

    def isolation_probe(self, image_digest: str) -> IsolationProbeReport:
        from vidaio.competition.runners.errors import SandboxProbeUnavailableError

        self.probe_calls.append(image_digest)
        if any(
            self.digest_for(tree) == image_digest for tree in self.probe_unavailable_for
        ):
            raise SandboxProbeUnavailableError(
                "host inspection of the probe container failed"
            )
        failed = any(
            self.digest_for(tree) == image_digest for tree in self.fail_probe_for
        )
        return IsolationProbeReport(
            network_blocked=not failed,
            secrets_absent=True,
            reference_mounts_absent=True,
            index_leak_absent=True,
            details="fake probe (forced failure)" if failed else "fake probe",
        )

    def run_batch(
        self, image_digest: str, items: Sequence[BatchItem], batch_index: int
    ) -> Sequence[BatchOutput]:
        self.batch_calls.append((image_digest, batch_index))
        if self.batch_fail_times > 0:
            self.batch_fail_times -= 1
            raise BatchExecutionError("forced infra failure")
        outputs = []
        for item in items:
            data = f"out:{image_digest[:8]}:{item.input_sha256}".encode()
            digest = hashlib.sha256(data).hexdigest()
            (self.outputs_dir / digest).write_bytes(data)
            outputs.append(
                BatchOutput(
                    item_id=item.item_id,
                    output_sha256=digest,
                    output_bytes=len(data),
                    wall_seconds=0.01,
                )
            )
        return outputs

    def available(self) -> bool:
        return True


class ContenderFaultRunner(FakeRunner):
    """FakeRunner whose run_batch fails, or produces nothing, for chosen images.

    Models the two an internal review scenarios without needing docker: a solution that
    exits non-zero, and a solution that exits 0 having written no output.
    """

    def __init__(self, outputs_dir: Path) -> None:
        super().__init__(outputs_dir)
        self.fault_for: dict[str, Exception] = {}  # tree_sha -> exception to raise
        self.silent_for: set[str] = set()  # tree_shas that produce zero outputs

    def run_batch(self, image_digest, items, batch_index):
        for tree_sha, exc in self.fault_for.items():
            if self.digest_for(tree_sha) == image_digest:
                self.batch_calls.append((image_digest, batch_index))
                raise exc
        if any(self.digest_for(t) == image_digest for t in self.silent_for):
            self.batch_calls.append((image_digest, batch_index))
            return []
        return super().run_batch(image_digest, items, batch_index)


class RecordingChain(InMemoryChain):
    """InMemoryChain that also records every anchor call it was handed.

    The control-API test asserts the payload reached THIS adapter — proving the
    orchestrator anchors through the injected ChainAdapter rather than only
    writing SQLite.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.anchor_calls: list[bytes] = []

    async def anchor_commitment(self, payload: bytes) -> str:
        self.anchor_calls.append(bytes(payload))
        return await super().anchor_commitment(payload)


class SimulatedCrash(BaseException):
    """Deliberately NOT an Exception: passes through every retry/except-Exception
    layer, modeling a hard process death mid-stage."""


class CrashingRunner(FakeRunner):
    """Crashes the whole process on the Nth build call (1-based)."""

    def __init__(self, outputs_dir: Path, crash_on_build_call: int) -> None:
        super().__init__(outputs_dir)
        self.crash_on_build_call = crash_on_build_call

    def build(self, contender) -> str:
        if len(self.build_calls) + 1 >= self.crash_on_build_call:
            raise SimulatedCrash("process died mid-BUILDING")
        return super().build(contender)


class FakeScoringClient:
    """CompetitionScoringClient producing REAL compose_item_score packets from the
    actual input/output byte sizes (compression track). An absent output (the
    canonical empty digest / 0 bytes) yields a gate-failed zero packet — never a
    substituted positive score. `conn` is assigned after the orchestrator exists."""

    #: What this "worker" advertises on GET /healthz — matched by the fixture
    #: manifest's `scoring_version`, because a manifest commits to the scorer
    #: identity and the orchestrator halts when the two disagree
    #: (vidaio/services/protocol.py: THE SCORER-IDENTITY CONTRACT).
    IDENTITY = "v1.0.0"

    def __init__(self, *, backend_versions: dict[str, str] | None = None) -> None:
        self.conn: sqlite3.Connection | None = None
        self.calls: list[tuple[int, int]] = []
        self.identity = self.IDENTITY
        self.backend_versions = dict(backend_versions or {})
        #: True = GET /healthz unreachable; the check defers (it cannot PROVE a
        #: disagreement) and the scoring stage's own retry/halt path owns it.
        self.identity_unavailable = False
        self.identity_calls = 0

    def scorer_identity(self) -> str:
        self.identity_calls += 1
        if self.identity_unavailable:
            raise ScorerIdentityUnavailable("scoring worker /healthz unreachable")
        return self.identity

    def score_item(
        self,
        competition_id: str,
        contender_id: int,
        item: BatchItem,
        output: BatchOutput,
    ) -> ScorePacket:
        assert self.conn is not None, "assign .conn after constructing the orchestrator"
        self.calls.append((contender_id, item.item_id))
        item_row = self.conn.execute(
            "SELECT * FROM evaluation_items WHERE item_id = ?", (item.item_id,)
        ).fetchone()
        contender = repo.get_contender(self.conn, contender_id)
        assert item_row is not None and contender is not None
        config = ScoringConfig()
        if output.output_bytes == 0:
            gate_passed = False
            violations = [
                ValidityViolation(
                    code=ReasonCode.METRIC_MISSING, detail="no output produced"
                )
            ]
            breakdown = None
            rate = None
        else:
            breakdown = score_compression(
                candidate_bytes=output.output_bytes,
                reference_bytes=item.input_bytes,
                vmaf=VMAF_FAKE,
                config=config,
                vmaf_threshold=VMAF_THRESHOLD_ITEM,
            )
            gate_passed = True
            violations = []
            rate = breakdown.compression_rate
        metrics: dict[str, Any] = {
            "cost": 1.0,
            "length_seconds": item_row["length_seconds"],
        }
        if rate is not None:
            metrics.update({"vmaf": VMAF_FAKE, "compression_rate": rate})
        packet = compose_item_score(
            item_id=item_row["scoring_item_id"],
            challenge_id=item_row["challenge_id"],
            track="compression",
            gate_passed=gate_passed,
            violations=violations,
            breakdown=breakdown,
            config=config,
            miner_hotkey=contender.hotkey,
            content_digest=output.output_sha256,
            metrics=metrics,
            backend_versions=self.backend_versions,
            # Model the worker contract faithfully: after a test swaps/restores
            # the advertised worker identity, measured packets carry that same
            # current identity rather than the fixture's original default.
            scorer_version=self.identity,
        )
        return ScorePacket(
            item_id=item.item_id,
            contender_id=contender_id,
            packet_bytes=packet.to_json().encode("utf-8"),
        )


class FlakyScoringClient(FakeScoringClient):
    """Fails the first `fail_times` calls (transport-style), then behaves."""

    def __init__(self, fail_times: int) -> None:
        super().__init__()
        self.fail_times = fail_times

    def score_item(self, competition_id, contender_id, item, output) -> ScorePacket:
        if self.fail_times > 0:
            self.fail_times -= 1
            self.calls.append((contender_id, item.item_id))
            raise ConnectionError("forced scoring transport failure")
        return super().score_item(competition_id, contender_id, item, output)


class AlwaysFailScoringClient(FakeScoringClient):
    def score_item(self, competition_id, contender_id, item, output) -> ScorePacket:
        self.calls.append((contender_id, item.item_id))
        raise ConnectionError("scoring worker permanently unreachable")


class CrashingScoringClient(FakeScoringClient):
    """Hard process death on the Nth score call (1-based)."""

    def __init__(self, crash_on_call: int) -> None:
        super().__init__()
        self.crash_on_call = crash_on_call

    def score_item(self, competition_id, contender_id, item, output) -> ScorePacket:
        if len(self.calls) + 1 >= self.crash_on_call:
            raise SimulatedCrash("process died mid-SCORING")
        return super().score_item(competition_id, contender_id, item, output)
