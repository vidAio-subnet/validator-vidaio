"""Parallel epoch auditing preserves ordered verdicts and drains its workers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, Lock, current_thread
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tests.auditor.fakes import HONEST_METRICS, HONEST_SCORE, NOW, SCORER, make_fake_bundle
from tests.auditor.test_service import _auditor, _epoch_with_items
from vidaio.audit.recompute import RECOMPUTE_ERROR, RecomputedScore, StaticRecomputer
from vidaio.audit.store import ArtifactKind, LocalFsStore
from vidaio.auditor import AuditorConfig, AuditStatus, ItemVerdictKind, SamplePolicy
from vidaio.auditor.content_cache import EpochRecomputer
from vidaio.core.config import load_raw_config, section


def test_recompute_concurrency_defaults_and_env_override(monkeypatch):
    assert AuditorConfig().recompute_concurrency == 1
    monkeypatch.setenv("VIDAIO__AUDITOR__RECOMPUTE_CONCURRENCY", "4")
    config = section(load_raw_config(), "auditor", AuditorConfig)
    assert config.recompute_concurrency == 4
    assert isinstance(config.recompute_concurrency, int)


@pytest.mark.parametrize("concurrency", [
    0, 9, -1, 1.0, 1.5, float("nan"), float("inf"), True, False, "invalid",
])
def test_recompute_concurrency_rejects_noninteger_or_out_of_range_values(concurrency):
    with pytest.raises(ValidationError, match="recompute_concurrency"):
        AuditorConfig(recompute_concurrency=concurrency)


@pytest.mark.parametrize("concurrency", [1, 8])
def test_recompute_concurrency_accepts_both_bounds(concurrency):
    assert AuditorConfig(recompute_concurrency=concurrency).recompute_concurrency == concurrency




@pytest.fixture
def audit_case(tmp_path):
    store = LocalFsStore(tmp_path / "audit")
    bundles = [make_fake_bundle(store, challenge_id="c1", item_id=f"i{uid}",
        miner_hotkey=f"hk{uid}") for uid in range(1, 8)]
    log, source = _epoch_with_items(store, bundles)
    return SimpleNamespace(store=store, log=log, source=source,
        bundles=tuple(bundles),
        item_ids=tuple(bundle.item_id for bundle in bundles),
        policy=SamplePolicy(sample_rate=1.0, all_items=True))


class BlockingRecomputer(StaticRecomputer):
    """Real bundle verification around a thread-safe, controllable CPU double."""

    def __init__(self, item_ids, *, block=True):
        super().__init__(HONEST_METRICS, SCORER, score=HONEST_SCORE)
        self.lock = Lock()
        self.active = 0
        self.peak = 0
        self.started = []
        self.completed = []
        self.workers = set()
        self.saturated = Event()
        self.entered = {item_id: Event() for item_id in item_ids}
        self.release = {item_id: Event() for item_id in item_ids}
        self.finished = {item_id: Event() for item_id in item_ids}
        if not block:
            self.release_all()

    def release_all(self):
        for release in self.release.values():
            release.set()

    def recompute(self, bundle, artifacts):
        item_id = bundle.item_id
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.append(item_id)
            self.workers.add(current_thread())
            if self.active == 4:
                self.saturated.set()
            self.entered[item_id].set()
        try:
            if not self.release[item_id].wait(timeout=5):
                raise TimeoutError(f"test did not release recompute {item_id}")
            return super().recompute(bundle, artifacts)
        finally:
            with self.lock:
                self.active -= 1
                self.completed.append(item_id)
            self.finished[item_id].set()


def test_four_parallel_items_preserve_serial_verdict_order_and_report_bytes(audit_case, caplog):
    caplog.set_level("INFO")
    case = audit_case
    serial = BlockingRecomputer(case.item_ids, block=False)
    serial_report = _auditor(case.source, recompute_concurrency=1).audit_epoch(
        case.log, case.store, case.policy, serial, NOW)
    assert serial.peak == 1
    assert len(serial.started) == len(case.item_ids)
    assert tuple(verdict.item_id for verdict in serial_report.item_verdicts) == tuple(serial.started)

    parallel = BlockingRecomputer(case.item_ids)
    auditor = _auditor(case.source, recompute_concurrency=4)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-audit-driver") as driver:
        future = driver.submit(auditor.audit_epoch, case.log, case.store, case.policy, parallel, NOW)
        try:
            assert parallel.saturated.wait(timeout=5), "four recomputes never overlapped"
            with parallel.lock:
                assert parallel.active == parallel.peak == 4
                assert set(parallel.started) == set(serial.started[:4])
            # Keep the first three calls blocked while the fourth slot serves the rest.
            completion_order = serial.started[3:] + list(reversed(serial.started[:3]))
            for item_id in completion_order:
                assert parallel.entered[item_id].wait(timeout=5)
                parallel.release[item_id].set()
                assert parallel.finished[item_id].wait(timeout=5)
            report = future.result(timeout=5)
        finally:
            parallel.release_all()

    assert parallel.peak == 4 and parallel.active == 0
    assert parallel.completed == completion_order
    assert len(parallel.workers) == 4
    assert all(not worker.is_alive() for worker in parallel.workers)
    assert report.item_verdicts == serial_report.item_verdicts
    assert report == serial_report
    assert report.canonical_bytes() == serial_report.canonical_bytes()
    assert report.model_dump_json() == serial_report.model_dump_json()
    assert report.overall is AuditStatus.CLEAN
    assert all(verdict.verdict is ItemVerdictKind.PASS for verdict in report.item_verdicts)
    records = [record for record in caplog.records if record.getMessage() == "epoch items audited"]
    assert len(records) == 2
    assert [record.fields["concurrency"] for record in records] == [1, 4]
    assert all(record.fields["epoch_id"] == case.log.epoch_id for record in records)
    assert all(record.fields["n_items"] == len(case.item_ids) for record in records)
    assert all(record.fields["seconds"] >= 0 for record in records)


def test_recomputer_error_remains_fail_with_identical_serial_and_parallel_report(audit_case):
    case = audit_case

    class FailingRecomputer(StaticRecomputer):
        def recompute(self, bundle, artifacts):
            if bundle.item_id == "i3":
                raise RuntimeError("injected recompute failure")
            return super().recompute(bundle, artifacts)

    reports = [
        _auditor(case.source, recompute_concurrency=concurrency).audit_epoch(
            case.log, case.store, case.policy,
            FailingRecomputer(HONEST_METRICS, SCORER, score=HONEST_SCORE), NOW)
        for concurrency in (1, 4)
    ]
    assert reports[0] == reports[1]
    assert reports[0].canonical_bytes() == reports[1].canonical_bytes()
    assert reports[0].model_dump_json() == reports[1].model_dump_json()
    verdicts = {verdict.item_id: verdict for verdict in reports[1].item_verdicts}
    assert verdicts["i3"].verdict is ItemVerdictKind.FAIL
    assert verdicts["i3"].code == RECOMPUTE_ERROR
    assert all(verdict.verdict is ItemVerdictKind.PASS
        for item_id, verdict in verdicts.items() if item_id != "i3")
    assert reports[1].overall is AuditStatus.DISPUTED


def test_unexpected_audit_item_exception_propagates_after_all_workers_drain(audit_case):
    case = audit_case
    # Reuse the serial sampled order so the first map result raises promptly.
    serial = BlockingRecomputer(case.item_ids, block=False)
    _auditor(case.source, recompute_concurrency=1).audit_epoch(
        case.log, case.store, case.policy, serial, NOW)
    first, *peers = serial.started[:4]
    delayed_peer = peers[0]
    recomputer = BlockingRecomputer(case.item_ids)
    auditor = _auditor(case.source, recompute_concurrency=4)
    audit_item = auditor._audit_item
    allow_crash, crashed = Event(), Event()
    worker_lock = Lock()
    workers = set()

    class UnexpectedAuditFailure(RuntimeError):
        pass

    def crash_first(item, *args, **kwargs):
        with worker_lock:
            workers.add(current_thread())
        if item.item_id == first:
            if not allow_crash.wait(timeout=5):
                raise TimeoutError("test did not release the unexpected audit failure")
            crashed.set()
            raise UnexpectedAuditFailure("unexpected audit item failure")
        return audit_item(item, *args, **kwargs)

    auditor._audit_item = crash_first
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-audit-driver") as driver:
        future = driver.submit(auditor.audit_epoch, case.log, case.store, case.policy, recomputer, NOW)
        try:
            assert all(recomputer.entered[item_id].wait(timeout=5) for item_id in peers)
            allow_crash.set()
            assert crashed.wait(timeout=5)
            # One in-flight item deliberately remains blocked through executor shutdown.
            for item_id, release in recomputer.release.items():
                if item_id != delayed_peer:
                    release.set()
            assert not future.done()
            assert not recomputer.finished[delayed_peer].is_set()
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            recomputer.release[delayed_peer].set()
            with pytest.raises(UnexpectedAuditFailure, match="unexpected audit item failure"):
                future.result(timeout=5)
        finally:
            allow_crash.set()
            recomputer.release_all()

    assert recomputer.active == 0
    assert sorted(recomputer.started) == sorted(recomputer.completed)
    assert all(not worker.is_alive() for worker in workers)


@pytest.fixture
def memo_inputs(tmp_path):
    from tests.auditor.test_content_evidence import content_case

    store, _, _, template, context, *_ = content_case(tmp_path)
    entries = []
    for member in context.roster:
        bundle = template.model_copy(update={
            "item_id": member.receipt.metadata.task_id,
            "miner_hotkey": member.hotkey,
            "miner_output": member.output,
            "miner_receipt": member.receipt,
            "score_packet": member.score_packet,
            "scorer_version": context.committed_scorer_version,
            "backend_versions": {},
        })
        artifacts = {ArtifactKind.SCORE_PACKET: store.get(member.score_packet)}
        entries.append((bundle, artifacts))
    return entries


class MemoRecomputer:
    def __init__(self, entries):
        self.scorer_version = entries[0][0].scorer_version
        self.calls = []
        self.lock = Lock()
        self.entered = {bundle.item_id: Event() for bundle, _ in entries}
        self.release = Event()
        self.fail = False

    def recompute(self, bundle, artifacts):
        with self.lock:
            self.calls.append(bundle.item_id)
            self.entered[bundle.item_id].set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the memoized recompute")
        if self.fail:
            raise RuntimeError("temporary recompute failure")
        return RecomputedScore(metrics={"final_score": 0.5}, scorer_version=self.scorer_version, backend_versions={},
            score=0.5, gate_passed=True, breakdown={"nested": {"values": [1, 2]}})


def test_epoch_memo_shares_simultaneous_same_key_work_and_copies_results(memo_inputs):
    inner = MemoRecomputer(memo_inputs)
    memo = EpochRecomputer(inner)
    bundle, artifacts = memo_inputs[0]
    ready = Barrier(2)

    def request():
        ready.wait(timeout=5)
        return memo.recompute(bundle, artifacts)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(request) for _ in range(2)]
        try:
            assert inner.entered[bundle.item_id].wait(timeout=5)
            with pytest.raises(TimeoutError):
                futures[0].result(timeout=0.05)
            assert not any(future.done() for future in futures)
            assert inner.calls == [bundle.item_id]
            inner.release.set()
            left, right = [future.result(timeout=5) for future in futures]
        finally:
            inner.release.set()
    assert inner.calls == [bundle.item_id]
    assert left == right and left is not right
    assert left.metrics is not right.metrics
    assert left.breakdown["nested"]["values"] is not right.breakdown["nested"]["values"]
    left.metrics["final_score"] = 0.1
    left.breakdown["nested"]["values"].append(99)
    cached = memo.recompute(bundle, artifacts)
    assert cached == right and cached is not right
    assert right.metrics == {"final_score": 0.5}
    assert right.breakdown == {"nested": {"values": [1, 2]}}
    assert inner.calls == [bundle.item_id]


def test_epoch_memo_different_keys_compute_concurrently(memo_inputs):
    inner = MemoRecomputer(memo_inputs)
    memo = EpochRecomputer(inner)
    ready = Barrier(2)

    def request(entry):
        ready.wait(timeout=5)
        return memo.recompute(*entry)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(request, entry) for entry in memo_inputs]
        try:
            assert all(entered.wait(timeout=5) for entered in inner.entered.values())
            assert len(inner.calls) == 2
            assert not any(future.done() for future in futures)
            inner.release.set()
            results = [future.result(timeout=5) for future in futures]
        finally:
            inner.release.set()
    assert all(result.score == 0.5 for result in results)
    assert sorted(inner.calls) == sorted(bundle.item_id for bundle, _ in memo_inputs)


def test_epoch_memo_failure_wakes_same_key_waiters_and_later_call_retries(memo_inputs):
    inner = MemoRecomputer(memo_inputs)
    inner.fail = True
    memo = EpochRecomputer(inner)
    bundle, artifacts = memo_inputs[0]
    ready = Barrier(2)

    def request():
        ready.wait(timeout=5)
        return memo.recompute(bundle, artifacts)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(request) for _ in range(2)]
        try:
            assert inner.entered[bundle.item_id].wait(timeout=5)
            with pytest.raises(TimeoutError):
                futures[0].result(timeout=0.05)
            assert inner.calls == [bundle.item_id]
            inner.release.set()
            for future in futures:
                with pytest.raises(RuntimeError, match="temporary recompute failure"):
                    future.result(timeout=5)
        finally:
            inner.release.set()
    assert inner.calls == [bundle.item_id]
    inner.fail = False
    assert memo.recompute(bundle, artifacts).score == 0.5
    assert inner.calls == [bundle.item_id, bundle.item_id]
    assert memo.recompute(bundle, artifacts).score == 0.5
    assert inner.calls == [bundle.item_id, bundle.item_id]


@pytest.fixture
def real_recomputer_case(tmp_path):
    from tests.scoring_worker.test_runtime_identity import _attestation, _fake_backends
    from vidaio.auditor import RealScoreRecomputer
    from vidaio.scoring_worker import ScoringWorkerConfig

    config = ScoringWorkerConfig(work_dir=tmp_path / "work", max_input_bytes=16,
        max_request_bytes=32, max_request_scratch_bytes=80, max_scratch_bytes=100,
        request_timeout=12)
    backends = _fake_backends(_attestation())
    recomputer = RealScoreRecomputer(config, backends,
        allow_noncanonical_pre_marker_build_or_test_runtime=True)
    artifacts = {ArtifactKind.SCORE_PACKET: b'{"track":"compression"}',
        ArtifactKind.REFERENCE_ORIGINAL: b"reference", ArtifactKind.CHALLENGE_INPUT: b"input",
        ArtifactKind.MINER_OUTPUT: b"output"}
    bundle = SimpleNamespace(challenge_id="scratch-challenge", item_id="first", miner_hotkey="miner")
    return SimpleNamespace(recomputer=recomputer, config=config, backends=backends,
        artifacts=artifacts, bundle=bundle)


def _pipeline_item(request, identity):
    from vidaio.scoring import ItemScore

    return ItemScore(item_id=request.item_id, challenge_id=request.challenge_id,
        track=request.track, miner_hotkey=request.miner_hotkey, score=0.5,
        gate_passed=True, metrics={"final_score": 0.5}, scorer_version=identity)


def test_worker_clones_share_scratch_ceiling_and_retry_after_lease_cleanup(real_recomputer_case, monkeypatch):
    import vidaio.auditor.recomputer as module

    case = real_recomputer_case
    first, second = case.recomputer, case.recomputer.for_worker()
    first_entered, release_first, first_finished = Event(), Event(), Event()
    attempts, budgets, scratch_paths, waits, observed_usage = [], [], [], [], []
    lock = Lock()

    def score(request, config, scoring, backends, identity, scope=None, budget=None):
        with lock:
            attempts.append(request.item_id)
            budgets.append(budget)
        assert budget is not None
        with budget.lease() as lease, TemporaryDirectory(dir=config.work_dir, prefix="test-score-") as tmp:
            path = Path(tmp)
            with lock:
                scratch_paths.append((request.item_id, path))
            lease.reserve_generated(kind="snapshot", nbytes=10)
            (path / "snapshot").write_bytes(b"x" * 10)
            lease.reserve_generated(kind="expansion", nbytes=50)
            with lock:
                observed_usage.append(budget.used_bytes)
            if request.item_id == "first":
                first_entered.set()
                assert release_first.wait(timeout=5)
            return _pipeline_item(request, identity)

    def wait_for_capacity(delay):
        waits.append(delay)
        assert first._scratch_budget.used_bytes == 60
        refused_paths = [path for item_id, path in scratch_paths if item_id == "second"]
        assert refused_paths and all(not path.exists() for path in refused_paths)
        release_first.set()
        assert first_finished.wait(timeout=5)
        assert first._scratch_budget.used_bytes == 0

    def run_first():
        try:
            return first.recompute(case.bundle, case.artifacts)
        finally:
            first_finished.set()

    monkeypatch.setattr(module, "_score_sync", score)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=wait_for_capacity))
    with ThreadPoolExecutor(max_workers=2) as workers:
        first_result = workers.submit(run_first)
        try:
            assert first_entered.wait(timeout=5)
            bundle = SimpleNamespace(**{**vars(case.bundle), "item_id": "second"})
            second_result = workers.submit(second.recompute, bundle, case.artifacts)
            assert first_result.result(timeout=5).score == second_result.result(timeout=5).score == 0.5
        finally:
            release_first.set()
    assert attempts == ["first", "second", "second"]
    assert all(budget is first._scratch_budget for budget in budgets)
    assert waits == [5]
    assert observed_usage == [60, 60]
    assert first._scratch_budget.used_bytes == second._scratch_budget.used_bytes == 0
    assert all(not path.exists() for _, path in scratch_paths)
    assert len({path for _, path in scratch_paths}) == 3


def test_persistent_scratch_contention_stops_at_request_timeout(real_recomputer_case, monkeypatch):
    import vidaio.auditor.recomputer as module
    from vidaio.scoring_worker.inputs import ScoreRejected

    case = real_recomputer_case
    attempts, waits, scratch_paths = [], [], []
    now = [0.0]

    def score(request, config, scoring, backends, identity, scope=None, budget=None):
        attempts.append(now[0])
        with budget.lease() as lease, TemporaryDirectory(dir=config.work_dir, prefix="test-score-") as tmp:
            scratch_paths.append(Path(tmp))
            lease.reserve_generated(kind="snapshot", nbytes=10)
            lease.reserve_generated(kind="expansion", nbytes=50)
        pytest.fail("persistent competing reservation unexpectedly admitted the request")

    def sleep(delay):
        assert case.recomputer._scratch_budget.used_bytes == 60
        assert all(not path.exists() for path in scratch_paths)
        waits.append(delay)
        now[0] += delay

    monkeypatch.setattr(module, "_score_sync", score)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    with case.recomputer._scratch_budget.lease() as competing:
        competing.reserve_generated(kind="other-live-item", nbytes=60)
        with pytest.raises(RuntimeError, match="scratch_budget_unavailable") as failure:
            case.recomputer.recompute(case.bundle, case.artifacts)
        assert isinstance(failure.value.__cause__, ScoreRejected)
        assert case.recomputer._scratch_budget.used_bytes == 60
    assert now[0] == case.config.request_timeout == 12
    assert attempts == [0, 5, 10] and waits == [5, 5, 2]
    assert case.recomputer._scratch_budget.used_bytes == 0
    assert all(not path.exists() for path in scratch_paths)


@pytest.mark.parametrize("status, error", [(413, "request_scratch_too_large"), (503, "different_unavailability")])
def test_non_temporary_score_rejection_keeps_existing_failure_path(real_recomputer_case, monkeypatch, status, error):
    import vidaio.auditor.recomputer as module
    from vidaio.scoring_worker.inputs import ScoreRejected

    case = real_recomputer_case
    rejected = ScoreRejected(status, {"error": error})
    calls = []

    def score(*args, **kwargs):
        calls.append(1)
        raise rejected

    monkeypatch.setattr(module, "_score_sync", score)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0,
        sleep=lambda delay: pytest.fail("a permanent or unrelated rejection must not wait")))
    with pytest.raises(RuntimeError, match=error) as failure:
        case.recomputer.recompute(case.bundle, case.artifacts)
    assert failure.value.__cause__ is rejected and calls == [1]


def test_worker_backend_factory_is_lazy_reused_per_thread_and_preserves_history(tmp_path, monkeypatch):
    import vidaio.auditor.recomputer as module
    from tests.scoring_worker.test_runtime_identity import _attestation, _fake_backends
    from vidaio.auditor import RealScoreRecomputer
    from vidaio.auditor.service import _WorkerRecomputer
    from vidaio.epoch.log import _HistoryEpochLogV16
    from vidaio.scoring import ScoringConfig
    from vidaio.scoring_worker import ScoringWorkerConfig

    built = []
    attestation = _attestation()
    config = ScoringWorkerConfig(work_dir=tmp_path / "factory-work", pieapp_device="cuda")
    scoring = ScoringConfig(compression_norm=1.13)

    def compose(configured, *, scoring_config, pieapp_device):
        assert configured is config and scoring_config is scoring
        assert pieapp_device == "cpu"
        backends = _fake_backends(attestation)
        built.append(backends)
        return backends

    monkeypatch.setattr(module, "real_backends", compose)
    original = RealScoreRecomputer.from_config(config, scoring_config=scoring,
        allow_noncanonical_pre_marker_build_or_test_runtime=True)
    historical = original.for_epoch_log(_HistoryEpochLogV16.model_construct())
    assert historical.scorer_version != original.scorer_version
    workers = _WorkerRecomputer(historical)
    assert len(built) == 1
    assert workers._backends is original._backends
    ready = Barrier(4)

    def access_twice():
        first = workers._backends
        ready.wait(timeout=5)
        return first, workers._backends, workers.scorer_version, workers._scratch_budget

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(access_twice) for _ in range(4)]
        results = [future.result(timeout=5) for future in futures]
    assert len(built) == 4
    assert len({id(first.pieapp) for first, *_ in results}) == 4
    assert sum(first is original._backends for first, *_ in results) == 1
    assert all(first is again for first, again, _, _ in results)
    assert all(identity == historical.scorer_version for _, _, identity, _ in results)
    assert all(budget is original._scratch_budget for _, _, _, budget in results)
    assert all(backend.runtime_attestation == attestation for backend in built)
    assert attestation["execution_policy"]["actual_torch_intraop_threads"] == 1
    assert workers._backends is original._backends and len(built) == 4


def test_worker_clone_retains_explicitly_injected_backends(real_recomputer_case, monkeypatch):
    import vidaio.auditor.recomputer as module

    original = real_recomputer_case.recomputer
    monkeypatch.setattr(module, "real_backends",
        lambda *args, **kwargs: pytest.fail("an injected composition must not be replaced"))
    worker = original.for_worker()
    assert worker is not original
    assert worker._backends is original._backends
    assert worker.scorer_version == original.scorer_version
    assert worker._scratch_budget is original._scratch_budget


@pytest.mark.parametrize("kind", ["s3", "hippius"])
@pytest.mark.parametrize("failure_type", [OSError, FileNotFoundError])
def test_public_store_initialization_failure_preserves_serial_missing_artifact_verdict(
    audit_case, monkeypatch, kind, failure_type,
):
    import vidaio.audit.store as stores
    from vidaio.audit import AuditConfig
    from vidaio.audit.recompute import ARTIFACT_MISSING
    from vidaio.auditor.service import StoredBundleSource

    case = audit_case
    attempts = []

    def unavailable(config, *, anonymous):
        assert anonymous is True
        attempts.append(current_thread())
        raise failure_type("public store initialization unavailable")

    monkeypatch.setattr(stores, f"_connect_{kind}_transport", unavailable)
    store_type = stores.S3Store if kind == "s3" else stores.HippiusStore
    reports = []
    for concurrency in (1, 4):
        remote = store_type(AuditConfig(), public_read_only=True)
        auditor = _auditor(StoredBundleSource(remote), recompute_concurrency=concurrency)
        reports.append(auditor.audit_epoch(case.log, remote, case.policy,
            StaticRecomputer(HONEST_METRICS, SCORER, score=HONEST_SCORE), NOW))

    serial, parallel = reports
    assert parallel == serial
    assert parallel.model_dump_json() == serial.model_dump_json()
    assert parallel.canonical_bytes() == serial.canonical_bytes()
    assert len(serial.item_verdicts) == len(case.item_ids)
    assert all(verdict.verdict is ItemVerdictKind.SKIP and verdict.code == ARTIFACT_MISSING
        for verdict in serial.item_verdicts)
    assert len(attempts) >= len(case.item_ids) * 2


@pytest.mark.parametrize("kind", ["s3", "hippius"])
def test_public_remote_store_connectors_initialize_once_for_simultaneous_reads(audit_case, monkeypatch, kind):
    import vidaio.audit.store as stores
    from vidaio.audit import AuditConfig
    from vidaio.auditor.service import StoredBundleSource, persist_bundle

    case = audit_case
    for bundle in case.bundles:
        persist_bundle(case.store, bundle)
    names = ("bundles", "artifacts")
    configs = {name: AuditConfig() for name in names}
    transports = {name: object() for name in names}
    connections = {name: [] for name in names}
    entered = {name: Event() for name in names}
    release = {name: Event() for name in names}
    readers = []
    lock = Lock()

    def connect(config, *, anonymous):
        name = next(name for name in names if config is configs[name])
        with lock:
            connections[name].append((current_thread(), anonymous))
        entered[name].set()
        assert release[name].wait(timeout=5)
        return transports[name]

    monkeypatch.setattr(stores, f"_connect_{kind}_transport", connect)
    store_type = stores.S3Store if kind == "s3" else stores.HippiusStore

    class PublicStore(store_type):
        def __init__(self, name):
            super().__init__(configs[name], public_read_only=True)
            self.name = name
            self.ready = Barrier(4)
            self.n_reads = 0

        def read(self, operation, *args, **kwargs):
            with lock:
                readers.append(current_thread())
                self.n_reads += 1
                n_reads = self.n_reads
            if n_reads <= 4:
                self.ready.wait(timeout=5)
            assert self._t is transports[self.name]
            return getattr(case.store, operation)(*args, **kwargs)

        def get(self, *args, **kwargs):
            return self.read("get", *args, **kwargs)

        def get_limited(self, *args, **kwargs):
            return self.read("get_limited", *args, **kwargs)

        def get_digest_limited(self, *args, **kwargs):
            return self.read("get_digest_limited", *args, **kwargs)

        def materialize(self, *args, **kwargs):
            return self.read("materialize", *args, **kwargs)

        def exists(self, *args, **kwargs):
            return self.read("exists", *args, **kwargs)

    artifacts = PublicStore("artifacts")
    bundles = PublicStore("bundles")
    assert all(not calls for calls in connections.values())
    auditor = _auditor(StoredBundleSource(bundles), recompute_concurrency=4)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-audit-driver") as driver:
        future = driver.submit(auditor.audit_epoch, case.log, artifacts, case.policy,
            StaticRecomputer(HONEST_METRICS, SCORER, score=HONEST_SCORE), NOW)
        try:
            for name in names:
                assert entered[name].wait(timeout=5)
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.05)
                assert len(connections[name]) == 1
                release[name].set()
            report = future.result(timeout=5)
        finally:
            for event in release.values():
                event.set()
    assert report.overall is AuditStatus.CLEAN
    assert all(len(calls) == 1 and calls[0][1] is True for calls in connections.values())
    assert any(reader is not current_thread() for reader in readers)
