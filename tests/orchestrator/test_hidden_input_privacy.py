"""Hidden inputs and contender outputs never reach the public store before the result.

``challenge_input`` and ``miner_output`` are public object kinds (keyless readers fetch
them by digest). A running competition therefore keeps both in its local digest pools;
they are uploaded, verified, and recorded right before the engine may mark the
competition COMPLETED, so epoch evidence and auditors find them exactly as before.
"""

from __future__ import annotations

import json

import pytest

from vidaio.audit import ArtifactKind, ArtifactRef
from vidaio.audit.store import backend_key
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase

from orchestrator_support import (
    END,
    FINALIZATION,
    M,
    T0,
    build_manifest,
    events_of,
    phase,
    seed_items,
    start_and_enroll,
)


def _public_refs(orch, cid) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    for row in orch.conn.execute(
        "SELECT input_sha256, input_bytes FROM evaluation_items WHERE competition_id = ?",
        (cid,),
    ).fetchall():
        refs.append(ArtifactRef(
            digest=row["input_sha256"], kind=ArtifactKind.CHALLENGE_INPUT,
            byte_size=row["input_bytes"],
            backend_key=backend_key(ArtifactKind.CHALLENGE_INPUT, row["input_sha256"]),
        ))
    for row in orch.conn.execute(
        "SELECT DISTINCT output_sha256, output_bytes FROM performance_history "
        "WHERE competition_id = ?",
        (cid,),
    ).fetchall():
        refs.append(ArtifactRef(
            digest=row["output_sha256"], kind=ArtifactKind.MINER_OUTPUT,
            byte_size=row["output_bytes"],
            backend_key=backend_key(ArtifactKind.MINER_OUTPUT, row["output_sha256"]),
        ))
    return refs


def _public_prefix_files(tmp_path) -> list:
    audit = tmp_path / "audit"
    return [
        path
        for prefix in ("challenge_input", "miner_output")
        for path in (audit / prefix).rglob("*")
        if path.is_file()
    ]


async def _drive_to_result_due(orch, cid, tmp_path) -> None:
    seed_items(orch, cid, tmp_path / "item-src")
    await orch.step(FINALIZATION)
    await orch.step(FINALIZATION + 2 * M)
    await orch.step(FINALIZATION + 3 * M)
    await orch.step(FINALIZATION + 10 * M)
    await orch.step(FINALIZATION + 15 * M)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME


async def test_nothing_public_before_the_result_everything_after(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await _drive_to_result_due(orch, cid, tmp_path)

    refs = _public_refs(orch, cid)
    assert {ref.kind for ref in refs} == {ArtifactKind.CHALLENGE_INPUT, ArtifactKind.MINER_OUTPUT}
    # Scored, audit-linked, bundles persisted -- and still nothing public.
    assert orch.engine.audit_linkage_gaps(orch.conn, cid) == []
    assert not any(orch.store.exists(ref) for ref in refs)
    assert _public_prefix_files(tmp_path) == []
    # The persisted bundles name exactly those (not yet uploaded) refs.
    for event in events_of(orch, cid, "audit_bundle_built"):
        bundle = json.loads(event["payload_json"])["bundle"]
        assert bundle["challenge_input"]["digest"] in {r.digest for r in refs}
        assert bundle["miner_output"]["digest"] in {r.digest for r in refs}

    await orch.step(END + M)

    assert phase(orch, cid) is Phase.COMPLETED
    for ref in refs:
        assert orch.store.exists(ref)
        pool = orch.inputs_dir if ref.kind is ArtifactKind.CHALLENGE_INPUT else orch.outputs_dir
        assert orch.store.get(ref) == (pool / ref.digest).read_bytes()
    [published] = events_of(orch, cid, pers.EVENT_COMPETITION_ARTIFACTS_PUBLISHED)
    payload = json.loads(published["payload_json"])
    assert sorted(payload["challenge_input"] + payload["miner_output"]) == sorted(
        ref.digest for ref in refs
    )


async def test_failed_upload_halts_and_never_completes_then_resumes(
    orchestrator_factory, fixture_repos, tmp_path, monkeypatch
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await _drive_to_result_due(orch, cid, tmp_path)

    real_put_file = orch.store.put_file
    calls = {"n": 0}

    def flaky_put_file(path, kind):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("object store unavailable")
        return real_put_file(path, kind)

    monkeypatch.setattr(orch.store, "put_file", flaky_put_file)
    await orch.step(END + M)

    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    assert pers.is_halted(orch.conn, cid)
    [halt] = events_of(orch, cid, pers.EVENT_HALTED)
    assert "publishing the hidden inputs/outputs" in halt["payload_json"]
    assert events_of(orch, cid, pers.EVENT_COMPETITION_ARTIFACTS_PUBLISHED) == []

    # Operator clears the halt; the retry re-verifies (idempotent puts) and completes.
    orch.clear_halt(cid, operator="ops", reason="store back", now=END + 2 * M)
    await orch.step(END + 3 * M)
    assert phase(orch, cid) is Phase.COMPLETED
    assert all(orch.store.exists(ref) for ref in _public_refs(orch, cid))


async def test_missing_local_pool_file_halts_instead_of_completing(
    orchestrator_factory, fixture_repos, tmp_path
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await _drive_to_result_due(orch, cid, tmp_path)
    victim = next(r for r in _public_refs(orch, cid) if r.kind is ArtifactKind.CHALLENGE_INPUT)
    (orch.inputs_dir / victim.digest).unlink()

    await orch.step(END + M)

    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    assert pers.is_halted(orch.conn, cid)
    [halt] = events_of(orch, cid, pers.EVENT_HALTED)
    assert "neither stored nor in the local pool" in halt["payload_json"]


async def test_publication_survives_restart_and_is_not_repeated(
    orchestrator_factory, fixture_repos, tmp_path, monkeypatch
):
    orch = orchestrator_factory(repos=fixture_repos)
    cid = await start_and_enroll(orch, build_manifest(), ["hk-a", "hk-b"])
    await _drive_to_result_due(orch, cid, tmp_path)
    # Hold completion open (as an open audit-linkage gate would) to watch re-entry.
    monkeypatch.setattr(type(orch.engine), "_complete", lambda self, conn, comp, now: False)
    await orch.step(END + M)
    assert phase(orch, cid) is Phase.AWAITING_END_TIME
    assert len(events_of(orch, cid, pers.EVENT_COMPETITION_ARTIFACTS_PUBLISHED)) == 1

    restarted = orchestrator_factory(repos=fixture_repos)
    puts: list[str] = []
    monkeypatch.setattr(restarted.store, "put_file", lambda path, kind: puts.append(kind))
    monkeypatch.setattr(
        restarted.store, "exists",
        lambda ref: pytest.fail("verified digests are not re-checked every tick"),
    )
    assert restarted._publish_due_competition_artifacts(cid, END + 2 * M) is True
    assert puts == []
    assert len(events_of(restarted, cid, pers.EVENT_COMPETITION_ARTIFACTS_PUBLISHED)) == 1


async def test_competition_inputs_registered_but_never_run_stay_private(
    orchestrator_factory, tmp_path
):
    orch = orchestrator_factory()
    manifest = build_manifest()
    orch.create_competition(manifest, T0)
    seed_items(orch, manifest.competition_id, tmp_path / "item-src")
    rows = orch.conn.execute(
        "SELECT input_sha256 FROM evaluation_items WHERE competition_id = ?",
        (manifest.competition_id,),
    ).fetchall()
    assert rows and all((orch.inputs_dir / row["input_sha256"]).is_file() for row in rows)
    assert _public_prefix_files(tmp_path) == []
