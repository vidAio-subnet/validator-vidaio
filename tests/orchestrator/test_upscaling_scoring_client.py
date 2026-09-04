"""The trusted scorer, unlike the miner runner, receives both upscaling inputs."""

from __future__ import annotations

import hashlib
import json

import httpx

from vidaio.competition import LifecycleEngine, evaluation_item_commitment, migrate
from vidaio.competition import repository as repo
from vidaio.competition.interfaces import BatchItem, BatchOutput
from vidaio.competition.orchestrator import HttpScoringClient
from vidaio.core.db import connect
from vidaio.services.protocol import (
    SCORER_RUNTIME_BACKEND_PREFIX,
    SCORER_RUNTIME_COMMITMENT_SCHEMA,
    ScoreRequest,
)

from orchestrator_support import COMMITMENT_ROOT, START, T0, build_manifest

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


def test_upscaling_request_uses_committed_reference_input_and_factor(tmp_path) -> None:
    input_digest = "1" * 64
    reference_digest = "2" * 64
    output_digest = "3" * 64
    competition_id = "comp-upscale-client"
    commitment = evaluation_item_commitment(
        competition_id=competition_id,
        item_index=0,
        reference_sha256=reference_digest,
        input_sha256=input_digest,
        upscale_factor=4,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    manifest = build_manifest(
        competition_id,
        track="upscaling",
        allowed_upscale_factors=[2, 4],
        evaluation_item_commitments=[commitment],
    )
    conn = connect(":memory:")
    migrate(conn)
    engine = LifecycleEngine()
    engine.create_competition(conn, manifest, T0)
    engine.mark_commitment_anchored(conn, competition_id, COMMITMENT_ROOT, T0)
    engine.tick(conn, START)
    contender_id = repo.enroll_contender(
        conn,
        competition_id,
        hotkey="hk-upscale",
        repo_url="local://hk-upscale",
        commit_sha="1a" * 20,
        tree_sha="2a" * 20,
        stake=1000.0,
        now=START,
    )
    item_id = repo.add_evaluation_item(
        conn,
        competition_id,
        item_index=0,
        input_sha256=input_digest,
        input_bytes=1024,
        reference_sha256=reference_digest,
        reference_bytes=4096,
        upscale_factor=4,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        threshold_commitment="f" * 64,
        challenge_id="chal-upscale",
        now=START,
    )

    attestation = {
        "schema": SCORER_RUNTIME_COMMITMENT_SCHEMA,
        "payout_backends": {
            "pieapp": "piq/0.8.0:cpu",
            "torch": "torch/2.8.0+cpu",
        },
    }
    attestation_bytes = json.dumps(
        attestation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    runtime_digest = hashlib.sha256(attestation_bytes).hexdigest()
    backend_versions = {
        **attestation["payout_backends"],
        "runtime": SCORER_RUNTIME_BACKEND_PREFIX + runtime_digest,
    }
    packet_json = json.dumps(
        {
            "item_id": input_digest,
            "challenge_id": "chal-upscale",
            "track": "upscaling",
            "miner_hotkey": "hk-upscale",
            "content_digest": output_digest,
            "score": 0.7,
            "gate_passed": True,
            "violations": [],
            "skips": [],
            "breakdown": None,
            "metrics": {},
            "scorer_version": "v1.0.0",
            "backend_versions": backend_versions,
            "canonicalization_plan_digest": None,
            "pieapp_start_frame": 0,
            "scoring_config_digest": "a" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(
                200,
                json={
                    "scorer_version": "v1.0.0",
                    "runtime_commitment": {
                        "digest": runtime_digest,
                        "attestation": attestation,
                    },
                },
            )
        body = ScoreRequest.model_validate_json(request.content)
        assert body.track == "upscaling"
        assert body.reference_digest == reference_digest
        assert body.miner_input_digest == input_digest
        assert body.reference_digest != body.miner_input_digest
        assert body.reference_path == str(tmp_path / "in" / reference_digest)
        assert body.miner_input_path == str(tmp_path / "in" / input_digest)
        assert body.params == {
            "upscale_factor": 4,
            "target_width": TARGET_WIDTH,
            "target_height": TARGET_HEIGHT,
        }
        return httpx.Response(
            200,
            json={
                "item_score_json": packet_json,
                "packet_digest": hashlib.sha256(packet_json.encode()).hexdigest(),
            },
        )

    client = HttpScoringClient(
        "http://scoring.test",
        conn,
        inputs_dir=tmp_path / "in",
        outputs_dir=tmp_path / "out",
        transport=httpx.MockTransport(handler),
        allow_noncanonical_runtime_for_report_or_tests=True,
    )
    item = BatchItem(
        item_id=item_id,
        item_index=0,
        input_sha256=input_digest,
        input_bytes=1024,
        upscale_factor=4,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    output = BatchOutput(
        item_id=item_id,
        output_sha256=output_digest,
        output_bytes=4096,
    )
    try:
        packet = client.score_item(competition_id, contender_id, item, output)
    finally:
        conn.close()
    assert packet.packet_bytes == packet_json.encode()
