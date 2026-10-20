"""Honest refusal when the CPU PieAPP dependency is explicitly unavailable.

No media tools needed — the recomputer refuses BEFORE any scoring, exactly like the
scoring worker's typed 501.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.scoring_worker.conftest import RoleKeyedBackend
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind
from vidaio.challenge.commitment import ChallengeCommitment
from vidaio.challenge.dag import build_dag, dag_rng_from_seed
from vidaio.scoring import (
    DeterministicFakeBackend,
    MediaInfo,
    PerceptualCheckResult,
    ScoringConfig,
)
from tests.legacy_validator_zero import forged_validator_zero_packet
from vidaio.scoring.backends_real import PieAppTorchBackend
from vidaio.scoring_worker import ScoringBackends, ScoringWorkerConfig
from vidaio.auditor import RealScoreRecomputer, RecomputeUnavailable


def _recomputer_with_unconfigured_pieapp(tmp_path) -> RealScoreRecomputer:
    fake = DeterministicFakeBackend()
    backends = ScoringBackends(
        probe=fake,
        vmaf_primary=fake,
        vmaf_secondary=fake,
        # An explicit seam: installing the media extra must not change this
        # refusal test into a configured-backend test.
        pieapp=PieAppTorchBackend(_backend_version="not-configured"),
        perceptual=fake,
        canonicalizer=None,
        versions={},
    )
    return RealScoreRecomputer(
        ScoringWorkerConfig(work_dir=tmp_path / "work"),
        backends,
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )


def test_upscaling_item_refuses_rather_than_false_clean(tmp_path) -> None:
    recomputer = _recomputer_with_unconfigured_pieapp(tmp_path)
    artifacts = {ArtifactKind.SCORE_PACKET: json.dumps({"track": "upscaling"}).encode()}

    # cheap probe: "cannot recompute" without touching media
    reason = recomputer.unsupported_reason(object(), artifacts)
    assert reason is not None and "PieAPP" in reason

    # recompute() honestly refuses — never a substituted (false CLEAN) verdict
    with pytest.raises(RecomputeUnavailable):
        recomputer.recompute(object(), artifacts)


def test_compression_item_is_recomputable(tmp_path) -> None:
    recomputer = _recomputer_with_unconfigured_pieapp(tmp_path)
    artifacts = {
        ArtifactKind.SCORE_PACKET: json.dumps({"track": "compression"}).encode()
    }
    assert recomputer.unsupported_reason(object(), artifacts) is None


def test_from_config_forces_cpu_even_when_scorer_is_configured_cuda(
    tmp_path, monkeypatch
) -> None:
    fake = DeterministicFakeBackend()
    captured = {}

    def compose(config, *, scoring_config, pieapp_device):
        captured["device"] = pieapp_device
        return ScoringBackends(
            probe=fake,
            vmaf_primary=fake,
            vmaf_secondary=fake,
            pieapp=fake,
            perceptual=fake,
            canonicalizer=None,
            versions={},
        )

    monkeypatch.setattr("vidaio.auditor.recomputer.real_backends", compose)
    config = ScoringWorkerConfig(work_dir=tmp_path / "work", pieapp_device="cuda")
    RealScoreRecomputer.from_config(
        config,
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    assert captured["device"] == "cpu"


def _committed_upscaling_reveal(seed: int = 123456789) -> tuple[object, bytes, int]:
    dag = build_dag("upscaling", dag_rng_from_seed(seed))
    raw = ChallengeCommitment.preimage_payload(
        "asset-upscale",
        dag.canonical_digest(),
        seed,
        "vidaio-scorer/test",
        "upscaling",
        7,
    )
    downscale = next(op for op in dag.ops if getattr(op, "op", "") == "downscale")
    return (
        SimpleNamespace(commitment_hash=sha256_hex(raw)),
        raw,
        round(1.0 / downscale.scale_factor),
    )


def test_upscaling_factor_is_reconstructed_from_the_committed_seed_dag(
    tmp_path, monkeypatch
) -> None:
    """The auditor supplies the honest file-size gate factor without trusting packet params."""
    fake = DeterministicFakeBackend()
    recomputer = RealScoreRecomputer(
        ScoringWorkerConfig(work_dir=tmp_path / "work"),
        ScoringBackends(
            probe=fake,
            vmaf_primary=fake,
            vmaf_secondary=fake,
            pieapp=fake,
            perceptual=fake,
            canonicalizer=None,
            versions={},
        ),
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    bundle, reveal, expected_factor = _committed_upscaling_reveal()
    bundle.challenge_id = "challenge-upscale"
    bundle.item_id = "item-upscale"
    bundle.miner_hotkey = "miner"
    artifacts = {
        ArtifactKind.REFERENCE_ORIGINAL: b"reference",
        ArtifactKind.CHALLENGE_INPUT: b"input",
        ArtifactKind.MINER_OUTPUT: b"output",
        ArtifactKind.DAG_REVEAL: reveal,
        # A malicious packet-side factor is deliberately ignored.
        ArtifactKind.SCORE_PACKET: json.dumps(
            {"track": "upscaling", "params": {"upscale_factor": 999}}
        ).encode(),
    }
    captured = {}

    def score(request, *_args):
        captured.update(request.params)
        return SimpleNamespace(
            metrics={},
            scorer_version=recomputer.scorer_version,
            backend_versions={},
            score=0.0,
            gate_passed=False,
            canonical_content_digest=None,
            content_fingerprint=None,
            encoded_size=None,
            canonicalization_plan_digest=None,
        )

    monkeypatch.setattr("vidaio.auditor.recomputer._score_sync", score)

    recomputer.recompute(bundle, artifacts)

    assert captured == {"upscale_factor": expected_factor}


def test_upscaling_auditor_recompute_uses_miner_input_perceptual_basis(
    tmp_path,
) -> None:
    """The CPU auditor reruns all three checks against the bytes the miner saw."""

    def media(width: int, height: int, byte_size: int) -> MediaInfo:
        return MediaInfo(
            codec="h264",
            width=width,
            height=height,
            fps=30.0,
            frame_count=60,
            duration=2.0,
            byte_size=byte_size,
        )

    fake = RoleKeyedBackend(
        vmaf={
            ("reference", "output"): 70.0,
            ("miner_input", "output"): 70.0,
        },
        pieapp={("reference", "output"): 0.2},
        media={
            "reference": media(320, 240, 1_000),
            "miner_input": media(160, 120, 200),
            "output": media(320, 240, 600),
        },
    )

    class BasisSensitivePerceptual:
        name = "basis-sensitive"
        version = "1"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def _check(self, reference: str, candidate: str) -> PerceptualCheckResult:
            roles = (RoleKeyedBackend.role(reference), RoleKeyedBackend.role(candidate))
            self.calls.append(roles)
            return PerceptualCheckResult(
                passed=roles == ("miner_input", "output"),
                measure=0.2 if roles[0] == "miner_input" else 2.0,
                limit=1.0,
                comparison="maximum",
            )

        check_tone_manipulation = _check
        check_color_grayscale = _check
        check_chroma_uv = _check

    perceptual = BasisSensitivePerceptual()
    recomputer = RealScoreRecomputer(
        ScoringWorkerConfig(work_dir=tmp_path / "work"),
        ScoringBackends(
            probe=fake,
            vmaf_primary=fake,
            vmaf_secondary=fake,
            pieapp=fake.pieapp,
            perceptual=perceptual,
            canonicalizer=None,
            versions=fake.versions(),
        ),
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    bundle, reveal, _factor = _committed_upscaling_reveal()
    bundle.challenge_id = "challenge-upscale"
    bundle.item_id = "item-upscale"
    bundle.miner_hotkey = "miner"
    artifacts = {
        ArtifactKind.REFERENCE_ORIGINAL: b"R" * 1_000,
        ArtifactKind.CHALLENGE_INPUT: b"I" * 200,
        ArtifactKind.MINER_OUTPUT: b"O" * 600,
        ArtifactKind.DAG_REVEAL: reveal,
        ArtifactKind.SCORE_PACKET: b'{"track":"upscaling"}',
    }

    result = recomputer.recompute(bundle, artifacts)

    assert result.gate_passed and result.score > 0.0
    assert perceptual.calls == [("miner_input", "output")] * 3


def test_upscaling_factor_rejects_a_self_hashed_reveal_that_seed_cannot_regenerate(
    tmp_path,
) -> None:
    """Hashing a hand-picked DAG into a new preimage does not make it seed-derived."""
    fake = DeterministicFakeBackend()
    recomputer = RealScoreRecomputer(
        ScoringWorkerConfig(work_dir=tmp_path / "work"),
        ScoringBackends(
            probe=fake,
            vmaf_primary=fake,
            vmaf_secondary=fake,
            pieapp=fake,
            perceptual=fake,
            canonicalizer=None,
            versions={},
        ),
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    _bundle, honest_reveal, _factor = _committed_upscaling_reveal()
    tampered_doc = json.loads(honest_reveal)
    tampered_doc["dag_digest"] = "0" * 64
    tampered_reveal = canonical_json_bytes(tampered_doc)
    bundle = SimpleNamespace(
        commitment_hash=sha256_hex(tampered_reveal),
        challenge_id="challenge-upscale",
        item_id="item-upscale",
        miner_hotkey="miner",
    )
    artifacts = {
        ArtifactKind.REFERENCE_ORIGINAL: b"reference",
        ArtifactKind.CHALLENGE_INPUT: b"input",
        ArtifactKind.MINER_OUTPUT: b"output",
        ArtifactKind.DAG_REVEAL: tampered_reveal,
        ArtifactKind.SCORE_PACKET: b'{"track":"upscaling"}',
    }

    with pytest.raises(RuntimeError, match="does not regenerate"):
        recomputer.recompute(bundle, artifacts)


def test_validator_attributed_upscaling_zero_is_explicitly_refused(tmp_path) -> None:
    recomputer = _recomputer_with_unconfigured_pieapp(tmp_path)
    bundle, reveal, _factor = _committed_upscaling_reveal()
    packet = forged_validator_zero_packet(
        item_id="item-upscale",
        challenge_id="challenge-upscale",
        track="upscaling",
        miner_hotkey="miner",
        committed_scorer_version="vidaio-scorer/test",
        failure_reason="timeout",
        config=ScoringConfig(),
    )
    bundle.item_id = packet.item_id
    bundle.challenge_id = packet.challenge_id
    bundle.miner_hotkey = packet.miner_hotkey
    bundle.scorer_version = packet.scorer_version
    artifacts = {
        ArtifactKind.REFERENCE_ORIGINAL: b"reference",
        ArtifactKind.CHALLENGE_INPUT: b"input",
        ArtifactKind.MINER_OUTPUT: b"",
        ArtifactKind.DAG_REVEAL: reveal,
        ArtifactKind.SCORE_PACKET: packet.to_json().encode(),
    }

    assert recomputer.unsupported_reason(bundle, artifacts) is None
    with pytest.raises(RuntimeError, match="not launch-valid economic evidence"):
        recomputer.recompute(bundle, artifacts)


def test_validator_zero_convention_is_refused_even_with_nonempty_output(
    tmp_path,
) -> None:
    recomputer = _recomputer_with_unconfigured_pieapp(tmp_path)
    bundle, reveal, _factor = _committed_upscaling_reveal()
    packet = forged_validator_zero_packet(
        item_id="item-upscale",
        challenge_id="challenge-upscale",
        track="upscaling",
        miner_hotkey="miner",
        committed_scorer_version="vidaio-scorer/test",
        failure_reason="timeout",
        config=ScoringConfig(),
    )
    bundle.scorer_version = packet.scorer_version
    artifacts = {
        ArtifactKind.MINER_OUTPUT: b"not empty",
        ArtifactKind.DAG_REVEAL: reveal,
        ArtifactKind.SCORE_PACKET: packet.to_json().encode(),
    }

    with pytest.raises(RuntimeError, match="not launch-valid economic evidence"):
        recomputer.recompute(bundle, artifacts)


def test_from_config_sweeps_stale_scratch_but_views_never_do(tmp_path, monkeypatch) -> None:
    fake = DeterministicFakeBackend()

    def compose(config, *, scoring_config, pieapp_device):
        return ScoringBackends(
            probe=fake, vmaf_primary=fake, vmaf_secondary=fake, pieapp=fake,
            perceptual=fake, canonicalizer=None, versions={},
        )

    monkeypatch.setattr("vidaio.auditor.recomputer.real_backends", compose)
    work = tmp_path / "work"
    (work / "score-killed" / "nested").mkdir(parents=True)
    (work / "score-killed" / "nested" / "copy.y4m").write_bytes(b"x" * 1024)
    (work / "vmaf-orphan").mkdir()
    (work / "operator-notes.txt").write_text("not ours")

    recomputer = RealScoreRecomputer.from_config(
        ScoringWorkerConfig(work_dir=work),
        scoring_config=ScoringConfig(),
        allow_noncanonical_pre_marker_build_or_test_runtime=True,
    )
    assert sorted(p.name for p in work.iterdir()) == ["operator-notes.txt"]

    # A live request's scratch created after composition survives worker and history views.
    (work / "score-live").mkdir()
    recomputer._owns_real_backends = False  # fakes carry no runtime attestation to re-check
    recomputer.for_worker()
    recomputer.for_epoch_log(object())
    assert (work / "score-live").is_dir()


def _faithful_upscaling_bundle(scoring: object = "faithful"):
    from vidaio.audit import CompetitionItemBinding, LifecycleStage
    from vidaio.competition import evaluation_item_commitment

    commitment = evaluation_item_commitment(
        competition_id="comp-faithful",
        item_index=0,
        reference_sha256="2" * 64,
        input_sha256="1" * 64,
        upscale_factor=4,
        target_width=1920,
        target_height=1080,
    )
    bundle = SimpleNamespace(
        stage=LifecycleStage.COMPETITION_SEALED,
        competition_item=CompetitionItemBinding(
            item_index=0,
            input_sha256="1" * 64,
            reference_sha256="2" * 64,
            upscale_factor=4,
            target_width=1920,
            target_height=1080,
            item_commitment=commitment,
        ),
    )
    manifest = {
        "competition_id": "comp-faithful",
        "track": "upscaling",
        "allowed_upscale_factors": [2, 4],
        "evaluation_item_commitments": [commitment],
        "upscaling_scoring": scoring,
    }
    artifacts = {
        ArtifactKind.SCORE_PACKET: json.dumps({"track": "upscaling"}).encode(),
        ArtifactKind.MANIFEST: json.dumps(manifest).encode(),
    }
    return bundle, artifacts


def test_faithful_upscaling_item_is_recomputable_without_pieapp(tmp_path) -> None:
    # the faithful scorer measures frames against the sealed original: no PieAPP
    recomputer = _recomputer_with_unconfigured_pieapp(tmp_path)
    bundle, artifacts = _faithful_upscaling_bundle()
    assert recomputer.unsupported_reason(bundle, artifacts) is None
    assert recomputer.unsupported_reason(bundle, artifacts, track="upscaling") is None


def test_faithful_upscaling_selector_comes_from_the_anchored_manifest(tmp_path) -> None:
    bundle, artifacts = _faithful_upscaling_bundle()
    # the packet's own params never pick the scorer
    packet = {"track": "upscaling", "params": {"scoring": "inference"}}
    params = RealScoreRecomputer._reconstruct_params(bundle, artifacts, packet)
    assert params == {
        "upscale_factor": 4,
        "target_width": 1920,
        "target_height": 1080,
        "scoring": "faithful",
    }
    bundle, artifacts = _faithful_upscaling_bundle(scoring=None)
    assert "scoring" not in RealScoreRecomputer._reconstruct_params(bundle, artifacts, packet)
    bundle, artifacts = _faithful_upscaling_bundle(scoring="generative")
    with pytest.raises(RuntimeError, match="unknown upscaling scorer"):
        RealScoreRecomputer._reconstruct_params(bundle, artifacts, packet)


def test_streaming_rules_come_from_the_anchored_compression_manifest() -> None:
    from vidaio.audit import LifecycleStage

    rules = {"bits_per_pixel": 0.05, "max_keyframe_interval_seconds": 2.0, "peak_ratio": 1.5,
             "peak_window_seconds": 2.0}
    bundle = SimpleNamespace(stage=LifecycleStage.PRE_REVEAL, competition_item=None)

    def artifacts(**manifest):
        return {
            ArtifactKind.SCORE_PACKET: json.dumps({"track": "compression"}).encode(),
            ArtifactKind.MANIFEST: json.dumps({"track": "compression", "vmaf_threshold": 90.0, **manifest}).encode(),
        }

    packet = {"track": "compression", "params": {"bits_per_pixel": 9.0}}
    params = RealScoreRecomputer._reconstruct_params(
        bundle, artifacts(compression_scoring="streaming", streaming_rules=rules), packet
    )
    assert params == {
        "vmaf_threshold": 90.0, "scoring": "streaming", "bits_per_pixel": 0.05,
        "max_keyframe_interval_s": 2.0, "peak_ratio": 1.5, "peak_window_s": 2.0,
    }
    assert RealScoreRecomputer._reconstruct_params(bundle, artifacts(), packet) == {"vmaf_threshold": 90.0}
    with pytest.raises(RuntimeError, match="unknown compression scorer"):
        RealScoreRecomputer._reconstruct_params(bundle, artifacts(compression_scoring="vbr"), packet)
    with pytest.raises(RuntimeError, match="streaming_rules invalid"):
        RealScoreRecomputer._reconstruct_params(
            bundle, artifacts(compression_scoring="streaming", streaming_rules={"bits_per_pixel": 0}), packet
        )
