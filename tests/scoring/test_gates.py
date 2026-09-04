"""Gates-first semantics: any failing gate zeroes even a perfect metric."""

from vidaio.scoring import (
    DeterministicFakeBackend,
    GateContext,
    MediaInfo,
    PerceptualCheckResult,
    ReasonCode,
    ScoringConfig,
    compose_item_score,
    default_pipeline,
    score_compression,
)
from vidaio.scoring.config import TRACK_COMPRESSION, TRACK_UPSCALING

CFG = ScoringConfig()


def info(**overrides) -> MediaInfo:
    base = dict(
        codec="h264",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        duration=10.0,
        byte_size=1_000_000,
        pix_fmt="yuv420p",
    )
    base.update(overrides)
    return MediaInfo(**base)


def make_ctx(**overrides) -> GateContext:
    kwargs = dict(
        track=TRACK_COMPRESSION,
        config=CFG,
        reference_info=info(),
        candidate_info=info(byte_size=400_000),
        reference_path="ref.mp4",
        candidate_path="cand.mp4",
        vmaf_primary=95.0,
        vmaf_secondary=94.0,
    )
    kwargs.update(overrides)
    return GateContext(**kwargs)


def pipeline(checks=None):
    return default_pipeline(
        DeterministicFakeBackend(perceptual_checks=checks or {})
    )


def test_all_gates_pass() -> None:
    passed, violations = pipeline().run(make_ctx())
    assert passed
    assert violations == []


def test_failing_gate_zeroes_a_perfect_metric_score() -> None:
    # Perfect compression metrics, but candidate uses a disallowed codec.
    ctx = make_ctx(candidate_info=info(codec="mpeg4", byte_size=100_000))
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert [v.code for v in violations] == [ReasonCode.ENCODING_NOT_ALLOWED]

    breakdown = score_compression(
        candidate_bytes=100_000, reference_bytes=1_000_000, vmaf=99.0, config=CFG
    )
    assert breakdown.final > 0.8  # the metric alone would have scored high
    item = compose_item_score(
        item_id="i1",
        challenge_id="c1",
        track=TRACK_COMPRESSION,
        gate_passed=passed,
        violations=violations,
        breakdown=breakdown,
        config=CFG,
    )
    assert item.score == 0.0  # gates-first: the gate wins
    assert not item.gate_passed


def test_frame_count_gate() -> None:
    ctx = make_ctx(candidate_info=info(frame_count=299, byte_size=400_000))
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.FRAME_COUNT_MISMATCH
    assert violations[0].measured == 299.0
    assert violations[0].limit == 300.0


def test_compression_rate_gate_at_exactly_080() -> None:
    ctx = make_ctx(candidate_info=info(byte_size=800_000))
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.COMPRESSION_RATE_TOO_HIGH
    assert violations[0].measured == 0.8


def test_file_size_cap_gate_upscaling() -> None:
    small_input = info(width=960, height=540, byte_size=100_000)
    ok = make_ctx(
        track=TRACK_UPSCALING,
        input_info=small_input,
        candidate_info=info(byte_size=800_000),
        upscale_factor=2,
        vmaf_primary=60.0,
        vmaf_secondary=60.0,
    )
    passed, violations = pipeline().run(ok)
    assert passed, violations  # exactly 8x the input is allowed

    over = make_ctx(
        track=TRACK_UPSCALING,
        input_info=small_input,
        candidate_info=info(byte_size=800_001),
        upscale_factor=2,
        vmaf_primary=60.0,
        vmaf_secondary=60.0,
    )
    passed, violations = pipeline().run(over)
    assert not passed
    assert violations[0].code == ReasonCode.FILE_SIZE_CAP_EXCEEDED

    # 4x factor gets the 20x cap
    ok4 = make_ctx(
        track=TRACK_UPSCALING,
        input_info=small_input,
        candidate_info=info(byte_size=2_000_000),
        upscale_factor=4,
        vmaf_primary=60.0,
        vmaf_secondary=60.0,
    )
    passed, _ = pipeline().run(ok4)
    assert passed


def test_vmaf_floor_gate_upscaling_at_half() -> None:
    # vmaf/100 < 0.5 -> fail; exactly 0.5 passes.
    at = make_ctx(
        track=TRACK_UPSCALING,
        vmaf_primary=50.0,
        vmaf_secondary=50.0,
        candidate_info=info(),
        upscale_factor=2,
    )
    passed, _ = pipeline().run(at)
    assert passed
    below = make_ctx(
        track=TRACK_UPSCALING,
        vmaf_primary=49.999,
        vmaf_secondary=49.999,
        candidate_info=info(),
        upscale_factor=2,
    )
    passed, violations = pipeline().run(below)
    assert not passed
    assert violations[0].code == ReasonCode.VMAF_BELOW_FLOOR
    assert violations[0].limit == 50.0


def test_vmaf_floor_gate_compression_uses_threshold_minus_band() -> None:
    below = make_ctx(vmaf_primary=84.9, vmaf_secondary=84.0)
    passed, violations = pipeline().run(below)
    assert not passed
    assert violations[0].code == ReasonCode.VMAF_BELOW_FLOOR
    at = make_ctx(vmaf_primary=85.0, vmaf_secondary=85.5)
    passed, _ = pipeline().run(at)
    assert passed


def test_vmaf_model_delta_gate() -> None:
    ok = make_ctx(vmaf_primary=95.0, vmaf_secondary=92.0)  # delta exactly 3.0
    passed, _ = pipeline().run(ok)
    assert passed
    bad = make_ctx(vmaf_primary=95.0, vmaf_secondary=91.9)
    passed, violations = pipeline().run(bad)
    assert not passed
    assert violations[0].code == ReasonCode.VMAF_MODEL_DELTA_EXCEEDED


def test_vmaf_model_delta_uses_miner_input_pair_not_scored_pristine_pair() -> None:
    # Pristine pair disagrees by 10 (would false-fail); miner-input pair agrees
    # by 2 and therefore passes at the unchanged 3-point threshold.
    ctx = make_ctx(
        vmaf_primary=95.0,
        vmaf_secondary=85.0,
        vmaf_delta_primary=92.0,
        vmaf_delta_secondary=90.0,
    )
    passed, violations = pipeline().run(ctx)
    assert passed and violations == []


# --- fail-closed: missing / non-finite metrics ---------------------------------


def test_missing_primary_vmaf_is_a_violation() -> None:
    ctx = make_ctx(vmaf_primary=None)
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert all(v.code == ReasonCode.METRIC_MISSING for v in violations)
    assert len(violations) == 2  # floor gate + model-delta gate both fail closed


def test_nan_and_inf_primary_vmaf_are_violations() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        ctx = make_ctx(vmaf_primary=bad)
        passed, violations = pipeline().run(ctx)
        assert not passed
        assert {v.code for v in violations} == {ReasonCode.METRIC_NON_FINITE}


def test_nan_secondary_vmaf_is_a_violation() -> None:
    ctx = make_ctx(vmaf_secondary=float("nan"))
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.METRIC_NON_FINITE


def test_secondary_vmaf_absent_fails_closed_by_default() -> None:
    ctx = make_ctx(vmaf_secondary=None)
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.METRIC_MISSING
    assert "secondary" in violations[0].detail
    assert ctx.skips == []


def test_secondary_vmaf_absent_with_flag_off_records_informational_skip() -> None:
    cfg = ScoringConfig(require_secondary_vmaf=False)
    ctx = make_ctx(config=cfg, vmaf_secondary=None)
    passed, violations = pipeline().run(ctx)
    assert passed, violations
    assert [s.gate for s in ctx.skips] == ["vmaf_model_delta"]
    assert "require_secondary_vmaf" in ctx.skips[0].detail
    # ...and the skip is persisted in the audit packet, not just the transient context
    item = compose_item_score(
        item_id="i-skip",
        challenge_id="c-skip",
        track=TRACK_COMPRESSION,
        gate_passed=passed,
        violations=violations,
        breakdown=None,
        config=cfg,
        skips=ctx.skips,
    )
    assert item.skips == ctx.skips


def test_absent_upscale_factor_fails_closed() -> None:
    ctx = make_ctx(
        track=TRACK_UPSCALING,
        vmaf_primary=60.0,
        vmaf_secondary=60.0,
        upscale_factor=None,
    )
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.UNSUPPORTED_SCALE_FACTOR


def test_unsupported_upscale_factor_fails_closed() -> None:
    # Supported factors are exactly the keys of file_size_caps ({2, 4} by default).
    ctx = make_ctx(
        track=TRACK_UPSCALING,
        vmaf_primary=60.0,
        vmaf_secondary=60.0,
        upscale_factor=3,
    )
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.UNSUPPORTED_SCALE_FACTOR
    assert violations[0].measured == 3.0


def test_non_finite_cap_limit_is_a_violation() -> None:
    # A non-finite cap can no longer be *configured* (ScoringConfig rejects it at
    # construction — see test_config), but the computed limit can still overflow to
    # inf from a finite cap times a huge input; the gate stays fail-closed on that.
    import pytest

    with pytest.raises(ValueError):
        ScoringConfig(file_size_caps={2: float("inf"), 4: 20.0})

    cfg = ScoringConfig(file_size_caps={2: 1e308, 4: 20.0})
    ctx = make_ctx(
        track=TRACK_UPSCALING,
        config=cfg,
        vmaf_primary=60.0,
        vmaf_secondary=60.0,
        upscale_factor=2,
    )
    assert not (1e308 * ctx.effective_input.byte_size < float("inf"))  # overflows
    passed, violations = pipeline().run(ctx)
    assert not passed
    assert violations[0].code == ReasonCode.METRIC_NON_FINITE


def test_perceptual_gates_backend_driven() -> None:
    checks = {
        "tone": PerceptualCheckResult(passed=False, measure=0.9, detail="tone shifted"),
        "chroma": PerceptualCheckResult(passed=False),
    }
    passed, violations = pipeline(checks).run(make_ctx())
    assert not passed
    codes = {v.code for v in violations}
    assert codes == {ReasonCode.TONE_MANIPULATION, ReasonCode.CHROMA_UV_MANIPULATION}


def test_pipeline_collects_all_violations_and_extra() -> None:
    from vidaio.scoring import ValidityViolation

    ctx = make_ctx(
        candidate_info=info(codec="theora", frame_count=10, byte_size=900_000),
        extra_violations=[
            ValidityViolation(code=ReasonCode.STREAM_PTS_INCONSISTENT, detail="x")
        ],
    )
    passed, violations = pipeline().run(ctx)
    assert not passed
    codes = [v.code for v in violations]
    assert codes[0] == ReasonCode.STREAM_PTS_INCONSISTENT  # extra first
    assert ReasonCode.ENCODING_NOT_ALLOWED in codes
    assert ReasonCode.FRAME_COUNT_MISMATCH in codes
    assert ReasonCode.COMPRESSION_RATE_TOO_HIGH in codes
