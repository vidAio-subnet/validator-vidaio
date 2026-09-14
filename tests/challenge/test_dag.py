import json
import random

import pytest

from vidaio.challenge import (
    DAG_VERSION,
    LAUNCH_UPSCALE_FACTORS,
    SUPPORTED_DAG_VERSIONS,
    OPERATOR_REGISTRY,
    TRACK_RULES,
    UPSCALE_FACTORS,
    DegradationDag,
    SourceVariant,
    build_dag,
    dag_rng_from_seed,
    source_ops,
    to_ffmpeg_plan,
    to_reference_plan,
    track_rules,
)
from vidaio.challenge.dag import (
    MAX_OPTIONAL_OPS,
    STAGE_ORDER,
)


def test_same_seed_identical_dag_digest_and_plan() -> None:
    a = build_dag("compression", random.Random(1234))
    b = build_dag("compression", random.Random(1234))
    assert a == b
    assert a.canonical_json() == b.canonical_json()
    assert a.canonical_digest() == b.canonical_digest()
    assert to_ffmpeg_plan(a, "in.mkv", "out.mp4") == to_ffmpeg_plan(
        b, "in.mkv", "out.mp4"
    )


def test_different_seeds_differ() -> None:
    a = build_dag("compression", random.Random(1))
    b = build_dag("compression", random.Random(2))
    assert a.canonical_digest() != b.canonical_digest()


def test_canonical_json_key_order_and_compactness() -> None:
    dag = build_dag("upscaling", random.Random(7))
    text = dag.canonical_json()

    def check_sorted(pairs):
        keys = [k for k, _ in pairs]
        assert keys == sorted(keys)  # every object serializes with sorted keys
        return dict(pairs)

    parsed = json.loads(text, object_pairs_hook=check_sorted)
    # round-trips byte-identically under the canonical dump rules
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == text


@pytest.mark.parametrize("track", ["compression", "upscaling"])
def test_operator_order_constraints(track: str) -> None:
    stage_index = {s: i for i, s in enumerate(STAGE_ORDER)}
    for seed in range(60):
        dag = build_dag(track, random.Random(seed))
        names = [op.op for op in dag.ops]
        # required operators present
        for req in TRACK_RULES[track].required:
            assert req in names
        # op count within version bounds
        n_optional = len(names) - len(TRACK_RULES[track].required)
        assert 0 <= n_optional <= min(MAX_OPTIONAL_OPS, len(TRACK_RULES[track].optional))
        assert len(set(names)) == len(names)  # no duplicate operators
        # stage order is monotone: capture -> edit -> delivery
        stages = [stage_index[type(op).stage] for op in dag.ops]
        assert stages == sorted(stages)
        # codec, when present, is the final operator (=> downscale precedes it)
        if "codec_compress" in names:
            assert names[-1] == "codec_compress"
        if track == "upscaling" and "codec_compress" in names:
            assert names.index("downscale") < names.index("codec_compress")


def test_upscaling_downscale_factor_is_discrete() -> None:
    """Upscaling-track Downscale must invert to exactly a supported scoring factor:
    scale_factor is 1/f for f in UPSCALE_FACTORS (file_size_caps keys), never
    continuous. Compression excludes Downscale entirely at launch because the
    pristine-reference VMAF threshold made severe draws unwinnable."""
    allowed = {round(1 / f, 4) for f in LAUNCH_UPSCALE_FACTORS}
    seen = set()
    for seed in range(80):
        dag = build_dag("upscaling", random.Random(seed))
        for op in dag.ops:
            if op.op == "downscale":
                assert op.scale_factor in allowed
                seen.add(op.scale_factor)
    assert seen == allowed
    # 4x remains supported/auditable for historical competition commitments,
    # but is not minted by the launch inference DAG until separately calibrated.
    assert set(LAUNCH_UPSCALE_FACTORS) < set(UPSCALE_FACTORS)


def test_launch_pools_are_explicitly_winnable_single_operator_tasks() -> None:
    """DAG v8 launch tasks: the source variant plus the calibrated invertible operation."""
    assert DAG_VERSION == 8
    assert TRACK_RULES["compression"].required == ("source_variant", "codec_compress")
    assert TRACK_RULES["compression"].optional == ()
    assert TRACK_RULES["upscaling"].required == ("source_variant", "downscale")
    assert TRACK_RULES["upscaling"].optional == ()
    for seed in range(1_000):
        compression = build_dag("compression", random.Random(seed))
        upscaling = build_dag("upscaling", random.Random(seed))
        assert [op.op for op in compression.ops] == ["source_variant", "codec_compress"]
        assert [op.op for op in upscaling.ops] == ["source_variant", "downscale"]
        for variant in (compression.ops[0], upscaling.ops[0]):
            assert isinstance(variant, SourceVariant)
            for frac in (variant.crop_left, variant.crop_right, variant.crop_top, variant.crop_bottom):
                assert 0.01 <= frac <= 0.03
            assert 0.94 <= variant.gamma <= 1.06
            assert 0.92 <= variant.saturation <= 1.08
            assert -4.0 <= variant.hue_deg <= 4.0
            assert 2.0 <= variant.noise_strength <= 5.0
            assert 1 <= variant.noise_seed < 2**31
        codec = compression.ops[1]
        assert codec.codec == "h264"
        assert codec.rate_mode == "crf"
        assert codec.crf in {8, 10, 12}
        assert codec.gop == 1
        assert (codec.chroma, codec.bit_depth) == ("420", 8)
        downscale = upscaling.ops[1]
        assert round(1 / downscale.scale_factor) in LAUNCH_UPSCALE_FACTORS


def test_upscale_factors_match_scoring_caps() -> None:
    from vidaio.scoring.config import ScoringConfig

    assert set(UPSCALE_FACTORS) == set(ScoringConfig().file_size_caps)


def test_dag_rng_is_derived_not_bare_seed() -> None:
    """dag_rng_from_seed must not reproduce the bare-seed MT stream."""
    seed = 987654321987654321
    derived = dag_rng_from_seed(seed)
    bare = random.Random(seed)
    assert [derived.getrandbits(64) for _ in range(4)] != [
        bare.getrandbits(64) for _ in range(4)
    ]
    # deterministic: same seed -> same derived stream
    assert dag_rng_from_seed(seed).getrandbits(64) == dag_rng_from_seed(
        seed
    ).getrandbits(64)


def test_launch_parameters_vary_across_seeds() -> None:
    digests = {
        build_dag("compression", random.Random(s)).canonical_digest()
        for s in range(40)
    }
    assert len(digests) == 40  # every seed draws its own source variant (v8)


def test_unknown_track_and_version_rejected() -> None:
    with pytest.raises(ValueError):
        build_dag("nope", random.Random(0))
    with pytest.raises(ValueError):
        build_dag("compression", random.Random(0), dag_version=DAG_VERSION + 1)


def test_registry_covers_spec_operators() -> None:
    # frame_drop stays registered (historical dag_version <= 2 documents must
    # deserialize) even though dag_version 3 excludes it from every pool.
    assert set(OPERATOR_REGISTRY) == {
        "source_variant",
        "downscale",
        "gaussian_blur",
        "noise",
        "codec_compress",
        "frame_drop",
        "tone_shift",
        "color_pipeline",
        "artifact_mask",
        "exposure",
        "motion_blur",
    }


def test_frame_drop_excluded_from_every_track_pool() -> None:
    """dag_version 3 P1 scope cut: FrameDrop shortens frame count/duration vs the
    reference, which the validity gates zero for HONEST miners, and no fps
    normalization can temporally re-align a decimated candidate for
    full-reference VMAF/PieAPP (vidaio/challenge/dag.py module docstring)."""
    from vidaio.challenge.dag import EXCLUDED_OPS

    assert "frame_drop" in EXCLUDED_OPS
    for rule in TRACK_RULES.values():
        assert "frame_drop" not in rule.required
        assert "frame_drop" not in rule.optional


@pytest.mark.parametrize("track", ["compression", "upscaling"])
def test_frame_drop_never_sampled(track: str) -> None:
    """Structural absence across a seed sweep: no produced DAG carries frame_drop."""
    for seed in range(300):
        dag = build_dag(track, random.Random(seed))
        assert all(op.op != "frame_drop" for op in dag.ops)


def test_ffmpeg_plan_snapshot_fixed_seed() -> None:
    """Full plan for seed 20260820 (compression). Any change here means the DAG's
    meaning changed and DAG_VERSION must be bumped. DAG_VERSION 8 pins the
    calibrated codec-only launch draw behind the source-variant stage."""
    dag = build_dag("compression", random.Random(20260820))
    assert dag.canonical_digest() == (
        "bce1bebe41cc905f24370c6e02dd43fa9b83ca219c5de6b7e55579a01c71bf2e"
    )
    assert [op.op for op in dag.ops] == ["source_variant", "codec_compress"]
    assert to_reference_plan(dag, "pristine.mkv", "reference.mkv") == [
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            "pristine.mkv",
            "-filter_complex",
            "[0:v]crop=trunc((iw-iw*0.0391)/2)*2:trunc((ih-ih*0.0516)/2)*2"
            ":trunc(iw*0.015/2)*2:trunc(ih*0.0284/2)*2,"
            "eq=gamma=0.9951:saturation=1.0404,hue=h=-0.9028,"
            "noise=alls=3.5841:allf=t:all_seed=154957849[out]",
            "-map",
            "[out]",
            "-c:v",
            "ffv1",
            "-an",
            "reference.mkv",
        ],
    ]
    plan = to_ffmpeg_plan(dag, "in.mkv", "out.mp4")
    assert plan == [
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            "in.mkv",
            "-c:v",
            "libx264",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "10",
            "-an",
            "out.mp4",
        ],
    ]


def test_dag_version_7_still_regenerates_byte_for_byte() -> None:
    """Commitments dispatched under v7 must keep verifying after the v8 roll: the v7
    pools (no source stage) stay buildable and reproduce the launch snapshot."""
    assert SUPPORTED_DAG_VERSIONS == (7, 8)
    assert track_rules(7)["compression"].required == ("codec_compress",)
    assert track_rules(7)["upscaling"].required == ("downscale",)
    dag = build_dag("compression", random.Random(20260820), dag_version=7)
    assert dag.dag_version == 7
    assert dag.canonical_digest() == (
        "c6958adc68ce911df89d05ee96f0ed92966e698188f9398711e0f6d8c07b0099"
    )
    assert [op.op for op in dag.ops] == ["codec_compress"]
    assert dag.ops[0].crf == 12
    assert to_reference_plan(dag, "pristine.mkv", "reference.mkv") == []
    with pytest.raises(ValueError):
        build_dag("compression", random.Random(0), dag_version=6)
    with pytest.raises(ValueError):
        track_rules(9)


def test_source_variant_is_reference_only_and_even_geometry() -> None:
    """The source stage never enters the miner-input plan; it renders the reference
    with even crop offsets/sizes (4:2:0 safe) and a seeded, reproducible grain."""
    for seed in range(50):
        dag = build_dag("compression", random.Random(seed))
        variant = dag.ops[0]
        assert isinstance(variant, SourceVariant)
        assert source_ops(dag) == [variant]
        input_plan = to_ffmpeg_plan(dag, "reference.mkv", "input.mkv")
        assert all("source_variant" not in " ".join(cmd) and "crop=" not in " ".join(cmd) for cmd in input_plan)
        assert input_plan[0][input_plan[0].index("-i") + 1] == "reference.mkv"
        ref_plan = to_reference_plan(dag, "pristine.mkv", "reference.mkv")
        assert len(ref_plan) == 1 and ref_plan[0][-1] == "reference.mkv"
        graph = ref_plan[0][ref_plan[0].index("-filter_complex") + 1]
        assert graph.count("trunc(") == 4 and "/2)*2" in graph
        assert f"all_seed={variant.noise_seed}" in graph
        assert ("hflip" in graph) == variant.flip
    # historical documents without a source stage keep the copy-is-reference path
    legacy = DegradationDag(dag_version=7, task_type="compression", ops=[build_dag("compression", random.Random(3)).ops[1]])
    assert to_reference_plan(legacy, "pristine.mkv", "reference.mkv") == []
    assert len(to_ffmpeg_plan(legacy, "reference.mkv", "input.mkv")) == 1


def test_artifact_mask_plan_snapshot() -> None:
    """Historical ArtifactMask realization remains deserializable after v7.

    The operator it replaced was a SILENT NO-OP: it wrote `x=iw*(x0+vx*t)` next
    to `t=fill`, and in ffmpeg's drawbox the `t` inside a coordinate expression
    is the box THICKNESS — so the box was positioned at the fill thickness, far
    off frame, and the stage's output was byte-identical (frame-md5) to no
    filter at all. drawbox has NO time or frame-number variable; crop and
    overlay do, which is why the box is now a darkened cut-out of the region it
    covers, moved by their real `t`.

    Two properties are load-bearing and asserted structurally below the
    snapshot: the drawbox/`t=fill` shape is gone for good, and the trajectory
    variable appears in filters that actually have one.
    """
    from vidaio.challenge.dag import ArtifactMask, DegradationDag

    mask = ArtifactMask(
        x0=0.2332, y0=0.5413, width_frac=0.1851, height_frac=0.1226,
        vx=0.041, vy=0.0483, opacity=0.9241,
    )
    dag = DegradationDag(dag_version=4, task_type="compression", ops=[mask])
    assert mask.model_dump(mode="json") == {
        "op": "artifact_mask",
        "x0": 0.2332,
        "y0": 0.5413,
        "width_frac": 0.1851,
        "height_frac": 0.1226,
        "vx": 0.041,
        "vy": 0.0483,
        "opacity": 0.9241,
    }
    assert to_ffmpeg_plan(dag, "in.mkv", "out.mkv")[0] == [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        "in.mkv",
        "-filter_complex",
        "[0:v]split[am_bg][am_dark];"
        "[am_dark]lutyuv=y='minval+(val-minval)*0.0759'"
        ":u='(minval+maxval)/2+(val-(minval+maxval)/2)*0.0759'"
        ":v='(minval+maxval)/2+(val-(minval+maxval)/2)*0.0759',"
        "crop=w=iw*0.1851:h=ih*0.1226"
        ":x='2*floor(min(max(iw*(0.2332+0.041*t),0),iw-ow)/2)'"
        ":y='2*floor(min(max(ih*(0.5413+0.0483*t),0),ih-oh)/2)'[am_box];"
        "[am_bg][am_box]overlay="
        "x='2*floor(min(max(W*(0.2332+0.041*t),0),W-w)/2)'"
        ":y='2*floor(min(max(H*(0.5413+0.0483*t),0),H-h)/2)':eval=frame[out]",
        "-map",
        "[out]",
        "-c:v",
        "ffv1",
        "-an",
        "out.mkv",
    ]


@pytest.mark.parametrize("seed", range(60))
def test_artifact_mask_never_uses_drawbox_again(seed: int) -> None:
    """The regression, structurally: no sampled mask may reach drawbox/t=fill."""
    from vidaio.challenge.dag import ArtifactMask

    mask = ArtifactMask.sample(random.Random(seed))
    graph = mask.filter_graph()
    assert "drawbox" not in graph and "t=fill" not in graph
    # the trajectory lives in crop and overlay, the two filters with a real `t`
    assert graph.count(f"{mask.vx}*t") == 2 and graph.count(f"{mask.vy}*t") == 2
    assert "overlay=" in graph and "crop=" in graph


def test_plan_chains_through_intermediates() -> None:
    dag = build_dag("compression", random.Random(11))
    plan = to_ffmpeg_plan(dag, "in.mkv", "out.mp4")
    assert len(plan) == len(dag.ops) - len(source_ops(dag))
    # every command reads the previous command's output; final writes out.mp4
    current = "in.mkv"
    for cmd in plan:
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-i") + 1] == current
        current = cmd[-1]
    assert current == "out.mp4"
