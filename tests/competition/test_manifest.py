from datetime import timedelta, timezone

import pytest
from pydantic import ValidationError

from vidaio.competition import (
    CompetitionConfig,
    ManifestBoundsError,
    validate_against_config,
)

from support import BASELINE, END, ENROLL_DEADLINE, FINALIZATION, START, build_manifest


def test_valid_manifest_and_digest_shape() -> None:
    m = build_manifest()
    digest = m.manifest_digest()
    assert len(digest) == 64
    assert int(digest, 16) >= 0  # hex


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_time": ENROLL_DEADLINE, "enrollment_deadline": START},  # start after enroll
        {"start_time": START, "enrollment_deadline": START},  # start == enroll deadline
        {"enrollment_deadline": FINALIZATION + timedelta(minutes=1)},  # enroll after finalize
        {"finalization_time": END, "end_time": FINALIZATION},  # finalize after end
        {"finalization_time": END},  # finalize == end
    ],
)
def test_times_must_be_ordered(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        build_manifest(**overrides)


def test_naive_datetimes_rejected() -> None:
    with pytest.raises(ValidationError):
        build_manifest(start_time=START.replace(tzinfo=None))


@pytest.mark.parametrize(
    "factors",
    [
        {"quality": 0.6, "cost_efficiency": 0.1, "length_coverage": 0.4},  # sums to 1.1
        {"quality": 0.5, "cost_efficiency": 0.0, "length_coverage": 0.4},  # sums to 0.9
        {"quality": 0.6, "length_coverage": 0.4},  # missing key
        {"quality": 0.6, "cost_efficiency": 0.0, "length_coverage": 0.4, "extra": 0.0},
    ],
)
def test_scoring_factors_validation(factors: dict) -> None:
    with pytest.raises(ValidationError):
        build_manifest(scoring_factors=factors)


@pytest.mark.parametrize(
    "bad_key,value",
    [
        ("hotkey", "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYJQX"),
        ("payout_hotkey", "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repo5EYJQX"),
        ("payout_address", "0xabc"),
        ("reward_share", 0.5),
        ("wallet", "w"),
        ("coldkey", "ck"),
    ],
)
def test_baseline_payout_and_identity_fields_forbidden(bad_key: str, value: object) -> None:
    baseline = dict(BASELINE)
    baseline[bad_key] = value
    with pytest.raises(ValidationError, match="archived baseline is not a participant"):
        build_manifest(baseline=baseline)


def test_baseline_unknown_field_forbidden() -> None:
    baseline = dict(BASELINE)
    baseline["notes"] = "x"
    with pytest.raises(ValidationError):
        build_manifest(baseline=baseline)


def test_baseline_valid_block_accepted() -> None:
    m = build_manifest(baseline=BASELINE)
    assert m.baseline is not None
    assert m.baseline.tree_sha == BASELINE["tree_sha"]
    # The schema has no payout/hotkey attribute at all on the baseline block.
    assert not hasattr(m.baseline, "hotkey")


def test_digest_stable_across_timezone_representation() -> None:
    plus2 = timezone(timedelta(hours=2))
    a = build_manifest()
    b = build_manifest(
        start_time=START.astimezone(plus2),
        enrollment_deadline=ENROLL_DEADLINE.astimezone(plus2),
        finalization_time=FINALIZATION.astimezone(plus2),
        end_time=END.astimezone(plus2),
    )
    assert a.manifest_digest() == b.manifest_digest()


def test_digest_changes_when_content_changes() -> None:
    assert build_manifest().manifest_digest() != build_manifest(vmaf_threshold=91.0).manifest_digest()
    assert build_manifest().manifest_digest() != build_manifest(baseline=BASELINE).manifest_digest()


@pytest.mark.parametrize(
    "overrides",
    [
        {"sealed_vmaf_variants": []},
        {"sealed_vmaf_variants": [85.0, 101.0]},
        {"allowed_gpus": []},
        {"allowed_gpus": ["L4", "L4"]},
        {"evaluation_batch_size": {"min": 5, "max": 1}},
        {"evaluation_batch_size": {"min": 0, "max": 5}},
        {"scoring_seed_commitment": "not-a-hash"},
        {"scoring_seed_commitment": "A" * 64},  # uppercase not canonical
        {"container_size_limit_gb": 0},
        {"minimum_alpha_stake": -1},
        {"competition_id": "Bad_ID!"},
        # Changing only the track omits the upscaling manifest's required item
        # commitment/binding fields; complete upscaling manifests are covered in
        # test_upscaling_items.py.
        {"track": "upscaling"},
        {"unknown_field": 1},
    ],
)
def test_field_validation(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        build_manifest(**overrides)


def test_manifest_is_frozen() -> None:
    m = build_manifest()
    with pytest.raises(ValidationError):
        m.vmaf_threshold = 50.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"vmaf_threshold": 30.0},  # below configured floor of 50
        {"sealed_vmaf_variants": [40.0, 89.0]},
        {"container_size_limit_gb": 26.0},  # over the 25 GB (unattested) cap
        {"evaluation_batch_size": {"min": 1, "max": 32}},
    ],
)
def test_config_bounds_enforced(overrides: dict) -> None:
    cfg = CompetitionConfig()
    manifest = build_manifest(**overrides)
    with pytest.raises(ManifestBoundsError):
        validate_against_config(manifest, cfg)


def test_config_bounds_pass_for_comp01_defaults() -> None:
    validate_against_config(build_manifest(), CompetitionConfig())
