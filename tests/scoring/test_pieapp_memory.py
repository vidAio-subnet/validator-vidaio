"""Bound PieAPP feature workspace while retaining exact full-matrix scoring."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from vidaio.scoring import backends_real as real


@pytest.fixture
def native_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.backends.mkldnn, "enabled", False)
    with torch.backends.nnpack.flags(enabled=False):
        yield torch


@pytest.mark.parametrize("count", [1, 64, 65, 129])
def test_feature_batches_preserve_order_and_release_chunk_outputs(native_torch, count):
    torch = native_torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.batches = []
            self.outputs = []
            self.differences = []

        def forward(self, patches):
            # Completed chunks must be released before the next model call.
            assert all(ref() is None for ref in self.outputs)
            self.batches.append(patches[:, 0].tolist())
            features = patches * self.scale
            weights = patches[:, :1].clone()
            self.outputs.extend((weakref.ref(features), weakref.ref(weights)))
            return features, weights

        def compute_difference(self, features, weights):
            self.differences.append((features, weights))
            return features, weights

    original = Model()
    bounded = real._bounded_pieapp_model(original, torch).eval()
    assert tuple(bounded.children()) == (original,)
    assert [id(value) for value in bounded.parameters()] == [id(original.scale)]
    assert not original.training
    bounded.to(dtype=torch.float64)
    assert original.scale.dtype == torch.float64

    patches = torch.arange(count * 3, dtype=torch.float64).reshape(count, 3)
    with torch.inference_mode():
        features, weights = bounded(patches)
    assert torch.equal(features, patches * 2)
    assert torch.equal(weights, patches[:, :1])
    assert features.is_contiguous() and weights.is_contiguous()
    assert max(map(len, original.batches)) <= 64
    assert [value for batch in original.batches for value in batch] == patches[:, 0].tolist()
    if count > 64:
        assert all(ref() is None for ref in original.outputs)

    # Score layers receive the complete matrices by identity, without splitting.
    result = bounded.compute_difference(features, weights)
    assert result[0] is features and result[1] is weights
    assert original.differences[0][0] is features
    assert original.differences[0][1] is weights


@pytest.mark.parametrize("mkldnn,nnpack", [(True, False), (False, True), (True, True)])
def test_accelerated_cpu_keeps_original_model_and_full_batch(monkeypatch, mkldnn, nnpack):
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batches = []

        def forward(self, patches):
            self.batches.append(tuple(patches.shape))
            return patches, patches

    monkeypatch.setattr(torch.backends.mkldnn, "enabled", mkldnn)
    with torch.backends.nnpack.flags(enabled=nnpack):
        original = Model()
        selected = real._bounded_pieapp_model(original, torch)
        assert selected is original
        patches = torch.zeros((129, 3))
        result = selected(patches)
        assert original.batches == [(129, 3)]
        assert result[0] is patches and result[1] is patches
        assert torch.backends.mkldnn.enabled is mkldnn
        assert torch._C._get_nnpack_enabled() is nnpack


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runtime_installs_feature_wrapper_only_on_cpu(monkeypatch, device):
    torch = pytest.importorskip("torch")
    piq = pytest.importorskip("piq")
    pytest.importorskip("cv2")
    from vidaio.scoring_worker import runtime_identity

    original = object()
    wrapped = object()
    metric = SimpleNamespace(model=original)
    metric.to = lambda target: metric
    metric.eval = lambda: metric
    wrapped_models = []

    def wrap(model, torch_module):
        wrapped_models.append((model, torch_module))
        return wrapped

    monkeypatch.setattr(piq, "PieAPP", lambda **kwargs: metric)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(runtime_identity, "canonical_release_marker_present", lambda: False)
    monkeypatch.setattr(real._PiqPieAppRuntime, "_ensure_weights", lambda *args: None)
    monkeypatch.setattr(real, "_bounded_pieapp_model", wrap)

    runtime = real._PiqPieAppRuntime(device)
    assert runtime._metric.model is (wrapped if device == "cpu" else original)
    assert wrapped_models == ([(original, torch)] if device == "cpu" else [])


@pytest.fixture(scope="module", params=["native", "configured"])
def cached_runtime(request):
    torch = pytest.importorskip("torch")
    pytest.importorskip("piq")
    pytest.importorskip("cv2")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / real.PIEAPP_WEIGHTS_FILENAME
    if not checkpoint.is_file():
        pytest.skip("pinned PieAPP weights must already be cached; test never downloads")

    previous_threads = torch.get_num_threads()
    previous_mkldnn = torch.backends.mkldnn.enabled
    try:
        # Exercise both the established native CPU convolution policy and the
        # configured host policy, including accelerated development runtimes.
        native = request.param == "native"
        if native:
            torch.backends.mkldnn.enabled = False
        with torch.backends.nnpack.flags(enabled=False) if native else nullcontext():
            runtime = real._PiqPieAppRuntime("cpu")
            optimized = not (torch.backends.mkldnn.enabled or torch._C._get_nnpack_enabled())
            assert hasattr(runtime._metric.model, "_model") is optimized
            original = getattr(runtime._metric.model, "_model", runtime._metric.model)
            yield torch, runtime, original
    finally:
        torch.backends.mkldnn.enabled = previous_mkldnn
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize(
    "seed,height,width", [(11, 64, 64), (23, 91, 118), (37, 128, 128), (41, 145, 145)]
)
def test_cached_model_preserves_feature_weight_score_bits_and_fc_shapes(
    cached_runtime, monkeypatch, seed, height, width
):
    torch, runtime, original = cached_runtime
    np = pytest.importorskip("numpy")
    # A smaller test-only batch exercises multiple chunks and uneven tails
    # with the actual pinned model while keeping peak RSS below about 1 GB.
    monkeypatch.setattr(real, "_PIEAPP_PATCH_BATCH_SIZE", 4)
    bounded = real._bounded_pieapp_model(original, torch)
    metric = runtime._metric
    rng = np.random.default_rng(seed)
    distorted = runtime._tensor(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    reference = runtime._tensor(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
    fc_shapes = []
    handles = []

    def record_shape(name):
        def record(module, inputs):
            fc_shapes.append((name, tuple(inputs[0].shape), inputs[0].stride()))

        return record

    for name in ("fc1_score", "fc2_score", "ref_score_subtract", "fc1_weight", "fc2_weight"):
        handles.append(getattr(original, name).register_forward_pre_hook(record_shape(name)))

    def assert_bits_equal(expected, actual):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype == torch.float32
        assert torch.equal(expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32))

    try:
        with torch.inference_mode():
            monkeypatch.setattr(metric, "model", original)
            expected_features = [metric.get_features(frame) for frame in (distorted, reference)]
            expected_score = metric(distorted, reference)
            expected_fc_shapes = list(fc_shapes)
            fc_shapes.clear()

            monkeypatch.setattr(metric, "model", bounded)
            actual_features = [metric.get_features(frame) for frame in (distorted, reference)]
            actual_score = metric(distorted, reference)

        for expected, actual in zip(expected_features, actual_features, strict=True):
            assert_bits_equal(expected[0], actual[0])
            assert_bits_equal(expected[1], actual[1])
        assert_bits_equal(expected_score, actual_score)
        assert fc_shapes == expected_fc_shapes
        patches = ((height - 64) // 27 + 1) * ((width - 64) // 27 + 1)
        assert len(fc_shapes) == 5
        assert all(shape[0] == patches for _, shape, _ in fc_shapes)
    finally:
        for handle in handles:
            handle.remove()
