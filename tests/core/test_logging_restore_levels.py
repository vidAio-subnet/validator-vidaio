"""restore_levels undoes the SDK import's logger clamp without touching the SDK's own loggers."""
import logging

from vidaio.core.logging import restore_levels


def test_restore_levels_resets_clamped_service_loggers_only():
    ours = logging.getLogger("test-scoring-worker-clamped")
    sdk = logging.getLogger("bittensor.test-clamp")
    ours.setLevel(logging.CRITICAL)
    sdk.setLevel(logging.CRITICAL)
    try:
        assert restore_levels() >= 1
        assert ours.level == logging.NOTSET
        assert sdk.level == logging.CRITICAL
    finally:
        ours.setLevel(logging.NOTSET)
        sdk.setLevel(logging.NOTSET)
