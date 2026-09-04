"""Shared foundation: config, logging, db, metrics, resilience.

Every service module imports from here and nowhere else in core. Module-specific
configuration models live in the module itself (e.g. vidaio.tokenomics.config) and are
parsed out of the shared raw config via `vidaio.core.config.section`.
"""

from vidaio.core.config import CoreConfig, load_raw_config, section
from vidaio.core.db import apply_migrations, connect, connect_read_only
from vidaio.core.logging import bound, get_logger, log_fields, setup_logging
from vidaio.core.metrics import HealthServer
from vidaio.core.resilience import RetriesExhausted, retry_async, with_timeout

__all__ = [
    "CoreConfig",
    "load_raw_config",
    "section",
    "connect",
    "connect_read_only",
    "apply_migrations",
    "setup_logging",
    "get_logger",
    "bound",
    "log_fields",
    "HealthServer",
    "retry_async",
    "with_timeout",
    "RetriesExhausted",
]
