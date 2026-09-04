"""Autoupdater — ships service-code VERSION updates to your validator/miner fleet.

Watches a version source (repo VERSION file now, HTTP later) and, gated on a
recorded CI pass, runs the deployment's own update command. Report-only by
default; never edits code itself. Champions are NOT shipped here — they go
through the registry's PromotionPipeline (see README.md).
"""

from vidaio.autoupdater.config import AutoupdaterConfig
from vidaio.autoupdater.service import (
    TARGET_RUNTIME_DIGEST_ENV,
    TARGET_SOURCE_DIGEST_ENV,
    TARGET_STAGED_ROOT_ENV,
    TARGET_VERSION_ENV,
    Autoupdater,
    VersionSourceError,
    compare_versions,
    version_key,
)

__all__ = [
    "Autoupdater",
    "AutoupdaterConfig",
    "TARGET_RUNTIME_DIGEST_ENV",
    "TARGET_SOURCE_DIGEST_ENV",
    "TARGET_STAGED_ROOT_ENV",
    "TARGET_VERSION_ENV",
    "VersionSourceError",
    "compare_versions",
    "version_key",
]
