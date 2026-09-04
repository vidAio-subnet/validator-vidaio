"""Service layer: long-running processes composed from the foundation modules.

Every service extends BaseService (config section, JSON logging, /health+/metrics,
graceful shutdown) and reaches the chain only through vidaio.chain.ChainAdapter.
"""

from vidaio.services.base import (
    FATAL_EXIT_CODE,
    BaseService,
    FatalServiceError,
    run_service,
)

__all__ = ["BaseService", "FatalServiceError", "FATAL_EXIT_CODE", "run_service"]
