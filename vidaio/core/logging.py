"""Structured JSON logging with contextvar-bound fields (round/contender/batch ids).

Logs go to stdout as one JSON object per line — shipped to a log store, never a
gitignored file. Secrets and PATs must never be passed as field values.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Iterator

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "vidaio_log_context", default={}
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context.get())
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers[:] = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def restore_levels(*, keep_prefixes: tuple[str, ...] = ("bittensor",)) -> int:
    """Undo third-party level clamps so :func:`setup_logging`'s root level applies again.

    Importing the bittensor SDK "enables default logging" by raising every logger that
    already exists to CRITICAL (level 50), keeping only its own. Any service logger
    created before that import — the scoring worker's, for one — then emits nothing at
    all. Reset every non-SDK logger with an explicit level back to NOTSET and return
    how many were reset.
    """
    reset = 0
    for name, obj in list(logging.root.manager.loggerDict.items()):
        if not isinstance(obj, logging.Logger) or name.startswith(keep_prefixes):
            continue
        if obj.level != logging.NOTSET:
            obj.setLevel(logging.NOTSET)
            reset += 1
    return reset


@contextlib.contextmanager
def bound(**fields: Any) -> Iterator[None]:
    """Bind structured fields to every log line emitted inside the block."""
    merged = dict(_context.get())
    merged.update(fields)
    token = _context.set(merged)
    try:
        yield
    finally:
        _context.reset(token)


def log_fields(**fields: Any) -> dict[str, Any]:
    """Helper for per-call fields: logger.info("msg", extra=log_fields(uid=5))."""
    return {"fields": fields}
