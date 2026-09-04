"""Canonical serialization + digest helpers shared across the audit module.

Determinism contract: canonical JSON is UTF-8, sorted keys, no whitespace,
non-ASCII preserved, NaN/Infinity rejected. Every digest, bundle hash, merkle
leaf, and on-chain commitment in this module is computed over bytes produced
here, so any change to this contract invalidates all recorded digests — it is
versioned through the commitment payload domain tag in commitments.py.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize obj to canonical JSON bytes (deterministic for equal values)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
