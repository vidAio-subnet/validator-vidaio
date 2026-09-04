"""Audit-store configuration (config section: ``audit``).

Credentials and holdout-encryption keys are never stored in config; only their
environment-variable names are. Both AWS-style S3 and Hippius' S3-compatible
gateway use the same verified content-addressed storage implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    backend: Literal["local", "s3", "hippius"] = "local"
    local_root: Path = Path("./data/audit")
    #: Explicit opt-in for a PLAINTEXT holdout at rest. make_store() refuses to
    #: build a store whose reference_original envelope is the no-op
    #: PassthroughEnvelope unless this is True (dev/test only) — otherwise the
    #: "sealed" holdout ground truth would be readable by anyone with storage
    #: access mid-competition.
    allow_plaintext_holdout: bool = False
    #: 32-byte AES-GCM key (hex or base64) used when plaintext is not explicitly
    #: allowed. Only the environment-variable NAME is stored here.
    holdout_key_env: str = "VIDAIO_AUDIT_HOLDOUT_KEY"
    # Env-var NAMES only (values are read at transport time, never persisted).
    hippius_endpoint_env: str = "VIDAIO_HIPPIUS_ENDPOINT"
    hippius_access_key_env: str = "VIDAIO_HIPPIUS_ACCESS_KEY_ID"
    hippius_secret_key_env: str = "VIDAIO_HIPPIUS_SECRET_ACCESS_KEY"
    hippius_bucket: str = ""
    hippius_region: str = "decentralized"
    hippius_prefix: str = ""
    # S3 backend (the production object store — the project design record §1). The bucket
    # + region + optional key prefix are plain config; CREDENTIALS are never stored
    # here — only the NAMES of the env vars the boto3 transport reads them from.
    # Requires the optional 'storage' extra:  uv pip install -e '.[storage]'
    s3_bucket: str = ""
    s3_region: str = ""
    s3_prefix: str = ""  # optional key prefix within the bucket
    s3_endpoint_url_env: str = "VIDAIO_S3_ENDPOINT"  # custom endpoint (MinIO/S3-compat)
    s3_access_key_env: str = "VIDAIO_S3_ACCESS_KEY_ID"
    s3_secret_key_env: str = "VIDAIO_S3_SECRET_ACCESS_KEY"
    # 0 = retain forever. Auditability is the point: only shrink this once
    # commitments referencing the artifacts have themselves expired.
    retention_days: int = Field(default=0, ge=0)
