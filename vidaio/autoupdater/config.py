"""Autoupdater configuration — schema for the `autoupdater:` section of config.

The autoupdater ships SERVICE CODE versions to your validator/miner fleet
(design spec §20 CI/CD row) by watching a version source and, on a change, running the
deployment's own update command. It NEVER edits code itself, and it ships CLOSED twice over:

  * `update_command` defaults to `[]` — REPORT-ONLY mode. A version change is
    logged and counted, and /health flips `update_pending`, but nothing runs.
  * `require_ci_pass` defaults to true — even with a command configured, a
    trigger is refused unless the CI gate (the release CI gate (development tree)) has recorded a pass
    FOR THAT EXACT VERSION in the `ci_pass_marker` file.

Champions are NOT shipped through this service — they travel through the model
registry's PromotionPipeline (see vidaio/autoupdater/README.md for why the two
paths are deliberately separate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AutoupdaterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: Where new versions appear. "file" reads `version_file`; "http" GETs
    #: `version_url` (a plain-text body whose first line is the version string —
    #: the later fleet-push endpoint). Both are implemented and tested now so
    #: switching is a config change, not a code change.
    version_source: Literal["file", "http"] = "file"

    #: The version FILE. Under `version_source: file` it is the source itself;
    #: under "http" it still matters: it names the version THIS deployment is
    #: running, which is the startup baseline when no state file exists yet
    #: (the "never trigger on first sight" rule). Default: the repo VERSION.
    version_file: Path = Path("./VERSION")

    #: GET endpoint returning the current fleet version as its response body
    #: (first line wins, whitespace stripped). Required when version_source is
    #: "http"; ignored under "file".
    version_url: str = ""

    #: Timeout for one version_url GET.
    http_timeout_seconds: float = Field(default=10.0, gt=0)

    #: How often the version source is polled.
    poll_seconds: float = Field(default=60.0, gt=0)

    #: The deployment's OWN update mechanism as an argv list (e.g.
    #: ["/usr/local/bin/vidaio-update.sh"] — typically git pull + service
    #: restart). Run via subprocess with a timeout and captured output; the
    #: target version is exported as $VIDAIO_AUTOUPDATER_TARGET_VERSION. The
    #: EMPTY default is REPORT-ONLY mode: changes are logged/counted, never
    #: acted on.
    update_command: list[str] = Field(default_factory=list)

    #: Hard wall-clock budget for one update_command run.
    update_timeout_seconds: float = Field(default=600.0, gt=0)

    #: Bounded retry envelope for a failing update command. After the budget is
    #: spent the failure becomes a PERSISTENT unhealthy signal (`update_failed`
    #: on /health) and that version is not retried again.
    update_retry_attempts: int = Field(default=3, ge=1)
    update_retry_delay_seconds: float = Field(default=5.0, ge=0)

    #: Refuse to trigger an update unless `ci_pass_marker` names the NEW version
    #: and binds the full source digest, shipped-runtime digest and runtime
    #: manifest bytes. the release CI gate (development tree) writes it after a FULL green run only.
    require_ci_pass: bool = True

    #: The marker the CI gate writes: first line = the version it passed on,
    #: followed by source-sha256, runtime-sha256 and manifest-sha256.
    ci_pass_marker: Path = Path("./data/ci-pass")

    #: Root whose release inputs must match the content digest recorded by CI.
    #: For an active image deployment this is a separately staged, read-only
    #: target artifact tree; the activation command switches to it only after
    #: its exact runtime manifest and CI marker verify. Report-only/checkouts may
    #: keep the default current tree.
    source_root: Path = Path(".")

    #: Generated immutable identity for the staged target. Relative paths are
    #: resolved below ``source_root`` so one atomic staging tree carries its
    #: VERSION, manifest, CI marker and runtime bytes together.
    runtime_manifest_file: Path = Path("runtime-release-manifest.json")

    #: A version ORDERED BELOW the current one (see the ordering rule in
    #: vidaio/autoupdater/service.py) is refused with a CRITICAL log unless this
    #: is explicitly set — a rollback must be a human decision.
    allow_downgrade: bool = False

    #: Small JSON state file persisting the last applied version, so a restart
    #: does not re-trigger the update it just applied.
    state_file: Path = Path("./data/autoupdater-state.json")

    #: Metrics/health port (service port map: vidaio/services/protocol.py).
    metrics_port: int = 9110

    def staged_path(self, value: Path) -> Path:
        """Resolve an artifact-owned path beneath ``source_root`` when relative."""
        return value if value.is_absolute() else self.source_root / value

    @property
    def resolved_ci_pass_marker(self) -> Path:
        return self.staged_path(self.ci_pass_marker)

    @property
    def resolved_runtime_manifest_file(self) -> Path:
        return self.staged_path(self.runtime_manifest_file)

    @model_validator(mode="after")
    def _http_source_needs_a_url(self) -> "AutoupdaterConfig":
        if self.version_source == "http" and not self.version_url.strip():
            raise ValueError(
                "autoupdater.version_source is 'http' but version_url is empty —"
                " there is nothing to poll"
            )
        return self
