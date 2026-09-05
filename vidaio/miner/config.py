"""Reference-miner configuration (config section: `miner`).

Encode levers (the honest defaults, chosen against the scoring gates):

  compress_crf 22 / preset medium — x264 CRF mode. This is the shipped launch
  baseline calibrated on real footage against the production VMAF=90 threshold
  and <0.80 miner-input size gate. Deployment-specific contender variants do not
  change this release default. `+faststart` keeps outputs streamable.
  It applies when the task names no codec/tier (every synthetic round). Organic
  tasks carry the old contract's `target_codec` (h264 | hevc | av1 | vp9),
  `codec_mode` (CRF | VBR + `target_bitrate`) and quality tier
  (`compression_type` High | Medium | Low); `vidaio.miner.encoding` maps those to
  the encoder and a per-codec tier CRF, and refuses codecs it cannot honour.

  upscale_crf 16 / preset medium — near-transparent quality for the upscaled
  output: the upscaling track is scored on VMAF against the pristine reference
  with per-factor file-size CAPS (not a shrink gate), so the miner spends bits on
  fidelity, capped by sanity.

Ingress bounds (the miner endpoint is PUBLIC — anyone who can reach the port can
make it spend CPU and disk):

  max_concurrent_tasks — how many tasks may run at once. Past it the miner
  answers 429 `busy` instead of queueing unbounded ffmpeg jobs; a validator that
  gets a 429 scores the round absent, which is the honest outcome for a miner
  that is over capacity.

  max_input_bytes / max_output_bytes — refuse an upload before work once its
  streamed bytes cross the input cap, and discard a backend output that crosses
  the output cap before it can be served back to a caller.

  artifact_ingress_timeout_seconds — server-owned wall-clock cap for receiving
  and durably staging the complete streamed request body. It cannot be enlarged
  by caller metadata, so slow clients cannot hold every task slot indefinitely.

  artifact_hotkey / replay settings — protocol-v2 responses are signed by this
  miner identity, while requests are accepted only from a fresh, currently
  registered validator signature. The timestamp window and bounded nonce cache
  make a captured request single-use. Unsigned v1 is an explicit dev-only opt-in.

  task_dir_ttl_seconds / retain_task_dirs — remote byte-stream responses delete
  their task dir after the response is sent. The TTL is the crash fallback and,
  only when the explicit legacy-path opt-in is enabled for local tests, the grace
  window for its JSON path response. Failed tasks are deleted immediately.
  retain_task_dirs: true disables all sweeping (local debugging only).

  api_token — optional shared secret for a controlled/reference miner. None (the
  default) leaves the bounded streamed endpoint open for permissionless testnet.
  When set, every task call must present it in `X-Miner-Token` or get a 401.

  enable_legacy_path_routes — local-test compatibility switch for the old JSON
  endpoints that dereference caller-supplied filesystem paths. It defaults false
  and production rejects it. The canonical streamed endpoint never dereferences
  a caller path.

TaskWarrant (`warrant_track`): ONE POOL PER MINER IDENTITY. The reference miner
binary ships both track backends and will process a task for either, but the
instance declares the single pool it competes in; the validator's GET /warrant
probe reads that declaration and buckets every score for this hotkey there. A
miner that wants to earn in both pools runs two identities (two hotkeys, two
endpoints). The value is validated against the known tracks here, because the
validator never defaults an unclassified miner into a track — an unknown warrant
answer means the miner is skipped for the round entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from vidaio.miner.remote_gpu import SUPPORTED_GPU_VARIANTS
from vidaio.scoring.config import TRACK_COMPRESSION, TRACK_UPSCALING

#: Pools a miner identity may declare (the scoring module's tracks).
KNOWN_TRACKS = (TRACK_COMPRESSION, TRACK_UPSCALING)


class MinerConfig(BaseModel):
    # HTTP API — see vidaio.services.protocol port table.
    http_host: str = "0.0.0.0"
    http_port: int = 8300
    metrics_port: int = 9106
    # Task outputs land in work_dir/<task_id>/output.mp4.
    work_dir: Path = Path("./data/miner")
    ffmpeg_path: str = "ffmpeg"
    # Hard wall-clock bound per ffmpeg subprocess (the request deadline may bound
    # it tighter per task).
    ffmpeg_timeout_seconds: float = Field(240.0, gt=0)
    #: ``ffmpeg`` is the self-contained CPU reference miner. ``remote_gpu`` keeps
    #: the signed public miner protocol on this process and delegates only the
    #: media transform to an authenticated GPU worker (the Modal launch shape).
    #: It is deliberately opt-in: no default configuration can spend cloud GPU.
    backend_mode: Literal["ffmpeg", "remote_gpu"] = "ffmpeg"
    #: HTTPS base URL of the dedicated GPU worker. The backend appends
    #: ``/process`` and the live preflight probes ``/healthz``. Query strings,
    #: fragments and embedded credentials are refused so bearer material can
    #: never leak through redirects or URL logging.
    remote_gpu_url: str | None = None
    #: Application bearer shared only by this miner ingress and its fresh Modal
    #: worker. SecretStr keeps config/model reprs from printing the credential.
    remote_gpu_auth_token: SecretStr | None = None
    #: Named solution profile. Deploy distinct hotkeys with different profiles
    #: to make score ranking and output-digest dedup observable on testnet.
    remote_gpu_solution_variant: str = Field(
        "balanced", pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$"
    )
    #: Separate connect cap; the request's signed deadline remains the tighter
    #: whole-call bound and is also sent to the worker.
    remote_gpu_connect_timeout_seconds: float = Field(10.0, gt=0, le=60.0)
    remote_gpu_allow_cpu_fallback: bool = False
    # Compression-track encode (see module docstring).
    compress_crf: int = Field(22, ge=0, le=51)
    compress_preset: str = "medium"
    # Upscaling-track encode.
    upscale_crf: int = Field(16, ge=0, le=51)
    upscale_preset: str = "medium"
    # -- ingress bounds (module docstring) --------------------------------------
    #: Tasks allowed to run concurrently; past it the endpoint answers 429 `busy`.
    max_concurrent_tasks: int = Field(2, ge=1)
    #: Largest accepted input; bigger inputs are refused (422) before any work.
    max_input_bytes: int = Field(2 * 1024 * 1024 * 1024, gt=0)
    #: Server-owned wall-clock cap for receiving + fsyncing a remote upload. The
    #: hard upper bound prevents a production configuration from restoring an
    #: effectively unbounded public slow-upload seam.
    artifact_ingress_timeout_seconds: float = Field(60.0, gt=0, le=300.0)
    #: The serving miner ss58 identity. V2 request intent and every response
    #: signature must bind to it; production also matches it to the loaded wallet.
    artifact_hotkey: str = ""
    #: Deprecated unsigned artifact v1, for isolated report/dev fixtures only.
    allow_unsigned_artifact_v1: bool = False
    #: Signed-request anti-replay window and tolerated positive clock skew.
    artifact_request_max_age_seconds: float = Field(120.0, gt=0, le=900.0)
    artifact_request_future_skew_seconds: float = Field(5.0, ge=0, le=60.0)
    #: Hard memory bound. Live entries are never evicted (that would permit replay);
    #: the endpoint returns 503 until one expires if this capacity is exhausted.
    artifact_replay_cache_entries: int = Field(10_000, ge=1, le=1_000_000)
    #: Isolation quota: one registered validator cannot consume the global cache.
    artifact_replay_cache_entries_per_validator: int = Field(256, ge=1, le=10_000)
    #: A metagraph snapshot older than this cannot prove a current validator.
    artifact_validator_snapshot_max_age_seconds: float = Field(300.0, gt=0, le=900.0)
    #: Largest output the backend may publish/stream. Checked while hashing,
    #: before response headers are emitted.
    max_output_bytes: int = Field(4 * 1024 * 1024 * 1024, gt=0)
    #: Crash cleanup plus grace for the deprecated JSON path response. Canonical
    #: remote responses delete their task dir after the output stream is sent.
    task_dir_ttl_seconds: float = Field(900.0, gt=0)
    #: How often the reaper sweeps work_dir (it also sweeps once at startup).
    task_sweep_interval_seconds: float = Field(60.0, gt=0)
    #: Debugging escape hatch: keep every task dir forever (no sweeping at all).
    retain_task_dirs: bool = False
    #: Optional shared secret required in the X-Miner-Token header. None = open
    #: permissionless endpoint; use only as a controlled reference-miner secret.
    api_token: str | None = None
    #: Explicit local-test-only opt-in for POST /v1/task and /task. These legacy
    #: routes dereference caller-supplied paths and must never be public.
    enable_legacy_path_routes: bool = False
    # The pool THIS identity competes in, served by GET /warrant (module docstring).
    warrant_track: str = TRACK_COMPRESSION

    @field_validator("warrant_track")
    @classmethod
    def _known_track(cls, value: str) -> str:
        if value not in KNOWN_TRACKS:
            raise ValueError(
                f"warrant_track {value!r} is not a known pool; choose one of"
                f" {list(KNOWN_TRACKS)} (one pool per miner identity)"
            )
        return value

    @model_validator(mode="after")
    def _per_validator_replay_quota_fits_global_cache(self) -> MinerConfig:
        if (
            self.artifact_replay_cache_entries_per_validator
            > self.artifact_replay_cache_entries
        ):
            raise ValueError(
                "artifact_replay_cache_entries_per_validator must be <= "
                "artifact_replay_cache_entries"
            )
        if self.backend_mode == "remote_gpu":
            if not self.remote_gpu_url:
                raise ValueError(
                    "miner.remote_gpu_url is required when backend_mode=remote_gpu"
                )
            token = (
                ""
                if self.remote_gpu_auth_token is None
                else self.remote_gpu_auth_token.get_secret_value()
            )
            if (
                not token
                or token != token.strip()
                or any(character in token for character in "\r\n\x00")
            ):
                raise ValueError(
                    "miner.remote_gpu_auth_token is required and must be a trimmed, "
                    "single-line secret when backend_mode=remote_gpu"
                )
            parsed = urlsplit(self.remote_gpu_url)
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError(
                    "miner.remote_gpu_url contains an invalid port"
                ) from exc
            if parsed.scheme.lower() != "https" or not parsed.hostname:
                raise ValueError(
                    "miner.remote_gpu_url must be an absolute HTTPS URL"
                )
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "miner.remote_gpu_url must not contain credentials, a query, or "
                    "a fragment"
                )
            if self.remote_gpu_solution_variant not in SUPPORTED_GPU_VARIANTS:
                raise ValueError(
                    "miner.remote_gpu_solution_variant must be one of "
                    f"{SUPPORTED_GPU_VARIANTS} when backend_mode=remote_gpu"
                )
        return self
