"""Versioned procedural degradation DAG with private-seed randomization.

Spec (design spec §18 anti-gaming): "Degradation must be a versioned procedural DAG, not a
preset list. Randomize operator order, resize kernel + subpixel phase, spatially-varying
blur, codec/rate-control/GOP/chroma/bit-depth, sensor/ISP noise,
exposure/rolling-shutter/motion, tone/color pipelines, and artifact masks/trajectories.
Task type public; parameters and seeds private."

All randomness flows through one injected ``random.Random``; same seed + dag_version
produce a byte-identical canonical JSON, digest, and ffmpeg plan.

Version history
---------------
  1  initial structure (continuous Downscale factor on every track)
  2  upscaling-track Downscale factors restricted to the discrete UPSCALE_FACTORS
     {2, 4} — the factors scoring's file_size_caps supports; compression-track
     Downscale keeps the continuous range.
  3  FrameDrop excluded from BOTH tracks' operator pools (P1 scope cut — see
     "FrameDrop exclusion" below). The operator class and its registry entry
     remain so historical dag_version <= 2 documents still deserialize.
  4  ArtifactMask actually masks. Its `drawbox` realization put `t` in the x/y
     expressions meaning "time" while ffmpeg's drawbox reads `t` there as the
     box THICKNESS — and the same expression set `t=fill`, so every box was
     positioned at the fill thickness, far off frame. The operator was a SILENT
     NO-OP (verified byte-identical by frame-md5 for every sampled velocity
     except an exact 0.0), which means a sampled `artifact_mask` degraded
     nothing. Rebuilt on darken -> crop -> overlay, which does have a real time
     variable (see ArtifactMask.filter_graph). SAMPLED PARAMETERS AND THE
     OPERATOR POOLS ARE UNCHANGED — only the realization — but a
     `dag_version 3` document now renders differently than it did, so the
     version moves and the digest with it.
  5  Challenge commitments fence the miner-input VMAF model-delta scorer
     semantics selected for launch. Degradation operators and sampling are
     unchanged from version 4; the version distinguishes the downstream scoring
     contract under which a produced challenge is valid.
  6  Perceptual manipulation gates use the miner input (not the pristine
     holdout), and compression no longer draws GaussianBlur or Downscale. Those
     operators destroyed enough information to put a large fraction of draws
     below the pristine-reference VMAF floor regardless of miner merit. They
     remain available to the restoration/upscaling track.
  7  Launch-calibrated task pools. Compression is codec-only and uses a
     minimally-lossy H.264/8-bit/4:2:0 CRF draw; upscaling is downscale-only.
     The broader restoration operator pool remains registered for historical
     DAG deserialization and a later, separately calibrated version. This
     prevents the validator from minting rounds whose own degradations make the
     production quality/size gates impossible for the shipped baselines.

FrameDrop exclusion (dag_version 3 — documented P1 scope cut)
-------------------------------------------------------------
FrameDrop's realization (``select='lt(mod(n,cycle),keep)'`` +
``setpts=N/FRAME_RATE/TB``) renumbers the surviving frames contiguously at the
SOURCE frame rate: the degraded clip keeps the reference's fps but carries only
keep/cycle of its frames and keep/cycle of its duration (measured: a 20-frame
2.0s clip at 10 fps becomes 14 frames / 1.4s / still 10 fps under keep=2,
cycle=3). An honest restoration of that input necessarily has the same
shortened temporal shape, so the scoring validity gates (FrameCountGate,
validate_stream's FRAME_COUNT_MISMATCH / STREAM_DURATION_MISMATCH) zero every
honest miner on any challenge whose DAG contains FrameDrop.

fps-normalization at the scoring seam CANNOT repair this, structurally:
  * the canonicalization plan's ``fps=`` lever applies one common rate to both
    sides, but the two sides genuinely differ in DURATION — for any pinned rate
    f the reference holds ~duration*f frames and the candidate ~0.7x of that,
    so frame count and duration can never both match;
  * for keep >= 2 the kept frames are non-uniform in source time (e.g. source
    indices 0,1,3,4,6,7,... for keep=2/cycle=3) while the fps filter resamples
    uniformly by timestamp — no uniform resampling aligns candidate frame i
    with its true reference frame. Full-reference metrics (VMAF, PieAPP) are
    frame-paired, so the comparison is temporally undefined, not just gated.
Temporal reference mismatch is therefore fundamental for full-reference
scoring; FrameDrop can only return together with a temporal-alignment-aware
scoring path (per-frame source-index mapping revealed with the DAG), which is
out of P1 scope. Reintroducing it is a DAG_VERSION bump.

Ordering constraints (unchanged since dag_version 3)
----------------------------------------------------
Operators are grouped into pipeline stages that model how real footage degrades:

  capture  (Exposure, MotionBlur, GaussianBlur, Noise)   — lens/sensor-time artifacts
  edit     (ToneShift, ColorPipeline, ArtifactMask)      — post-production artifacts
  delivery (Downscale, CodecCompress)                    — distribution artifacts
           (FrameDrop belongs here structurally but is excluded from the pools
            since dag_version 3 — see "FrameDrop exclusion" above)

Stage order is fixed capture -> edit -> delivery (noise-after-encode etc. would be
physically implausible and easy to fingerprint); operator order WITHIN a stage is
shuffled by the rng. Two hard constraints inside delivery:

  * CodecCompress, when present, is always the FINAL operator — it produces the
    delivered bitstream, so nothing may re-process (and thus re-encode) after it.
  * Downscale therefore always precedes CodecCompress (required for upscaling tasks:
    the miner receives a small encoded clip, never an encode of the full-res frame).

Track rules: "compression" requires CodecCompress; "upscaling" requires Downscale.
The remaining operators form the optional pool; the rng picks how many and which.

Operator -> ffmpeg mapping table (plan builder only; execution is out of scope)
-------------------------------------------------------------------------------
Each operator maps to exactly one ffmpeg command; commands chain through lossless
FFV1/MKV intermediates, and the last command writes the final output path.

  Downscale      scale=iw*4:ih*4:flags=neighbor (4x oversample), crop at a quarter-pel
                 integer offset (subpixel phase), then scale by `scale_factor` with the
                 sampled kernel (lanczos|bicubic|bilinear), even-dimension rounded.
  GaussianBlur   spatially-varying blur: geq gradient map (horizontal|vertical|radial)
                 feeding varblur=min_r:max_r (radius ~ 2*sigma).
  Noise          noise=alls=<strength>:allf=<flags>; gaussian -> default noise,
                 iso -> u (uniform), sensor -> a (averaged); +t when temporal.
  CodecCompress  -c:v libx264|libx265, -crf N or -b:v/-maxrate/-bufsize kbps,
                 -g GOP, -pix_fmt from chroma (420|422) x bit depth (8|10).
  FrameDrop      select='lt(mod(n,cycle),keep)' + setpts=N/FRAME_RATE/TB
                 (keeps `keep` of every `cycle` frames).
  ToneShift      eq=gamma:brightness:contrast:saturation.
  ColorPipeline  hue=h=<deg>, colortemperature=temperature=<K>,
                 colorbalance=rs=<r>:bs=<b>.
  ArtifactMask   a translucent black box on a time-linear trajectory (x0+vx*t,
                 y0+vy*t in frame fractions per second, clamped inside the frame),
                 realized as split -> lutyuv darken -> moving crop -> overlay
                 (dag_version 4; drawbox has no time variable — see the operator).
  Exposure       exposure=exposure=<EV> + eq=eval=frame per-frame sinusoidal
                 brightness flicker.
  MotionBlur     tmix=frames=N (temporal accumulation blur).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# v7 is the explicitly calibrated launch pool: codec-only compression and
# downscale-only upscaling. Historical operator models remain deserializable.
DAG_VERSION = 7

# Optional-operator count range (unchanged since dag_version 3; structure is
# versioned, not config).
MIN_OPTIONAL_OPS = 2
MAX_OPTIONAL_OPS = 5

# Discrete factors the protocol/scorer understands. MUST stay in sync with the keys of
# vidaio.scoring config `file_size_caps` — the file-size gate is keyed by these factors,
# and an unsupported factor would make the gate unenforceable.
UPSCALE_FACTORS: tuple[int, ...] = (2, 4)

# The v7 launch generator is deliberately narrower than protocol support. Real-media
# calibration showed the shipped Lanczos baseline clears the production VMAF floor for
# 2x draws, while some honest 4x draws do not. Keeping 4x in UPSCALE_FACTORS preserves
# already-committed competition items and the future geometry-v2 contract; adding 4x
# back to live inference generation requires calibration and a DAG_VERSION bump.
LAUNCH_UPSCALE_FACTORS: tuple[int, ...] = (2,)

STAGE_CAPTURE = "capture"
STAGE_EDIT = "edit"
STAGE_DELIVERY = "delivery"
STAGE_ORDER = (STAGE_CAPTURE, STAGE_EDIT, STAGE_DELIVERY)


def canonical_json_dumps(payload: Any) -> str:
    """Stable canonical JSON: sorted keys, no whitespace. Shared by digests/commitments."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


# --- private-seed key derivation ------------------------------------------------------


def seed_to_bytes(private_seed: int) -> bytes:
    """Canonical big-endian byte encoding of a private seed for key derivation."""
    if private_seed < 0:
        raise ValueError("private seeds must be non-negative")
    return private_seed.to_bytes(max(1, (private_seed.bit_length() + 7) // 8), "big")


def dag_rng_from_seed(private_seed: int) -> random.Random:
    """The ONLY sanctioned way to turn a private seed into the DAG rng.

    The Mersenne Twister is seeded from sha256(b"dag" || seed_bytes), never from the
    bare seed. Any public value derived from the seed (e.g. the challenge id) must use
    a different sha256 domain tag, so no raw MT output tied to the bare seed is ever
    observable — a miner cannot brute-force public material back to the DAG stream.
    """
    digest = hashlib.sha256(b"dag" + seed_to_bytes(private_seed)).digest()
    return random.Random(int.from_bytes(digest, "big"))


# --- sampling helpers (all values rounded so serialization is platform-stable) -------


def _u(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 4)


def _log_u(rng: random.Random, lo: float, hi: float) -> float:
    return round(math.exp(rng.uniform(math.log(lo), math.log(hi))), 4)


def _log_u_int(rng: random.Random, lo: float, hi: float) -> int:
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


# --- operators ------------------------------------------------------------------------


class DegradationOp(BaseModel):
    """Base degradation operator: sampled concrete params + its ffmpeg realization."""

    model_config = ConfigDict(frozen=True)

    stage: ClassVar[str]

    @classmethod
    def sample(cls, rng: random.Random) -> "DegradationOp":
        raise NotImplementedError

    def filter_expr(self) -> str:
        raise NotImplementedError

    def filter_graph(self) -> str:
        return f"[0:v]{self.filter_expr()}[out]"

    def command(self, input_path: str, output_path: str) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            input_path,
            "-filter_complex",
            self.filter_graph(),
            "-map",
            "[out]",
            "-c:v",
            "ffv1",
            "-an",
            output_path,
        ]


class Downscale(DegradationOp):
    stage: ClassVar[str] = STAGE_DELIVERY
    op: Literal["downscale"] = "downscale"
    kernel: Literal["lanczos", "bicubic", "bilinear"]
    scale_factor: float  # compression track: log-uniform [0.3, 0.75];
    # upscaling track: exactly 1/f for f in UPSCALE_FACTORS (sample_discrete)
    phase_x: float  # quarter-pel subpixel phase in {0, .25, .5, .75}
    phase_y: float

    @classmethod
    def sample(cls, rng: random.Random) -> "Downscale":
        return cls(
            kernel=rng.choice(["lanczos", "bicubic", "bilinear"]),
            scale_factor=_log_u(rng, 0.3, 0.75),
            phase_x=rng.choice([0.0, 0.25, 0.5, 0.75]),
            phase_y=rng.choice([0.0, 0.25, 0.5, 0.75]),
        )

    @classmethod
    def sample_discrete(cls, rng: random.Random) -> "Downscale":
        """Upscaling-track variant (dag_version >= 2): the miner must invert this
        downscale, so the factor is restricted to the discrete UPSCALE_FACTORS that
        scoring's file-size caps support."""
        factor = rng.choice(UPSCALE_FACTORS)
        return cls(
            kernel=rng.choice(["lanczos", "bicubic", "bilinear"]),
            scale_factor=round(1 / factor, 4),
            phase_x=rng.choice([0.0, 0.25, 0.5, 0.75]),
            phase_y=rng.choice([0.0, 0.25, 0.5, 0.75]),
        )

    @classmethod
    def sample_launch(cls, rng: random.Random) -> "Downscale":
        """Launch-calibrated upscaling draw (DAG v7).

        ``sample_discrete`` retains the complete 2x/4x protocol surface for
        historical/future callers. Live v7 inference tasks use only the subset
        whose shipped baseline clears production gates.
        """
        factor = rng.choice(LAUNCH_UPSCALE_FACTORS)
        return cls(
            kernel=rng.choice(["lanczos", "bicubic", "bilinear"]),
            scale_factor=round(1 / factor, 4),
            phase_x=rng.choice([0.0, 0.25, 0.5, 0.75]),
            phase_y=rng.choice([0.0, 0.25, 0.5, 0.75]),
        )

    def filter_expr(self) -> str:
        ox, oy = int(self.phase_x * 4), int(self.phase_y * 4)
        f = self.scale_factor
        return (
            f"scale=iw*4:ih*4:flags=neighbor,"
            f"crop=iw-3:ih-3:{ox}:{oy},"
            f"scale=trunc(iw*{f}/8)*2:trunc(ih*{f}/8)*2:flags={self.kernel}"
        )


class GaussianBlur(DegradationOp):
    stage: ClassVar[str] = STAGE_CAPTURE
    op: Literal["gaussian_blur"] = "gaussian_blur"
    sigma_min: float  # uniform [0.3, 1.2]
    sigma_max: float  # uniform [1.5, 5.0]
    axis: Literal["horizontal", "vertical", "radial"]

    _GRADIENTS: ClassVar[dict[str, str]] = {
        "horizontal": "255*X/W",
        "vertical": "255*Y/H",
        "radial": "255*hypot(X-W/2,Y-H/2)/hypot(W/2,H/2)",
    }

    @classmethod
    def sample(cls, rng: random.Random) -> "GaussianBlur":
        return cls(
            sigma_min=_u(rng, 0.3, 1.2),
            sigma_max=_u(rng, 1.5, 5.0),
            axis=rng.choice(["horizontal", "vertical", "radial"]),
        )

    def filter_graph(self) -> str:
        min_r = max(0, round(2 * self.sigma_min))
        max_r = max(min_r + 1, round(2 * self.sigma_max))
        grad = self._GRADIENTS[self.axis]
        return (
            f"[0:v]split[vb_in][vb_ref];"
            f"[vb_ref]format=gray,geq=lum='{grad}'[vb_map];"
            f"[vb_in][vb_map]varblur=min_r={min_r}:max_r={max_r}[out]"
        )


class Noise(DegradationOp):
    stage: ClassVar[str] = STAGE_CAPTURE
    op: Literal["noise"] = "noise"
    kind: Literal["gaussian", "iso", "sensor"]
    strength: float  # log-uniform [4, 24] (ffmpeg noise 0-100 scale)
    temporal: bool

    _KIND_FLAG: ClassVar[dict[str, str]] = {"gaussian": "", "iso": "u", "sensor": "a"}

    @classmethod
    def sample(cls, rng: random.Random) -> "Noise":
        return cls(
            kind=rng.choice(["gaussian", "iso", "sensor"]),
            strength=_log_u(rng, 4, 24),
            temporal=rng.random() < 0.7,
        )

    def filter_expr(self) -> str:
        flags = [
            f for f in (self._KIND_FLAG[self.kind], "t" if self.temporal else "") if f
        ]
        expr = f"noise=alls={self.strength}"
        if flags:
            expr += f":allf={'+'.join(flags)}"
        return expr


class CodecCompress(DegradationOp):
    stage: ClassVar[str] = STAGE_DELIVERY
    op: Literal["codec_compress"] = "codec_compress"
    codec: Literal["h264", "h265"]
    rate_mode: Literal["crf", "bitrate"]
    crf: int | None = None  # randint [24, 40] when rate_mode == crf
    bitrate_kbps: int | None = (
        None  # log-uniform int [300, 3000] when rate_mode == bitrate
    )
    gop: int  # choice of common GOP sizes
    chroma: Literal["420", "422"]
    bit_depth: Literal[8, 10]

    _ENCODER: ClassVar[dict[str, str]] = {"h264": "libx264", "h265": "libx265"}
    _PIX_FMT: ClassVar[dict[tuple[str, int], str]] = {
        ("420", 8): "yuv420p",
        ("422", 8): "yuv422p",
        ("420", 10): "yuv420p10le",
        ("422", 10): "yuv422p10le",
    }

    @classmethod
    def sample(cls, rng: random.Random) -> "CodecCompress":
        rate_mode = rng.choice(["crf", "bitrate"])
        return cls(
            codec=rng.choice(["h264", "h265"]),
            rate_mode=rate_mode,
            crf=rng.randint(24, 40) if rate_mode == "crf" else None,
            bitrate_kbps=_log_u_int(rng, 300, 3000) if rate_mode == "bitrate" else None,
            gop=rng.choice([12, 24, 48, 96, 120, 250]),
            chroma=rng.choice(["420", "422"]),
            bit_depth=rng.choice([8, 10]),
        )

    @classmethod
    def sample_launch(cls, rng: random.Random) -> "CodecCompress":
        """Minimally-lossy launch input for the compression track (DAG v7).

        Production scoring requires a contender to make the *miner input* at
        least 20% smaller while retaining pristine-reference VMAF >= 90. The
        former v6 sampler could hand miners an already tiny CRF-40/300-kbps
        input, so an honest re-encode grew the file and the round was
        structurally unwinnable. A high-quality CRF 8/10/12 input leaves real
        compression headroom for the shipped CRF-18 quality baseline without
        making the validator input lossless or byte-identical to the holdout.

        Fixing codec/chroma/depth also removes format-conversion quality loss
        from the task generator. An all-intra GOP is intentional: even very
        low-entropy clips then retain enough codec-only size headroom for an
        ordinary inter-frame contender to satisfy the strict <0.80 size gate.
        CRF remains a private draw, so this is still procedural rather than a
        static preset file.
        """
        return cls(
            codec="h264",
            rate_mode="crf",
            crf=rng.choice([8, 10, 12]),
            bitrate_kbps=None,
            gop=1,
            chroma="420",
            bit_depth=8,
        )

    def command(self, input_path: str, output_path: str) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            input_path,
            "-c:v",
            self._ENCODER[self.codec],
            "-g",
            str(self.gop),
            "-pix_fmt",
            self._PIX_FMT[(self.chroma, self.bit_depth)],
        ]
        if self.rate_mode == "crf":
            cmd += ["-crf", str(self.crf)]
        else:
            k = self.bitrate_kbps
            cmd += ["-b:v", f"{k}k", "-maxrate", f"{k}k", "-bufsize", f"{2 * k}k"]
        cmd += ["-an", output_path]
        return cmd


class FrameDrop(DegradationOp):
    """EXCLUDED from all track pools since dag_version 3 (see EXCLUDED_OPS and
    the module docstring) — kept only so historical DAG documents deserialize."""

    stage: ClassVar[str] = STAGE_DELIVERY
    op: Literal["frame_drop"] = "frame_drop"
    keep: int  # keep `keep` of every `cycle` frames
    cycle: int  # choice of {2, 3, 4}

    @classmethod
    def sample(cls, rng: random.Random) -> "FrameDrop":
        cycle = rng.choice([2, 3, 4])
        return cls(cycle=cycle, keep=rng.randint(1, cycle - 1))

    def filter_expr(self) -> str:
        return f"select='lt(mod(n,{self.cycle}),{self.keep})',setpts=N/FRAME_RATE/TB"


class ToneShift(DegradationOp):
    stage: ClassVar[str] = STAGE_EDIT
    op: Literal["tone_shift"] = "tone_shift"
    gamma: float  # uniform [0.7, 1.4]
    brightness: float  # uniform [-0.15, 0.15]
    contrast: float  # uniform [0.8, 1.25]
    saturation: float  # uniform [0.6, 1.4]

    @classmethod
    def sample(cls, rng: random.Random) -> "ToneShift":
        return cls(
            gamma=_u(rng, 0.7, 1.4),
            brightness=_u(rng, -0.15, 0.15),
            contrast=_u(rng, 0.8, 1.25),
            saturation=_u(rng, 0.6, 1.4),
        )

    def filter_expr(self) -> str:
        return (
            f"eq=gamma={self.gamma}:brightness={self.brightness}"
            f":contrast={self.contrast}:saturation={self.saturation}"
        )


class ColorPipeline(DegradationOp):
    stage: ClassVar[str] = STAGE_EDIT
    op: Literal["color_pipeline"] = "color_pipeline"
    hue_shift_deg: float  # uniform [-25, 25]
    temperature_k: int  # uniform int [3500, 8500]
    balance_r: float  # uniform [-0.25, 0.25] (shadow red balance)
    balance_b: float  # uniform [-0.25, 0.25] (shadow blue balance)

    @classmethod
    def sample(cls, rng: random.Random) -> "ColorPipeline":
        return cls(
            hue_shift_deg=_u(rng, -25, 25),
            temperature_k=rng.randint(3500, 8500),
            balance_r=_u(rng, -0.25, 0.25),
            balance_b=_u(rng, -0.25, 0.25),
        )

    def filter_expr(self) -> str:
        return (
            f"hue=h={self.hue_shift_deg},"
            f"colortemperature=temperature={self.temperature_k},"
            f"colorbalance=rs={self.balance_r}:bs={self.balance_b}"
        )


class ArtifactMask(DegradationOp):
    stage: ClassVar[str] = STAGE_EDIT
    op: Literal["artifact_mask"] = "artifact_mask"
    x0: float  # start position, frame fraction, uniform [0.05, 0.7]
    y0: float
    width_frac: float  # uniform [0.08, 0.25]
    height_frac: float
    vx: float  # drift in frame fractions / second, uniform [-0.05, 0.05]
    vy: float
    opacity: float  # uniform [0.6, 1.0]

    @classmethod
    def sample(cls, rng: random.Random) -> "ArtifactMask":
        return cls(
            x0=_u(rng, 0.05, 0.7),
            y0=_u(rng, 0.05, 0.7),
            width_frac=_u(rng, 0.08, 0.25),
            height_frac=_u(rng, 0.08, 0.25),
            vx=_u(rng, -0.05, 0.05),
            vy=_u(rng, -0.05, 0.05),
            opacity=_u(rng, 0.6, 1.0),
        )

    #: Trajectory, in FRAME FRACTIONS, as an ffmpeg expression over the frame
    #: width/height variable names of the filter it lands in (`crop` speaks
    #: iw/ih/ow/oh, `overlay` speaks W/H/w/h). The position is clamped so the
    #: box stays wholly inside the frame — a mask that drifts off frame stops
    #: masking, which is the failure mode this operator just came out of.
    def _position(
        self, frame_dim: str, box_dim: str, start: float, velocity: float
    ) -> str:
        travel = f"{frame_dim}*({start}+{velocity}*t)"
        clamped = f"min(max({travel},0),{frame_dim}-{box_dim})"
        # Snap to an even offset: 4:2:0 chroma is subsampled 2x, and crop and
        # overlay must land the box on the SAME pixel or the chroma tears.
        return f"2*floor({clamped}/2)"

    def filter_graph(self) -> str:
        """A moving translucent black box, realized as darken -> crop -> overlay.

        NOT `drawbox`. In drawbox, `t` inside a coordinate expression is the box
        THICKNESS, not time — there is no time or frame-number variable in that
        filter at all (verified against ffmpeg 9: `n`, `pts` and `T` are all
        "Undefined constant"). The previous realization wrote
        `x=iw*(x0+vx*t)` next to the `t=fill` sentinel, so `t` evaluated to the
        fill thickness (INT_MAX) and every box landed far off frame: the
        operator was a SILENT NO-OP whose output was byte-identical (frame-md5)
        to no filter at all, for every sampled `vx`/`vy` except an exact 0.0.

        `overlay` and `crop` do have a real `t` (presentation time in seconds),
        which is the unit `vx`/`vy` are documented in. So the frame is split,
        one copy is darkened toward black by `1 - opacity` (`lutyuv`, a 256/1024
        entry table — no per-pixel expression evaluation), the moving rectangle
        is cropped out of THAT copy, and it is pasted back over the original at
        the same moving position. Compositing a darkened cut-out of the region
        it covers is exactly an alpha blend with black at `opacity`, and it does
        it without introducing an alpha plane: `lutyuv`, `crop` and an
        alpha-free `overlay` all preserve yuv420p, so the stage's pixel format
        survives (an RGBA/`yuva420p` detour would have leaked out of this
        operator into every downstream stage).

        `minval`/`maxval` are lut's own range constants, so limited-range black
        (16 / 64) and the chroma neutral are correct at 8 and 10 bit alike.
        """
        keep = round(1.0 - self.opacity, 4)  # what survives of the covered region
        neutral = "(minval+maxval)/2"
        darken = (
            f"lutyuv=y='minval+(val-minval)*{keep}'"
            f":u='{neutral}+(val-{neutral})*{keep}'"
            f":v='{neutral}+(val-{neutral})*{keep}'"
        )
        crop_x = self._position("iw", "ow", self.x0, self.vx)
        crop_y = self._position("ih", "oh", self.y0, self.vy)
        over_x = self._position("W", "w", self.x0, self.vx)
        over_y = self._position("H", "h", self.y0, self.vy)
        return (
            f"[0:v]split[am_bg][am_dark];"
            f"[am_dark]{darken},"
            f"crop=w=iw*{self.width_frac}:h=ih*{self.height_frac}"
            f":x='{crop_x}':y='{crop_y}'[am_box];"
            f"[am_bg][am_box]overlay=x='{over_x}':y='{over_y}':eval=frame[out]"
        )


class Exposure(DegradationOp):
    stage: ClassVar[str] = STAGE_CAPTURE
    op: Literal["exposure"] = "exposure"
    ev: float  # uniform [-1.2, 1.2]
    flicker_amp: float  # uniform [0.0, 0.06]
    flicker_rate: float  # cycles per frame, uniform [0.05, 0.3]

    @classmethod
    def sample(cls, rng: random.Random) -> "Exposure":
        return cls(
            ev=_u(rng, -1.2, 1.2),
            flicker_amp=_u(rng, 0.0, 0.06),
            flicker_rate=_u(rng, 0.05, 0.3),
        )

    def filter_expr(self) -> str:
        return (
            f"exposure=exposure={self.ev},"
            f"eq=eval=frame:brightness={self.flicker_amp}*sin(2*PI*n*{self.flicker_rate})"
        )


class MotionBlur(DegradationOp):
    stage: ClassVar[str] = STAGE_CAPTURE
    op: Literal["motion_blur"] = "motion_blur"
    frames: int  # randint [2, 5]

    @classmethod
    def sample(cls, rng: random.Random) -> "MotionBlur":
        return cls(frames=rng.randint(2, 5))

    def filter_expr(self) -> str:
        return f"tmix=frames={self.frames}"


OperatorUnion = Annotated[
    Union[
        Downscale,
        GaussianBlur,
        Noise,
        CodecCompress,
        FrameDrop,
        ToneShift,
        ColorPipeline,
        ArtifactMask,
        Exposure,
        MotionBlur,
    ],
    Field(discriminator="op"),
]

_OPERATOR_CLASSES: tuple[type[DegradationOp], ...] = (
    Downscale,
    GaussianBlur,
    Noise,
    CodecCompress,
    FrameDrop,
    ToneShift,
    ColorPipeline,
    ArtifactMask,
    Exposure,
    MotionBlur,
)

OPERATOR_REGISTRY: dict[str, type[DegradationOp]] = {
    cls.model_fields["op"].default: cls for cls in _OPERATOR_CLASSES
}


# --- track rules ----------------------------------------------------------------------


class TrackRule(BaseModel):
    model_config = ConfigDict(frozen=True)
    required: tuple[str, ...]
    optional: tuple[str, ...]  # kept sorted: it is an rng.sample population


#: Operators structurally excluded from EVERY track's pools (dag_version 3).
#: FrameDrop shortens the candidate's frame count/duration relative to the
#: reference, which the validity gates rightly zero, and full-reference metrics
#: cannot be temporally aligned to a decimated candidate (module docstring,
#: "FrameDrop exclusion"). Registry entry retained for historical documents.
EXCLUDED_OPS: tuple[str, ...] = ("frame_drop",)


def _rule(
    required: tuple[str, ...], *, track_excluded: tuple[str, ...] = ()
) -> TrackRule:
    excluded = set(EXCLUDED_OPS) | set(track_excluded)
    if set(required) & excluded:
        raise ValueError(f"excluded ops cannot be required: {required}")
    return TrackRule(
        required=required,
        optional=tuple(sorted(set(OPERATOR_REGISTRY) - set(required) - excluded)),
    )


# These are intentionally explicit rather than "all registry entries except X".
# Launch pools must never grow merely because a new operator is registered. Every
# future pool expansion needs real-media calibration at production gates and a
# DAG_VERSION bump.
TRACK_RULES: dict[str, TrackRule] = {
    "compression": TrackRule(required=("codec_compress",), optional=()),
    "upscaling": TrackRule(required=("downscale",), optional=()),
}


# --- the DAG --------------------------------------------------------------------------


class DegradationDag(BaseModel):
    """Ordered, fully-sampled degradation pipeline. Serialization is canonical."""

    model_config = ConfigDict(frozen=True)

    dag_version: int
    task_type: str
    ops: list[OperatorUnion]

    def canonical_json(self) -> str:
        return canonical_json_dumps(self.model_dump(mode="json"))

    def canonical_digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def _order_ops(names: set[str], rng: random.Random) -> list[str]:
    """Stage-constrained order: fixed stage sequence, shuffled within, codec pinned last."""
    ordered: list[str] = []
    for stage in STAGE_ORDER:
        members = sorted(n for n in names if OPERATOR_REGISTRY[n].stage == stage)
        pin_codec = stage == STAGE_DELIVERY and "codec_compress" in members
        if pin_codec:
            members.remove("codec_compress")
        rng.shuffle(members)
        ordered.extend(members)
        if pin_codec:
            ordered.append("codec_compress")
    return ordered


def build_dag(
    task_type: str, rng: random.Random, *, dag_version: int = DAG_VERSION
) -> DegradationDag:
    """Build a randomized degradation DAG for a public task type from a private rng.

    Everything non-public — operator subset, order, and every parameter — is drawn
    solely from `rng`, which the caller seeds from the private challenge seed.
    """
    if dag_version != DAG_VERSION:
        raise ValueError(
            f"unsupported dag_version {dag_version}; this build supports {DAG_VERSION}"
        )
    rule = TRACK_RULES.get(task_type)
    if rule is None:
        raise ValueError(
            f"unknown task_type {task_type!r}; known: {sorted(TRACK_RULES)}"
        )

    max_opt = min(MAX_OPTIONAL_OPS, len(rule.optional))
    n_optional = rng.randint(min(MIN_OPTIONAL_OPS, max_opt), max_opt)
    chosen = rng.sample(rule.optional, n_optional)
    names = set(rule.required) | set(chosen)

    ordered = _order_ops(names, rng)
    ops = [_sample_op(name, task_type, rng) for name in ordered]
    return DegradationDag(dag_version=dag_version, task_type=task_type, ops=ops)


def _sample_op(name: str, task_type: str, rng: random.Random) -> DegradationOp:
    if name == "codec_compress" and task_type == "compression":
        return CodecCompress.sample_launch(rng)
    if name == "downscale" and task_type == "upscaling":
        return Downscale.sample_launch(rng)
    return OPERATOR_REGISTRY[name].sample(rng)


def to_ffmpeg_plan(
    dag: DegradationDag, input_path: str, output_path: str
) -> list[list[str]]:
    """Ordered ffmpeg argv plans realizing the DAG. Pure builder — nothing is executed.

    Non-final stages write lossless FFV1/MKV intermediates named
    `<output_path>.stageNN.mkv`; the final stage writes `output_path`.
    """
    plan: list[list[str]] = []
    current = input_path
    for i, op in enumerate(dag.ops):
        out = (
            output_path if i == len(dag.ops) - 1 else f"{output_path}.stage{i:02d}.mkv"
        )
        plan.append(op.command(current, out))
        current = out
    return plan
