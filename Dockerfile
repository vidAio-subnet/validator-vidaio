# syntax=docker/dockerfile:1.7
# VidAIO release image — ONE locked image for every service (they share the codebase;
# docker-compose runs it with a different command per service, so it builds once
# and layer-caches). See the project design record.
#
# What this image must provide, and how it gets it:
#   * ffmpeg + ffprobe WITH libvmaf and the vmaf_v0.6.1 / vmaf_v0.6.1neg models —
#     copied as static binaries from mwader/static-ffmpeg, whose libvmaf is built
#     with the version models COMPILED IN. The scoring code invokes the models by
#     `model=version=vmaf_v0.6.1[neg]` (vidaio/scoring/backends_real.py), i.e. no
#     external model files are needed — verified empirically (a real 2-clip VMAF
#     computes for both models). Debian's own ffmpeg does NOT ship libvmaf, which
#     is why the static build is copied in rather than apt-installed.
#     VERSION MATTERS: pinned to 9.0 to match the developers' host ffmpeg (9.x).
#     ffmpeg 7.1's `-fps_mode cfr` with no explicit rate defaults the y4m output to
#     25fps, inflating a 20fps source's canonicalized frame count (120 -> 150) past
#     the scoring worker's scratch projection -> every /score 413s
#     `canonicalized_output_too_large`. 9.0 preserves the source rate (measured), so
#     canonicalization matches the projection and rounds actually score.
#   * the `docker` CLI (client only) for the orchestrator's docker-out-of-docker —
#     copied from the official docker:cli image; it talks to the host daemon over
#     the socket the compose file bind-mounts.
#   * the repo installed editable, with the dependency layer separated from the
#     source layer so a code edit rebuilds in seconds (only the tiny editable
#     re-register runs; the heavy dependency wheel layer stays cached).

FROM ghcr.io/astral-sh/uv:0.10.2@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 AS uv-bin

FROM python:3.13-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1 AS runtime

# Fixed unprivileged runtime identity. Build/install stages remain root; the final
# release drops permanently to this uid/gid. Production state directories are
# provisioned on the host for 10001:10001 before Compose starts.
RUN groupadd --system --gid 10001 vidaio \
 && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent \
      --shell /usr/sbin/nologin vidaio

# --- native tools -------------------------------------------------------------------
# HTTPS-only pinned repository checkouts need a real Git client and the system
# trust store. They are installed in both the release and inherited test image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

# ffmpeg + ffprobe (static, libvmaf + models built in) and the docker CLI client.
COPY --from=mwader/static-ffmpeg:9.0@sha256:b90574a4e2ae62b763c39c384526689e7eb435da6398f4fb3f6c3f1c6a14ce33 /ffmpeg /ffprobe /usr/local/bin/
COPY --from=docker:27-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c /usr/local/bin/docker /usr/local/bin/docker
COPY --from=uv-bin /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    OMP_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_NUM_THREADS=1 \
    MKL_DYNAMIC=FALSE \
    MKL_CBWR=COMPATIBLE \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    ATEN_CPU_CAPABILITY=default \
    TORCH_HOME=/opt/vidaio/torch

WORKDIR /app

# --- locked production dependency layer ---------------------------------------------
# `uv.lock` is the source of truth.  The release runtime deliberately installs all
# production extras: the default config selects Bittensor + S3, while every scorer
# and auditor must have the CPU media path.  Tests/pytest are excluded from this stage.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev \
      --extra chain --extra storage --extra media --extra modal

# --- source layer (the only layers a code edit invalidates) --------------------------
COPY VERSION ./
COPY vidaio ./vidaio
COPY config ./config
COPY scripts ./scripts
COPY examples ./examples
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      --extra chain --extra storage --extra media --extra modal

ENV PATH="/app/.venv/bin:${PATH}"

# Sanity: the tools the stack cannot run without are actually present in the image.
RUN ffmpeg -hide_banner -filters | grep -qi vmaf \
 && ffprobe -version >/dev/null \
 && docker --version >/dev/null \
 && git --version >/dev/null \
 && python scripts/verify_release_dependencies.py --preload-media

# Bind the complete CI/build source identity to the exact Python/config/runtime
# inputs retained by the lean image. The full checkout exists only in this
# throw-away stage; the release receives its deterministic manifest, not tests,
# docs, Git metadata, caches or credentials. Keep these COPY declarations aligned
# with integrity.py's SOURCE_FILES/SOURCE_DIRS. Explicit inputs make it impossible
# for a weakened .dockerignore to copy an operator .env or another undeclared file
# into an image/cache layer before the manifest walk ignores it.
FROM runtime AS runtime-manifest
COPY .dockerignore .env.example .gitignore DEPS.md Dockerfile Makefile README.md STATUS.md VERSION docker-compose.yml pyproject.toml uv.lock /release-source/
COPY vidaio /release-source/vidaio
COPY scripts /release-source/scripts
COPY config /release-source/config
COPY tests /release-source/tests
COPY docs /release-source/docs
COPY deploy /release-source/deploy
COPY examples /release-source/examples
RUN python vidaio/autoupdater/integrity.py /release-source --runtime-root /app \
      --write-runtime-manifest /runtime-release-manifest.json >/dev/null

# Tests live only in the explicit test target.  The default/final `runtime` image
# therefore cannot accidentally ship pytest or the repository's test corpus.
FROM runtime AS test
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
      --extra chain --extra storage --extra media --extra modal --extra dev
RUN python -c "import pytest, pytest_asyncio"

FROM runtime AS release
COPY --from=runtime-manifest /runtime-release-manifest.json /app/runtime-release-manifest.json
# Image-only marker consumed by the payout-runtime identity. A source checkout
# can have the same manifest bytes, but cannot accidentally present itself as the
# canonical scoring/auditing environment. Create it only in the final stage:
# the earlier dependency smoke intentionally runs before a release manifest
# exists and must not claim that it is already a qualified release. Keep these
# bytes aligned with CANONICAL_RUNTIME_MARKER_BYTES in runtime_identity.py.
RUN printf '%s\n' \
      'vidaio-release-runtime/1' \
      'os=linux' \
      'arch=amd64' \
      'aten_cpu_capability=default' \
      'torch_intraop_threads=1' \
      'torch_interop_threads=1' \
      'torch_deterministic_algorithms=error' \
      'torch_mkldnn=disabled' \
      'torch_nnpack=disabled' \
      'mkl_cbwr=COMPATIBLE' \
      > /app/.vidaio-release-runtime \
 && chmod 0444 /app/.vidaio-release-runtime \
 && chmod 0444 /app/runtime-release-manifest.json \
 && python vidaio/autoupdater/integrity.py /app --runtime-root /app \
      --verify-runtime-manifest /app/runtime-release-manifest.json >/dev/null \
 && mkdir -p /opt/vidaio/torch \
 && chown -R 10001:10001 /opt/vidaio/torch

# Bittensor inspects the process home even for wallet-free reads. Keep the system
# account non-login/nonpersistent, but resolve its runtime home into the fresh,
# hardened /tmp tmpfs required by the production Compose contract. Defining this
# only in the final stage prevents the root build smoke from seeding /tmp with
# root-owned SDK state.
ENV HOME=/tmp
USER 10001:10001

# No default CMD: docker-compose supplies `python scripts/service_entrypoint.py <svc>`.
