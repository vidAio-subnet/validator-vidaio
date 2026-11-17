# Generative upscaling example contender (GPU). The build has network access but no GPU;
# the evaluation run has one A100 80 GB and no network, so everything the run needs
# (code, libraries AND model weights) is installed here.
#
# The sandbox builder is not Docker/BuildKit (docs/MINING.md): only plain FROM, COPY,
# RUN, ENV and WORKDIR; no ADD; USER is ignored; an image reference carries a digest OR
# a tag, never both. The image must build within 30 minutes and stay under the
# manifest's container_size_limit_gb (40 GB for the generative competition).
#
# pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime (PyTorch 2.8.0, CUDA 12.8, Python 3.11),
# pinned by digest. About 4.3 GB compressed.
FROM pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385

# ffmpeg + ffprobe 9.0, static build with libx264 (mwader/static-ffmpeg:9.0), the same
# build the VidAIO upscaling example uses.
COPY --from=mwader/static-ffmpeg@sha256:b90574a4e2ae62b763c39c384526689e7eb435da6398f4fb3f6c3f1c6a14ce33 /ffmpeg /ffprobe /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- model weights go here --------------------------------------------------------
# A real entry downloads its weights at build time and checks them by sha256 in the
# SAME RUN step (the pre-check and the reviewers require it), for example:
#
#   RUN mkdir -p /app/weights \
#    && curl -fsSL -o /app/weights/model.safetensors \
#       https://huggingface.co/<org>/<repo>/resolve/<40-hex commit>/model.safetensors \
#    && echo "<sha256>  /app/weights/model.safetensors" | sha256sum -c -
#
# Code AND weights must be Apache-2.0, MIT or BSD licensed (list them in LICENSES.md).
# This example ships no weights: its "model" is a fixed, weight-free detail filter.
# -----------------------------------------------------------------------------------

COPY run.sh generative_example.py /app/
RUN chmod 0755 /app/run.sh && chmod 0644 /app/generative_example.py

# Fail the build, not the evaluation, when something the run relies on is missing.
# (No GPU at build time: the CUDA check happens at run time and falls back loudly.)
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
 && ffprobe -version >/dev/null \
 && command -v timeout >/dev/null && command -v du >/dev/null && command -v date >/dev/null \
 && python -c "import torch, generative_example; assert torch.__version__.startswith('2.8.0')"
