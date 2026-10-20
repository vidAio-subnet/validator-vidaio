# Streaming compression example contender (CPU only). The build has network access; the evaluation
# run has none, so everything the run needs is installed here.
#
# The sandbox builder is not Docker/BuildKit (docs/MINING.md): only plain FROM, COPY,
# RUN, ENV and WORKDIR; no ADD; USER is ignored; an image reference carries a digest OR
# a tag, never both. Both images below are pinned by their multi-arch index digest.
#
# python:3.13-slim-bookworm (Python 3.13.15, Debian 12), the base of the VidAIO images.
FROM python@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1

# ffmpeg + ffprobe 9.0, static build with libx264 (mwader/static-ffmpeg:9.0), the same
# build the VidAIO scorer decodes with. Debian's ffmpeg is not needed.
COPY --from=mwader/static-ffmpeg@sha256:b90574a4e2ae62b763c39c384526689e7eb435da6398f4fb3f6c3f1c6a14ce33 /ffmpeg /ffprobe /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY run.sh streaming_example.py /app/
RUN chmod 0755 /app/run.sh && chmod 0644 /app/streaming_example.py

# Fail the build, not the evaluation, when something the run relies on is missing.
RUN ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264 \
 && ffmpeg -hide_banner -filters 2>/dev/null | grep -q libx264 \
 && ffprobe -version >/dev/null \
 && command -v timeout >/dev/null && command -v du >/dev/null && command -v date >/dev/null \
 && python3 -c "import streaming_example"
