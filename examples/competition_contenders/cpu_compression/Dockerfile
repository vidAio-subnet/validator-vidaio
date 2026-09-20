# CPU-only compression contender. Build has network access; the evaluation run has none.
# Pin the base image by digest and the ffmpeg archive by sha256 in a real submission so
# the build is reproducible.
FROM ubuntu:24.04

# Do not use the distribution ffmpeg here: Ubuntu 24.04 ships SVT-AV1 1.7, which crashes
# on frame widths that are not a multiple of 8 (common in real clips), and its ffmpeg has
# no libvmaf filter, so the quality search below would silently fall back to a fixed CRF.
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
 && curl -fsSL "$FFMPEG_URL" -o /tmp/ffmpeg.tar.xz \
 && mkdir -p /opt/ffmpeg \
 && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
 && ln -s /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
 && ln -s /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
 && rm -f /tmp/ffmpeg.tar.xz \
 && apt-get purge -y curl xz-utils && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*
# Fail the build, not the evaluation, when a capability the profiles rely on is missing.
RUN ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libsvtav1 \
 && ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx265

WORKDIR /app
COPY run.sh variant.env /app/
RUN chmod 0755 /app/run.sh
