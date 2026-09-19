# CPU-only compression contender. Build has network access; the evaluation run has none.
# Pin the base image by digest in a real submission so the build is reproducible.
FROM ubuntu:24.04

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY run.sh variant.env /app/
RUN chmod 0755 /app/run.sh
