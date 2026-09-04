# syntax=docker/dockerfile:1
# linux/amd64 digests resolved from the NVIDIA registry on 2026-08-24. Modal L4
# Sandboxes are amd64; pinning prevents a mutable tag from changing a contender's
# enrolled tree into different build inputs.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749 AS build

COPY gpu_transform.cu /src/gpu_transform.cu
RUN nvcc -O3 --use_fast_math -std=c++17 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    /src/gpu_transform.cu -o /vidaio-next-gpu-transform

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04@sha256:cff3a0d82d2c2b47bab252d67fa9b34a20ef4c50781d98501b5c7367ea9afd10

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /vidaio-next-gpu-transform /app/gpu_transform
COPY run.sh variant.env /app/
RUN chmod 0555 /app/run.sh /app/gpu_transform \
    && chmod 0444 /app/variant.env

ENTRYPOINT []
CMD ["/bin/sh"]
