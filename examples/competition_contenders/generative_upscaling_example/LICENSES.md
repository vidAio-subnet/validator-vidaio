# Third-party components

Everything this entry ships besides its own code (MIT, see `LICENSE`). It ships no model
weights: the "model" is a fixed, weight-free filter. A real entry adds one row per weight
file and per model-code repository, each Apache-2.0, MIT or BSD licensed.

| Component | Version | Source | Licence | Location |
| --- | --- | --- | --- | --- |
| PyTorch official image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` | 2.8.0, pinned by digest | https://hub.docker.com/r/pytorch/pytorch | PyTorch BSD-3-Clause; Python and Conda packages under their own licences; NVIDIA CUDA 12.8 and cuDNN 9 runtime libraries under NVIDIA's redistribution licences (system runtime, not model code or weights) | base image |
| FFmpeg and ffprobe, static build `mwader/static-ffmpeg` | 9.0, pinned by digest | https://github.com/wader/static-ffmpeg | GPL-3.0-or-later (built with `--enable-gpl --enable-version3`), run as a separate program | `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe` |
