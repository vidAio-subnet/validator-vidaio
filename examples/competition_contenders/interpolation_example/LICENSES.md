# Third-party components

Everything this entry ships besides its own code (MIT, see `LICENSE`). All of it allows
commercial use. It ships no model weights (the example is the blend).

| Component | Version | Source | Licence | Location |
| --- | --- | --- | --- | --- |
| Python, official `python:3.13-slim-bookworm` image | 3.13, pinned by digest | https://hub.docker.com/_/python | PSF-2.0 (Python); the Debian 12 packages under their own free licences | base image |
| FFmpeg and ffprobe, static build `mwader/static-ffmpeg` | 9.0, pinned by digest | https://github.com/wader/static-ffmpeg | GPL-3.0-or-later (built with `--enable-gpl --enable-version3`), run as a separate program | `/usr/local/bin/ffmpeg`, `/usr/local/bin/ffprobe` |
| NumPy | 2.2.6, pinned by sha256 (`requirements.txt`) | https://numpy.org | BSD-3-Clause; the wheel bundles OpenBLAS (BSD-3-Clause) and the GCC Fortran runtime (GPL-3.0 with the GCC Runtime Library Exception) | Python site-packages |
