# Generative upscaling example contender (GPU)

A minimal, working entry for the **generative** upscaling competition: the GPU pipeline a
real generative upscaler needs (ffmpeg decodes raw frames into a PyTorch model on the GPU,
the model's frames go to the encoder) with a weight-free stub in place of the model. It
builds and runs without downloading anything, so it scores like a plain interpolation
by design. Use it as the skeleton for your own entry.

| File | Purpose |
| --- | --- |
| `Dockerfile` | digest-pinned PyTorch 2.8 + CUDA 12.8 runtime image + static ffmpeg 9.0; where the weights go |
| `run.sh` | the competition entry point: iterates the items, enforces time and size budgets |
| `generative_example.py` | one item: task file, decode, model (stub) on the GPU, encode |
| `LICENSE`, `LICENSES.md` | your licence and every shipped third-party component |

## The contract

Identical to the faithful upscaling track (see `../upscaling_example/README.md` and
`docs/MINING.md`): `/bin/sh /app/run.sh <input_dir> <output_dir>`, no network at run time,
read-only root filesystem (write only to `/tmp` and the output directory).

- **Input**: every regular, non-hidden file in the input directory is one item, named by
  its 64-hex sha256: a 1080p original degraded and downscaled by 2 (960×540) or by 4
  (480×270), or the portrait mirror, H.264 in MP4, up to 10 seconds long, at a constant frame rate up
  to 60 fps.
- **Task file**: the hidden file `.vidaio-next-upscale-task-<sha256>` next to each item,
  for example `{"target_height":1080,"target_width":1920,"upscale_factor":4}`. Always read
  the target from it.
- **Output**: one file per item under the same name: an MP4 (or Matroska) with one H.264,
  HEVC, VP9 or AV1 stream, yuv420p, **exactly the target size**, the input's frame count and
  frame rate, tagged BT.709 limited range. Encode at high quality (this example uses x264
  CRF 12): 512 MiB per file and 2 GiB per batch at most.
- `run.sh` must exit 0. A non-zero exit, a timeout or an oversized output directory zeroes
  the whole batch; a missing output zeroes only that item.

## The sandbox

- **GPU**: one NVIDIA A100 80 GB per run (Modal `A100-80GB`). The build gets no GPU.
- **Image**: at most the manifest's `container_size_limit_gb` (40 GB for this competition),
  measured after the build. Bake the weights into the image; nothing can be downloaded at
  run time.
- **Build**: must finish within 30 minutes (`linux/amd64`).
- **Time**: the manifest's `batch_timeout_seconds` bounds one whole `run.sh` call
  (`evaluation_batch_size` items). Your model is reloaded for every batch: load it once
  per `run.sh` call, not once per item, in a real entry. When the manifest carries a speed
  term, your run time also counts.

## How it is scored (summary)

Each output is compared with the sealed 1080p original it was degraded from, on the CPU,
deterministically:

- **Gates** (fail = the clip scores 0; more than the anchored share of failed clips, 10 %
  for this competition, and the entry holds no paid rank): exact geometry, frame count and
  frame rate; **low-resolution consistency** (your output, downscaled back to the input size
  and slightly blurred, must match the blurred input: invented structure the input does not
  support fails it); **colour** (mean CIEDE2000 against the original at most 6).
- **Score** = 0.55 × fidelity + 0.15 × temporal consistency + 0.30 × no-reference quality,
  each placed between the weakest (0) and the strongest (1) of three public baseline upscalers:
  - *fidelity*: DISTS and LPIPS against the original on sampled full 1080p frames;
  - *temporal*: warp error under the original's own motion, relative to the original's
    (flicker costs points; being smoother than the original earns nothing extra);
  - *no-reference quality*: CLIP-IQA, capped at the original's own value + 5 % ("sharper
    than reality" earns nothing).

The code is `vidaio/scoring/upscaling_generative.py` and
`vidaio/scoring/upscaling_generative_formula.py` in the VidAIO repository.

## Making it earn

Replace `StubGenerator` in `generative_example.py` with a real generative video upscaler,
load its weights from `/app/weights` (downloaded and sha256-checked in the Dockerfile), and
batch frames in temporal windows if it is a video model. Keep every frame, in order; keep
colours and composition faithful to the input (the gates), and avoid temporal flicker.

## Licences: permissive only

Everything that makes your output, **code and weights**, must be Apache-2.0, MIT or BSD
licensed: the winner's model is deployed commercially. Conditional or community licences
(for example the CogVideoX licence or the LTX community licence), non-commercial,
academic-use-only, territory-restricted or unlicensed code or weights are **not eligible**. List every
component in `LICENSES.md`, and pin every download by content (a sha256 check in the same
`RUN` step, a 40-hex commit or a Hugging Face revision). Check before enrolling:

```sh
python scripts/competition_precheck.py /path/to/your/repo --commercial
```
