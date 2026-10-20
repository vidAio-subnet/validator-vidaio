# Upscaling example contender

A minimal, working entry for the upscaling competition track (faithful scoring): the
input/output contract end to end and a starting point for your own entry. CPU only and
deterministic; a few seconds per 4-second 1080p item on a laptop.

| File | Purpose |
| --- | --- |
| `Dockerfile` | digest-pinned Python 3.13 slim + static ffmpeg 9.0 |
| `run.sh` | the competition entry point: iterates the items, enforces time and size budgets |
| `upscaling_example.py` | the algorithm for one item (stdlib; ffmpeg via subprocess) |
| `test_local.py` | self-test on a synthetic item, no Docker needed |

## The upscaling track contract

The evaluator runs `/bin/sh /app/run.sh <input_dir> /output` in a sandbox with no
network, a read-only root filesystem (only `/tmp` and the output directory are writable)
and no injected environment variables.
Read the input folder from the first argument: on the live evaluator it is
`/evaluation-inputs/clips`, and it holds only the items and their task files, nothing
else.

- **Input**: every regular, non-hidden file in the input directory is one item, named
  by its 64-hex sha256 with no extension: a sealed 1080p original, degraded and
  downscaled by 2 (to 540p) or by 4 (to 270p), H.264 in MP4, about four seconds at a
  constant frame rate.
- **Task file**: next to each item, the hidden file
  `.vidaio-next-upscale-task-<sha256>` holds the task as JSON, for example
  `{"target_height":1080,"target_width":1920,"upscale_factor":2}`. Originals are
  landscape or portrait, so the target is 1920x1080 or 1080x1920: always read it from
  this file.
- **Output**: one file per item in the output directory, under the same name: an MP4 with
  a single H.264, HEVC, VP9 or AV1 stream, yuv420p, **exactly the target width and
  height**, the input's frame count and frame rate. Plain regular files, mode 0644. Tag
  the stream BT.709 limited range (`-colorspace bt709 -color_primaries bt709 -color_trc
  bt709 -color_range tv`); an untagged stream is read as exactly that.
- `run.sh` must exit 0. A non-zero exit, a timeout or an oversized output directory
  (2 GiB for all files together, 512 MiB per file) zeroes the whole batch; a missing
  output zeroes only that item.

Every output frame is scored against the original it must restore, with PSNR, SSIM and
LPIPS, relative to the answer the validator computes for free: the input Lanczos-upscaled
to the target size. Each term credits only the gain over that floor, as a share of what is
left to gain:

```
s_psnr  = clamp((psnr - psnr_floor) / 6 dB, 0, 1)
s_ssim  = clamp((ssim - ssim_floor) / (1 - ssim_floor), 0, 1)
s_lpips = clamp((lpips_floor - lpips) / lpips_floor, 0, 1)
score   = 0.3 * s_psnr + 0.3 * s_ssim + 0.4 * s_lpips
```

**This example is the Lanczos floor**, so it scores about zero by design. The code is in
`vidaio/scoring/upscaling_faithful.py` and `vidaio/scoring/upscaling_faithful_formula.py`
of the VidAIO repository.

## Making it earn

Replace `upscale_filter` in `upscaling_example.py`, or the whole ffmpeg step, with a real
restoration model: decode the input to raw frames, run the model, and pipe its frames to
the encoder (the frame-interpolation example shows the raw-frame pipe pattern). Things
that cost entries points:

- **Faithful, not generative.** The score compares your frames with the original. Detail
  that was not there (hallucinated texture, over-sharpening, halos) moves your frames
  away from it and costs PSNR, SSIM and LPIPS. Models trained for restoration fidelity
  beat models trained to look sharp.
- **Undo the real degradation.** The inputs are not clean downscales: expect blur, H.264
  artifacts at web bitrates and, on some items, low-light noise.
- **Keep every frame, in order.** Never resample the frame rate or drop frames; a frame
  count or size mismatch zeroes the item.
- **Encode at high quality.** Your encode is measured too; this example uses x264 CRF 10.
- **Budget time per batch.** The manifest's `batch_timeout_seconds` bounds one whole
  `run.sh` call (`evaluation_batch_size` items); keep a margin. When the manifest carries
  a speed term, your run time also counts in the score.

## Licences: commercial use only

The same rules as the frame-interpolation track (see its README and `docs/MINING.md`):
everything you ship must allow commercial use, your own code is Apache-2.0 or MIT with a
`LICENSE` file at the repository root, every third-party component is listed in
`LICENSES.md`, and every download is pinned by content. Weights published by a model's
authors from its repository take that repository's licence; a file your solution never
loads at run time is not judged.

This example ships its own `LICENSE` (MIT) and `LICENSES.md`; keep both up to date when
you build on it. `python scripts/competition_precheck.py <repo> --commit <sha>
--commercial` checks the files and flags unpinned downloads.

## Testing locally

```
python3 test_local.py
```

Runs `run.sh` on a synthetic item with its task file and checks the target size, the
frame count, the frame rate and the codec. Run from inside the VidAIO repository it also
measures the output with the real faithful upscaling metrics.
