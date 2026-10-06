# Frame-interpolation example contender

A minimal, working entry for the frame-interpolation competition track: the input/output
contract end to end and a starting point for your own entry. CPU only and deterministic;
a few seconds per 1080p, 5-second item on a laptop.

| File | Purpose |
| --- | --- |
| `Dockerfile` | digest-pinned Python 3.13 slim + static ffmpeg 9.0, numpy pinned by hash |
| `requirements.txt` | the numpy wheel, with sha256 hashes |
| `run.sh` | the competition entry point: iterates the items, enforces time and size budgets |
| `interpolation_example.py` | the algorithm for one item (stdlib + numpy; ffmpeg via subprocess) |
| `test_local.py` | self-test on a synthetic item, no Docker needed |

## The interpolation track contract

The evaluator runs `/bin/sh /app/run.sh /evaluation-inputs /output` in a sandbox with no
network, a read-only root filesystem (only `/tmp` and the output directory are writable)
and no injected environment variables.

- **Input**: every regular file in the input directory is one item, named by its 64-hex
  sha256 with no extension. It is a Matroska file with one video stream (FFV1, yuv420p,
  up to 1920x1080, a few seconds): a genuine high-frame-rate clip with every other frame
  removed, so its frame rate is half the original (for example 30 fps made from 60 fps).
- **Output**: one file per item in the output directory, under the same name: an MP4 with
  a single H.264, HEVC, VP9 or AV1 stream, yuv420p, the same width and height, **twice
  the input frame rate** and **exactly 2N - 1 frames** for N input frames. Frame 2k is
  input frame k, unchanged; frame 2k + 1 is your new frame between input frames k and
  k + 1. Plain regular files, mode 0644.
- **The input frames must come back unchanged**: a mean absolute RGB change above 3.0
  on any input frame zeroes the item. It also catches an output shifted off the frame
  grid. A near-lossless encode (this example: x264 CRF 8) stays well below the limit.
- `run.sh` must exit 0. A non-zero exit, a timeout or an oversized output directory
  (2 GiB for all files together, 512 MiB per file) zeroes the whole batch; a missing
  output zeroes only that item.

Only the inserted frames are scored, against the real frames that were removed, with
PSNR, SSIM and LPIPS, relative to two answers the validator computes for free: repeating
the previous frame (hold) and averaging the two neighbours (blend). Each term credits
only the gain over the better of the two, as a share of what is left to gain:

```
s_psnr  = clamp((psnr - psnr_floor) / 8 dB, 0, 1)
s_ssim  = clamp((ssim - ssim_floor) / (1 - ssim_floor), 0, 1)
s_lpips = clamp((lpips_floor - lpips) / lpips_floor, 0, 1)
score   = 0.3 * s_psnr + 0.3 * s_ssim + 0.4 * s_lpips
```

**This example is the blend**, so it scores about zero by design. The code is in
`vidaio/scoring/interpolation.py` and `vidaio/scoring/interpolation_formula.py` of the
VidAIO repository.

## Making it earn

Replace `insert_frame(previous, following)` in `interpolation_example.py` with a real
interpolation model (it receives the two neighbouring frames as raw yuv420p planes and
returns the new frame in the same layout), or rewrite the item loop around your model.
Things that cost entries points:

- **Drive the model on the exact frame grid.** Popular wrappers skip near-duplicate
  frames, duplicate frames across scene cuts or round the frame rate. Any of these
  changes the frame count or shifts the grid, and the item scores zero.
- **Keep the input frames bit-for-bit or near-lossless.** Do not denoise, sharpen or
  re-grade them: they must match the input within the tolerance above.
- **Large motion and occlusions** are where simple methods fail and where the score is
  decided. Test on fast pans, sports, hands, vehicles and thin structures.
- **Budget time per batch.** The manifest's `batch_timeout_seconds` bounds one whole
  `run.sh` call (`evaluation_batch_size` items); keep a margin.

## Licences: commercial use only

The winning solution of this competition runs on the VidAIO website, so every entry must
be usable commercially end to end:

- everything you ship (code, libraries, model weights) must be under a licence that
  allows commercial use (Apache-2.0, MIT, BSD, ISC, MPL-2.0, CC0/Unlicense, CC-BY-4.0, or
  GPL/LGPL such as ffmpeg). Non-commercial, research-only, no-derivatives, AGPL/SSPL or
  unlicensed components are refused, including fine-tuned, merged, adapted or distilled
  versions of such models. Licences are judged on what you ship, not on training data;
- your own code must be released under Apache-2.0 or MIT (a `LICENSE` file at the
  repository root at the commit you enroll);
- list every third-party component in a `LICENSES.md` at the repository root: name,
  source URL, licence and where it sits in your repository;
- pin every file your Dockerfile downloads by content (a sha256 check in the same `RUN`
  step, a git commit, a Hugging Face revision, `pip install --require-hashes`), so the
  image that is built is the one that was reviewed.

This example ships its own `LICENSE` (MIT) and `LICENSES.md`; keep both up to date when
you build on it. `python scripts/competition_precheck.py <repo> --commit <sha>
--commercial` checks the files and flags unpinned downloads.

Entries are checked in the 24 h after enrollment closes; an entry that breaks these rules
is rejected before evaluation and never scored. A violation found later, until the
competition completes, takes the entry out of the scoring and the result.

## Testing locally

```
python3 test_local.py
```

Runs `run.sh` on a synthetic item and checks the frame grid, the frame rate and the
kept-frame tolerance. Run from inside the VidAIO repository it also measures the output
with the real interpolation metrics.
