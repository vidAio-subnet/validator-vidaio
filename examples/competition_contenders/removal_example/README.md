# Object-removal example contender

A minimal, working entry for the object-removal competition track: the input/output
contract end to end and a starting point for your own entry. CPU only and deterministic: a few seconds per 720p, 5-second item
on a laptop, peak memory under 0.5 GB.

| File | Purpose |
| --- | --- |
| `Dockerfile` | digest-pinned Python 3.13 slim + static ffmpeg 9.0, numpy and OpenCV pinned by hash |
| `requirements.txt` | the two Python wheels, with sha256 hashes |
| `run.sh` | the competition entry point: iterates the items, enforces time and size budgets |
| `removal_example.py` | the algorithm for one item (stdlib, numpy, OpenCV; ffmpeg via subprocess) |
| `test_local.py` | self-test on synthetic items, no Docker needed |

## The removal track contract

The evaluator runs `/bin/sh /app/run.sh /evaluation-inputs /output` in a sandbox with no
network, a read-only root filesystem (only `/tmp` and the output directory are
writable) and no injected environment variables.

- **Input**: every regular file in the input directory is one item, named by its
  64-hex sha256 with no extension. It is a Matroska file with two video streams:
  stream 0 is the clip with an object on it (FFV1, yuv420p, up to 1280x720, 24 fps,
  a few seconds), stream 1 is the per-frame mask (FFV1, gray, same size and frame
  count; luma > 127 = reconstruct, 0 = keep). Some frames have an empty mask.
- **Output**: one file per item in the output directory, under the same name: an MP4
  with a single H.264 stream, the same width, height, frame count and frame rate,
  yuv420p. Masked pixels hold a plausible background; everything outside the mask must
  stay as it was (a mean absolute RGB change above 3.0 outside the mask zeroes the
  item). Plain regular files, mode 0644.
- `run.sh` must exit 0. A non-zero exit, a timeout or an oversized output directory
  (2 GiB for all files together, 512 MiB per file) zeroes the whole batch; a missing
  output zeroes only that item.

Only the masked region is scored, against the clean original, and relative to a
free fill the validator computes itself: the per-pixel temporal median of the unmasked
frames. **This example is that free fill**, so it scores about zero by design; an
entry earns only by beating it clearly and by reaching the competition's published
score bars (the metrics are in `vidaio/scoring/removal.py` of the
VidAIO repository).

## What it does

1. Decodes stream 0 as planar YUV 4:2:0 (no colour conversion) and stream 1 as masks.
2. Fills every masked pixel with the temporal median of that pixel over the frames in
   which it is not masked (exact on a static camera, ghosting under motion).
3. Pixels masked in every frame have no clean sample: OpenCV Telea inpainting,
   radius 5, per frame, from the median-filled frame around them.
4. Leaves unmasked pixels and empty-mask frames untouched. A chroma sample is replaced
   when any of the four luma pixels it covers is masked, so the object's colour does
   not bleed into the fill.
5. Encodes losslessly with libx264 (`-qp 0 -g 1`): the decoded output equals the
   composited frames bit for bit, so the outside-mask change is essentially zero
   (about 0.01). Lossless 4:2:0 H.264 uses the High 4:4:4 profiles; ffmpeg decodes it
   (that is what the scorer uses), many hardware and browser decoders do not. Outputs
   are about 20 MB per item on typical footage and up to about 110 MB on heavy grain.
   If a lossless file would not fit the batch's size budget, that item is re-encoded
   with `-crf 10` (near-lossless, about 1 mean abs RGB change on heavy grain).

Items are processed one at a time. A failing item is logged to stderr and skipped with
no file written (outputs are written to a hidden temporary file and renamed only after
they were checked), and the script still exits 0. The time and size budgets are at the
top of `run.sh` (`BATCH_BUDGET_SECONDS`, `ITEM_TIMEOUT_SECONDS`,
`OUTPUT_BUDGET_BYTES`). Nothing is injected at run time, so edit the defaults to fit
the manifest's `batch_timeout_seconds` and batch size.

## Running it locally

Without Docker (needs ffmpeg/ffprobe on `PATH`, numpy and opencv-python-headless):

```sh
python3 test_local.py                      # toy items at 320x240, checks, run.sh end to end
python3 test_local.py --size 1280x720      # timing at the largest frame size
python3 test_local.py --size 1280x720 --background your_clip.mp4   # real footage
```

With Docker, the way the sandbox runs it:

```sh
docker build --platform linux/amd64 -t removal-example .
docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp \
  -v "$PWD/inputs:/evaluation-inputs:ro" -v "$PWD/outputs:/output" \
  removal-example /bin/sh /app/run.sh /evaluation-inputs /output
```

To enter, put these files at the root of a fresh private repository (the `Dockerfile`
must be at the root), commit, and enroll the commit and tree SHA as described in
[docs/MINING.md](../../../docs/MINING.md), "The competition track". Enrolling this example
unchanged earns nothing.

## Doing better

The median cannot follow camera or object motion, and per-frame Telea blurs and
flickers. Directions that address this, roughly in order of effort:

- **Align before you aggregate.** Estimate global motion between frames (OpenCV
  `findTransformECC` or feature matching plus `findHomography`), warp the clean
  samples into the current frame, then take the median or a weighted average of the
  aligned samples. Handles pans and zooms on CPU.
- **Flow-guided propagation.** Complete optical flow inside the hole, then propagate
  real pixels from the nearest frames where they are visible (OpenCV's DIS flow is a
  fast CPU start). This is the core of current video-inpainting methods such as
  ProPainter and E2FGVI, which add learned flow completion and a transformer for the
  regions never seen.
- **Image inpainting for what is never visible**, such as LaMa or diffusion inpainting,
  kept temporally consistent: the scorer caps temporal warp error, so a fill that
  flickers from frame to frame is zeroed.
- **Keep the outside untouched.** Composite only inside the mask and encode
  losslessly or near-losslessly.

Models and weights must be inside the image, because runs have no network. Download
them in the `Dockerfile` (builds have network) and verify a checksum; the repository
itself is capped at 512 MiB. Check the licence of both code and weights before you
ship them: several well-known video-inpainting releases are for non-commercial use
only. Whether a GPU is available is set per competition (`allowed_gpus` in the
manifest).
