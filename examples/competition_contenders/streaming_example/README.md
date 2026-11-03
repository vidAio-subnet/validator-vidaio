# Streaming compression example contender

A minimal, working entry for the streaming compression competition track: the
input/output contract end to end and a starting point for your own entry. CPU only (x264).

| File | Purpose |
| --- | --- |
| `Dockerfile` | digest-pinned Python 3.13 slim + static ffmpeg 9.0 |
| `run.sh` | the competition entry point: iterates the items, enforces the time budget, carries the competition's streaming rules |
| `streaming_example.py` | the encode for one item (stdlib; ffmpeg/ffprobe via subprocess) |
| `test_local.py` | self-test on a synthetic item, no Docker needed |

## The streaming track contract

The evaluator runs `/bin/sh /app/run.sh <input_dir> /output` in a sandbox with no
network, a read-only root filesystem (only `/tmp` and the output directory are writable)
and no injected environment variables. Read the input folder from the first argument: on
the live evaluator it is `/evaluation-inputs/clips`, and it holds only the items.

- **Input**: every regular file in the input directory is one item, named by its 64-hex
  sha256 with no extension: a lossless clip (FFV1 in Matroska).
- **Output**: one file per item in the output directory, under the same name: a single
  H.264, HEVC, VP9 or AV1 stream with the input's width, height, frame count and frame
  rate. Plain regular files, mode 0644.
- **The budget**: the competition's manifest anchors `streaming_rules`. Each clip's
  bitrate budget is `bits_per_pixel × width × height × fps / 1000` kbps, computed from the
  clip itself, so the same rule gives every clip its own budget.
- **The delivery rules** are measured on your encoded video packets (container overhead is
  not counted). Breaking any one zeroes the item:
  - the average bitrate stays within the budget (1 % slack for rounding);
  - no window of `peak_window_seconds` consecutive frames goes above `peak_ratio` × the budget;
  - the first frame is a keyframe, and keyframes are at most `max_keyframe_interval_seconds` apart, so the stream can be cut into segments.

Within the rules the score is quality: `(VMAF − 60) / 40` clamped to [0, 1], with the
same pinned VMAF model and anti-sharpening checks as the compression track. When the
manifest carries a speed term, your `run.sh` time counts as well (seconds per output
frame, as in the other tracks).

**This example** is a single x264 encode per item:
- a fixed GOP;
- a VBV cap well under the peak rule;
- an average aimed just under the budget, re-encoded lower if its own packet measurement does not fit.

It is a correct starting point, not a competitive one.

## Making it earn

The budget is fixed, so everything that raises VMAF per bit counts:
- the codec (AV1 and HEVC beat H.264 at equal bitrate);
- the preset and tuning;
- two-pass rate control that spends bits where the clip needs them;
- per-clip decisions.

Keep the rules exact:
- measure your own packets, as `streaming_example.py` does;
- leave a margin under the budget and the peak cap;
- keep keyframes on a fixed grid. Scene-cut keyframes are fine only if the gaps still stay within the interval.

Encode speed counts when the manifest has a speed term, so a slower preset must pay for itself in VMAF.

Set the four values at the top of `run.sh` to the `streaming_rules` of the competition you
enter (the sandbox injects no environment variables).

## Licences: commercial use only

The same rules as the other competition tracks (`docs/MINING.md`):
- everything you ship must allow commercial use;
- everything you train or tune on must allow training a model for commercial use
  ("Training data" in `docs/MINING.md`);
- your own code is Apache-2.0 or MIT, with a `LICENSE` file at the repository root;
- every third-party component is listed in `LICENSES.md`;
- every download is pinned by content.

`python scripts/competition_precheck.py <repo> --commit <sha> --commercial` checks the
files and flags unpinned downloads.

## Testing locally

```
python3 test_local.py
```

Runs `run.sh` on a synthetic item and checks the contract. Run from inside the VidAIO
repository, it also measures every delivery rule with the validator's own packet code.
