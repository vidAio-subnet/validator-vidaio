#!/bin/sh
# Competition contract, streaming compression: /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
#   - every regular, non-hidden file in INPUT_DIR is one item, named by its 64-hex sha256
#     with no extension: a lossless clip (FFV1 in Matroska);
#   - write exactly one encoded file per item into OUTPUT_DIR under the SAME name: one
#     H.264, HEVC, VP9 or AV1 stream with the input's size, frame count and frame rate,
#     inside the clip's bitrate budget (kbps = BITS_PER_PIXEL * width * height * fps / 1000,
#     BITS_PER_PIXEL from the competition manifest), a keyframe at least every
#     MAX_KEYFRAME_INTERVAL seconds, and no window of PEAK_WINDOW seconds above PEAK_RATIO x
#     the budget;
#   - no network, read-only root filesystem: write only under OUTPUT_DIR and /tmp;
#   - outputs are plain regular files readable by the evaluator (mode 0644);
#   - a non-zero exit zeroes the whole batch, so a failing item is logged and skipped
#     (it scores zero on its own) and the script still exits 0.
# The streaming rules of the competition you enter (its manifest's "streaming_rules"):
BITS_PER_PIXEL=${BITS_PER_PIXEL:-0.05}
MAX_KEYFRAME_INTERVAL=${MAX_KEYFRAME_INTERVAL:-2.0}
PEAK_WINDOW=${PEAK_WINDOW:-2.0}
PEAK_RATIO=${PEAK_RATIO:-1.5}
# No `set -e` on purpose: one bad item must not abort the batch.
set -u
# Outputs must be plain regular files the evaluator can read after the run.
umask 022

[ "$#" -eq 2 ] || { echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2; exit 64; }
input_dir=$1
output_dir=$2
app_dir=${CONTENDER_APP_DIR:-/app}
python=${PYTHON:-python3}
# The sandbox injects no environment variables; do not depend on the caller's PATH.
PATH=/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}
PYTHONDONTWRITEBYTECODE=1
export PATH PYTHONDONTWRITEBYTECODE

# Time limits. The manifest's batch_timeout_seconds bounds this whole script and a
# batch holds evaluation_batch_size items: keep BATCH_BUDGET_SECONDS below the timeout.
# A 10-second 1080p item takes well under a minute on 8 cores.
ITEM_TIMEOUT_SECONDS=${ITEM_TIMEOUT_SECONDS:-300}
BATCH_BUDGET_SECONDS=${BATCH_BUDGET_SECONDS:-780}
MIN_ITEM_SECONDS=${MIN_ITEM_SECONDS:-30}

is_item() {  # regular file, not hidden
    [ -f "$1" ] || return 1
    case "${1##*/}" in .*) return 1 ;; esac
}
total=0
for input_path in "$input_dir"/*; do
    is_item "$input_path" && total=$((total + 1))
done
[ "$total" -gt 0 ] || { echo "no input items" >&2; exit 66; }

have_timeout=0
command -v timeout >/dev/null 2>&1 && have_timeout=1
start=$(date +%s)
index=0; done_count=0; failed=0; skipped=0
for input_path in "$input_dir"/*; do
    is_item "$input_path" || continue
    name=${input_path##*/}
    index=$((index + 1))
    left=$((BATCH_BUDGET_SECONDS - ($(date +%s) - start)))
    if [ "$left" -lt "$MIN_ITEM_SECONDS" ]; then
        echo "skip $name: batch time budget exhausted" >&2
        skipped=$((skipped + 1))
        continue
    fi
    limit=$ITEM_TIMEOUT_SECONDS
    [ "$left" -lt "$limit" ] && limit=$left
    if [ "$have_timeout" -eq 1 ]; then
        timeout -k 10 "$limit" "$python" "$app_dir/streaming_example.py" \
            --bits-per-pixel "$BITS_PER_PIXEL" --max-keyframe-interval "$MAX_KEYFRAME_INTERVAL" \
            --peak-window "$PEAK_WINDOW" --peak-ratio "$PEAK_RATIO" "$input_path" "$output_dir/$name"
    else
        "$python" "$app_dir/streaming_example.py" --bits-per-pixel "$BITS_PER_PIXEL" \
            --max-keyframe-interval "$MAX_KEYFRAME_INTERVAL" --peak-window "$PEAK_WINDOW" \
            --peak-ratio "$PEAK_RATIO" "$input_path" "$output_dir/$name"
    fi
    rc=$?
    # The module removes its temporary file itself; this only matters after a SIGKILL.
    rm -f -- "$output_dir"/.streaming-partial-* 2>/dev/null
    if [ "$rc" -eq 0 ] && [ -s "$output_dir/$name" ]; then
        done_count=$((done_count + 1))
        echo "done $name" >&2
    else
        failed=$((failed + 1))
        [ "$rc" -eq 124 ] && echo "timeout after ${limit}s: $name" >&2
        echo "FAILED $name (exit $rc): no output, this item scores zero" >&2
    fi
done
echo "streaming example: $done_count done, $failed failed, $skipped skipped" >&2
exit 0
