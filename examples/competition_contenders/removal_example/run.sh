#!/bin/sh
# Competition contract, object-removal track: /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
#   - every regular file in INPUT_DIR is one item, named by its 64-hex sha256 with no
#     extension: a Matroska file whose video stream 0 is the clip and stream 1 the
#     per-frame mask (luma > 127 = reconstruct);
#   - write exactly one MP4 per item into OUTPUT_DIR under the SAME name: one H.264
#     stream, same width, height, frame count and frame rate, pixels outside the mask
#     unchanged;
#   - no network, read-only root filesystem: write only under OUTPUT_DIR and /tmp;
#   - outputs are plain regular files readable by the evaluator (mode 0644);
#   - a non-zero exit zeroes the whole batch, so a failing item is logged and skipped
#     (it scores zero on its own) and the script still exits 0.
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
# A 720p, 5-second item takes well under a minute on 4 cores.
ITEM_TIMEOUT_SECONDS=${ITEM_TIMEOUT_SECONDS:-300}
BATCH_BUDGET_SECONDS=${BATCH_BUDGET_SECONDS:-780}
MIN_ITEM_SECONDS=${MIN_ITEM_SECONDS:-30}
# Size limits. All outputs of a batch together must stay under 2 GiB (512 MiB each).
# Outputs are lossless (about 20 MB for a 720p, 5-second real clip, up to about
# 110 MB for heavy grain), so each item gets a fair share of what is left and falls
# back to a near-lossless encode only when its lossless file would not fit.
OUTPUT_BUDGET_BYTES=${OUTPUT_BUDGET_BYTES:-1800000000}
MAX_OUTPUT_BYTES=500000000

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
    # shellcheck disable=SC2046
    set -- $(du -sk "$output_dir" 2>/dev/null) 0
    allowance=$(((OUTPUT_BUDGET_BYTES - $1 * 1024) / (total - index + 1)))
    [ "$allowance" -gt "$MAX_OUTPUT_BYTES" ] && allowance=$MAX_OUTPUT_BYTES
    [ "$allowance" -gt 0 ] || allowance=1
    left=$((BATCH_BUDGET_SECONDS - ($(date +%s) - start)))
    if [ "$left" -lt "$MIN_ITEM_SECONDS" ]; then
        echo "skip $name: batch time budget exhausted" >&2
        skipped=$((skipped + 1))
        continue
    fi
    limit=$ITEM_TIMEOUT_SECONDS
    [ "$left" -lt "$limit" ] && limit=$left
    if [ "$have_timeout" -eq 1 ]; then
        timeout -k 10 "$limit" "$python" "$app_dir/removal_example.py" \
            --max-bytes "$allowance" "$input_path" "$output_dir/$name"
    else
        "$python" "$app_dir/removal_example.py" --max-bytes "$allowance" "$input_path" "$output_dir/$name"
    fi
    rc=$?
    # The module removes its temporary file itself; this only matters after a SIGKILL.
    rm -f -- "$output_dir"/.removal-partial-* 2>/dev/null
    if [ "$rc" -eq 0 ] && [ -s "$output_dir/$name" ]; then
        done_count=$((done_count + 1))
        echo "done $name" >&2
    else
        failed=$((failed + 1))
        [ "$rc" -eq 124 ] && echo "timeout after ${limit}s: $name" >&2
        echo "FAILED $name (exit $rc): no output, this item scores zero" >&2
    fi
done
echo "removal example: $done_count done, $failed failed, $skipped skipped" >&2
exit 0
