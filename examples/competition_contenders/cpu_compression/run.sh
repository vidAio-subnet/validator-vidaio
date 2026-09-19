#!/bin/sh
# Competition contract: /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
#   - every regular file in INPUT_DIR is one clip, named by its 64-hex sha256;
#   - write exactly one MP4 per clip into OUTPUT_DIR under the SAME name;
#   - no network, read-only root filesystem: work only under OUTPUT_DIR and /tmp;
#   - outputs are regular files (no links) readable by the evaluator (mode 0644);
#   - exit 0 only when every clip was produced.
# Scoring rewards fewer bytes at VMAF >= the published threshold; the frame count and
# geometry of the input must be preserved.
set -eu
# Outputs must be plain regular files the evaluator can read after the run.
umask 022

[ "$#" -eq 2 ] || { echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2; exit 64; }
input_dir=$1
output_dir=$2
# shellcheck disable=SC1091
. "${CONTENDER_APP_DIR:-/app}/variant.env"
: "${CODEC:?}" "${CRF:?}" "${PRESET:?}" "${MODE:?}"
TARGET_VMAF=${TARGET_VMAF:-91}
THREADS=$(nproc 2>/dev/null || echo 4)

encode() {  # encode <input> <crf> <output>
    case "$CODEC" in
        x264)   set -- "$1" "$2" "$3" -c:v libx264 -preset "$PRESET" -crf "$2" ;;
        x265)   set -- "$1" "$2" "$3" -c:v libx265 -preset "$PRESET" -crf "$2" -x265-params log-level=error -tag:v hvc1 ;;
        vp9)    set -- "$1" "$2" "$3" -c:v libvpx-vp9 -b:v 0 -crf "$2" -cpu-used "$PRESET" -row-mt 1 ;;
        svtav1) set -- "$1" "$2" "$3" -c:v libsvtav1 -preset "$PRESET" -crf "$2" ;;
        *) echo "unknown codec: $CODEC" >&2; exit 65 ;;
    esac
    in=$1; out=$3; shift 3
    ffmpeg -nostdin -hide_banner -loglevel error -threads "$THREADS" -i "$in" \
        -map 0:v:0 -an -pix_fmt yuv420p "$@" -movflags +faststart -f mp4 -y "$out"
}

measure_vmaf() {  # measure_vmaf <distorted> <reference> -> prints the pooled mean
    ffmpeg -nostdin -hide_banner -threads "$THREADS" -i "$1" -i "$2" \
        -lavfi "[0:v][1:v]libvmaf=n_threads=$THREADS" -f null - 2>&1 \
        | sed -n 's/.*VMAF score: \([0-9.]*\).*/\1/p' | tail -n 1
}

have_vmaf=0
if [ "$MODE" = search ] && ffmpeg -hide_banner -filters 2>/dev/null | grep -q libvmaf; then
    have_vmaf=1
fi

found=0
for input_path in "$input_dir"/*; do
    [ -f "$input_path" ] || continue
    name=${input_path##*/}
    case "$name" in .*) continue ;; esac
    work=$(mktemp -d "$output_dir/.work-XXXXXX")
    best=$work/best.mp4
    if [ "$have_vmaf" -eq 1 ]; then
        # Highest CRF (fewest bytes) that still clears the quality target: bisection.
        low=${CRF_MIN:-20}; high=${CRF_MAX:-55}
        while [ "$low" -le "$high" ]; do
            mid=$(( (low + high) / 2 ))
            encode "$input_path" "$mid" "$work/try.mp4"
            score=$(measure_vmaf "$work/try.mp4" "$input_path")
            if [ -n "$score" ] && awk "BEGIN{exit !($score >= $TARGET_VMAF)}"; then
                mv "$work/try.mp4" "$best"; low=$(( mid + 1 ))
            else
                high=$(( mid - 1 ))
            fi
        done
    fi
    [ -s "$best" ] || encode "$input_path" "$CRF" "$best"
    [ -s "$best" ] || { echo "empty output for $name" >&2; exit 70; }
    mv "$best" "$output_dir/$name"
    chmod 0644 "$output_dir/$name"
    rm -rf -- "$work"
    found=1
    echo "done $name codec=$CODEC mode=$MODE" >&2
done
[ "$found" -eq 1 ] || { echo "no input clips" >&2; exit 66; }
