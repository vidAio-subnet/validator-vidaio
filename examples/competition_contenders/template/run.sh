#!/bin/sh
# Modal competition contract: /app/run.sh INPUT_DIR OUTPUT_DIR.
set -eu
umask 077

if [ "$#" -ne 2 ]; then
    echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2
    exit 64
fi

input_dir=$1
output_dir=$2
app_dir=${VIDAIO_NEXT_APP_DIR:-/app}
gpu_filter=${VIDAIO_NEXT_GPU_FILTER_BIN:-"$app_dir/gpu_transform"}
ffmpeg_bin=${VIDAIO_NEXT_FFMPEG_BIN:-ffmpeg}
ffprobe_bin=${VIDAIO_NEXT_FFPROBE_BIN:-ffprobe}
nvidia_smi_bin=${VIDAIO_NEXT_NVIDIA_SMI_BIN:-nvidia-smi}

# variant.env is immutable submission input selected before the repo is committed.
# shellcheck disable=SC1091
. "$app_dir/variant.env"

: "${VIDAIO_NEXT_TRACK:?missing track profile}"
: "${VIDAIO_NEXT_VARIANT:?missing variant profile}"
: "${VIDAIO_NEXT_SCALE:?missing scale profile}"
: "${VIDAIO_NEXT_INTERPOLATION:?missing interpolation profile}"
: "${VIDAIO_NEXT_SHARPEN:?missing sharpen profile}"
: "${VIDAIO_NEXT_CRF:?missing CRF profile}"
: "${VIDAIO_NEXT_PRESET:?missing preset profile}"

case "$VIDAIO_NEXT_TRACK:$VIDAIO_NEXT_SCALE" in
    compression:1|upscaling:committed) ;;
    *) echo "invalid track/scale profile" >&2; exit 65 ;;
esac
case "$VIDAIO_NEXT_INTERPOLATION" in
    nearest|bilinear) ;;
    *) echo "invalid interpolation profile" >&2; exit 65 ;;
esac
[ -d "$input_dir" ] || { echo "input directory missing" >&2; exit 66; }
[ -d "$output_dir" ] || { echo "output directory missing" >&2; exit 66; }
[ -x "$gpu_filter" ] || { echo "GPU transform missing" >&2; exit 69; }

# Two independent checks: the NVIDIA control plane must see a device, and the
# helper must allocate device memory + launch/synchronize a CUDA kernel. There is
# no CPU fallback that can claim a GPU run.
"$nvidia_smi_bin" --query-gpu=name,uuid --format=csv,noheader >/dev/null
"$gpu_filter" --probe >/dev/null

work_dir=
decoder_pid=
filter_pid=
encoder_pid=
cleanup() {
    for pid in "$decoder_pid" "$filter_pid" "$encoder_pid"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
    done
    [ -n "$work_dir" ] && rm -rf -- "$work_dir"
}
trap cleanup EXIT HUP INT TERM

found=0
for input_path in "$input_dir"/*; do
    [ -f "$input_path" ] || continue
    name=${input_path##*/}
    case "$name" in
        *[!0-9a-f]*) echo "input name is not lowercase sha256: $name" >&2; exit 65 ;;
    esac
    [ "${#name}" -eq 64 ] || {
        echo "input name is not 64-character sha256: $name" >&2
        exit 65
    }
    task_scale=$VIDAIO_NEXT_SCALE
    if [ "$VIDAIO_NEXT_TRACK" = upscaling ]; then
        task_path=$input_dir/.vidaio-next-upscale-task-$name
        [ -f "$task_path" ] && [ -r "$task_path" ] || {
            echo "committed upscale-task sidecar missing for input: $name" >&2
            exit 66
        }
        task_contract=$(sed -n '1p' "$task_path")
        task_height=$(printf '%s\n' "$task_contract" | sed -n \
            's/^{"target_height":\([0-9][0-9]*\),"target_width":[0-9][0-9]*,"upscale_factor":[24]}$/\1/p')
        task_width=$(printf '%s\n' "$task_contract" | sed -n \
            's/^{"target_height":[0-9][0-9]*,"target_width":\([0-9][0-9]*\),"upscale_factor":[24]}$/\1/p')
        task_scale=$(printf '%s\n' "$task_contract" | sed -n \
            's/^{"target_height":[0-9][0-9]*,"target_width":[0-9][0-9]*,"upscale_factor":\([24]\)}$/\1/p')
        expected_contract=$(printf \
            '{"target_height":%s,"target_width":%s,"upscale_factor":%s}' \
            "$task_height" "$task_width" "$task_scale")
        [ -n "$task_height" ] && [ -n "$task_width" ] \
            && [ "$task_contract" = "$expected_contract" ] \
            && [ "$(wc -c <"$task_path")" -eq "$((${#task_contract} + 1))" ] || {
            echo "invalid committed upscale-task sidecar for input: $name" >&2
            exit 65
        }
        case "$task_scale" in
            2|4) ;;
            *) echo "unsupported committed upscale factor: $task_scale" >&2; exit 65 ;;
        esac
    fi
    output_path=$output_dir/$name
    [ ! -e "$output_path" ] || {
        echo "refusing to overwrite output: $output_path" >&2
        exit 73
    }

    geometry=$("$ffprobe_bin" -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=p=0:s=x "$input_path")
    width=${geometry%x*}
    height=${geometry#*x}
    case "$width:$height" in
        *[!0-9:]*|:*|*:) echo "invalid input geometry: $geometry" >&2; exit 65 ;;
    esac
    fps=$("$ffprobe_bin" -v error -select_streams v:0 \
        -show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$input_path")
    case "$fps" in
        ""|0/0|*[!0-9./]*) echo "invalid input frame rate: $fps" >&2; exit 65 ;;
    esac
    if [ "$VIDAIO_NEXT_TRACK" = upscaling ]; then
        output_width=$task_width
        output_height=$task_height
    else
        output_width=$width
        output_height=$height
    fi

    work_dir=$(mktemp -d "$output_dir/.vidaio-next-${name}.XXXXXX")
    decode_fifo=$work_dir/decoded.rgb
    encode_fifo=$work_dir/transformed.rgb
    temporary_output=$work_dir/output.mp4
    mkfifo "$decode_fifo" "$encode_fifo"

    "$ffmpeg_bin" -nostdin -hide_banner -loglevel error \
        -i "$input_path" -map 0:v:0 -an -f rawvideo -pix_fmt rgb24 \
        -y "$decode_fifo" &
    decoder_pid=$!
    "$gpu_filter" "$width" "$height" "$output_width" "$output_height" \
        "$VIDAIO_NEXT_INTERPOLATION" "$VIDAIO_NEXT_SHARPEN" \
        <"$decode_fifo" >"$encode_fifo" &
    filter_pid=$!
    "$ffmpeg_bin" -nostdin -hide_banner -loglevel error \
        -f rawvideo -pix_fmt rgb24 -video_size "${output_width}x${output_height}" \
        -framerate "$fps" -i "$encode_fifo" -an -c:v libx264 \
        -preset "$VIDAIO_NEXT_PRESET" -crf "$VIDAIO_NEXT_CRF" \
        -pix_fmt yuv420p -movflags +faststart -f mp4 "$temporary_output" &
    encoder_pid=$!

    status=0
    wait "$decoder_pid" || status=1
    wait "$filter_pid" || status=1
    wait "$encoder_pid" || status=1
    decoder_pid=
    filter_pid=
    encoder_pid=
    [ "$status" -eq 0 ] || { echo "media pipeline failed" >&2; exit 70; }
    [ -s "$temporary_output" ] || { echo "empty media output" >&2; exit 70; }
    mv "$temporary_output" "$output_path"
    rm -rf -- "$work_dir"
    work_dir=
    found=1
    printf '%s\n' \
        "vidaio-next contender complete track=$VIDAIO_NEXT_TRACK variant=$VIDAIO_NEXT_VARIANT input=$name factor=$task_scale geometry=${width}x${height} output=${output_width}x${output_height}" >&2
done

[ "$found" -eq 1 ] || { echo "no digest-named inputs" >&2; exit 66; }
trap - EXIT HUP INT TERM
