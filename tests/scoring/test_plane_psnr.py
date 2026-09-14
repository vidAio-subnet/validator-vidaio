"""Chroma-plane PSNR: exact integer accumulation over canonical yuv420p y4m streams."""

import math

import numpy as np
import pytest

from vidaio.scoring.plane_psnr import (
    IDENTICAL_PLANES_PSNR_DB,
    PlanePsnrError,
    chroma_psnr_uv,
    chroma_residual,
    read_y4m_header,
)


def write_y4m(path, frames, width=8, height=6):
    cw, ch = (width + 1) // 2, (height + 1) // 2
    with open(path, "wb") as fh:
        fh.write(f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420jpeg XYSCSS=420JPEG\n".encode())
        for y, u, v in frames:
            assert y.shape == (height, width) and u.shape == (ch, cw) and v.shape == (ch, cw)
            fh.write(b"FRAME\n")
            fh.write(y.astype(np.uint8).tobytes() + u.astype(np.uint8).tobytes() + v.astype(np.uint8).tobytes())


def planes(seed, width=8, height=6):
    rng = np.random.default_rng(seed)
    cw, ch = (width + 1) // 2, (height + 1) // 2
    return (rng.integers(0, 256, (height, width)), rng.integers(0, 256, (ch, cw)), rng.integers(0, 256, (ch, cw)))


def test_header_parse_and_unsupported_layout(tmp_path):
    path = tmp_path / "a.y4m"
    write_y4m(path, [planes(1)])
    with open(path, "rb") as fh:
        header = read_y4m_header(fh)
    assert (header.width, header.height, header.chroma) == (8, 6, "420jpeg")
    assert header.frame_bytes == 8 * 6 + 2 * 4 * 3
    (tmp_path / "bad.y4m").write_bytes(b"YUV4MPEG2 W8 H6 C444\nFRAME\n")
    with pytest.raises(PlanePsnrError, match="chroma layout"):
        with open(tmp_path / "bad.y4m", "rb") as fh:
            read_y4m_header(fh)


def test_identical_streams_hit_the_finite_cap_and_luma_is_ignored(tmp_path):
    a, b = tmp_path / "a.y4m", tmp_path / "b.y4m"
    y, u, v = planes(2)
    write_y4m(a, [(y, u, v)])
    write_y4m(b, [((y + 40) % 256, u, v)])  # luma differs, chroma identical
    assert chroma_psnr_uv(str(a), str(b)) == IDENTICAL_PLANES_PSNR_DB


def test_psnr_matches_a_direct_integer_computation(tmp_path):
    a, b = tmp_path / "a.y4m", tmp_path / "b.y4m"
    fa = [planes(3), planes(4)]
    fb = [planes(5), planes(6)]
    write_y4m(a, fa)
    write_y4m(b, fb)
    expected = 0.0
    for (ya, ua, va), (yb, ub, vb) in zip(fa, fb):
        sq = int(((ua.astype(int) - ub.astype(int)) ** 2).sum() + ((va.astype(int) - vb.astype(int)) ** 2).sum())
        expected += 10 * math.log10(255 * 255 / (sq / (2 * ua.size)))
    assert chroma_psnr_uv(str(a), str(b)) == pytest.approx(expected / 2, abs=1e-12)
    # Order matters only through the frame pairing, not the operand order.
    assert chroma_psnr_uv(str(b), str(a)) == pytest.approx(expected / 2, abs=1e-12)


def test_residual_sign_tracks_the_closer_source(tmp_path):
    ref, inp, cand = (tmp_path / n for n in ("ref.y4m", "input.y4m", "cand.y4m"))
    y, u, v = planes(7)
    write_y4m(ref, [(y, u, v)])
    write_y4m(inp, [(y, (u + 3) % 256, (v + 3) % 256)])
    write_y4m(cand, [(y, (u + 1) % 256, (v + 1) % 256)])  # nearer the reference
    residual = chroma_residual(str(cand), str(ref), str(inp))
    assert residual.psnr_uv_reference > residual.psnr_uv_input
    assert residual.residual == residual.psnr_uv_reference - residual.psnr_uv_input > 0


@pytest.mark.parametrize("fault", ["frames", "geometry", "truncated"])
def test_mismatched_streams_fail_closed(tmp_path, fault):
    a, b = tmp_path / "a.y4m", tmp_path / "b.y4m"
    write_y4m(a, [planes(8), planes(9)])
    if fault == "frames":
        write_y4m(b, [planes(8)])
        match = "frame count"
    elif fault == "geometry":
        write_y4m(b, [planes(8, 10, 6)], width=10)
        match = "geometry"
    else:
        write_y4m(b, [planes(8), planes(9)])
        data = b.read_bytes()
        b.write_bytes(data[:-5])
        match = "truncated"
    with pytest.raises(PlanePsnrError, match=match):
        chroma_psnr_uv(str(a), str(b))


def test_cancellation_is_honoured(tmp_path):
    a, b = tmp_path / "a.y4m", tmp_path / "b.y4m"
    write_y4m(a, [planes(10)])
    write_y4m(b, [planes(11)])
    with pytest.raises(PlanePsnrError, match="cancelled"):
        chroma_psnr_uv(str(a), str(b), cancelled=lambda: True)
