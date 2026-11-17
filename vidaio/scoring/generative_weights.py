"""Pinned model weights of the generative upscaling scorer (dependency-light: no NumPy).

Every file is verified by sha256 BEFORE torch deserializes it, exactly like the PieAPP
weights (vidaio/scoring/backends_real.py). The image build fetches them once
(scripts/verify_release_dependencies.py --preload-media); scoring never downloads.
:func:`generative_weights_digest` names the whole set in every score packet's
``backend_versions["generative_weights"]``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "CLIP_RN50_WEIGHTS",
    "DISTS_WEIGHTS",
    "GENERATIVE_WEIGHTS",
    "VGG16_WEIGHTS",
    "GenerativeWeightsError",
    "PinnedWeights",
    "generative_weights_digest",
    "pinned_weights_path",
]


class GenerativeWeightsError(RuntimeError):
    """A pinned model weight file is missing or does not match its digest."""


# --- pinned weights ------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedWeights:
    """An immutable weight file, verified by sha256 BEFORE torch deserializes it."""

    url: str
    filename: str
    sha256: str
    #: byte-identical copies tried in order when ``url`` cannot be fetched (same sha256)
    mirrors: tuple[str, ...] = ()


#: torchvision's ImageNet VGG16 (the DISTS backbone; LPIPS loads the same file).
VGG16_WEIGHTS = PinnedWeights(
    url="https://download.pytorch.org/models/vgg16-397923af.pth",
    filename="vgg16-397923af.pth",
    sha256="397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0",
)
#: piq 0.8.0's DISTS alpha/beta weights (immutable release asset).
DISTS_WEIGHTS = PinnedWeights(
    url="https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt",
    filename="dists_weights.pt",
    sha256="f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218",
)
#: The original CLIP RN50 release (MIT); its URL path carries the published sha256.
CLIP_RN50_WEIGHTS = PinnedWeights(
    url=(
        "https://openaipublic.azureedge.net/clip/models/"
        "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"
    ),
    filename="clip-RN50-afeb0e10.pt",
    sha256="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
    # piq 0.8.0's snapshot of the same file (verified byte-identical)
    mirrors=("https://github.com/photosynthesis-team/piq/releases/download/v0.7.1/RN50.pt",),
)
GENERATIVE_WEIGHTS = (VGG16_WEIGHTS, DISTS_WEIGHTS, CLIP_RN50_WEIGHTS)

_VERIFIED: set[tuple[str, int, float]] = set()
_VERIFIED_LOCK = threading.Lock()


def generative_weights_digest() -> str:
    """One digest naming every pinned generative weight file (stamped in score packets)."""
    joined = "\n".join(f"{w.filename}={w.sha256}" for w in GENERATIVE_WEIGHTS)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, weights: PinnedWeights) -> None:
    try:
        stat = path.stat()
    except OSError as exc:
        raise GenerativeWeightsError(f"cannot read {weights.filename} at {path}: {exc}") from exc
    key = (str(path), stat.st_size, stat.st_mtime)
    with _VERIFIED_LOCK:
        if key in _VERIFIED:
            return
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise GenerativeWeightsError(f"cannot read {weights.filename} at {path}: {exc}") from exc
    if actual != weights.sha256:
        raise GenerativeWeightsError(
            f"{weights.filename} digest mismatch at {path}: {actual} != {weights.sha256}"
        )
    with _VERIFIED_LOCK:
        _VERIFIED.add(key)


def pinned_weights_path(weights: PinnedWeights, *, download: bool = False) -> Path:
    """The verified local copy of ``weights`` in torch.hub's checkpoint directory.

    ``download=True`` is for the image build (scripts/verify_release_dependencies.py
    --preload-media): a missing file is fetched to a temporary name, verified, then moved
    into place. Scoring never downloads: a missing file is a deployment error."""
    import torch

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    target = checkpoint_dir / weights.filename
    if not target.exists():
        if not download:
            raise GenerativeWeightsError(
                f"{weights.filename} is not cached in {checkpoint_dir}; pre-cache the pinned "
                "generative weights in the scorer/auditor image (verify_release_dependencies.py "
                "--preload-media)"
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="generative-", suffix=".part", dir=checkpoint_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            errors: list[str] = []
            for url in (weights.url, *weights.mirrors):
                try:
                    torch.hub.download_url_to_file(url, str(temporary), progress=False)
                    break
                except Exception as exc:  # noqa: BLE001 - try the next byte-identical copy
                    errors.append(f"{url}: {type(exc).__name__}: {exc}")
            else:
                raise GenerativeWeightsError(
                    f"could not fetch {weights.filename}: " + "; ".join(errors)
                )
            _verify(temporary, weights)
            os.replace(temporary, target)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    _verify(target, weights)
    return target
