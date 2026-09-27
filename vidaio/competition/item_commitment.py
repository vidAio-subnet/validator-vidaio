"""Canonical commitment for one hidden upscaling competition item.

The manifest is anchored before enrollment.  Its ordered
``evaluation_item_commitments`` list therefore binds the exact pristine reference,
the exact low-resolution bytes exposed to miner code, the scale factor, and the
exact output geometry before any contender runs.  The database copies these values
for orchestration, but those copies are never the auditor's trust root.
"""

from __future__ import annotations

import hashlib
import json
import re

EVALUATION_ITEM_COMMITMENT_V1_DOMAIN = "vidaio.competition.evaluation-item.v1"
EVALUATION_ITEM_COMMITMENT_DOMAIN = "vidaio.competition.evaluation-item.v2"
COMPRESSION_ITEM_COMMITMENT_DOMAIN = "vidaio.competition.compression-item.v1"

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_UPSCALE_FACTORS = frozenset({2, 4})


def evaluation_item_preimage(
    *,
    competition_id: str,
    item_index: int,
    reference_sha256: str,
    input_sha256: str,
    upscale_factor: int,
    target_width: int | None = None,
    target_height: int | None = None,
) -> bytes:
    """Return the exact canonical preimage committed by an upscaling manifest.

    Calls without target geometry deliberately retain the v1 preimage so already
    anchored manifests remain verifiable after migration. New evaluation items
    are required by the repository to provide both dimensions and therefore use
    v2. Supplying only one dimension is always an error.
    """
    if not competition_id:
        raise ValueError("competition_id must be non-empty")
    if item_index < 0:
        raise ValueError("item_index must be non-negative")
    for field, digest in (
        ("reference_sha256", reference_sha256),
        ("input_sha256", input_sha256),
    ):
        if _SHA256_HEX.fullmatch(digest) is None:
            raise ValueError(f"{field} must be lowercase sha256 hex")
    if reference_sha256 == input_sha256:
        raise ValueError("upscaling reference and miner input must be distinct")
    if upscale_factor not in _UPSCALE_FACTORS:
        raise ValueError(
            f"unsupported upscale_factor {upscale_factor}; expected one of "
            f"{sorted(_UPSCALE_FACTORS)}"
        )
    if (target_width is None) != (target_height is None):
        raise ValueError("target_width and target_height must appear together")
    payload: dict[str, str | int] = {
        "competition_id": competition_id,
        "domain": EVALUATION_ITEM_COMMITMENT_V1_DOMAIN,
        "input_sha256": input_sha256,
        "item_index": item_index,
        "reference_sha256": reference_sha256,
        "upscale_factor": upscale_factor,
    }
    if target_width is not None and target_height is not None:
        for field, dimension in (
            ("target_width", target_width),
            ("target_height", target_height),
        ):
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(f"{field} must be a positive integer")
        payload.update(
            {
                "domain": EVALUATION_ITEM_COMMITMENT_DOMAIN,
                "target_width": target_width,
                "target_height": target_height,
            }
        )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def evaluation_item_commitment(
    *,
    competition_id: str,
    item_index: int,
    reference_sha256: str,
    input_sha256: str,
    upscale_factor: int,
    target_width: int | None = None,
    target_height: int | None = None,
) -> str:
    """sha256 of :func:`evaluation_item_preimage`."""
    return hashlib.sha256(
        evaluation_item_preimage(
            competition_id=competition_id,
            item_index=item_index,
            reference_sha256=reference_sha256,
            input_sha256=input_sha256,
            upscale_factor=upscale_factor,
            target_width=target_width,
            target_height=target_height,
        )
    ).hexdigest()


__all__ = [
    "EVALUATION_ITEM_COMMITMENT_DOMAIN",
    "EVALUATION_ITEM_COMMITMENT_V1_DOMAIN",
    "evaluation_item_commitment",
    "evaluation_item_preimage",
]


def compression_item_commitment(
    *, competition_id: str, item_index: int, input_sha256: str
) -> str:
    """Commitment to one hidden COMPRESSION item: the exact bytes contenders receive.

    A compression item's reference is its input, so the digest of those bytes plus the
    ordered index inside the named competition is the whole preimage. The ordered list
    lives in the manifest digest anchored before enrollment, which lets anyone prove
    afterwards which hidden clips were evaluated and that none was swapped.
    """
    if not competition_id:
        raise ValueError("competition_id must be non-empty")
    if type(item_index) is not int or item_index < 0:
        raise ValueError("item_index must be a non-negative integer")
    if _SHA256_HEX.fullmatch(input_sha256) is None:
        raise ValueError("input_sha256 must be lowercase sha256 hex")
    preimage = json.dumps(
        {
            "competition_id": competition_id,
            "domain": COMPRESSION_ITEM_COMMITMENT_DOMAIN,
            "input_sha256": input_sha256,
            "item_index": item_index,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


REMOVAL_ITEM_COMMITMENT_DOMAIN = "vidaio.competition.removal-item.v1"
#: Tracks whose scorer reference differs from the miner input: the pristine
#: reference is sealed during the competition and released at completion so a
#: keyless auditor can recompute every packet.
SEALED_REFERENCE_TRACKS = frozenset({"upscaling", "removal"})
#: The served removal input carries the per-frame mask as this video stream
#: (stream 0 = the frames to repair, stream 1 = the mask).  The index is part of the
#: committed preimage so a scorer can never be pointed at another stream.
REMOVAL_MASK_STREAM_INDEX = 1


def removal_item_commitment(
    *,
    competition_id: str,
    item_index: int,
    input_sha256: str,
    reference_sha256: str,
    mask_stream_index: int = REMOVAL_MASK_STREAM_INDEX,
) -> str:
    """Commitment to one hidden OBJECT-REMOVAL item.

    The preimage binds the exact two-stream input contenders receive (frames with the
    object + the mask), the sealed clean reference the masked region is scored
    against, and the mask stream index.  The reference is released only after the
    competition, so the ordered list anchored in the manifest proves that no item or
    reference was swapped in between.
    """
    if not competition_id:
        raise ValueError("competition_id must be non-empty")
    if type(item_index) is not int or item_index < 0:
        raise ValueError("item_index must be a non-negative integer")
    for field, digest in (
        ("input_sha256", input_sha256),
        ("reference_sha256", reference_sha256),
    ):
        if _SHA256_HEX.fullmatch(digest) is None:
            raise ValueError(f"{field} must be lowercase sha256 hex")
    if input_sha256 == reference_sha256:
        raise ValueError("removal reference and miner input must be distinct")
    if mask_stream_index != REMOVAL_MASK_STREAM_INDEX:
        raise ValueError(
            f"removal mask stream index must be {REMOVAL_MASK_STREAM_INDEX}"
        )
    preimage = json.dumps(
        {
            "competition_id": competition_id,
            "domain": REMOVAL_ITEM_COMMITMENT_DOMAIN,
            "input_sha256": input_sha256,
            "item_index": item_index,
            "mask_stream_index": mask_stream_index,
            "reference_sha256": reference_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()
