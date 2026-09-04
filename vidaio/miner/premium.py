"""Premium solution variant — not part of the public build.

The `premium` variant is not part of the public build. This stub keeps
the import surface (`DEVICE_PREFIX`, `PremiumEncodeError`,
`run_premium_compression`) so the shared worker code imports cleanly; selecting
the variant in a public deployment fails with a clear error instead of
pretending to implement it. Public miners run the `quality` / `balanced` /
`compact` variants, which are fully implemented here.
"""

from __future__ import annotations

DEVICE_PREFIX = "abav1:"


class PremiumEncodeError(RuntimeError):
    """The premium variant is unavailable in the public build."""


def run_premium_compression(*_args: object, **_kwargs: object) -> str:
    raise PremiumEncodeError(
        "the 'premium' solution variant is not part of the public build; "
        "run one of: quality, balanced, compact"
    )
