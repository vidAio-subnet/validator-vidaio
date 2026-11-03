"""certifi stays on a bundle without the two roots behind the compromised eMudhra intermediates.

httpx, boto3, requests and modal verify TLS against certifi (modal hard-codes it), so an
older lock would let anyone holding those intermediate keys impersonate S3, Modal or GitHub.
"""

from __future__ import annotations

from importlib.metadata import version

import certifi

FIXED = (2026, 10, 19)
DISTRUSTED_ROOTS = ("CN=AAA Certificate Services", "CN=emSign ECC Root CA - G3")


def test_certifi_is_at_least_the_fixed_release():
    assert tuple(int(p) for p in version("certifi").split(".")[:3]) >= FIXED


def test_certifi_bundle_omits_the_distrusted_roots():
    bundle = open(certifi.where(), encoding="utf-8").read()
    for root in DISTRUSTED_ROOTS:
        assert root not in bundle, root
