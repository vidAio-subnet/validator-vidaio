"""Wire contracts between services (HTTP request/response bodies).

Miner artifacts cross the network as bounded byte streams.  The validator and
organic gateway send a metadata header plus the raw challenge bytes to
``POST /v1/task/artifact``; the miner returns the raw output with its digest and
task binding in response headers.  No peer-supplied path or URL is dereferenced
on either side.  The older JSON path contract remains available only as a
deprecated local/shared-filesystem compatibility route.

Scoring-worker artifacts still use the separately documented path-and-digest
snapshot contract below. Ports are each service's config default:

  scoring worker 8201 · challenge service 8210 · reference miner 8300 ·
  organic gateway 29996 · chainsim 8400 · dashboard 8600 ·
  scoring authority 8700 · audit-results api 8710 · baseline registry 8720 — metrics:
  validator 9101, weight-setter 9102, scoring 9103, orchestrator 9104,
  challenge 9105, miner 9106, gateway 9107, chainsim 9108, dashboard 9109,
  autoupdater 9110, scoring-authority 9111, audit-results-api 9112,
  authority-finalizer 9120, beacon auditor 9121, own-auditor 9122,
  baseline registry 9123.

Canonical routes (one path per contract; every client uses these):

  POST {miner}/v1/task/artifact
                              base64url(MinerArtifactTaskRequest) in
                              X-Vidaio-Task-Metadata + raw bounded input bytes ->
                              raw bounded output bytes with X-Vidaio-* binding
                              headers. This is the production/default validator
                              and gateway contract.
  POST {miner}/v1/task        DEPRECATED local/shared-filesystem JSON
                              MinerTaskRequest -> MinerTaskResponse.
  POST {miner}/task           DEPRECATED alias of the JSON path route.
  GET  {miner}/warrant        -> {"track": ...} the TaskWarrant probe: the one
                              pool that miner identity competes in.
  POST {scoring_worker}/score ScoreRequest -> ScoreResponse.
  GET  {scoring_worker}/healthz -> {"scorer_version": <identity>, ...} — the
                              scorer-identity discovery route (see below).
  POST {challenge_service}/challenge/next  {track, owner?} -> the produced
                              challenge; `owner` (validator identity) is recorded
                              on the challenge and enforced on resolve/list
                              (miner-facing `dispatch` payload + the
                              validator-private reference/input artifacts).
  GET  {challenge_service}/challenges?status=dispatched&older_than_seconds=N
                              -> the dispatched-challenge sweep list; the
                              validator's blind-spot closer for a LOST
                              /challenge/next response (see below).

================================================================================
THE SCORER-IDENTITY CONTRACT (one model, every service)
================================================================================

A score packet is only auditable if everyone agrees on WHICH scorer produced it.
The scoring worker therefore has exactly one name, and it is the worker's to
mint:

    identity = f"{scoring_worker.scorer_version}+{identity_digest[:12]}"

where ``identity_digest`` (vidaio.scoring_worker.service.scorer_identity_digest)
is a sha256 over every configured lever that can change a measured score plus
the complete payout-runtime commitment: verified release runtime marker,
Linux/amd64 policy, deterministic thread policy, native media versions and the
Python/PIQ/PyTorch/OpenCV/NumPy versions. The full runtime commitment digest is
also stamped in ``backend_versions["runtime"]`` and its public preimage is
published on ``GET /healthz``.
Ports, paths, timeouts and concurrency are deliberately NOT in it: they cannot
change a packet, so two identically-scoring deployments must not refuse each
other's work over them.

**Identity is the FULL effective string.** `<name>+<digest12>` — never the bare
configured name — is what gets recorded in packets and audit bundles, pinned by
validators, and committed to by competition manifests and challenge commitments.
A bare name (e.g. "scorer-v1") is not an identity and any caller asserting one
is refused.

The worker publishes its identity on ``GET /healthz`` (``scorer_version``) and
stamps it into every packet it emits. ``ScoreRequest.scorer_version`` is a
caller ASSERTION, never an instruction: absent/empty means "whichever scorer you
are", an equal value means agreement, and anything else is a 409
``scorer_version_mismatch``. Use :func:`fetch_scorer_identity` /
:func:`fetch_scorer_identity_async` to discover it.

Each consumer adopts that identity as follows.

1. **Inference validator** — PIN ON FIRST CONTACT.
   On startup (and again whenever it has no pin yet, e.g. the worker was not up
   when the loop started) the validator reads the identity from ``GET /healthz``
   and pins it in memory. Requests OMIT ``scorer_version`` unless an operator
   pin exists, but every returned packet's ``scorer_version`` MUST equal the
   pinned identity — a mismatch is the existing binding rejection (non-punitive
   to the miner, counted as a scoring failure).
   ``ValidatorConfig.scorer_version``, when non-empty, is an EXPLICIT OPERATOR
   PIN: it is asserted in the request (so the worker 409s a stranger) AND the
   discovered identity must equal it, or the validator fails loudly at startup
   with a config error rather than silently drifting onto another scorer.

2. **Competition orchestrator** — MANIFEST COMMITS TO THE IDENTITY.
   ``CompetitionManifest.scoring_version`` is audit-critical: the manifest
   digest is anchored on chain before enrollment, so it must be the full
   effective identity, not a label. The orchestrator compares the worker's
   advertised identity to the persisted manifest at competition start and again
   before SCORING; disagreement is an INFRA HALT with an explicit reason (never
   a FAILED competition, and never a surprise 409 in the middle of scoring).
   Before any packet can be persisted or ranked, the HTTP client additionally
   requires its track/challenge/item/miner/output/scorer fields to equal the exact
   request and its COMPLETE ``backend_versions`` map to equal the independently
   re-hashed health runtime attestation (including the derived runtime stamp).
   Missing/moved/CUDA backend stamps are infra failures, never delayed audit
   findings. Author manifests with :func:`fetch_scorer_identity` against the
   worker that will actually run.

3. **Challenge service** — THE COMMITMENT PREIMAGE NAMES THE SCORER.
   ``challenge_service.scorer_version`` is bound into every challenge
   commitment's preimage, so it is the same identity concept. When
   ``challenge_service.scoring_worker_url`` is configured the service verifies
   the configured value against the worker's advertised identity at startup;
   on mismatch it logs CRITICAL and REFUSES to produce challenges (503) —
   a commitment that names a scorer nobody runs is unverifiable. With no worker
   URL configured the literal stands and is logged once as UNVERIFIED.

Nobody ever rewrites the worker's stamp: a MEASURED packet's ``scorer_version``
is always the value the worker minted, which is exactly what makes the later
cross-check (packet vs bundle vs manifest vs commitment) meaningful.

RESERVED NAMESPACE: ``orchestrator-zero/*``
-------------------------------------------------------------------------------
Exactly one packet is NOT minted by a scoring worker. When a competition item has
no measurable bytes (the contender produced no output, or the worker rejected the
contender's own output), the competition orchestrator records a gate-failed ZERO
locally instead of feeding an empty file to ffmpeg. That packet asserts no
measurement — ``gate_passed=False`` forces ``score=0.0`` structurally — and it is
attributed HONESTLY rather than stamped with the worker's identity::

    scorer_version = "orchestrator-zero/1+<identity digest[:12]>"

Same ``<name>+<digest12>`` shape, so every consumer parses it; a reserved name, so
nothing can be confused with a measurement. Consequences for consumers:

- such a packet's ``scorer_version`` legitimately DIFFERS from the manifest's
  ``scoring_version``; that is an orchestrator fact, not scorer drift, and must
  not be flagged as a mismatch. Its audit bundle carries the same
  orchestrator-zero identity (so the packet-vs-bundle cross-check still holds)
  while the bundle's manifest still names the committed worker;
- NO scoring worker may advertise or stamp an identity in the
  ``orchestrator-zero/`` namespace. A worker that does is refused, because it
  would make measured packets indistinguishable from these records.

The convention, its digest inputs and the refusal helpers are defined in
``vidaio.competition.orchestrator.zero_packets``
(``is_orchestrator_zero_identity`` / ``assert_not_reserved``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field

from vidaio.challenge.commitment import ChallengeAnchor

SHA256_HEX = r"^[0-9a-f]{64}$"

# Stable production health/metrics ports for the two script-owned epoch loops.
# Their report-mode fallback remains port 0 (an ephemeral test/demo listener), but
# Bittensor startup requires these canonical ports so supervisors, dashboards and
# Prometheus never have to discover an OS-selected port.
AUTHORITY_FINALIZER_METRICS_PORT = 9120
AUDITOR_METRICS_PORT = 9121
OWN_AUDITOR_METRICS_PORT = 9122

#: Remote miner artifact-exchange route.  Clients must never accept an absolute
#: output URL from a miner; the response body itself is the artifact.
MINER_ARTIFACT_ROUTE = "/v1/task/artifact"

#: Request metadata is base64url-encoded canonical JSON so it remains an ASCII
#: header and cannot smuggle CR/LF.  4 KiB raw keeps the encoded header below
#: common 8 KiB proxy limits and bounds JSON parsing independently of the body.
MINER_TASK_METADATA_HEADER = "X-Vidaio-Task-Metadata"
MAX_MINER_TASK_METADATA_BYTES = 4 * 1024

#: Artifact protocol v1 is the deprecated unsigned compatibility contract. V2 is
#: the canonical hotkey-authenticated contract used by every production caller.
MINER_TASK_ID_HEADER = "X-Vidaio-Task-Id"
MINER_OUTPUT_DIGEST_HEADER = "X-Vidaio-Output-SHA256"
MINER_PROCESSING_SECONDS_HEADER = "X-Vidaio-Processing-Seconds"
MINER_ARTIFACT_VERSION_HEADER = "X-Vidaio-Artifact-Version"
MINER_ARTIFACT_VERSION = "1"
MINER_ARTIFACT_AUTH_VERSION = "2"

#: V2 request authentication. The signature covers canonical task metadata plus
#: the exact input size, signer/intended identities, timestamp, and nonce.
MINER_VALIDATOR_HOTKEY_HEADER = "X-Vidaio-Validator-Hotkey"
MINER_HOTKEY_HEADER = "X-Vidaio-Miner-Hotkey"
MINER_REQUEST_TIMESTAMP_HEADER = "X-Vidaio-Request-Timestamp"
MINER_REQUEST_NONCE_HEADER = "X-Vidaio-Request-Nonce"
MINER_INPUT_SIZE_HEADER = "X-Vidaio-Input-Size"
MINER_REQUEST_SIGNATURE_HEADER = "X-Vidaio-Request-Signature"

#: V2 response authentication. Output size is independent of Content-Length so
#: it is explicitly included in the miner-hotkey signature.
MINER_OUTPUT_SIZE_HEADER = "X-Vidaio-Output-Size"
MINER_RESPONSE_SIGNATURE_HEADER = "X-Vidaio-Response-Signature"

#: Route on the scoring worker that publishes its effective scorer identity.
SCORER_IDENTITY_ROUTE = "/healthz"

#: Field carrying the identity in that route's JSON body.
SCORER_IDENTITY_FIELD = "scorer_version"

#: Public payout-runtime contract carried by the scoring worker's health body.
SCORER_RUNTIME_COMMITMENT_FIELD = "runtime_commitment"
SCORER_RUNTIME_COMMITMENT_SCHEMA = "vidaio-payout-runtime/1"
SCORER_RUNTIME_BACKEND_KEY = "runtime"
SCORER_RUNTIME_BACKEND_PREFIX = f"{SCORER_RUNTIME_COMMITMENT_SCHEMA}+"

# Independent wire-side definition of the release runtime.  This module is used
# by validators, the challenge service and the competition orchestrator, so it
# deliberately does not import scoring_worker.runtime_identity (which would turn
# the producer's implementation into its consumers' verifier and creates import
# cycles through scoring/audit modules).  Keep these bytes and required fields in
# lockstep with the versioned public contract documented by the scoring worker.
SCORER_CANONICAL_RUNTIME_MARKER_BYTES = (
    b"vidaio-release-runtime/1\nos=linux\narch=amd64\n"
    b"aten_cpu_capability=default\ntorch_intraop_threads=1\n"
    b"torch_interop_threads=1\ntorch_deterministic_algorithms=error\n"
    b"torch_mkldnn=disabled\ntorch_nnpack=disabled\nmkl_cbwr=COMPATIBLE\n"
)
SCORER_CANONICAL_RUNTIME_MARKER_SHA256 = hashlib.sha256(
    SCORER_CANONICAL_RUNTIME_MARKER_BYTES
).hexdigest()
SCORER_PIEAPP_WEIGHTS_SHA256 = (
    "0937b01480c7a637ae3018af755faa8ecde4788b52bb246b7ae62cf96fb6baf0"
)
SCORER_REQUIRED_PAYOUT_BACKENDS = frozenset(
    {
        "ffmpeg",
        "ffprobe",
        "libvmaf",
        "pieapp",
        "perceptual",
        "pieapp_weights",
        "torch",
        "torchvision",
        "piq",
        "opencv",
        "numpy",
        "python",
    }
)
_SCORER_RELEASE_FIELDS = frozenset(
    {
        "manifest_verified",
        "manifest_sha256",
        "release_version",
        "source_sha256",
        "runtime_sha256",
        "marker_sha256",
        "marker_verified",
    }
)
_SCORER_EXECUTION_POLICY_FIELDS = frozenset(
    {
        "required_os",
        "required_arch",
        "actual_os",
        "actual_arch",
        "libc",
        "torch_intraop_threads",
        "torch_interop_threads",
        "mkl_threads",
        "openblas_threads",
        "omp_dynamic",
        "mkl_dynamic",
        "mkl_cbwr",
        "aten_cpu_capability_override",
        "actual_torch_intraop_threads",
        "actual_torch_interop_threads",
        "actual_torch_deterministic_algorithms",
        "actual_torch_deterministic_warn_only",
        "actual_torch_mkldnn_enabled",
        "actual_torch_mkldnn_deterministic",
        "actual_torch_nnpack_enabled",
        "actual_torch_cpu_capability",
        "actual_openmp_threads",
        "actual_mkl_threads",
        "actual_mkl_cbwr",
        "actual_mkl_dynamic",
    }
)
_SCORER_FIXED_EXECUTION_POLICY: dict[str, Any] = {
    "required_os": "linux",
    "required_arch": "amd64",
    "actual_os": "linux",
    "actual_arch": "amd64",
    "torch_intraop_threads": "1",
    "torch_interop_threads": "1",
    "mkl_threads": "1",
    "openblas_threads": "1",
    "omp_dynamic": "FALSE",
    "mkl_dynamic": "FALSE",
    "mkl_cbwr": "COMPATIBLE",
    "aten_cpu_capability_override": "default",
    "actual_torch_intraop_threads": 1,
    "actual_torch_interop_threads": 1,
    "actual_torch_deterministic_algorithms": True,
    "actual_torch_deterministic_warn_only": False,
    "actual_torch_mkldnn_enabled": False,
    "actual_torch_mkldnn_deterministic": True,
    "actual_torch_nnpack_enabled": False,
    "actual_torch_cpu_capability": "NO AVX",
    "actual_openmp_threads": 1,
    "actual_mkl_threads": 1,
    "actual_mkl_cbwr": "COMPATIBLE",
    "actual_mkl_dynamic": False,
}


class ScorerIdentityUnavailable(RuntimeError):
    """The scoring worker's identity could not be read from GET /healthz.

    Transport failure, a non-JSON body, or a payload with no (or an empty)
    ``scorer_version``. Callers must treat this as "unknown", never as
    "whatever I had configured" — the whole point of discovery is that the
    worker, not the caller, names the scorer.
    """


class ScorerRuntimeUnavailable(ScorerIdentityUnavailable):
    """The scorer health body cannot establish a self-consistent runtime pin.

    Competition scores are economic inputs, so a caller must not accept a packet
    when ``GET /healthz`` omits the runtime commitment, when its public preimage
    does not hash to its advertised digest, or when its complete payout-backend
    map is malformed.  This is a stricter subtype of identity unavailability:
    the worker may have named itself, but has not supplied enough trustworthy
    information to bind a measured packet to that name.
    """


class ScorerRuntimeMismatch(RuntimeError):
    """A live scorer differs from this process's local release contract."""


class ScorerRuntimeContract(BaseModel):
    """Validated packet-runtime expectations derived from ``GET /healthz``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer_version: str
    runtime_digest: str = Field(pattern=SHA256_HEX)
    #: Complete public preimage. Production consumers compare this object, not
    #: merely its self-reported digest, to an independently constructed local
    #: release-image expectation.
    attestation: dict[str, Any]
    #: Exact measured-packet map: health ``payout_backends`` plus its derived
    #: ``runtime`` commitment stamp.  Equality is intentional; subset matching
    #: could silently accept a moved or CUDA backend under a reused runtime stamp.
    backend_versions: dict[str, str]


def _lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def canonical_scorer_runtime_problems(attestation: Mapping[str, Any]) -> list[str]:
    """Independent wire-policy check for an earning scorer attestation.

    This proves that the *claim* is a complete canonical Linux/amd64 CPU claim.
    Production clients additionally compare the whole claim and its digest with a
    local attestation independently derived from their own pinned release image;
    therefore copying a scorer name or inventing a self-consistent claim is not
    sufficient to enter an earning path.
    """

    problems: list[str] = []
    expected_top = {"schema", "release", "execution_policy", "payout_backends"}
    if set(attestation) != expected_top:
        problems.append(
            "runtime attestation fields differ from the canonical schema "
            f"(missing={sorted(expected_top - set(attestation))}, "
            f"unexpected={sorted(set(attestation) - expected_top)})"
        )
    if attestation.get("schema") != SCORER_RUNTIME_COMMITMENT_SCHEMA:
        problems.append(
            f"schema is {attestation.get('schema')!r}, expected "
            f"{SCORER_RUNTIME_COMMITMENT_SCHEMA!r}"
        )

    release = attestation.get("release")
    if not isinstance(release, Mapping):
        problems.append("release identity is missing")
    else:
        if set(release) != _SCORER_RELEASE_FIELDS:
            problems.append("release identity fields differ from the canonical schema")
        if release.get("manifest_verified") is not True:
            problems.append("runtime release manifest is not verified")
        if release.get("marker_verified") is not True:
            problems.append("release-image runtime marker is not verified")
        if release.get("marker_sha256") != SCORER_CANONICAL_RUNTIME_MARKER_SHA256:
            problems.append("release-image runtime marker digest is not canonical")
        for field in ("manifest_sha256", "source_sha256", "runtime_sha256"):
            if not _lower_sha256(release.get(field)):
                problems.append(f"release {field} is not lowercase sha256")
        version = release.get("release_version")
        if not isinstance(version, str) or not version.strip():
            problems.append("release version is empty or malformed")

    policy = attestation.get("execution_policy")
    if not isinstance(policy, Mapping):
        problems.append("execution policy is missing")
    else:
        if set(policy) != _SCORER_EXECUTION_POLICY_FIELDS:
            problems.append("execution policy fields differ from the canonical schema")
        for field, expected in _SCORER_FIXED_EXECUTION_POLICY.items():
            if policy.get(field) != expected:
                problems.append(
                    f"execution policy {field} is {policy.get(field)!r}, "
                    f"expected {expected!r}"
                )
        libc = policy.get("libc")
        if (
            not isinstance(libc, str)
            or not libc.strip()
            or "unknown" in libc.lower()
            or "unavailable" in libc.lower()
        ):
            problems.append("execution policy libc identity is unavailable")

    backends = attestation.get("payout_backends")
    if not isinstance(backends, Mapping):
        problems.append("payout backend versions are missing")
    else:
        backend_names = set(backends)
        if backend_names != SCORER_REQUIRED_PAYOUT_BACKENDS:
            problems.append(
                "payout backend set is not canonical "
                f"(missing={sorted(SCORER_REQUIRED_PAYOUT_BACKENDS - backend_names)}, "
                f"unexpected={sorted(backend_names - SCORER_REQUIRED_PAYOUT_BACKENDS)})"
            )
        malformed = sorted(
            repr(name)
            for name, version in backends.items()
            if not isinstance(name, str)
            or not name.strip()
            or not isinstance(version, str)
            or not version.strip()
            or "not-configured" in version.lower()
            or version.lower().endswith("/unknown")
        )
        if malformed:
            problems.append(
                "payout backend versions are malformed or unavailable: "
                + ", ".join(malformed)
            )
        pieapp = str(backends.get("pieapp", ""))
        if not pieapp.endswith(":cpu"):
            problems.append("PieAPP payout backend is not CPU-only")
        torch = str(backends.get("torch", "")).lower()
        if "+cpu" not in torch or "+cu" in torch or "cuda" in torch:
            problems.append("PyTorch payout backend is not the CPU-only wheel")
        torchvision = str(backends.get("torchvision", "")).lower()
        if "+cpu" not in torchvision or "+cu" in torchvision or "cuda" in torchvision:
            problems.append("torchvision payout backend is not the CPU-only wheel")
        if backends.get("pieapp_weights") != (
            f"sha256:{SCORER_PIEAPP_WEIGHTS_SHA256}"
        ):
            problems.append("PieAPP weights digest is not the pinned release asset")
        if not str(backends.get("perceptual", "")).startswith(
            "cpu-perceptual-checks/"
        ):
            problems.append("perceptual payout backend is not the CPU implementation")
    return problems


def require_canonical_scorer_runtime(attestation: Mapping[str, Any]) -> None:
    problems = canonical_scorer_runtime_problems(attestation)
    if problems:
        raise ScorerRuntimeMismatch(
            "scoring worker does not attest the canonical marker-qualified "
            "Linux/amd64 CPU payout runtime: " + "; ".join(problems)
        )


def require_matching_scorer_runtime_contract(
    observed: ScorerRuntimeContract,
    expected: ScorerRuntimeContract,
    *,
    context: str = "",
) -> None:
    """Exact-match a live health contract to a locally derived release contract.

    Validating a remote attestation proves only that it is a well-formed claim.
    Earning callers must also derive ``expected`` from their own pinned release
    image and compare every identity, attestation and backend-map byte represented
    by the models.  This prevents a worker from reusing an identity while moving a
    backend or inventing a different, self-consistent runtime claim.
    """

    if observed == expected:
        return
    fields = [
        name
        for name in (
            "scorer_version",
            "runtime_digest",
            "attestation",
            "backend_versions",
        )
        if getattr(observed, name) != getattr(expected, name)
    ]
    where = f" ({context})" if context else ""
    raise ScorerRuntimeMismatch(
        "scorer payout-runtime contract mismatch"
        f"{where}: differing field(s): {', '.join(fields)}. The live worker must "
        "run the exact same marker-qualified release image and payout backend "
        "map as this consumer."
    )


class ScorerIdentityMismatch(RuntimeError):
    """A discovered identity disagrees with the one this caller is committed to.

    Raised where the disagreement is a configuration/commitment error rather
    than a transient one: an operator pin that no live worker matches, a
    competition manifest naming another scorer, a challenge commitment naming a
    scorer nobody runs.
    """

    def __init__(self, expected: str, discovered: str, *, context: str = "") -> None:
        where = f" ({context})" if context else ""
        super().__init__(
            f"scorer identity mismatch{where}: expected {expected!r}, the worker "
            f"advertises {discovered!r}. The identity is the worker's to mint "
            "(<name>+<config/runtime digest>, published on GET /healthz) — align the "
            "configuration with the scorer that will actually measure."
        )
        self.expected = expected
        self.discovered = discovered


def scorer_identity_from_healthz(payload: Mapping[str, Any]) -> str:
    """The effective identity out of a parsed ``GET /healthz`` body."""
    identity = str(payload.get(SCORER_IDENTITY_FIELD) or "").strip()
    if not identity:
        raise ScorerIdentityUnavailable(
            f"the scoring worker's {SCORER_IDENTITY_ROUTE} body carries no "
            f"{SCORER_IDENTITY_FIELD}: {dict(payload)!r}"
        )
    return identity


def scorer_runtime_contract_from_healthz(
    payload: Mapping[str, Any],
    *,
    require_canonical: bool = True,
) -> ScorerRuntimeContract:
    """Validate and derive the complete packet runtime contract from healthz.

    The worker publishes the canonical attestation preimage and its sha256.  We
    independently re-hash that preimage, require the versioned schema, and derive
    the exact backend map a measured packet must carry.  This intentionally does
    not import the scoring-worker implementation: the wire verifier is an
    independent consumer of the public protocol.
    """

    identity = scorer_identity_from_healthz(payload)
    commitment = payload.get(SCORER_RUNTIME_COMMITMENT_FIELD)
    if not isinstance(commitment, Mapping):
        raise ScorerRuntimeUnavailable(
            f"the scoring worker's {SCORER_IDENTITY_ROUTE} body carries no "
            f"{SCORER_RUNTIME_COMMITMENT_FIELD} object"
        )
    digest = commitment.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise ScorerRuntimeUnavailable(
            "the scoring worker's runtime commitment digest is not lowercase sha256"
        )
    attestation = commitment.get("attestation")
    if not isinstance(attestation, Mapping):
        raise ScorerRuntimeUnavailable(
            "the scoring worker's runtime commitment has no attestation object"
        )
    try:
        canonical = json.dumps(
            dict(attestation),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScorerRuntimeUnavailable(
            f"the scoring worker's runtime attestation is not canonical-JSON data: {exc}"
        ) from exc
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if actual_digest != digest:
        raise ScorerRuntimeUnavailable(
            "the scoring worker's runtime attestation does not hash to its "
            f"advertised digest (computed {actual_digest}, advertised {digest})"
        )
    if require_canonical:
        require_canonical_scorer_runtime(attestation)
    elif attestation.get("schema") != SCORER_RUNTIME_COMMITMENT_SCHEMA:
        raise ScorerRuntimeUnavailable(
            "the scoring worker's runtime attestation schema is not "
            f"{SCORER_RUNTIME_COMMITMENT_SCHEMA!r}"
        )
    payout_backends = attestation.get("payout_backends")
    if not isinstance(payout_backends, Mapping) or not payout_backends:
        raise ScorerRuntimeUnavailable(
            "the scoring worker's runtime attestation has no payout_backends map"
        )
    malformed = sorted(
        repr(name)
        for name, version in payout_backends.items()
        if not isinstance(name, str)
        or not name.strip()
        or not isinstance(version, str)
        or not version.strip()
    )
    if malformed:
        raise ScorerRuntimeUnavailable(
            "the scoring worker's runtime attestation has malformed payout backend(s): "
            + ", ".join(malformed)
        )
    if SCORER_RUNTIME_BACKEND_KEY in payout_backends:
        raise ScorerRuntimeUnavailable(
            "the runtime commitment recursively includes the derived runtime backend stamp"
        )
    backend_versions = {
        str(name): str(version) for name, version in sorted(payout_backends.items())
    }
    backend_versions[SCORER_RUNTIME_BACKEND_KEY] = (
        SCORER_RUNTIME_BACKEND_PREFIX + digest
    )
    return ScorerRuntimeContract(
        scorer_version=identity,
        runtime_digest=digest,
        attestation=json.loads(canonical),
        backend_versions=backend_versions,
    )


def _health_payload_from_response(response: httpx.Response) -> Mapping[str, Any]:
    # A DEGRADED worker (503) still knows who it is, and its identity is exactly
    # what a caller needs to decide whether to wait for it or fail loudly.
    try:
        payload = response.json()
    except ValueError as exc:
        raise ScorerIdentityUnavailable(
            f"{SCORER_IDENTITY_ROUTE} did not return JSON (HTTP "
            f"{response.status_code}): {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ScorerIdentityUnavailable(
            f"{SCORER_IDENTITY_ROUTE} returned {type(payload).__name__}, not an object"
        )
    return payload


def _identity_from_response(response: httpx.Response) -> str:
    return scorer_identity_from_healthz(_health_payload_from_response(response))


def fetch_scorer_identity(
    base_url: str, *, timeout: float = 10.0, client: httpx.Client | None = None
) -> str:
    """Read the worker's effective scorer identity (sync). See the module doc.

    `client` lets a caller supply its own transport (an ASGI transport in tests,
    a pooled client in a service); `base_url` is then only used for the path.
    """
    url = base_url.rstrip("/") + SCORER_IDENTITY_ROUTE
    try:
        if client is not None:
            response = client.get(url, timeout=timeout)
        else:
            response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ScorerIdentityUnavailable(
            f"could not reach the scoring worker at {url}: {type(exc).__name__}: {exc}"
        ) from exc
    return _identity_from_response(response)


def fetch_scorer_runtime_contract(
    base_url: str,
    *,
    timeout: float = 10.0,
    client: httpx.Client | None = None,
    require_canonical: bool = True,
) -> ScorerRuntimeContract:
    """Read and validate scorer identity plus its complete payout runtime pin."""

    url = base_url.rstrip("/") + SCORER_IDENTITY_ROUTE
    try:
        if client is not None:
            response = client.get(url, timeout=timeout)
        else:
            response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ScorerRuntimeUnavailable(
            f"could not reach the scoring worker at {url}: {type(exc).__name__}: {exc}"
        ) from exc
    return scorer_runtime_contract_from_healthz(
        _health_payload_from_response(response),
        require_canonical=require_canonical,
    )


async def fetch_scorer_identity_async(
    base_url: str, *, timeout: float = 10.0, client: httpx.AsyncClient | None = None
) -> str:
    """Async twin of :func:`fetch_scorer_identity`."""
    url = base_url.rstrip("/") + SCORER_IDENTITY_ROUTE
    try:
        if client is not None:
            response = await client.get(url, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as own:
                response = await own.get(url)
    except httpx.HTTPError as exc:
        raise ScorerIdentityUnavailable(
            f"could not reach the scoring worker at {url}: {type(exc).__name__}: {exc}"
        ) from exc
    return _identity_from_response(response)


async def fetch_scorer_runtime_contract_async(
    base_url: str,
    *,
    timeout: float = 10.0,
    client: httpx.AsyncClient | None = None,
    require_canonical: bool = True,
) -> ScorerRuntimeContract:
    """Async twin of :func:`fetch_scorer_runtime_contract`."""

    url = base_url.rstrip("/") + SCORER_IDENTITY_ROUTE
    try:
        if client is not None:
            response = await client.get(url, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as own:
                response = await own.get(url)
    except httpx.HTTPError as exc:
        raise ScorerRuntimeUnavailable(
            f"could not reach the scoring worker at {url}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return scorer_runtime_contract_from_healthz(
        _health_payload_from_response(response),
        require_canonical=require_canonical,
    )


class MinerTaskRequest(BaseModel):
    """Deprecated local/shared-filesystem miner request.

    Production/default HTTP clients derive :class:`MinerArtifactTaskRequest`
    from this internal request and stream ``input_path``'s bytes.  The path is
    intentionally absent from the remote metadata.
    """

    task_id: str
    track: str  # "compression" | "upscaling"
    input_path: str
    input_digest: str = Field(pattern=SHA256_HEX)
    #: track params the miner may see (never seeds/DAG): e.g. upscale factor, bitrate cap
    params: dict[str, float | int | str] = Field(default_factory=dict)
    #: Finalized external commitment receipt supplied by the challenge service.
    #: Artifact-v2 signs it into the request, so the miner's response signature
    #: proves this exact chain receipt was presented before the output existed.
    commitment_anchor: ChallengeAnchor | None = None
    #: seconds the miner has before the validator scores it absent
    deadline_seconds: float = 300.0


class MinerTaskResponse(BaseModel):
    task_id: str
    output_path: str
    output_digest: str = Field(pattern=SHA256_HEX)
    #: wall-clock processing seconds as measured by the miner (advisory only)
    processing_seconds: float | None = None
    #: Canonical artifact-v2 miner signature receipt.  It is a JSON object here to
    #: keep the base wire model independent of the auth implementation; production
    #: constructs and verifies it as ``MinerArtifactReceipt``.
    artifact_receipt: dict[str, Any] | None = None


class MinerArtifactTaskRequest(BaseModel):
    """Path-free metadata bound to one remote miner artifact upload."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    track: str
    input_digest: str = Field(pattern=SHA256_HEX)
    params: dict[str, float | int | str] = Field(default_factory=dict)
    commitment_anchor: ChallengeAnchor | None = None
    deadline_seconds: float = Field(default=300.0, gt=0)

    @classmethod
    def from_local_request(cls, request: MinerTaskRequest) -> "MinerArtifactTaskRequest":
        return cls(
            task_id=request.task_id,
            track=request.track,
            input_digest=request.input_digest,
            params=request.params,
            commitment_anchor=request.commitment_anchor,
            deadline_seconds=request.deadline_seconds,
        )

    def as_local_request(self, input_path: str) -> MinerTaskRequest:
        """Bind received bytes to the legacy backend's local staging path."""
        return MinerTaskRequest(
            task_id=self.task_id,
            track=self.track,
            input_path=input_path,
            input_digest=self.input_digest,
            params=self.params,
            commitment_anchor=self.commitment_anchor,
            deadline_seconds=self.deadline_seconds,
        )


class MinerTaskMetadataError(ValueError):
    """The bounded remote-task metadata header is missing or malformed."""


def encode_miner_task_metadata(metadata: MinerArtifactTaskRequest) -> str:
    """Canonical, ASCII-safe header value for a remote miner task."""
    raw = json.dumps(
        metadata.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    if len(raw) > MAX_MINER_TASK_METADATA_BYTES:
        raise MinerTaskMetadataError(
            f"task metadata is {len(raw)} bytes; maximum is "
            f"{MAX_MINER_TASK_METADATA_BYTES}"
        )
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_miner_task_metadata(value: str | None) -> MinerArtifactTaskRequest:
    """Decode and validate a bounded metadata header without reading the body."""
    if not value:
        raise MinerTaskMetadataError(f"missing {MINER_TASK_METADATA_HEADER} header")
    # Encoded length preflight: ceil(raw / 3) * 4, minus at most two padding
    # bytes. Refuse before base64 or JSON allocates attacker-sized objects.
    max_encoded = ((MAX_MINER_TASK_METADATA_BYTES + 2) // 3) * 4
    if len(value) > max_encoded:
        raise MinerTaskMetadataError(
            f"encoded task metadata exceeds {MAX_MINER_TASK_METADATA_BYTES} bytes"
        )
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MinerTaskMetadataError("task metadata header is not ASCII") from exc
    padding = b"=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MinerTaskMetadataError("task metadata header is not valid base64url") from exc
    if len(raw) > MAX_MINER_TASK_METADATA_BYTES:
        raise MinerTaskMetadataError(
            f"task metadata exceeds {MAX_MINER_TASK_METADATA_BYTES} bytes"
        )
    try:
        payload = json.loads(raw)
        return MinerArtifactTaskRequest.model_validate(payload)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise MinerTaskMetadataError(f"task metadata is invalid: {exc}") from exc


class ScoreRequest(BaseModel):
    """Validator/orchestrator -> scoring worker: score one miner output.

    Path/digest contract: every ``*_path`` names a file on the shared filesystem
    and every ``*_digest`` is the sha256 the caller claims for it. The worker
    verifies each digest through a single symlink-refusing descriptor and scores
    an immutable private COPY taken from that same descriptor, so the packet's
    ``content_digest`` always names the bytes that were actually measured — a
    post-request rewrite of the caller-named path cannot change the score.
    Symlinks, fifos, devices and directories are refused (422), never followed.

    ``scorer_version`` is an ASSERTION, not an instruction: it says which scorer
    the caller expects. The worker always stamps its OWN identity into the packet
    (``<configured name>+<identity digest[:12]>``, published on the worker's GET
    /healthz). Sending a different value is a 409 ``scorer_version_mismatch`` —
    a validator expecting scorer X must never silently accumulate packets from
    scorer Y, because the audit bundle later cross-checks the two. Omit the field
    (or send an empty string) to accept whichever scorer answers; discover the
    identity with :func:`fetch_scorer_identity` and see the module docstring's
    SCORER-IDENTITY CONTRACT for who pins what.
    """

    track: str
    challenge_id: str
    item_id: str
    miner_hotkey: str | None = None
    reference_path: str
    reference_digest: str = Field(pattern=SHA256_HEX)
    miner_input_path: str
    miner_input_digest: str = Field(pattern=SHA256_HEX)
    output_path: str
    output_digest: str = Field(pattern=SHA256_HEX)
    #: scoring params (e.g. vmaf_threshold, upscale_factor) — validator-chosen
    params: dict[str, float | int | str] = Field(default_factory=dict)
    #: the scorer the caller EXPECTS; None/"" = accept the worker's own (see above)
    scorer_version: str | None = None


class ScoreResponse(BaseModel):
    """The audit-grade result: the full ItemScore packet as its exact JSON bytes
    (base for the score_packet_digest), plus that digest for convenience."""

    item_score_json: str
    packet_digest: str = Field(pattern=SHA256_HEX)
