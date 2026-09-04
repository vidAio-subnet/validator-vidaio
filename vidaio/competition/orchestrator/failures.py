"""Failure CLASSIFICATION — one untrusted contender must never halt everything.

review service-review #14: the orchestrator treated every exception escaping a
batch (including a solution's deliberate ``exit 1``) as a systemic infrastructure
blocker, so any contender could requeue-then-HALT the whole competition; and a
code-0 run that produced no output was sent to real ffmpeg, whose inevitable 502
halted it too.

This module is the single place that answers "whose fault was that?".

CONTENDER fault (``Fault.CONTENDER``) — attributable to the submission:
    the solution exited non-zero, blew its per-contender timeout, wrote an unsafe
    or oversize output, produced no (or an empty) output, shipped a Dockerfile that
    does not build, failed an isolation probe that actually RAN, or made the
    trusted scorer reject ITS OWN OUTPUT bytes.
    Outcome: THAT contender's item/batch is failed and zero-scored with a reason
    code; the competition CONTINUES.

INFRA fault (``Fault.INFRA``) — attributable to us:
    docker daemon unreachable, image missing, a sealed input missing from the
    pool, the isolation contract not holding on a container we launched, a probe
    that could not be RUN, a checkout we could not materialize, DB errors,
    scoring-worker 5xx/transport failures, and every scorer rejection that names
    OUR half of the request (reference/miner_input paths, invalid params).
    Outcome: the existing bounded requeue/retry path, then HALT (never a
    competition FAILED, never a substituted score).

DEFAULT: anything unrecognized is INFRA. Zeroing a contender because of OUR bug
would silently corrupt the competition's result; halting is loud, recoverable and
leaves the DB truthful. Fail closed, not convenient.

ROUND 2. Four stages BYPASSED this module
entirely and hard-coded a verdict; they now all route through ``classify_failure``:
  - a failed submission-backup / validation checkout no longer auto-rejects the
    contender (an unreachable git host is not a bad submission),
  - an arbitrary build failure no longer auto-marks BUILD_FAILED (see
    runners.errors.ContenderBuildError vs BuildError),
  - a probe that could not RUN no longer disqualifies anyone (see
    runners.errors.SandboxProbeUnavailableError),
  - a scorer 422 is no longer contender fault by status alone (see below).
"""

from __future__ import annotations

import sqlite3
from enum import StrEnum

from vidaio.core.resilience import RetriesExhausted
from vidaio.competition.repository import ScorePacketError
from vidaio.competition.runners.errors import ContenderFaultError

#: Statuses that mean "the contender's MEDIA is unusable" on their own:
#:   400 the worker could not parse/decode the submitted media,
#:   415 unsupported media type.
#: An allowlist, not a 4xx catch-all, on purpose: 401/403 are OUR credentials,
#: 404 is a wrong URL, 409 is a scorer-identity/version disagreement between
#: validator and worker, 408/425/429 are timing. None of those say anything about
#: the submission, so zeroing a contender for them would silently corrupt the
#: competition — they are INFRA and must halt loudly.
_CONTENDER_STATUSES = frozenset({400, 415})

#: 422 is DELIBERATELY NOT in that set. The
#: scoring worker answers 422 for three different things:
#:   - the OUTPUT the contender produced is missing/unreadable/not a regular
#:     file/hash-mismatched  -> the contender's fault,
#:   - the REFERENCE or MINER INPUT *we* named is missing/unreadable/mismatched
#:     -> our fault (a broken input pool, a stale path, a lost sealed input),
#:   - invalid_param / unsupported_track -> our request, our bug.
#: Treating every 422 as contender fault zeroed contenders for OUR failures, which
#: is exactly the silent corruption the DEFAULT rule below exists to prevent. So a
#: 422 is contender-attributable only when the worker's own typed error names one
#: of these input problems AND names the OUTPUT field.
_REJECTED_INPUT_CODES = frozenset(
    {
        "file_missing",
        "symlink_rejected",
        "unreadable_input",
        "not_a_regular_file",
        "digest_mismatch",
        "undecodable_media",
        "unsupported_media_type",
    }
)

#: The one request artifact the CONTENDER produced. `reference` and `miner_input`
#: are ours (the sealed input pool); a rejection naming them is an infra failure.
_CONTENDER_FIELD = "output"


class Fault(StrEnum):
    CONTENDER = "contender"
    INFRA = "infra"


def unwrap(exc: BaseException) -> BaseException:
    """Peel retry wrappers so classification sees the real cause.

    ``retry_async`` raises RetriesExhausted with the last failure as ``__cause__``;
    a contender fault that exhausted a retry budget is still a contender fault.
    """
    seen: set[int] = set()
    current = exc
    while isinstance(current, RetriesExhausted) and current.__cause__ is not None:
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.__cause__
    return current


def scorer_rejection_is_contender_fault(
    status: int | None, error_code: str | None, error_field: str | None
) -> bool:
    """Is this scoring-worker rejection attributable to the CONTENDER's bytes?

    Split out so it is testable on its own and so the rule is stated once:
    - 400/415: the worker could not decode the media it was handed -> contender;
    - 422: only when the worker's TYPED error names an input problem on the
      ``output`` field. A 422 about ``reference``/``miner_input``, an
      ``invalid_param``/``unsupported_track``, or an untypeable 422 is OURS;
    - anything else: not contender-attributable.
    """
    if status in _CONTENDER_STATUSES:
        return True
    if status != 422:
        return False
    return error_code in _REJECTED_INPUT_CODES and error_field == _CONTENDER_FIELD


def classify_failure(exc: BaseException) -> Fault:
    """Map an exception to its fault class. See the module docstring for the rules."""
    root = unwrap(exc)
    if isinstance(root, ContenderFaultError):
        return Fault.CONTENDER
    status = getattr(root, "status_code", None)
    if isinstance(status, int) and scorer_rejection_is_contender_fault(
        status,
        getattr(root, "error_code", None),
        getattr(root, "error_field", None),
    ):
        # The trusted scorer rejected the CONTENDER's own bytes as unusable.
        return Fault.CONTENDER
    if isinstance(root, ScorePacketError):
        # A packet that cannot be bound to this contender/item is a scorer or
        # transport defect, not something the contender chose — infra.
        return Fault.INFRA
    if isinstance(root, sqlite3.Error):
        return Fault.INFRA
    return Fault.INFRA


def fault_code(exc: BaseException) -> str:
    """Stable machine-readable reason code for the event log / batch row."""
    root = unwrap(exc)
    code = getattr(root, "code", None)
    if isinstance(code, str) and code:
        return code
    status = getattr(root, "status_code", None)
    error_code = getattr(root, "error_code", None)
    if isinstance(error_code, str) and error_code:
        # The worker's own typed reason is more useful in the event log than the
        # bare status it happened to arrive with.
        return f"SCORER_{error_code.upper()}"
    if isinstance(status, int):
        return f"SCORER_HTTP_{status}"
    return type(root).__name__.upper()
