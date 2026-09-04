"""ORCHESTRATOR-MINTED ZERO PACKETS — an honest, distinct scorer identity.

review service-review, round 2, finding new-3
-------------------------------------------------------------------------------
The orchestrator has one legitimate reason to mint a score packet itself: an item
for which the contender produced NO OUTPUT (absent or zero-byte), or whose output
the trusted worker rejected as its own. Those items must not be handed to real
ffmpeg, so they
are gate-failed to zero locally with a machine-readable reason code.

What was WRONG about the previous version: the packet was stamped with
``manifest.scoring_version`` — the scoring WORKER's identity. A SCORE_PACKET
artifact therefore claimed the worker had produced bytes the worker never saw.
Anyone auditing the store could not tell a measured packet from an orchestrator
bookkeeping record, and the whole point of the identity contract
(vidaio/services/protocol.py) is that a packet names whoever actually minted it.

THE CONVENTION (this module is its definition)
-------------------------------------------------------------------------------
An orchestrator-minted zero carries::

    scorer_version = "orchestrator-zero/1+<identity digest[:12]>"

which is the same ``<name>+<digest12>`` SHAPE as a worker identity — so every
existing consumer parses it — under a name the orchestrator RESERVES. The digest
covers everything that determines what such a packet asserts:

    {"convention": "orchestrator-zero/1",
     "scoring_config_digest": <sha256 of the ScoringConfig in force>,
     "committed_scoring_version": <the manifest's committed worker identity>,
     "track": <the manifest's track>}

so the identity is recomputable from the anchored manifest plus the config, it
moves when either moves, and it RECORDS which worker the competition committed to
without ever claiming to be that worker.

WHY THIS IS HONEST, not a second scorer: the packet asserts no measurement.
``gate_passed=False`` forces ``score=0.0`` structurally (compose_item_score and
repository.record_item_score both enforce gates-first), the violation carries the
machine-readable reason (METRIC_MISSING), ``content_digest`` is the canonical
empty digest, and ``metrics`` is empty. It is a gate-failure
RECORD attributed to the orchestrator — an orchestrator fact, not a measurement.

CONSUMERS: a packet whose ``scorer_version`` is an orchestrator-zero identity is
LEGITIMATELY different from the manifest's ``scoring_version``. That is not drift
and must not be flagged as one. The audit bundle for such a row is built with the
SAME orchestrator-zero identity (vidaio/audit/recompute.py cross-checks packet
against bundle, and those two must agree), while the manifest artifact in the same
bundle still names the committed worker — so the bundle says, precisely: "this
competition committed to worker X; for this item there were no bytes to measure,
and the orchestrator recorded a zero". The CPU recomputer re-hashes the empty
output, derives this identity from its own locked scoring config plus that
manifest, and refuses any metric/backend/scoring-work claim.

IMPERSONATION IS IMPOSSIBLE IN BOTH DIRECTIONS:
- the orchestrator never stamps a worker identity on a packet it minted itself
  (this module is the only place a local packet's scorer_version comes from);
- a worker may not claim the reserved name either: ``assert_not_reserved`` is
  applied to the manifest's committed identity and to the live worker's advertised
  identity, and a claim on ``orchestrator-zero/*`` is refused (an INFRA halt), so
  an untrusted worker cannot make its measured packets indistinguishable from
  orchestrator gate-failure records.
"""

from __future__ import annotations

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.scoring.config import ScoringConfig
from vidaio.scoring.result import ItemScore, compose_item_score, config_digest
from vidaio.scoring.gates import ReasonCode, ValidityViolation

#: The reserved scorer NAME the orchestrator mints gate-failure zeros under.
#: Versioned like a worker name: bump it if the meaning of these packets changes.
ORCHESTRATOR_ZERO_SCORER_NAME = "orchestrator-zero/1"

#: Any identity starting with this is orchestrator-minted, by definition.
ORCHESTRATOR_ZERO_PREFIX = f"{ORCHESTRATOR_ZERO_SCORER_NAME}+"

#: The whole reserved namespace (any version), used to refuse impersonation.
_RESERVED_NAMESPACE = "orchestrator-zero/"


class ReservedScorerIdentity(ValueError):
    """Someone other than the orchestrator claimed the orchestrator-zero namespace.

    Raised for a manifest that commits to such an identity and for a live worker
    that advertises one. Allowing it would make measured packets indistinguishable
    from orchestrator gate-failure records — the exact confusion this convention
    exists to remove.
    """


def is_orchestrator_zero_identity(identity: str | None) -> bool:
    """True for an identity minted by :func:`orchestrator_zero_identity`.

    Consumers comparing a packet's ``scorer_version`` with the manifest's
    ``scoring_version`` use this to recognise the legitimate difference.
    """
    return bool(identity) and str(identity).startswith(ORCHESTRATOR_ZERO_PREFIX)


def assert_not_reserved(identity: str | None, *, what: str) -> None:
    """Refuse a NON-orchestrator identity inside the reserved namespace."""
    if identity and str(identity).startswith(_RESERVED_NAMESPACE):
        raise ReservedScorerIdentity(
            f"{what} claims the reserved scorer namespace "
            f"{_RESERVED_NAMESPACE!r} ({identity!r}). That namespace belongs to "
            "orchestrator-minted gate-failure records (packets that assert no "
            "measurement); a scoring worker answering under it would make measured "
            "packets indistinguishable from them. Rename the scorer."
        )


def orchestrator_zero_identity(
    *,
    committed_scoring_version: str,
    track: str,
    config: ScoringConfig | None = None,
) -> str:
    """The effective identity for orchestrator-minted zeros in this competition.

    Deterministic and recomputable from the anchored manifest (its
    ``scoring_version`` and ``track``) plus the ScoringConfig in force — see the
    module docstring for why each input is in the digest.
    """
    cfg = config if config is not None else ScoringConfig()
    digest = sha256_hex(
        canonical_json_bytes(
            {
                "convention": ORCHESTRATOR_ZERO_SCORER_NAME,
                "scoring_config_digest": config_digest(cfg),
                "committed_scoring_version": committed_scoring_version,
                "track": track,
            }
        )
    )
    return f"{ORCHESTRATOR_ZERO_SCORER_NAME}+{digest[:12]}"


def mint_zero_packet(
    *,
    scoring_item_id: str,
    challenge_id: str,
    track: str,
    committed_scoring_version: str,
    miner_hotkey: str | None,
    empty_digest: str,
    code: ReasonCode,
    detail: str,
    config: ScoringConfig | None = None,
) -> tuple[ItemScore, str]:
    """Mint the gate-failed ZERO for one (contender, item) and its identity.

    Returns ``(packet, scorer_version)``; the caller persists the packet bytes and
    MUST use the same ``scorer_version`` for the item's audit bundle so packet and
    bundle agree (vidaio/audit/recompute.py cross-checks exactly that).

    The packet is self-evidently an orchestrator attribution: reserved identity,
    ``gate_passed=False`` (so the score is structurally 0.0), the reason code in
    the violation, and the canonical empty digest as content — there is no path
    here that can emit a positive score or a measured metric.
    """
    cfg = config if config is not None else ScoringConfig()
    scorer_version = orchestrator_zero_identity(
        committed_scoring_version=committed_scoring_version, track=track, config=cfg
    )
    packet = compose_item_score(
        item_id=scoring_item_id,
        challenge_id=challenge_id,
        track=track,
        gate_passed=False,
        violations=[ValidityViolation(code=code, detail=detail)],
        breakdown=None,
        config=cfg,
        miner_hotkey=miner_hotkey,
        content_digest=empty_digest,
        # This record asserts only the independently hashable absence of output.
        # Do not copy a DB-declared fact into the metric surface: a CPU auditor
        # must reproduce every numeric metric without trusting that database.
        metrics={},
        scorer_version=scorer_version,
    )
    return packet, scorer_version
