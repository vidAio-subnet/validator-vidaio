"""The per-epoch EPOCH-LOG model — the ONE artifact that drives BOTH sides.

This is the shared, dependency-light data model (pydantic + stdlib only) imported
by BOTH the central Scoring Authority (which PRODUCES an `EpochLog` each epoch via
`vidaio.authority.EpochFinalizer`) AND the thin validator/auditor (which CONSUME
it: validators converge from `weight_shares`/`weight_u16`, auditors verify from
`audit_manifest`). Spec: the project design record §3.1 (`EpochResultsSnapshot`)
and the project design record §2.2, build-wave 3.

It deliberately reuses the two designated shared primitives and nothing heavier:

- `vidaio.tokenomics.quantize.quantize_u16` — THE deterministic float->u16 grid
  (wave 1), so the u16 vector carried here is byte-identical to what every
  validator recomputes;
- `vidaio.audit.canonical.canonical_json_bytes` / `sha256_hex` — THE canonical
  JSON + digest contract (sorted keys, no whitespace), so `log_digest()` is the
  same sha256 on every machine and can be anchored on chain.

The per-uid authoritative inference inputs (`MinerSnapshot`) and live competition
reward-window models come from `vidaio.tokenomics.state`. Schema v15 commits
the exact competition packet/bundle matrix and per-subject dedup decisions needed to
rederive those economics.

Convergence-critical property (tested): the SAME epoch state yields BYTE-IDENTICAL
`EpochLog` bytes and digest on any machine, regardless of dict/list input order —
`to_json()` normalizes every collection to a canonical order and delegates key
ordering to `canonical_json_bytes`. Two validators that fetch the same log agree
on the weight vector without any coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidaio.audit.canonical import (
    SHA256_HEX_PATTERN,
    canonical_json_bytes,
    sha256_hex,
)
from vidaio.audit.commitments import COMMITMENT_DOMAIN
from vidaio.challenge.dag import TRACK_RULES
from vidaio.tokenomics.quantize import quantize_u16
from vidaio.tokenomics.state import (
    CompetitionResult,
    ContenderResult,
    MinerSnapshot,
    RewardWindowState,
)

#: The PROTOCOL track set: the tracks the protocol actually scores +
#: pays out — the keys of the challenge module's ``TRACK_RULES`` (the same set
#: ``tokenomics.track_weights`` is keyed by). A committed/log track OUTSIDE this set is
#: not a real protocol track: tokenomics silently drops such a miner from every track
#: pool (``inference_shares``), so committing positive evidence under an out-of-set track
#: (e.g. ``"unknown"``) substitutes a 100%-burn while every self-consistent declaration
#: still agrees. The log-validation boundary refuses any out-of-set track here (the
#: auditor adds a defense-in-depth DISPUTED verdict for bytes that bypass the finalizer).
PROTOCOL_TRACKS: frozenset[str] = frozenset(TRACK_RULES)
GIT_SHA1_HEX_PATTERN = r"^[0-9a-f]{40}$"

#: Bump on ANY change to the EpochLog canonical-JSON shape (it changes every
#: recorded `log_digest`, so a mixed-version fleet must fence on it — this is the
#: `version_key` convergence fence, the project design record §8).
#: v2 (2026-08-21): `AuditFileRef` now carries a per-item merkle `inclusion_proof`
#: so an auditor can PROVE committed-set membership against the manifest's
#: `score_packet_merkle_root` (closes the wave-6 tamper-evidence gap).
#: v3 (2026-08-21): the EARNING-STATE re-fold spine (#1). The manifest now carries,
#: per uid, an `EarningInput` (the prior-epoch carry-in + the ordered cycle scores
#: that fold into that uid's `accumulate_score`), the `EpochLog` carries
#: `prior_log_digest` (chaining epochs back to genesis so the fold is verifiable),
#: and the SCORE_PACKET `AuditFileRef` carries the COMMITTED `committed_track` (so
#: recompute-ability and the recompute track come from the committed challenge, not
#: the authority's packet — #9). All additions are deterministic from sorted/ordered
#: inputs, so byte-identity survives.
#: v4 (historical): made the competition-cycle staleness input explicit. That scalar
#: was later removed; schema v14 instead binds economic time to ``created_at`` and each
#: result's ``applied_at`` at the exact epoch-close chain time.
#: v5 (2026-08-21): the earning fold is now EVIDENCE-BOUND. A
#: uid's `cycle_scores` are no longer bare floats an authority can reorder or pad: each
#: is a `CycleScore` BOUND to a committed SCORE_PACKET — `packet_digest` (a per_uid
#: merkle leaf), `ordering_key` (the packet's committed monotonic cycle-sequence, the
#: VERIFIABLE fold order), and `score` (the packet's recorded value, or -1 for a
#: committed exclusion). EWMA is order-dependent, so binding the order to committed
#: evidence closes the reorder hole (`[0.1,0.9]`→`[0.9,0.1]`) and the unbacked-sentinel
#: hole (a padded 0.0 = extra decay, a substituted -1 = phantom exclusion). And the
#: SCORE_PACKET `AuditFileRef.committed_track` is now REQUIRED (#9), so the auditor
#: recomputes over the committed track and NEVER the packet-controlled one. Deterministic
#: (sorted leaves / ascending ordering keys), so byte-identity survives.
#: v6: the manifest now carries, per nonzero-weight uid, a `WindowInput`
#: — the COMMITTED windowed evidence (retention-window start/end blocks, alpha-stake at each
#: endpoint, emission over the window, and the registration block) the auditor independently
#: re-derives the WINDOWED weight inputs from (alpha_stake_delta_window / emission_window /
#: has_full_retention_window), cross-checks the endpoint against the close-block metagraph,
#: and chains the window START against the prior epoch. It is SOURCED from the authority's
#: real windowed observations, NOT re-derived from the (self-attested) MinerSnapshot fields,
#: so a substituted retention claim no longer re-derives cleanly. Deterministic (sorted uids /
#: plain scalar fields), so byte-identity survives; bumping this is the `version_key` fence.
#: v7: the manifest now carries, when a competition COMPLETED this epoch, a
#: `CompetitionInput` — the COMMITTED competition evidence binding the `competition_result`: per
#: contender {hotkey, uid, score_packet_digest, audit_bundle_digest}, the executable-baseline
#: packets, and the committed cycle/completion time. The auditor re-derives each contender's
#: margin against the executable baseline, reconstructs the result, and re-derives the reward
#: designation — never trusting stated margins or rank. Contender bundles are also carried in
#: `competition_bundles` (parallel to
#: `baseline_bundles`) so they route through the existing media-sample/verify_bundle path. An epoch
#: with no competition (`competition_result is None`) carries none of this. Deterministic
#: (ranked contender order preserved, sorted refs), so byte-identity survives; bumping this is
#: the `version_key` convergence fence.
#: v8: the
#: retention-window multiplier is GONE for v1, so the manifest no longer carries the per-uid
#: `WindowInput` (committed-window evidence) and the `MinerSnapshot` no longer carries the
#: three windowed fields (alpha_stake_delta_window / emission_window / has_full_retention_window).
#: The canonical-JSON shape of every MinerSnapshot and of the AuditManifest changed, so a
#: mixed-version fleet MUST fence on it — this bump is the `version_key` convergence fence
#: (the project design record §8).
#: v9 (historical): competition input added executable-baseline audit-bundle digests alongside
#: the already-committed baseline score-packet digests. The baseline packets became committed +
#: bundle-bound + merkle-included EXACTLY like contenders (their bundles route into
#: `competition_bundles`, their packets are merkle leaves), so the auditor resolves B_t ONLY from
#: committed, bound baseline evidence — never an arbitrary stored score. Deterministic (two
#: optional scalars), so byte-identity survives; bumping this is the `version_key` convergence fence.
#: v10 (historical): temporarily removed the competition economic/audit path. The canonical
#: shape changed, so mixed-version fleets had to fence on the schema version.
#: v11 (earning replay integrity): introduced a cumulative per-uid highest committed dispatch
#: ordering key. The map is carried
#: through every epoch — including carry-only, excluded, deregistered, and empty/burn epochs —
#: and is part of the anchored canonical bytes. ``EpochLog.miner_census`` separately commits the
#: full registered subnet identity set at the close block, including registrations with
#: no known economic track; ``miners`` remains the eligible/economic subset. Auditors enforce
#: exact predecessor cursor continuity and exact metagraph/census identity. A one-epoch
#: omission, an idle epoch, or a hotkey change therefore cannot erase the replay boundary.
#: v12 (historical testnet competition + availability evidence): restored score-derived
#: competition economics without human-review inputs. The log carried completed competition
#: and reward data; the manifest carried every contender/baseline packet and bundle grouped
#: by an unambiguous audit subject. Auditors rebuilt subject aggregates from the committed
#: packet set, reran selected media on CPU, derived the podium, then derived weights.
#: ``burn_uid`` also became the canonical
#: sink for conditionally withheld empty/below-floor pools, so it may coexist with earners.
#: v13 (competition provenance + duplicate suppression): every competition subject now
#: commits ``dedup_excluded``. The authority derives it from the close-block census
#: (lowest-uid IP/coldkey winner, with unspecified-IP exemption) plus the subject's exact
#: full output-digest matrix; the CPU auditor independently re-derives the bit before
#: accepting the podium. Competition audit bundles also bind one stable execution-image
#: digest per subject, with the baseline required to match its pre-enrollment commitment.
#: v14 (total fold-cursor boundary): replaces the partial replay-boundary map with
#: ``fold_cursors: uid -> int|null``. Every current census uid has an entry: ``null`` means
#: the uid slot has been observed but has never folded a cycle, while an integer is the
#: greatest ordering key ever folded. The entire predecessor map is carried as tombstones,
#: so first fold after null is valid and every later fold must strictly advance. V14 also
#: commits the executable baseline and contender release provenance, chain-time result
#: application, the exact pre-enrollment raw anchor inclusion/finality receipt, and the
#: replay-safe ``reward_window_state`` used to derive the vector.
#: v15 (validator-permit role correction): keeps the v14 canonical field shape but
#: defines ``miner_census`` as every registered subnet identity. Validator permit is
#: a dynamic capability, not an exclusive role; a serving miner that earns enough
#: stake to acquire one must remain scoreable, payable, and replay-bound. Old v14
#: producers/auditors would disagree on the same close-block metagraph, so this
#: semantic consensus change receives a lockstep fleet fence despite no JSON field
#: being added.
EPOCH_LOG_SCHEMA_VERSION = 16


class EpochLogInvalid(Exception):
    """An EpochLog violates a convergence/audit invariant and is REJECTED.

    Raised when the u16 vector is not `quantize_u16` of the float vector, when the
    weight-vector digest does not bind the u16 vector, or when a nonzero-weight uid
    has NO audit-manifest entry (a weight an auditor could never reproduce).

    Deliberately NOT a `ValueError` subclass: pydantic wraps `ValueError`/`AssertionError`
    raised inside a validator into its own `ValidationError`, which would hide the
    domain error. As a plain `Exception` it propagates from `EpochLog(...)` construction
    unchanged, so callers (finalizer, auditor) catch a single, meaningful error type.
    """


class AuditFileKind(StrEnum):
    """What kind of audit-store object an `AuditFileRef` points at.

    SCORE_PACKET is the stored `ItemScore` packet (a real, content-addressed
    `ArtifactKind.SCORE_PACKET` blob an auditor fetches by digest and recomputes
    over the real scoring engine). AUDIT_BUNDLE is the `AuditBundle.bundle_digest()`
    that binds the item's whole artifact set — the auditor verifies the bundle by
    recompute (`vidaio.audit.recompute.verify_bundle`), it is not a standalone blob.
    """

    SCORE_PACKET = "score_packet"
    AUDIT_BUNDLE = "audit_bundle"


#: The ONLY media strata an audit ref may be labelled with. The
#: auditor's coverage-floor roll-up (`vidaio.auditor.report.overall_status`) recognises
#: exactly these; a ref labelled anything else would SKIP without counting toward the
#: floor, so it is refused at construction. Kept here (the model layer) so it is the
#: single source of truth for both the manifest and the roll-up.
KNOWN_AUDIT_SOURCES: frozenset[str] = frozenset({"competition", "inference"})


class AuditFileRef(BaseModel):
    """One entry on the auditor's worklist: an audit file that backs a weight.

    `digest` is the content address / object key in the audit store (sha256 hex).
    `kind` says whether it is a stored score-packet blob or a bundle-binding digest.
    `challenge_id`/`item_id` pin exactly which scored item it covers; `source` says
    whether the score came from a competition run or an inference round. An auditor
    fetches `digest` from the same object store and recomputes it — the manifest is
    the EXPLICIT, followable index of what must be reproduced (never a vague promise).

    `inclusion_proof` is the per-item merkle inclusion proof — a
    `((sibling_hex, "left"|"right"), ...)` path (the project design record §3.1) that
    opens `digest` against the manifest's `score_packet_merkle_root`. It is populated on
    the SCORE_PACKET ref (whose `digest` is a merkle leaf) by the finalizer, and left
    None on the AUDIT_BUNDLE ref (bundles are proven by recompute, not by leaf
    inclusion). Deterministic from the sorted leaves, so it does not perturb the
    byte-identity of the EpochLog: an empty tuple `()` is the valid proof for a
    single-leaf tree; None means "no proof carried" (the pre-v2 / non-packet shape).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AuditFileKind
    digest: str = Field(pattern=SHA256_HEX_PATTERN)
    challenge_id: str
    item_id: str
    source: str = "inference"  # "competition" | "inference"
    #: merkle inclusion proof opening `digest` against `score_packet_merkle_root`;
    #: only on the SCORE_PACKET ref. `()` = single-leaf tree, None = no proof carried.
    inclusion_proof: tuple[tuple[str, str], ...] | None = None
    #: The item's COMMITTED scoring track (the project design record §9 fix): the
    #: track fixed by the committed challenge, carried on the SCORE_PACKET ref so the
    #: auditor decides recompute-ability and recomputes over THIS track, never the
    #: authority's packet-declared track — a packet substituting `track=upscaling` to
    #: force a GPU-unavailable SKIP is caught (its declared track will not match the
    #: committed one). REQUIRED on every SCORE_PACKET ref (enforced below, #9): a
    #: SCORE_PACKET ref without a committed track would let the auditor fall back to the
    #: packet-controlled track — the exact hole this closes. None on the AUDIT_BUNDLE ref.
    committed_track: str | None = None

    @model_validator(mode="after")
    def _require_known_source(self) -> "AuditFileRef":
        # an internal review: `source` is the label the roll-up's media-coverage floor
        # keys on (`overall_status` counts an all-SKIP media sample as INCONCLUSIVE only
        # for the literal sources competition/inference). An arbitrary string here let a
        # sampled-but-unavailable item be labelled with some OTHER source, so it SKIPped
        # yet dodged the coverage floor and the epoch washed to CLEAN. A source outside
        # the known media set is itself tampering — refuse it at the model level so the
        # floor cannot be side-stepped by a spoofed label (the roll-up is also made
        # label-independent as defense in depth).
        if self.source not in KNOWN_AUDIT_SOURCES:
            raise EpochLogInvalid(
                f"AuditFileRef.source {self.source!r} is not a known media source "
                f"{sorted(KNOWN_AUDIT_SOURCES)} — an unknown source would dodge the "
                "all-SKIP media-coverage floor"
            )
        return self

    @model_validator(mode="after")
    def _require_committed_track_on_packets(self) -> "AuditFileRef":
        # #9: a SCORE_PACKET ref MUST pin the committed track. Optional-and-absent is
        # exactly what let a substituted packet track dodge recompute-ability; refuse it.
        if self.kind is AuditFileKind.SCORE_PACKET and self.committed_track is None:
            raise EpochLogInvalid(
                "a SCORE_PACKET AuditFileRef must carry a committed_track — the auditor "
                "recomputes over the COMMITTED track and never the packet-controlled one "
                f"(item {self.item_id!r}, digest {self.digest})"
            )
        return self

    def _sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.source,
            self.challenge_id,
            self.item_id,
            f"{self.kind.value}:{self.digest}",
        )


class CycleScore(BaseModel):
    """ONE evidence-bound per-cycle score that EWMA-folds into `accumulate_score`.

    EWMA is order-dependent (`[0.1,0.9]`→0.24375 but `[0.9,0.1]`→0.19375 at decay 0.75),
    so a bare ordered float list lets the authority REORDER honest scores to change the
    accumulator while a multiset check still passes — and PAD the list with an unbacked
    0.0 (extra decay) or -1 (substituted exclusion). This binds each cycle score to
    COMMITTED evidence so neither is possible:

    - `packet_digest` — the committed evidence digest this cycle's score comes from. It
      is normally a SCORE_PACKET in the uid's `per_uid` merkle leaves; an exact zero may
      instead name a signed `AvailabilityInput` embedded in the manifest. Every entry
      MUST have exactly one of those backing forms or it FAILS the audit.
    - `ordering_key` — that packet's committed monotonic cycle-sequence index (recorded
      IN the content-addressed packet, so it cannot be changed without changing the
      digest / breaking merkle inclusion). It is the VERIFIABLE fold order: the auditor
      rejects any `cycle_scores` order that is not ascending by the packets' committed
      sequence.
    - `score` — the folded value: the packet's recorded score, or the -1.0 exclusion
      sentinel when (and only when) the committed packet marks an exclusion. A value that
      does not match its committed packet FAILS.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    packet_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    ordering_key: int
    score: float


class EarningInput(BaseModel):
    """The verifiable derivation of ONE uid's earning state (#1).

    The whole point: an auditor must be able to RECONSTRUCT `accumulate_score` from
    audited evidence, so an authority cannot publish honest score packets while
    assigning a uid a SUBSTITUTED `accumulate_score`. This carries exactly the two
    inputs of that reconstruction:

    - `prior_accumulate_score` — the carry-in: the SAME uid's `accumulate_score` in
      the PREVIOUS epoch's log (chained via `EpochLog.prior_log_digest`, back to a
      genesis where it is 0.0). An auditor holding the prior log verifies this equals
      the prior log's stated value — so the fold is verifiable all the way back.
    - `cycle_scores` — the EVIDENCE-BOUND sequence of per-cycle `CycleScore`s that were
      EWMA-folded into `accumulate_score` THIS epoch (`vidaio.tokenomics.ewma.accumulate`,
      decay from config), in ASCENDING committed `ordering_key` order (EWMA is
      history-dependent, and the order is bound to committed evidence — NOT
      authority-chosen). Every entry is bound to committed media or availability
      evidence; the auditor cross-checks each entry's value + ordering key and rejects
      an unbacked entry, a substituted value, or an order that does not match the evidence.

    The auditor re-folds `prior_accumulate_score` over `cycle_scores` and compares to
    the log's stated `accumulate_score`; a mismatch is EARNING_STATE_MISMATCH.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prior_accumulate_score: float = 0.0
    cycle_scores: tuple[CycleScore, ...] = ()

    @model_validator(mode="after")
    def _ordering_keys_ascending(self) -> "EarningInput":
        # The fold order is the committed ordering — strictly ascending, so a reordered
        # (or duplicated-leaf) sequence cannot even be represented in a valid log.
        keys = [c.ordering_key for c in self.cycle_scores]
        if any(b <= a for a, b in zip(keys, keys[1:])):
            raise EpochLogInvalid(
                "EarningInput.cycle_scores must be in strictly ascending committed "
                f"ordering_key order (the evidence-bound fold order); got {keys}"
            )
        return self

    def folded_scores(self) -> tuple[float, ...]:
        """The per-cycle score values in evidence order (what EWMA folds)."""
        return tuple(c.score for c in self.cycle_scores)

    def _canonical_obj(self) -> dict[str, Any]:
        # cycle_scores stays in committed ordering_key order (EWMA is order-dependent).
        return {
            "prior_accumulate_score": self.prior_accumulate_score,
            "cycle_scores": [
                {
                    "packet_digest": c.packet_digest,
                    "ordering_key": c.ordering_key,
                    "score": c.score,
                }
                for c in self.cycle_scores
            ],
        }


class AvailabilityInput(BaseModel):
    """One canonical, persisted availability zero committed into an epoch manifest.

    The observation remains embedded as canonical JSON rather than importing the
    validator's richer availability model into this shared schema module. The
    authority finalizer parses and verifies that model and its signatures before
    constructing a manifest; this dependency-light boundary independently pins the
    exact bytes, digest, identity and committed fold order that an auditor will read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=128)
    challenge_id: str = Field(min_length=1, max_length=128)
    item_id: str = Field(min_length=1, max_length=128)
    track: str = Field(min_length=1, max_length=32)
    ordering_key: int = Field(ge=1)
    observation_json: str = Field(min_length=1)
    observation_digest: str = Field(pattern=SHA256_HEX_PATTERN)

    @model_validator(mode="after")
    def _canonical_observation(self) -> "AvailabilityInput":
        import json

        if self.track not in PROTOCOL_TRACKS:
            raise EpochLogInvalid(
                f"availability input track {self.track!r} is not a protocol track "
                f"{sorted(PROTOCOL_TRACKS)}"
            )
        try:
            value = json.loads(self.observation_json)
        except (TypeError, ValueError) as exc:
            raise EpochLogInvalid(
                "availability observation_json is malformed JSON"
            ) from exc
        if not isinstance(value, dict):
            raise EpochLogInvalid("availability observation_json must be a JSON object")
        try:
            canonical = canonical_json_bytes(value)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise EpochLogInvalid(
                "availability observation_json is not canonically serializable"
            ) from exc
        if canonical != self.observation_json.encode("utf-8"):
            raise EpochLogInvalid(
                "availability observation_json must be the exact canonical JSON bytes"
            )
        expected_digest = sha256_hex(canonical)
        if self.observation_digest != expected_digest:
            raise EpochLogInvalid(
                "availability observation_digest does not equal sha256 of canonical "
                f"observation_json: expected {expected_digest}, got "
                f"{self.observation_digest}"
            )
        return self

    def _sort_key(self) -> tuple[int, int, str, str, str]:
        return (
            self.uid,
            self.ordering_key,
            self.challenge_id,
            self.item_id,
            self.observation_digest,
        )

    def _canonical_obj(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "hotkey": self.hotkey,
            "challenge_id": self.challenge_id,
            "item_id": self.item_id,
            "track": self.track,
            "ordering_key": self.ordering_key,
            "observation_json": self.observation_json,
            "observation_digest": self.observation_digest,
        }


# WindowInput REMOVED with the retention multiplier for v1 (retention removed — owner
# decision; an internal review): with no windowed weight inputs there is
# no committed-window evidence for the auditor to re-derive, so the whole per-uid window
# evidence subsystem is gone from the manifest.


class CompetitionAuditItem(BaseModel):
    """One hidden evaluation item and its pre-enrollment commitments.

    Every competition subject must carry exactly this item set.  Keeping the item
    identities and commitments outside the per-subject packet list prevents a
    producer from evaluating the baseline and a contender on different holdouts while
    still publishing self-consistent means.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    challenge_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    threshold_commitment: str = Field(pattern=SHA256_HEX_PATTERN)
    item_index: int | None = Field(default=None, ge=0)
    input_sha256: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    reference_sha256: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    upscale_factor: Literal[2, 4] | None = None
    target_width: int | None = Field(default=None, gt=0)
    target_height: int | None = Field(default=None, gt=0)
    item_commitment: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)

    def _canonical_obj(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "item_id": self.item_id,
            "threshold_commitment": self.threshold_commitment,
            "item_index": self.item_index,
            "input_sha256": self.input_sha256,
            "reference_sha256": self.reference_sha256,
            "upscale_factor": self.upscale_factor,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "item_commitment": self.item_commitment,
        }


class CompetitionAuditSubject(BaseModel):
    """One competition participant/baseline and the exact packet set behind it.

    Competition evaluation intentionally reuses an evaluation item's challenge/item identity
    across contenders. ``subject_id`` therefore namespaces its manifest refs and prevents two
    contenders from collapsing into one sampler entry.  The economic aggregate is the arithmetic
    mean of ``packet_digests`` under ``mean_item_score.v2``; the auditor reads those exact packets,
    checks every paired bundle, and derives the aggregate itself.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_id: str = Field(min_length=1, max_length=512)
    role: Literal["contender", "baseline"]
    uid: int | None = Field(default=None, ge=0)
    hotkey: str | None = None
    #: Deterministic close-census/exact-output duplicate suppression. The CPU
    #: auditor independently re-derives this bit before accepting the result.
    dedup_excluded: bool = False
    submission_archive_digest: str | None = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )
    submission_archive_bytes: int | None = Field(default=None, gt=0)
    execution_image_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    repo_url: str | None = Field(default=None, min_length=1, max_length=2048)
    commit_sha: str | None = Field(default=None, pattern=GIT_SHA1_HEX_PATTERN)
    tree_sha: str | None = Field(default=None, pattern=GIT_SHA1_HEX_PATTERN)
    packet_digests: tuple[str, ...] = Field(min_length=1)
    audit_bundle_digests: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _identity_and_pairs(self) -> "CompetitionAuditSubject":
        if self.role == "contender":
            if self.uid is None or not self.hotkey:
                raise EpochLogInvalid(
                    "a competition contender audit subject requires a uid and hotkey"
                )
            missing_release_fields = [
                name
                for name in (
                    "submission_archive_digest",
                    "submission_archive_bytes",
                    "repo_url",
                    "commit_sha",
                    "tree_sha",
                )
                if getattr(self, name) is None
            ]
            if missing_release_fields:
                raise EpochLogInvalid(
                    "a competition contender audit subject lacks sealed release identity: "
                    + ", ".join(missing_release_fields)
                )
        else:
            if self.uid is not None or self.hotkey is not None:
                raise EpochLogInvalid(
                    "the non-earning baseline audit subject cannot carry a uid or hotkey"
                )
            if self.dedup_excluded:
                raise EpochLogInvalid("the non-earning baseline cannot be dedup-excluded")
            contender_only = (
                self.submission_archive_digest,
                self.submission_archive_bytes,
                self.repo_url,
                self.commit_sha,
                self.tree_sha,
            )
            if any(value is not None for value in contender_only):
                raise EpochLogInvalid(
                    "the non-earning baseline cannot carry contender submission identity"
                )
        if len(self.packet_digests) != len(self.audit_bundle_digests):
            raise EpochLogInvalid(
                f"competition subject {self.subject_id!r} has "
                f"{len(self.packet_digests)} packet digest(s) but "
                f"{len(self.audit_bundle_digests)} bundle digest(s)"
            )
        if len(set(self.packet_digests)) != len(self.packet_digests):
            raise EpochLogInvalid(
                f"competition subject {self.subject_id!r} repeats a score packet digest"
            )
        if len(set(self.audit_bundle_digests)) != len(self.audit_bundle_digests):
            raise EpochLogInvalid(
                f"competition subject {self.subject_id!r} repeats an audit bundle digest"
            )
        return self

    def _canonical_obj(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "role": self.role,
            "uid": self.uid,
            "hotkey": self.hotkey,
            "dedup_excluded": self.dedup_excluded,
            "submission_archive_digest": self.submission_archive_digest,
            "submission_archive_bytes": self.submission_archive_bytes,
            "execution_image_digest": self.execution_image_digest,
            "repo_url": self.repo_url,
            "commit_sha": self.commit_sha,
            "tree_sha": self.tree_sha,
            "packet_digests": list(self.packet_digests),
            "audit_bundle_digests": list(self.audit_bundle_digests),
        }


class CompetitionInput(BaseModel):
    """Committed, CPU-recomputable inputs of one economic competition result.

    Human review never enters this structure.  The ranked result is derived only from the
    subject packet means, then by stable ``(-score, hotkey, uid)`` ordering.  The manifest and
    its pre-enrollment commitment root are retained as provenance; every score packet and bundle
    is also included in the epoch merkle/audit worklist.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    competition_id: str = Field(min_length=1)
    track: Literal["compression", "upscaling"]
    cycle: int = Field(ge=1)
    completed_at: datetime
    applied_at: datetime
    manifest_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    commitment_root: str = Field(pattern=SHA256_HEX_PATTERN)
    #: Exact finalized pre-enrollment commitment receipt.  The payload is carried
    #: verbatim (hex-encoded only so canonical JSON remains text) and is required
    #: to be the protocol's domain-tagged payload for ``commitment_root``.  The
    #: inclusion block/hash and the finalized head observed by the orchestrator
    #: let an independent CPU auditor re-read the same raw commitment record from
    #: archive state instead of trusting a database-local ``commitment_root``.
    anchor_netuid: int = Field(ge=0)
    anchor_payload_hex: str = Field(pattern=r"^(?:[0-9a-f]{2}){1,128}$")
    anchor_payload_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    anchor_block: int = Field(ge=0)
    anchor_block_hash: str = Field(pattern=SHA256_HEX_PATTERN)
    anchor_finalized_block: int = Field(ge=0)
    baseline_version: int = Field(ge=0)
    baseline_artifact_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    baseline_artifact_bytes: int = Field(gt=0)
    baseline_execution_image_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    baseline_provenance_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    baseline_provenance_bytes: int = Field(gt=0)
    aggregation_version: Literal["mean_item_score.v2"] = "mean_item_score.v2"
    items: tuple[CompetitionAuditItem, ...] = Field(min_length=1)
    subjects: tuple[CompetitionAuditSubject, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _complete_subject_set(self) -> "CompetitionInput":
        tz = self.completed_at.tzinfo
        if tz is None or tz.utcoffset(self.completed_at) is None:
            raise EpochLogInvalid("competition completed_at must be timezone-aware")
        applied_tz = self.applied_at.tzinfo
        if applied_tz is None or applied_tz.utcoffset(self.applied_at) is None:
            raise EpochLogInvalid("competition applied_at must be timezone-aware")
        if self.completed_at > self.applied_at:
            raise EpochLogInvalid(
                "competition completed_at cannot be after its epoch applied_at"
            )
        expected_payload = (
            f"{COMMITMENT_DOMAIN}:competition:{self.commitment_root}".encode("ascii")
        )
        try:
            committed_payload = bytes.fromhex(self.anchor_payload_hex)
        except ValueError as exc:  # defensive: the field pattern already fences this
            raise EpochLogInvalid(
                "competition anchor payload is not canonical lowercase hex"
            ) from exc
        if committed_payload != expected_payload:
            raise EpochLogInvalid(
                "competition anchor payload does not bind the committed competition root"
            )
        if sha256_hex(committed_payload) != self.anchor_payload_digest:
            raise EpochLogInvalid(
                "competition anchor payload digest does not bind the exact payload bytes"
            )
        if self.anchor_finalized_block < self.anchor_block:
            raise EpochLogInvalid(
                "competition anchor receipt finalized block precedes its inclusion block"
            )
        subject_ids = [subject.subject_id for subject in self.subjects]
        if len(set(subject_ids)) != len(subject_ids):
            raise EpochLogInvalid("competition audit subject_id values must be unique")
        contenders = [
            subject for subject in self.subjects if subject.role == "contender"
        ]
        baselines = [
            subject for subject in self.subjects if subject.role == "baseline"
        ]
        if not contenders:
            raise EpochLogInvalid(
                "an earning competition requires at least one contender"
            )
        if len(baselines) != 1:
            raise EpochLogInvalid(
                "an earning competition requires exactly one non-earning baseline"
            )
        if (
            len(baselines) == 1
            and baselines[0].execution_image_digest
            != self.baseline_execution_image_digest
        ):
            raise EpochLogInvalid(
                "competition baseline subject execution image does not match the committed "
                "baseline execution image"
            )
        identities = [(subject.uid, subject.hotkey) for subject in contenders]
        if len(set(identities)) != len(identities):
            raise EpochLogInvalid(
                "competition contender uid/hotkey identities must be unique"
            )
        packet_digests = [
            digest for subject in self.subjects for digest in subject.packet_digests
        ]
        if len(set(packet_digests)) != len(packet_digests):
            raise EpochLogInvalid(
                "competition packet digests must be unique across audit subjects"
            )
        bundle_digests = [
            digest
            for subject in self.subjects
            for digest in subject.audit_bundle_digests
        ]
        if len(set(bundle_digests)) != len(bundle_digests):
            raise EpochLogInvalid(
                "competition bundle digests must be unique across audit subjects"
            )
        item_ids = [(item.challenge_id, item.item_id) for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise EpochLogInvalid("competition audit item identities must be unique")
        if self.track == "upscaling":
            for expected_index, item in enumerate(self.items):
                if item.item_index != expected_index:
                    raise EpochLogInvalid(
                        "upscaling competition audit items must cover ordered indices "
                        f"0..N-1; expected {expected_index}, got {item.item_index}"
                    )
                if (
                    item.input_sha256 is None
                    or item.reference_sha256 is None
                    or item.upscale_factor is None
                    or item.item_commitment is None
                ):
                    raise EpochLogInvalid(
                        f"upscaling competition item {expected_index} lacks its "
                        "reference/input/factor commitment preimage"
                    )
                if item.input_sha256 == item.reference_sha256:
                    raise EpochLogInvalid(
                        f"upscaling competition item {expected_index} aliases "
                        "reference and miner input"
                    )
                if (item.target_width is None) != (item.target_height is None):
                    raise EpochLogInvalid(
                        f"upscaling competition item {expected_index} has incomplete "
                        "target geometry"
                    )
        return self

    def _canonical_obj(self) -> dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "track": self.track,
            "cycle": self.cycle,
            "completed_at": self.completed_at.isoformat(),
            "applied_at": self.applied_at.isoformat(),
            "manifest_digest": self.manifest_digest,
            "commitment_root": self.commitment_root,
            "anchor_netuid": self.anchor_netuid,
            "anchor_payload_hex": self.anchor_payload_hex,
            "anchor_payload_digest": self.anchor_payload_digest,
            "anchor_block": self.anchor_block,
            "anchor_block_hash": self.anchor_block_hash,
            "anchor_finalized_block": self.anchor_finalized_block,
            "baseline_version": self.baseline_version,
            "baseline_artifact_digest": self.baseline_artifact_digest,
            "baseline_artifact_bytes": self.baseline_artifact_bytes,
            "baseline_execution_image_digest": self.baseline_execution_image_digest,
            "baseline_provenance_digest": self.baseline_provenance_digest,
            "baseline_provenance_bytes": self.baseline_provenance_bytes,
            "aggregation_version": self.aggregation_version,
            "items": [item._canonical_obj() for item in self.items],
            "subjects": [subject._canonical_obj() for subject in self.subjects],
        }


class AuditManifest(BaseModel):
    """The audit manifest: which audit files back each weight (the auditor's index).

    `per_uid[uid]` is exactly the `list[AuditFileRef]` an auditor must fetch and
    recompute to reproduce uid's score — the literal `dict[uid, list[AuditFileRef]]`
    worklist. `baseline_bundles` carries non-earning calibration rows that remain
    auditable. `score_packet_merkle_root`, when
    present, is the committed-set root the per-item inclusion proofs open against
    (the project design record §3.1).

    `earning_inputs[uid]` is the verifiable earning-state derivation for that uid
    (#1) — present for every nonzero-weight uid the finalizer produces, so the
    auditor can re-fold `accumulate_score` from the audited packet scores + the
    chained prior carry-in rather than trusting the authority's stated value.

    ``competition_input`` commits the deterministic economic result inputs and
    ``competition_bundles`` namespaces the corresponding packet/bundle worklists by subject.
    ``availability_inputs`` embeds canonical signed non-media zeros that back matching
    earning cycles without entering media sampling or the score-packet Merkle tree.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_uid: dict[int, tuple[AuditFileRef, ...]] = Field(default_factory=dict)
    baseline_bundles: tuple[AuditFileRef, ...] = ()
    score_packet_merkle_root: str | None = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )
    earning_inputs: dict[int, EarningInput] = Field(default_factory=dict)
    availability_inputs: tuple[AvailabilityInput, ...] = ()
    competition_input: CompetitionInput | None = None
    competition_bundles: dict[str, tuple[AuditFileRef, ...]] = Field(
        default_factory=dict
    )
    #: Total cumulative replay boundary. Every current census uid appears: ``None`` means the
    #: uid slot has been observed but has never folded a cycle; an integer is the greatest
    #: committed dispatch ordering key folded in this or any predecessor epoch. Entries remain
    #: as tombstones across idle, exclusion, deregistration, hotkey reuse, and empty epochs.
    fold_cursors: dict[int, int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _competition_evidence_complete(self) -> "AuditManifest":
        if self.competition_input is None:
            if self.competition_bundles:
                raise EpochLogInvalid(
                    "competition_bundles are present without a competition_input"
                )
            return self
        subjects = {
            subject.subject_id: subject for subject in self.competition_input.subjects
        }
        if set(self.competition_bundles) != set(subjects):
            raise EpochLogInvalid(
                "competition_bundles subjects do not exactly match competition_input: "
                f"expected {sorted(subjects)}, got {sorted(self.competition_bundles)}"
            )
        for subject_id, subject in subjects.items():
            refs = self.competition_bundles[subject_id]
            packet_refs = [
                ref for ref in refs if ref.kind is AuditFileKind.SCORE_PACKET
            ]
            bundle_refs = [
                ref for ref in refs if ref.kind is AuditFileKind.AUDIT_BUNDLE
            ]
            if any(ref.source != "competition" for ref in refs):
                raise EpochLogInvalid(
                    f"competition subject {subject_id!r} carries a non-competition audit ref"
                )
            if any(
                ref.committed_track != self.competition_input.track
                for ref in packet_refs
            ):
                raise EpochLogInvalid(
                    f"competition subject {subject_id!r} packet refs do not all pin track "
                    f"{self.competition_input.track!r}"
                )
            if sorted(ref.digest for ref in packet_refs) != sorted(
                subject.packet_digests
            ):
                raise EpochLogInvalid(
                    f"competition subject {subject_id!r} packet refs do not exactly match "
                    "its committed packet digest set"
                )
            if sorted(ref.digest for ref in bundle_refs) != sorted(
                subject.audit_bundle_digests
            ):
                raise EpochLogInvalid(
                    f"competition subject {subject_id!r} bundle refs do not exactly match "
                    "its committed audit-bundle digest set"
                )
            expected_items = sorted(
                (item.challenge_id, item.item_id)
                for item in self.competition_input.items
            )
            if (
                sorted((ref.challenge_id, ref.item_id) for ref in packet_refs)
                != expected_items
            ):
                raise EpochLogInvalid(
                    f"competition subject {subject_id!r} packet refs do not exactly cover "
                    "the committed evaluation item set"
                )
            if (
                sorted((ref.challenge_id, ref.item_id) for ref in bundle_refs)
                != expected_items
            ):
                raise EpochLogInvalid(
                    f"competition subject {subject_id!r} bundle refs do not exactly cover "
                    "the committed evaluation item set"
                )
        return self

    @model_validator(mode="after")
    def _availability_evidence_complete(self) -> "AuditManifest":
        digests = [evidence.observation_digest for evidence in self.availability_inputs]
        if len(set(digests)) != len(digests):
            raise EpochLogInvalid(
                "availability_inputs must contain unique observation digests"
            )

        identities = [
            (evidence.uid, evidence.challenge_id, evidence.item_id)
            for evidence in self.availability_inputs
        ]
        if len(set(identities)) != len(identities):
            raise EpochLogInvalid(
                "availability_inputs must contain unique uid/challenge/item identities"
            )

        ordering_keys = [
            (evidence.uid, evidence.ordering_key)
            for evidence in self.availability_inputs
        ]
        if len(set(ordering_keys)) != len(ordering_keys):
            raise EpochLogInvalid(
                "availability_inputs must contain unique per-uid committed ordering keys"
            )

        media_digests = {
            ref.digest
            for refs in (
                tuple(self.per_uid.values())
                + (self.baseline_bundles,)
                + tuple(self.competition_bundles.values())
            )
            for ref in refs
        }
        overlap = sorted(set(digests) & media_digests)
        if overlap:
            raise EpochLogInvalid(
                "availability observations are non-media evidence and cannot also appear "
                f"as audit file refs: {overlap}"
            )

        for evidence in self.availability_inputs:
            matches = [
                (uid, cycle)
                for uid, earning_input in self.earning_inputs.items()
                for cycle in earning_input.cycle_scores
                if cycle.packet_digest == evidence.observation_digest
            ]
            if len(matches) != 1:
                raise EpochLogInvalid(
                    f"availability observation {evidence.observation_digest} must back "
                    "exactly one CycleScore for its uid"
                )
            uid, cycle = matches[0]
            if (
                uid != evidence.uid
                or cycle.ordering_key != evidence.ordering_key
                or cycle.score != 0.0
            ):
                raise EpochLogInvalid(
                    f"availability observation {evidence.observation_digest} must bind an "
                    "exact score=0 CycleScore at its committed ordering key"
                )
        return self

    def refs_for(self, uid: int) -> tuple[AuditFileRef, ...]:
        """The worklist backing `uid` (empty if none)."""
        return self.per_uid.get(uid, ())

    def earning_for(self, uid: int) -> EarningInput | None:
        """The earning-state derivation for `uid` (None if none carried)."""
        return self.earning_inputs.get(uid)

    def availability_for(self, uid: int) -> tuple[AvailabilityInput, ...]:
        """Canonical non-media zero observations folded for ``uid`` this epoch."""
        return tuple(
            evidence for evidence in self.availability_inputs if evidence.uid == uid
        )

    def _canonical_obj(self) -> dict[str, Any]:
        return {
            "per_uid": {
                str(uid): [
                    r.model_dump(mode="json")
                    for r in sorted(refs, key=lambda x: x._sort_key())
                ]
                for uid, refs in self.per_uid.items()
            },
            "baseline_bundles": [
                r.model_dump(mode="json")
                for r in sorted(self.baseline_bundles, key=lambda x: x._sort_key())
            ],
            "score_packet_merkle_root": self.score_packet_merkle_root,
            # dict keys canonicalize (sorted) via canonical_json_bytes; stringify uids.
            "earning_inputs": {
                str(uid): ei._canonical_obj() for uid, ei in self.earning_inputs.items()
            },
            "availability_inputs": [
                evidence._canonical_obj()
                for evidence in sorted(
                    self.availability_inputs, key=lambda evidence: evidence._sort_key()
                )
            ],
            "competition_input": (
                self.competition_input._canonical_obj()
                if self.competition_input is not None
                else None
            ),
            "competition_bundles": {
                subject_id: [
                    ref.model_dump(mode="json")
                    for ref in sorted(refs, key=lambda ref: ref._sort_key())
                ]
                for subject_id, refs in self.competition_bundles.items()
            },
            "fold_cursors": {
                str(uid): ordering_key for uid, ordering_key in self.fold_cursors.items()
            },
        }


class MinerCensusEntry(BaseModel):
    """One registered subnet identity at the epoch's exact close block.

    It deliberately carries no track or earning state. A registered identity may be offline,
    newly registered, control-only, or have an unresolved warrant track and still belongs in
    the independently bindable chain census, while ``EpochLog.miners`` contains the
    eligible/economic subset. Validator permit is a non-exclusive capability and does not
    remove an otherwise serving miner.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uid: int = Field(ge=0)
    hotkey: str
    coldkey: str
    ip: str

    @classmethod
    def from_miner(cls, miner: MinerSnapshot) -> "MinerCensusEntry":
        return cls(
            uid=miner.uid,
            hotkey=miner.hotkey,
            coldkey=miner.coldkey,
            ip=miner.ip,
        )


def _census_obj(entry: MinerCensusEntry) -> dict[str, Any]:
    return {
        "uid": entry.uid,
        "hotkey": entry.hotkey,
        "coldkey": entry.coldkey,
        "ip": entry.ip,
    }


# ---- deterministic serializers for the tokenomics state dataclasses ------------------
#
# The state models are frozen stdlib dataclasses (vidaio.tokenomics.state). We serialize
# them EXPLICITLY (not via pydantic dataclass handling) so the canonical shape, and every
# collection's order, is fully under our control — the byte-identity property depends on it.


def _miner_obj(m: MinerSnapshot) -> dict[str, Any]:
    return {
        "uid": m.uid,
        "hotkey": m.hotkey,
        "coldkey": m.coldkey,
        "ip": m.ip,
        "track": m.track,
        "accumulate_score": m.accumulate_score,
        "excluded": m.excluded,
    }


def _miner_from_obj(d: dict[str, Any]) -> MinerSnapshot:
    return MinerSnapshot(
        uid=int(d["uid"]),
        hotkey=d["hotkey"],
        coldkey=d["coldkey"],
        ip=d["ip"],
        track=d["track"],
        accumulate_score=float(d["accumulate_score"]),
        excluded=bool(d["excluded"]),
    )


def _contender_obj(contender: ContenderResult) -> dict[str, Any]:
    return {
        "hotkey": contender.hotkey,
        "uid": contender.uid,
        "score": contender.score,
    }


def _competition_obj(result: CompetitionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "competition_id": result.competition_id,
        "track": result.track,
        "cycle": result.cycle,
        "applied_at": result.applied_at.isoformat(),
        "contenders": [_contender_obj(contender) for contender in result.contenders],
        "baseline_score": result.baseline_score,
        "baseline_version": result.baseline_version,
        "baseline_artifact_digest": result.baseline_artifact_digest,
    }


def _competition_from_obj(value: dict[str, Any] | None) -> CompetitionResult | None:
    if value is None:
        return None
    return CompetitionResult(
        competition_id=value["competition_id"],
        track=value["track"],
        cycle=int(value["cycle"]),
        applied_at=datetime.fromisoformat(value["applied_at"]),
        contenders=tuple(
            ContenderResult(
                hotkey=contender["hotkey"],
                uid=int(contender["uid"]),
                score=float(contender["score"]),
            )
            for contender in value.get("contenders", [])
        ),
        baseline_score=(
            None
            if value.get("baseline_score") is None
            else float(value["baseline_score"])
        ),
        baseline_version=int(value["baseline_version"]),
        baseline_artifact_digest=value["baseline_artifact_digest"],
    )


def _reward_window_obj(state: RewardWindowState) -> dict[str, Any]:
    return {
        "kind": state.kind.value,
        "starts_at": state.starts_at.isoformat() if state.starts_at is not None else None,
        "ends_at": state.ends_at.isoformat() if state.ends_at is not None else None,
        "podium_hotkeys": list(state.podium_hotkeys),
        "winner_hotkey": state.winner_hotkey,
        "winner_uid": state.winner_uid,
        "winner_score": state.winner_score,
        "winner_margin": state.winner_margin,
        "baseline_score": state.baseline_score,
        "baseline_version": state.baseline_version,
        "baseline_artifact_digest": state.baseline_artifact_digest,
        "source_competition_id": state.source_competition_id,
        "source_track": state.source_track,
        "source_cycle": state.source_cycle,
        "last_applied_cycle": state.last_applied_cycle,
    }


def _reward_window_from_obj(value: dict[str, Any]) -> RewardWindowState:
    return RewardWindowState(
        kind=value["kind"],
        starts_at=(
            None
            if value.get("starts_at") is None
            else datetime.fromisoformat(value["starts_at"])
        ),
        ends_at=(
            None
            if value.get("ends_at") is None
            else datetime.fromisoformat(value["ends_at"])
        ),
        podium_hotkeys=tuple(value.get("podium_hotkeys", [])),
        winner_hotkey=value.get("winner_hotkey"),
        winner_uid=(
            None if value.get("winner_uid") is None else int(value["winner_uid"])
        ),
        winner_score=(
            None
            if value.get("winner_score") is None
            else float(value["winner_score"])
        ),
        winner_margin=(
            None
            if value.get("winner_margin") is None
            else float(value["winner_margin"])
        ),
        baseline_score=(
            None
            if value.get("baseline_score") is None
            else float(value["baseline_score"])
        ),
        baseline_version=(
            None
            if value.get("baseline_version") is None
            else int(value["baseline_version"])
        ),
        baseline_artifact_digest=value.get("baseline_artifact_digest"),
        source_competition_id=value.get("source_competition_id"),
        source_track=value.get("source_track"),
        source_cycle=(
            None
            if value.get("source_cycle") is None
            else int(value["source_cycle"])
        ),
        last_applied_cycle=(
            None
            if value.get("last_applied_cycle") is None
            else int(value["last_applied_cycle"])
        ),
    )


def weight_vector_digest(weight_u16: dict[int, int]) -> str:
    """sha256 over the canonical u16 weight-vector document (uid-ascending pairs).

    THE digest a validator cross-checks and the on-chain publication binds — a
    function of the u16 pairs only, so it is identical wherever the same vector is
    quantized. (the project design record Part 5 `weight_vector_document`.)
    """
    pairs = [[uid, weight_u16[uid]] for uid in sorted(weight_u16)]
    return sha256_hex(canonical_json_bytes(pairs))


class EpochLog(BaseModel):
    """The immutable per-epoch log: weights + the inputs they came from + audit manifest.

    Frozen and content-addressed: `log_digest()` = sha256 over `to_json()` (canonical
    bytes), the value anchored on chain and verified against fetched bytes by every
    validator. Carries, per epoch:

    - identity/pinning: `epoch_id`, `close_block` (the epoch-close block it is pinned
      to), `scorer_version` (the scorer identity), `schema_version`, `created_at`;
    - convergence inputs: the inference `MinerSnapshot` set, an optional newly applied
      packet-derived `CompetitionResult`, and predecessor-folded `RewardWindowState`;
    - the canonical weight vector BOTH as `weight_shares` (float, `build_weight_vector`
      output) AND `weight_u16` (the deterministic `quantize_u16` of it), plus
      `weight_vector_digest`;
    - `audit_manifest`: per nonzero-weight uid, the audit files that back its score.

    Construction VALIDATES (else `EpochLogInvalid`): `weight_u16 == quantize_u16(
    weight_shares)` (the convergence cross-check), `weight_vector_digest` binds the
    u16 vector, and every nonzero-weight uid (other than the empty-epoch `burn_uid`)
    has a non-empty manifest entry — a weight with no audit backing is rejected
    because an auditor could never reproduce it.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    schema_version: int = EPOCH_LOG_SCHEMA_VERSION
    epoch_id: int
    close_block: int
    scorer_version: str
    created_at: datetime
    #: The PREVIOUS epoch's `log_digest()` — chains the epoch logs so the earning-state
    #: carry-in (`EarningInput.prior_accumulate_score`) is verifiable back to a genesis
    #: (#1). None at genesis / when no prior epoch is referenced. An auditor holding the
    #: prior log checks `prior_log_digest == prior_log.log_digest()` before trusting a
    #: nonzero carry-in.
    prior_log_digest: str | None = Field(default=None, pattern=SHA256_HEX_PATTERN)
    #: Runtime epochs SKIPPED between the chained predecessor and this epoch (schema
    #: v16, the outage-gap recovery of P1.5). Empty for normal succession. When
    #: non-empty it is the FULL contiguous range ``prior_epoch_id+1 .. epoch_id-1``:
    #: epochs whose un-grindable anchor windows (``close + K``) had already elapsed
    #: when this log was built, so no log for them can ever be published without
    #: becoming a late, grindable anchor. Declaring them here (inside an epoch that
    #: IS anchored on time) makes the outage a signed, on-chain, auditable fact
    #: instead of a permanently wedged spine: the earning carry-in and the digest
    #: chain continue from the last real predecessor, and auditors advance their
    #: contiguous cursor over exactly this declared range — never over a silent 404.
    gap_epochs: tuple[int, ...] = ()
    #: Canonical subnet-owner sink for emission that cannot be allocated without
    #: cross-subsidising an empty/below-floor pool.  It is also the sole recipient in a
    #: genuinely empty epoch and is always exempt from measurement-evidence coverage.
    burn_uid: int | None = None

    #: Newly completed, packet-score-derived economic result applied at this epoch close.
    competition_result: CompetitionResult | None = None
    #: Reward window resolved from the chained predecessor plus ``competition_result``.
    reward_window_state: RewardWindowState = Field(default_factory=RewardWindowState)

    miners: tuple[MinerSnapshot, ...]
    #: Complete registered subnet identity set at ``close_block``. This is distinct
    #: from ``miners``, the eligible/economic subset: no track is required here, so an offline
    #: or unknown-track registration cannot force a fabricated track or disappear silently.
    miner_census: tuple[MinerCensusEntry, ...] = ()

    weight_shares: dict[int, float]
    weight_u16: dict[int, int]
    weight_vector_digest: str = Field(pattern=SHA256_HEX_PATTERN)

    audit_manifest: AuditManifest

    @model_validator(mode="after")
    def _validate(self) -> "EpochLog":
        # an internal review: ENFORCE the schema version. `schema_version` DEFAULTS to
        # EPOCH_LOG_SCHEMA_VERSION but was never checked — `from_json` accepted ANY integer,
        # so canonical CURRENT-shape bytes LABELLED schema 9 or 11 could be anchored, audited
        # CLEAN, and submitted, defeating the version_key convergence fence (validators on a
        # DIFFERENT code version must NOT converge on a foreign-schema log). A log whose
        # schema_version is not the code's own EPOCH_LOG_SCHEMA_VERSION is refused at the shared
        # construction / from_json boundary (EpochLogInvalid) — the mixed-version fence.
        if self.schema_version != EPOCH_LOG_SCHEMA_VERSION:
            raise EpochLogInvalid(
                f"schema_version {self.schema_version} != EPOCH_LOG_SCHEMA_VERSION "
                f"{EPOCH_LOG_SCHEMA_VERSION} — a log on a foreign schema is refused (a "
                "current-shape payload mislabelled to a different version would defeat the "
                "version_key convergence fence, an internal review)"
            )
        # v16 outage-gap declaration (P1.5). A gap is only meaningful against an
        # anchored predecessor, and only as the FULL contiguous range immediately
        # below this epoch: anything else would let a log skip history selectively
        # while still chaining a digest (a partial skip is indistinguishable from
        # censorship, so it is refused at the shared construction/from_json boundary).
        if self.gap_epochs:
            if self.prior_log_digest is None:
                raise EpochLogInvalid(
                    "gap_epochs declared on a genesis log (prior_log_digest is None) — "
                    "an outage gap requires an anchored predecessor to chain from; a "
                    "fresh deployment expresses its start with auditor_cursor_floor, "
                    "never with a gap"
                )
            expected = tuple(range(self.gap_epochs[0], self.epoch_id))
            if self.gap_epochs != expected:
                raise EpochLogInvalid(
                    f"gap_epochs {self.gap_epochs} is not the full contiguous range "
                    f"ending at epoch_id-1 ({self.epoch_id - 1}) — a declared outage "
                    "gap must cover every skipped epoch between the chained "
                    "predecessor and this epoch, with no holes and no reordering"
                )
            if self.gap_epochs[0] < 1:
                raise EpochLogInvalid(
                    f"gap_epochs starts at {self.gap_epochs[0]} — the chained "
                    "predecessor would have a negative epoch id"
                )
        # an internal review: REQUIRE a timezone-AWARE created_at. `from_json` parses it with
        # `datetime.fromisoformat`, which happily yields an offset-NAIVE datetime for bytes
        # that omit the offset. The round-9 #6 close-block-time compare subtracts created_at
        # from an AWARE chain timestamp — `aware - naive` raises TypeError, an uncaught crash
        # that blocks the audit cursor. Rejecting naive here (the shared construction /
        # from_json boundary) keeps canonical bytes always tz-aware, so the subtraction is
        # always well-typed and a malformed time is a clean EpochLogInvalid, never a crash.
        tz = self.created_at.tzinfo
        if tz is None or tz.utcoffset(self.created_at) is None:
            raise EpochLogInvalid(
                f"created_at {self.created_at!r} is timezone-NAIVE — an epoch log's created_at "
                "must be tz-aware (the auditor compares it against an aware close-block time; a "
                "naive value would crash that subtraction, an internal review)"
            )
        if self.competition_result is not None:
            applied_tz = self.competition_result.applied_at.tzinfo
            if (
                applied_tz is None
                or applied_tz.utcoffset(self.competition_result.applied_at) is None
            ):
                raise EpochLogInvalid(
                    "competition_result.applied_at must be timezone-aware"
                )
            if self.competition_result.applied_at != self.created_at:
                raise EpochLogInvalid(
                    "competition_result.applied_at must equal the epoch close-block "
                    "created_at; database-local completion time cannot drive economics"
                )
            comp_input = self.audit_manifest.competition_input
            if comp_input is None:
                raise EpochLogInvalid(
                    "competition_result is present without committed competition_input evidence"
                )
            if (
                comp_input.competition_id != self.competition_result.competition_id
                or comp_input.track != self.competition_result.track
                or comp_input.cycle != self.competition_result.cycle
                or comp_input.applied_at != self.competition_result.applied_at
                or comp_input.baseline_version
                != self.competition_result.baseline_version
                or comp_input.baseline_artifact_digest
                != self.competition_result.baseline_artifact_digest
            ):
                raise EpochLogInvalid(
                    "competition_input identity/applied time/baseline provenance do not bind "
                    "competition_result"
                )
            if comp_input.completed_at > self.competition_result.applied_at:
                raise EpochLogInvalid(
                    "competition_input.completed_at is after the result application time"
                )
            if comp_input.anchor_block >= self.close_block:
                raise EpochLogInvalid(
                    "earning competition pre-enrollment anchor inclusion block must "
                    "strictly precede the earning epoch close block"
                )
            if comp_input.anchor_finalized_block >= self.close_block:
                raise EpochLogInvalid(
                    "earning competition finalized receipt must strictly precede the "
                    "earning epoch close block"
                )
            expected_identities = sorted(
                (subject.uid, subject.hotkey)
                for subject in comp_input.subjects
                if subject.role == "contender" and not subject.dedup_excluded
            )
            result_identities = sorted(
                (contender.uid, contender.hotkey)
                for contender in self.competition_result.contenders
            )
            if expected_identities != result_identities:
                raise EpochLogInvalid(
                    "competition_result contenders do not exactly match committed subjects"
                )
            # Every newly applied contender is a registered close-block identity,
            # even when it is absent from the narrower economic/inference snapshot
            # (offline, no resolved warrant track, below the inference floor, ...).
            # Such an absent podium rank is intentionally *unpayable*: the pure
            # composer leaves its fixed share for the canonical sink.  Requiring the
            # contender to appear in ``miners`` here would turn that documented sink
            # path into an epoch-wide HOLD.  Bind the immutable result identity to the
            # complete registered census instead.
            census_identities = {
                (entry.uid, entry.hotkey) for entry in self.miner_census
            }
            unregistered = [
                (contender.uid, contender.hotkey)
                for contender in self.competition_result.contenders
                if (contender.uid, contender.hotkey) not in census_identities
            ]
            if unregistered:
                raise EpochLogInvalid(
                    "competition_result contender identities are absent from the "
                    f"complete close-block miner_census: {unregistered}"
                )
        elif self.audit_manifest.competition_input is not None:
            raise EpochLogInvalid(
                "competition_input is present but competition_result is absent"
            )
        state = self.reward_window_state
        reward_active = (
            state.starts_at is not None
            and state.ends_at is not None
            and state.starts_at <= self.created_at < state.ends_at
        )
        # Reward-window recipients are the committed podium *hotkeys*.  ``winner_uid``
        # remains immutable provenance for the source result, not a seven-day lease on
        # one numeric chain slot.  On later epochs the same registered hotkey may occupy
        # a different uid; ``build_weight_vector`` pays its current economic snapshot
        # and the weight-setter binds that uid back to the current census.  A hotkey that
        # is absent from ``miners`` receives nothing and its fixed share goes to the sink.
        # an internal review: REJECT duplicate snapshot uids. `build_weight_vector`
        # consumes EVERY snapshot row (folding each duplicate into the vector before
        # `by_uid` collapses them by uid), while the auditor's uid->snapshot maps keep
        # only the LAST row — so a read-only probe could add a duplicate uid on another
        # track and turn an honest 40/40/20 vector into 50/40/10 that STILL re-derives
        # cleanly against the collapsed map. A snapshot set is a per-uid census; two rows
        # for one uid is itself malformed, so refuse it here where both the finalizer and
        # `from_json` (untrusted bytes) re-run the invariant.
        uids = [m.uid for m in self.miners]
        if len(set(uids)) != len(uids):
            from collections import Counter

            dupes = sorted(uid for uid, n in Counter(uids).items() if n > 1)
            raise EpochLogInvalid(
                f"duplicate miner snapshot uid(s) {dupes} — a snapshot set is a per-uid "
                "census; duplicate rows let unaudited rows influence the weight vector "
                "while the auditor's uid->snapshot map keeps only the last"
            )
        census_uids = [entry.uid for entry in self.miner_census]
        if len(set(census_uids)) != len(census_uids):
            from collections import Counter

            dupes = sorted(uid for uid, n in Counter(census_uids).items() if n > 1)
            raise EpochLogInvalid(
                f"duplicate miner_census uid(s) {dupes} — the close-block registration "
                "census must contain exactly one identity row per uid (schema v11)"
            )
        census_by_uid = {entry.uid: entry for entry in self.miner_census}
        for miner in self.miners:
            entry = census_by_uid.get(miner.uid)
            if entry is None:
                raise EpochLogInvalid(
                    f"economic miner uid {miner.uid} is absent from miner_census — every "
                    "eligible snapshot must be a member of the full close-block registration "
                    "census (schema v11)"
                )
            economic_identity = (miner.hotkey, miner.coldkey, miner.ip)
            census_identity = (entry.hotkey, entry.coldkey, entry.ip)
            if economic_identity != census_identity:
                raise EpochLogInvalid(
                    f"economic miner uid {miner.uid} identity {economic_identity!r} does not "
                    f"match miner_census identity {census_identity!r} (schema v11)"
                )
        # an internal review: every committed/log TRACK must be a MEMBER of the protocol
        # track set. `_require_committed_track_on_packets` only requires a NON-NULL committed
        # track, never that it be a real protocol track, and tokenomics `inference_shares`
        # SILENTLY drops a miner whose track is absent from `track_weights` — so an authority
        # could commit positive evidence consistently under an out-of-protocol track (e.g.
        # "unknown"), collapse the vector to `{burn_uid: 1.0}`, and audit CLEAN (every
        # self-consistent declaration agrees; nothing ever validated the track against the
        # protocol). Refuse any out-of-set `MinerSnapshot.track` / `AuditFileRef.committed_track`
        # at the shared construction / from_json boundary (the auditor adds a defense-in-depth
        # DISPUTED verdict for bytes that bypass the finalizer).
        for m in self.miners:
            if m.track not in PROTOCOL_TRACKS:
                raise EpochLogInvalid(
                    f"miner uid {m.uid} declares scoring track {m.track!r}, which is NOT a "
                    f"protocol track {sorted(PROTOCOL_TRACKS)} — an out-of-protocol track is "
                    "silently dropped from every tokenomics pool and substitutes a burn "
                    ""
                )
        all_refs = list(self.audit_manifest.baseline_bundles)
        for refs in self.audit_manifest.per_uid.values():
            all_refs.extend(refs)
        for refs in self.audit_manifest.competition_bundles.values():
            all_refs.extend(refs)
        for ref in all_refs:
            if (
                ref.committed_track is not None
                and ref.committed_track not in PROTOCOL_TRACKS
            ):
                raise EpochLogInvalid(
                    f"audit ref for item {ref.item_id!r} carries committed_track "
                    f"{ref.committed_track!r}, which is NOT a protocol track "
                    f"{sorted(PROTOCOL_TRACKS)} — an out-of-protocol committed track "
                    "substitutes a canonical burn while every declaration self-agrees "
                    ""
                )
        # v14: every current census identity has an explicit cursor, including ``null`` before
        # its first fold. Tombstones outside the census are intentionally retained.
        cursor_uids = set(self.audit_manifest.fold_cursors)
        census_uids = {entry.uid for entry in self.miner_census}
        missing_cursors = sorted(census_uids - cursor_uids)
        if missing_cursors:
            raise EpochLogInvalid(
                "current miner_census uid(s) are missing fold_cursors: "
                f"{missing_cursors} (schema v14 requires null before first fold)"
            )
        # Every CURRENT cycle must advance the cumulative replay cursor. Cross-epoch
        # exactness needs the predecessor and is enforced by the auditor/finalizer, but a log is
        # locally malformed if its own committed cycles are already above (or absent from) the
        # cursor it claims to have folded through.
        for uid, earning_input in self.audit_manifest.earning_inputs.items():
            if not earning_input.cycle_scores:
                continue
            current_max = max(c.ordering_key for c in earning_input.cycle_scores)
            cursor = self.audit_manifest.fold_cursors.get(uid)
            if uid not in self.audit_manifest.fold_cursors or cursor is None or cursor < current_max:
                raise EpochLogInvalid(
                    f"uid {uid} folds through ordering_key {current_max} this epoch but its "
                    f"cumulative fold cursor is {cursor!r} — every folded packet must be "
                    "covered by the anchored replay boundary (schema v14)"
                )
        # Every fixed pool must be represented explicitly before the chain's
        # mandatory normalization. A partial float vector with no sink can otherwise
        # quantize to a full u16 grid and silently donate the omitted allocation to
        # its surviving earners while remaining internally self-consistent.
        allocated_total = sum(float(weight) for weight in self.weight_shares.values())
        if abs(allocated_total - 1.0) > 1e-12:
            raise EpochLogInvalid(
                "weight_shares must explicitly allocate the complete fixed emission "
                "vector (including every canonical-sink residual); "
                f"allocated total is {allocated_total:.17g}"
            )
        expected_u16 = quantize_u16(self.weight_shares)
        if self.weight_u16 != expected_u16:
            raise EpochLogInvalid(
                "weight_u16 is not quantize_u16(weight_shares): the u16 vector must be "
                "the deterministic quantization of the float vector (convergence crux)"
            )
        expected_digest = weight_vector_digest(self.weight_u16)
        if self.weight_vector_digest != expected_digest:
            raise EpochLogInvalid(
                f"weight_vector_digest {self.weight_vector_digest} does not bind the u16 "
                f"vector (expected {expected_digest})"
            )
        if self.burn_uid is not None:
            if self.weight_shares.get(self.burn_uid, 0.0) <= 0.0:
                raise EpochLogInvalid(
                    f"burn_uid {self.burn_uid} is set but has no positive withheld share"
                )
            # an internal review: the RESERVED burn uid must NOT double as a census/evidence
            # identity. `_validate` already exempts burn_uid from manifest coverage below, and
            # the auditor excludes it from the snapshot/identity/dedup/track and earning folds —
            # so an untrusted log could seat the CANONICAL burn uid in `miners` carrying another miner's
            # evidence + a self-attested hotkey and publish `{burn_uid: 1.0}`: NO metagraph
            # binding or earning fold ever runs for the reserved uid ⇒ CLEAN. Refuse any overlap
            # between burn_uid and the census / earning / manifest evidence, so a tampered log is
            # caught at the shared construction / from_json boundary (the auditor adds a
            # defense-in-depth burn verdict for bytes that bypass the finalizer).
            if any(m.uid == self.burn_uid for m in self.miners):
                raise EpochLogInvalid(
                    f"burn_uid {self.burn_uid} is ALSO seated as a census miner — the reserved "
                    "withheld-pool burn uid must not double as an evidence/census identity (a "
                    "untrusted log could re-attribute another miner's evidence under the unaudited reserved "
                    "uid, an internal review)"
                )
            if (
                self.burn_uid in self.audit_manifest.earning_inputs
                or self.audit_manifest.refs_for(self.burn_uid)
            ):
                raise EpochLogInvalid(
                    f"burn_uid {self.burn_uid} carries earning/manifest evidence — the reserved "
                    "withheld-pool burn uid must not double as an evidence identity (the auditor "
                    "never folds the reserved uid, so its evidence would ride free, review "
                    "round-18 #2)"
                )
            if any(
                subject.uid == self.burn_uid
                for subject in (
                    self.audit_manifest.competition_input.subjects
                    if self.audit_manifest.competition_input is not None
                    else ()
                )
            ):
                raise EpochLogInvalid(
                    f"burn_uid {self.burn_uid} is also a competition contender — the "
                    "canonical withheld-pool sink cannot receive an auditable miner payout"
                )
        acc_by_uid = {m.uid: m.accumulate_score for m in self.miners}
        competition_backed_uids = {
            subject.uid
            for subject in (
                self.audit_manifest.competition_input.subjects
                if self.audit_manifest.competition_input is not None
                else ()
            )
            if (
                subject.role == "contender"
                and subject.uid is not None
                and acc_by_uid.get(subject.uid, 0.0) <= 0.0
            )
        }
        # A still-active reward window is chained from the predecessor and can pay its
        # committed podium even when this epoch carries no new inference fold for
        # those identities. Weight derivation/auditing independently verifies the
        # predecessor state and its time window; the schema only needs to avoid
        # misclassifying that legitimate non-inference payout as an unbacked EWMA.
        reward_hotkeys = set(self.reward_window_state.podium_hotkeys) if reward_active else set()
        competition_backed_uids.update(
            miner.uid
            for miner in self.miners
            if miner.hotkey in reward_hotkeys and miner.accumulate_score <= 0.0
        )
        for uid, weight in self.weight_shares.items():
            if weight <= 0.0 or uid == self.burn_uid:
                continue
            if uid in competition_backed_uids:
                continue
            if self.audit_manifest.refs_for(
                uid
            ) or self.audit_manifest.availability_for(uid):
                continue  # backed by current media refs and/or signed availability evidence
            # an internal review: a nonzero-weight uid with NO current manifest refs is allowed
            # ONLY as a pure CARRY-FORWARD — an IDLE prior earner still weighted by its carried
            # accumulator, with NO current earning input. Its weight is backed not by CURRENT
            # evidence but by the PRIOR epoch's fold, whose provenance the AUDITOR verifies by
            # chaining it (`_carry_forward_verdict`: same uid/hotkey, identical accumulator, back
            # to a committed genesis fold). The schema cannot see the prior epoch, so it defers
            # that cross-epoch check to the auditor — but it still refuses the two provable
            # local faults: (a) a uid that carries an earning INPUT (current cycles) yet has NO
            # refs backing those cycles, and (b) a nonzero weight whose snapshot accumulator is
            # not positive (weight must derive from a positive carried accumulator).
            if uid in self.audit_manifest.earning_inputs:
                raise EpochLogInvalid(
                    f"uid {uid} has nonzero weight {weight} and an earning input but NO "
                    "audit-manifest refs backing its cycles — a weight with no audit backing "
                    "is rejected (unverifiable by an auditor)"
                )
            if acc_by_uid.get(uid, 0.0) <= 0.0:
                raise EpochLogInvalid(
                    f"uid {uid} has nonzero weight {weight} but NO audit-manifest entry and a "
                    f"non-positive accumulate_score {acc_by_uid.get(uid)} — a weight not derived "
                    "from a positive carried accumulator has no audit backing (round-20 #2)"
                )
            # A pure carry-forward: nonzero weight, no refs, no earning input, positive carried
            # accumulator. The auditor chains its provenance to the prior epoch.
        return self

    @property
    def prior_epoch_id(self) -> int | None:
        """The runtime epoch id of the chained predecessor, gap-aware (v16).

        ``None`` at genesis. With no declared gap the predecessor is ``epoch_id-1``;
        with a gap it is the epoch just below the declared contiguous range. Every
        consumer that locates the prior log by id MUST use this instead of the
        literal ``epoch_id - 1`` (P1.5).
        """
        if self.prior_log_digest is None:
            return None
        if self.gap_epochs:
            return self.gap_epochs[0] - 1
        return self.epoch_id - 1

    # -- canonical serialization -------------------------------------------------------

    def _canonical_obj(self) -> dict[str, Any]:
        """The plain-python canonical shape; every collection in a deterministic order."""
        return {
            "schema_version": self.schema_version,
            "epoch_id": self.epoch_id,
            "close_block": self.close_block,
            "scorer_version": self.scorer_version,
            "created_at": self.created_at.isoformat(),
            "prior_log_digest": self.prior_log_digest,
            "gap_epochs": list(self.gap_epochs),
            "burn_uid": self.burn_uid,
            "competition_result": _competition_obj(self.competition_result),
            "reward_window_state": _reward_window_obj(self.reward_window_state),
            "miner_census": [
                _census_obj(entry)
                for entry in sorted(self.miner_census, key=lambda entry: entry.uid)
            ],
            "miners": [_miner_obj(m) for m in sorted(self.miners, key=lambda m: m.uid)],
            # dict keys canonicalize (sorted) via canonical_json_bytes; stringify uids.
            "weight_shares": {str(uid): w for uid, w in self.weight_shares.items()},
            "weight_u16": [
                [uid, self.weight_u16[uid]] for uid in sorted(self.weight_u16)
            ],
            "weight_vector_digest": self.weight_vector_digest,
            "audit_manifest": self.audit_manifest._canonical_obj(),
        }

    def to_json(self) -> bytes:
        """Canonical JSON bytes — byte-identical for equal epoch state on any machine."""
        return canonical_json_bytes(self._canonical_obj())

    def log_digest(self) -> str:
        """sha256 of `to_json()` — the digest anchored on chain / verified on fetch."""
        return sha256_hex(self.to_json())

    @classmethod
    def from_json(cls, data: bytes | str) -> "EpochLog":
        """Reconstruct an `EpochLog` from canonical bytes (re-runs every invariant)."""
        import json

        obj = json.loads(data)
        schema_version = int(obj["schema_version"])
        # Reject a legacy shape before indexing required fields. Otherwise a genuine
        # older payload fails with a raw KeyError rather than the domain-level
        # mixed-schema fence, which makes callers unable to distinguish incompatibility from a
        # corrupt object-store read.
        if schema_version != EPOCH_LOG_SCHEMA_VERSION:
            raise EpochLogInvalid(
                f"schema_version {schema_version} != EPOCH_LOG_SCHEMA_VERSION "
                f"{EPOCH_LOG_SCHEMA_VERSION} — a log on a foreign schema is refused"
            )
        manifest_obj = obj["audit_manifest"]
        canonical_top_fields = {
            "schema_version",
            "epoch_id",
            "close_block",
            "scorer_version",
            "created_at",
            "prior_log_digest",
            "gap_epochs",
            "burn_uid",
            "competition_result",
            "reward_window_state",
            "miner_census",
            "miners",
            "weight_shares",
            "weight_u16",
            "weight_vector_digest",
            "audit_manifest",
        }
        canonical_manifest_fields = {
            "per_uid",
            "baseline_bundles",
            "score_packet_merkle_root",
            "earning_inputs",
            "availability_inputs",
            "competition_input",
            "competition_bundles",
            "fold_cursors",
        }
        if (set(obj) - canonical_top_fields) or (
            set(manifest_obj) - canonical_manifest_fields
        ):
            raise EpochLogInvalid(
                f"schema-v{EPOCH_LOG_SCHEMA_VERSION} epoch log carries retired canonical "
                "field(s) or omits/adds "
                "a canonical field"
            )
        competition_input_obj = manifest_obj.get("competition_input")
        competition_result_obj = obj.get("competition_result")
        reward_state_obj = obj.get("reward_window_state")
        input_fields = {
            "competition_id",
            "track",
            "cycle",
            "completed_at",
            "applied_at",
            "manifest_digest",
            "commitment_root",
            "anchor_netuid",
            "anchor_payload_hex",
            "anchor_payload_digest",
            "anchor_block",
            "anchor_block_hash",
            "anchor_finalized_block",
            "baseline_version",
            "baseline_artifact_digest",
            "baseline_artifact_bytes",
            "baseline_execution_image_digest",
            "baseline_provenance_digest",
            "baseline_provenance_bytes",
            "aggregation_version",
            "items",
            "subjects",
        }
        subject_fields = {
            "subject_id",
            "role",
            "uid",
            "hotkey",
            "dedup_excluded",
            "submission_archive_digest",
            "submission_archive_bytes",
            "execution_image_digest",
            "repo_url",
            "commit_sha",
            "tree_sha",
            "packet_digests",
            "audit_bundle_digests",
        }
        result_fields = {
            "competition_id",
            "track",
            "cycle",
            "applied_at",
            "contenders",
            "baseline_score",
            "baseline_version",
            "baseline_artifact_digest",
        }
        reward_fields = {
            "kind",
            "starts_at",
            "ends_at",
            "podium_hotkeys",
            "winner_hotkey",
            "winner_uid",
            "winner_score",
            "winner_margin",
            "baseline_score",
            "baseline_version",
            "baseline_artifact_digest",
            "source_competition_id",
            "source_track",
            "source_cycle",
            "last_applied_cycle",
        }
        missing_current = [
            path
            for path, present in (
                ("miner_census", "miner_census" in obj),
                ("audit_manifest.fold_cursors", "fold_cursors" in manifest_obj),
                ("competition_result", "competition_result" in obj),
                ("reward_window_state", "reward_window_state" in obj),
                (
                    "audit_manifest.baseline_bundles",
                    "baseline_bundles" in manifest_obj,
                ),
                (
                    "audit_manifest.competition_input",
                    "competition_input" in manifest_obj,
                ),
                (
                    "audit_manifest.competition_bundles",
                    "competition_bundles" in manifest_obj,
                ),
                (
                    "audit_manifest.availability_inputs",
                    "availability_inputs" in manifest_obj,
                ),
                (
                    "audit_manifest.competition_input.v14_fields",
                    competition_input_obj is None
                    or input_fields.issubset(competition_input_obj),
                ),
                (
                    "audit_manifest.competition_input.subjects[].v14_fields",
                    competition_input_obj is None
                    or all(
                        subject_fields.issubset(subject)
                        for subject in competition_input_obj.get("subjects", [])
                    ),
                ),
                (
                    "competition_result.v14_fields",
                    competition_result_obj is None
                    or result_fields.issubset(competition_result_obj),
                ),
                (
                    "competition_result.contenders[].score",
                    competition_result_obj is None
                    or all(
                        {"hotkey", "uid", "score"}.issubset(contender)
                        for contender in competition_result_obj.get("contenders", [])
                    ),
                ),
                (
                    "reward_window_state.v14_fields",
                    isinstance(reward_state_obj, dict)
                    and reward_fields.issubset(reward_state_obj),
                ),
            )
            if not present
        ]
        if missing_current:
            raise EpochLogInvalid(
                f"schema-v{EPOCH_LOG_SCHEMA_VERSION} epoch log is missing required "
                "canonical field(s): " + ", ".join(missing_current)
            )
        competition_input = (
            None
            if competition_input_obj is None
            else CompetitionInput(
                competition_id=competition_input_obj["competition_id"],
                track=competition_input_obj["track"],
                cycle=int(competition_input_obj["cycle"]),
                completed_at=datetime.fromisoformat(
                    competition_input_obj["completed_at"]
                ),
                applied_at=datetime.fromisoformat(
                    competition_input_obj["applied_at"]
                ),
                manifest_digest=competition_input_obj["manifest_digest"],
                commitment_root=competition_input_obj["commitment_root"],
                anchor_netuid=int(competition_input_obj["anchor_netuid"]),
                anchor_payload_hex=competition_input_obj["anchor_payload_hex"],
                anchor_payload_digest=competition_input_obj[
                    "anchor_payload_digest"
                ],
                anchor_block=int(competition_input_obj["anchor_block"]),
                anchor_block_hash=competition_input_obj["anchor_block_hash"],
                anchor_finalized_block=int(
                    competition_input_obj["anchor_finalized_block"]
                ),
                baseline_version=int(competition_input_obj["baseline_version"]),
                baseline_artifact_digest=competition_input_obj[
                    "baseline_artifact_digest"
                ],
                baseline_artifact_bytes=int(
                    competition_input_obj["baseline_artifact_bytes"]
                ),
                baseline_execution_image_digest=competition_input_obj[
                    "baseline_execution_image_digest"
                ],
                baseline_provenance_digest=competition_input_obj[
                    "baseline_provenance_digest"
                ],
                baseline_provenance_bytes=int(
                    competition_input_obj["baseline_provenance_bytes"]
                ),
                aggregation_version=competition_input_obj["aggregation_version"],
                items=tuple(
                    CompetitionAuditItem(
                        challenge_id=item["challenge_id"],
                        item_id=item["item_id"],
                        threshold_commitment=item["threshold_commitment"],
                        item_index=(
                            None
                            if item.get("item_index") is None
                            else int(item["item_index"])
                        ),
                        input_sha256=item.get("input_sha256"),
                        reference_sha256=item.get("reference_sha256"),
                        upscale_factor=item.get("upscale_factor"),
                        target_width=item.get("target_width"),
                        target_height=item.get("target_height"),
                        item_commitment=item.get("item_commitment"),
                    )
                    for item in competition_input_obj.get("items", [])
                ),
                subjects=tuple(
                    CompetitionAuditSubject(
                        subject_id=subject["subject_id"],
                        role=subject["role"],
                        uid=None if subject.get("uid") is None else int(subject["uid"]),
                        hotkey=subject.get("hotkey"),
                        dedup_excluded=bool(subject["dedup_excluded"]),
                        submission_archive_digest=subject.get(
                            "submission_archive_digest"
                        ),
                        submission_archive_bytes=(
                            None
                            if subject.get("submission_archive_bytes") is None
                            else int(subject["submission_archive_bytes"])
                        ),
                        execution_image_digest=subject["execution_image_digest"],
                        repo_url=subject.get("repo_url"),
                        commit_sha=subject.get("commit_sha"),
                        tree_sha=subject.get("tree_sha"),
                        packet_digests=tuple(subject.get("packet_digests", [])),
                        audit_bundle_digests=tuple(
                            subject.get("audit_bundle_digests", [])
                        ),
                    )
                    for subject in competition_input_obj.get("subjects", [])
                ),
            )
        )
        manifest = AuditManifest(
            per_uid={
                int(uid): tuple(AuditFileRef(**r) for r in refs)
                for uid, refs in manifest_obj.get("per_uid", {}).items()
            },
            baseline_bundles=tuple(
                AuditFileRef(**r) for r in manifest_obj.get("baseline_bundles", [])
            ),
            score_packet_merkle_root=manifest_obj.get("score_packet_merkle_root"),
            earning_inputs={
                int(uid): EarningInput(
                    prior_accumulate_score=float(ei["prior_accumulate_score"]),
                    cycle_scores=tuple(
                        CycleScore(
                            packet_digest=c["packet_digest"],
                            ordering_key=int(c["ordering_key"]),
                            score=float(c["score"]),
                        )
                        for c in ei.get("cycle_scores", [])
                    ),
                )
                for uid, ei in manifest_obj.get("earning_inputs", {}).items()
            },
            availability_inputs=tuple(
                AvailabilityInput(
                    uid=int(evidence["uid"]),
                    hotkey=evidence["hotkey"],
                    challenge_id=evidence["challenge_id"],
                    item_id=evidence["item_id"],
                    track=evidence["track"],
                    ordering_key=int(evidence["ordering_key"]),
                    observation_json=evidence["observation_json"],
                    observation_digest=evidence["observation_digest"],
                )
                for evidence in manifest_obj.get("availability_inputs", [])
            ),
            competition_input=competition_input,
            competition_bundles={
                subject_id: tuple(AuditFileRef(**ref) for ref in refs)
                for subject_id, refs in manifest_obj.get(
                    "competition_bundles", {}
                ).items()
            },
            fold_cursors={
                int(uid): (None if ordering_key is None else int(ordering_key))
                for uid, ordering_key in manifest_obj.get("fold_cursors", {}).items()
            },
        )
        return cls(
            schema_version=schema_version,
            epoch_id=int(obj["epoch_id"]),
            close_block=int(obj["close_block"]),
            scorer_version=obj["scorer_version"],
            created_at=datetime.fromisoformat(obj["created_at"]),
            prior_log_digest=obj.get("prior_log_digest"),
            gap_epochs=tuple(int(g) for g in obj["gap_epochs"]),
            burn_uid=obj["burn_uid"],
            competition_result=_competition_from_obj(obj["competition_result"]),
            reward_window_state=_reward_window_from_obj(obj["reward_window_state"]),
            miner_census=tuple(
                MinerCensusEntry(**entry) for entry in obj["miner_census"]
            ),
            miners=tuple(_miner_from_obj(m) for m in obj["miners"]),
            weight_shares={
                int(uid): float(w) for uid, w in obj["weight_shares"].items()
            },
            weight_u16={int(uid): int(v) for uid, v in obj["weight_u16"]},
            weight_vector_digest=obj["weight_vector_digest"],
            audit_manifest=manifest,
        )


@dataclass(frozen=True, slots=True)
class EpochLogInputs:
    """Everything the finalizer needs to assemble an EpochLog, in one struct.

    A convenience for the authority producer (`vidaio.authority.finalizer`); the
    model above stays the shared, storable artifact.
    """

    epoch_id: int
    close_block: int
    snapshots: tuple[MinerSnapshot, ...]
    audit_manifest: AuditManifest
    burn_uid: int
    now: datetime
    miner_census: tuple[MinerCensusEntry, ...] | None = None
