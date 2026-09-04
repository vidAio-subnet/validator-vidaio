"""The AuditReport — the auditor's signed, deterministic verdict on an epoch.

An auditor produces exactly one of these per epoch it audits and POSTs it to the
Audit Results API (wave 7) through :mod:`vidaio.auditor.client`. It aggregates:

- one :class:`ItemVerdict` per SAMPLED audit item (PASS / FAIL / SKIP + a stable
  reason code straight from ``vidaio.audit.recompute`` — the auditor never invents
  a code, it forwards what ``verify_bundle`` returned);
- one :class:`WeightVerdict` — the cheap, media-free re-derivation of the epoch
  log's own weight vector from its stated inputs (catches a weight that does not
  follow from the scores even before any media is recomputed; also the channel the
  EARNING-STATE re-fold disputes travel on, see :mod:`vidaio.auditor.service`);
- a tuple of ``earning_verdicts`` — the per-uid re-derivation of the EARNING STATE
  (each nonzero-weight uid's ``accumulate_score`` re-folded from the audited packet
  scores + the chained prior-epoch carry-in), so a substituted earning state with
  honest packets can no longer pass (#1). Schema v14 also re-derives the packet-economic
  competition result and predecessor-derived reward window. These are
  reported for transparency; a FAIL among them is ALSO reflected into the
  ``weight_verdict`` so the epoch-level roll-up (and the Audit Results API, which
  reads only ``item_verdicts`` + ``weight_verdict``) sees the dispute;
- the ``overall`` status — a DERIVED, validated property (never a free field): it is
  always recomputed at construction from ``item_verdicts`` + ``weight_verdict`` via
  :func:`overall_status`, so a report can NEVER claim CLEAN while carrying a fault
  (#7). DISPUTED if any item/weight FAILs; INCONCLUSIVE if the sampled media items
  all SKIP (nothing was actually recomputed — not clean, needs attention, #8); else
  CLEAN. A SKIP never disputes ("unknown is never assume-fine"), but any selected
  media item or required weight/earning check that SKIPs holds the epoch
  INCONCLUSIVE. Launch production selects every media item, so CLEAN proves complete
  CPU recompute coverage rather than merely one successful sample.

Determinism: ``canonical_bytes()`` is canonical JSON over every field except the
signature, with all collections in a fixed order, so the SAME audit run yields
byte-identical report bytes on any machine — the bytes a hotkey signs and a digest
addresses. Signing is an injectable seam (:class:`ReportSigner`); the report stays
useful unsigned (``auditor_signature == ""``).
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex

#: The two media strata (competition/inference); an ItemVerdict on any OTHER source
#: (e.g. the synthetic "earning" rows) is a re-derivation record, not a recomputed
#: media item, and does not count toward the recompute coverage floor.
_MEDIA_SOURCES = ("competition", "inference")

#: Weight-vector re-derivation failure: the log's published weight vector is NOT
#: what ``build_weight_vector`` + ``quantize_u16`` produce from the log's own stated
#: inputs (a substituted weight, or one that does not follow from the scores). A NEW
#: code (the ``vidaio.audit.recompute`` vocabulary covers per-item packet checks; a
#: whole-log weight derivation is the auditor's own check). Stable string.
WEIGHT_DERIVATION_MISMATCH = "WEIGHT_DERIVATION_MISMATCH"

#: The content-addressed, on-chain-anchored epoch bytes violate a current-schema
#: model invariant before a normal item/economic audit can run. This is a provable
#: authority artifact defect, so the auditor emits a signed DISPUTED report instead
#: of crash-looping without central visibility.
EPOCH_LOG_INVALID = "EPOCH_LOG_INVALID"

#: The anchored epoch could not be decoded by the local strict audit model, but the
#: failure was not one of the known schema/model validation exceptions. Treat a
#: possible auditor implementation fault as unknown: signed INCONCLUSIVE, never a
#: false accusation and never a silent crash loop.
EPOCH_LOG_UNVERIFIED = "EPOCH_LOG_UNVERIFIED"

#: Earning-state re-derivation failure (#1): a nonzero-weight uid's stated
#: ``accumulate_score`` is NOT the EWMA fold of its audited packet score(s) + the
#: chained prior-epoch carry-in, or its cycle scores are not backed by audited
#: packets, or the reward window does not re-derive from the predecessor state + the
#: audited competition result. The authority cannot publish honest packets while
#: assigning a substituted earning state: the fold will not reproduce it. Stable
#: string; a distinct code from WEIGHT_DERIVATION_MISMATCH (that check re-derives
#: the WEIGHTS from the stated inputs; this one re-derives the stated INPUTS
#: themselves from the audited evidence).
EARNING_STATE_MISMATCH = "EARNING_STATE_MISMATCH"

#: A CROSS-EPOCH packet REPLAY. The per-epoch earning fold proves each
#: ``CycleScore`` is backed by a committed packet in the CURRENT manifest and that the numeric
#: carry-in chains to the prior epoch — but NOT that the packet was not ALREADY folded by a
#: predecessor epoch. An untrusted authority could re-use an EARLIER genuine packet (its
#: commitment still valid), refold it over the carried accumulator, and derive a self-consistent
#: (and INFLATED) accumulate_score: backing, commitment, media recompute, carry-in linkage, and
#: weight derivation ALL pass ⇒ the same inference is silently awarded again. The committed
#: dispatch ``ordering_key`` is MONOTONIC per uid, so this epoch's folded cycles must every one
#: exceed the MAXIMUM ordering_key the uid already folded through the prior epoch. A cycle whose
#: ordering_key is <= the prior watermark is a re-fold ⇒ FAIL (DISPUTED). When the prior
#: watermark is UNAVAILABLE (the immediately-prior epoch carried the uid forward with no earning
#: input, so its max folded key is not in that log — the same gap as the producer-side watermark
#: stall), non-replay cannot be PROVEN for a uid that already had a positive prior accumulator, so
#: it is fail-closed to INCONCLUSIVE (HOLD), never a CLEAN. A genuinely NEW uid (no positive prior
#: accumulator) has nothing to replay and folds freely from genesis.
EARNING_PACKET_REPLAY = "EARNING_PACKET_REPLAY"

#: The anchored total fold-cursor map is not the exact monotonic continuation of the
#: predecessor map (schema v14): an observed identity/cursor is missing, a uid boundary was
#: dropped/regressed, a boundary advanced without a committed cycle, or a current cycle was
#: not reflected in it. The full map (including ``None`` tombstones) is carried across idle,
#: exclusion, deregistration, burn, and uid/hotkey reuse, so identity churn cannot erase history.
FOLD_CURSOR_MISMATCH = "FOLD_CURSOR_MISMATCH"

#: The audit manifest carries an ambiguous DUPLICATE audit identity: two items share
#: the (source, challenge_id, item_id) the sampler keys on. The
#: sample would be un-attributable and a duplicate identity is itself a tamper signal,
#: so the whole epoch is DISPUTED before any sampling. Reflected through the weight
#: verdict (the dispute channel), so the roll-up and the Audit Results API both see it.
DUPLICATE_AUDIT_IDENTITY = "DUPLICATE_AUDIT_IDENTITY"

#: The audit manifest is STRUCTURALLY malformed: an item does not
#: pair an AUDIT_BUNDLE with a SCORE_PACKET ref (``ManifestIncomplete``), or a similar
#: structural defect prevents even sampling it. A defect in the AUTHORITY's own manifest
#: is a provable fault, so the epoch is DISPUTED with a SIGNED report rather than left to
#: retry forever as an uncaught exception (which would block the auditor cursor). Like
#: DUPLICATE_AUDIT_IDENTITY it is reflected through the weight verdict (the dispute
#: channel the roll-up and the Audit Results API both read).
MANIFEST_INCOMPLETE = "MANIFEST_INCOMPLETE"

#: A STILL-REGISTERED, prior-POSITIVE miner's earning state was SILENTLY RESET to 0.0 /
#: the exclusion sentinel this epoch WITHOUT an evidenced reason. The
#: earning re-fold + census only look at the CURRENT positive accumulators / current
#: evidence, so an authority could zero (or exclude, or drop from the census) a miner that
#: (a) had a positive accumulator in the prior epoch's chained log, (b) is STILL present in
#: the close-block metagraph under the SAME (uid, hotkey), yet (c) carries NO current
#: EarningInput justifying the drop — erasing its accrued earnings while another miner's new
#: evidence re-derives the vector CLEAN. EWMA DECAYS but never zeroes a positive accumulator
#: in one epoch, and a no-evidence miner must CARRY its value forward unchanged, so a
#: still-registered positive accumulator that becomes 0.0 / excluded / absent with no
#: committed exclusion is a censored/erased earning state ⇒ DISPUTED. A genuine
#: deregistration (absent from the metagraph) or an evidenced exclusion is legitimate and is
#: NOT flagged. DISTINCT from the sanctioned all-carry `items=[]` burn (whose miners carry
#: their positive accumulators FORWARD, staying eligible — they are never reset).
EARNING_STATE_RESET = "EARNING_STATE_RESET"

#: A committed / log TRACK is NOT a member of the protocol track set.
#: `AuditFileRef` requires a non-null committed track but never that it be a real protocol
#: track, commitment parsing accepts any non-empty string, and tokenomics SILENTLY drops a
#: miner whose track is absent from `track_weights` from every pool — so an authority could
#: commit positive evidence consistently under an out-of-protocol track (e.g. "unknown"),
#: collapse the vector to `{burn_uid: 1.0}`, and audit CLEAN (every self-consistent
#: declaration agrees; the existing track binding only compares declarations to each other).
#: The auditor validates every `MinerSnapshot.track` / `AuditFileRef.committed_track` against
#: the protocol set (the keys of `tokenomics.track_weights`) — an out-of-set track is a
#: provable substituted-burn fault ⇒ DISPUTED. Provable from the log's own bytes (no metagraph),
#: defense-in-depth for bytes that bypassed the finalizer (`EpochLog._validate` refuses them).
UNKNOWN_TRACK = "UNKNOWN_TRACK"

#: A nonzero-weight uid's earning state could not be verified (the log carried no
#: earning derivation for it, or the carry-in could not be chained without the
#: prior epoch's log). NOT a provable fault — recorded as an earning SKIP, surfaced
#: to the dashboard, never washed into a PASS. Production finalizers always emit the
#: earning inputs, so this signals a legacy/partial log or a missing prior epoch.
EARNING_STATE_UNVERIFIED = "EARNING_STATE_UNVERIFIED"

#: Close-block metagraph binding failures for the SNAPSHOT-derivable weight inputs
#:. The auditor reads the metagraph AS OF close_block ITSELF and
#: binds each nonzero-weight uid's identity/dedup/track to it, never trusting the
#: authority's self-attested MinerSnapshot fields:
#: - a relabelled uid->hotkey/coldkey/ip (or a uid absent from the metagraph) is a
#:   conclusive IDENTITY_MISMATCH (reused from vidaio.audit.recompute) ⇒ DISPUTED;
#: - a mis-declared IP/coldkey DEDUP outcome (`excluded` flag) — a real collision the
#:   authority did NOT exclude, or a distinct miner it WRONGLY excluded — is
#:   METAGRAPH_DEDUP_MISMATCH ⇒ DISPUTED;
#: - a scoring `track` inconsistent with the committed-challenge track of the uid's
#:   earning evidence is METAGRAPH_TRACK_MISMATCH ⇒ DISPUTED.
METAGRAPH_DEDUP_MISMATCH = "METAGRAPH_DEDUP_MISMATCH"
METAGRAPH_TRACK_MISMATCH = "METAGRAPH_TRACK_MISMATCH"

#: The close-block metagraph binding could not be COMPLETED for a nonzero-weight uid —
#: the metagraph read failed / is unavailable, or the committed track is unresolvable
#:. NOT a provable fault: recorded as a SKIP so the epoch rolls up
#: INCONCLUSIVE (HOLD), never a PASS on the authority's word (fail-closed). Distinct
#: from EARNING_STATE_UNVERIFIED (that is the fold/carry-in; this is the snapshot bind).
SNAPSHOT_UNVERIFIED = "SNAPSHOT_UNVERIFIED"

# (WINDOW_EVIDENCE_MISMATCH / WINDOW_UNVERIFIED REMOVED with the retention multiplier for
# v1 — retention removed — owner decision; an internal review — the
# auditor no longer verifies (or needs) any committed windowed evidence.)

# Schema v14 carries explicit competition/reward-window verdicts. These stable codes distinguish
# an evidence matrix that cannot be verified from one that verifies but derives a different
# packet-economic result or reward window.
COMPETITION_MISMATCH = "COMPETITION_MISMATCH"
COMPETITION_UNVERIFIED = "COMPETITION_UNVERIFIED"
REWARD_WINDOW_MISMATCH = "REWARD_WINDOW_MISMATCH"

#: The log's ``created_at`` does not agree with the epoch's CLOSE BLOCK time.
#: The auditor reads the close_block's timestamp from the chain ITSELF and requires ``created_at``
#: to match within a tolerance — a disagreement is a provable mismatch ⇒ DISPUTED. It also
#: binds the reward-window time base to chain time rather than an authority-selected timestamp.
CREATED_AT_MISMATCH = "CREATED_AT_MISMATCH"

#: The epoch's CLOSE BLOCK time could not be READ: the chain adapter exposes
#: no block clock, or the close_block's time is unavailable. NOT a provable fault: recorded as a
#: SKIP so the epoch rolls up INCONCLUSIVE (HOLD), never a PASS on an unverifiable ``created_at``.
CREATED_AT_UNVERIFIED = "CREATED_AT_UNVERIFIED"

#: The log's MINER CENSUS contradicts its own committed manifest evidence.
#: The snapshot / earning / competition re-derivations above are all scoped to the POSITIVE-weight
#: set, so when that set is empty or only ``burn_uid`` they return early WITHOUT ever comparing the
#: log's ``miners`` census to the committed evidence — letting an authority STORE earning/competition
#: evidence for real miners and then OMIT (or zero-out) every one of them from the census, publishing
#: ``miners=[], {burn_uid:1.0}`` and receiving CLEAN (a censored empty-burn epoch indistinguishable
#: from a genuinely-empty one). The auditor cross-checks the census against the manifest ITSELF (no
#: metagraph needed — provable from the log's own bytes): any uid the manifest carries committed
#: EARNING evidence (a SCORE_PACKET ref / EarningInput) or COMPETITION evidence (a contender) for,
#: but which is OMITTED from ``log.miners`` — or present but ZEROED without an evidenced exclusion —
#: is a censored miner ⇒ FAIL ⇒ DISPUTED. A GENUINELY empty epoch (no committed evidence at all →
#: burn) carries no evidenced uids, so no census verdict fires and it stays CLEAN (the legitimate
#: empty-epoch burn, the project design record #11). RESIDUAL/milestone: this proves the EVIDENCE-vs-census
#: cross-check (the authority stored evidence then censored the miner); it does NOT prove
#: "registered-but-never-dispatched" censorship (a miner that SHOULD have been challenged but was
#: not), which needs challenge-dispatch-record binding — a documented future milestone.
CENSUS_MISMATCH = "CENSUS_MISMATCH"

#: The log's declared empty-epoch BURN RECIPIENT is not the CANONICAL burn uid the auditor
#: resolves INDEPENDENTLY of the untrusted log. ``EpochLog._validate``
#: only requires ``burn_uid`` to be the SOLE positive-weight uid — it lets the UNTRUSTED
#: authority CHOOSE the recipient, so an authority could anchor an empty log burning 100%
#: to a registered beneficiary IT controls (``burn_uid=<beneficiary>``, ``{beneficiary:1.0}``)
#: and audit CLEAN, diverting the whole epoch's emission to itself. The burn recipient must
#: be CANONICAL: each isolated auditor resolves it independently from config —
#: ``AuditorConfig.burn_uid``, the SAME value the Scoring Authority is configured with
#: (``authority.burn_uid``) — never from the log. A ``log.burn_uid`` that does not equal the
#: canonical value is a provable fault ⇒ DISPUTED. RESIDUAL/Rung-2: config is the source of
#: truth today; a chain/registry-derived ``get_burn_uid()`` (the subnet-owner uid) is the
#: production hardening.
BURN_UID_MISMATCH = "BURN_UID_MISMATCH"

#: The canonical subnet-owner uid could not be resolved from the auditor's
#: independent chain connection. This is UNKNOWN, not evidence that the declared
#: recipient is correct: the epoch remains INCONCLUSIVE/HOLD and uid 0 is never
#: substituted. Report/test adapters may use an explicit overlay fallback.
BURN_UID_UNVERIFIED = "BURN_UID_UNVERIFIED"

#: The epoch BREAKS the predecessor chain — a LOG-LEVEL fault that fires regardless of
#: whether any uid is audited. Every per-uid carry-in / carry-forward
#: chain check (``_carry_in_check`` / ``_carry_forward_verdict``) only runs for a uid that
#: is actually SELECTED for earning audit (positive weight, a current ``EarningInput``, or a
#: positive stated accumulator). An EMPTY canonical burn log (``miners=[]``,
#: ``{burn_uid:1.0}``) selects NO earning uid, so the chain was never enforced at all: a
#: NON-genesis authority could OMIT ``prior_log_digest`` (or point it at an unavailable
#: predecessor), publish the empty burn vector, and audit CLEAN — silently RESETTING all
#: prior positive earning state (an item-scoped check sees no nonzero carry-in and could
#: otherwise report the epoch clean). This is the log-level guard: a non-genesis epoch
#: that OMITS ``prior_log_digest`` (a chain reset) or supplies one that does NOT match the
#: chained prior is a provable fault ⇒ DISPUTED. Provable from the log's bytes + the loop's
#: independent genesis determination; the true genesis (``is_genesis``, ``prior_log_digest``
#: None) is exempt.
PREDECESSOR_CHAIN_BROKEN = "PREDECESSOR_CHAIN_BROKEN"

#: The epoch REFERENCES a predecessor (``prior_log_digest`` is not None) that could not be
#: loaded (pruned / unreadable), so the predecessor chain is UNVERIFIABLE at the log level
#:. Like a referenced-but-unavailable carry-in (round-8 #6), removing the
#: prior object is exactly how an authority would reset/censor accumulated earnings, so this
#: is fail-closed to INCONCLUSIVE (HOLD) — never a CLEAN that would advance the cursor past
#: an unverifiable reset. Fires for an empty/burn log too (no earning uid need be selected).
PREDECESSOR_UNVERIFIED = "PREDECESSOR_UNVERIFIED"


class ItemVerdictKind(StrEnum):
    """A per-item audit outcome.

    PASS — ``verify_bundle`` proved the item honest.
    FAIL — a provable fault (SCORE_MISMATCH, MERKLE_EXCLUSION, IDENTITY_MISMATCH,
           REVEAL_INVALID, ARTIFACT_CORRUPT, …); conclusive regardless of sampling.
    SKIP — the auditor could not verify (unreachable artifact or unavailable required
           CPU recompute backend). Never a
           PASS-in-disguise (the project design record §6).
    """

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class AuditStatus(StrEnum):
    """The epoch-level roll-up an auditor asserts.

    CLEAN — every selected item and every required weight/earning check passed.
    DISPUTED — a provable fault (a FAIL among items or the weight verdict, which
            also carries a reflected EARNING_STATE_MISMATCH). Conclusive.
    INCONCLUSIVE — at least one selected media item or required weight/earning check
            SKIPped. NOT clean — it surfaces as needs-attention on the dashboard.
            Distinct from DISPUTED: no fault was proven, but complete verification
            was not achieved.
    """

    CLEAN = "CLEAN"
    DISPUTED = "DISPUTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class AuditMode(StrEnum):
    """The independent audit path that produced a report.

    ``BEACON`` is the historical/default validator audit loop. ``OWN_AUDIT`` is
    the validator's full post-submission observer, reported through the same central
    API. Keeping the mode in the signed report lets both paths report the same
    hotkey+epoch without one being mistaken for a divergent resubmission.
    """

    BEACON = "beacon"
    OWN_AUDIT = "own_audit"


class ItemVerdict(BaseModel):
    """One sampled item's verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str  # "competition" | "inference"
    challenge_id: str
    item_id: str
    miner_hotkey: str | None = None
    uid: int | None = None
    bundle_digest: str
    packet_digest: str
    verdict: ItemVerdictKind
    #: "" when PASS/SKIP-clean; else the recompute reason code.
    code: str = ""
    detail: str = ""

    def _sort_key(self) -> tuple[str, str, str, str]:
        return (self.source, self.challenge_id, self.item_id, self.bundle_digest)

    def _canonical_obj(self) -> dict[str, object]:
        return {
            "source": self.source,
            "challenge_id": self.challenge_id,
            "item_id": self.item_id,
            "miner_hotkey": self.miner_hotkey,
            "uid": self.uid,
            "bundle_digest": self.bundle_digest,
            "packet_digest": self.packet_digest,
            "verdict": self.verdict.value,
            "code": self.code,
            "detail": self.detail,
        }


class WeightVerdict(BaseModel):
    """The media-free re-derivation of the epoch log's weight vector.

    The auditor re-runs ``build_weight_vector`` + ``quantize_u16`` over the log's own
    ``miners`` and compares the resulting u16 weight-vector digest against the one the
    log published (and, transitively, the one anchored on chain). A mismatch is
    WEIGHT_DERIVATION_MISMATCH. Schema v14 includes the re-derived competition result
    and reward window in that deterministic vector input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    recomputed_weight_vector_digest: str
    published_weight_vector_digest: str
    verdict: ItemVerdictKind
    code: str = ""
    detail: str = ""

    def _canonical_obj(self) -> dict[str, object]:
        return {
            "recomputed_weight_vector_digest": self.recomputed_weight_vector_digest,
            "published_weight_vector_digest": self.published_weight_vector_digest,
            "verdict": self.verdict.value,
            "code": self.code,
            "detail": self.detail,
        }


class AuditReport(BaseModel):
    """A hotkey-attributable, deterministic verdict on one epoch (frozen)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auditor_hotkey: str
    epoch_id: int
    #: Report origin. Omitted from the canonical signed payload for the historical
    #: BEACON default so existing beacon signatures and report digests remain valid.
    audit_mode: AuditMode = AuditMode.BEACON
    #: The ``EpochLog.log_digest()`` this report audits — binds the verdict to the
    #: exact bytes (== on-chain anchor), so it cannot be re-pointed at another log.
    snapshot_digest: str
    pipeline_version: str  # the log's scorer_identity
    sampled_at: datetime
    #: Coverage summary (the project design record §3.2 sample_policy shape).
    competition_n: int = 0
    inference_n: int = 0
    item_verdicts: tuple[ItemVerdict, ...] = ()
    #: Per-uid EARNING-STATE re-derivation verdicts (source="earning"): the fold of
    #: audited packet scores + chained carry-in vs the stated accumulate_score, plus
    #: competition-result/reward-window re-derivation in schema v14.
    #: Transparency records; a FAIL here is also
    #: reflected into ``weight_verdict`` so the roll-up (and the Audit Results API)
    #: disputes on it (#1).
    earning_verdicts: tuple[ItemVerdict, ...] = ()
    weight_verdict: WeightVerdict
    #: DERIVED at construction from ``item_verdicts`` + ``weight_verdict`` (see the
    #: validator below) — a caller-supplied value is always overwritten with the
    #: consistent one, so a CLEAN report can never carry a fault (#7).
    overall: AuditStatus = AuditStatus.CLEAN
    #: hex signature over ``canonical_bytes()``; "" when unsigned (a signer is an
    #: injectable seam — the report is still a valid, verifiable artifact unsigned).
    auditor_signature: str = ""

    @model_validator(mode="after")
    def _derive_overall(self) -> "AuditReport":
        """Force ``overall`` to the value derived from the verdicts it carries.

        The roll-up is NEVER a free field: whatever was passed in, construction
        overwrites it with :func:`overall_status` over this report's own verdicts.
        So the auditor (and anyone who deserializes a POSTed report — the Audit
        Results API constructs the same model) can never emit or accept a CLEAN
        report that contains a FAIL, or wash an all-SKIP sample to CLEAN (#7, #8).
        """
        derived = overall_status(
            self.item_verdicts, self.weight_verdict, self.earning_verdicts
        )
        if self.overall is not derived:
            object.__setattr__(self, "overall", derived)
        return self

    def _canonical_obj(self, *, include_signature: bool) -> dict[str, object]:
        obj: dict[str, object] = {
            "auditor_hotkey": self.auditor_hotkey,
            "epoch_id": self.epoch_id,
            "snapshot_digest": self.snapshot_digest,
            "pipeline_version": self.pipeline_version,
            "sampled_at": self.sampled_at.isoformat(),
            "competition_n": self.competition_n,
            "inference_n": self.inference_n,
            "item_verdicts": [
                v._canonical_obj()
                for v in sorted(self.item_verdicts, key=lambda x: x._sort_key())
            ],
            "earning_verdicts": [
                v._canonical_obj()
                for v in sorted(self.earning_verdicts, key=lambda x: x._sort_key())
            ],
            "weight_verdict": self.weight_verdict._canonical_obj(),
            "overall": self.overall.value,
        }
        # Backward compatibility is deliberate: v0.3.x beacon reports did not
        # carry an audit_mode field. Their canonical bytes, signatures, and report
        # digests must continue to verify after parsing as the default BEACON mode.
        if self.audit_mode is not AuditMode.BEACON:
            obj["audit_mode"] = self.audit_mode.value
        if include_signature:
            obj["auditor_signature"] = self.auditor_signature
        return obj

    def canonical_bytes(self) -> bytes:
        """Canonical JSON of the report WITHOUT the signature — the signable bytes."""
        return canonical_json_bytes(self._canonical_obj(include_signature=False))

    def report_digest(self) -> str:
        """sha256 of the unsigned canonical bytes — a stable id for the report."""
        return sha256_hex(self.canonical_bytes())

    def signed(self, signer: "ReportSigner") -> "AuditReport":
        """Return a copy carrying ``signer``'s signature over ``canonical_bytes()``."""
        return self.model_copy(
            update={"auditor_signature": signer.sign(self.canonical_bytes())}
        )

    def failures(self) -> tuple[ItemVerdict, ...]:
        return tuple(v for v in self.item_verdicts if v.verdict is ItemVerdictKind.FAIL)


def overall_status(
    item_verdicts: tuple[ItemVerdict, ...],
    weight_verdict: WeightVerdict,
    earning_verdicts: tuple[ItemVerdict, ...] = (),
) -> AuditStatus:
    """The epoch roll-up: DISPUTED > INCONCLUSIVE > CLEAN, derived from the verdicts.

    - DISPUTED if the weight verdict FAILs (which also carries any reflected
      EARNING_STATE_MISMATCH), ANY item verdict FAILs, or ANY earning verdict FAILs —
      a provable fault. A signed report can NEVER derive CLEAN while carrying an
      earning FAIL.
    - else INCONCLUSIVE if the weight verdict, any EARNING verdict, or any selected
      media verdict is SKIP/unverifiable. Production uses the uncapped ``all_items``
      policy, so accepting PASS+SKIP would let one GPU-produced score influence
      emissions without CPU reproduction. Earning rows remain a separate channel,
      but an unverified earning state is equally fail-closed.
    - else CLEAN.

    A SKIP never disputes (unknown is never a provable fault), but no required or
    selected SKIP is ever washed to CLEAN. ``earning_verdicts`` is an optional third
    channel; passing it keeps a re-derivation (e.g. the Audit Results API) consistent
    with the report's own ``overall``.
    """
    if weight_verdict.verdict is ItemVerdictKind.FAIL:
        return AuditStatus.DISPUTED
    if any(v.verdict is ItemVerdictKind.FAIL for v in item_verdicts):
        return AuditStatus.DISPUTED
    if any(v.verdict is ItemVerdictKind.FAIL for v in earning_verdicts):
        return AuditStatus.DISPUTED
    if weight_verdict.verdict is ItemVerdictKind.SKIP:
        return AuditStatus.INCONCLUSIVE
    if any(v.verdict is ItemVerdictKind.SKIP for v in earning_verdicts):
        return AuditStatus.INCONCLUSIVE
    # an internal review: the coverage floor must NOT depend on a spoofable `source`
    # label. `item_verdicts` are ONLY the sampled media-recompute verdicts (earning
    # re-derivations arrive via `earning_verdicts` above and are never mixed in here), so
    # ANY selected item that SKIPs means complete CPU reproduction was not achieved
    # → INCONCLUSIVE, regardless of the label it carries. Filtering by
    # `_MEDIA_SOURCES` here would let an off-list label bypass the fail-closed rule;
    # the source is constrained at the model level too, so this is defense in depth.
    # `_MEDIA_SOURCES` is retained as the documented media-strata set.
    if any(v.verdict is ItemVerdictKind.SKIP for v in item_verdicts):
        return AuditStatus.INCONCLUSIVE
    return AuditStatus.CLEAN


# --- signing seam --------------------------------------------------------------------


class ReportSigner(Protocol):
    """Signs an auditor's canonical report bytes; returns a hex signature.

    Production wires the validator's hotkey keypair here; tests use a deterministic
    double. Kept a Protocol so the auditor never hard-depends on a crypto backend.
    """

    def sign(self, payload: bytes) -> str: ...


class Sha256Signer:
    """Deterministic signer double: ``sha256(secret || payload)``.

    NOT a real signature scheme — a stand-in for the hotkey signer so the report's
    sign/verify seam is exercised deterministically in tests. Two runs over the same
    report bytes produce the same signature; a different secret produces a different
    one.
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret.encode("utf-8")

    def sign(self, payload: bytes) -> str:
        return hashlib.sha256(self._secret + b"\x00" + payload).hexdigest()

    def verify(self, payload: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(payload), signature)
