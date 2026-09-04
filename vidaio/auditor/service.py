"""The Auditor — the decentralized honesty check on the central Scoring Authority.

``Auditor.audit_epoch`` reads an epoch log's audit manifest, deterministically
SAMPLES a subset of its items, independently RECOMPUTES each over the real scoring
engine (``verify_bundle`` + a :class:`~vidaio.auditor.recomputer.RealScoreRecomputer`),
RE-DERIVES the log's own weight vector from its stated inputs, and aggregates the
result into a signed, deterministic :class:`~vidaio.auditor.report.AuditReport`
(the project design record §1c, build wave 6).

What it proves vs samples (the project design record §5): a passing
``verify_bundle`` PROVES that item honest; a typed FAIL PROVES it invalid; the
weight re-derivation PROVES the published vector follows (or does not) from the
inputs — all before any wall-clock trust. It only SAMPLES the item population, so
whole-epoch honesty is probabilistic (raise the sample to raise confidence) — but
any single provable FAIL flips the epoch to DISPUTED.

Two seams the manifest cannot itself supply:
- BUNDLES — the manifest names a ``bundle_digest`` but an ``AuditBundle`` is not a
  standalone blob (vidaio/epoch/log.py). The auditor resolves it through an
  injected :class:`BundleSource`; the resolved bundle is digest-checked against the
  manifest ref inside ``verify_bundle`` (DIGEST_MISMATCH otherwise).
- REVEAL — the deep commit-reveal verifier is the challenge module's; it is injected
  here (``reveal_verifier``), exactly as ``verify_bundle`` expects.

The v2 manifest carries the committed ``score_packet_merkle_root`` and a per-item
merkle INCLUSION PROOF on each SCORE_PACKET ``AuditFileRef`` (built by the finalizer),
so the auditor now runs STRICT merkle inclusion by default (``strict=True``): a sampled
item whose bundle is not provably in the committed root fails ``MERKLE_EXCLUSION`` and
disputes the epoch, alongside the substitution-catching recompute/identity/digest checks.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import replace
from datetime import datetime
from typing import Callable, Mapping, Protocol

from pydantic import ValidationError

from vidaio.audit.bundle import AuditBundle
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.commitments import (
    load_competition_commitment,
    pin_git_sha,
    reward_parameter_digest,
)
from vidaio.audit.recompute import (
    ARTIFACT_MISSING,
    IDENTITY_MISMATCH,
    CompetitionAuditContext,
    verify_bundle,
)
from vidaio.challenge.commitment import CHALLENGE_ANCHOR_DOMAIN, ChallengeCommitment
from vidaio.audit.store import (
    ArtifactKind,
    ArtifactRef,
    AuditStore,
    IntegrityError,
)
from vidaio.epoch.log import (
    AuditFileKind,
    AuditFileRef,
    AvailabilityInput,
    EarningInput,
    EpochLog,
    weight_vector_digest,
)
from vidaio.tokenomics.ewma import accumulate, is_excluded
from vidaio.tokenomics.quantize import quantize_u16
from vidaio.tokenomics.rank_curve import dedup_excluded
from vidaio.tokenomics.weights import build_weight_vector
from vidaio.tokenomics.breakthrough import resolve_reward_window
from vidaio.tokenomics.state import CompetitionResult, EmissionState, RewardWindowState
from vidaio.validator.availability import (
    AvailabilityObservation,
    verify_availability_observation,
)
from vidaio.competition.economic_result import (
    CompetitionDedupCandidate,
    CompetitionEconomicResultError,
    competition_dedup_losers,
    derive_competition_economics,
)
from vidaio.competition.anchor_evidence import (
    CompetitionAnchorMismatch,
    CompetitionAnchorUnavailable,
    verify_competition_anchor_on_chain,
)
from vidaio.competition.item_commitment import evaluation_item_commitment
from vidaio.competition.manifest import CompetitionManifest

from vidaio.auditor.config import AuditorConfig, SamplePolicy
from vidaio.auditor.chronology import (
    ChronologyKind,
    verify_challenge_chronology,
)
from vidaio.auditor.client import AuditResultsClient, SubmitAck
from vidaio.auditor.recomputer import RecomputeUnavailable
from vidaio.auditor.report import (
    BURN_UID_MISMATCH,
    BURN_UID_UNVERIFIED,
    CENSUS_MISMATCH,
    COMPETITION_MISMATCH,
    COMPETITION_UNVERIFIED,
    CREATED_AT_MISMATCH,
    CREATED_AT_UNVERIFIED,
    DUPLICATE_AUDIT_IDENTITY,
    EPOCH_LOG_INVALID,
    EPOCH_LOG_UNVERIFIED,
    MANIFEST_INCOMPLETE,
    EARNING_PACKET_REPLAY,
    EARNING_STATE_MISMATCH,
    EARNING_STATE_RESET,
    EARNING_STATE_UNVERIFIED,
    FOLD_CURSOR_MISMATCH,
    METAGRAPH_DEDUP_MISMATCH,
    METAGRAPH_TRACK_MISMATCH,
    PREDECESSOR_CHAIN_BROKEN,
    PREDECESSOR_UNVERIFIED,
    REWARD_WINDOW_MISMATCH,
    SNAPSHOT_UNVERIFIED,
    UNKNOWN_TRACK,
    WEIGHT_DERIVATION_MISMATCH,
    AuditReport,
    ItemVerdict,
    ItemVerdictKind,
    ReportSigner,
    WeightVerdict,
)
from vidaio.chain.adapter import resolve_burn_uid
from vidaio.auditor.sampling import (
    NO_BEACON,
    AuditItem,
    DuplicateAuditIdentity,
    ManifestIncomplete,
    sample_items,
)

#: Earning re-fold tolerance (float EWMA rounding across machines).
_FOLD_TOL = 1e-9
#: The synthetic source label for earning-state re-derivation verdicts (NOT a media
#: stratum, so it never counts toward the recompute coverage floor — see report.py).
_EARNING_SOURCE = "earning"
#: The synthetic source label for the close-block metagraph SNAPSHOT-binding verdicts
#:. Also NOT a media stratum: it rides the
#: earning-verdicts channel (FAIL ⇒ DISPUTED, SKIP ⇒ INCONCLUSIVE) but never counts toward
#: the media coverage floor.
_SNAPSHOT_SOURCE = "snapshot"
# The committed WINDOWED-evidence verdicts remain removed with the retired retention
# multiplier. Competition economics use their own schema-v14 synthetic source below.
#: The synthetic source label for the weight-INPUT time-base verdict (created_at vs the
#: close-block time). Rides the SAME earning-verdicts channel (FAIL ⇒
#: DISPUTED, SKIP ⇒ INCONCLUSIVE); NOT a media stratum, so it never counts toward the media
#: coverage floor. Competition completion/cycle chronology is checked by the dedicated
#: schema-v14 competition verdicts.
_TIMEBASE_SOURCE = "weight-timebase"
#: Tolerance (seconds) for the created_at vs close-block-time compare. Small
#: enough to catch a reward-window-extending backdate, generous for clock/rounding.
_CREATED_AT_TOL_SECONDS = 600.0
_MAX_AUDIT_BUNDLE_BYTES = 1 * 1024 * 1024
_MAX_AUDIT_METADATA_BYTES = 16 * 1024 * 1024
#: The synthetic source label for the MINER-CENSUS-vs-committed-evidence cross-check (review
#: round-9 #1). Rides the SAME earning-verdicts channel (FAIL ⇒ DISPUTED); NOT a media stratum, so
#: it never counts toward the media coverage floor. Unlike the snapshot/competition bindings this
#: check is PROVABLE from the log's own bytes (no metagraph), so it fires even for a burn/empty log.
_CENSUS_SOURCE = "census"
_COMPETITION_SOURCE = "competition-economics"


class MetagraphReader(Protocol):
    """The read seam the auditor binds the SNAPSHOT-derivable weight inputs against.

    In report mode this is the chainsim's registered-neuron view (``self._chain`` in the
    auditor loop, an ``HttpChainAdapter`` / ``InMemoryChain``); production uses the optional
    ``neurons_at(close_block)`` extension to read the archive-node metagraph at the exact
    epoch boundary. Read failures make the auditor fail CLOSED
    (INCONCLUSIVE), never PASS the authority's self-attested identities.
    """

    def neurons(self) -> list: ...


def _fold_matches(a: float, b: float) -> bool:
    """Compare two EWMA values under the fold tolerance (exclusion sentinel exact)."""
    import math

    if is_excluded(a) or is_excluded(b):
        return a == b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=_FOLD_TOL)


def _cycle_scores_backing_error(cycle_scores, committed: dict[str, dict]) -> str | None:
    """None if every cycle score is bound to a committed packet in its evidenced order.

    Closes the round-2 EWMA/sentinel hole. `committed` maps each of the uid's committed
    SCORE_PACKET digests to its committed ``{"score", "cycle_sequence", "excluded"}``.
    Each ``CycleScore`` must (1) reference a committed packet (an UNBACKED entry — a
    padded 0.0 or a substituted -1 — has no committed packet and FAILS), (2) carry the
    packet's committed ``cycle_sequence`` as its ``ordering_key`` (the fold ORDER is
    evidence-bound, so a reorder that changes the accumulator is caught), and (3) equal
    the committed value: the packet's recorded score, or the -1 exclusion sentinel iff
    the committed packet marks an exclusion. Finally the folded set must be EXACTLY the
    uid's committed packets — no dropped or duplicated cycle. Returns a reason on violation.
    """
    used: list[str] = []
    for cs in cycle_scores:
        fields = committed.get(cs.packet_digest)
        if fields is None:
            return (
                f"cycle score {cs.score} (ordering_key {cs.ordering_key}) references packet "
                f"{cs.packet_digest} that is not one of the uid's committed score packets — "
                "an unbacked cycle entry (padded 0.0 / substituted -1) cannot enter the fold"
            )
        if cs.ordering_key != fields["cycle_sequence"]:
            return (
                f"cycle ordering_key {cs.ordering_key} != the committed packet's "
                f"cycle_sequence {fields['cycle_sequence']} — the fold order is not "
                "evidence-backed (a reorder that changes the EWMA accumulator)"
            )
        if fields["excluded"]:
            if not is_excluded(cs.score):
                return (
                    f"the committed packet {cs.packet_digest} marks an exclusion but the "
                    f"cycle score {cs.score} is not the -1 exclusion sentinel"
                )
        else:
            if is_excluded(cs.score):
                return (
                    f"cycle score is the -1 exclusion sentinel but the committed packet "
                    f"{cs.packet_digest} records no exclusion — a substituted exclusion"
                )
            if not _fold_matches(cs.score, fields["score"]):
                return (
                    f"cycle score {cs.score} != the committed packet score {fields['score']} "
                    f"(packet {cs.packet_digest}) — a substituted cycle value"
                )
        used.append(cs.packet_digest)
    if sorted(used) != sorted(committed):
        return (
            "the folded cycle set is not exactly the uid's committed score packets — the "
            "fold drops or repeats committed evidence (used "
            f"{sorted(used)} vs committed {sorted(committed)})"
        )
    return None


def _challenge_commitment_backing_error(
    cycle_scores, committed_challenge: dict[str, dict]
) -> tuple[str, str] | None:
    """None if every cycle traces its ORDER + TRACK back to the CHALLENGE COMMITMENT.

    This is what makes the earning fold NON-CIRCULAR. The
    fold ORDER (`ordering_key`) and the item's TRACK are read from the pre-dispatch,
    independently-anchored challenge commitment (the item's DAG_REVEAL preimage) — NOT
    from the finalization-time score packet whose `cycle_sequence`/`track` the authority
    controls. `committed_challenge` maps each of the uid's committed SCORE_PACKET digests
    to its CHALLENGE-committed ``{"track", "ordering_key", "ref_committed_track"}``. So:

    - a reordered fold — ordering keys reassigned at finalization to fold a chosen order —
      is caught (EARNING_STATE_MISMATCH): the committed dispatch order is fixed BEFORE any
      score exists, so `cs.ordering_key` must equal it;
    - a substituted `committed_track` (e.g. stamping `upscaling` over a committed-compression
      challenge to dodge recompute-ability) is caught in the EARNING path without any media
      sampling (IDENTITY_MISMATCH, #9): the manifest ref's committed_track must equal the
      challenge-committed track.

    Returns ``(code, reason)`` on violation, else None.
    """
    for cs in cycle_scores:
        fields = committed_challenge.get(cs.packet_digest)
        if fields is None:
            return (
                EARNING_STATE_MISMATCH,
                f"packet {cs.packet_digest} has no CHALLENGE-committed dispatch record — the "
                "fold order/track cannot be traced to a pre-dispatch commitment",
            )
        if cs.ordering_key != fields["ordering_key"]:
            return (
                EARNING_STATE_MISMATCH,
                f"cycle ordering_key {cs.ordering_key} != the CHALLENGE-committed dispatch order "
                f"{fields['ordering_key']} (packet {cs.packet_digest}) — a reordered earning fold; "
                "the committed order is fixed pre-scoring and independently anchored, so the "
                "authority cannot choose the fold order at finalization",
            )
        if fields["ref_committed_track"] != fields["track"]:
            return (
                IDENTITY_MISMATCH,
                f"committed_track {fields['ref_committed_track']!r} != the CHALLENGE-committed "
                f"track {fields['track']!r} (packet {cs.packet_digest}) — a substituted track "
                "cannot dodge the pre-dispatch committed challenge",
            )
    return None


class BundleUnavailable(Exception):
    """A bundle the manifest named cannot be resolved (unreachable, not a finding).

    The auditor records SKIP — "never report PASS for what it couldn't verify"
    (the project design record §6). A bundle that resolves but whose digest does
    not match is a different thing (DIGEST_MISMATCH, a FAIL) and is caught inside
    ``verify_bundle``.
    """


class BundleSource(Protocol):
    """Resolves a manifest ``AuditFileRef`` (AUDIT_BUNDLE kind) to its AuditBundle.

    Production backs this with wherever the authority persisted the bundle (the
    competition event log / inference record); tests use :class:`InMemoryBundleSource`.
    Raises :class:`BundleUnavailable` when the bundle cannot be reached.
    """

    def bundle_for(self, ref: AuditFileRef) -> AuditBundle: ...


class InMemoryBundleSource:
    """Test double: a dict of ``bundle_digest -> AuditBundle``."""

    def __init__(self) -> None:
        self._by_digest: dict[str, AuditBundle] = {}

    def add(self, bundle: AuditBundle) -> AuditBundle:
        self._by_digest[bundle.bundle_digest()] = bundle
        return bundle

    def bundle_for(self, ref: AuditFileRef) -> AuditBundle:
        try:
            return self._by_digest[ref.digest]
        except KeyError as exc:
            raise BundleUnavailable(
                f"no bundle for digest {ref.digest} (item {ref.item_id})"
            ) from exc


def persist_bundle(store: AuditStore, bundle: AuditBundle) -> ArtifactRef:
    """Persist an ``AuditBundle`` as a content-addressed object; return its ref.

    The bundle's JSON is put under :attr:`ArtifactKind.AUDIT_BUNDLE`, so the store
    digest IS the ``bundle_digest()`` (both are ``sha256`` over the SAME canonical
    JSON). That is exactly the address the epoch manifest's AUDIT_BUNDLE ref carries,
    so :class:`StoredBundleSource` can resolve the bundle straight back by that digest.
    This is how the authority makes each epoch's bundles resolvable to auditors — the
    object store IS the bundle store (the project design record §1c).
    """
    data = canonical_json_bytes(bundle.model_dump(mode="json"))
    ref = store.put(data, ArtifactKind.AUDIT_BUNDLE)
    if (
        ref.digest != bundle.bundle_digest()
    ):  # invariant: content address == bundle digest
        raise RuntimeError(
            f"persisted bundle digest {ref.digest} != bundle_digest() "
            f"{bundle.bundle_digest()} — canonicalization drift"
        )
    return ref


class StoredBundleSource:
    """The REAL ``BundleSource``: resolves a bundle by digest from the object store.

    Production wiring (the project design record §1c): the authority persists each
    epoch's `AuditBundle`s into the shared object store (via :func:`persist_bundle`),
    content-addressed so the address equals the ``bundle_digest`` the manifest names.
    ``bundle_for`` fetches those bytes back, VERIFIES ``sha256(bytes) == ref.digest``
    (verify-on-read), parses the `AuditBundle`, and re-checks ``bundle_digest()`` — a
    resolved bundle whose digest disagrees is refused (it would fail DIGEST_MISMATCH in
    ``verify_bundle`` anyway; refusing here keeps a corrupt object from even being
    scored). An unreachable/corrupt/malformed bundle raises :class:`BundleUnavailable`
    → the auditor records SKIP (never a PASS-in-disguise, the project design record §6).

    Backend-agnostic: uses a digest-only bounded read (AUDIT_BUNDLE is not sealed),
    so it needs neither a trusted byte size nor a specific backend. Oversized bundle
    JSON is refused before a remote body is loaded into memory.
    """

    def __init__(self, store: AuditStore) -> None:
        self._store = store

    def bundle_for(self, ref: AuditFileRef) -> AuditBundle:
        try:
            data = self._store.get_digest_limited(
                ArtifactKind.AUDIT_BUNDLE,
                ref.digest,
                max_bytes=_MAX_AUDIT_BUNDLE_BYTES,
            )
        except FileNotFoundError as exc:
            raise BundleUnavailable(
                f"no stored bundle for digest {ref.digest} (item {ref.item_id})"
            ) from exc
        except (IntegrityError, OSError) as exc:
            raise BundleUnavailable(
                f"bundle {ref.digest} unreadable from the object store: {exc}"
            ) from exc
        if sha256_hex(data) != ref.digest:
            raise BundleUnavailable(
                f"stored bundle bytes for {ref.digest} do not match the content address "
                "(verify-on-read failed) — refusing to score a corrupt object"
            )
        try:
            bundle = AuditBundle.model_validate_json(data)
        except ValidationError as exc:
            raise BundleUnavailable(
                f"stored object {ref.digest} is not a valid AuditBundle: {exc}"
            ) from exc
        if bundle.bundle_digest() != ref.digest:
            raise BundleUnavailable(
                f"resolved bundle re-digests to {bundle.bundle_digest()} != manifest ref "
                f"{ref.digest} — refusing a bundle that does not match its manifest entry"
            )
        return bundle


#: The recompute Protocol plus the auditor's honest-refusal probe. Any
#: ScoreRecomputer works; RealScoreRecomputer adds ``unsupported_reason``.
class _Recomputer(Protocol):
    def recompute(self, bundle: object, artifacts: Mapping[ArtifactKind, bytes]): ...


class Auditor:
    """Audits one epoch and produces (optionally submits) a signed AuditReport."""

    def __init__(
        self,
        config: AuditorConfig,
        bundle_source: BundleSource,
        *,
        chain: MetagraphReader | None = None,
        reveal_verifier: Callable[[bytes], bool] | None = None,
        miner_receipt_verifier: Callable[[object], bool] | None = None,
        availability_verify_fn: Callable[[str, bytes, str], bool] | None = None,
        signer: ReportSigner | None = None,
        client: AuditResultsClient | None = None,
    ) -> None:
        self._config = config
        self._bundle_source = bundle_source
        # The close-block metagraph read seam: the auditor binds each
        # nonzero-weight uid's identity/dedup/track to the metagraph it reads ITSELF, never
        # the authority's self-attested MinerSnapshot fields. None ⇒ no metagraph wired ⇒
        # the SNAPSHOT binding is UNVERIFIABLE ⇒ fail closed (INCONCLUSIVE), never a PASS.
        self._chain = chain
        self._reveal_verifier = reveal_verifier
        self._miner_receipt_verifier = miner_receipt_verifier
        self._availability_verify_fn = availability_verify_fn
        self._signer = signer
        self._client = client

    @classmethod
    def over_store(
        cls,
        config: AuditorConfig,
        store: AuditStore,
        *,
        chain: MetagraphReader | None = None,
        reveal_verifier: Callable[[bytes], bool] | None = None,
        miner_receipt_verifier: Callable[[object], bool] | None = None,
        availability_verify_fn: Callable[[str, bytes, str], bool] | None = None,
        signer: ReportSigner | None = None,
        client: AuditResultsClient | None = None,
    ) -> "Auditor":
        """PRODUCTION construction: resolve bundles from the shared object store.

        Wires a :class:`StoredBundleSource` over ``store`` — where the authority
        persisted each epoch's bundles (via :func:`persist_bundle`) — so a real
        auditor resolves every sampled item's bundle by digest from the same content
        layer it mirrors the epoch log from. ``chain`` is the close-block metagraph read
        seam; production wires the archive-node/commitment adapter, the
        auditor loop wires its read-only chain adapter. Tests keep injecting the
        :class:`InMemoryBundleSource` fake through the plain constructor.
        """
        return cls(
            config,
            StoredBundleSource(store),
            chain=chain,
            reveal_verifier=reveal_verifier,
            miner_receipt_verifier=miner_receipt_verifier,
            availability_verify_fn=availability_verify_fn,
            signer=signer,
            client=client,
        )

    def audit_epoch(
        self,
        epoch_log: EpochLog,
        store: AuditStore,
        sample_policy: SamplePolicy,
        recomputer: _Recomputer,
        now: datetime,
        *,
        beacon: str = NO_BEACON,
        prior_log: EpochLog | None = None,
        is_genesis: bool = True,
    ) -> AuditReport:
        """Sample, recompute, re-derive weights + EARNING STATE, aggregate → a report.

        ``beacon`` is the post-finalization on-chain anchor value (#10) mixed into the
        sample seed so the authority could not steer which items get audited when it
        built the manifest. ``prior_log`` is the PREVIOUS epoch's log (#1) — when
        supplied the earning-state carry-in is chained against it (and the reward window is
        re-derived from it), closing the substituted-earning-state hole all the way back
        to genesis.

        ``is_genesis`` is the LOOP's independent determination of whether THIS epoch is
        the genuine genesis epoch. It is the ONLY authorisation for a
        MISSING ``prior_log_digest`` (None): omitting the digest resets ALL chained state
        (earning carry-in and reward-window state), so a NON-genesis epoch that omits it is a
        BROKEN CHAIN ⇒ DISPUTED — NEVER re-treated as genesis. Only the true genesis (the
        discovered/configured floor, an epoch with no predecessor the cursor advanced past)
        legitimately carries ``prior_log_digest`` None. The loop threads the fact from the
        cursor + genesis floor; the default True is the pure-function convenience for the
        genesis unit tests (production ALWAYS passes the loop-derived value).
        """
        snapshot_digest = epoch_log.log_digest()
        try:
            sampled = sample_items(
                epoch_log.audit_manifest,
                epoch_id=epoch_log.epoch_id,
                auditor_hotkey=self._config.auditor_hotkey,
                policy=sample_policy,
                beacon=beacon,
            )
        except DuplicateAuditIdentity as exc:
            # an internal review(b): an ambiguous duplicate audit identity is conclusive
            # tampering — the manifest cannot even be sampled un-steerably. Short-circuit
            # to a DISPUTED report (reflected through the weight verdict, the dispute
            # channel the Audit Results API reads) rather than audit it as if honest.
            return self._structural_defect_report(
                epoch_log, snapshot_digest, now, exc, DUPLICATE_AUDIT_IDENTITY
            )
        except ManifestIncomplete as exc:
            # an internal review: a structurally malformed manifest (an item missing its
            # bundle/packet pair) cannot be sampled. Previously this uncaught exception
            # made the auditor runner retry forever with NO signed report — BLOCKING the
            # cursor. A structural defect in the authority's own manifest is a provable
            # fault, so emit a SIGNED DISPUTED report (same dispute channel as the
            # duplicate-identity short-circuit), never an uncaught exception.
            return self._structural_defect_report(
                epoch_log, snapshot_digest, now, exc, MANIFEST_INCOMPLETE
            )

        published_root = epoch_log.audit_manifest.score_packet_merkle_root
        competition_input = epoch_log.audit_manifest.competition_input
        competition_items = {
            (item.challenge_id, item.item_id): item
            for item in (
                competition_input.items if competition_input is not None else ()
            )
        }
        # uid -> hotkey, so a SAMPLED item's recompute can bind the packet's miner to the
        # identity the manifest attributes the item to — even when the bundle pins no miner
        #. None for an archived-baseline row (item.uid is None).
        hotkey_by_uid = {m.uid: m.hotkey for m in epoch_log.miners}
        item_verdicts = tuple(
            self._audit_item(
                item,
                store,
                recomputer,
                published_root,
                # an internal review: a COMPETITION contender item carries its committed
                # `expected_hotkey` (the CompetitionContenderInput identity), so a SAMPLED contender
                # bundle binds the packet's miner to the committed contender — never the log's
                # self-attested per-uid hotkey. Earning/baseline rows keep the uid->log-hotkey path.
                expected_miner_hotkey=(
                    item.expected_hotkey
                    if item.expected_hotkey is not None
                    else (hotkey_by_uid.get(item.uid) if item.uid is not None else None)
                ),
                competition_context=(
                    CompetitionAuditContext(
                        competition_id=competition_input.competition_id,
                        track=competition_input.track,
                        manifest_digest=competition_input.manifest_digest,
                        threshold_commitment=competition_items[
                            (item.challenge_id, item.item_id)
                        ].threshold_commitment,
                        item_index=competition_items[
                            (item.challenge_id, item.item_id)
                        ].item_index,
                        input_sha256=competition_items[
                            (item.challenge_id, item.item_id)
                        ].input_sha256,
                        reference_sha256=competition_items[
                            (item.challenge_id, item.item_id)
                        ].reference_sha256,
                        upscale_factor=competition_items[
                            (item.challenge_id, item.item_id)
                        ].upscale_factor,
                        target_width=competition_items[
                            (item.challenge_id, item.item_id)
                        ].target_width,
                        target_height=competition_items[
                            (item.challenge_id, item.item_id)
                        ].target_height,
                        item_commitment=competition_items[
                            (item.challenge_id, item.item_id)
                        ].item_commitment,
                    )
                    if item.source == "competition"
                    and competition_input is not None
                    and (item.challenge_id, item.item_id) in competition_items
                    else None
                ),
            )
            for item in sampled
        )
        # an internal review: read the close-block METAGRAPH ourselves and bind the
        # SNAPSHOT-derivable weight inputs (uid->hotkey/coldkey/ip identity, the IP/coldkey
        # DEDUP `excluded` outcome, and the committed track) to it — never the authority's
        # self-attested MinerSnapshot fields. An unreadable metagraph is UNVERIFIABLE, so
        # the binding fails closed (INCONCLUSIVE), never a PASS.
        metagraph = self._read_metagraph(epoch_log.close_block)
        snapshot_verdicts = tuple(self._snapshot_verdicts(epoch_log, metagraph))
        (
            derived_competition_result,
            derived_reward_window_state,
            competition_verdicts,
        ) = self._competition_verdicts(epoch_log, store, prior_log, is_genesis)
        # (The windowed-evidence re-derivation was REMOVED with the retention multiplier for
        # v1 — retention removed — owner decision; an internal review.)
        # The WEIGHT verdict is computed over INDEPENDENTLY-VERIFIED inputs (the metagraph
        # identity/dedup/track + accumulate_score from the earning fold); a provable snapshot
        # mismatch DISPUTES, an unverifiable one HOLDs (INCONCLUSIVE).
        weight_verdict = self._weight_verdict(
            epoch_log,
            metagraph,
            snapshot_verdicts,
            reward_window_state=derived_reward_window_state,
        )
        # Competition result/reward window were rederived immediately above from the exact
        # namespaced score-packet and bundle evidence.
        # #1: re-derive the EARNING STATE (accumulate_score fold) from the audited packets +
        # chained carry-in — the honesty crux. Covers EVERY nonzero-weight uid cheaply (the
        # fold is arithmetic over already-committed packet scores); the expensive media
        # recompute stays sampled above.
        earning_verdicts = tuple(
            self._earning_verdicts(epoch_log, store, prior_log, is_genesis)
        )
        # an internal review: bind the `created_at` weight-INPUT time base to independently-verifiable
        # state — it must agree with the CLOSE BLOCK time read from the chain (a disagreement is a
        # provable mismatch ⇒ DISPUTED; an unreadable close-block time ⇒ INCONCLUSIVE, never a PASS).
        # Competition completion and cycle chronology are checked in
        # `_competition_verdicts`; this pass binds the epoch-level close time.
        timebase_verdicts: tuple[ItemVerdict, ...] = (
            self._created_at_verdict(epoch_log),
        )
        # an internal review: cross-check the MINER CENSUS against the log's OWN committed evidence.
        # The snapshot/earning verdicts above are all scoped to the POSITIVE-weight set, so a log
        # with an empty/burn positive set (miners=[], {burn_uid:1.0}) skips them entirely — letting
        # an authority store earning evidence for real miners then OMIT every one of them from the
        # census and receive CLEAN. This check is PROVABLE from the log's own bytes (no metagraph),
        # so it fires even for a burn/empty log: any evidenced-but-censored miner is a FAIL ⇒
        # DISPUTED. A genuinely-empty epoch (no committed evidence) yields no verdicts ⇒ CLEAN.
        census_verdicts = tuple(self._census_verdicts(epoch_log, metagraph))
        # Schema v11: verify the COMPLETE cumulative replay-boundary map against the anchored
        # predecessor.  This is independent of the current census/accumulator sign and therefore
        # survives carry-only, exclusion, deregistration, empty-burn, and hotkey-change epochs.
        watermark_verdicts = tuple(
            self._fold_cursor_verdicts(epoch_log, prior_log, is_genesis)
        )
        # an internal review: detect a prior-POSITIVE, STILL-REGISTERED miner SILENTLY RESET to
        # 0.0 / excluded (or dropped from the census) this epoch with NO evidenced reason — the
        # earning fold + census only look at CURRENT accumulators/evidence, so an erased earning
        # state slips through. Chained against the prior epoch log + bound to the close-block
        # metagraph; a genuine deregistration / evidenced exclusion is legitimate ⇒ not flagged.
        reset_verdicts = tuple(
            self._reset_earning_verdicts(epoch_log, prior_log, metagraph)
        )
        # an internal review: LOG-LEVEL predecessor-chain enforcement — the per-uid carry-in /
        # carry-forward chain checks only run for a SELECTED earning uid, so an EMPTY canonical
        # burn log (miners=[], {burn_uid:1.0}) selects none and NEVER enforces the chain. A
        # non-genesis authority could omit prior_log_digest (or reference an unavailable prior),
        # publish the empty burn vector, and audit CLEAN — silently resetting all prior earning
        # state. This log-level guard fires regardless of the audited set: a broken chain FAILs
        # (DISPUTED), an unverifiable one HOLDs; a genuine empty epoch that MAINTAINS the chain
        # stays CLEAN.
        predecessor_verdict = self._predecessor_chain_verdict(
            epoch_log, prior_log, is_genesis
        )
        # an internal review: validate every committed / log TRACK against the PROTOCOL track set —
        # an out-of-protocol track (silently dropped from every tokenomics pool) substitutes a burn
        # while every self-consistent declaration agrees. Provable from the log's own bytes.
        track_verdicts = tuple(self._track_membership_verdicts(epoch_log))
        # an internal review: the burn RECIPIENT must be CANONICAL — resolved from OUR config
        # (`AuditorConfig.burn_uid`, the SAME value the authority is configured with), NEVER the
        # log. `EpochLog._validate` only checks the burn uid is the SOLE positive uid, so an
        # untrusted authority could anchor an empty log burning 100% to a beneficiary IT controls
        # and audit CLEAN. A log.burn_uid that is not the canonical value is a conclusive fault ⇒
        # DISPUTED (BURN_UID_MISMATCH). Fires only on a burn epoch (log.burn_uid is not None); a
        # None burn uid is a normal epoch with no burn recipient to canonicalize.
        burn_verdict = self._burn_uid_verdict(epoch_log)
        # The snapshot-binding + time-base + census + burn verdicts ride the same earning-verdicts
        # channel (a FAIL disputes, a SKIP holds INCONCLUSIVE) — never counted toward the media
        # coverage floor (their sources are not media strata).
        earning_verdicts = (
            earning_verdicts
            + snapshot_verdicts
            + timebase_verdicts
            + census_verdicts
            + watermark_verdicts
            + reset_verdicts
            + ((predecessor_verdict,) if predecessor_verdict is not None else ())
            + track_verdicts
            + ((burn_verdict,) if burn_verdict is not None else ())
            + competition_verdicts
        )
        # An earning/snapshot FAIL disputes the epoch. It is reflected THROUGH the weight
        # verdict so the roll-up (and the Audit Results API, whose disputed-items channel
        # reads item_verdicts + weight_verdict) sees it without a signature change.
        weight_verdict = self._fold_earning_into_weight(
            weight_verdict, earning_verdicts
        )

        report = AuditReport(
            auditor_hotkey=self._config.auditor_hotkey,
            epoch_id=epoch_log.epoch_id,
            audit_mode=self._config.audit_mode,
            snapshot_digest=snapshot_digest,
            pipeline_version=epoch_log.scorer_version,
            sampled_at=now,
            competition_n=sum(1 for it in sampled if it.source == "competition"),
            inference_n=sum(1 for it in sampled if it.source == "inference"),
            item_verdicts=item_verdicts,
            earning_verdicts=earning_verdicts,
            weight_verdict=weight_verdict,
            # overall is DERIVED at construction (report.py) — never trusted from here.
        )
        if self._signer is not None:
            report = report.signed(self._signer)
        return report

    def audit_and_submit(
        self,
        epoch_log: EpochLog,
        store: AuditStore,
        sample_policy: SamplePolicy,
        recomputer: _Recomputer,
        now: datetime,
        *,
        beacon: str = NO_BEACON,
        prior_log: EpochLog | None = None,
        is_genesis: bool = True,
    ) -> tuple[AuditReport, SubmitAck]:
        """Audit the epoch and POST the report through the results client."""
        if self._client is None:
            raise RuntimeError("no AuditResultsClient wired — cannot submit")
        report = self.audit_epoch(
            epoch_log,
            store,
            sample_policy,
            recomputer,
            now,
            beacon=beacon,
            prior_log=prior_log,
            is_genesis=is_genesis,
        )
        return report, self._client.submit(report)

    def invalid_epoch_report(
        self,
        *,
        epoch_id: int,
        snapshot_digest: str,
        pipeline_version: str,
        published_weight_vector_digest: str,
        now: datetime,
        error: Exception,
        conclusive: bool,
    ) -> AuditReport:
        """Sign a central finding when strict epoch decoding cannot start an audit.

        The caller must first authenticate the raw canonical bytes against the
        authority pointer and on-chain anchor and extract the narrow submission
        identity. Known schema/model validation failures are conclusive defects and
        therefore DISPUTED. An unexpected local exception may be an auditor bug, so
        it is INCONCLUSIVE. Neither outcome has an enforcement path to weight-setting.
        """
        verdict = ItemVerdictKind.FAIL if conclusive else ItemVerdictKind.SKIP
        code = EPOCH_LOG_INVALID if conclusive else EPOCH_LOG_UNVERIFIED
        detail = f"{type(error).__name__}: {error}"
        if len(detail) > 4096:
            detail = detail[:4093] + "..."
        report = AuditReport(
            auditor_hotkey=self._config.auditor_hotkey,
            epoch_id=epoch_id,
            audit_mode=self._config.audit_mode,
            snapshot_digest=snapshot_digest,
            pipeline_version=pipeline_version,
            sampled_at=now,
            item_verdicts=(),
            earning_verdicts=(),
            weight_verdict=WeightVerdict(
                recomputed_weight_vector_digest="",
                published_weight_vector_digest=published_weight_vector_digest,
                verdict=verdict,
                code=code,
                detail=(
                    "authenticated authority epoch failed strict audit-model "
                    f"validation: {detail}"
                ),
            ),
        )
        if self._signer is not None:
            report = report.signed(self._signer)
        return report

    def _structural_defect_report(
        self,
        epoch_log: EpochLog,
        snapshot_digest: str,
        now: datetime,
        exc: Exception,
        code: str,
    ) -> AuditReport:
        """A conclusive DISPUTED report for a manifest with a STRUCTURAL defect.

        an internal review(b) / round-8 #8: the manifest cannot even be sampled — an
        ambiguous duplicate identity (DUPLICATE_AUDIT_IDENTITY) or an item missing its
        bundle/packet pair (MANIFEST_INCOMPLETE). There is nothing to recompute, so the
        fault is surfaced through the weight verdict — the dispute channel the roll-up
        (`overall_status`) and the Audit Results API both read — so the epoch derives
        DISPUTED without any sampling, with a SIGNED report (never an uncaught exception
        that would block the auditor cursor).
        """
        weight_verdict = WeightVerdict(
            recomputed_weight_vector_digest="",
            published_weight_vector_digest=epoch_log.weight_vector_digest,
            verdict=ItemVerdictKind.FAIL,
            code=code,
            detail=str(exc),
        )
        report = AuditReport(
            auditor_hotkey=self._config.auditor_hotkey,
            epoch_id=epoch_log.epoch_id,
            audit_mode=self._config.audit_mode,
            snapshot_digest=snapshot_digest,
            pipeline_version=epoch_log.scorer_version,
            sampled_at=now,
            item_verdicts=(),
            earning_verdicts=(),
            weight_verdict=weight_verdict,
        )
        if self._signer is not None:
            report = report.signed(self._signer)
        return report

    @staticmethod
    def _fold_earning_into_weight(
        weight_verdict: WeightVerdict, earning_verdicts: tuple[ItemVerdict, ...]
    ) -> WeightVerdict:
        """Reflect any earning-state FAIL into the weight verdict (dispute channel)."""
        fails = [v for v in earning_verdicts if v.verdict is ItemVerdictKind.FAIL]
        if weight_verdict.verdict is ItemVerdictKind.FAIL:
            return weight_verdict
        if fails:
            uids = ", ".join(str(v.uid) for v in fails if v.uid is not None) or "-"
            return weight_verdict.model_copy(
                update={
                    "verdict": ItemVerdictKind.FAIL,
                    "code": EARNING_STATE_MISMATCH,
                    "detail": (
                        "the stated earning state does not re-derive from the audited "
                        f"evidence for uid(s) {uids}: {fails[0].detail}"
                    ),
                }
            )
        skips = [v for v in earning_verdicts if v.verdict is ItemVerdictKind.SKIP]
        if skips and weight_verdict.verdict is ItemVerdictKind.PASS:
            return weight_verdict.model_copy(
                update={
                    "verdict": ItemVerdictKind.SKIP,
                    "code": skips[0].code,
                    "detail": (
                        "the stated earning state is not independently verifiable: "
                        f"{skips[0].detail}"
                    ),
                }
            )
        return weight_verdict

    # -- per-item -------------------------------------------------------------------

    def _audit_item(
        self,
        item: AuditItem,
        store: AuditStore,
        recomputer: _Recomputer,
        published_root: str | None,
        *,
        expected_miner_hotkey: str | None = None,
        competition_context: CompetitionAuditContext | None = None,
    ) -> ItemVerdict:
        base = dict(
            source=item.source,
            challenge_id=item.challenge_id,
            item_id=item.item_id,
            uid=item.uid,
            bundle_digest=item.bundle_ref.digest,
            packet_digest=item.packet_ref.digest,
        )
        # an internal review: a SAMPLED item attributed to a uid (item.uid is not None)
        # whose log hotkey is missing/empty is a fault, not a pass. verify_bundle binds
        # the packet's miner to `expected_miner_hotkey`, but that comparison SKIPS when the
        # expected hotkey is null — so a null log hotkey would let a foreign miner's packet
        # pass the media recompute. Fail closed here BEFORE recompute. Baseline rows
        # (item.uid is None) legitimately carry no expected hotkey and are exempt.
        if item.uid is not None and not expected_miner_hotkey:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=IDENTITY_MISMATCH,
                detail=(
                    f"sampled item is attributed to uid {item.uid} but its log hotkey is "
                    f"missing/empty ({expected_miner_hotkey!r}) — a null expected identity "
                    "cannot bind the packet's miner; failing closed rather than skipping the "
                    "miner check (the null-hotkey bypass)"
                ),
                **base,
            )
        if item.source == "competition" and competition_context is None:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=COMPETITION_MISMATCH,
                detail=(
                    "competition audit ref is not bound to a committed competition "
                    "manifest/evaluation item"
                ),
                **base,
            )
        try:
            bundle = self._bundle_source.bundle_for(item.bundle_ref)
        except BundleUnavailable as exc:
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=ARTIFACT_MISSING,
                detail=str(exc),
                **base,
            )

        if competition_context is None:
            chronology = self._challenge_chronology(bundle, store)
            if chronology.kind is not ChronologyKind.PASS:
                return ItemVerdict(
                    verdict=(
                        ItemVerdictKind.FAIL
                        if chronology.kind is ChronologyKind.FAIL
                        else ItemVerdictKind.SKIP
                    ),
                    code=chronology.code,
                    detail=chronology.detail,
                    **base,
                )

        # #9: the COMMITTED track governs recompute-ability and the recompute path —
        # never the authority's packet-declared track. A packet substituting
        # another track to force an unavailable-backend SKIP over a real committed
        # (compression) item is a FAIL here (its declared track != the committed one),
        # so it cannot dodge verification.
        committed_track = item.packet_ref.committed_track
        if committed_track is not None:
            declared = self._packet_track(store, bundle)
            if declared is not None and declared != committed_track:
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=IDENTITY_MISMATCH,
                    detail=(
                        f"packet declares track {declared!r} but the committed challenge "
                        f"pins track {committed_track!r} — a substituted track cannot dodge "
                        "recompute-ability"
                    ),
                    **base,
                )

        # Honest-refusal probe BEFORE any media work: an item this CPU backend cannot
        # recompute is a SKIP, never a false CLEAN. Cheap — reads
        # only the track, and prefers the COMMITTED track when the manifest carries it.
        probe_reason = self._unsupported_reason(
            recomputer, bundle, store, track=committed_track
        )
        if probe_reason is not None:
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP, code="", detail=probe_reason, **base
            )

        report = verify_bundle(
            bundle,
            store,
            recomputer,  # type: ignore[arg-type]
            expected_bundle_digest=item.bundle_ref.digest,
            # Bind the packet's miner to the uid this item is attributed to (review
            # round-6 #1): a foreign miner's packet cannot pass recompute by nulling
            # the bundle miner. None for an archived-baseline row.
            expected_miner_hotkey=expected_miner_hotkey,
            # an internal review: for a uid-attributed item a null/empty expected hotkey is
            # itself a fault (defense-in-depth behind the fail-closed guard above), never a
            # silent fall-back to the bundle's pinned miner.
            require_expected_miner=item.uid is not None,
            # v2 manifest: PROVE strict merkle inclusion of the sampled packet against
            # the committed root. A packet that is not provably in the committed set
            # (missing/invalid proof, or a root that excludes it) → MERKLE_EXCLUSION.
            published_root=published_root,
            inclusion_proof=item.packet_ref.inclusion_proof,
            reveal_verifier=self._reveal_verifier,
            strict=self._config.strict,
            competition_context=competition_context,
        )
        verdict, code, detail = self._verdict_from(report)
        return ItemVerdict(verdict=verdict, code=code, detail=detail, **base)

    def _challenge_chronology(self, bundle: AuditBundle, store: AuditStore):
        kwargs = {}
        if self._miner_receipt_verifier is not None:
            kwargs["receipt_verifier"] = self._miner_receipt_verifier
        return verify_challenge_chronology(
            bundle,
            store,
            self._chain,
            require_anchor=self._config.require_external_challenge_anchors,
            expected_netuid=self._config.challenge_anchor_netuid,
            scoring=self._config.scoring,
            **kwargs,
        )

    def _unsupported_reason(
        self,
        recomputer: _Recomputer,
        bundle: AuditBundle,
        store: AuditStore,
        *,
        track: str | None = None,
    ) -> str | None:
        probe = getattr(recomputer, "unsupported_reason", None)
        if probe is None:
            return None
        packet_bytes: bytes | None = None
        try:
            packet_bytes = store.get_limited(
                bundle.score_packet, _MAX_AUDIT_METADATA_BYTES
            )
        except (IntegrityError, FileNotFoundError, OSError):
            packet_bytes = None  # verify_bundle will surface the artifact problem
        artifacts = (
            {ArtifactKind.SCORE_PACKET: packet_bytes}
            if packet_bytes is not None
            else {}
        )
        try:
            # Prefer the COMMITTED track (#9); recomputers that ignore the kwarg still
            # work — this Protocol call passes it only when the backend accepts it.
            return self._probe(probe, bundle, artifacts, track)
        except RecomputeUnavailable as exc:  # a probe that decides by attempting
            return exc.reason

    @staticmethod
    def _probe(probe, bundle: AuditBundle, artifacts, track: str | None):
        try:
            return probe(bundle, artifacts, track=track)
        except TypeError:
            return probe(bundle, artifacts)  # legacy probe without a track kwarg

    @staticmethod
    def _packet_track(store: AuditStore, bundle: AuditBundle) -> str | None:
        """The track the recorded packet DECLARES (for the committed-track cross-check)."""
        try:
            raw = store.get_limited(bundle.score_packet, _MAX_AUDIT_METADATA_BYTES)
        except (IntegrityError, FileNotFoundError, OSError):
            return None
        import json

        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        track = payload.get("track")
        return str(track) if track is not None else None

    @staticmethod
    def _verdict_from(report) -> tuple[ItemVerdictKind, str, str]:
        """Map a VerificationReport to (verdict, code, detail).

        - a real (non-skipped) ARTIFACT_MISSING ⇒ SKIP: unreachable, the auditor
          could not verify it (never PASS-in-disguise);
        - any other real failure ⇒ FAIL with its stable code (the provable fault);
        - a STRICT-mode skipped check that FAILED (``skipped and not passed``) ⇒ SKIP
          (INCONCLUSIVE), NEVER a wash to PASS: under ``strict=True``
          ``verify_bundle`` records a missing published merkle root / missing reveal
          verifier as ``CheckResult(passed=False, skipped=True)`` — "strict mode treats
          skipped checks as failures". The prior filter excluded EVERY ``skipped`` check
          regardless of ``passed``, so a strict verification FAILURE mapped to PASS. A
          strict skip is an UNVERIFIABLE anchor/verifier, so it fails CLOSED to
          INCONCLUSIVE (not a provable DISPUTE, but never CLEAN);
        - otherwise ⇒ PASS. Only a GENUINELY benign skip (``passed`` truthy / not
          applicable — an absent anchor / unwired reveal under ``strict=False``) is
          ignored and does not dispute.
        """
        real_failures = [c for c in report.checks if not c.passed and not c.skipped]
        # A strict skip that FAILED is unverifiable, not benign — fail closed to SKIP.
        strict_skip_failures = [c for c in report.checks if c.skipped and not c.passed]
        missing = [c for c in real_failures if c.code == ARTIFACT_MISSING]
        findings = [c for c in real_failures if c.code != ARTIFACT_MISSING]
        if findings:
            first = findings[0]
            return ItemVerdictKind.FAIL, first.code or "", first.reason
        if missing:
            return ItemVerdictKind.SKIP, ARTIFACT_MISSING, missing[0].reason
        if strict_skip_failures:
            first = strict_skip_failures[0]
            return ItemVerdictKind.SKIP, first.code or "", first.reason
        return ItemVerdictKind.PASS, "", ""

    def _competition_verdicts(
        self,
        log: EpochLog,
        store: AuditStore,
        prior_log: EpochLog | None,
        is_genesis: bool,
    ) -> tuple[CompetitionResult | None, RewardWindowState, tuple[ItemVerdict, ...]]:
        """Rebuild competition result/reward window from every committed packet score.

        Packet arithmetic is exhaustive and cheap; media measurement remains subject to
        the same beacon-stratified sampling as inference (the validator own-audit policy
        can and does select the complete competition stratum before submission).
        """

        def verdict(
            kind: ItemVerdictKind,
            code: str = "",
            detail: str = "",
            *,
            item_id: str = "competition-result",
        ) -> ItemVerdict:
            return ItemVerdict(
                source=_COMPETITION_SOURCE,
                challenge_id=(
                    log.audit_manifest.competition_input.competition_id
                    if log.audit_manifest.competition_input is not None
                    else ""
                ),
                item_id=item_id,
                miner_hotkey=None,
                uid=None,
                bundle_digest="",
                packet_digest="",
                verdict=kind,
                code=code,
                detail=detail,
            )

        comp_input = log.audit_manifest.competition_input
        packet_scores: dict[str, float] = {}
        if comp_input is not None:
            try:
                commitment = load_competition_commitment(
                    store, comp_input.commitment_root
                )
                manifest_raw = store.get_digest_limited(
                    ArtifactKind.MANIFEST,
                    comp_input.manifest_digest,
                    max_bytes=_MAX_AUDIT_METADATA_BYTES,
                )
                manifest = CompetitionManifest.model_validate_json(manifest_raw)
                if manifest.canonical_json().encode("utf-8") != manifest_raw:
                    raise ValueError(
                        "competition manifest object is not its canonical JSON preimage"
                    )
                if manifest.manifest_digest() != comp_input.manifest_digest:
                    raise ValueError(
                        "competition manifest object does not open the committed digest"
                    )
                if (
                    manifest.competition_id != comp_input.competition_id
                    or manifest.track != comp_input.track
                ):
                    raise ValueError(
                        "competition manifest id/track differs from epoch evidence"
                    )
                verify_competition_anchor_on_chain(
                    self._chain,
                    comp_input,
                    expected_netuid=self._config.challenge_anchor_netuid,
                    competition_start_time=manifest.start_time,
                    epoch_close_block=log.close_block,
                )
                if manifest.baseline is None:
                    raise ValueError(
                        "earning competition manifest has no archived executable baseline"
                    )
                expected_commitment = {
                    "manifest_digest": comp_input.manifest_digest,
                    "baseline_version": manifest.baseline.version,
                    "baseline_artifact_digest": manifest.baseline.artifact_digest,
                    "baseline_provenance_digest": manifest.baseline.provenance_digest,
                    "baseline_tree_digest": pin_git_sha(manifest.baseline.tree_sha),
                    "baseline_image_digest": manifest.baseline.image_digest,
                    "dataset_selection_seed_commitment": (
                        manifest.scoring_seed_commitment
                    ),
                    "reward_param_digest": reward_parameter_digest(
                        self._config.tokenomics
                    ),
                }
                for field, expected in expected_commitment.items():
                    observed = getattr(commitment, field)
                    if observed != expected:
                        raise ValueError(
                            f"competition commitment {field} {observed} differs from "
                            f"the executed epoch policy {expected}"
                        )
                if (
                    comp_input.baseline_version != manifest.baseline.version
                    or comp_input.baseline_artifact_digest
                    != manifest.baseline.artifact_digest
                    or comp_input.baseline_artifact_bytes
                    != manifest.baseline.artifact_bytes
                    or comp_input.baseline_execution_image_digest
                    != manifest.baseline.image_digest
                    or comp_input.baseline_provenance_digest
                    != manifest.baseline.provenance_digest
                    or comp_input.baseline_provenance_bytes
                    != manifest.baseline.provenance_bytes
                ):
                    raise ValueError(
                        "competition input baseline provenance differs from the "
                        "pre-enrollment manifest/commitment"
                    )
                if comp_input.applied_at != log.created_at:
                    raise ValueError(
                        "competition applied_at must equal the epoch's chain-bound "
                        "created_at; reward windows cannot use a database-local clock"
                    )

                if manifest.track == "upscaling":
                    commitments = manifest.evaluation_item_commitments or []
                    if len(comp_input.items) != len(commitments):
                        raise ValueError(
                            f"upscaling competition input has {len(comp_input.items)} "
                            f"item(s), but the anchored manifest commits "
                            f"{len(commitments)}"
                        )
                    allowed_factors = set(manifest.allowed_upscale_factors or ())
                    for expected_index, (item, committed) in enumerate(
                        zip(comp_input.items, commitments, strict=True)
                    ):
                        if (
                            item.item_index != expected_index
                            or item.input_sha256 is None
                            or item.reference_sha256 is None
                            or item.upscale_factor is None
                            or item.item_commitment is None
                        ):
                            raise ValueError(
                                f"upscaling evaluation item {expected_index} has an "
                                "incomplete or reordered commitment preimage"
                            )
                        if item.upscale_factor not in allowed_factors:
                            raise ValueError(
                                f"upscaling evaluation item {expected_index} factor "
                                f"{item.upscale_factor} is not allowed by the manifest"
                            )
                        derived_item_commitment = evaluation_item_commitment(
                            competition_id=comp_input.competition_id,
                            item_index=item.item_index,
                            reference_sha256=item.reference_sha256,
                            input_sha256=item.input_sha256,
                            upscale_factor=item.upscale_factor,
                            target_width=item.target_width,
                            target_height=item.target_height,
                        )
                        if (
                            item.item_commitment != derived_item_commitment
                            or committed != derived_item_commitment
                        ):
                            raise ValueError(
                                f"upscaling evaluation item {expected_index} does not "
                                "open its anchored manifest commitment"
                            )

                census_by_hotkey = {entry.hotkey: entry for entry in log.miner_census}
                if len(census_by_hotkey) != len(log.miner_census):
                    raise ValueError("competition close-block census repeats a hotkey")
                dedup_candidates: list[CompetitionDedupCandidate] = []
                for subject in comp_input.subjects:
                    refs = log.audit_manifest.competition_bundles[subject.subject_id]
                    bundle_refs = {
                        ref.digest: ref
                        for ref in refs
                        if ref.kind is AuditFileKind.AUDIT_BUNDLE
                    }
                    packet_refs = {
                        ref.digest: ref
                        for ref in refs
                        if ref.kind is AuditFileKind.SCORE_PACKET
                    }
                    output_digests: list[str] = []
                    image_digests: set[str] = set()
                    expected_hotkey = (
                        None if subject.role == "baseline" else subject.hotkey
                    )
                    for item, packet_digest, bundle_digest in zip(
                        comp_input.items,
                        subject.packet_digests,
                        subject.audit_bundle_digests,
                        strict=True,
                    ):
                        packet_ref = packet_refs[packet_digest]
                        bundle_ref = bundle_refs[bundle_digest]
                        expected_item = (item.challenge_id, item.item_id)
                        if (
                            packet_ref.challenge_id,
                            packet_ref.item_id,
                        ) != expected_item or (
                            bundle_ref.challenge_id,
                            bundle_ref.item_id,
                        ) != expected_item:
                            raise ValueError(
                                "competition subject digest ordering differs from the "
                                "committed evaluation-item ordering"
                            )
                        bundle = self._bundle_source.bundle_for(bundle_ref)
                        if bundle.bundle_digest() != bundle_ref.digest:
                            raise ValueError(
                                "competition bundle does not re-digest to its manifest ref"
                            )
                        if bundle.score_packet.digest != packet_ref.digest:
                            raise ValueError(
                                "competition bundle score-packet digest differs from manifest ref"
                            )
                        if bundle.manifest.digest != comp_input.manifest_digest:
                            raise ValueError(
                                "competition bundle manifest differs from the epoch commitment"
                            )
                        expected_bundle_identity = (
                            item.challenge_id,
                            item.item_id,
                            expected_hotkey,
                            packet_ref.digest,
                            comp_input.manifest_digest,
                            item.threshold_commitment,
                        )
                        actual_bundle_identity = (
                            bundle.challenge_id,
                            bundle.item_id,
                            bundle.miner_hotkey,
                            bundle.score_packet.digest,
                            bundle.manifest.digest,
                            bundle.commitment_hash,
                        )
                        if actual_bundle_identity != expected_bundle_identity:
                            raise ValueError(
                                f"competition bundle identity {actual_bundle_identity!r} "
                                f"differs from committed item/subject identity "
                                f"{expected_bundle_identity!r}"
                            )
                        if comp_input.track == "upscaling":
                            binding = bundle.competition_item
                            expected_binding = (
                                item.item_index,
                                item.input_sha256,
                                item.reference_sha256,
                                item.upscale_factor,
                                item.target_width,
                                item.target_height,
                                item.item_commitment,
                            )
                            actual_binding = (
                                None
                                if binding is None
                                else (
                                    binding.item_index,
                                    binding.input_sha256,
                                    binding.reference_sha256,
                                    binding.upscale_factor,
                                    binding.target_width,
                                    binding.target_height,
                                    binding.item_commitment,
                                )
                            )
                            if (
                                bundle.stage.value != "competition_sealed"
                                or actual_binding != expected_binding
                            ):
                                raise ValueError(
                                    "upscaling competition bundle does not bind the "
                                    "committed evaluation-item preimage"
                                )
                        if bundle.execution_image_digest is None:
                            raise ValueError(
                                "competition bundle does not bind its execution image"
                            )
                        image_digests.add(bundle.execution_image_digest)
                        output_digests.append(bundle.miner_output.digest)
                        raw = store.get_limited(
                            bundle.score_packet, _MAX_AUDIT_METADATA_BYTES
                        )
                        payload = json.loads(raw)
                        if not isinstance(payload, dict):
                            raise ValueError(
                                "competition score packet is not a JSON object"
                            )
                        score = payload.get("score")
                        if (
                            not isinstance(score, numbers.Real)
                            or isinstance(score, bool)
                            or not math.isfinite(float(score))
                            or not 0.0 <= float(score) <= 1.0
                        ):
                            raise ValueError(
                                "competition packet score is not finite in [0,1]"
                            )
                        if (
                            payload.get("challenge_id") != packet_ref.challenge_id
                            or payload.get("item_id") != packet_ref.item_id
                            or payload.get("track") != comp_input.track
                        ):
                            raise ValueError(
                                "competition packet identity/track differs from committed refs"
                            )
                        if payload.get("miner_hotkey") != expected_hotkey:
                            raise ValueError(
                                "competition packet miner identity differs from audit subject"
                            )
                        if payload.get("content_digest") != bundle.miner_output.digest:
                            raise ValueError(
                                "competition packet content digest differs from bundle "
                                "miner output"
                            )
                        packet_scores[packet_ref.digest] = float(score)
                    if len(image_digests) != 1:
                        raise ValueError(
                            f"competition subject {subject.subject_id!r} does not use "
                            "one stable execution image across its full matrix"
                        )
                    if next(iter(image_digests)) != subject.execution_image_digest:
                        raise ValueError(
                            f"competition subject {subject.subject_id!r} bundle execution "
                            "image differs from its committed subject image"
                        )
                    if (
                        subject.role == "baseline"
                        and next(iter(image_digests))
                        != commitment.baseline_image_digest
                    ):
                        raise ValueError(
                            "competition baseline execution image differs from the "
                            "pre-enrollment commitment"
                        )
                    if subject.role == "contender":
                        census = census_by_hotkey.get(subject.hotkey or "")
                        if (
                            census is None
                            or census.uid != subject.uid
                            or census.hotkey != subject.hotkey
                        ):
                            raise ValueError(
                                f"competition contender {subject.subject_id!r} does not "
                                "match its close-block census uid/hotkey"
                            )
                        dedup_candidates.append(
                            CompetitionDedupCandidate(
                                subject_id=subject.subject_id,
                                uid=census.uid,
                                coldkey=census.coldkey,
                                ip=census.ip,
                                output_digests=tuple(output_digests),
                            )
                        )

                dedup_losers = competition_dedup_losers(tuple(dedup_candidates))
                committed_losers = frozenset(
                    subject.subject_id
                    for subject in comp_input.subjects
                    if subject.role == "contender" and subject.dedup_excluded
                )
                if dedup_losers != committed_losers:
                    raise ValueError(
                        "competition dedup exclusions differ from the independent "
                        "close-census/coldkey/IP/exact-output derivation: "
                        f"expected {sorted(dedup_losers)}, committed "
                        f"{sorted(committed_losers)}"
                    )
            except (
                BundleUnavailable,
                CompetitionAnchorUnavailable,
                FileNotFoundError,
                OSError,
                IntegrityError,
            ) as exc:
                unresolved = verdict(
                    ItemVerdictKind.SKIP,
                    COMPETITION_UNVERIFIED,
                    f"competition commitment/manifest/packet evidence is unavailable: {exc}",
                )
                return (
                    log.competition_result,
                    log.reward_window_state,
                    (unresolved,),
                )
            except (
                CompetitionAnchorMismatch,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                failed = verdict(
                    ItemVerdictKind.FAIL,
                    COMPETITION_MISMATCH,
                    "competition commitment/manifest/packet evidence is malformed or "
                    f"inconsistent: {exc}",
                )
                return None, RewardWindowState(), (failed,)
            try:
                derivation = derive_competition_economics(comp_input, packet_scores)
                derived_result = derivation.result
            except CompetitionEconomicResultError as exc:
                failed = verdict(
                    ItemVerdictKind.FAIL,
                    COMPETITION_MISMATCH,
                    str(exc),
                )
                return None, RewardWindowState(), (failed,)
            if derived_result != log.competition_result:
                failed = verdict(
                    ItemVerdictKind.FAIL,
                    COMPETITION_MISMATCH,
                    "published competition result does not equal the exact committed-packet "
                    "mean/score/ranking/provenance derivation",
                )
                result_verdicts: tuple[ItemVerdict, ...] = (failed,)
            else:
                result_verdicts = (verdict(ItemVerdictKind.PASS),)
        else:
            derived_result = None
            if log.competition_result is not None:
                result_verdicts = (
                    verdict(
                        ItemVerdictKind.FAIL,
                        COMPETITION_MISMATCH,
                        "competition result is present without committed packet evidence",
                    ),
                )
            else:
                result_verdicts = ()

        if prior_log is not None and log.prior_log_digest == prior_log.log_digest():
            prior_reward_window = prior_log.reward_window_state
        elif is_genesis and log.prior_log_digest is None:
            prior_reward_window = RewardWindowState()
        elif log.prior_log_digest is not None and prior_log is None:
            return (
                derived_result,
                log.reward_window_state,
                result_verdicts
                + (
                    verdict(
                        ItemVerdictKind.SKIP,
                        COMPETITION_UNVERIFIED,
                        "prior epoch is unavailable, so reward-window carry-in cannot "
                        "be verified",
                        item_id="reward-window-state",
                    ),
                ),
            )
        else:
            prior_reward_window = RewardWindowState()
        try:
            derived_reward_window = resolve_reward_window(
                self._config.tokenomics,
                prior_reward_window,
                derived_result,
            )
        except (TypeError, ValueError) as exc:
            failed = verdict(
                ItemVerdictKind.FAIL,
                REWARD_WINDOW_MISMATCH,
                "reward-window derivation rejected the predecessor/result chronology: "
                f"{exc}",
                item_id="reward-window-state",
            )
            return None, RewardWindowState(), result_verdicts + (failed,)
        release_verdicts: tuple[ItemVerdict, ...] = ()
        if (
            derived_reward_window.kind is EmissionState.CROWN
            and derived_result is not None
            and derived_result.contenders
        ):
            winner = derived_result.contenders[0]
            winning_subject = next(
                (
                    subject
                    for subject in comp_input.subjects
                    if subject.role == "contender"
                    and subject.uid == winner.uid
                    and subject.hotkey == winner.hotkey
                ),
                None,
            ) if comp_input is not None else None
            if (
                winning_subject is None
                or winning_subject.submission_archive_digest is None
                or winning_subject.submission_archive_bytes is None
                or winning_subject.submission_archive_bytes <= 0
            ):
                release_verdicts = (
                    verdict(
                        ItemVerdictKind.FAIL,
                        COMPETITION_MISMATCH,
                        "CROWN winner does not carry a positive, content-addressed "
                        "submission archive commitment",
                        item_id="winner-submission-archive",
                    ),
                )
            else:
                try:
                    archive = store.get_digest_limited(
                        ArtifactKind.SUBMISSION_ARCHIVE,
                        winning_subject.submission_archive_digest,
                        max_bytes=winning_subject.submission_archive_bytes,
                    )
                    if len(archive) != winning_subject.submission_archive_bytes:
                        raise IntegrityError(
                            "released winner submission size differs from its commitment"
                        )
                except IntegrityError as exc:
                    release_verdicts = (
                        verdict(
                            ItemVerdictKind.FAIL,
                            COMPETITION_MISMATCH,
                            f"released CROWN winner archive fails integrity: {exc}",
                            item_id="winner-submission-archive",
                        ),
                    )
                except (FileNotFoundError, OSError, ValueError) as exc:
                    release_verdicts = (
                        verdict(
                            ItemVerdictKind.SKIP,
                            COMPETITION_UNVERIFIED,
                            "released CROWN winner archive is not publicly readable: "
                            f"{type(exc).__name__}: {exc}",
                            item_id="winner-submission-archive",
                        ),
                    )
        reward_window_verdict = verdict(
            (
                ItemVerdictKind.PASS
                if derived_reward_window == log.reward_window_state
                else ItemVerdictKind.FAIL
            ),
            (
                ""
                if derived_reward_window == log.reward_window_state
                else REWARD_WINDOW_MISMATCH
            ),
            (
                ""
                if derived_reward_window == log.reward_window_state
                else "published reward-window state does not equal the exact "
                "predecessor+result fold"
            ),
            item_id="reward-window-state",
        )
        return (
            derived_result,
            derived_reward_window,
            result_verdicts + release_verdicts + (reward_window_verdict,),
        )

    # -- weight re-derivation -------------------------------------------------------

    def _weight_verdict(
        self,
        log: EpochLog,
        metagraph: dict[int, object] | None,
        snapshot_verdicts: tuple[ItemVerdict, ...],
        *,
        reward_window_state: RewardWindowState,
    ) -> WeightVerdict:
        """Re-derive the log's weight vector over INDEPENDENTLY-VERIFIED inputs.

        Cheap and media-free: ``build_weight_vector`` + ``quantize_u16`` over a
        RECONSTRUCTED miner set whose identity (uid->hotkey/coldkey/ip), IP/coldkey DEDUP
        (`excluded`) and TRACK are the close-block METAGRAPH / committed values just
        verified — NOT the authority's raw ``log.miners`` — with
        ``accumulate_score`` from the earning fold. (The windowed-input re-derivation was
        REMOVED with the retention multiplier for v1 — retention removed — owner decision;
        an internal review.) Fail-closed, in precedence order:

        - a provable SNAPSHOT mismatch (identity/dedup/track) ⇒ FAIL with that code
          (reflected here so it lands in the Audit Results API disputed-items channel);
        - an UNVERIFIABLE snapshot input (SNAPSHOT_UNVERIFIED) ⇒ SKIP ⇒ INCONCLUSIVE,
          NEVER a PASS;
        - otherwise the reconstructed digest is compared to the published one: a raise or
          a mismatch is WEIGHT_DERIVATION_MISMATCH (a substituted weight — caught pre-media).

        Schema v15 also supplies the independently predecessor/result-derived reward
        window to the same pure composer. Window activity is evaluated at the log's
        independently chain-bound ``created_at``.
        """
        published = log.weight_vector_digest
        # A provable snapshot mismatch is a weight-INPUT fault — surface its specific code
        # (IDENTITY_MISMATCH / METAGRAPH_DEDUP_MISMATCH / METAGRAPH_TRACK_MISMATCH).
        fails = [v for v in snapshot_verdicts if v.verdict is ItemVerdictKind.FAIL]
        if fails:
            return WeightVerdict(
                recomputed_weight_vector_digest="",
                published_weight_vector_digest=published,
                verdict=ItemVerdictKind.FAIL,
                code=fails[0].code,
                detail=fails[0].detail,
            )
        # An unverifiable snapshot input is UNVERIFIABLE weight — HOLD, never PASS.
        snap_skips = [v for v in snapshot_verdicts if v.verdict is ItemVerdictKind.SKIP]
        if snap_skips:
            return WeightVerdict(
                recomputed_weight_vector_digest="",
                published_weight_vector_digest=published,
                verdict=ItemVerdictKind.SKIP,
                code=SNAPSHOT_UNVERIFIED,
                detail=(
                    "the SNAPSHOT-derivable weight inputs (metagraph identity/dedup/track) "
                    f"could not be independently verified: {snap_skips[0].detail}"
                ),
            )
        miners = self._reconstructed_miners(log, metagraph)
        if miners is None:
            # No nonzero-weight uid was verifiable enough to reconstruct (metagraph
            # unavailable with nonzero uids). Fail closed rather than derive over raw log.
            return WeightVerdict(
                recomputed_weight_vector_digest="",
                published_weight_vector_digest=published,
                verdict=ItemVerdictKind.SKIP,
                code=SNAPSHOT_UNVERIFIED,
                detail=(
                    "the close-block metagraph is unavailable, so the weight vector cannot "
                    "be re-derived over independently-verified snapshot inputs — INCONCLUSIVE"
                ),
            )
        try:
            shares = build_weight_vector(
                self._config.tokenomics,
                miners,
                burn_uid=log.burn_uid,
                reward_state=reward_window_state,
                now=log.created_at,
            )
            recomputed_u16 = quantize_u16(shares)
            recomputed_digest = weight_vector_digest(recomputed_u16)
        except Exception as exc:  # a derivation that cannot even run is a fault
            return WeightVerdict(
                recomputed_weight_vector_digest="",
                published_weight_vector_digest=published,
                verdict=ItemVerdictKind.FAIL,
                code=WEIGHT_DERIVATION_MISMATCH,
                detail=f"weight re-derivation raised {type(exc).__name__}: {exc}",
            )

        if recomputed_u16 == log.weight_u16 and recomputed_digest == published:
            return WeightVerdict(
                recomputed_weight_vector_digest=recomputed_digest,
                published_weight_vector_digest=published,
                verdict=ItemVerdictKind.PASS,
            )
        return WeightVerdict(
            recomputed_weight_vector_digest=recomputed_digest,
            published_weight_vector_digest=published,
            verdict=ItemVerdictKind.FAIL,
            code=WEIGHT_DERIVATION_MISMATCH,
            detail=(
                "the published weight vector does not follow from the independently-verified "
                "inputs: build_weight_vector+quantize_u16 yields "
                f"{recomputed_digest}, log published {published}"
            ),
        )

    # -- close-block metagraph SNAPSHOT binding ------------------

    def _read_metagraph(self, close_block: int) -> dict[int, object] | None:
        """Read the close-block metagraph via our OWN chain adapter.

        Production adapters expose ``neurons_at(close_block)`` and MUST use that exact
        historical view; head churn after finalization cannot relabel the log's census.
        Report/in-memory adapters without the optional historical seam use their deterministic
        snapshot via ``neurons()``. Returns ``uid -> ChainNeuron`` or None when no metagraph can
        be read: the binding then fails CLOSED (INCONCLUSIVE), never a PASS on the authority's
        self-attested identities.
        """
        chain = self._chain
        if chain is None:
            return None
        try:
            historical = getattr(chain, "neurons_at", None)
            neurons = (
                historical(close_block) if callable(historical) else chain.neurons()
            )
        except Exception:
            return None  # unreadable/unavailable metagraph ⇒ fail closed (INCONCLUSIVE)
        return {int(n.uid): n for n in neurons}

    def _snapshot_verdicts(
        self, log: EpochLog, metagraph: dict[int, object] | None
    ) -> list[ItemVerdict]:
        """Bind each SNAPSHOT-derivable weight input to the close-block metagraph.

        For every NONZERO-weight uid (other than the empty-epoch burn uid): verify its
        uid->hotkey/coldkey/ip identity, its committed TRACK, and its IP/coldkey DEDUP
        (`excluded`) outcome against the metagraph the auditor reads ITSELF. The DEDUP is
        additionally checked for ZERO-weight miners — a real collision the authority did
        NOT exclude gets ZERO weight (so it is not "nonzero"), yet its `excluded` flag must
        still bind, or the mis-declaration would slip through. Fail-closed:

        - relabel / uid absent from metagraph ⇒ IDENTITY_MISMATCH (FAIL ⇒ DISPUTED);
        - `excluded` disagrees with the re-derived dedup ⇒ METAGRAPH_DEDUP_MISMATCH;
        - `track` != committed track ⇒ METAGRAPH_TRACK_MISMATCH;
        - metagraph unavailable / track unresolvable ⇒ SNAPSHOT_UNVERIFIED (SKIP ⇒
          INCONCLUSIVE), NEVER a PASS.
        """
        miners_by_uid = {m.uid: m for m in log.miners}
        nonzero = sorted(
            uid
            for uid, w in log.weight_shares.items()
            if w > 0.0 and uid != log.burn_uid
        )
        nonzero_set = set(nonzero)
        # an internal review: even a BURN-ONLY vector ({burn_uid: 1.0}, NO nonzero uid) must still
        # bind the ZERO-weight census miners' identity + dedup. Otherwise an untrusted authority
        # can seat evidence-backed miners, mark every distinct one `excluded=True` (or spoof a
        # zero-weight identity) to SUBSTITUTE a 100%-burn vector, and slip through — the old
        # `if not nonzero: return []` early-out skipped the zero-weight pass entirely. So the
        # bindable set is EVERY non-burn census miner (nonzero + zero-weight); we only skip when
        # there is genuinely nothing to bind (an empty census).
        zero_census = [
            m for m in log.miners if m.uid not in nonzero_set and m.uid != log.burn_uid
        ]
        if not nonzero and not zero_census:
            return []  # empty epoch: nothing snapshot-derivable to bind
        if metagraph is None:
            # Fail closed for EVERYTHING bindable (nonzero AND zero-weight census) — including a
            # burn-only epoch: a substituted burn must not wash to CLEAN when the metagraph that
            # would expose it cannot be read.
            bindable = list(nonzero) + [m.uid for m in zero_census]
            return [
                self._snapshot_verdict(
                    uid,
                    miners_by_uid.get(uid),
                    ItemVerdictKind.SKIP,
                    SNAPSHOT_UNVERIFIED,
                    "the close-block metagraph is unavailable (chain read failed / no "
                    "adapter wired) — the uid's identity, IP/coldkey dedup and track "
                    "cannot be bound; failing closed to INCONCLUSIVE, never a PASS on the "
                    "authority's self-attested snapshot",
                )
                for uid in bindable
            ]
        excluded_set = self._recompute_excluded(log, metagraph)
        verdicts = [
            self._snapshot_verdict_for_uid(
                uid, miners_by_uid.get(uid), log, metagraph, excluded_set
            )
            for uid in nonzero
        ]
        # IDENTITY + dedup + TRACK binding for ZERO-weight census miners (an internal review,
        # extended round-17 #2/#3). A zero-weight record earns nothing THIS epoch and so never
        # appears in `nonzero` above, but the untrusted authority must not seat a census entry
        # that does not match the close-block metagraph / its own committed evidence — a tampered
        # identity/track on a zero-weight, evidence-carrying record is an unverified entry that
        # can SUBSTITUTE a burn (relabel evidenced miners → all excluded / non-paying track →
        # {burn_uid:1.0}) or later become load-bearing (re-attribute a carry-in). So bind, exactly
        # as the nonzero `_snapshot_verdict_for_uid` does:
        #
        # - an internal review: a census miner carrying COMMITTED evidence but whose uid is
        #   ABSENT from the close-block metagraph is an unbindable/relabelled identity ⇒ FAIL
        #   IDENTITY_MISMATCH (was a silent skip — the relabel-to-absent-uid burn hole). A
        #   zero-weight NO-evidence miner absent from the metagraph stays a benign skip (it
        #   cannot be bound and carries no evidence to relabel);
        # - a real uid with mismatched uid->hotkey/coldkey/ip ⇒ FAIL IDENTITY_MISMATCH;
        # - an internal review: an EVIDENCED miner whose `track` != its committed track ⇒ FAIL
        #   METAGRAPH_TRACK_MISMATCH (the tampered-track burn hole); an unresolvable committed
        #   track ⇒ SKIP SNAPSHOT_UNVERIFIED (never a PASS on the log-stated track). An honest
        #   below-cutoff evidenced loser has a resolvable, matching track ⇒ no verdict (CLEAN);
        # - the `excluded` dedup flag, unchanged.
        evidenced = self._evidenced_uids(log)
        for m in log.miners:
            if m.uid in nonzero_set or m.uid == log.burn_uid:
                continue
            if m.uid not in metagraph:
                if m.uid in evidenced:
                    verdicts.append(
                        self._snapshot_verdict(
                            m.uid,
                            m,
                            ItemVerdictKind.FAIL,
                            IDENTITY_MISMATCH,
                            f"zero-weight uid {m.uid} (hotkey {m.hotkey!r}) carries COMMITTED "
                            "earning evidence but is ABSENT from the close-block metagraph — its "
                            "identity cannot be bound (an evidenced miner RELABELLED to a uid not "
                            "in the metagraph, e.g. to substitute a burn, an internal review)",
                        )
                    )
                elif not is_excluded(m.accumulate_score) and m.accumulate_score > 0.0:
                    # an internal review: a POSITIVE IMPLICIT carry-forward miner (positive
                    # accumulator, NO current committed evidence) is ALSO bindable EVIDENCE for
                    # snapshot purposes — a real prior earner carried forward. The round-17 #2
                    # absent-uid FAIL only covered CURRENT-evidence miners (`evidenced`), so a
                    # positive implicit carry with a correct prior value/hotkey/track but a
                    # TAMPERED `excluded=True`, ABSENT from the close-block metagraph, was a benign
                    # skip: its identity/dedup went unbound and the burn-only reconstruct
                    # preserved the tampered exclusion ⇒ substituted burn CLEAN. A positive
                    # carry-forward miner that cannot be bound to the metagraph (can't verify
                    # identity/dedup, can't re-derive `excluded`) FAILs CLOSED — IDENTITY_MISMATCH
                    # (⇒ DISPUTED), never a wash on the authority's self-attested exclusion.
                    verdicts.append(
                        self._snapshot_verdict(
                            m.uid,
                            m,
                            ItemVerdictKind.FAIL,
                            IDENTITY_MISMATCH,
                            f"zero-weight uid {m.uid} (hotkey {m.hotkey!r}) carries a POSITIVE "
                            f"implicit carry-forward accumulator {m.accumulate_score} but is ABSENT "
                            "from the close-block metagraph — a positive prior earner is bindable "
                            "evidence; its identity/dedup (`excluded`) cannot be bound, so it fails "
                            "closed rather than let a tampered exclusion substitute a burn (round-19 #3)",
                        )
                    )
                # zero-weight, NO evidence, NO positive carry, absent: benign, cannot bind.
                continue
            neuron = metagraph[m.uid]
            if (
                neuron.hotkey != m.hotkey
                or neuron.coldkey != m.coldkey
                or neuron.ip != m.ip
            ):
                verdicts.append(
                    self._snapshot_verdict(
                        m.uid,
                        m,
                        ItemVerdictKind.FAIL,
                        IDENTITY_MISMATCH,
                        f"zero-weight uid {m.uid} identity does not match the close-block "
                        f"metagraph: log states {m.hotkey!r}/{m.coldkey!r}/{m.ip!r} but the "
                        f"metagraph binds {neuron.hotkey!r}/{neuron.coldkey!r}/{neuron.ip!r} — "
                        "a tampered identity seated in the census",
                    )
                )
                continue
            # TRACK binding for EVIDENCED zero-weight miners: compare the
            # log-stated track to the committed track from evidence, exactly as the nonzero
            # path does. Only for evidenced miners — a zero-weight miner with NO committed
            # track (e.g. a dedup-excluded / unscored record) has none to bind, and forcing a
            # SKIP there would false-INCONCLUSIVE honest epochs.
            if m.uid in evidenced:
                committed_track = self._committed_track_for_uid(log, m.uid)
                if committed_track is None:
                    competition_only = (
                        not log.audit_manifest.refs_for(m.uid)
                        and log.audit_manifest.competition_input is not None
                        and any(
                            subject.role == "contender" and subject.uid == m.uid
                            for subject in log.audit_manifest.competition_input.subjects
                        )
                    )
                    if competition_only:
                        expected = m.uid in excluded_set
                        if expected != bool(m.excluded):
                            verdicts.append(self._dedup_verdict(m, expected))
                        continue
                    verdicts.append(
                        self._snapshot_verdict(
                            m.uid,
                            m,
                            ItemVerdictKind.SKIP,
                            SNAPSHOT_UNVERIFIED,
                            f"zero-weight uid {m.uid} is evidenced but its track cannot be "
                            "resolved from committed earning evidence — UNVERIFIED "
                            "(INCONCLUSIVE), never a PASS on the log-stated track (round-17 #3)",
                        )
                    )
                    continue
                if m.track != committed_track:
                    verdicts.append(
                        self._snapshot_verdict(
                            m.uid,
                            m,
                            ItemVerdictKind.FAIL,
                            METAGRAPH_TRACK_MISMATCH,
                            f"zero-weight uid {m.uid} declares scoring track {m.track!r} but its "
                            f"committed earning evidence is track {committed_track!r} — a "
                            "mis-declared track (e.g. a non-paying track that substitutes a burn "
                            "for an evidence-backed positive-score miner, an internal review)",
                        )
                    )
                    continue
            expected = m.uid in excluded_set
            if expected != bool(m.excluded):
                verdicts.append(self._dedup_verdict(m, expected))
        return verdicts

    def _snapshot_verdict_for_uid(
        self,
        uid: int,
        miner,
        log: EpochLog,
        metagraph: dict[int, object],
        excluded_set: set[int],
    ) -> ItemVerdict:
        if miner is None:
            # Nonzero weight but no MinerSnapshot at all — the earning path FAILs this too;
            # bind it here as an identity fault so the weight verdict reflects it directly.
            return self._snapshot_verdict(
                uid,
                None,
                ItemVerdictKind.FAIL,
                IDENTITY_MISMATCH,
                f"uid {uid} has nonzero weight but no miner snapshot to bind to the "
                "close-block metagraph",
            )
        neuron = metagraph.get(uid)
        if neuron is None:
            return self._snapshot_verdict(
                uid,
                miner,
                ItemVerdictKind.FAIL,
                IDENTITY_MISMATCH,
                f"uid {uid} (hotkey {miner.hotkey!r}) has nonzero weight but is ABSENT "
                "from the close-block metagraph — its identity cannot be bound",
            )
        if (
            neuron.hotkey != miner.hotkey
            or neuron.coldkey != miner.coldkey
            or neuron.ip != miner.ip
        ):
            return self._snapshot_verdict(
                uid,
                miner,
                ItemVerdictKind.FAIL,
                IDENTITY_MISMATCH,
                f"uid {uid} identity does not match the close-block metagraph: log states "
                f"hotkey/coldkey/ip {miner.hotkey!r}/{miner.coldkey!r}/{miner.ip!r} but the "
                f"metagraph binds {neuron.hotkey!r}/{neuron.coldkey!r}/{neuron.ip!r} — a "
                "relabelled/tampered identity",
            )
        committed_track = self._committed_track_for_uid(log, uid)
        if committed_track is None:
            # an internal review: a nonzero-weight uid with NO current earning evidence (no
            # manifest refs, no earning input) is a pure CARRY-FORWARD — an idle prior earner.
            # It has no CURRENT committed track to resolve, so its track is bound by the earning
            # CARRY-FORWARD path instead (`_carry_forward_verdict`, round-18 #1: the track must
            # CHAIN to the prior epoch's snapshot for this (uid, hotkey)). Identity + dedup above
            # are already bound to the metagraph, so defer the track to that chained check and
            # PASS here rather than SKIP (which would falsely HOLD every idle earner). An
            # EVIDENCED uid whose track is nonetheless unresolvable stays a red-flag SKIP.
            if (
                not log.audit_manifest.refs_for(uid)
                and uid not in log.audit_manifest.earning_inputs
            ):
                expected_excluded = uid in excluded_set
                if expected_excluded != bool(miner.excluded):
                    return self._dedup_verdict(miner, expected_excluded)
                return self._snapshot_verdict(uid, miner, ItemVerdictKind.PASS, "", "")
            return self._snapshot_verdict(
                uid,
                miner,
                ItemVerdictKind.SKIP,
                SNAPSHOT_UNVERIFIED,
                f"uid {uid}'s track cannot be resolved from committed earning evidence — "
                "UNVERIFIED (INCONCLUSIVE), never a PASS on the log-stated track",
            )
        if miner.track != committed_track:
            return self._snapshot_verdict(
                uid,
                miner,
                ItemVerdictKind.FAIL,
                METAGRAPH_TRACK_MISMATCH,
                f"uid {uid} declares scoring track {miner.track!r} but its committed "
                f"earning evidence is track {committed_track!r} — a mis-declared track",
            )
        expected_excluded = uid in excluded_set
        if expected_excluded != bool(miner.excluded):
            return self._dedup_verdict(miner, expected_excluded)
        return self._snapshot_verdict(uid, miner, ItemVerdictKind.PASS, "", "")

    def _dedup_verdict(self, miner, expected_excluded: bool) -> ItemVerdict:
        if expected_excluded:
            detail = (
                f"uid {miner.uid} (ip {miner.ip!r}, coldkey {miner.coldkey!r}) is a real "
                "IP/coldkey dedup COLLISION with a lower uid but the log did NOT exclude it "
                f"(excluded={miner.excluded!r}) — a mis-declared dedup outcome"
            )
        else:
            detail = (
                f"uid {miner.uid} (ip {miner.ip!r}, coldkey {miner.coldkey!r}) is a DISTINCT "
                "miner (no IP/coldkey collision in the metagraph) but the log excluded it "
                f"(excluded={miner.excluded!r}) — a wrongly-excluded distinct miner"
            )
        return self._snapshot_verdict(
            miner.uid, miner, ItemVerdictKind.FAIL, METAGRAPH_DEDUP_MISMATCH, detail
        )

    @staticmethod
    def _snapshot_verdict(
        uid: int, miner, verdict: ItemVerdictKind, code: str, detail: str
    ) -> ItemVerdict:
        return ItemVerdict(
            source=_SNAPSHOT_SOURCE,
            challenge_id="",
            item_id=f"snapshot:{uid}",
            miner_hotkey=(miner.hotkey if miner is not None else None),
            uid=uid,
            bundle_digest="",
            packet_digest="",
            verdict=verdict,
            code=code,
            detail=detail,
        )

    # -- committed-evidence set (shared by census + zero-weight snapshot binding) ----

    def _evidenced_uids(self, log: EpochLog) -> set[int]:
        """Uids the manifest carries committed EARNING evidence for (an internal review).

        The UNION of ``earning_inputs`` and any uid with a committed SCORE_PACKET ref — so
        dropping either half cannot hide a miner. Shared by the census cross-check (an
        evidenced-but-omitted miner is censorship) and the zero-weight snapshot binding (an
        evidenced miner ABSENT from the metagraph, or on a tampered track, is unbindable).
        """
        evidenced: set[int] = set(log.audit_manifest.earning_inputs)
        evidenced.update(
            evidence.uid for evidence in log.audit_manifest.availability_inputs
        )
        for uid, refs in log.audit_manifest.per_uid.items():
            if any(r.kind is AuditFileKind.SCORE_PACKET for r in refs):
                evidenced.add(uid)
        if log.audit_manifest.competition_input is not None:
            evidenced.update(
                subject.uid
                for subject in log.audit_manifest.competition_input.subjects
                if subject.role == "contender" and subject.uid is not None
            )
        return evidenced

    # -- canonical burn-recipient binding -----------------------

    def _burn_uid_verdict(self, log: EpochLog) -> ItemVerdict | None:
        """The empty-epoch burn recipient must be the CANONICAL burn uid.

        ``EpochLog._validate`` only requires ``burn_uid`` to be the SOLE positive-weight uid,
        so it lets the UNTRUSTED authority CHOOSE the recipient: an authority could anchor an
        empty log with ``burn_uid=<a registered beneficiary it controls>`` + ``{beneficiary:
        1.0}`` and — with every real miner censored (the census check catches THAT) OR a
        genuinely-empty epoch — burn 100% of the epoch's emission to itself while auditing
        CLEAN. So the burn recipient is resolved HERE, INDEPENDENTLY of the log, from
        ``AuditorConfig.burn_uid`` (the SAME canonical value the Scoring Authority is
        configured with, ``authority.burn_uid``). A ``log.burn_uid`` that is not the canonical
        value is a provable fault ⇒ FAIL (BURN_UID_MISMATCH ⇒ DISPUTED).

        an internal review (defense in depth): additionally, the reserved burn uid must NOT double
        as a census/evidence identity — a burn epoch whose ``burn_uid`` appears in ``log.miners``
        OR carries earning/manifest evidence is a provable fault ⇒ FAIL (even if the recipient is
        canonical). ``EpochLog._validate`` refuses such a log, so this catches only bytes that
        bypassed the finalizer/validation.

        Returns None on a normal epoch (``burn_uid`` None — no burn recipient to canonicalize)
        or when the declared recipient IS canonical AND does not double as a census/evidence
        identity (an honest empty-epoch burn stays CLEAN, the project design record rule 11). Rides the
        census/earning dispute channel.
        """
        if log.burn_uid is None:
            return None
        # an internal review: DEFENSE IN DEPTH — the reserved burn uid must NOT double as a
        # census/evidence identity. `EpochLog._validate` refuses such a log, but a tampered log
        # that BYPASSED the finalizer/validation could seat the burn uid in `miners` (excluded
        # from the snapshot/identity/dedup/track pass) carrying another miner's evidence (excluded from
        # the earning fold) and publish `{burn_uid: 1.0}` — CLEAN, because no binding ever runs
        # for the reserved uid. Catch it here regardless of whether the recipient is canonical: a
        # burn epoch whose burn_uid appears in the census OR carries earning/manifest evidence is
        # a provable fault ⇒ DISPUTED (rides the census/burn channel).
        seated = any(m.uid == log.burn_uid for m in log.miners)
        has_evidence = (
            log.burn_uid in self._evidenced_uids(log)
            or log.burn_uid in log.audit_manifest.earning_inputs
        )
        if seated or has_evidence:
            return ItemVerdict(
                source=_CENSUS_SOURCE,
                challenge_id="",
                item_id=f"burn:{log.burn_uid}",
                miner_hotkey=None,
                uid=log.burn_uid,
                bundle_digest="",
                packet_digest="",
                verdict=ItemVerdictKind.FAIL,
                code=BURN_UID_MISMATCH,
                detail=(
                    f"the reserved burn uid {log.burn_uid} ALSO appears as a census miner "
                    f"(seated={seated}) / carries earning evidence (evidence={has_evidence}) — "
                    "the reserved empty-epoch burn uid must not double as an evidence/census "
                    "identity; the auditor never folds or binds it, so an untrusted log could "
                    "re-attribute another miner's evidence under it and burn CLEAN"
                ),
            )
        try:
            canonical = resolve_burn_uid(
                self._chain, report_fallback=self._config.burn_uid
            )
        except Exception as exc:
            return ItemVerdict(
                source=_CENSUS_SOURCE,
                challenge_id="",
                item_id=f"burn:{log.burn_uid}",
                miner_hotkey=None,
                uid=log.burn_uid,
                bundle_digest="",
                packet_digest="",
                verdict=ItemVerdictKind.SKIP,
                code=BURN_UID_UNVERIFIED,
                detail=(
                    "the auditor could not resolve the subnet-owner burn uid from its "
                    "independent chain connection, so this empty epoch cannot be marked "
                    f"CLEAN: {type(exc).__name__}: {exc}"
                ),
            )
        if log.burn_uid == canonical:
            return None
        return ItemVerdict(
            source=_CENSUS_SOURCE,
            challenge_id="",
            item_id=f"burn:{log.burn_uid}",
            miner_hotkey=None,
            uid=log.burn_uid,
            bundle_digest="",
            packet_digest="",
            verdict=ItemVerdictKind.FAIL,
            code=BURN_UID_MISMATCH,
            detail=(
                f"the log burns 100% to uid {log.burn_uid}, but the CANONICAL burn recipient "
                f"(resolved independently of the log, from chain state) is uid {canonical} — an "
                "untrusted authority does not get to CHOOSE the burn recipient; a substituted "
                "burn to a non-canonical uid is DISPUTED"
            ),
        )

    # -- miner census vs committed evidence cross-check -----------

    def _census_verdicts(
        self, log: EpochLog, metagraph: dict[int, object] | None
    ) -> list[ItemVerdict]:
        """Bind registration census, economic miners, evidence, and close-block metagraph.

        The snapshot / earning / competition re-derivations are scoped to the POSITIVE-weight
        set, so an empty/burn positive set (``miners=[], {burn_uid:1.0}``) bypasses all of them.
        This closes that census-censorship hole with a check that needs NO
        metagraph — it is provable from the log's own bytes:

        For every uid the manifest carries committed EARNING evidence for (an ``EarningInput`` or
        a SCORE_PACKET ref) OR committed COMPETITION evidence for (a contender):

        - OMITTED from ``log.miners`` ⇒ FAIL (CENSUS_MISMATCH ⇒ DISPUTED): the authority stored
          the miner's evidence then censored it from the census (evidence present, miner not in
          the census AT ALL);
        - PRESENT in ``log.miners`` ⇒ PASS. Genuine censorship is OMISSION; a miner that IS in the
          census with zero weight is NOT censored — it is either a legitimate below-cutoff RANK
          LOSER or a DEDUP exclusion, both of which are bound elsewhere and must NOT surface here
. Distinguishing the two:
            * below-cutoff RANK LOSER: ``build_weight_vector`` → ``track_shares`` pays only the
              top ``top_n_per_track`` by score and zeroes rank N+1+ (rank_curve.py). The finalizer
              carries earning evidence for EVERY scored miner (build_audit_manifest keys off the
              SCORED items, not the final weight), so an honest epoch with MORE miners than
              ``top_n`` has evidenced, present, ZERO-weight, non-excluded losers — HONEST, CLEAN.
              (The prior round-9 #1 rule FAILed exactly these ⇒ a false-DISPUTE regression.)
            * DEDUP exclusion (``excluded`` True): the exclusion's LEGITIMACY is bound to the
              close-block metagraph by ``_snapshot_verdicts`` — a wrongly-excluded distinct miner
              is a METAGRAPH_DEDUP_MISMATCH there (DISPUTED), and an UNREADABLE metagraph makes it
              SNAPSHOT_UNVERIFIED (INCONCLUSIVE), never a wash to CLEAN. So the "evidenced
              exclusion" is verified on the dedup channel, not re-litigated (and cannot be) from
              the census's metagraph-free bytes; census only proves the miner is IN the census.

        NO committed evidence at all may still be a genuinely-empty economic epoch → burn, while
        ``miner_census`` independently carries every registered payout identity, including
        offline/new/unknown-track registrations that cannot enter tokenomics. The census must
        exactly match the close-block metagraph; ``miners`` must be an identity-matching subset.
        An unavailable metagraph is SKIP / INCONCLUSIVE even when both collections are empty.
        """
        miners_by_uid = {m.uid: m for m in log.miners}
        census_by_uid = {entry.uid: entry for entry in log.miner_census}
        verdicts: list[ItemVerdict] = []

        if metagraph is None:
            # Keep evaluating the self-contained evidence/economic-subset proofs below. An
            # outage makes exact registration membership unverifiable, but cannot erase a
            # conclusive inconsistency already present in the anchored log.
            verdicts.append(
                ItemVerdict(
                    source=_CENSUS_SOURCE,
                    challenge_id="",
                    item_id="census:metagraph",
                    miner_hotkey=None,
                    uid=None,
                    bundle_digest="",
                    packet_digest="",
                    verdict=ItemVerdictKind.SKIP,
                    code=SNAPSHOT_UNVERIFIED,
                    detail=(
                        "the close-block metagraph is unavailable, so the registered-miner "
                        "census cannot be bound — an empty authority census is NOT treated as "
                        "a genuinely empty subnet during an outage; HOLDING INCONCLUSIVE"
                    ),
                )
            )
        else:
            # Validator permit is a capability, not an exclusive role. A miner may
            # acquire it after earning stake and must not disappear from the replay
            # boundary or payout census. Therefore bind every registered identity;
            # non-serving control hotkeys remain census-only and cannot earn without
            # valid warrant/score evidence.
            expected = dict(metagraph)
            expected_uids = set(expected)
            actual_uids = set(census_by_uid)
            for uid in sorted(expected_uids - actual_uids):
                neuron = expected[uid]
                verdicts.append(
                    ItemVerdict(
                        source=_CENSUS_SOURCE,
                        challenge_id="",
                        item_id=f"census:{uid}",
                        miner_hotkey=getattr(neuron, "hotkey", None),
                        uid=uid,
                        bundle_digest="",
                        packet_digest="",
                        verdict=ItemVerdictKind.FAIL,
                        code=CENSUS_MISMATCH,
                        detail=(
                            f"registered payout identity uid {uid} (hotkey "
                            f"{getattr(neuron, 'hotkey', None)!r}) is present in the "
                            "independently-read close-block metagraph but OMITTED from "
                            "miner_census — a census omission cannot erase earning/replay "
                            "history (schema v11)"
                        ),
                    )
                )
            for uid in sorted(actual_uids - expected_uids):
                entry = census_by_uid[uid]
                verdicts.append(
                    self._census_verdict(
                        uid,
                        entry,
                        ItemVerdictKind.FAIL,
                        f"uid {uid} (hotkey {entry.hotkey!r}) appears in miner_census but "
                        "is not registered in the independently-read "
                        "close-block metagraph — the census must exactly bind to chain "
                        "membership (schema v11)",
                    )
                )
            for uid in sorted(expected_uids & actual_uids):
                neuron = expected[uid]
                entry = census_by_uid[uid]
                expected_identity = (neuron.hotkey, neuron.coldkey, neuron.ip)
                actual_identity = (entry.hotkey, entry.coldkey, entry.ip)
                if actual_identity != expected_identity:
                    verdicts.append(
                        self._census_verdict(
                            uid,
                            entry,
                            ItemVerdictKind.FAIL,
                            f"miner_census uid {uid} identity {actual_identity!r} does not "
                            f"match close-block metagraph identity {expected_identity!r} "
                            "(schema v11)",
                        )
                    )
                else:
                    verdicts.append(
                        self._census_verdict(uid, entry, ItemVerdictKind.PASS, "")
                    )

        # Defense in depth for model-constructed payloads: every eligible/economic snapshot
        # must be a matching member of the registered census. EpochLog validates this normally.
        for uid, miner in sorted(miners_by_uid.items()):
            entry = census_by_uid.get(uid)
            if entry is None or (entry.hotkey, entry.coldkey, entry.ip) != (
                miner.hotkey,
                miner.coldkey,
                miner.ip,
            ):
                verdicts.append(
                    ItemVerdict(
                        source=_CENSUS_SOURCE,
                        challenge_id="",
                        item_id=f"census-economic:{uid}",
                        miner_hotkey=miner.hotkey,
                        uid=uid,
                        bundle_digest="",
                        packet_digest="",
                        verdict=ItemVerdictKind.FAIL,
                        code=CENSUS_MISMATCH,
                        detail=(
                            f"economic miner uid {uid} is absent from miner_census or has a "
                            "different registered identity — every economic row must be a "
                            "matching subset member (schema v11)"
                        ),
                    )
                )

        # Committed earning evidence still requires an economic row. This is provable from the
        # log alone and remains conclusive even when the metagraph is unavailable.
        for uid in sorted(self._evidenced_uids(log)):
            if uid in miners_by_uid:
                continue
            verdicts.append(
                ItemVerdict(
                    source=_CENSUS_SOURCE,
                    challenge_id="",
                    item_id=f"census-economic:{uid}",
                    miner_hotkey=census_by_uid.get(uid).hotkey
                    if uid in census_by_uid
                    else None,
                    uid=uid,
                    bundle_digest="",
                    packet_digest="",
                    verdict=ItemVerdictKind.FAIL,
                    code=CENSUS_MISMATCH,
                    detail=(
                        f"uid {uid} carries committed earning evidence in the manifest but is "
                        "OMITTED from the eligible/economic miners set — the authority stored "
                        "the miner's evidence then censored it from weight derivation"
                    ),
                )
            )
        return verdicts

    def _fold_cursor_verdicts(
        self,
        log: EpochLog,
        prior_log: EpochLog | None,
        is_genesis: bool,
    ) -> list[ItemVerdict]:
        """Verify the schema-v15 total, tamper-evident replay boundary.

        The expected current map is exactly ``prior.fold_cursors`` (including deregistered
        tombstones), with every uid in the current census inserted as ``None`` when never seen
        and each uid that folds current cycles advanced to that cycle set's maximum key. No
        predecessor entry may disappear, no cursor may regress, and no cursor may advance
        without the committed cycles that justify it. A first fold after ``None`` is valid;
        numeric uid history survives hotkey replacement and later uid reuse.
        """
        current = {
            int(uid): None if key is None else int(key)
            for uid, key in log.audit_manifest.fold_cursors.items()
        }
        census_uids = {int(entry.uid) for entry in log.miner_census}
        cycle_keys = {
            int(uid): [c.ordering_key for c in earning_input.cycle_scores]
            for uid, earning_input in log.audit_manifest.earning_inputs.items()
            if earning_input.cycle_scores
        }

        def verdict(
            kind: ItemVerdictKind,
            detail: str,
            *,
            uid: int | None = None,
            code: str = FOLD_CURSOR_MISMATCH,
        ) -> ItemVerdict:
            miner = next((m for m in log.miners if m.uid == uid), None)
            return ItemVerdict(
                source=_EARNING_SOURCE,
                challenge_id="",
                item_id=(
                    f"fold-cursor:{uid}" if uid is not None else "fold-cursor"
                ),
                miner_hotkey=(miner.hotkey if miner is not None else None),
                uid=uid,
                bundle_digest="",
                packet_digest="",
                verdict=kind,
                code=code,
                detail=detail,
            )

        # Local identity coverage is checked even for model-constructed payloads that bypassed
        # EpochLog's v14 validator. Presence is membership-sensitive: a missing key is not the
        # same state as an explicitly observed, never-folded ``None`` cursor.
        missing_census = sorted(census_uids - set(current))
        if missing_census:
            return [
                verdict(
                    ItemVerdictKind.FAIL,
                    f"current census uid {uid} has no anchored fold cursor; schema v14 requires "
                    "an explicit null cursor for every observed identity before its first fold",
                    uid=uid,
                )
                for uid in missing_census
            ]

        for uid, keys in sorted(cycle_keys.items()):
            cursor = current.get(uid)
            if uid not in current or cursor is None or cursor < max(keys):
                return [
                    verdict(
                        ItemVerdictKind.FAIL,
                        f"uid {uid} folds current ordering_key(s) {sorted(keys)} but its "
                        f"anchored cumulative cursor is {cursor!r}; every current "
                        "cycle must be covered by the replay boundary",
                        uid=uid,
                    )
                ]

        if prior_log is None:
            if log.prior_log_digest is not None:
                # Predecessor-chain verdict also HOLDs; make watermark continuity explicit.
                return [
                    verdict(
                        ItemVerdictKind.SKIP,
                        "the log references a predecessor that could not be loaded, so its "
                        "cumulative fold cursor continuity is UNVERIFIABLE",
                        code=EARNING_STATE_UNVERIFIED,
                    )
                ]
            if not is_genesis:
                return []  # predecessor-chain guard emits the conclusive reset fault
            expected: dict[int, int | None] = {
                uid: None for uid in census_uids
            }
            for uid, keys in cycle_keys.items():
                expected[uid] = max(keys)
        elif log.prior_log_digest != prior_log.log_digest():
            return []  # predecessor-chain guard emits the conclusive digest mismatch
        else:
            expected = {
                int(uid): None if key is None else int(key)
                for uid, key in prior_log.audit_manifest.fold_cursors.items()
            }
            for uid in census_uids:
                expected.setdefault(uid, None)
            for uid, keys in sorted(cycle_keys.items()):
                prior_key = expected.get(uid)
                if prior_key is not None and min(keys) <= prior_key:
                    # `_replay_check` emits the per-uid EARNING_PACKET_REPLAY; keep the expected
                    # map unchanged here so this helper does not obscure the more specific code.
                    continue
                expected[uid] = max(keys)

        if current == expected:
            return []
        all_uids = sorted(set(current) | set(expected))
        missing = object()
        mismatched = [
            uid for uid in all_uids
            if current.get(uid, missing) != expected.get(uid, missing)
        ]
        return [
            verdict(
                ItemVerdictKind.FAIL,
                f"uid {uid}'s cumulative fold cursor is {current.get(uid, '<missing>')!r}, "
                f"expected {expected.get(uid, '<missing>')!r} from the chained predecessor, "
                "current census, and this epoch's committed "
                "cycles — replay history was dropped, regressed, or advanced without evidence",
                uid=uid,
            )
            for uid in mismatched
        ]

    @staticmethod
    def _census_verdict(
        uid: int, miner, verdict: ItemVerdictKind, detail: str
    ) -> ItemVerdict:
        return ItemVerdict(
            source=_CENSUS_SOURCE,
            challenge_id="",
            item_id=f"census:{uid}",
            miner_hotkey=(miner.hotkey if miner is not None else None),
            uid=uid,
            bundle_digest="",
            packet_digest="",
            verdict=verdict,
            code=CENSUS_MISMATCH if verdict is ItemVerdictKind.FAIL else "",
            detail=detail,
        )

    def _recompute_excluded(
        self, log: EpochLog, metagraph: dict[int, object]
    ) -> set[int]:
        """Re-derive the IP/coldkey dedup ``excluded`` set INDEPENDENTLY.

        Rebuilds each log miner's identity from the METAGRAPH (coldkey/ip) — never the
        log's self-attested identity — keeping the log's score for the eligibility gate,
        then applies the SAME dedup rule ``build_weight_vector`` uses (the shared
        ``dedup_excluded`` → ``dedup_losers``). Miners absent from the metagraph are left
        out (a nonzero one is already an IDENTITY_MISMATCH; a zero-weight one cannot be
        bound). The returned uids are the dedup LOSERS — the ones whose ``excluded`` must
        be True.
        """
        identity_miners = [
            replace(m, coldkey=metagraph[m.uid].coldkey, ip=metagraph[m.uid].ip)
            for m in log.miners
            if m.uid in metagraph
        ]
        return dedup_excluded(
            identity_miners,
            minimum_payout_score=self._config.tokenomics.minimum_payout_score,
        )

    def _reconstructed_miners(self, log: EpochLog, metagraph: dict[int, object] | None):
        """The miner set the weight vector is re-derived over — SNAPSHOT inputs replaced by
        the independently-verified metagraph/committed values.

        identity ← metagraph; dedup ``excluded`` ← re-derived; track ← committed evidence;
        accumulate_score ← the log (earning fold verifies it), so the FULL weight vector is
        derived over independently-verified inputs. (The windowed-input re-derivation was
        REMOVED with the retention multiplier for v1 — retention removed — owner decision;
        an internal review.) Returns None when a nonzero-weight uid
        cannot be reconstructed (metagraph unavailable) so the caller HOLDs. Only reached
        once the per-uid snapshot verdicts carry no FAIL/SKIP, so every nonzero uid is
        present-and-matching in the metagraph here.
        """
        nonzero = {
            uid
            for uid, w in log.weight_shares.items()
            if w > 0.0 and uid != log.burn_uid
        }
        if not nonzero:
            # burn/empty epoch: nothing snapshot-derivable. The raw log miners are returned
            # here, but this reconstruct is only reached from `_weight_verdict` AFTER the
            # per-uid snapshot verdicts carry no FAIL/SKIP
            # POSITIVE implicit carry-forward miner that is ABSENT from the metagraph (tampered
            # `excluded` unbindable), so a burn substituted by a raw-preserved tampered exclusion
            # can never reach this raw return without first disputing on the snapshot channel.
            return list(log.miners)
        if metagraph is None:
            return None
        excluded_set = self._recompute_excluded(log, metagraph)
        out = []
        for m in log.miners:
            neuron = metagraph.get(m.uid)
            if neuron is None:
                if m.uid in nonzero:
                    return None  # a nonzero uid we cannot bind ⇒ HOLD
                out.append(m)  # zero-weight, unbindable: keep as-is (no weight impact)
                continue
            committed_track = self._committed_track_for_uid(log, m.uid)
            track = committed_track if committed_track is not None else m.track
            out.append(
                replace(
                    m,
                    hotkey=neuron.hotkey,
                    coldkey=neuron.coldkey,
                    ip=neuron.ip,
                    track=track,
                    excluded=m.uid in excluded_set,
                )
            )
        return out

    def _committed_track_for_uid(self, log: EpochLog, uid: int) -> str | None:
        """The committed scoring track of a uid's earning evidence, or None if it cannot
        be resolved.

        Sourced from the uid's SCORE_PACKET ``AuditFileRef.committed_track`` — the same
        committed track the earning path independently anchors to the CHALLENGE commitment
        (the DAG_REVEAL preimage) and FAILs/SKIPs on if it disagrees, so the value here is
        transitively bound to real committed evidence. None when no committed track can be
        read (the earning path also holds such a uid INCONCLUSIVE) or the uid's refs carry
        more than one distinct committed track (ambiguous ⇒ unresolvable).
        """
        tracks = {
            ref.committed_track
            for ref in log.audit_manifest.refs_for(uid)
            if ref.kind is AuditFileKind.SCORE_PACKET
            and ref.committed_track is not None
        }
        tracks.update(
            evidence.track
            for evidence in log.audit_manifest.availability_inputs
            if evidence.uid == uid
        )
        if len(tracks) != 1:
            return None
        return next(iter(tracks))

    # (The committed windowed-evidence re-derivation — _windowed_verdicts /
    # _windowed_verdict_for_uid / _window_start_check / _window_verdict — was REMOVED
    # with the retention multiplier for v1 — retention removed — owner decision; deletes
    # an internal review.)

    # -- earning-state re-derivation (#1) -------------------------------------------

    @staticmethod
    def _competition_window_only_earning_uids(log: EpochLog) -> set[int]:
        """Return paid uids whose weight is backed only by competition evidence.

        An active global competition window can pay a registered miner
        that has no inference EWMA at all.  Those weights are verified by
        ``_competition_verdicts`` and the competition-aware weight re-derivation; they
        must not be misrouted through the inference carry-forward verifier, where a
        legitimate genesis ``0.0`` would (correctly for inference) look like an
        unaudited injected accumulator.

        The exemption is deliberately narrow.  The uid must be bound to either an
        exact current contender identity or a reward-window podium hotkey, must have no payable
        inference state (non-positive or explicitly excluded), and must carry no
        inference packet, earning input, or signed availability observation.  If any
        inference evidence/state exists, the ordinary EWMA audit still runs.
        """
        manifest = log.audit_manifest
        miners_by_uid = {miner.uid: miner for miner in log.miners}
        competition_identities: set[tuple[int, str]] = set()
        if manifest.competition_input is not None:
            competition_identities.update(
                (subject.uid, subject.hotkey)
                for subject in manifest.competition_input.subjects
                if (
                    subject.role == "contender"
                    and subject.uid is not None
                    and subject.hotkey is not None
                )
            )
        window_hotkeys = set(log.reward_window_state.podium_hotkeys)

        out: set[int] = set()
        for uid, miner in miners_by_uid.items():
            competition_backed = (
                uid,
                miner.hotkey,
            ) in competition_identities or miner.hotkey in window_hotkeys
            inference_inactive = (
                miner.excluded
                or is_excluded(miner.accumulate_score)
                or miner.accumulate_score <= 0.0
            )
            has_inference_evidence = (
                uid in manifest.earning_inputs
                or bool(manifest.refs_for(uid))
                or bool(manifest.availability_for(uid))
            )
            if competition_backed and inference_inactive and not has_inference_evidence:
                out.add(uid)
        return out

    def _earning_verdicts(
        self,
        log: EpochLog,
        store: AuditStore,
        prior_log: EpochLog | None,
        is_genesis: bool = True,
    ) -> list[ItemVerdict]:
        """Re-derive every AUDITED uid's EARNING STATE from audited evidence.

        For each audited uid (other than the empty-epoch burn uid): re-fold
        ``accumulate_score`` from the manifest's ``EarningInput`` (prior carry-in + ordered
        cycle scores), CHECK that the nonzero cycle scores are backed by the uid's audited
        SCORE_PACKETs, and CHAIN the carry-in against the prior epoch's log. A substituted
        accumulate_score published alongside honest packets no longer passes: the fold will
        not reproduce it (EARNING_STATE_MISMATCH). Cheap (arithmetic over already-committed
        packet scores) — the expensive media recompute stays sampled.

        an internal review(a): the audited set is NOT just the current positive-weight uids.
        It also covers every ZERO-weight uid that carries committed earning evidence (an
        ``EarningInput``) OR a positive stated accumulator. A zero-weight miner used to be
        entirely unaudited, so it could carry a SUBSTITUTED ``accumulate_score`` in epoch E
        (nothing re-folded it); when it entered top-N in E+1 the carry-in check accepted that
        prior STATED accumulator verbatim — re-attributing the substitution into paid weight. Now a
        zero-weight uid's accumulator is ALSO re-derived from evidence (a bad fold FAILs) and,
        when it carries none this epoch, verified as a chained CARRY-FORWARD of the prior
        epoch's value (an injected jump FAILs) — so it cannot be substituted then re-attributed.

        The reward-window re-derivation is NOT here: it is computed in
        ``audit_epoch`` from the RE-DERIVED competition result, not the log's stated one.
        """
        decay = self._config.tokenomics.ewma_decay
        miners_by_uid = {m.uid: m for m in log.miners}
        verdicts: list[ItemVerdict] = []
        audited: set[int] = {uid for uid, w in log.weight_shares.items() if w > 0.0}
        # #2(a): every uid the manifest carries committed earning evidence for, AND every uid
        # with a positive stated accumulator — so a zero-weight miner's accumulator is verified
        # too (not trusted verbatim as a later carry-in). The exclusion sentinel (-1) and a zero
        # accumulator carry no re-attributable value, so they need no earning verdict of their own.
        audited |= set(log.audit_manifest.earning_inputs)
        audited.update(
            evidence.uid for evidence in log.audit_manifest.availability_inputs
        )
        for m in log.miners:
            if not is_excluded(m.accumulate_score) and m.accumulate_score > 0.0:
                audited.add(m.uid)
        audited.discard(log.burn_uid)
        # Competition-window payees with no inference state are verified by the
        # competition-result/reward-window fold above, not by the inference EWMA fold.
        audited.difference_update(self._competition_window_only_earning_uids(log))
        for uid in sorted(audited):
            verdicts.append(
                self._earning_verdict_for_uid(
                    uid,
                    miners_by_uid.get(uid),
                    log,
                    store,
                    prior_log,
                    decay,
                    is_genesis,
                )
            )
        # The reward-window re-derivation is no longer folded here: it is computed in
        # `audit_epoch` from the RE-DERIVED competition result, then appended
        # to the earning-verdicts channel there.
        return verdicts

    def _reset_earning_verdicts(
        self,
        log: EpochLog,
        prior_log: EpochLog | None,
        metagraph: dict[int, object] | None,
    ) -> list[ItemVerdict]:
        """Detect a prior-POSITIVE, STILL-REGISTERED miner SILENTLY RESET this epoch (round-19 #1).

        The earning re-fold + census only look at the CURRENT positive accumulators / current
        evidence, and ``_earning_verdicts`` only audits uids with a positive weight, a current
        ``EarningInput``, or a positive CURRENT accumulator. So a miner reset to 0.0 / the
        exclusion sentinel (or dropped from the census) with NO current evidence is in NONE of
        those sets — it is silently skipped, and the vector re-derives CLEAN off another miner's
        new evidence. That erases a still-registered miner's accrued earnings.

        Using the prior epoch log (already chained via the carry-in), FAIL any miner that:
          (a) had a POSITIVE accumulator in the prior log (accrued earnings a reset erases);
          (b) is STILL present in the close-block metagraph under the SAME (uid, hotkey); yet
          (c) is reset to 0.0 / excluded (or dropped from ``log.miners``) THIS epoch WITHOUT an
              evidenced reason — no current ``EarningInput`` justifying the drop.

        EWMA DECAYS but never zeroes a positive accumulator in one epoch, and a no-evidence miner
        must CARRY its value forward unchanged (``_carry_forward_verdict``), so a still-registered
        positive accumulator that becomes 0.0 / excluded / absent with no committed exclusion is a
        censored/erased earning state ⇒ DISPUTED (EARNING_STATE_RESET, rides the earning channel
        into the weight verdict). Legitimate and NOT flagged:
          - a GENUINE deregistration (absent from the metagraph) — a fresh uid carries in 0.0;
          - a hotkey CHANGE (re-registration) — a fresh identity carries in 0.0;
          - an EVIDENCED exclusion / any current ``EarningInput`` — the earning fold audits it
            (an evidenced exclusion folds to -1; a substituted one FAILs there), so defer to it;
          - a value CARRIED FORWARD positive (or reduced-but-positive) — the carry-forward path
            audits the value (an injected jump FAILs there), so defer to it.
        DISTINCT from the sanctioned all-carry ``items=[]`` burn: those miners carry their
        POSITIVE accumulators FORWARD (staying eligible), so none is reset and none is flagged.
        """
        if prior_log is None or metagraph is None:
            # No chained prior to compare, or no metagraph to establish "still registered".
            # Both are handled fail-closed elsewhere (a referenced-but-unavailable prior SKIPs the
            # carry-in ⇒ INCONCLUSIVE; an unreadable metagraph HOLDs the snapshot binding), so
            # reset-detection must not FAIL on an unverifiable premise.
            return []
        if log.prior_log_digest != prior_log.log_digest():
            # The supplied prior is not the chained prior (broken chain): the carry-in check FAILs
            # the audited uids; comparing reset state against a non-matching prior is meaningless.
            return []
        cur_by_uid = {m.uid: m for m in log.miners}
        earning_inputs = log.audit_manifest.earning_inputs
        verdicts: list[ItemVerdict] = []
        for pm in prior_log.miners:
            if is_excluded(pm.accumulate_score) or pm.accumulate_score <= 0.0:
                continue  # (a) only a prior POSITIVE accumulator has earnings a reset erases
            neuron = metagraph.get(pm.uid)
            if neuron is None:
                continue  # genuine deregistration — legitimate (a fresh uid carries in 0.0)
            if getattr(neuron, "hotkey", None) != pm.hotkey:
                continue  # re-registered under a new hotkey — a fresh identity, carries in 0.0
            # (b) still registered under the SAME (uid, hotkey). An EVIDENCED drop is legitimate:
            # a current EarningInput is audited by the earning fold (evidenced exclusion ⇒ -1; a
            # substituted one FAILs there), so defer to it and never double-verdict.
            if pm.uid in earning_inputs:
                continue
            cm = cur_by_uid.get(pm.uid)
            if cm is None:
                detail = (
                    f"uid {pm.uid} (hotkey {pm.hotkey!r}) had a POSITIVE accumulator "
                    f"{pm.accumulate_score} in the prior epoch and is STILL registered in the "
                    "close-block metagraph under the same (uid, hotkey), but is DROPPED from "
                    "log.miners this epoch with NO current earning evidence — a still-registered "
                    "positive earner's accrued earnings erased/censored"
                )
            elif is_excluded(cm.accumulate_score) or cm.accumulate_score <= 0.0:
                detail = (
                    f"uid {pm.uid} (hotkey {pm.hotkey!r}) had a POSITIVE accumulator "
                    f"{pm.accumulate_score} in the prior epoch and is STILL registered in the "
                    "close-block metagraph under the same (uid, hotkey), but is RESET to "
                    f"{cm.accumulate_score} this epoch with NO current earning evidence — EWMA "
                    "decays but never zeroes a positive accumulator in one epoch, and a "
                    "no-evidence miner must carry its value forward; a silent reset without a "
                    "committed exclusion is a censored earning state"
                )
            else:
                continue  # carried forward positive — the carry-forward path audits the value
            verdicts.append(
                ItemVerdict(
                    source=_EARNING_SOURCE,
                    challenge_id="",
                    item_id=f"reset:{pm.uid}",
                    miner_hotkey=pm.hotkey,
                    uid=pm.uid,
                    bundle_digest="",
                    packet_digest="",
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_RESET,
                    detail=detail,
                )
            )
        return verdicts

    def _predecessor_chain_verdict(
        self,
        log: EpochLog,
        prior_log: EpochLog | None,
        is_genesis: bool,
    ) -> ItemVerdict | None:
        """LOG-LEVEL predecessor-chain enforcement — fires even for an EMPTY/BURN log (round-20 #1).

        Every per-uid carry-in / carry-forward chain check (``_carry_in_check`` /
        ``_carry_forward_verdict``) runs ONLY for a uid SELECTED for earning audit (positive
        weight, a current ``EarningInput``, or a positive stated accumulator — see
        ``_earning_verdicts``). An EMPTY canonical burn log (``miners=[]``, ``{burn_uid:1.0}``,
        no ``earning_inputs``) selects NO earning uid, so the chain enforcement NEVER RUNS: a
        NON-genesis authority could OMIT ``prior_log_digest`` (or reference an unavailable
        predecessor), publish the empty burn vector, and audit CLEAN — silently RESETTING all
        prior positive earning state, while item-scoped checks (seeing no nonzero carry-in)
        could report the epoch clean and advance an auditor cursor past the erased history.

        So the chain is also enforced at the LOG level, provable from the log's bytes + the
        loop's INDEPENDENT genesis determination (``is_genesis``), independent of any audited
        uid — exactly the guard the empty-burn case bypassed:

        - true genesis (``is_genesis``, ``prior_log_digest`` None): legitimate ⇒ no verdict;
        - NON-genesis + ``prior_log_digest`` None: a chain reset (omitting the digest resets ALL
          chained state) ⇒ FAIL PREDECESSOR_CHAIN_BROKEN (DISPUTED);
        - a supplied prior whose digest does NOT match ``prior_log_digest``: broken chain ⇒
          FAIL PREDECESSOR_CHAIN_BROKEN;
        - ``prior_log_digest`` set but the prior could not be LOADED (pruned / unreadable —
          ``prior_log`` None): UNVERIFIABLE ⇒ SKIP PREDECESSOR_UNVERIFIED (INCONCLUSIVE / HOLD),
          never a CLEAN that advances the cursor past an unverifiable reset (round-8 #6 at the
          log level).

        Rides the earning-verdicts channel (a FAIL disputes, a SKIP holds). This subsumes the
        per-uid chain checks for the empty-log case they cannot reach; for a non-empty log it is
        redundant defense-in-depth with them (a chain break is caught either way). A genuinely
        EMPTY epoch that MAINTAINS the chain (valid ``prior_log_digest`` matching an available
        prior) is CLEAN — only breaking/censoring the chain is faulted.
        """
        base = dict(
            source=_EARNING_SOURCE,
            challenge_id="",
            item_id="predecessor-chain",
            miner_hotkey=None,
            uid=None,
            bundle_digest="",
            packet_digest="",
        )
        if log.prior_log_digest is None:
            if is_genesis:
                return None  # the true genesis legitimately has no predecessor
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=PREDECESSOR_CHAIN_BROKEN,
                detail=(
                    f"epoch {log.epoch_id} OMITS prior_log_digest at a NON-genesis epoch — a "
                    "missing chain digest RESETS all chained earning state; only the true "
                    "genesis may omit it. An empty/burn log selects no earning uid, so the "
                    "per-uid carry-in chain check never runs; this log-level guard catches the "
                    "reset the empty-burn case bypasses"
                ),
                **base,
            )
        if prior_log is None:
            # A digest is referenced but the prior could not be loaded (pruned / unreadable):
            # UNVERIFIABLE, fail closed to INCONCLUSIVE (round-8 #6 at the log level).
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=PREDECESSOR_UNVERIFIED,
                detail=(
                    f"epoch {log.epoch_id} references prior_log_digest "
                    f"{log.prior_log_digest!r} but that predecessor could not be loaded — the "
                    "predecessor chain is UNVERIFIABLE (an unavailable prior is how earnings "
                    "would be reset/censored); HOLDING rather than advancing past an "
                    "unverifiable reset"
                ),
                **base,
            )
        if log.prior_log_digest != prior_log.log_digest():
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=PREDECESSOR_CHAIN_BROKEN,
                detail=(
                    f"epoch {log.epoch_id} prior_log_digest {log.prior_log_digest!r} does not "
                    f"match the supplied prior epoch log ({prior_log.log_digest()!r}) — the "
                    "predecessor chain is broken"
                ),
                **base,
            )
        return None

    def _track_membership_verdicts(self, log: EpochLog) -> list[ItemVerdict]:
        """Validate every committed / log TRACK against the PROTOCOL track set (round-19 #2).

        ``AuditFileRef`` requires a non-null ``committed_track`` but never that it be a MEMBER of
        the protocol track set; commitment parsing accepts any non-empty string; and tokenomics
        ``inference_shares`` SILENTLY drops a miner whose track is absent from ``track_weights``.
        So committing positive evidence CONSISTENTLY under an out-of-protocol track (e.g.
        "unknown") collapses the vector to ``{burn_uid: 1.0}`` and audits CLEAN — the existing
        track binding only compares self-consistent DECLARATIONS to each other, never to the
        protocol. This validates every ``MinerSnapshot.track`` and ``AuditFileRef.committed_track``
        against the authoritative protocol set (the keys of ``tokenomics.track_weights``, the SAME
        set the finalizer's tokenomics is keyed by): an out-of-set track is a provable substituted
        burn ⇒ FAIL (UNKNOWN_TRACK ⇒ DISPUTED). Provable from the log's own bytes (no metagraph),
        so it fires even for a burn/empty log — defense-in-depth for bytes that BYPASSED the
        finalizer (``EpochLog._validate`` refuses such a log). Honest in-set tracks are unaffected.
        """
        protocol = frozenset(self._config.tokenomics.track_weights)
        verdicts: list[ItemVerdict] = []
        for m in log.miners:
            if m.uid == log.burn_uid:
                continue
            if m.track not in protocol:
                verdicts.append(
                    self._track_verdict(
                        m.uid,
                        m.hotkey,
                        f"track:{m.uid}",
                        f"uid {m.uid} declares scoring track {m.track!r}, which is NOT a protocol "
                        f"track {sorted(protocol)} — an out-of-protocol track is silently dropped "
                        "from every tokenomics pool and substitutes a burn",
                    )
                )
        seen: set[str] = set()
        all_refs = list(log.audit_manifest.baseline_bundles)
        for refs in log.audit_manifest.per_uid.values():
            all_refs.extend(refs)
        for refs in log.audit_manifest.competition_bundles.values():
            all_refs.extend(refs)
        for ref in all_refs:
            track = ref.committed_track
            if track is None or track in protocol:
                continue
            item_id = f"track-ref:{ref.item_id}:{track}"
            if item_id in seen:
                continue
            seen.add(item_id)
            verdicts.append(
                self._track_verdict(
                    None,
                    None,
                    item_id,
                    f"audit ref for item {ref.item_id!r} carries committed_track {track!r}, which "
                    f"is NOT a protocol track {sorted(protocol)} — an out-of-protocol committed "
                    "track substitutes a canonical burn while every declaration self-agrees "
                    "",
                )
            )
        for evidence in log.audit_manifest.availability_inputs:
            if evidence.track in protocol:
                continue
            verdicts.append(
                self._track_verdict(
                    evidence.uid,
                    evidence.hotkey,
                    f"track-availability:{evidence.item_id}:{evidence.track}",
                    f"availability evidence for item {evidence.item_id!r} carries track "
                    f"{evidence.track!r}, which is NOT a protocol track "
                    f"{sorted(protocol)}",
                )
            )
        return verdicts

    @staticmethod
    def _track_verdict(
        uid: int | None, miner_hotkey: str | None, item_id: str, detail: str
    ) -> ItemVerdict:
        """A protocol-track-membership FAIL. Rides the CENSUS channel
        (byte-provable, no metagraph; FAIL ⇒ DISPUTED) and never counts toward the media floor."""
        return ItemVerdict(
            source=_CENSUS_SOURCE,
            challenge_id="",
            item_id=item_id,
            miner_hotkey=miner_hotkey,
            uid=uid,
            bundle_digest="",
            packet_digest="",
            verdict=ItemVerdictKind.FAIL,
            code=UNKNOWN_TRACK,
            detail=detail,
        )

    def _earning_verdict_for_uid(
        self,
        uid: int,
        miner,
        log: EpochLog,
        store: AuditStore,
        prior_log: EpochLog | None,
        decay: float,
        is_genesis: bool = True,
    ) -> ItemVerdict:
        base = dict(
            source=_EARNING_SOURCE,
            challenge_id="",
            item_id=f"uid:{uid}",
            miner_hotkey=miner.hotkey if miner is not None else None,
            uid=uid,
            bundle_digest="",
            packet_digest="",
        )
        ei = log.audit_manifest.earning_for(uid)
        if ei is None:
            if log.audit_manifest.availability_for(uid):
                # A schema-v14 availability observation is itself committed earning
                # evidence.  It must never fall through to the evidence-free carry path:
                # that would accept the signed zero without binding it to a CycleScore,
                # replay boundary, or EWMA fold (possible only for model-constructed /
                # otherwise malformed bytes, since AuditManifest normally rejects it).
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_MISMATCH,
                    detail=(
                        "availability evidence is present for this uid but no earning "
                        "input binds it into the committed zero-score fold"
                    ),
                    **base,
                )
            weight = log.weight_shares.get(uid, 0.0)
            if weight > 0.0 and miner is None:
                # A nonzero-weight uid that is not even in the snapshots cannot be attributed
                # or carried — a conclusive fault, never a SKIP that washes CLEAN.
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_MISMATCH,
                    detail="nonzero weight but the uid is absent from the miner snapshots and "
                    "carries no earning input",
                    **base,
                )
            # an internal review(a) / round-20 #2: a uid with a positive stated accumulator but NO
            # committed earning evidence this epoch (no cycles) is a pure CARRY-FORWARD — an idle
            # prior earner. This is the NORMAL idle/carry epoch: an EWMA accumulator DECAYS but is
            # not re-earned every epoch, so a miner that did no new work this epoch STILL holds
            # (and is weighted by) its carried accumulator. Round-20 #2: this now covers the
            # NONZERO-weight carry-forward too (previously it fell through to an INCONCLUSIVE HOLD,
            # so a positive-weight idle earner could NOT be represented — the report finalizer
            # either stalled or dropped every miner to an empty burn that round-19 then disputed).
            # `_carry_forward_verdict` chains it exactly like the zero-weight path: its accumulator
            # must equal the prior epoch's audited value for the SAME (uid, hotkey) and its track
            # must chain (an injected jump / inherited-across-hotkey-change / track switch FAILs;
            # genesis or an unavailable prior fails closed). So a nonzero-weight carry-forward is
            # VERIFIED against the chain, never trusted verbatim and never a blanket HOLD.
            return self._carry_forward_verdict(
                uid, miner, log, prior_log, is_genesis, base
            )
        if miner is None:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail="nonzero weight but the uid is absent from the miner snapshots",
                **base,
            )
        # 0a. NULL/EMPTY LOG HOTKEY: a NONZERO-weight uid whose log
        #     hotkey is missing/empty is itself a fault, NOT a skip. `MinerSnapshot` is an
        #     unchecked dataclass and `_miner_from_obj` accepts JSON null, so the authority
        #     could publish a uid with a null hotkey wrapping another miner's packet — and
        #     every identity check below (which compares against `expected_hotkey=miner.hotkey`)
        #     would SKIP, because it only fires when the expected hotkey is non-null. A null
        #     expected identity means the score can never be attributed to this uid, so the
        #     earning path FAILs CLOSED here (IDENTITY_MISMATCH ⇒ DISPUTED) rather than let
        #     the null-hotkey bypass wash the miner check.
        if not miner.hotkey:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=IDENTITY_MISMATCH,
                detail=(
                    f"uid {uid} has nonzero weight but its log hotkey is missing/empty "
                    f"({miner.hotkey!r}) — a null expected identity cannot attribute any "
                    "packet's score to this uid; the earning path fails closed rather than "
                    "skip the miner check (the null-hotkey bypass)"
                ),
                **base,
            )
        # Authenticate every embedded availability observation before it can stand in
        # for a media SCORE_PACKET.  The helper also binds its signed request to the
        # close-block census/economic identity and independently archive-checks the
        # finalized challenge anchor.  Only then are its exact zero and committed order
        # exposed in the same map shapes used by the ordinary backing checks below.
        (
            availability_packets,
            availability_challenges,
            availability_error,
        ) = self._availability_evidence_fields(log, uid, miner)
        if availability_error is not None:
            verdict, code, detail = availability_error
            return ItemVerdict(verdict=verdict, code=code, detail=detail, **base)
        # 0. IDENTITY BINDING: the manifest pairs an AUDIT_BUNDLE ref
        #    and a SCORE_PACKET ref by AUTHORITY-supplied (challenge_id, item_id) LABELS.
        #    Before trusting a resolved bundle's committed evidence (its DAG_REVEAL binds
        #    a (track, ordering_key)) or folding its packet's score, PROVE the resolved
        #    bundle actually AUTHENTICATES this packet and belongs to this uid: its own
        #    score_packet.digest must equal the SCORE_PACKET ref's digest, and its
        #    challenge/item/miner must match. Otherwise a misreporting authority could point
        #    the bundle ref at an UNRELATED but resolvable bundle (its DAG_REVEAL
        #    well-formed) to back a packet the bundle never scored — a substitution that
        #    would otherwise fold an authority-minted score straight to PASS/CLEAN at zero
        #    media sampling. A proven substitution is a conclusive fault (DISPUTED), not a
        #    mere SKIP; an UNRESOLVABLE bundle is left to the fail-closed SKIP paths below.
        identity_err = self._earning_identity_error(log, uid, store, miner.hotkey)
        if identity_err is not None:
            code, detail = identity_err
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL, code=code, detail=detail, **base
            )
        # 1. evidence-bound backing: every cycle score is bound to a committed packet, in
        #    the committed order (a reorder / unbacked 0.0 / substituted -1 all FAIL here).
        committed = self._committed_packet_fields(log, uid, store)
        if committed is None:
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=EARNING_STATE_UNVERIFIED,
                detail="could not read the uid's committed score packets to back the fold",
                **base,
            )
        overlap = set(committed) & set(availability_packets)
        if overlap:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail=(
                    "availability observation digest is also committed as a media score "
                    f"packet for this uid: {sorted(overlap)}"
                ),
                **base,
            )
        committed.update(availability_packets)
        backing = _cycle_scores_backing_error(ei.cycle_scores, committed)
        if backing is not None:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail=backing,
                **base,
            )
        # 1b. NON-CIRCULAR binding: the fold ORDER and the TRACK must
        #     trace to the pre-dispatch CHALLENGE COMMITMENT (the anchored DAG_REVEAL),
        #     not the finalization-time packet. A reordered fold or a substituted track is
        #     caught here even though every packet is internally self-consistent.
        #     FAIL CLOSED: for a NONZERO-weight uid this evidence is
        #     MANDATORY. The finalizer REQUIRES a resolvable AUDIT_BUNDLE carrying a
        #     committed DAG_REVEAL for every earning item (#8), so a well-formed epoch
        #     ALWAYS populates `committed_challenge`. If it is None — the bundle,
        #     DAG_REVEAL, or commitment preimage cannot be resolved/verified — the
        #     committed ordering/track binding is UNREACHABLE, and a misreporting authority
        #     could otherwise OMIT the DAG_REVEAL to make this cross-check unreachable and
        #     fold self-consistent authority-minted packets straight to PASS/CLEAN. So the
        #     packet-bound checks above may only ever yield SKIP/INCONCLUSIVE or a FAIL for
        #     a nonzero uid: an unresolvable commitment ⇒ EARNING_STATE_UNVERIFIED (a SKIP
        #     that rolls up to INCONCLUSIVE / HOLD), NEVER a PASS via the packet-bound
        #     fallback. (The legitimate genesis/no-prior case is the CARRY-IN check below,
        #     not this one — genesis still carries resolvable committed evidence.)
        committed_challenge = self._committed_challenge_fields(log, uid, store)
        if committed_challenge is None:
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=EARNING_STATE_UNVERIFIED,
                detail=(
                    "the committed challenge evidence (bundle → DAG_REVEAL → commitment "
                    "preimage binding the (track, dispatch_ordering_key)) could not be "
                    "resolved and verified for this nonzero-weight uid — the committed "
                    "fold order/track is unverifiable, so the earning state is UNVERIFIED "
                    "(never a PASS via the packet-bound fallback); a well-formed epoch "
                    "always carries it, so missing-at-audit-time is a red flag → HOLD"
                ),
                **base,
            )
        overlap = set(committed_challenge) & set(availability_challenges)
        if overlap:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail=(
                    "availability observation digest is also challenge-backed as a media "
                    f"packet for this uid: {sorted(overlap)}"
                ),
                **base,
            )
        committed_challenge.update(availability_challenges)
        commit_err = _challenge_commitment_backing_error(
            ei.cycle_scores, committed_challenge
        )
        if commit_err is not None:
            code, detail = commit_err
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL, code=code, detail=detail, **base
            )
        # 2. re-fold (in the evidence-bound order) and compare to accumulate_score.
        folded = ei.prior_accumulate_score
        for score in ei.folded_scores():
            folded = accumulate(folded, score, decay)
        if not _fold_matches(folded, miner.accumulate_score):
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail=(
                    f"re-folding the audited cycle scores {list(ei.folded_scores())} over the "
                    f"carry-in {ei.prior_accumulate_score} yields {folded}, but the log states "
                    f"accumulate_score {miner.accumulate_score} — a substituted earning state"
                ),
                **base,
            )
        # 2b. NON-REPLAY across epochs: the fold above proves each cycle is
        #     backed by a committed packet in THIS manifest and the carry-in below chains the
        #     numeric value, but NEITHER proves the packet was not ALREADY folded by a PRIOR
        #     epoch. The committed dispatch ordering_key is MONOTONIC per uid, so every cycle
        #     folded THIS epoch must exceed the cumulative uid-slot watermark anchored by the
        #     prior epoch; a cycle at/below it is a re-fold of an earlier inference ⇒ FAIL (an
        #     inflated accumulator that the numeric checks alone accept).  Schema-v11 producers
        #     carry the complete map through idle, exclusion, deregistration and empty epochs.
        replay_kind, replay_detail = self._replay_check(uid, ei, log, prior_log)
        if replay_kind is ItemVerdictKind.FAIL:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_PACKET_REPLAY,
                detail=replay_detail,
                **base,
            )
        if replay_kind is ItemVerdictKind.SKIP:
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=EARNING_STATE_UNVERIFIED,
                detail=replay_detail,
                **base,
            )
        # 3. chain the carry-in against the prior epoch's log (back to genesis).
        kind, detail = self._carry_in_check(uid, ei, log, prior_log, is_genesis)
        if kind is ItemVerdictKind.FAIL:
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail=detail,
                **base,
            )
        if kind is ItemVerdictKind.SKIP:
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=EARNING_STATE_UNVERIFIED,
                detail=detail,
                **base,
            )
        return ItemVerdict(verdict=ItemVerdictKind.PASS, **base)

    def _replay_check(
        self,
        uid: int,
        ei: EarningInput,
        log: EpochLog,
        prior_log: EpochLog | None,
    ) -> tuple[ItemVerdictKind, str]:
        """Reject a CROSS-EPOCH packet REPLAY.

        The committed dispatch ``ordering_key`` is MONOTONIC per uid (the producer only folds a
        packet whose key exceeds the highest already folded), so every cycle folded THIS epoch
        must have an ordering_key STRICTLY GREATER than the maximum the uid folded THROUGH the
        prior epoch. A current cycle at/below that watermark is a re-fold of an earlier inference
        ⇒ FAIL. Returns:

        - PASS: all current cycles are strictly above the prior watermark (or there are no current
          cycles, or the uid is genuinely new with no prior census row or watermark — nothing to
          replay);
        - FAIL: some current cycle's ordering_key <= the prior watermark (a re-fold);
        - SKIP: malformed/model-constructed predecessor state carries the uid but lacks its v11
          cumulative boundary, so non-replay cannot be proven ⇒ INCONCLUSIVE (HOLD), never CLEAN.
        """
        cur_keys = [c.ordering_key for c in ei.cycle_scores]
        if not cur_keys:
            return (
                ItemVerdictKind.PASS,
                "",
            )  # a pure carry-in / no new cycle — nothing to fold
        if prior_log is None or log.prior_log_digest != prior_log.log_digest():
            # No verifiable prior chain here; the carry-in check separately downgrades this uid
            # (genesis PASS / non-genesis broken-chain FAIL / unavailable-prior SKIP). Defer.
            return (ItemVerdictKind.PASS, "")
        prior_miner = next((m for m in prior_log.miners if m.uid == uid), None)
        # Schema v11: replay history belongs to the numeric uid SLOT and survives hotkey
        # changes.  The accumulator still resets for a fresh hotkey (`_carry_in_check`), but the
        # dispatch sequence must remain above every key ever folded at that uid.  Resetting the
        # replay boundary on A -> B -> A is exactly the miner-reachable ping-pong exploit.
        prior_cursors = prior_log.audit_manifest.fold_cursors
        if uid not in prior_cursors:
            # A schema-v14 predecessor that observed this uid but omitted its cursor is a
            # conclusive structural fault, not an inconclusive first-fold case. A uid absent from
            # both predecessor census and cursor map is genuinely new and may fold its first key.
            if prior_miner is not None or any(c.uid == uid for c in prior_log.miner_census):
                return (
                    ItemVerdictKind.FAIL,
                    f"uid {uid} folds cycles {sorted(cur_keys)} this epoch but the prior epoch "
                    "observed this identity without the mandatory schema-v14 fold cursor",
                )
            return (
                ItemVerdictKind.PASS,
                "",
            )  # absent from prior — a first appearance, nothing to replay
        prior_max_key = prior_cursors[uid]
        if prior_max_key is None:
            # Explicit null means observed but never folded, so the first fold is provably clean.
            return (ItemVerdictKind.PASS, "")
        min_cur = min(cur_keys)
        if min_cur <= prior_max_key:
            return (
                ItemVerdictKind.FAIL,
                f"uid {uid} folds cycle ordering_key(s) {sorted(k for k in cur_keys if k <= prior_max_key)} "
                f"this epoch that are at/below the max key {prior_max_key} the uid ALREADY folded "
                "through the prior epoch — a CROSS-EPOCH re-fold of an earlier inference (committed "
                "dispatch keys are monotonic per uid, so a new cycle must exceed the prior "
                "watermark); replaying it inflates the accumulator and double-awards the same work "
                "",
            )
        return (ItemVerdictKind.PASS, "")

    def _carry_in_check(
        self,
        uid: int,
        ei: EarningInput,
        log: EpochLog,
        prior_log: EpochLog | None,
        is_genesis: bool = True,
    ) -> tuple[ItemVerdictKind, str]:
        """Verify the carry-in against the prior epoch (or note it unverifiable)."""
        if prior_log is not None:
            if log.prior_log_digest != prior_log.log_digest():
                return (
                    ItemVerdictKind.FAIL,
                    "prior_log_digest does not match the supplied prior epoch log — the "
                    "earning carry-in chain is broken",
                )
            prior_miner = next((m for m in prior_log.miners if m.uid == uid), None)
            # an internal review(b): the carry-in is keyed by (uid, hotkey), NOT uid alone. A uid
            # whose hotkey CHANGED vs the prior epoch is a FRESH identity — the validator registry
            # resets that uid's accumulate_score to 0.0 on a hotkey change
            # (miner_manager.sync_neurons) — so a re-registered hotkey carries in 0.0 and must NOT
            # inherit the previous owner's accumulator. A nonzero stated carry-in across a hotkey
            # change is a re-attributed inheritance ⇒ DISPUTED.
            cur_miner = next((m for m in log.miners if m.uid == uid), None)
            if (
                prior_miner is not None
                and cur_miner is not None
                and prior_miner.hotkey != cur_miner.hotkey
            ):
                if _fold_matches(ei.prior_accumulate_score, 0.0):
                    return (ItemVerdictKind.PASS, "")
                return (
                    ItemVerdictKind.FAIL,
                    f"uid {uid} carry-in {ei.prior_accumulate_score} is nonzero but the uid's "
                    f"hotkey CHANGED vs the prior epoch ({prior_miner.hotkey!r} -> "
                    f"{cur_miner.hotkey!r}) — a re-registered hotkey is a fresh identity that "
                    "carries in 0.0; inheriting the previous owner's accumulator is substituted "
                    "",
                )
            expected = prior_miner.accumulate_score if prior_miner is not None else 0.0
            if not _fold_matches(ei.prior_accumulate_score, expected):
                return (
                    ItemVerdictKind.FAIL,
                    f"carry-in {ei.prior_accumulate_score} != the prior epoch's stated "
                    f"accumulate_score {expected} for this uid — substituted carry-in",
                )
            return (ItemVerdictKind.PASS, "")
        # No prior log supplied. If the log declares itself a GENESIS (no
        # prior_log_digest), that is legitimate ONLY at the TRUE genesis epoch (review
        # round-9 #3). A NON-genesis epoch that OMITS prior_log_digest is a broken chain —
        # omitting the digest is exactly how an authority RESETs the earning carry-in to
        # zero at an arbitrary epoch (round-8 #6 only closed "digest present but prior
        # unavailable"). So a missing digest at a non-genesis epoch is a CONCLUSIVE fault
        # (DISPUTED), NOT re-treated as genesis.
        if log.prior_log_digest is None:
            if not is_genesis:
                return (
                    ItemVerdictKind.FAIL,
                    f"uid {uid}: the log OMITS prior_log_digest at a NON-genesis epoch "
                    f"({log.epoch_id}) — a missing prior chain digest resets the earning "
                    "carry-in to zero; only the true genesis may omit it",
                )
            if _fold_matches(ei.prior_accumulate_score, 0.0):
                return (ItemVerdictKind.PASS, "")
            return (
                ItemVerdictKind.FAIL,
                f"carry-in {ei.prior_accumulate_score} is nonzero but the log declares no "
                "prior epoch (genesis) — a genesis fold starts from 0.0",
            )
        # The log EXPLICITLY references a prior epoch (prior_log_digest is not None) but
        # that prior could not be loaded (pruned / unreadable / digest mismatch — see
        # load_prior_epoch_log). an internal review: a zero carry-in here is NOT safe to
        # PASS. Removing/censoring the prior object is exactly how the authority would
        # RESET accumulated earnings to zero — a zero carry-in against a referenced-but-
        # unavailable prior is UNVERIFIABLE, not "nothing to inflate". So it is a SKIP
        # (EARNING_STATE_UNVERIFIED ⇒ INCONCLUSIVE / HOLD), never a PASS. Only a GENUINE
        # genesis (prior_log_digest is None, handled above) may PASS a zero carry-in. A
        # nonzero carry-in is likewise unverifiable here. The referenced-but-unavailable
        # case is distinguished from genesis purely by prior_log_digest, so the auditor
        # tells them apart without conflating load_prior_epoch_log's two None reasons.
        return (
            ItemVerdictKind.SKIP,
            "the log references a prior epoch whose log could not be loaded, so the "
            f"carry-in {ei.prior_accumulate_score} is UNVERIFIABLE (even a zero one — an "
            "unavailable prior is how earnings would be reset/censored); chain it by "
            "supplying the prior_log — surfaced INCONCLUSIVE, never assumed honest",
        )

    def _carry_forward_verdict(
        self,
        uid: int,
        miner,
        log: EpochLog,
        prior_log: EpochLog | None,
        is_genesis: bool,
        base: dict,
    ) -> ItemVerdict:
        """Verify a uid's positive accumulator that carries NO earning evidence this epoch as
        a chained CARRY-FORWARD of the prior epoch.

        A positive ``accumulate_score`` can only ever be produced by a committed EWMA fold.
        A uid that carries one but has NO ``EarningInput`` this epoch (no cycles) must therefore
        be a pure carry-forward: its accumulator equals the prior epoch's audited value for the
        SAME (uid, hotkey). This is the normal idle/carry case and applies at ANY weight —
        round-20 #2 routes the NONZERO-weight carry-forward (an idle prior earner still weighted
        by its carried accumulator) through here too, not just the zero-weight one. Anything else
        is an accumulator injected while unaudited (nothing re-folds it) that could be re-attributed:

        - genesis (no prior): a positive accumulator with no committed fold cannot exist ⇒ FAIL;
        - prior available, hotkey CHANGED (#2b): a re-registered uid is a fresh identity that
          carries 0.0 — a positive accumulator is the previous owner's, inherited ⇒ FAIL;
        - prior available, TRACK CHANGED (round-18 #1): a carry-forward miner carries NO current
          evidence to bind a track to, so its track must CHAIN — it must match the track carried
          in the prior epoch's snapshot for this (uid, hotkey). A switch (e.g. to a non-paying
          track that substitutes a burn while preserving the accumulator) ⇒ METAGRAPH_TRACK_MISMATCH;
        - prior available, value != prior's ⇒ FAIL (an injected jump with no evidence);
        - prior available, value == prior's AND track unchanged ⇒ PASS (an honest carry-forward);
        - prior referenced but unavailable ⇒ SKIP (INCONCLUSIVE, fail closed).
        """
        if miner is None:  # defensive: only reached for a uid drawn from log.miners
            return ItemVerdict(
                verdict=ItemVerdictKind.SKIP,
                code=EARNING_STATE_UNVERIFIED,
                detail="a positive accumulator for a uid absent from the snapshots",
                **base,
            )
        acc = miner.accumulate_score
        if prior_log is not None:
            if log.prior_log_digest != prior_log.log_digest():
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_MISMATCH,
                    detail="prior_log_digest does not match the supplied prior epoch log — the "
                    "carry-forward chain is broken",
                    **base,
                )
            prior_miner = next((m for m in prior_log.miners if m.uid == uid), None)
            if prior_miner is not None and prior_miner.hotkey != miner.hotkey:
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_MISMATCH,
                    detail=f"uid {uid} carries a positive accumulator {acc} but its hotkey CHANGED "
                    f"vs the prior epoch ({prior_miner.hotkey!r} -> {miner.hotkey!r}) — a "
                    "re-registered hotkey is a fresh identity (0.0), not the previous owner's "
                    "accumulator",
                    **base,
                )
            # an internal review: BIND the TRACK of an IMPLICIT carry-forward miner. Track
            # verification (round-17 #3) only fires for miners carrying CURRENT-epoch evidence;
            # a carry-forward miner has NONE, so its track went unbound — and the burn-only
            # reconstruct preserves the raw log track. So an authority could carry a CLEAN
            # predecessor's positive accumulator forward under the SAME (uid, hotkey), switch its
            # paying track to a non-paying `unknown`, omit current evidence, and publish the
            # canonical burn vector: snapshot skips the unevidenced track, carry-forward passes,
            # reconstruct keeps `unknown`, weight collapses to burn ⇒ CLEAN. The carry-forward
            # track must CHAIN like the accumulator: it must match the track carried in the prior
            # epoch's snapshot for this (uid, hotkey). A changed/non-paying track that does NOT
            # match the carried track is a substituted burn ⇒ METAGRAPH_TRACK_MISMATCH (the FAIL
            # rides the earning channel and folds into the weight verdict ⇒ DISPUTED).
            if prior_miner is not None and prior_miner.track != miner.track:
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=METAGRAPH_TRACK_MISMATCH,
                    detail=f"uid {uid} carries accumulator {acc} forward with NO committed earning "
                    f"evidence this epoch but its track CHANGED vs the prior epoch "
                    f"({prior_miner.track!r} -> {miner.track!r}) — an implicit carry-forward miner "
                    "must carry the SAME track forward (a switch to a non-paying track with no "
                    "evidence substitutes a burn while preserving the accumulator, an internal review"
                    "#1)",
                    **base,
                )
            expected = prior_miner.accumulate_score if prior_miner is not None else 0.0
            if not _fold_matches(acc, expected):
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_MISMATCH,
                    detail=f"uid {uid} carries accumulator {acc} with NO committed earning evidence "
                    f"this epoch, but the prior epoch's value for this (uid, hotkey) is {expected} "
                    "— an accumulator injected while zero-weight",
                    **base,
                )
            return ItemVerdict(verdict=ItemVerdictKind.PASS, **base)
        # No prior log supplied.
        if log.prior_log_digest is None:
            if not is_genesis:
                return ItemVerdict(
                    verdict=ItemVerdictKind.FAIL,
                    code=EARNING_STATE_MISMATCH,
                    detail=f"uid {uid}: the log OMITS prior_log_digest at a NON-genesis epoch "
                    f"({log.epoch_id}) while carrying a positive accumulator {acc} — a broken "
                    "chain reset",
                    **base,
                )
            return ItemVerdict(
                verdict=ItemVerdictKind.FAIL,
                code=EARNING_STATE_MISMATCH,
                detail=f"uid {uid} carries a positive accumulator {acc} at GENESIS with no "
                "committed earning evidence — a genesis accumulator starts at 0.0 and a positive "
                "value requires a committed fold",
                **base,
            )
        return ItemVerdict(
            verdict=ItemVerdictKind.SKIP,
            code=EARNING_STATE_UNVERIFIED,
            detail=f"uid {uid} carries accumulator {acc} with no earning evidence this epoch and "
            "the referenced prior epoch could not be loaded — the carry-forward is UNVERIFIABLE "
            "(INCONCLUSIVE, never assumed honest)",
            **base,
        )

    # -- time-base weight-input binding --------------------------

    def _timebase_verdict(
        self, item_id: str, verdict: ItemVerdictKind, code: str, detail: str
    ) -> ItemVerdict:
        return ItemVerdict(
            source=_TIMEBASE_SOURCE,
            challenge_id="",
            item_id=item_id,
            miner_hotkey=None,
            uid=None,
            bundle_digest="",
            packet_digest="",
            verdict=verdict,
            code=code,
            detail=detail,
        )

    def _close_block_time(self, close_block: int) -> "datetime | None":
        """The CLOSE BLOCK's wall-clock time read from our OWN chain adapter (round-9 #6).

        None when the adapter is not wired, exposes no ``block_time`` (BlockTimeReadable), the
        time is unavailable, or the read RAISES — the caller then fails CLOSED to INCONCLUSIVE,
        never a PASS on an unverifiable ``created_at``. Never trusts the log's self-attested time.
        """
        chain = self._chain
        probe = getattr(chain, "block_time", None)
        if probe is None:
            return None
        try:
            return probe(close_block)
        except Exception:
            return None

    def _created_at_verdict(self, log: EpochLog) -> ItemVerdict:
        """Bind ``log.created_at`` to the epoch's CLOSE BLOCK time.

        ``created_at`` is the time base used to evaluate reward-window activity, so a
        BACKDATED value could keep an expired competition window active — a substituted
        weight input. The auditor reads the close_block's timestamp
        from the chain ITSELF and requires ``created_at`` to match within
        ``_CREATED_AT_TOL_SECONDS``:

        - close-block time UNREADABLE ⇒ SKIP (CREATED_AT_UNVERIFIED ⇒ INCONCLUSIVE), never a PASS;
        - ``|created_at - close_block_time| > tol`` ⇒ FAIL (CREATED_AT_MISMATCH ⇒ DISPUTED);
        - otherwise PASS.
        """
        close_time = self._close_block_time(log.close_block)
        if close_time is None:
            return self._timebase_verdict(
                "created_at",
                ItemVerdictKind.SKIP,
                CREATED_AT_UNVERIFIED,
                f"the close_block ({log.close_block}) time could not be read from the chain — "
                "created_at is UNVERIFIABLE; INCONCLUSIVE, never a PASS",
            )
        # an internal review: an offset-NAIVE created_at (EpochLog._validate rejects it, but a
        # log handed in directly / legacy bytes might carry one) cannot be subtracted from the
        # aware close-block time without a TypeError crash that would block the cursor. Treat a
        # naive created_at as a provable fault (FAIL ⇒ DISPUTED) rather than crash — fail closed.
        tz = log.created_at.tzinfo
        if tz is None or tz.utcoffset(log.created_at) is None:
            return self._timebase_verdict(
                "created_at",
                ItemVerdictKind.FAIL,
                CREATED_AT_MISMATCH,
                f"log created_at {log.created_at.isoformat()} is timezone-NAIVE — it cannot be "
                f"bound to the aware close_block ({log.close_block}) time; a naive time is "
                "unverifiable and fails closed",
            )
        skew = abs((log.created_at - close_time).total_seconds())
        if skew > _CREATED_AT_TOL_SECONDS:
            return self._timebase_verdict(
                "created_at",
                ItemVerdictKind.FAIL,
                CREATED_AT_MISMATCH,
                f"log created_at {log.created_at.isoformat()} disagrees with the close_block "
                f"({log.close_block}) time {close_time.isoformat()} by {skew:.0f}s (> "
                f"{_CREATED_AT_TOL_SECONDS:.0f}s tolerance) — the log's created_at does not "
                "bind to the epoch's close-block time",
            )
        return self._timebase_verdict("created_at", ItemVerdictKind.PASS, "", "")

    def _availability_anchor_error(
        self, observation: AvailabilityObservation, close_block: int
    ) -> tuple[ItemVerdictKind, str, str] | None:
        """Independently archive-check the signed request's finalized challenge anchor."""
        anchor = observation.attempt.request.metadata.commitment_anchor
        if anchor is None:
            return (
                ItemVerdictKind.FAIL,
                EARNING_STATE_MISMATCH,
                "availability observation has no finalized pre-dispatch challenge anchor",
            )
        if anchor.netuid != self._config.challenge_anchor_netuid:
            return (
                ItemVerdictKind.FAIL,
                EARNING_STATE_MISMATCH,
                f"availability anchor netuid {anchor.netuid} != expected "
                f"{self._config.challenge_anchor_netuid}",
            )
        if anchor.block > close_block:
            return (
                ItemVerdictKind.FAIL,
                EARNING_STATE_MISMATCH,
                f"availability challenge anchor block {anchor.block} is after epoch close "
                f"block {close_block}",
            )
        if anchor.block_hash is None:
            return (
                ItemVerdictKind.SKIP,
                EARNING_STATE_UNVERIFIED,
                "availability challenge anchor has no finalized block hash",
            )
        chain = self._chain
        if chain is None:
            return (
                ItemVerdictKind.SKIP,
                EARNING_STATE_UNVERIFIED,
                "no independent chain adapter is wired for availability challenge evidence",
            )
        finalized_reader = getattr(chain, "finalized_block", None)
        block_hash_reader = getattr(chain, "block_hash", None)
        archive_reader = getattr(chain, "read_anchor_at", None)
        if not all(
            callable(reader)
            for reader in (finalized_reader, block_hash_reader, archive_reader)
        ):
            return (
                ItemVerdictKind.SKIP,
                EARNING_STATE_UNVERIFIED,
                "chain adapter lacks finalized_block/block_hash/read_anchor_at seams for "
                "availability challenge evidence",
            )
        try:
            finalized = int(finalized_reader())
            observed_block_hash = block_hash_reader(anchor.block)
            observed_digest = archive_reader(
                netuid=anchor.netuid,
                epoch_id=anchor.dispatch_ordering_key,
                domain=CHALLENGE_ANCHOR_DOMAIN,
                block_number=anchor.block,
            )
        except Exception as exc:
            return (
                ItemVerdictKind.SKIP,
                EARNING_STATE_UNVERIFIED,
                "availability challenge archive/finality read unavailable: "
                f"{type(exc).__name__}: {exc}",
            )
        if finalized < anchor.block:
            return (
                ItemVerdictKind.SKIP,
                EARNING_STATE_UNVERIFIED,
                f"availability challenge anchor block {anchor.block} is not finalized "
                f"(finalized={finalized})",
            )
        if observed_block_hash is None:
            return (
                ItemVerdictKind.SKIP,
                EARNING_STATE_UNVERIFIED,
                f"availability anchor block hash {anchor.block} is not independently readable",
            )
        normalized_hash = str(observed_block_hash).lower().removeprefix("0x")
        if normalized_hash != anchor.block_hash:
            return (
                ItemVerdictKind.FAIL,
                EARNING_STATE_MISMATCH,
                "availability challenge anchor block hash does not match chain history",
            )
        if observed_digest != anchor.commitment_hash:
            return (
                ItemVerdictKind.FAIL,
                EARNING_STATE_MISMATCH,
                "archive state at the availability anchor block does not contain the "
                "signed challenge commitment",
            )
        return None

    def _availability_evidence_fields(
        self, log: EpochLog, uid: int, miner
    ) -> tuple[
        dict[str, dict],
        dict[str, dict],
        tuple[ItemVerdictKind, str, str] | None,
    ]:
        """Authenticate canonical non-media zeros and expose packet-compatible fields.

        The returned maps deliberately use the same shapes as media packet/challenge
        evidence. That keeps exact-set, ordering, EWMA, replay and watermark validation on
        one path instead of inventing a weaker availability-only fold.
        """
        raw_inputs = tuple(log.audit_manifest.availability_inputs)
        if not raw_inputs:
            return ({}, {}, None)
        try:
            evidence_inputs = tuple(
                AvailabilityInput.model_validate(evidence.model_dump(mode="python"))
                for evidence in raw_inputs
            )
        except Exception as exc:
            return (
                {},
                {},
                (
                    ItemVerdictKind.FAIL,
                    EARNING_STATE_MISMATCH,
                    "availability manifest input is malformed/non-canonical: "
                    f"{type(exc).__name__}: {exc}",
                ),
            )

        digests = [evidence.observation_digest for evidence in evidence_inputs]
        identities = [
            (evidence.uid, evidence.challenge_id, evidence.item_id)
            for evidence in evidence_inputs
        ]
        ordering_keys = [
            (evidence.uid, evidence.ordering_key) for evidence in evidence_inputs
        ]
        if (
            len(set(digests)) != len(digests)
            or len(set(identities)) != len(identities)
            or len(set(ordering_keys)) != len(ordering_keys)
        ):
            return (
                {},
                {},
                (
                    ItemVerdictKind.FAIL,
                    EARNING_STATE_MISMATCH,
                    "availability manifest repeats a digest, uid/challenge/item identity, "
                    "or per-uid committed ordering key",
                ),
            )
        media_digests = {
            ref.digest
            for refs in (
                tuple(log.audit_manifest.per_uid.values())
                + (log.audit_manifest.baseline_bundles,)
                + tuple(log.audit_manifest.competition_bundles.values())
            )
            for ref in refs
        }
        if set(digests) & media_digests:
            return (
                {},
                {},
                (
                    ItemVerdictKind.FAIL,
                    EARNING_STATE_MISMATCH,
                    "availability observation digest is also presented as media evidence",
                ),
            )

        packet_fields: dict[str, dict] = {}
        challenge_fields: dict[str, dict] = {}
        census_by_uid = {entry.uid: entry for entry in log.miner_census}
        for evidence in evidence_inputs:
            if evidence.uid != uid:
                continue
            try:
                observation = AvailabilityObservation.model_validate_json(
                    evidence.observation_json
                )
            except (TypeError, ValueError) as exc:
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        EARNING_STATE_MISMATCH,
                        "availability observation is malformed: "
                        f"{type(exc).__name__}: {exc}",
                    ),
                )
            if (
                observation.canonical_bytes().decode("utf-8")
                != evidence.observation_json
                or observation.digest() != evidence.observation_digest
            ):
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        EARNING_STATE_MISMATCH,
                        "availability observation canonical bytes/digest do not match the "
                        "manifest commitment",
                    ),
                )
            try:
                signature_ok = (
                    verify_availability_observation(observation)
                    if self._availability_verify_fn is None
                    else verify_availability_observation(
                        observation, verify_fn=self._availability_verify_fn
                    )
                )
            except (
                Exception
            ) as exc:  # defense for an injected verifier outside the helper
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.SKIP,
                        EARNING_STATE_UNVERIFIED,
                        "availability signature verifier unavailable: "
                        f"{type(exc).__name__}: {exc}",
                    ),
                )
            if not signature_ok:
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        IDENTITY_MISMATCH,
                        "availability request/observation/miner-receipt signature is invalid",
                    ),
                )
            attempt = observation.attempt
            declared = (
                evidence.uid,
                evidence.hotkey,
                evidence.challenge_id,
                evidence.item_id,
                evidence.track,
            )
            observed = (
                attempt.uid,
                attempt.miner_hotkey,
                attempt.challenge_id,
                attempt.item_id,
                attempt.track,
            )
            if declared != observed:
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        IDENTITY_MISMATCH,
                        f"availability manifest identity/track {declared!r} does not match "
                        f"the signed observation {observed!r}",
                    ),
                )
            if observation.score != 0.0:
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        EARNING_STATE_MISMATCH,
                        "availability evidence commits a nonzero economic score",
                    ),
                )
            anchor = attempt.request.metadata.commitment_anchor
            if anchor is None or anchor.dispatch_ordering_key != evidence.ordering_key:
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        EARNING_STATE_MISMATCH,
                        "availability CycleScore ordering key is not the signed, committed "
                        "dispatch ordering key",
                    ),
                )
            census = census_by_uid.get(uid)
            if census is None or census.hotkey != evidence.hotkey:
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        CENSUS_MISMATCH,
                        "availability evidence uid/hotkey is not the same close-block "
                        "miner_census identity",
                    ),
                )
            if (
                miner is None
                or miner.hotkey != evidence.hotkey
                or miner.track != evidence.track
            ):
                return (
                    {},
                    {},
                    (
                        ItemVerdictKind.FAIL,
                        IDENTITY_MISMATCH,
                        "availability evidence identity/track is not the same economic "
                        "miner snapshot identity/track",
                    ),
                )
            anchor_error = self._availability_anchor_error(observation, log.close_block)
            if anchor_error is not None:
                return ({}, {}, anchor_error)
            packet_fields[evidence.observation_digest] = {
                "score": 0.0,
                "cycle_sequence": evidence.ordering_key,
                "excluded": False,
            }
            challenge_fields[evidence.observation_digest] = {
                "track": evidence.track,
                "ordering_key": evidence.ordering_key,
                "ref_committed_track": evidence.track,
            }
        return (packet_fields, challenge_fields, None)

    def _committed_packet_fields(
        self, log: EpochLog, uid: int, store: AuditStore
    ) -> dict[str, dict] | None:
        """`digest -> {"score","cycle_sequence","excluded"}` for the uid's committed
        SCORE_PACKETs (None if any cannot be read / lacks a committed ordering key — the
        earning fold is then UNVERIFIABLE, a SKIP not a PASS)."""
        out: dict[str, dict] = {}
        for ref in log.audit_manifest.refs_for(uid):
            if ref.kind is not AuditFileKind.SCORE_PACKET:
                continue
            fields = self._read_packet_fields(store, ref.digest)
            if fields is None:
                return None
            out[ref.digest] = fields
        return out

    def _earning_identity_error(
        self, log: EpochLog, uid: int, store: AuditStore, expected_hotkey: str | None
    ) -> tuple[str, str] | None:
        """Prove each resolved bundle AUTHENTICATES the packet it is paired with (an internal review, extended round-6 #1). Returns ``(IDENTITY_MISMATCH, detail)`` on a
        PROVABLE substitution — a resolved bundle whose own ``score_packet.digest`` is not
        the manifest's SCORE_PACKET ref digest, or whose ``challenge_id``/``item_id`` does
        not match the manifest ref, or whose ``miner_hotkey`` is not NON-NULL-and-equal to
        the uid's hotkey, or whose SCORE PACKET's OWN INTERNAL identity (the
        challenge_id/item_id/miner_hotkey inside the packet JSON) does not match the ref /
        the uid's hotkey. Returns None when everything binds OR when a bundle/packet is
        merely UNRESOLVABLE (that is not a fault — it is left to the fail-closed SKIP
        paths, which surface UNVERIFIED/INCONCLUSIVE).

        Round-6 #1 closed two holes: (a) the miner check was SKIPPED when the bundle's
        ``miner_hotkey`` was None, so nulling it dodged attribution; and (b) the packet's
        OWN internal identity was never read, so a bundle LABELLED for the targeted uid could
        wrap a foreign miner's high-scoring packet. Both now fail closed as substitutions.

        The manifest pairs refs by authority-supplied (challenge_id, item_id) labels; this
        is the check that a resolved bundle is genuinely THIS packet's bundle before its
        committed (track, ordering_key) and score are trusted."""
        refs = log.audit_manifest.refs_for(uid)
        bundle_refs = {
            (r.challenge_id, r.item_id): r
            for r in refs
            if r.kind is AuditFileKind.AUDIT_BUNDLE
        }
        for ref in refs:
            if ref.kind is not AuditFileKind.SCORE_PACKET:
                continue
            bref = bundle_refs.get((ref.challenge_id, ref.item_id))
            if bref is None:
                continue  # missing bundle ref → fail-closed SKIP path handles it
            try:
                bundle = self._bundle_source.bundle_for(bref)
            except BundleUnavailable:
                continue  # unresolvable → not a fault; SKIP path handles it
            if bundle.score_packet.digest != ref.digest:
                return (
                    IDENTITY_MISMATCH,
                    f"uid {uid}: the bundle paired with score packet {ref.digest} "
                    f"({ref.challenge_id}/{ref.item_id}) authenticates a DIFFERENT packet "
                    f"{bundle.score_packet.digest} — a substituted bundle cannot back this "
                    "packet's earning score/committed order",
                )
            if bundle.challenge_id != ref.challenge_id or bundle.item_id != ref.item_id:
                return (
                    IDENTITY_MISMATCH,
                    f"uid {uid}: resolved bundle identity {bundle.challenge_id}/"
                    f"{bundle.item_id} does not match its manifest ref {ref.challenge_id}/"
                    f"{ref.item_id} — a substituted bundle",
                )
            # (a) the resolved bundle must PIN a miner, and it must be THIS uid's hotkey
            #. A None (calibration-style, unattributed) or a foreign
            # miner on an EARNING-backing bundle is an IDENTITY_MISMATCH: a score whose
            # miner cannot be attributed to the uid must never fold into that uid. The
            # bundle model permits a None miner for non-attributed calibration runs, but
            # those never legitimately back a uid's earning state — so None fails closed
            # here (the previous check SKIPPED the comparison when miner was None, letting
            # the authority dodge it by nulling the bundle miner).
            if expected_hotkey is not None and bundle.miner_hotkey != expected_hotkey:
                return (
                    IDENTITY_MISMATCH,
                    f"uid {uid}: the resolved bundle's miner {bundle.miner_hotkey!r} is not "
                    f"the uid's hotkey {expected_hotkey!r} — a None/foreign miner on an "
                    "earning-backing bundle cannot attribute this score to the uid",
                )
            # (b) the SCORE PACKET's OWN INTERNAL identity (the challenge_id/item_id/
            # miner_hotkey inside the packet JSON) must bind to the manifest ref's
            # (challenge_id, item_id) and the uid's hotkey. The outer
            # bundle labels alone do not prove the packet INSIDE was minted for this uid:
            # an authority could wrap hk99's high-scoring packet in a bundle LABELLED for
            # the targeted uid/hotkey. An UNREADABLE packet is not faulted here (the
            # fail-closed SKIP paths below surface it as UNVERIFIED); a READABLE packet
            # whose self-declared identity contradicts the ref / the uid's hotkey is a
            # conclusive substitution.
            ident = self._read_packet_identity(store, ref.digest)
            if ident is not None:
                if (
                    ident["challenge_id"] != ref.challenge_id
                    or ident["item_id"] != ref.item_id
                ):
                    return (
                        IDENTITY_MISMATCH,
                        f"uid {uid}: score packet {ref.digest} declares its own identity "
                        f"{ident['challenge_id']!r}/{ident['item_id']!r}, which does not "
                        f"match its manifest ref {ref.challenge_id!r}/{ref.item_id!r} — a "
                        "packet minted for a different item cannot back this uid's earning",
                    )
                if (
                    expected_hotkey is not None
                    and ident["miner_hotkey"] != expected_hotkey
                ):
                    return (
                        IDENTITY_MISMATCH,
                        f"uid {uid}: score packet {ref.digest} was minted for miner "
                        f"{ident['miner_hotkey']!r}, not the uid's hotkey "
                        f"{expected_hotkey!r} — a packet mined by another hotkey cannot "
                        "fold into this uid's earning state",
                    )
        return None

    def _committed_challenge_fields(
        self, log: EpochLog, uid: int, store: AuditStore
    ) -> dict[str, dict] | None:
        """`packet_digest -> {"track","ordering_key","ref_committed_track"}` sourced from
        each item's CHALLENGE COMMITMENT (the pre-dispatch, anchored DAG_REVEAL preimage,
        resolved via the item's bundle) — the NON-circular binding for the fold order/track
. None if any commitment cannot be read: for a nonzero-weight
        uid the caller then FAILS CLOSED (EARNING_STATE_UNVERIFIED → INCONCLUSIVE/HOLD),
        because the committed binding is UNVERIFIABLE and must never wash to PASS/CLEAN via
        the packet-bound fallback."""
        refs = log.audit_manifest.refs_for(uid)
        bundle_refs = {
            (r.challenge_id, r.item_id): r
            for r in refs
            if r.kind is AuditFileKind.AUDIT_BUNDLE
        }
        out: dict[str, dict] = {}
        for ref in refs:
            if ref.kind is not AuditFileKind.SCORE_PACKET:
                continue
            bref = bundle_refs.get((ref.challenge_id, ref.item_id))
            if bref is None:
                return None
            try:
                bundle = self._bundle_source.bundle_for(bref)
            except BundleUnavailable:
                return None
            if (
                self._challenge_chronology(bundle, store).kind
                is not ChronologyKind.PASS
            ):
                return None
            committed = self._read_committed_dispatch(store, bundle)
            if committed is None:
                return None
            track, ordering_key = committed
            out[ref.digest] = {
                "track": track,
                "ordering_key": ordering_key,
                "ref_committed_track": ref.committed_track,
            }
        return out

    @staticmethod
    def _read_committed_dispatch(
        store: AuditStore, bundle: AuditBundle
    ) -> tuple[str, int] | None:
        """Fetch the item's DAG_REVEAL (the anchored commitment preimage), VERIFY it hashes
        to the bundle's `commitment_hash` (the anchored commit), and read the committed
        `(track, dispatch_ordering_key)`. None if unreadable / not a well-formed preimage."""
        reveal_ref = getattr(bundle, "dag_reveal", None)
        if reveal_ref is None:
            return None
        try:
            data = store.get_limited(reveal_ref, _MAX_AUDIT_METADATA_BYTES)
        except (IntegrityError, FileNotFoundError, OSError):
            return None
        if sha256_hex(data) != bundle.commitment_hash:
            return None
        return ChallengeCommitment.committed_dispatch_from_preimage(data)

    @staticmethod
    def _read_packet_fields(store: AuditStore, digest: str) -> dict | None:
        """Fetch a SCORE_PACKET by digest (verify-on-read); return its committed
        earning fields, or None if unreadable / missing the committed ordering key.

        The ``cycle_sequence`` (the fold ORDER's evidence anchor) MUST be present and
        integral: a packet that does not commit its cycle sequence cannot anchor a
        verifiable fold order, so the fold is unverifiable (SKIP → INCONCLUSIVE), never
        a PASS. ``excluded`` (default False) marks an evidenced exclusion cycle.
        """
        import json

        try:
            data = store.get_digest_limited(
                ArtifactKind.SCORE_PACKET,
                digest,
                max_bytes=_MAX_AUDIT_METADATA_BYTES,
            )
        except (IntegrityError, FileNotFoundError, OSError):
            return None
        try:
            payload = json.loads(data)
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict) or "score" not in payload:
            return None
        seq = payload.get("cycle_sequence")
        if not isinstance(seq, int) or isinstance(seq, bool):
            return None
        try:
            score = float(payload["score"])
        except (TypeError, ValueError):
            return None
        return {
            "score": score,
            "cycle_sequence": seq,
            "excluded": bool(payload.get("excluded", False)),
        }

    @staticmethod
    def _read_packet_identity(store: AuditStore, digest: str) -> dict | None:
        """Fetch a SCORE_PACKET by digest (verify-on-read) and return its OWN INTERNAL
        identity ``{"challenge_id", "item_id", "miner_hotkey"}`` — the fields the packet
        JSON carries about itself — or None if unreadable / not a well-formed packet.

        Deliberately SEPARATE from ``_read_packet_fields`` (which reads the earning fields)
        so the earning-field reads are not weakened. Used to bind a packet's SELF-declared
        identity to the manifest ref + the uid's hotkey: the outer
        bundle labels alone do not prove the packet inside was minted for THIS uid. A None
        return is NOT a fault (the fail-closed SKIP paths handle an unreadable packet);
        only a READABLE packet whose identity contradicts the ref/uid is a substitution.
        """
        import json

        try:
            data = store.get_digest_limited(
                ArtifactKind.SCORE_PACKET,
                digest,
                max_bytes=_MAX_AUDIT_METADATA_BYTES,
            )
        except (IntegrityError, FileNotFoundError, OSError):
            return None
        try:
            payload = json.loads(data)
        except (ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        return {
            "challenge_id": payload.get("challenge_id"),
            "item_id": payload.get("item_id"),
            "miner_hotkey": payload.get("miner_hotkey"),
        }
