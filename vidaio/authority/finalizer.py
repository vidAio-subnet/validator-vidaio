"""The Scoring Authority's epoch-log FINALIZER (the central producer).

At each epoch close the central authority has the folded inference `MinerSnapshot`
set, the audit bundles + score packets that back each score, and (when completed)
the packet-derived competition input/result that advances the reward window. This module
turns those inputs into the ONE immutable `EpochLog` artifact and writes it to the
object store behind a `_FINALIZED` marker, so validators can mirror byte-identical
bytes and auditors can verify them (the project design record §1, §4; build-wave 3).

Two pieces:

- `build_audit_manifest(scored_items, store=...)` — the manifest-assembly contract:
  given the epoch's scored items and their real, stored `bundle_digest`/`packet_digest`
  from the competition/performance rows, it builds the `AuditManifest`. HONEST by
  construction — when a store is supplied every score-packet digest is probed and a
  missing one raises `AuditFileMissingError`: a weight is never backed by an audit
  file that was never stored (the project design record integrity invariants).
- `EpochFinalizer.finalize(...)` — computes the canonical weight vector via
  `build_weight_vector` (with the empty-epoch `burn_uid` path), quantizes via
  `quantize_u16`, assembles + validates the `EpochLog`, writes it as a `_FINALIZED`
  set (member first, marker LAST), and returns a `FinalizedEpoch` pointer
  (`snapshot_key` + `log_digest` + `weight_vector_digest`) for the API + on-chain anchor.
  Idempotent: re-finalizing an already-`_FINALIZED` epoch is a NO-OP that reads the
  stored log back and returns the SAME key/digest (never rewrites — a finalized set is
  immutable, wave-2 convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from vidaio.audit.canonical import SHA256_HEX_PATTERN, sha256_hex
from vidaio.audit.commitments import merkle_proof, merkle_root
from vidaio.audit.store import (
    ArtifactKind,
    ArtifactRef,
    backend_key,
    set_member_key,
)
from vidaio.competition.economic_result import derive_competition_result
from vidaio.epoch.log import (
    AuditFileKind,
    AuditFileRef,
    AuditManifest,
    AvailabilityInput,
    CompetitionInput,
    CycleScore,
    EarningInput,
    EpochLog,
    EpochLogInvalid,
    MinerCensusEntry,
    weight_vector_digest,
)
from vidaio.tokenomics.breakthrough import resolve_reward_window
from vidaio.tokenomics.config import TokenomicsConfig
from vidaio.tokenomics.ewma import EXCLUDED_SCORE as EXCLUSION_SENTINEL
from vidaio.tokenomics.ewma import accumulate, is_excluded
from vidaio.tokenomics.quantize import quantize_u16
from vidaio.tokenomics.state import (
    CompetitionResult,
    EmissionState,
    MinerSnapshot,
    RewardWindowState,
)
from vidaio.tokenomics.weights import build_weight_vector
from vidaio.validator.availability import (
    AvailabilityObservation,
    verify_availability_observation,
)

#: Tolerance for the finalizer's own earning-fold self-check (float EWMA rounding).
_FOLD_TOL = 1e-9

#: The object-store member name of the epoch-log inside its `_FINALIZED` set, and
#: the set-prefix scheme (the project design record §4 key scheme).
EPOCH_LOG_MEMBER = "log.json"


def epoch_prefix(epoch_id: int) -> str:
    """`finalized/epoch={N}` — the set prefix an epoch's log lives under."""
    return f"finalized/epoch={epoch_id}"


class AuditFileMissingError(ValueError):
    """A manifest references an audit file that is not present in the object store.

    The finalizer refuses to publish a manifest that points an auditor at bytes that
    were never stored — the manifest must be a followable index, not a dead link.
    """


class ChallengeCommitmentSource(Protocol):
    """Looks up a challenge's PRE-DISPATCH committed `(track, dispatch_ordering_key)`.

    Production backs this with the challenge store / anchored commitment records (the
    `challenge_commitments` table, or the committed DAG_REVEAL preimage) — the SAME
    commitment the auditor independently re-reads. It is what makes the earning fold
    non-circular: the finalizer may NOT invent the fold
    order or the track at finalization; it SOURCES both from the commitment that was
    fixed and anchored before any score existed. Returns None when the item's challenge
    has no such commitment — the finalizer then REFUSES to enter it into an auditable
    earning fold.
    """

    def committed_dispatch(self, challenge_id: str) -> tuple[str, int] | None: ...


class _FinalizedSetStore(Protocol):
    """The slice of the object store the finalizer needs (the `_FINALIZED` set
    convention from wave 2; `LocalFsStore` / `S3Store` both provide it)."""

    def is_finalized(self, prefix: str) -> bool: ...

    def put_set_member(
        self, prefix: str, name: str, data: bytes, kind: ArtifactKind = ...
    ) -> ArtifactRef: ...

    def finalize_set(self, prefix: str) -> None: ...

    def get_set_member(
        self,
        prefix: str,
        name: str,
        *,
        expected_digest: str | None = None,
        byte_size: int | None = None,
    ) -> bytes: ...

    def exists(self, ref: ArtifactRef) -> bool: ...


class _PublicReleaseStore(Protocol):
    """Anonymous/keyless release view used for the CROWN publication interlock."""

    @property
    def public_read_only(self) -> bool: ...

    def is_released(self, ref: ArtifactRef) -> bool: ...


@dataclass(frozen=True, slots=True)
class ScoredItem:
    """One scored (uid, item) row the authority produced this epoch.

    Carries the REAL, stored digests of the audit files backing it: `bundle_digest`
    (the `AuditBundle.bundle_digest()`) and `packet_digest` (the stored
    `ArtifactKind.SCORE_PACKET` object). `baseline=True` marks a non-earning baseline
    calibration row (the project design record #1) — audited but never mapped to a weight.
    """

    uid: int
    hotkey: str
    challenge_id: str
    item_id: str
    bundle_digest: str
    packet_digest: str
    #: The item's COMMITTED scoring track (from the committed challenge), stamped onto
    #: the SCORE_PACKET ref so the auditor recomputes over the committed track, not the
    #: authority's packet-declared one (#9). REQUIRED — the finalizer always emits it,
    #: and a SCORE_PACKET ref without it is refused (`AuditFileRef`); it must equal the
    #: `track` the committed score packet records (the auditor cross-checks).
    committed_track: str
    source: str = "inference"  # "inference" | "competition"
    baseline: bool = False
    #: Manifest namespace for a competition contender/baseline. Competition rows
    #: never enter the inference EWMA fold; their packet means derive CompetitionResult.
    competition_subject: str | None = None
    #: The recorded score of THIS scored item — the per-cycle score that EWMA-folds
    #: into the uid's `accumulate_score`. Folds into the manifest's `EarningInput` so
    #: the auditor can re-derive the earning state from the audited packets (#1). None
    #: means the caller did not attest a cycle score for this item — then NO earning
    #: input is emitted for its uid and the finalizer REFUSES to finalize a nonzero-weight
    #: uid that lacks a complete earning input (#1). Production always sets it.
    score: float | None = None
    #: The COMMITTED monotonic cycle-sequence index of this scored item — the VERIFIABLE
    #: fold order. It is recorded IN the content-addressed score packet
    #: (as `cycle_sequence`), so the auditor re-reads it from the committed packet and
    #: rejects any fold order that is not ascending by it. REQUIRED whenever `score` is
    #: attested (an earning cycle); build_audit_manifest raises otherwise.
    cycle_sequence: int | None = None
    #: True marks this an EVIDENCED exclusion cycle: the committed packet records the
    #: uid as excluded and the folded value is the -1.0 exclusion sentinel (never a
    #: packet score). An exclusion entry that no committed packet backs is refused.
    excluded_cycle: bool = False


def _refs_for_item(
    item: ScoredItem,
    *,
    committed_track: str,
    inclusion_proof: tuple[tuple[str, str], ...] | None = None,
) -> tuple[AuditFileRef, AuditFileRef]:
    """The two audit files an auditor fetches for one item: its bundle + its packet.

    The SCORE_PACKET ref carries the per-item merkle `inclusion_proof` (opening its
    digest against the manifest's committed `score_packet_merkle_root`) and the
    COMMITTED `committed_track` (sourced from the challenge commitment when a
    `commitment_source` is supplied, never invented at finalization);
    the AUDIT_BUNDLE ref carries neither (bundles are proven by recompute, not leaf
    inclusion).
    """
    common = dict(
        challenge_id=item.challenge_id, item_id=item.item_id, source=item.source
    )
    return (
        AuditFileRef(
            kind=AuditFileKind.AUDIT_BUNDLE, digest=item.bundle_digest, **common
        ),
        AuditFileRef(
            kind=AuditFileKind.SCORE_PACKET,
            digest=item.packet_digest,
            inclusion_proof=inclusion_proof,
            committed_track=committed_track,
            **common,
        ),
    )


def _committed_dispatch_for(
    item: ScoredItem, commitment_source: ChallengeCommitmentSource | None
) -> tuple[str, int | None]:
    """The `(committed_track, committed_ordering_key)` to stamp for `item`.

    With a `commitment_source` (production), BOTH are SOURCED from the challenge's
    pre-dispatch commitment — the finalizer may not invent them. An
    item whose challenge has no such commitment is REFUSED: it cannot enter an auditable
    earning fold whose order/track was never pre-committed. Without a source (pure-model
    tests) the item's own stated fields are used, exactly as before.
    """
    if item.source == "competition":
        # Competition provenance is the pre-enrollment CompetitionInput/manifest
        # commitment, not an inference degradation-DAG dispatch commitment.
        return item.committed_track, None
    if commitment_source is None:
        return item.committed_track, item.cycle_sequence
    committed = commitment_source.committed_dispatch(item.challenge_id)
    if committed is None:
        raise AuditFileMissingError(
            f"scored item {item.item_id} (challenge {item.challenge_id}) has NO pre-dispatch "
            "challenge commitment of (track, dispatch_ordering_key) — the finalizer refuses to "
            "enter an item whose fold order/track was not committed before scoring into an "
            "auditable earning fold"
        )
    track, ordering_key = committed
    return track, ordering_key


def _verify_availability_input(
    evidence: AvailabilityInput,
    *,
    commitment_source: ChallengeCommitmentSource | None,
    verify_fn: Callable[[str, bytes, str], bool] | None,
) -> AvailabilityInput:
    """Revalidate and authenticate one persisted non-media zero before folding it."""
    # Reconstruct from plain data so callers cannot bypass the dependency-light model's
    # canonical/digest checks with ``model_construct`` or an unvalidated ``model_copy``.
    normalized = AvailabilityInput.model_validate(evidence.model_dump(mode="python"))
    try:
        observation = AvailabilityObservation.model_validate_json(
            normalized.observation_json
        )
    except (TypeError, ValueError) as exc:
        raise AuditFileMissingError(
            f"availability evidence for item {normalized.item_id} is not a valid "
            "AvailabilityObservation"
        ) from exc

    if observation.canonical_bytes().decode("utf-8") != normalized.observation_json:
        raise AuditFileMissingError(
            f"availability evidence for item {normalized.item_id} is not canonical"
        )
    if observation.digest() != normalized.observation_digest:
        raise AuditFileMissingError(
            f"availability evidence for item {normalized.item_id} has a digest mismatch"
        )
    verified = (
        verify_availability_observation(observation)
        if verify_fn is None
        else verify_availability_observation(observation, verify_fn=verify_fn)
    )
    if not verified:
        raise AuditFileMissingError(
            f"availability evidence for item {normalized.item_id} has an invalid signature"
        )
    if observation.score != 0.0:
        raise AuditFileMissingError(
            f"availability evidence for item {normalized.item_id} is not exact score=0"
        )

    attempt = observation.attempt
    declared_identity = (
        normalized.uid,
        normalized.hotkey,
        normalized.challenge_id,
        normalized.item_id,
        normalized.track,
    )
    observed_identity = (
        attempt.uid,
        attempt.miner_hotkey,
        attempt.challenge_id,
        attempt.item_id,
        attempt.track,
    )
    if declared_identity != observed_identity:
        raise AuditFileMissingError(
            f"availability evidence identity/track mismatch for item {normalized.item_id}: "
            f"declared {declared_identity!r}, observation {observed_identity!r}"
        )

    anchor = attempt.request.metadata.commitment_anchor
    if anchor is None or anchor.dispatch_ordering_key != normalized.ordering_key:
        observed_key = None if anchor is None else anchor.dispatch_ordering_key
        raise AuditFileMissingError(
            f"availability evidence for item {normalized.item_id} declares committed "
            f"ordering_key {normalized.ordering_key}, but its signed request anchors "
            f"{observed_key!r}"
        )
    if commitment_source is not None:
        committed = commitment_source.committed_dispatch(normalized.challenge_id)
        expected = (normalized.track, normalized.ordering_key)
        if committed != expected:
            raise AuditFileMissingError(
                f"availability evidence for item {normalized.item_id} does not match its "
                f"pre-dispatch challenge commitment: expected {expected!r}, got "
                f"{committed!r}"
            )
    return normalized


def build_audit_manifest(
    scored_items: Iterable[ScoredItem],
    *,
    store: _FinalizedSetStore | None = None,
    score_packet_merkle_root: str | None = None,
    prior_accumulate: Mapping[int, float] | None = None,
    prior_fold_cursors: Mapping[int, int | None] | None = None,
    current_census_uids: Iterable[int] = (),
    commitment_source: ChallengeCommitmentSource | None = None,
    competition_input: CompetitionInput | None = None,
    availability_evidence: Iterable[AvailabilityInput] = (),
    availability_verify_fn: Callable[[str, bytes, str], bool] | None = None,
) -> AuditManifest:
    """Assemble the `AuditManifest` from the epoch's scored items (the contract).

    Groups each earning row under its uid (two refs: bundle + packet) and collects
    baseline rows into `baseline_bundles`. When `store` is given, every SCORE_PACKET **and
    every AUDIT_BUNDLE** digest is probed and a missing one raises
    `AuditFileMissingError` — a weight is never backed by an audit file (packet OR
    resolvable bundle) that was never stored (#8: the finalizer guarantees a
    resolvable manifest, so an auditor can never be pointed at a dead link).

    The score-packet digests of ALL rows (earning + baseline) are the leaves of a merkle
    tree; the manifest carries its `score_packet_merkle_root` and each SCORE_PACKET ref
    carries its own **inclusion proof**, so an auditor can PROVE committed-set membership
    of every sampled item (MERKLE_EXCLUSION otherwise). Both are DETERMINISTIC from the
    sorted leaves — the manifest, and therefore the EpochLog it feeds, stays
    byte-identical across machines. The root is also folded into `log_digest` and thus
    anchored on chain, so post-finalization tampering is provable. A caller-supplied
    `score_packet_merkle_root` overrides the computed one (used only by report-mode
    fixtures); by default it is computed here.

    `earning_inputs` (#1): per earning uid, the ordered `cycle_scores` (the scored
    items' scores, in the order given — EWMA is history-dependent) plus the carry-in
    from `prior_accumulate` (the prior epoch's `accumulate_score` for that uid, 0.0 at
    genesis). This is what lets the auditor RE-FOLD `accumulate_score` from audited
    evidence rather than trusting the authority's stated value.

    ``prior_fold_cursors`` (schema v14) is the complete cumulative replay-boundary map
    from the chained predecessor, including ``None`` tombstones for identities observed but
    never folded. It is copied even when this epoch has no items, every uid in
    ``current_census_uids`` is added as ``None`` if it has never been seen, and each uid's
    current committed keys must strictly exceed an integer prior cursor before the map is
    advanced. A first fold after ``None`` is allowed. This makes carry-only and empty epochs
    preserve identity history instead of reopening old packets for a later fold.

    `commitment_source`: when supplied, an INFERENCE
    item's committed track and cycle ordering key are sourced from its pre-dispatch
    challenge commitment — never from the finalization-time ``ScoredItem``. Competition
    items are not inference-EWMA cycles and live in the separately pre-committed
    ``CompetitionInput`` matrix, so they take their track from that input and carry no
    fold ordering key. An inference item whose challenge has no commitment is refused.

    ``availability_evidence`` carries signed, request-bound economic zeros read from
    the committed validator ledger. Each observation is canonical/digest checked,
    cryptographically verified, and bound to its declared uid/hotkey/item/track and
    pre-dispatch ordering key. Its digest enters the same ordered ``CycleScore`` fold
    and replay watermark as a media packet, but never enters media refs or the media
    score-packet Merkle tree. ``availability_verify_fn`` is an injectable CPU verifier;
    ``None`` uses the production Bittensor hotkey verifier.

    The old `window_inputs` parameter remains removed with the retention multiplier.
    Competition evidence instead uses the explicit `competition_input` plus namespaced
    packet/bundle worklists and never enters the inference EWMA fold.
    """
    items = list(scored_items)
    availability_rows = list(availability_evidence)
    priors = dict(prior_accumulate or {})
    fold_cursors = {
        int(uid): None if ordering_key is None else int(ordering_key)
        for uid, ordering_key in (prior_fold_cursors or {}).items()
    }
    for uid in current_census_uids:
        fold_cursors.setdefault(int(uid), None)
    # The committed leaf set: every item's score-packet digest (earning + baseline).
    packet_digests = [item.packet_digest for item in items]
    root = score_packet_merkle_root
    if root is None and packet_digests:
        root = merkle_root(packet_digests)

    per_uid: dict[int, list[AuditFileRef]] = {}
    cycle_scores: dict[int, list[CycleScore]] = {}
    baseline: list[AuditFileRef] = []
    competition_bundles: dict[str, list[AuditFileRef]] = {}
    availability_inputs: list[AvailabilityInput] = []
    earning_item_uids: set[int] = set()
    for item in items:
        # Inference rows enter the history-dependent EWMA fold, so both track and
        # ordering key must come from their pre-dispatch challenge commitment. An
        # economic competition row does not enter that fold: its entire symmetric
        # challenge/item matrix and track are committed by CompetitionInput instead.
        if item.source == "competition" and not (
            item.baseline and item.competition_subject is None
        ):
            if competition_input is None or not item.competition_subject:
                raise AuditFileMissingError(
                    f"competition item {item.item_id} lacks its committed competition "
                    "input/subject namespace"
                )
            if item.committed_track != competition_input.track:
                raise AuditFileMissingError(
                    f"competition item {item.item_id} declares track "
                    f"{item.committed_track!r}, expected committed competition track "
                    f"{competition_input.track!r}"
                )
            committed_track, committed_ordering_key = competition_input.track, None
        else:
            # SOURCE the committed track + fold-order key from the pre-dispatch
            # inference challenge.
            committed_track, committed_ordering_key = _committed_dispatch_for(
                item, commitment_source
            )
        # Proof opens this packet's digest against `root`; None only when there are no
        # leaves at all (an empty epoch — no manifest to prove). `()` for a lone leaf.
        proof: tuple[tuple[str, str], ...] | None = None
        if packet_digests:
            proof = tuple(merkle_proof(packet_digests, item.packet_digest))
        bundle_ref, packet_ref = _refs_for_item(
            item, committed_track=committed_track, inclusion_proof=proof
        )
        if store is not None:
            _require_stored(
                store, item.packet_digest, ArtifactKind.SCORE_PACKET, item.item_id
            )
            # #8: the earning item must resolve to a stored BUNDLE too, not just a
            # packet — a packet-only ref an auditor cannot recompute is a dead link.
            _require_stored(
                store, item.bundle_digest, ArtifactKind.AUDIT_BUNDLE, item.item_id
            )
        if item.baseline and item.competition_subject is None:
            # Legacy/inference calibration row; economic competitions namespace their
            # baseline explicitly and follow the branch below.
            baseline.extend((bundle_ref, packet_ref))
            continue
        if item.source == "competition":
            competition_bundles.setdefault(item.competition_subject, []).extend(
                (bundle_ref, packet_ref)
            )
            continue
        if item.competition_subject is not None:
            raise AuditFileMissingError(
                f"inference item {item.item_id} unexpectedly carries competition subject "
                f"{item.competition_subject!r}"
            )
        if item.baseline:
            baseline.extend((bundle_ref, packet_ref))
            continue
        per_uid.setdefault(item.uid, []).extend((bundle_ref, packet_ref))
        earning_item_uids.add(item.uid)
        # An earning cycle is any non-baseline item that attests a score OR marks an
        # evidenced exclusion. Its fold value + committed ordering key are bound to THIS
        # item's committed packet, so the auditor can re-derive the fold order/value.
        if item.score is None and not item.excluded_cycle:
            continue
        if committed_ordering_key is None:
            raise AuditFileMissingError(
                f"scored item {item.item_id} (uid {item.uid}) attests a cycle score but no "
                "committed dispatch ordering key — the committed fold ORDER is unverifiable "
                ""
            )
        value = EXCLUSION_SENTINEL if item.excluded_cycle else float(item.score)
        cycle_scores.setdefault(item.uid, []).append(
            CycleScore(
                packet_digest=item.packet_digest,
                ordering_key=int(committed_ordering_key),
                score=value,
            )
        )
    for evidence in availability_rows:
        normalized = _verify_availability_input(
            evidence,
            commitment_source=commitment_source,
            verify_fn=availability_verify_fn,
        )
        availability_inputs.append(normalized)
        earning_item_uids.add(normalized.uid)
        cycle_scores.setdefault(normalized.uid, []).append(
            CycleScore(
                packet_digest=normalized.observation_digest,
                ordering_key=normalized.ordering_key,
                score=0.0,
            )
        )
    # Emit an earning input for every uid that carries an earning ROW (attested cycles or
    # an explicit prior carry-in). cycle_scores are sorted ASCENDING by committed
    # ordering_key — the evidence-bound fold order, not the item-arrival order.
    earning_uids = (set(cycle_scores) | set(priors)) & earning_item_uids
    earning_inputs = {
        uid: EarningInput(
            prior_accumulate_score=float(priors.get(uid, 0.0)),
            cycle_scores=tuple(
                sorted(cycle_scores.get(uid, ()), key=lambda c: c.ordering_key)
            ),
        )
        for uid in sorted(earning_uids)
    }
    for uid, scores in sorted(cycle_scores.items()):
        if not scores:
            continue
        prior_cursor = fold_cursors.get(uid)
        current_keys = sorted(c.ordering_key for c in scores)
        if prior_cursor is not None and current_keys[0] <= prior_cursor:
            raise AuditFileMissingError(
                f"uid {uid} attempts to fold ordering_key(s) {current_keys} at/below its "
                f"cumulative prior cursor {prior_cursor} — refusing a cross-epoch "
                "packet replay before publication (schema v14)"
            )
        fold_cursors[uid] = current_keys[-1]
    return AuditManifest(
        per_uid={uid: tuple(refs) for uid, refs in per_uid.items()},
        baseline_bundles=tuple(baseline),
        score_packet_merkle_root=root,
        earning_inputs=earning_inputs,
        availability_inputs=tuple(availability_inputs),
        competition_input=competition_input,
        competition_bundles={
            subject_id: tuple(refs) for subject_id, refs in competition_bundles.items()
        },
        fold_cursors=fold_cursors,
    )


def _require_stored(
    store: _FinalizedSetStore, digest: str, kind: ArtifactKind, item_id: str
) -> None:
    """Probe that a content-addressed audit file exists, else `AuditFileMissingError`."""
    probe = ArtifactRef(
        digest=digest,
        kind=kind,
        byte_size=0,  # exists() keys off (kind, digest) only
        backend_key=backend_key(kind, digest),
    )
    if not store.exists(probe):
        raise AuditFileMissingError(
            f"{kind.value} {digest} for item {item_id} is not in the object store — "
            "refusing to back a weight with an unstored/unresolvable audit file"
        )


class FinalizedEpoch(BaseModel):
    """The pointer the finalizer returns: what the API publishes + the anchor binds.

    `snapshot_key` is the object-store key of the mirrored epoch-log bytes;
    `log_digest` == sha256(those bytes) == the on-chain anchored digest a validator
    verifies against. `log` is the assembled `EpochLog` (so the API can serve pointer
    fields without re-fetching). `already_finalized` is True on an idempotent re-run.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    epoch_id: int
    close_block: int
    snapshot_key: str
    log_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    weight_vector_digest: str = Field(pattern=SHA256_HEX_PATTERN)
    finalized: bool = True
    already_finalized: bool = False
    log: EpochLog


class EpochFinalizer:
    """The Scoring Authority's producer: state -> immutable, `_FINALIZED` epoch log.

    Constructed once with the tokenomics config + the scorer identity; `finalize`
    is called per epoch close. Pure/deterministic up to the object-store writes: the
    same inputs assemble byte-identical `EpochLog` bytes on any machine.
    """

    def __init__(
        self,
        config: TokenomicsConfig,
        *,
        scorer_version: str,
        schema_version: int | None = None,
    ) -> None:
        self._config = config
        self._scorer_version = scorer_version
        self._schema_version = schema_version

    def build_log(
        self,
        *,
        epoch_id: int,
        close_block: int,
        snapshots: Sequence[MinerSnapshot],
        miner_census: Sequence[MinerCensusEntry] | None = None,
        burn_uid: int,
        audit_manifest: AuditManifest,
        now: datetime,
        prior_log_digest: str | None = None,
        gap_epochs: tuple[int, ...] = (),
        prior_earning: Mapping[int, tuple[str, float]] | None = None,
        prior_fold_cursors: Mapping[int, int | None] | None = None,
        competition_result: CompetitionResult | None = None,
        prior_reward_window_state: RewardWindowState | None = None,
        competition_packet_scores: Mapping[str, float] | None = None,
    ) -> EpochLog:
        """Assemble (and validate) the `EpochLog` — the pure core of `finalize`.

        ``competition_result`` and the predecessor reward window are folded into the
        schema-v15 reward state and vector when competition emissions are enabled.
        ``miner_census`` is the complete registered payout-identity set at
        ``close_block``. Pure-model callers may omit it, in which case it is derived
        from ``snapshots``; production supplies the full set so offline/unknown-track
        registrations remain independently visible without entering the economic vector.
        """
        resolved_census = tuple(
            miner_census
            if miner_census is not None
            else (MinerCensusEntry.from_miner(miner) for miner in snapshots)
        )
        if competition_result is not None:
            if audit_manifest.competition_input is None:
                raise EpochLogInvalid(
                    "competition result has no committed competition input"
                )
            if competition_packet_scores is None:
                raise EpochLogInvalid(
                    "competition result cannot be finalized without its exact packet scores"
                )
            if audit_manifest.competition_input.applied_at != now:
                raise EpochLogInvalid(
                    "competition input applied_at must equal the epoch close-block time"
                )
            derived_result = derive_competition_result(
                audit_manifest.competition_input,
                competition_packet_scores,
            )
            if derived_result != competition_result:
                raise EpochLogInvalid(
                    "competition result does not equal the deterministic committed-packet "
                    "score derivation"
                )
        prior_reward_state = prior_reward_window_state or RewardWindowState()
        if (
            competition_result is not None
            and prior_reward_state.last_applied_cycle is not None
            and competition_result.cycle <= prior_reward_state.last_applied_cycle
        ):
            raise EpochLogInvalid(
                f"competition cycle {competition_result.cycle} is not newer than the last "
                f"applied reward cycle {prior_reward_state.last_applied_cycle}"
            )
        reward_window_state = resolve_reward_window(
            self._config,
            prior_reward_state,
            competition_result,
        )
        weight_shares = build_weight_vector(
            self._config,
            snapshots,
            burn_uid=burn_uid,
            reward_state=reward_window_state,
            now=now,
        )
        weight_u16 = quantize_u16(weight_shares)
        # Mark the empty-epoch burn path so the burn uid is manifest-exempt (rule 11).
        # It is the burn vector exactly when the ONLY positive weight is the burn uid AND
        # that uid is not itself a scored miner — so a real lone earner (however unlikely)
        # is never misclassified as the empty-epoch anchor.
        snapshot_uids = {m.uid for m in snapshots}
        log_burn_uid = (
            burn_uid
            if weight_shares.get(burn_uid, 0.0) > 0.0 and burn_uid not in snapshot_uids
            else None
        )
        # #1: the authority's OWN attestation must be internally consistent AND COMPLETE
        # — EVERY nonzero-weight uid must carry an earning input whose EWMA fold
        # reproduces the snapshot's stated accumulate_score. Refuse to publish a log
        # where a nonzero-weight uid lacks a complete earning derivation (which would
        # otherwise become a silent auditor SKIP that rolls up CLEAN) or whose fold does
        # not reproduce its own weights' inputs (the auditor re-checks this against the
        # audited packets; this catches a producer bug before it ever ships).
        self._require_complete_earning(
            snapshots,
            audit_manifest,
            weight_shares,
            log_burn_uid,
            prior_earning,
            reward_window_state,
            now,
        )
        audit_manifest = self._bind_complete_fold_cursors(
            audit_manifest,
            prior_fold_cursors,
            (entry.uid for entry in resolved_census),
        )
        self._require_complete_competition(audit_manifest, competition_result)
        # The obsolete retention-window completeness check remains removed. Competition
        # coverage is enforced immediately above by `_require_complete_competition`.
        return EpochLog(
            schema_version=self._schema_version
            or EpochLog.model_fields["schema_version"].default,
            epoch_id=epoch_id,
            close_block=close_block,
            scorer_version=self._scorer_version,
            created_at=now,
            prior_log_digest=prior_log_digest,
            gap_epochs=gap_epochs,
            burn_uid=log_burn_uid,
            competition_result=competition_result,
            reward_window_state=reward_window_state,
            miners=tuple(snapshots),
            miner_census=resolved_census,
            weight_shares=weight_shares,
            weight_u16=weight_u16,
            weight_vector_digest=weight_vector_digest(weight_u16),
            audit_manifest=audit_manifest,
        )

    @staticmethod
    def _bind_complete_fold_cursors(
        manifest: AuditManifest,
        prior_fold_cursors: Mapping[int, int | None] | None,
        current_census_uids: Iterable[int],
    ) -> AuditManifest:
        """Complete and validate the schema-v15 total fold-cursor map.

        Pure-model callers that omit predecessor context have no prior map to compare, so their
        claimed tombstones are preserved and completed with current census ``None`` entries.
        Once predecessor context is supplied (including ``{}`` at genesis), the manifest must
        already
        equal the one exact value the auditor derives: the whole predecessor map (including
        tombstones), every current census uid inserted as ``None`` if unseen, and current cycle
        maxima. Anything partial, invented, or regressed is refused.
        """
        expected = {
            int(uid): None if ordering_key is None else int(ordering_key)
            for uid, ordering_key in (prior_fold_cursors or {}).items()
        }
        census_uids = tuple(int(uid) for uid in current_census_uids)
        for uid in census_uids:
            expected.setdefault(uid, None)
        for uid, earning_input in manifest.earning_inputs.items():
            keys = [c.ordering_key for c in earning_input.cycle_scores]
            if not keys:
                continue
            prior = expected.get(uid)
            if prior is not None and min(keys) <= prior:
                raise EpochLogInvalid(
                    f"uid {uid} folds ordering_key(s) {sorted(keys)} at/below cumulative "
                    f"predecessor cursor {prior} — cross-epoch packet replay"
                )
            current_max = max(keys)
            expected[int(uid)] = current_max
        actual = {
            int(uid): None if key is None else int(key)
            for uid, key in manifest.fold_cursors.items()
        }
        if prior_fold_cursors is None:
            completed = dict(actual)
            for uid in census_uids:
                completed.setdefault(uid, None)
            if completed == actual:
                return manifest
            return manifest.model_copy(update={"fold_cursors": completed})
        if actual != expected:
            raise EpochLogInvalid(
                "audit manifest fold_cursors do not exactly carry and advance the chained "
                f"replay boundary: expected {expected}, got {actual}"
            )
        if actual == expected:
            return manifest
        return manifest.model_copy(update={"fold_cursors": expected})

    def _require_complete_earning(
        self,
        snapshots: Sequence[MinerSnapshot],
        manifest: AuditManifest,
        weight_shares: Mapping[int, float],
        burn_uid: int | None,
        prior_earning: Mapping[int, tuple[str, float]] | None = None,
        reward_window_state: RewardWindowState | None = None,
        now: datetime | None = None,
    ) -> None:
        """Every NONZERO-WEIGHT uid must carry a re-derivable earning state.

        The consistency check is over ALL nonzero-weight uids (not merely the ones that
        happen to have an entry): a nonzero-weight uid whose earning state an auditor could
        not re-derive is refused — it can no longer become a silent auditor SKIP that rolls
        up CLEAN (#1). A nonzero-weight uid is re-derivable in one of two ways:

        - it carries a current ``EarningInput`` whose EWMA fold reproduces the snapshot's
          stated accumulate_score (a miner that did new work this epoch); OR
        - an internal review: it is a pure CARRY-FORWARD — an IDLE prior earner still weighted
          by its carried accumulator, with NO current cycle. It has no ``EarningInput`` this
          epoch, and the auditor re-derives it via ``_carry_forward_verdict`` by chaining it to
          the prior epoch's value for the SAME (uid, hotkey). Refusing to publish such a log
          (the pre-round-20 behaviour) meant a normal idle/carry epoch could not be represented:
          with another miner's new evidence present the producer STALLED here; with none it
          dropped every miner to an empty burn that round-19 then disputed. So a nonzero-weight
          uid with NO ``EarningInput`` is ACCEPTED iff ``prior_earning`` proves it a genuine
          carry-forward — same (uid, hotkey) and identical accumulate_score in the prior epoch
          (the exact predicate the auditor re-checks). Absent ``prior_earning`` (pure-model
          callers / genesis) there is nothing to carry, so it is still refused.
        """
        decay = self._config.ewma_decay
        by_uid = {m.uid: m for m in snapshots}
        priors = prior_earning or {}
        competition_uids = {
            subject.uid
            for subject in (
                manifest.competition_input.subjects
                if manifest.competition_input is not None
                else ()
            )
            if (
                subject.role == "contender"
                and subject.uid is not None
                and (
                    subject.uid not in by_uid
                    or by_uid[subject.uid].excluded
                    or by_uid[subject.uid].accumulate_score <= 0.0
                )
            )
        }
        state = reward_window_state or RewardWindowState()
        reward_active = (
            now is not None
            and state.starts_at is not None
            and state.ends_at is not None
            and state.starts_at <= now < state.ends_at
        )
        rewarded_hotkeys = set(state.podium_hotkeys) if reward_active else set()
        competition_uids.update(
            miner.uid
            for miner in snapshots
            if miner.hotkey in rewarded_hotkeys
            and (miner.excluded or miner.accumulate_score <= 0.0)
        )
        nonzero = sorted(
            uid
            for uid, w in weight_shares.items()
            if w > 0.0 and uid != burn_uid and uid not in competition_uids
        )
        for uid in nonzero:
            miner = by_uid.get(uid)
            if miner is None:
                raise EpochLogInvalid(
                    f"uid {uid} has nonzero weight but is absent from the miner snapshots"
                )
            ei = manifest.earning_for(uid)
            if ei is None:
                # Accept a genuine pure carry-forward: same uid/hotkey and value.
                prior = priors.get(uid)
                if (
                    prior is not None
                    and prior[0] == miner.hotkey
                    and not is_excluded(miner.accumulate_score)
                    and abs(float(prior[1]) - float(miner.accumulate_score))
                    <= _FOLD_TOL
                ):
                    continue
                raise EpochLogInvalid(
                    f"uid {uid} has nonzero weight but NO earning input and is not a "
                    "verifiable carry-forward of the prior epoch (same uid/hotkey, identical "
                    "accumulate_score) — refusing to publish a log whose earning state an "
                    "auditor could not re-derive"
                )
            folded = ei.prior_accumulate_score
            for score in ei.folded_scores():
                folded = accumulate(folded, score, decay)
            if is_excluded(miner.accumulate_score) or is_excluded(folded):
                if folded != miner.accumulate_score:
                    raise EpochLogInvalid(
                        f"uid {uid} earning input folds to {folded} but the snapshot's "
                        f"accumulate_score is {miner.accumulate_score} (exclusion mismatch)"
                    )
            elif abs(folded - miner.accumulate_score) > _FOLD_TOL:
                raise EpochLogInvalid(
                    f"uid {uid} earning input (prior={ei.prior_accumulate_score}, "
                    f"cycles={ei.folded_scores()}) EWMA-folds to {folded}, not the "
                    f"snapshot's stated accumulate_score {miner.accumulate_score} — a "
                    "substituted or inconsistent earning state is refused before publication"
                )

    @staticmethod
    def _require_complete_competition(
        manifest: AuditManifest,
        result: CompetitionResult | None,
    ) -> None:
        """Require exact committed packet/bundle coverage for an economic result."""
        comp_input = manifest.competition_input
        if result is None:
            if comp_input is not None:
                raise EpochLogInvalid(
                    "competition evidence is present without an economic result"
                )
            return
        if comp_input is None:
            raise EpochLogInvalid(
                "an economic competition result has no committed audit input"
            )
        if (
            comp_input.competition_id != result.competition_id
            or comp_input.track != result.track
            or comp_input.cycle != result.cycle
            or comp_input.applied_at != result.applied_at
            or comp_input.baseline_version != result.baseline_version
            or comp_input.baseline_artifact_digest
            != result.baseline_artifact_digest
        ):
            raise EpochLogInvalid(
                "competition result identity/application/baseline provenance is not bound "
                "by its audit input"
            )
        if comp_input.completed_at > result.applied_at:
            raise EpochLogInvalid(
                "competition operational completion time is after economic application"
            )

    # `_require_complete_window` remains removed with the retired retention multiplier.

    def finalize(
        self,
        *,
        epoch_id: int,
        close_block: int,
        snapshots: Sequence[MinerSnapshot],
        miner_census: Sequence[MinerCensusEntry] | None = None,
        burn_uid: int,
        audit_manifest: AuditManifest,
        store: _FinalizedSetStore,
        public_store: _PublicReleaseStore | None = None,
        now: datetime,
        prior_log_digest: str | None = None,
        gap_epochs: tuple[int, ...] = (),
        prior_earning: Mapping[int, tuple[str, float]] | None = None,
        prior_fold_cursors: Mapping[int, int | None] | None = None,
        competition_result: CompetitionResult | None = None,
        prior_reward_window_state: RewardWindowState | None = None,
        competition_packet_scores: Mapping[str, float] | None = None,
    ) -> FinalizedEpoch:
        """Produce + publish the epoch log; return its pointer (key + digests).

        Writes the log object first, then the `_FINALIZED` marker LAST, so no
        validator can mirror a half-written epoch. Idempotent: if the epoch is already
        finalized this is a NO-OP that reads the stored bytes and returns the SAME
        key/digest (a finalized set is immutable — never rewritten).
        """
        prefix = epoch_prefix(epoch_id)
        key = set_member_key(prefix, EPOCH_LOG_MEMBER)

        if store.is_finalized(prefix):
            data = store.get_set_member(prefix, EPOCH_LOG_MEMBER)
            log = EpochLog.from_json(data)
            self._require_public_crown_winner(log, public_store)
            # #16: on recovery from an already-`_FINALIZED` set the pointer MUST come
            # from the STORED log's own fields (epoch_id / close_block / weight digest),
            # NOT the caller's arguments. A crash after `_FINALIZED` but before indexing,
            # followed by changed epoch parameters (a new tempo/close_block), would
            # otherwise index+publish a pointer whose metadata contradicts its anchored
            # bytes — and the provider's identity check (shared_snapshot) would then
            # reject the honest log. Bind the pointer to the bytes that are actually
            # anchored: read the fields off the log.
            return FinalizedEpoch(
                epoch_id=log.epoch_id,
                close_block=log.close_block,
                snapshot_key=set_member_key(
                    epoch_prefix(log.epoch_id), EPOCH_LOG_MEMBER
                ),
                log_digest=sha256_hex(data),
                weight_vector_digest=log.weight_vector_digest,
                already_finalized=True,
                log=log,
            )

        log = self.build_log(
            epoch_id=epoch_id,
            close_block=close_block,
            snapshots=snapshots,
            miner_census=miner_census,
            burn_uid=burn_uid,
            audit_manifest=audit_manifest,
            now=now,
            prior_log_digest=prior_log_digest,
            gap_epochs=gap_epochs,
            prior_earning=prior_earning,
            prior_fold_cursors=prior_fold_cursors,
            competition_result=competition_result,
            prior_reward_window_state=prior_reward_window_state,
            competition_packet_scores=competition_packet_scores,
        )
        # A CROWN starts real earnings in this very log.  Its exact winning source
        # archive must therefore already be readable through the same keyless view
        # an independent auditor uses.  The evidence builder performs the explicit
        # winner-only release; this final boundary independently proves the result
        # before either the epoch member or its `_FINALIZED` marker can be written.
        self._require_public_crown_winner(log, public_store)
        data = log.to_json()
        digest = sha256_hex(data)
        # member first ...
        store.put_set_member(prefix, EPOCH_LOG_MEMBER, data, ArtifactKind.EPOCH_LOG)
        # ... then the marker LAST (atomic w.r.t. readers via is_finalized probe).
        store.finalize_set(prefix)
        return FinalizedEpoch(
            epoch_id=epoch_id,
            close_block=close_block,
            snapshot_key=key,
            log_digest=digest,
            weight_vector_digest=log.weight_vector_digest,
            log=log,
        )

    @staticmethod
    def _require_public_crown_winner(
        log: EpochLog, public_store: _PublicReleaseStore | None
    ) -> None:
        """Refuse a new CROWN unless its winner archive passes anonymous readback.

        Carry-forward epochs retain the already-anchored reward window and do not
        repeat the originating competition input.  The gate therefore applies to
        the source epoch that introduces a packet-derived CROWN result; every later
        CROWN carry chains back to that gated log.
        """
        result = log.competition_result
        if result is None or log.reward_window_state.kind is not EmissionState.CROWN:
            return
        competition_input = log.audit_manifest.competition_input
        if competition_input is None or not result.contenders:
            raise EpochLogInvalid(
                "CROWN publication has no committed winner submission identity"
            )
        winner = result.contenders[0]
        matching = [
            subject
            for subject in competition_input.subjects
            if subject.role == "contender"
            and subject.uid == winner.uid
            and subject.hotkey == winner.hotkey
        ]
        if len(matching) != 1:
            raise EpochLogInvalid(
                "CROWN winner does not resolve to exactly one committed contender archive"
            )
        subject = matching[0]
        if (
            subject.submission_archive_digest is None
            or subject.submission_archive_bytes is None
        ):
            raise EpochLogInvalid(
                "CROWN winner lacks a content-addressed submission archive"
            )
        ref = ArtifactRef(
            digest=subject.submission_archive_digest,
            kind=ArtifactKind.SUBMISSION_ARCHIVE,
            byte_size=subject.submission_archive_bytes,
            backend_key=backend_key(
                ArtifactKind.SUBMISSION_ARCHIVE,
                subject.submission_archive_digest,
            ),
        )
        if public_store is None:
            raise AuditFileMissingError(
                "CROWN publication requires an independent keyless audit-store view; "
                "none was supplied"
            )
        if not public_store.public_read_only:
            raise AuditFileMissingError(
                "CROWN publication requires the anonymous read-only audit-store view; "
                "a credentialed/private store is not public-access proof"
            )
        if not public_store.is_released(ref):
            raise AuditFileMissingError(
                f"CROWN winner submission archive {ref.digest} is not anonymously "
                "readable and content-valid"
            )


__all__ = [
    "EpochFinalizer",
    "FinalizedEpoch",
    "ScoredItem",
    "ChallengeCommitmentSource",
    "build_audit_manifest",
    "AuditFileMissingError",
    "epoch_prefix",
    "EPOCH_LOG_MEMBER",
    "EpochLogInvalid",
]
