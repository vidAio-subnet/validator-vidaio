"""Champion promotion pipeline — the design spec §20 gate.

"Promotes the holdout-winner, not a public-board winner." The gate originally
took the caller's word for the whole story (competition id, winner, score,
artifact, bundle were all parameters). A first pass added self-consistency
checks — but self-consistency is not provenance: a caller who substitutes a
winner, a packet, a bundle referencing that packet and an artifact, all agreeing
with each other, still passed every check and installed an arbitrary backend as
the quality floor.

So the pipeline no longer ACCEPTS the facts, it DERIVES them. Given a
competition id it reads the competition database itself
(:mod:`vidaio.registry.competition_source`) and takes from it: the track, the
winner (best-ranked non-calibration contender), the winner's holdout score, the
winner's per-item `score_packet_digest` / `audit_bundle_digest` linkage, the
holdout's challenge ids, the winner's build identity, and the audit-store address
of the submission tarball the competition archived FOR THAT WINNER. Anything the
caller passes is an ASSERTION (:class:`PromotionAssertions`) that must EQUAL what
the database says — a mismatch is a typed error, never a promotion.

THE SCORE THAT IS PROMOTED IS AN AGGREGATE, SO ALL OF ITS EVIDENCE IS VERIFIED
(round-3 finding #6). `contenders.final_score` is derived from EVERY holdout item
the winner was scored on. An earlier pass verified only the ONE bundle the caller
offered, which meant a two-item winner could promote with item 1's packet or
bundle missing, corrupt, or minted by a different scorer — while the promoted
number silently included item 1's score. Verification is therefore per-item over
the winner's whole item set; the caller's offered bundle is kept only as an
ASSERTION about which item it believes it is handing over.

Gates, in order (each owns its error):

  1. the competition exists                   -> CompetitionNotFoundError
  2. it is COMPLETED                          -> CompetitionNotCompletedError
  3. it has a rankable, scored winner         -> NoEligibleWinnerError
  4. the promotion track is the contested one -> RegistryError
  5. every caller assertion equals the DB     -> EvidenceAssertionError
                                                 (artifact ones: ArtifactLinkageError)
  6. the winner has items, and EVERY item carries both a score_packet_digest and
     an audit_bundle_digest                    -> AuditLinkageGapError
  7. the offered bundle IS one the competition recorded for this winner
     (an assertion about WHICH item, not the evidence itself)
                                               -> AuditBundleMismatchError /
                                                  MissingAuditLinkageError
  8. for EVERY one of the winner's items, in item order:
       a. the recorded bundle exists, verifies on read, parses, and its stored
          bytes are the canonical AuditBundle serialization whose bundle_digest
          is the recorded content address
                                               -> MissingAuditLinkageError /
                                                  AuditBundleMismatchError
       b. its identity fields equal the DB's row (winner, challenge, item), its
          challenge belongs to the holdout, and the bundle's input/reference
          artifact digests equal the media digests in `evaluation_items`; the
          threshold commitment matches; and an upscaling bundle carries the
          exact persisted item index/factor/commitment preimage; its policy
          manifest and execution image equal the competition's persisted ones
                                               -> AuditBundleMismatchError
       c. every artifact the bundle references exists under its exact kind and
          verifies by digest and byte size; an upscaling pristine reference is
          publicly released at completion       -> MissingAuditLinkageError
       d. its score packet IS the digest the competition recorded for that item,
          exists, verifies on read and parses   -> ScorePacketMismatchError /
                                                  MissingAuditLinkageError
       e. the packet is attributed to the same winner/challenge/item/track
                                               -> ScorePacketMismatchError
       f. the packet's `content_digest` IS the digest of the miner output the
          bundle archives — a packet scoring output A over a
          bundle archiving output B would otherwise pass, and a packet with a
          NULL content_digest is refused too: every packet the write side mints
          carries one (the scoring worker stamps the digest of the private copy
          it measured; the orchestrator's zero packets stamp the canonical empty
          digest), so null-with-an-archived-output is unauditable
                                               -> ScorePacketMismatchError
       g. the packet's scorer identity equals the bundle's
                                               -> ScorerIdentityMismatchError
       h. the packet's score equals the score the competition persisted for that
          item                                  -> HoldoutScoreMismatchError
     Every one of these errors NAMES the item it failed on.
  9. the winner was actually BUILT, the competition certified a submission
     backup, the offered artifact IS the tarball the competition archived FOR
     THIS WINNER (derived from its per-contender archival event, not asserted),
     and it verifies in the store
                                               -> ArtifactNotArchivedError /
                                                  ArtifactLinkageError
 10. only then the registry's own guard (strictly beat the reigning champion)
                                               -> ChampionNotBeatenError

The self-consistency checks the earlier pass added are kept as an ADDITIONAL
layer (gate 8b's holdout-membership check, the packet↔bundle field comparisons):
belt and braces, because the two sources are independently corruptible.

Every store read is verify-on-read: bytes that do not hash to their content
address are treated as absent. The items the caller did NOT hand over are known
only by digest, so they are fetched by content address and re-hashed here
(:meth:`PromotionPipeline._read_verified`) — the digest IS the address, and a
recorded byte size adds nothing that the hash does not already prove.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from vidaio.audit.bundle import AuditBundle, LifecycleStage
from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import (
    ArtifactKind,
    ArtifactRef,
    AuditStore,
    IntegrityError,
)
from vidaio.registry import registry
from vidaio.registry.competition_source import (
    BUILT,
    CompetitionFacts,
    CompetitionSource,
    ItemLinkage,
    WinnerFacts,
)
from vidaio.registry.registry import (
    ChampionCandidate,
    ChampionRecord,
    LegacyRegistryWriteDisabledError,
    RegistryError,
)
from vidaio.scoring.result import ItemScore

#: A persisted score must be the packet's score, not "close to" it. A float-repr
#: round trip is the only slack allowed.
_SCORE_TOLERANCE = 1e-12
_MAX_METADATA_BYTES = 4 * 1024 * 1024


class MissingAuditLinkageError(RegistryError):
    """Candidate lacks a linked, verified audit bundle — promotion is impossible."""


class ArtifactNotArchivedError(RegistryError):
    """The champion executable is not (verifiably) present in the audit store."""


class ArtifactLinkageError(RegistryError):
    """The artifact offered is not the executable that competition archived."""


class AuditBundleMismatchError(RegistryError):
    """The audit bundle is not one the competition recorded for this winner."""


class ScorePacketMismatchError(RegistryError):
    """The bundle's score packet is not the packet the competition recorded."""


class HoldoutScoreMismatchError(RegistryError):
    """The packet's score is not the score the competition persisted."""


class CompetitionNotFoundError(RegistryError):
    """No such competition in the authoritative database — nothing to promote."""


class CompetitionNotCompletedError(RegistryError):
    """The competition has not COMPLETED, so it has no winner to promote."""


class NoEligibleWinnerError(RegistryError):
    """The competition COMPLETED without a rankable, scored, eligible winner."""


class AuditLinkageGapError(RegistryError):
    """The winner has scored items with no linked audit bundle — unauditable."""


class EvidenceAssertionError(RegistryError):
    """A caller-asserted fact contradicts the competition database."""


class ScorerIdentityMismatchError(RegistryError):
    """The score packet and its audit bundle name different scorers."""


@dataclass(frozen=True)
class PromotionAssertions:
    """What the CALLER claims about the win. Never trusted — only checked.

    Every field is optional: supply what you believe and the pipeline proves it
    against the competition database. Supplying nothing is perfectly safe (the
    facts are derived either way); supplying something wrong is a typed refusal,
    which is what makes this useful as a caller-side sanity harness.
    """

    competition_id: str | None = None
    track: str | None = None
    winner_hotkey: str | None = None
    #: The winner's overall holdout score (`contenders.final_score`).
    holdout_score: float | None = None
    #: The per-item score the promoted bundle's packet must carry.
    item_score: float | None = None
    artifact_digest: str | None = None
    artifact_kind: ArtifactKind | None = None
    audit_bundle_digest: str | None = None
    score_packet_digest: str | None = None
    challenge_id: str | None = None
    item_id: str | None = None
    #: The winner's build identity (`contenders.image_digest`).
    image_digest: str | None = None


class PromotionPipeline:
    """Derive a competition's winner from the DB, verify it, then promote."""

    def __init__(self, store: AuditStore, competitions: CompetitionSource) -> None:
        self._store = store
        #: The AUTHORITY. Required: a pipeline with no competition database
        #: cannot verify anything, and "verify nothing" must not be reachable
        #: by omitting an argument.
        self._competitions = competitions

    def evaluate_and_promote(
        self,
        conn: sqlite3.Connection,
        *,
        competition_id: str,
        artifact_ref: ArtifactRef,
        audit_bundle_ref: ArtifactRef | None,
        now: datetime,
        track: str | None = None,
        asserted: PromotionAssertions | None = None,
    ) -> ChampionRecord:
        """Refuse the retired schema-v13 promotion path.

        This compatibility method remains importable so an old process fails with
        a typed, explicit error instead of silently falling back to caller/DB
        operational ranking.  It never reads evidence and never writes state.
        """
        raise LegacyRegistryWriteDisabledError(
            "schema-v13 competition promotion is disabled; use "
            "BaselinePromotionPipeline with VerifiedCrownEpochSource"
        )

    def _retired_evaluate_and_promote(
        self,
        conn: sqlite3.Connection,
        *,
        competition_id: str,
        artifact_ref: ArtifactRef,
        audit_bundle_ref: ArtifactRef | None,
        now: datetime,
        track: str | None = None,
        asserted: PromotionAssertions | None = None,
    ) -> ChampionRecord:
        """Unreachable historical implementation kept for evidence-read diagnostics."""
        raise LegacyRegistryWriteDisabledError("retired promotion is sealed")
        facts = self._facts(competition_id)
        winner = self._winner(facts)
        if track is not None and track != facts.track:
            raise RegistryError(
                f"promotion track {track!r} does not match the competition's "
                f"track {facts.track!r}"
            )
        self._check_assertions(facts, winner, artifact_ref, audit_bundle_ref, asserted)
        # The caller's bundle only SELECTS which item it claims to be handing
        # over. What is verified is every item that fed the aggregate score.
        linkage = self._offered_linkage(facts, winner, audit_bundle_ref)
        self._verify_every_item(facts, winner)
        self._check_item_assertions(linkage, asserted)
        self._verify_artifact(facts, winner, artifact_ref)
        candidate = ChampionCandidate(
            track=facts.track,
            artifact_digest=artifact_ref.digest,
            artifact_kind=artifact_ref.kind,
            artifact_bytes=artifact_ref.byte_size,
            source_competition_id=facts.competition_id,
            contender_hotkey=winner.hotkey,
            holdout_score=winner.final_score,
            audit_bundle_digest=linkage.audit_bundle_digest or "",
        )
        return registry.promote(conn, facts.track, candidate, now)

    # -- the authoritative read ----------------------------------------------

    def _facts(self, competition_id: str) -> CompetitionFacts:
        facts = self._competitions.facts(competition_id)
        if facts is None:
            raise CompetitionNotFoundError(
                f"competition {competition_id!r} does not exist in the competition "
                "database — a promotion cannot be derived from a competition that "
                "was never run"
            )
        if not facts.completed:
            raise CompetitionNotCompletedError(
                f"competition {competition_id!r} is {facts.status} — only a "
                "COMPLETED competition has a holdout winner to promote"
            )
        return facts

    def _winner(self, facts: CompetitionFacts) -> WinnerFacts:
        if facts.winner is None:
            raise NoEligibleWinnerError(
                f"competition {facts.competition_id!r} COMPLETED with no ranked, "
                "scored, eligible contender — there is nobody to promote"
            )
        return facts.winner

    # -- assertions ----------------------------------------------------------

    def _check_assertions(
        self,
        facts: CompetitionFacts,
        winner: WinnerFacts,
        artifact_ref: ArtifactRef,
        audit_bundle_ref: ArtifactRef | None,
        asserted: PromotionAssertions | None,
    ) -> None:
        """Every claim the caller made must equal the database's answer."""
        if asserted is None:
            return
        # Artifact claims keep their own error type: they are about the blob the
        # caller handed over, not about the competition's record.
        _require(
            ArtifactLinkageError,
            "artifact_digest",
            asserted.artifact_digest,
            artifact_ref.digest,
            "the artifact handed to the promotion",
        )
        _require(
            ArtifactLinkageError,
            "artifact_kind",
            asserted.artifact_kind,
            artifact_ref.kind,
            "the artifact handed to the promotion",
        )
        _require(
            AuditBundleMismatchError,
            "audit_bundle_digest",
            asserted.audit_bundle_digest,
            audit_bundle_ref.digest if audit_bundle_ref is not None else None,
            "the bundle handed to the promotion",
        )
        for field, claimed, actual in (
            ("competition_id", asserted.competition_id, facts.competition_id),
            ("track", asserted.track, facts.track),
            ("winner_hotkey", asserted.winner_hotkey, winner.hotkey),
            ("holdout_score", asserted.holdout_score, winner.final_score),
            ("image_digest", asserted.image_digest, winner.image_digest),
        ):
            _require(
                EvidenceAssertionError,
                field,
                claimed,
                actual,
                f"competition {facts.competition_id}",
            )

    def _check_item_assertions(
        self, linkage: ItemLinkage, asserted: PromotionAssertions | None
    ) -> None:
        if asserted is None:
            return
        for field, claimed, actual in (
            ("challenge_id", asserted.challenge_id, linkage.challenge_id),
            ("item_id", asserted.item_id, linkage.scoring_item_id),
            (
                "score_packet_digest",
                asserted.score_packet_digest,
                linkage.score_packet_digest,
            ),
            ("item_score", asserted.item_score, linkage.item_score),
        ):
            _require(
                EvidenceAssertionError,
                field,
                claimed,
                actual,
                f"the winner's item {linkage.item_index}",
            )

    # -- gates ---------------------------------------------------------------

    def _offered_linkage(
        self,
        facts: CompetitionFacts,
        winner: WinnerFacts,
        ref: ArtifactRef | None,
    ) -> ItemLinkage:
        """Which item the caller BELIEVES it handed over. An assertion, not evidence.

        Promotion no longer depends on this selection — `_verify_every_item`
        verifies the winner's whole item set regardless. What this still buys is
        a refusal when the caller offers a bundle this competition never linked
        to this winner at all, and a subject for the per-item assertions.
        """
        track, hotkey = facts.track, winner.hotkey
        if ref is None:
            raise MissingAuditLinkageError(
                f"{track} winner {hotkey} from {facts.competition_id} has no "
                "linked audit bundle — an unauditable candidate cannot be promoted"
            )
        if not facts.items:
            raise AuditLinkageGapError(
                f"{track} winner {hotkey} has no persisted item scores in "
                f"competition {facts.competition_id} — there is no audit trail "
                "to promote from"
            )
        gaps = facts.audit_linkage_gaps()
        if gaps:
            raise AuditLinkageGapError(
                f"{track} winner {hotkey} has scored items with incomplete audit "
                f"linkage in competition {facts.competition_id} "
                f"({'; '.join(gaps)}) — a partially audited win never promotes, "
                "because the promoted score is the aggregate of every one of them"
            )
        linkage = facts.linkage_for_bundle(ref.digest)
        if linkage is None:
            raise AuditBundleMismatchError(
                f"audit bundle {ref.digest} is not a bundle competition "
                f"{facts.competition_id} recorded for {hotkey}; it recorded "
                f"{sorted(str(i.audit_bundle_digest) for i in facts.items)}"
            )
        return linkage

    # -- the aggregate's whole evidence set ----------------------------------

    def _verify_every_item(self, facts: CompetitionFacts, winner: WinnerFacts) -> None:
        """Verify the bundle AND packet of every item that fed `final_score`.

        The winner's item set is what the aggregate was computed over, so it is
        exactly the set that has to hold up. Bounded by construction: one
        holdout winner's items, two small content-addressed reads each.
        """
        for linkage in facts.items:
            bundle = self._verify_item_bundle(facts, winner, linkage)
            self._verify_item_packet(facts, winner, bundle, linkage)

    def _verify_item_bundle(
        self, facts: CompetitionFacts, winner: WinnerFacts, linkage: ItemLinkage
    ) -> AuditBundle:
        track, hotkey, where = facts.track, winner.hotkey, _item(linkage)
        digest = linkage.audit_bundle_digest or ""
        try:
            raw = self._read_verified(digest, ArtifactKind.AUDIT_BUNDLE)
        except (FileNotFoundError, OSError, IntegrityError) as exc:
            raise MissingAuditLinkageError(
                f"audit bundle {digest} recorded for {where} of {track} winner "
                f"{hotkey} failed store verification: {exc}"
            ) from exc
        try:
            bundle = AuditBundle.model_validate_json(raw)
        except ValueError as exc:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} recorded for {where} of {track} winner "
                f"{hotkey} does not parse as an AuditBundle: {exc}"
            ) from exc
        canonical = canonical_json_bytes(bundle.model_dump(mode="json"))
        if raw != canonical or bundle.bundle_digest() != digest:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} recorded for {where} of {track} winner "
                f"{hotkey} is not its canonical AuditBundle serialization — the "
                "recorded content address must equal bundle.bundle_digest()"
            )
        if bundle.miner_hotkey != hotkey:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) audits miner "
                f"{bundle.miner_hotkey!r}, not the {track} winner {hotkey!r} the "
                "competition recorded"
            )
        if bundle.challenge_id != linkage.challenge_id:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} audits challenge "
                f"{bundle.challenge_id!r}, but the competition linked it to "
                f"{where}, whose challenge is {linkage.challenge_id!r}"
            )
        if bundle.item_id != linkage.scoring_item_id:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) audits item {bundle.item_id!r}, "
                f"but the competition linked it to {linkage.scoring_item_id!r}"
            )
        if bundle.manifest.digest != facts.manifest_digest:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) names policy manifest "
                f"{bundle.manifest.digest}, but competition "
                f"{facts.competition_id} committed manifest_digest "
                f"{facts.manifest_digest} before enrollment"
            )
        if winner.image_digest is not None and (
            bundle.execution_image_digest is None
            or bundle.execution_image_digest != winner.image_digest
        ):
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) names execution image "
                f"{bundle.execution_image_digest!r}, but the winner's persisted "
                f"build identity is {winner.image_digest!r} — promotion requires "
                "the exact image that produced every scored output"
            )
        if bundle.commitment_hash != linkage.threshold_commitment:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) commits score policy "
                f"{bundle.commitment_hash}, but evaluation_items records "
                f"threshold_commitment {linkage.threshold_commitment} — the "
                "promoted score must use the precommitted evaluation gate"
            )
        self._verify_item_media(facts, bundle, linkage, digest=digest, where=where)
        # Additional self-consistency layer (kept from the previous pass): the
        # bundle's challenge must also be one the holdout was built from.
        if bundle.challenge_id not in facts.holdout_challenge_ids:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) audits challenge "
                f"{bundle.challenge_id!r}, which is not part of competition "
                f"{facts.competition_id}'s hidden holdout"
            )
        self._verify_bundle_artifacts(facts, bundle, linkage)
        return bundle

    def _verify_bundle_artifacts(
        self,
        facts: CompetitionFacts,
        bundle: AuditBundle,
        linkage: ItemLinkage,
    ) -> None:
        """Stream-verify every blob needed to recompute this competition item.

        Score packets are verified and parsed separately because they have
        additional DB/identity checks. ``dag_reveal`` cannot occur in either
        currently accepted competition shape (PRE_REVEAL compression or
        COMPETITION_SEALED upscaling), but it remains in this exhaustive list so
        a future accepted stage cannot silently introduce an unchecked ref.
        """
        where = _item(linkage)
        refs = (
            bundle.challenge_input,
            bundle.miner_output,
            bundle.manifest,
            bundle.reference_original,
            bundle.dag_reveal,
        )
        for ref in refs:
            if ref is None:
                continue
            try:
                self._verify_artifact_ref(ref)
            except (FileNotFoundError, OSError, IntegrityError) as exc:
                raise MissingAuditLinkageError(
                    f"{ref.kind.value} artifact {ref.digest} referenced by the "
                    f"audit bundle for {where} of competition "
                    f"{facts.competition_id} failed store verification: {exc}"
                ) from exc
        reference = bundle.reference_original
        if facts.track == "upscaling" and reference is not None:
            try:
                released = self._store.is_released(reference)
            except (OSError, IntegrityError) as exc:
                raise MissingAuditLinkageError(
                    f"upscaling reference_original {reference.digest} for {where} "
                    f"could not be checked for public release: {exc}"
                ) from exc
            if not released:
                raise MissingAuditLinkageError(
                    f"upscaling reference_original {reference.digest} for {where} "
                    "is not publicly released — a completed champion must be "
                    "recomputable by a keyless auditor"
                )

    def _verify_artifact_ref(self, ref: ArtifactRef) -> None:
        """Verify a typed artifact ref without buffering a large media object."""
        digest = hashlib.sha256()
        size = 0
        with contextlib.closing(self._store.open_stream(ref)) as stream:
            while chunk := stream.read(1 << 20):
                size += len(chunk)
                if size > ref.byte_size:
                    raise IntegrityError(
                        f"artifact {ref.kind.value}/{ref.digest} exceeds its "
                        f"committed {ref.byte_size}-byte size"
                    )
                digest.update(chunk)
        if size != ref.byte_size or digest.hexdigest() != ref.digest:
            raise IntegrityError(
                f"artifact {ref.kind.value}/{ref.digest} failed verify-on-read: "
                "bytes do not match the committed size/content address"
            )

    @staticmethod
    def _verify_item_media(
        facts: CompetitionFacts,
        bundle: AuditBundle,
        linkage: ItemLinkage,
        *,
        digest: str,
        where: str,
    ) -> None:
        """Bind the competition's media row to the bundle's archived bytes.

        Challenge/item strings are routing identity, not content identity. A
        self-consistent bundle can preserve those strings while scoring entirely
        different media, so promotion compares the independent content addresses.
        """
        if bundle.challenge_input.digest != linkage.input_sha256:
            raise AuditBundleMismatchError(
                f"audit bundle {digest} ({where}) archives challenge input "
                f"{bundle.challenge_input.digest}, but evaluation_items records "
                f"input_sha256 {linkage.input_sha256} — the promoted score must "
                "audit the exact miner-visible bytes selected for the competition"
            )
        reference_digest = linkage.reference_sha256
        if not reference_digest:
            raise AuditBundleMismatchError(
                f"competition {facts.competition_id} records no reference_sha256 "
                f"for {where}; promotion cannot prove which reference media the "
                "item score used"
            )
        if facts.track == "compression":
            # Migration 0002 normalizes compression rows this way: the exposed
            # source is also the quality reference.
            if reference_digest != linkage.input_sha256:
                raise AuditBundleMismatchError(
                    f"compression {where} records reference_sha256 "
                    f"{reference_digest}, but its input_sha256 is "
                    f"{linkage.input_sha256} — compression must score against "
                    "the exact challenge input"
                )
            if (
                bundle.reference_original is not None
                and bundle.reference_original.digest != reference_digest
            ):
                raise AuditBundleMismatchError(
                    f"compression audit bundle {digest} ({where}) archives "
                    f"reference_original {bundle.reference_original.digest}, but "
                    f"evaluation_items records normalized reference_sha256 "
                    f"{reference_digest} — recomputation must never substitute "
                    "different reference bytes"
                )
            if bundle.stage is not LifecycleStage.PRE_REVEAL:
                raise AuditBundleMismatchError(
                    f"compression audit bundle {digest} ({where}) has stage "
                    f"{bundle.stage.value!r}; competition compression requires "
                    "pre_reveal so challenge_input is unambiguously both miner "
                    "input and scoring reference"
                )
            if (
                bundle.reference_original is not None
                or bundle.competition_item is not None
                or linkage.upscale_factor is not None
                or linkage.item_commitment is not None
            ):
                raise AuditBundleMismatchError(
                    f"compression audit bundle {digest} ({where}) carries "
                    "upscaling/reference-only fields — compression must bind its "
                    "single normalized challenge-input reference"
                )
            return
        if facts.track == "upscaling":
            if bundle.stage is not LifecycleStage.COMPETITION_SEALED:
                raise AuditBundleMismatchError(
                    f"upscaling audit bundle {digest} ({where}) has stage "
                    f"{bundle.stage.value!r}, expected competition_sealed"
                )
            reference = bundle.reference_original
            if reference is None:
                raise AuditBundleMismatchError(
                    f"upscaling audit bundle {digest} ({where}) has no "
                    "reference_original artifact, so its score cannot be bound "
                    "to the competition's pristine reference"
                )
            if reference.digest != reference_digest:
                raise AuditBundleMismatchError(
                    f"upscaling audit bundle {digest} ({where}) archives pristine "
                    f"reference {reference.digest}, but evaluation_items records "
                    f"reference_sha256 {reference_digest} — promotion cannot use "
                    "a score measured against different reference bytes"
                )
            binding = bundle.competition_item
            expected_binding = (
                linkage.item_index,
                linkage.input_sha256,
                reference_digest,
                linkage.upscale_factor,
                linkage.item_commitment,
            )
            actual_binding = (
                None
                if binding is None
                else (
                    binding.item_index,
                    binding.input_sha256,
                    binding.reference_sha256,
                    binding.upscale_factor,
                    binding.item_commitment,
                )
            )
            if (
                linkage.upscale_factor is None
                or linkage.item_commitment is None
                or actual_binding != expected_binding
            ):
                raise AuditBundleMismatchError(
                    f"upscaling audit bundle {digest} ({where}) carries item "
                    f"preimage {actual_binding!r}, but evaluation_items records "
                    f"{expected_binding!r} — factor and commitment are part of "
                    "the score's pre-enrollment identity"
                )
            return
        raise AuditBundleMismatchError(
            f"competition {facts.competition_id} has unsupported promotion track "
            f"{facts.track!r}; its evaluation-media binding is undefined"
        )

    def _read_verified(self, digest: str, kind: ArtifactKind) -> bytes:
        """Content-addressed, verify-on-read fetch of `(kind, digest)`.

        `AuditStore.get()` needs a full ArtifactRef, and for every item the
        caller did NOT hand over the competition database records only the
        DIGEST. The digest is the whole content address — a byte size proves
        nothing the hash does not — so this streams the stored bytes and
        recomputes sha256 itself, which is the identical guarantee: bytes that
        do not hash to their address are treated as absent.
        """
        raw = self._store.get_digest_limited(
            kind, digest, max_bytes=_MAX_METADATA_BYTES
        )
        if sha256_hex(raw) != digest:
            raise IntegrityError(
                f"artifact {kind.value}/{digest} failed verify-on-read: stored "
                "bytes do not match the content address"
            )
        return raw

    def _verify_item_packet(
        self,
        facts: CompetitionFacts,
        winner: WinnerFacts,
        bundle: AuditBundle,
        linkage: ItemLinkage,
    ) -> None:
        where = _item(linkage)
        ref = bundle.score_packet
        if ref.digest != linkage.score_packet_digest:
            raise ScorePacketMismatchError(
                f"the audit bundle for {where} points at score packet "
                f"{ref.digest}, but competition {facts.competition_id} recorded "
                f"{linkage.score_packet_digest} as the packet that item's score "
                "was derived from"
            )
        try:
            raw = self._read_verified(ref.digest, ArtifactKind.SCORE_PACKET)
        except (FileNotFoundError, OSError, IntegrityError) as exc:
            raise MissingAuditLinkageError(
                f"score packet {ref.digest} referenced by the audit bundle for "
                f"{where} failed store verification: {exc}"
            ) from exc
        try:
            packet = ItemScore.from_json(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ScorePacketMismatchError(
                f"score packet {ref.digest} for {where} does not parse as an "
                f"ItemScore: {exc}"
            ) from exc
        mismatches = _mismatches(
            (
                ("miner_hotkey", winner.hotkey, packet.miner_hotkey),
                ("challenge_id", linkage.challenge_id, packet.challenge_id),
                ("item_id", linkage.scoring_item_id, packet.item_id),
                ("track", facts.track, packet.track),
            )
        )
        if mismatches:
            detail = "; ".join(
                f"{name}: packet says {actual!r}, the competition says {expected!r}"
                for name, expected, actual in mismatches
            )
            raise ScorePacketMismatchError(
                f"score packet {ref.digest} for {where} is not bound to this "
                f"promotion — {detail}"
            )
        # The packet must score THE OUTPUT THE BUNDLE ARCHIVES (round-4 review
        # #2). Identity fields alone leave the evidence unbound to the bytes: a
        # scoring response naming output A can otherwise sit in a bundle that
        # archives output B, both blobs verify by content address, and the
        # promoted "audited" score was measured on bytes the audit trail does
        # not preserve. Null is refused too, not treated as "unknown": every
        # packet the write side mints carries a content_digest (the scoring
        # worker stamps the digest of the private snapshot it actually measured;
        # the orchestrator's zero packets stamp the canonical empty digest), so
        # a packet that names no output cannot be audited against a bundle that
        # archives one — and `miner_output` is a required bundle slot, so it
        # always does.
        archived_output = bundle.miner_output.digest
        if packet.content_digest is None:
            raise ScorePacketMismatchError(
                f"score packet {ref.digest} for {where} carries no content_digest, "
                f"but its audit bundle archives miner output {archived_output} — a "
                "packet that does not name the bytes it measured cannot be audited "
                "against the output the bundle preserves"
            )
        if packet.content_digest != archived_output:
            raise ScorePacketMismatchError(
                f"score packet {ref.digest} for {where} scored content "
                f"{packet.content_digest}, but its audit bundle archives miner "
                f"output {archived_output} — the score must be bound to the very "
                "output the bundle preserves, or the archived bytes are not the "
                "bytes the promoted score was measured on"
            )
        # Scorer identity (vidaio/services/protocol.py): the packet is only
        # auditable against the bundle if both name the SAME scorer — a bundle
        # claiming scorer X over a packet minted by scorer Y cannot be recomputed
        # by anyone, and would let a promotion slip in an unverifiable score.
        if packet.scorer_version != bundle.scorer_version:
            raise ScorerIdentityMismatchError(
                f"score packet {ref.digest} for {where} was produced by scorer "
                f"{packet.scorer_version!r}, but its audit bundle claims "
                f"{bundle.scorer_version!r} — the packet and the bundle must name "
                "the same scorer or the score cannot be independently recomputed"
            )
        if not math.isclose(
            packet.score, linkage.item_score, rel_tol=0.0, abs_tol=_SCORE_TOLERANCE
        ):
            raise HoldoutScoreMismatchError(
                f"competition {facts.competition_id} persisted {linkage.item_score} "
                f"for {where}, but packet {ref.digest} records {packet.score} — "
                "the promoted score aggregates every item's persisted score, so "
                "each one must be the score the audit trail can reproduce"
            )

    def _verify_artifact(
        self, facts: CompetitionFacts, winner: WinnerFacts, ref: ArtifactRef
    ) -> None:
        """The artifact is DERIVED from the competition's own record, not asserted.

        `contenders.image_digest` proves the winner was BUILT; the combined
        `submission_backup_completed` reference proves the competition certified a
        backup SET; and the per-contender `contender_submission_archived` event
        names the audit-store address of THIS winner's tarball. The offered
        artifact must BE that address — a blob that merely verifies in the store
        is any blob at all, which is exactly the hole the derivation closes.
        """
        track, hotkey = facts.track, winner.hotkey
        if winner.status != BUILT or not winner.image_digest:
            raise ArtifactNotArchivedError(
                f"{track} winner {hotkey} is {winner.status} with image_digest "
                f"{winner.image_digest!r} — competition {facts.competition_id} "
                "never built an executable for this contender, so there is "
                "nothing archived to serve"
            )
        if facts.submission_backup_ref is None:
            raise ArtifactNotArchivedError(
                f"competition {facts.competition_id} has no recorded submission "
                "backup — its submissions were never archived, so no artifact "
                "from it can be proven to be a competition submission"
            )
        archived = winner.archived_artifact_digest
        if not archived:
            # Finalization refuses to certify the combined backup until every
            # contender that can still win has an archival event, so a certified
            # backup with no archived WINNER is a contradiction, not a legacy DB.
            raise ArtifactNotArchivedError(
                f"competition {facts.competition_id} certified a submission "
                f"backup but records no archived submission for {track} winner "
                f"{hotkey} (contender {winner.contender_id}) — its artifact "
                "cannot be derived, so nothing may be promoted from it"
            )
        if ref.digest != archived:
            raise ArtifactLinkageError(
                f"artifact {ref.digest} is not the submission competition "
                f"{facts.competition_id} archived for {track} winner {hotkey}: "
                f"the competition recorded {archived}"
            )
        if (
            winner.archived_artifact_bytes
            and ref.byte_size != winner.archived_artifact_bytes
        ):
            raise ArtifactLinkageError(
                f"artifact {ref.digest} is {ref.byte_size} bytes but competition "
                f"{facts.competition_id} archived {winner.archived_artifact_bytes} "
                "bytes under that digest"
            )
        if ref.byte_size <= 0:
            raise ArtifactLinkageError(
                f"artifact {ref.digest} is empty — an empty champion executable "
                "cannot serve anything"
            )
        try:
            self._store.get(ref)  # verify-on-read
        except (FileNotFoundError, OSError, IntegrityError) as exc:
            raise ArtifactNotArchivedError(
                f"champion artifact {ref.digest} for {track} winner {hotkey} is "
                f"not verifiably archived: {exc}"
            ) from exc


# ---- helpers -------------------------------------------------------------------


def _item(linkage: ItemLinkage) -> str:
    """How every per-item failure names the item it failed on."""
    return f"item {linkage.item_index} (item_id={linkage.item_id})"


def _mismatches(pairs: Iterable[tuple[str, Any, Any]]) -> list[tuple[str, Any, Any]]:
    return [
        (name, expected, actual)
        for name, expected, actual in pairs
        if expected != actual
    ]


def _require(
    error: type[RegistryError],
    field: str,
    claimed: Any,
    actual: Any,
    subject: str,
) -> None:
    """Raise `error` when a caller's claim about `field` is not what we derived.

    `None` means "not asserted" — the pipeline derives that field either way.
    Floats compare with the packet tolerance so a JSON round trip is not a lie.
    """
    if claimed is None:
        return
    if isinstance(claimed, float) and isinstance(actual, float):
        if math.isclose(claimed, actual, rel_tol=0.0, abs_tol=_SCORE_TOLERANCE):
            return
    elif claimed == actual:
        return
    raise error(
        f"asserted {field}={claimed!r} contradicts {subject}, which says {actual!r} "
        "— promotion derives its facts from the record, it does not take them on trust"
    )
