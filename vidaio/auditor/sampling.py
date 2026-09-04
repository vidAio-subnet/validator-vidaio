"""Deterministic, non-cherry-pickable audit sampling.

Sampling is [PENDING DECISION] (the project design record §8); this is the wave-6
default, HARDENED against an operator-steerable seed (#10). Two properties matter:

- REPRODUCIBLE — the seed is a pure function of ``(beacon, epoch_id, auditor_hotkey)``
  and nothing else (no wall-clock, no PRNG global state), so re-running the same
  auditor on the same epoch with the same (public, post-hoc) beacon draws the SAME
  sample; anyone can check which items it should have recomputed.
- NON-CHERRY-PICKABLE — the seed mixes in an **UNPREDICTABLE BEACON** the authority
  cannot know when it BUILDS the manifest: the on-chain anchor of THIS epoch's log
  (extrinsic hash / block hash), which only exists AFTER finalization + anchoring.
  Previously the seed was only ``sha256(epoch_id || auditor_hotkey)`` — both public
  and fixed before finalization, so the authority (which controls the manifest item
  keys) could precompute every known auditor's sample ranks and grind invalid
  item/challenge IDs to land outside every sample. Anchoring the seed to a value
  chosen by the chain AFTER the manifest is fixed removes that freedom entirely.

Beacon source (the project design record §4/§5): the anchor's ``extrinsic_hash``
from the epoch pointer; if the extrinsic hash is not yet available to the auditor,
the anchor BLOCK hash (a chain block hash at/after the anchor block) is used. Either
way it is a public, post-finalization value, so audits stay fully reproducible while
being unpredictable at manifest-build time. The value is passed in by the auditor
service (which reads the pointer); this module stays pure.

Selection is stratified by SOURCE (competition vs inference) so both tracks always
get coverage, then within each stratum items are ordered by
``sha256(seed || item_key)`` (a stable keyed shuffle) and the first
``SamplePolicy.target_count`` taken. A different auditor hotkey OR a different beacon
yields a different keyed order and therefore a different sample.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vidaio.epoch.log import AuditFileKind, AuditFileRef, AuditManifest

from vidaio.auditor.config import SamplePolicy

#: Used when no on-chain beacon is available (report/dev overlays, or an
#: un-anchored epoch). Sampling stays deterministic and reproducible but loses the
#: unpredictability guarantee — acceptable only where there is no adversarial
#: authority (tests/report mode); production always passes the real anchor beacon.
NO_BEACON = ""


@dataclass(frozen=True, slots=True)
class AuditItem:
    """One auditable (source, challenge, item): its bundle + packet refs.

    The manifest carries two ``AuditFileRef``s per item (an AUDIT_BUNDLE binding and
    the SCORE_PACKET blob); this pairs them into the single unit the auditor
    recomputes. ``uid`` is the earning uid (None for an archived-baseline row).
    """

    source: str
    challenge_id: str
    item_id: str
    uid: int | None
    bundle_ref: AuditFileRef
    packet_ref: AuditFileRef
    #: Competition subject namespace. Empty for inference/baseline calibration rows.
    subject_id: str | None = None
    #: The committed identity the sampled bundle's miner must bind to. For an
    #: EARNING row it stays None (the auditor derives the expected hotkey from `uid`->log hotkey);
    #: for a COMPETITION CONTENDER row the sampler stamps the committed `CompetitionContenderInput`
    #: hotkey here, so a sampled contender bundle enforces identity through `verify_bundle` EXACTLY
    #: like an inference/earning item (a contender bundle mined by another hotkey ⇒ FAIL). None on a
    #: archived-baseline calibration row (no chain identity to bind).
    expected_hotkey: str | None = None

    def key(self) -> str:
        """The item's IDENTITY tuple (source, challenge_id, item_id).

        This is the tuple two distinct items must NOT share (see
        :class:`DuplicateAuditIdentity`); it is NOT what the rank is computed over
.
        """
        return (
            f"{self.source}\x00{self.subject_id or ''}\x00"
            f"{self.challenge_id}\x00{self.item_id}"
        )

    def rank_key(self) -> str:
        """The TIE-FREE key the beacon-seeded rank is computed over.

        The old rank keyed only on (source, challenge_id, item_id) — it EXCLUDED the
        uid and both artifact digests — so two DIFFERENT uids reusing the same
        (source, challenge_id, item_id) got IDENTICAL ranks for every beacon/auditor.
        The authority could then stage honest evidence on early-ranked uids and
        substituted packets on later-ranked ones that are NEVER media-recomputed, yet
        the full earning fold still trusts the committed packet scores → a
        self-consistent substituted epoch reaches CLEAN. Mixing the uid AND the
        committed bundle+packet digests in makes the ranked key unique to a concrete
        committed item, so no two DISTINCT items can share a rank (barring a sha256
        collision). Un-steerability is preserved: the beacon is mixed into the SEED
        (:func:`_seed`), not the item key, so the authority still cannot know any
        rank while it controls the manifest.
        """
        uid = "baseline" if self.uid is None else str(self.uid)
        return (
            f"{self.source}\x00{self.subject_id or ''}\x00"
            f"{self.challenge_id}\x00{self.item_id}\x00{uid}"
            f"\x00{self.bundle_ref.digest}\x00{self.packet_ref.digest}"
        )


class ManifestIncomplete(Exception):
    """A manifest entry does not pair one AUDIT_BUNDLE with one SCORE_PACKET ref."""


class DuplicateAuditIdentity(Exception):
    """Two audit items share the (source, challenge_id, item_id) the sampler keys on.

    an internal review(b): the manifest permits DIFFERENT uids to reuse the same
    (source, challenge_id, item_id) identity tuple. Even with the tie-free rank key
    above (which no longer lets colliding items share a rank), a DUPLICATE IDENTITY
    is itself evidence of tampering — it is exactly what let the authority put honest
    evidence and a substituted packet under one audit identity, so an auditor could
    never tell which concrete item a rank/verdict referred to. An epoch/manifest
    carrying one is REFUSED (raised from :func:`manifest_items` / :func:`sample_items`,
    surfaced as DISPUTED on the audit path), never audited as if honest.
    """


def _pair_refs(
    refs: tuple[AuditFileRef, ...],
    uid: int | None,
    *,
    subject_id: str | None = None,
    expected_hotkey: str | None = None,
) -> list[AuditItem]:
    """Group a ref list by (source, challenge, item) into paired AuditItems."""
    by_item: dict[tuple[str, str, str], dict[AuditFileKind, AuditFileRef]] = {}
    order: list[tuple[str, str, str]] = []
    for ref in refs:
        key = (ref.source, ref.challenge_id, ref.item_id)
        if key not in by_item:
            by_item[key] = {}
            order.append(key)
        by_item[key][ref.kind] = ref
    items: list[AuditItem] = []
    for key in order:
        slots = by_item[key]
        bundle_ref = slots.get(AuditFileKind.AUDIT_BUNDLE)
        packet_ref = slots.get(AuditFileKind.SCORE_PACKET)
        if bundle_ref is None or packet_ref is None:
            raise ManifestIncomplete(
                f"item {key} is missing a "
                f"{'bundle' if bundle_ref is None else 'packet'} ref — an audit item "
                "must pair an AUDIT_BUNDLE binding with its SCORE_PACKET blob"
            )
        source, challenge_id, item_id = key
        items.append(
            AuditItem(
                source=source,
                challenge_id=challenge_id,
                item_id=item_id,
                uid=uid,
                bundle_ref=bundle_ref,
                packet_ref=packet_ref,
                subject_id=subject_id,
                expected_hotkey=expected_hotkey,
            )
        )
    return items


def _reject_duplicate_identities(items: list[AuditItem]) -> None:
    """Refuse a manifest where two audit items share the sampler's identity tuple.

    an internal review(b): the manifest lets different uids reuse the same
    (source, challenge_id, item_id). A duplicate identity is itself a tamper signal,
    so it is refused BEFORE ranking/sampling — DISPUTED, never audited as honest.
    """
    seen: dict[tuple[str, str, str, str], AuditItem] = {}
    for item in items:
        identity = (
            item.source,
            item.subject_id or "",
            item.challenge_id,
            item.item_id,
        )
        prior = seen.get(identity)
        if prior is not None:
            raise DuplicateAuditIdentity(
                "two audit items share the sampler identity "
                f"{identity} (uid {prior.uid} and uid {item.uid}) — a duplicate "
                "audit identity is evidence of tampering; refusing to audit the epoch"
            )
        seen[identity] = item


def manifest_items(manifest: AuditManifest) -> list[AuditItem]:
    """Every auditable item in a manifest (earning rows + baseline calibration rows).

    Raises :class:`DuplicateAuditIdentity` if two items share the sampler's identity
    tuple (an internal review(b)) — an ambiguous manifest is refused, not sampled.

    Schema-v14 competition evidence uses the same paired packet/bundle discipline as
    inference evidence. Every committed competition subject/item therefore enters the
    competition stratum and routes through the media-sample/verify-bundle path.
    """
    items: list[AuditItem] = []
    for uid in sorted(manifest.per_uid):
        items.extend(_pair_refs(manifest.per_uid[uid], uid))
    items.extend(_pair_refs(manifest.baseline_bundles, None))
    if manifest.competition_input is not None:
        for subject in manifest.competition_input.subjects:
            items.extend(
                _pair_refs(
                    manifest.competition_bundles[subject.subject_id],
                    subject.uid if subject.role == "contender" else None,
                    subject_id=subject.subject_id,
                    expected_hotkey=(
                        subject.hotkey if subject.role == "contender" else None
                    ),
                )
            )
    _reject_duplicate_identities(items)
    return items


def _seed(beacon: str, epoch_id: int, auditor_hotkey: str) -> bytes:
    """The sampling seed: sha256(beacon || epoch_id || auditor_hotkey).

    The beacon (post-finalization on-chain anchor value) is mixed in FIRST so the
    authority cannot know the seed while it still controls the manifest item keys.
    """
    return hashlib.sha256(
        beacon.encode("utf-8")
        + b"\x00"
        + f"{epoch_id}".encode("utf-8")
        + b"\x00"
        + auditor_hotkey.encode("utf-8")
    ).digest()


def _rank(seed: bytes, item_key: str) -> bytes:
    """The keyed-shuffle rank of an item: sha256(seed || item_key)."""
    return hashlib.sha256(seed + b"\x00" + item_key.encode("utf-8")).digest()


def sample_items(
    manifest: AuditManifest,
    *,
    epoch_id: int,
    auditor_hotkey: str,
    policy: SamplePolicy,
    beacon: str = NO_BEACON,
) -> list[AuditItem]:
    """Deterministically draw the items THIS auditor recomputes for this epoch.

    Stratified per source; within a stratum, ordered by keyed shuffle and truncated
    to ``policy.target_count``. Pure function of (manifest, beacon, epoch_id,
    auditor_hotkey, policy) — reproducible and un-steerable: the ``beacon`` is the
    post-finalization on-chain anchor value the authority could not predict when it
    built the manifest (#10). The returned list is in a stable order (source, then
    keyed rank) so a report over it is byte-deterministic.
    """
    seed = _seed(beacon, epoch_id, auditor_hotkey)
    all_items = manifest_items(manifest)

    by_source: dict[str, list[AuditItem]] = {}
    for item in all_items:
        by_source.setdefault(item.source, []).append(item)

    selected: list[AuditItem] = []
    for source in sorted(by_source):
        stratum = by_source[source]
        # Rank over the TIE-FREE key: uid + committed digests are
        # mixed in so no two DISTINCT items can share a rank. The beacon lives in the
        # SEED, so un-steerability is unchanged.
        ranked = sorted(stratum, key=lambda it: _rank(seed, it.rank_key()))
        count = policy.target_count(len(ranked))
        selected.extend(ranked[:count])
    return selected
