"""Archive-only commit-window and cumulative round-cursor verification.

Commit heights are authority declarations authenticated by the epoch log. Anchor
heights and dispatch keys must agree with the existing archived challenge evidence.
This pass covers the complete index in both modes, independently of media sampling.
Economics never keys off ``round_id``: that authority-declared grouping label alone
causes no fold, zero or skip. Exact-once evidence is the anchored challenge/dispatch
key; the label only supports a within-log commit-height consistency check (D-033).
"""
from __future__ import annotations

import json

from vidaio.audit.bundle import AuditBundle
from vidaio.audit.canonical import sha256_hex
from vidaio.audit.store import ArtifactKind, IntegrityError
from vidaio.auditor.report import (
    ItemVerdict, ItemVerdictKind, ROUND_COMMIT_MISMATCH,
    ROUND_COMMIT_STALE, ROUND_COMMIT_UNVERIFIED,
)
from vidaio.challenge import ChallengeCommitment
from vidaio.epoch.log import AuditFileKind, RoundCommitInput, _HistoryEpochLogV16
from vidaio.validator.availability import AvailabilityObservation, verify_availability_observation

MAX_METADATA = 16 * 1024 * 1024


class MembershipInvalid(ValueError):
    """Archived bytes or declared membership contradict a required invariant."""


class MembershipUnavailable(Exception):
    """The evidence needed for an independent comparison cannot be read."""


def _verdict(kind, detail, *, challenge="", code=None):
    return ItemVerdict(
        source="round_membership", challenge_id=challenge,
        item_id="round-commit", bundle_digest="", packet_digest="",
        verdict=kind, code=code or (ROUND_COMMIT_MISMATCH if kind is ItemVerdictKind.FAIL
                                  else ROUND_COMMIT_UNVERIFIED if kind is ItemVerdictKind.SKIP else ""),
        detail=detail,
    )


def _read(store, ref):
    raw = store.get_limited(ref, MAX_METADATA)
    if len(raw) != ref.byte_size or sha256_hex(raw) != ref.digest:
        raise MembershipInvalid("archived artifact differs from its hash/size reference")
    return raw


def _bundle(service, ref):
    from vidaio.auditor.service import BundleUnavailable
    try:
        bundle = service._bundle_source.bundle_for(ref)
    except BundleUnavailable as exc:
        if isinstance(exc.__cause__, (IntegrityError, ValueError)):
            raise MembershipInvalid("round bundle integrity/shape differs") from exc
        raise MembershipUnavailable("round bundle cannot be read") from exc
    if bundle.bundle_digest() != ref.digest:
        raise MembershipInvalid("round bundle canonical digest differs")
    if (bundle.challenge_id, bundle.item_id) != (ref.challenge_id, ref.item_id):
        raise MembershipInvalid("round bundle identity differs from its epoch reference")
    return bundle


def _bind_anchor(entry, anchor, *, track=None, ordering_key=None):
    if anchor is None:
        raise MembershipInvalid("round evidence lacks a pre-dispatch challenge anchor")
    if anchor.block != entry.anchor_block or anchor.dispatch_ordering_key != entry.ordering_key:
        raise MembershipInvalid("round index anchor height/dispatch key differs from archived challenge evidence")
    if ordering_key is not None and ordering_key != entry.ordering_key:
        raise MembershipInvalid("economic fold ordering differs from its round commit entry")
    return (anchor.commitment_hash, anchor.block, anchor.dispatch_ordering_key, track)


def _bind_bundle(entry, bundle, store, *, expected_track=None, packet=None):
    anchor = bundle.challenge_anchor
    identity = _bind_anchor(entry, anchor, track=expected_track)
    if anchor.commitment_hash != bundle.commitment_hash:
        raise MembershipInvalid("bundle commitment differs from its archived challenge anchor")
    if bundle.dag_reveal is None:
        raise MembershipInvalid("round bundle lacks its committed dispatch preimage")
    raw = _read(store, bundle.dag_reveal)
    if sha256_hex(raw) != bundle.commitment_hash:
        raise MembershipInvalid("dispatch preimage does not hash to the anchored commitment")
    dispatch = ChallengeCommitment.committed_dispatch_from_preimage(raw)
    if dispatch is None or dispatch[1] != entry.ordering_key:
        raise MembershipInvalid("round ordering key differs from committed dispatch preimage")
    if expected_track is not None and dispatch[0] != expected_track:
        raise MembershipInvalid("round track differs from committed dispatch preimage")
    if packet is not None:
        if (type(packet.get("cycle_sequence")) is not int
                or packet.get("challenge_id") != entry.challenge_id
                or packet.get("cycle_sequence") != entry.ordering_key
                or packet.get("track") != dispatch[0]):
            raise MembershipInvalid("packet challenge/track/sequence differs from its committed round")
    return (*identity[:3], dispatch[0])


def audit_round_membership(service, log, store, prior_log=None, is_genesis=True):
    """Return conclusive membership faults, honest refusals, or report-only lateness."""
    if isinstance(log, _HistoryEpochLogV16):
        return ()
    if log.schema_version != 17 or log.round_membership != "commit/1":
        return (_verdict(ItemVerdictKind.FAIL, "current epoch lacks mandatory commit/1 membership"),)
    manifest = log.audit_manifest
    try:
        entries = tuple(RoundCommitInput.model_validate(e.model_dump(mode="python"))
                        for e in manifest.round_commits)
        if manifest.round_commit_cursor is not None and type(manifest.round_commit_cursor) is not int:
            raise MembershipInvalid("round cursor must be a strict integer or null")
        by_challenge = {e.challenge_id: e for e in entries}
        keys = [e.ordering_key for e in entries]
        if len(by_challenge) != len(entries) or len(set(keys)) != len(keys):
            raise MembershipInvalid("round commit index repeats a challenge or global dispatch key")
        round_blocks = {}
        for entry in entries:
            if entry.ordering_key < 1:
                raise MembershipInvalid("round dispatch ordering key must be positive")
            previous = round_blocks.setdefault(entry.round_id, entry.commit_block)
            if previous != entry.commit_block:
                raise MembershipInvalid("one atomic round claims different commit heights")
            if not entry.anchor_block <= entry.commit_block <= log.close_block:
                raise MembershipInvalid("round must satisfy anchor_block <= commit_block <= epoch close")
        required = {ref.challenge_id for refs in manifest.per_uid.values() for ref in refs
                    if ref.source == "inference"}
        required.update(e.challenge_id for e in manifest.availability_inputs)
        required.update(e.challenge_id for e in manifest.content_rounds)
        if required != set(by_challenge):
            raise MembershipInvalid("round commit index must exactly cover admitted media/availability/content contexts; "
                                    f"missing={sorted(required - set(by_challenge))}, unbacked={sorted(set(by_challenge) - required)}")
        if prior_log is None:
            if log.prior_log_digest is not None:
                return (_verdict(ItemVerdictKind.SKIP, "published predecessor unavailable; round commit window/cursor cannot be verified"),)
            if not is_genesis:
                return (_verdict(ItemVerdictKind.FAIL, "non-genesis epoch omitted its published predecessor"),)
            prior_close, prior_cursor = -1, None
        else:
            if log.prior_log_digest != prior_log.log_digest():
                raise MembershipInvalid("round commit predecessor does not match the archived prior digest")
            prior_close = prior_log.close_block
            if isinstance(prior_log, _HistoryEpochLogV16):
                prior_cursor = max((k for k in prior_log.audit_manifest.fold_cursors.values() if k is not None), default=None)
            else:
                prior_cursor = prior_log.audit_manifest.round_commit_cursor
            if prior_close >= log.close_block:
                raise MembershipInvalid("round membership close blocks do not advance")
        if any(e.commit_block <= prior_close for e in entries):
            raise MembershipInvalid("round is outside strict prior_published_close < commit_block <= close; late/repeated fold")
        if prior_cursor is not None and any(k <= prior_cursor for k in keys):
            raise MembershipInvalid("round dispatch key was already consumed by the cumulative round cursor")
        expected_cursor = max(keys + ([] if prior_cursor is None else [prior_cursor]), default=None)
        if manifest.round_commit_cursor != expected_cursor:
            raise MembershipInvalid("round cursor must preserve the predecessor and advance exactly through this epoch's complete index")
    except (ValueError, TypeError, AttributeError) as exc:
        return (_verdict(ItemVerdictKind.FAIL, str(exc)),)

    verdicts = []
    observed = {}

    def check(challenge, operation):
        try:
            identity = operation()
            old = observed.setdefault(challenge, identity)
            if old != identity:
                raise MembershipInvalid("one challenge claims inconsistent archived anchor/track evidence")
        except (MembershipUnavailable, FileNotFoundError, OSError) as exc:
            verdicts.append(_verdict(ItemVerdictKind.SKIP, str(exc), challenge=challenge))
        except Exception as exc:
            verdicts.append(_verdict(ItemVerdictKind.FAIL, f"{type(exc).__name__}: {exc}", challenge=challenge))

    for uid, refs in manifest.per_uid.items():
        packets = {(r.challenge_id, r.item_id): r for r in refs
                   if r.source == "inference" and r.kind is AuditFileKind.SCORE_PACKET}
        bundles = {(r.challenge_id, r.item_id): r for r in refs
                   if r.source == "inference" and r.kind is AuditFileKind.AUDIT_BUNDLE}
        if packets.keys() != bundles.keys():
            verdicts.append(_verdict(ItemVerdictKind.FAIL, f"uid {uid} inference packet/bundle pairing differs"))
            continue
        for identity, packet_ref in packets.items():
            entry = by_challenge[identity[0]]
            def media(entry=entry, packet_ref=packet_ref, bundle_ref=bundles[identity]):
                bundle = _bundle(service, bundle_ref)
                if bundle.score_packet.digest != packet_ref.digest:
                    raise MembershipInvalid("round bundle refers to another economic packet")
                packet = json.loads(_read(store, bundle.score_packet))
                return _bind_bundle(entry, bundle, store, expected_track=packet_ref.committed_track, packet=packet)
            check(entry.challenge_id, media)

    for item in manifest.availability_inputs:
        entry = by_challenge[item.challenge_id]
        def availability(item=item, entry=entry):
            observation = AvailabilityObservation.model_validate_json(item.observation_json)
            if observation.canonical_bytes().decode() != item.observation_json or observation.digest() != item.observation_digest:
                raise MembershipInvalid("availability canonical observation/digest differs")
            verify_fn = service._availability_verify_fn
            try:
                valid = (verify_availability_observation(observation) if verify_fn is None else
                         verify_availability_observation(observation, verify_fn=verify_fn))
            except Exception as exc:
                raise MembershipUnavailable("availability signature verifier unavailable") from exc
            if not valid:
                raise MembershipInvalid("availability signatures are invalid")
            attempt = observation.attempt
            if (attempt.challenge_id, attempt.uid, attempt.miner_hotkey, attempt.item_id, attempt.track) != (
                    item.challenge_id, item.uid, item.hotkey, item.item_id, item.track):
                raise MembershipInvalid("availability declaration differs from signed request identity")
            return _bind_anchor(entry, attempt.request.metadata.commitment_anchor,
                                track=item.track, ordering_key=item.ordering_key)
        check(entry.challenge_id, availability)

    for item in manifest.content_rounds:
        entry = by_challenge[item.challenge_id]
        def content(item=item, entry=entry):
            from vidaio.scoring.content_duplicate_evidence import parse_content_round
            context = parse_content_round(item.round_json)
            if context.digest() != item.round_digest or context.challenge_id != entry.challenge_id:
                raise MembershipInvalid("content context identity/digest differs")
            template = AuditBundle.model_validate_json(_read(store, item.template_bundle))
            if template.bundle_digest() != item.template_bundle.digest or template.challenge_id != entry.challenge_id:
                raise MembershipInvalid("content template canonical identity/digest differs")
            if template.challenge_anchor != context.commitment_anchor:
                raise MembershipInvalid("content template/context anchors differ")
            return _bind_bundle(entry, template, store, expected_track=context.track)
        check(entry.challenge_id, content)

    failed = {v.challenge_id for v in verdicts}
    for entry in entries:
        if entry.challenge_id in failed:
            continue
        age = entry.commit_block - entry.anchor_block
        if age > service._config.round_commit_max_blocks:
            verdicts.append(_verdict(ItemVerdictKind.PASS,
                f"round commit lag {age} blocks exceeds configured {service._config.round_commit_max_blocks}; report-only operational finding",
                challenge=entry.challenge_id, code=ROUND_COMMIT_STALE))
    return tuple(verdicts)
