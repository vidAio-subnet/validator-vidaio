"""Replay validates provenance and coverage before classifying content."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json

import pytest

from scripts import replay_distinct_content as replay
from tests.auditor.test_duplicate_evidence import _case, _signed_receipt, _receipt_ok
from vidaio.audit import ArtifactKind, canonical_json_bytes
from vidaio.audit.commitments import merkle_proof, merkle_root
from vidaio.scoring.result import ItemScore


@pytest.fixture
def case(tmp_path, monkeypatch):
    store, _, _, _, template, *_ = _case(tmp_path)
    uid, miner = 23, 'miner-23'
    output = store.put(b'owned output fixture', ArtifactKind.MINER_OUTPUT)
    receipt = _signed_receipt(validator='validator-hotkey', miner=miner, uid=uid,
        challenge_id=template.challenge_id, track='compression', anchor=template.challenge_anchor,
        input_digest=template.challenge_input.digest, input_size=template.challenge_input.byte_size,
        output_digest=output.digest, output_size=output.byte_size)
    original = ItemScore(item_id=receipt.metadata.task_id, challenge_id=template.challenge_id,
        track='compression', miner_hotkey=miner, content_digest=output.digest,
        score=0.5, gate_passed=True, scorer_version=template.scorer_version,
        backend_versions=template.backend_versions)
    packet = original.model_copy(update={'item_id': f'{original.item_id}-c{template.challenge_anchor.dispatch_ordering_key}'})
    packet_ref = store.put(packet.to_json().encode(), ArtifactKind.SCORE_PACKET)
    bundle = template.model_copy(update={'item_id': packet.item_id, 'miner_hotkey': miner,
        'miner_output': output, 'miner_receipt': receipt, 'score_packet': packet_ref})
    ref = {'kind': 'audit_bundle', 'source': 'inference', 'digest': bundle.bundle_digest(),
           'challenge_id': bundle.challenge_id, 'item_id': bundle.item_id,
           'committed_track': None, 'inclusion_proof': None}
    leaves = [packet_ref.digest]
    peer = ref | {'kind': 'score_packet', 'digest': packet_ref.digest, 'committed_track': packet.track,
                  'inclusion_proof': merkle_proof(leaves, packet_ref.digest)}
    log = {'created_at': '2026-09-06T00:00:00+00:00', 'epoch_id': 24963,
           'audit_manifest': {'per_uid': {str(uid): [ref, peer]}, 'score_packet_merkle_root': merkle_root(leaves)}}
    monkeypatch.setattr(replay, 'verify_miner_artifact_receipt', _receipt_ok)
    return uid, bundle, packet, original, ref, log


def test_binding_accepts_exact_archive_and_original_round(case):
    uid, bundle, packet, original, ref, log = case
    replay.validate_packet_binding(uid, ref, bundle, packet, log)
    replay.validate_signed_binding(uid, bundle, packet)
    raw = original.to_json()
    row = {'round_id': replay.TARGET_ROUND, 'uid': uid, 'item_id': original.item_id,
           'challenge_id': original.challenge_id, 'track': original.track,
           'miner_hotkey': original.miner_hotkey, 'score': original.score,
           'packet_json': raw, 'packet_digest': hashlib.sha256(raw.encode()).hexdigest()}
    index = replay.load_round_index({'packets': [row]})
    assert replay.bind_round(uid, bundle, packet, index) == replay.TARGET_ROUND
    with pytest.raises(ValueError, match='beyond chunk'):
        replay.bind_round(uid, bundle, packet.model_copy(update={'score': 0.6}), index)
    with pytest.raises(ValueError, match='digest'):
        replay.load_round_index({'packets': [row | {'packet_digest': 'a' * 64}]})


@pytest.mark.parametrize('field,value', [
    ('item_id', 'other'), ('challenge_id', 'other'), ('miner_hotkey', 'other'),
    ('scorer_version', 'other'), ('backend_versions', {'different': 'backend'}),
])
def test_packet_identity_substitution_is_refused(case, field, value):
    uid, bundle, packet, _, ref, log = case
    with pytest.raises(ValueError, match='binding'):
        replay.validate_packet_binding(uid, ref, bundle, packet.model_copy(update={field: value}), log)


def test_merkle_and_canonical_bundle_binding_required(case):
    uid, bundle, packet, _, ref, log = case
    with pytest.raises(ValueError, match='canonical digest'):
        replay.validate_packet_binding(uid, ref | {'digest': 'a' * 64}, bundle, packet, log)
    changed = deepcopy(log)
    changed['audit_manifest']['score_packet_merkle_root'] = 'a' * 64
    with pytest.raises(ValueError, match='Merkle'):
        replay.validate_packet_binding(uid, ref, bundle, packet, changed)
    changed = deepcopy(log)
    changed['audit_manifest']['per_uid'][str(uid)][1]['committed_track'] = 'upscaling'
    with pytest.raises(ValueError, match='track'):
        replay.validate_packet_binding(uid, ref, bundle, packet, changed)


@pytest.mark.parametrize('change', ['uid', 'track', 'digest', 'input', 'anchor', 'commitment', 'signature'])
def test_signed_task_media_anchor_and_signature_are_bound(case, change):
    uid, bundle, packet, *_ = case
    if change == 'uid': uid += 1
    if change == 'track': packet = packet.model_copy(update={'track': 'upscaling'})
    if change == 'digest': packet = packet.model_copy(update={'content_digest': 'a' * 64})
    if change == 'input': bundle = bundle.model_copy(update={'challenge_input': bundle.challenge_input.model_copy(update={'byte_size': 999})})
    if change == 'anchor': bundle = bundle.model_copy(update={'challenge_anchor': bundle.challenge_anchor.model_copy(update={'block_hash': 'a' * 64})})
    if change == 'commitment': bundle = bundle.model_copy(update={'commitment_hash': 'a' * 64})
    if change == 'signature': bundle = bundle.model_copy(update={'miner_receipt': bundle.miner_receipt.model_copy(update={'response_signature': 'a' * 128})})
    with pytest.raises(ValueError):
        replay.validate_signed_binding(uid, bundle, packet)


def test_inventory_missing_log_cannot_disappear_from_denominator(case, tmp_path):
    uid, _, _, _, ref, log = case
    inventory = [{'epoch_id': log['epoch_id'], 'uid': uid, 'created_at': log['created_at'], 'ref': r}
                 for r in log['audit_manifest']['per_uid'][str(uid)]]
    since = datetime.fromisoformat('2026-09-04T14:30:00+00:00')
    entries, _, coverage = replay.load_entries(tmp_path, since, inventory)
    assert not entries and coverage['expected_bundles'] == 1
    assert len(coverage['problems']) == 2
    path = tmp_path / 'finalized/epoch=24963/log.json'
    path.parent.mkdir(parents=True)
    path.write_bytes(canonical_json_bytes(log))
    # JSON round-trip normalizes Merkle tuple proofs to arrays.
    inventory = json.loads(json.dumps(inventory))
    entries, _, coverage = replay.load_entries(tmp_path, since, inventory)
    assert len(entries) == 1 and not coverage['problems']
    with pytest.raises(ValueError, match='duplicate expected'):
        replay.load_entries(tmp_path, since, inventory + [inventory[0]])


def test_external_inventory_pin_and_cache_paths(tmp_path):
    inventory = tmp_path / 'inventory.json'
    inventory.write_text('[]')
    assert replay.pinned_json(inventory, hashlib.sha256(b'[]').hexdigest()) == []
    with pytest.raises(ValueError): replay.pinned_json(inventory, 'a' * 64)
    with pytest.raises(ValueError): replay.object_path(tmp_path, {'kind': '../../outside', 'digest': 'a' * 64})
    with pytest.raises(ValueError): replay.object_path(tmp_path, {'kind': 'miner_output', 'digest': '../../outside'})
    outside = tmp_path / 'outside'
    outside.write_bytes(b'secret outside cache')
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    cache = tmp_path / 'cache'
    key = f'miner_output/{digest[:2]}/{digest[2:4]}/{digest}'
    link = cache / key
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        replay.object_path(cache, {'kind': 'miner_output', 'digest': digest})


def test_domain_keeps_epochs_anchors_tracks_and_content_origins_separate():
    base = dict(epoch_id=1, challenge_id='challenge', commitment_hash='c', anchor_identity='a',
                track='compression', reference_digest='r', challenge_input_digest='i', canonicalization_plan_digest='p')
    for field in base:
        assert replay.content_domain(base) != replay.content_domain(base | {field: 'different'})


def test_gate_passed_zero_is_ineligible_like_a_gate_failure(case):
    _, _, packet, *_ = case
    assert replay.ineligible_status(packet) is None
    assert replay.ineligible_status(packet.model_copy(update={'score': 0.0})) == 'not_eligible_zero_score'
    assert replay.ineligible_status(packet.model_copy(update={'gate_passed': False, 'score': 0.0})) == 'not_eligible_gate_failed'
