"""One audit invocation's bounded memo of independently measured original packets.

Only CPU scoring is memoized. verify_bundle still fetches and integrity-checks
all archived artifacts before every call, including a cached call. No memo is
stored on the long-lived service/recomputer or survives an audit invocation.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from concurrent.futures import Future
from threading import Lock

from vidaio.audit.canonical import canonical_json_bytes, sha256_hex
from vidaio.audit.store import ArtifactKind


class EpochRecomputer:
    def __init__(self, inner, *, max_entries: int = 8192):
        self._inner = inner
        self._max_entries = max_entries
        self._memo = OrderedDict()
        self._memo_lock = Lock()
        self._in_flight: dict[str, Future] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @staticmethod
    def _key(bundle, artifacts, runtime):
        packet = json.loads(artifacts[ArtifactKind.SCORE_PACKET])
        # The finalizer adds only a cycle suffix and fold metadata. Normalize
        # precisely that transformation; preserve every measured packet field.
        receipt, anchor = bundle.miner_receipt, bundle.challenge_anchor
        if receipt is None or anchor is None:
            return None
        task = receipt.metadata.task_id
        item = packet.get("item_id")
        if item not in (task, f"{task}-c{anchor.dispatch_ordering_key}"):
            return None
        if "cycle_sequence" in packet:
            if packet["cycle_sequence"] != anchor.dispatch_ordering_key or packet.get("excluded") is not False:
                return None
            packet.pop("cycle_sequence")
            packet.pop("excluded")
        packet["item_id"] = task
        # These are exactly the media/challenge inputs used by RealScoreRecomputer.
        # Metadata manifest and creation time are still verified by verify_bundle,
        # but neither changes the independent scoring computation.
        inputs = {
            "packet": packet, "runtime": runtime,
            "challenge_id": bundle.challenge_id, "miner_hotkey": bundle.miner_hotkey,
            "commitment_hash": bundle.commitment_hash,
            "anchor": anchor.model_dump(mode="json"),
            "receipt": receipt.model_dump(mode="json"),
            "scorer_version": bundle.scorer_version,
            "backend_versions": bundle.backend_versions,
            "stage": bundle.stage,
            "competition_item": bundle.competition_item,
        }
        for field in ("reference_original", "challenge_input", "miner_output", "dag_reveal"):
            ref = getattr(bundle, field)
            inputs[field] = None if ref is None else ref.model_dump(mode="json")
        return sha256_hex(canonical_json_bytes(inputs))

    def recompute(self, bundle, artifacts):
        try:
            key = self._key(bundle, artifacts, getattr(self._inner, "scorer_version", None))
        except (KeyError, TypeError, ValueError):
            key = None
        if key is None:
            return self._inner.recompute(bundle, artifacts)

        with self._memo_lock:
            if key in self._memo:
                cached = self._memo[key]
                self._memo.move_to_end(key)
                pending = None
            else:
                cached = None
                pending = self._in_flight.get(key)
                owns_call = pending is None
                if owns_call:
                    pending = Future()
                    self._in_flight[key] = pending
        if cached is not None:
            return cached.model_copy(deep=True)
        if not owns_call:
            return pending.result().model_copy(deep=True)

        # Neither CPU work nor waiting holds the LRU lock. Content witnesses can
        # recursively verify other original packets through this same wrapper.
        try:
            fresh = self._inner.recompute(bundle, artifacts)
            cached = fresh.model_copy(deep=True)
            with self._memo_lock:
                self._memo[key] = cached
                if len(self._memo) > self._max_entries:
                    self._memo.popitem(last=False)
        except BaseException as exc:
            # Publish even cancellation/system exceptions so peers cannot stay
            # blocked. A later call retries; failed results never enter the memo.
            pending.set_exception(exc)
            raise
        else:
            pending.set_result(cached)
            return fresh
        finally:
            with self._memo_lock:
                del self._in_flight[key]

    def recompute_content_duplicate(self, bundle, artifacts, store):
        return self._inner.recompute_content_duplicate(
            bundle, artifacts, store, member_recomputer=self,
        )
