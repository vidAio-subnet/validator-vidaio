"""The control result is the immutable, auditable packet-derived economy."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import httpx

from vidaio.audit import ArtifactKind
from vidaio.chain.adapter import ChainNeuron, InMemoryChain
from vidaio.competition.epoch_evidence import build_competition_epoch_evidence
from vidaio.competition.orchestrator.control import create_control_app
from vidaio.competition.orchestrator.results import result_payload
from vidaio.competition.orchestrator.service import Orchestrator
from vidaio.epoch import MinerCensusEntry
from vidaio.tokenomics.config import TokenomicsConfig

from integration_support import COMPLETED_AT, GoldenWorld

TOKEN = "economic-result-control-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _neuron(uid: int, hotkey: str) -> ChainNeuron:
    return ChainNeuron(
        uid=uid,
        hotkey=hotkey,
        coldkey=f"ck-{hotkey}",
        ip=f"203.0.113.{uid}",
        alpha_stake=1_000.0,
        emission=0.0,
    )


def _chain(*hotkeys: str) -> InMemoryChain:
    uid_by_hotkey = {"hk-a": 10, "hk-b": 11}
    return InMemoryChain(
        _neurons=[_neuron(uid_by_hotkey[hotkey], hotkey) for hotkey in hotkeys],
        block_time_anchor=(1, COMPLETED_AT),
    )


@dataclass
class _ResultOnlyOrchestrator:
    """Minimal host for the real Orchestrator.build_result implementation."""

    conn: object
    store: object
    chain: object | None
    tokenomics: object
    name: str = "competition-orchestrator"

    def build_result(self, competition_id: str, **kwargs):
        return Orchestrator.build_result(self, competition_id, **kwargs)


async def test_result_requires_census_and_evidence_and_ignores_manual_review_state(
    fresh_world: GoldenWorld, monkeypatch
) -> None:
    world = fresh_world
    competition_id = world.manifest.competition_id
    host = _ResultOnlyOrchestrator(
        conn=world.comp_conn,
        store=world.store,
        chain=None,
        tokenomics=TokenomicsConfig(competition_emissions_enabled=True),
    )
    app = create_control_app(host, token=TOKEN)  # type: ignore[arg-type]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://control"
    ) as client:
        # A completed DB row without a current chain census cannot silently emit
        # UNKNOWN_UID contenders or a smaller payout set.
        no_census = await client.get(
            f"/competitions/{competition_id}/result", headers=AUTH
        )
        assert no_census.status_code == 409
        assert no_census.json()["detail"]["code"] == "unauditable_result"

        # Likewise, a partial census may not cherry-pick the registered contender.
        host.chain = _chain("hk-a")
        partial_census = await client.get(
            f"/competitions/{competition_id}/result", headers=AUTH
        )
        assert partial_census.status_code == 409
        assert partial_census.json()["detail"]["code"] == "unauditable_result"
        assert "absent from the close-block census" in partial_census.text

        host.chain = _chain("hk-a", "hk-b")
        request_thread = threading.get_ident()
        census_threads: list[int] = []
        refresh_threads: list[int] = []
        raw_neurons = host.chain.neurons
        raw_refresh = host.chain.refresh

        def traced_refresh():
            refresh_threads.append(threading.get_ident())
            return raw_refresh()

        def traced_neurons():
            census_threads.append(threading.get_ident())
            return raw_neurons()

        monkeypatch.setattr(host.chain, "refresh", traced_refresh)
        monkeypatch.setattr(host.chain, "neurons", traced_neurons)
        expected_evidence = build_competition_epoch_evidence(
            world.comp_conn,
            census_by_hotkey={
                neuron.hotkey: MinerCensusEntry(
                    uid=neuron.uid,
                    hotkey=neuron.hotkey,
                    coldkey=neuron.coldkey,
                    ip=neuron.ip,
                )
                for neuron in raw_neurons()
            },
            store=world.store,
            tokenomics=TokenomicsConfig(competition_emissions_enabled=True),
            competition_id=competition_id,
            through_time=COMPLETED_AT,
        )
        assert expected_evidence is not None
        expected = result_payload(expected_evidence.result)
        expected["source"] = "packet_mean.current_census_preview.v1"
        expected["identity_snapshot"] = "current_unpinned_chain_head"
        expected["authoritative_emitted_result"] = False
        before_review = await client.get(
            f"/competitions/{competition_id}/result", headers=AUTH
        )
        assert before_review.status_code == 200, before_review.text
        assert before_review.json() == expected
        assert [row["uid"] for row in before_review.json()["contenders"]] == [10, 11]
        assert refresh_threads and refresh_threads[-1] != request_thread
        assert census_threads and census_threads[-1] != request_thread
        assert refresh_threads[-1] == census_threads[-1]

        # Manual review/ranking columns are operational metadata. They cannot
        # rewrite already-earned packet economics or remove an audited contender.
        world.comp_conn.execute(
            "UPDATE contenders SET manual_disqualified = 1, eligible = 0, "
            "final_rank = NULL WHERE competition_id = ? AND hotkey = 'hk-a'",
            (competition_id,),
        )
        after_review = await client.get(
            f"/competitions/{competition_id}/result", headers=AUTH
        )
        assert after_review.status_code == 200, after_review.text
        assert after_review.json() == before_review.json()

        # Make one committed bundle unresolvable without mutating any economic
        # score. The endpoint must fail closed instead of paying from DB aggregates.
        missing_digest = world.bundle_digests["hk-b"]
        original_get = world.store.get_digest_limited

        def missing_bundle(kind, digest, *, max_bytes):
            if kind is ArtifactKind.AUDIT_BUNDLE and digest == missing_digest:
                raise FileNotFoundError(digest)
            return original_get(kind, digest, max_bytes=max_bytes)

        monkeypatch.setattr(world.store, "get_digest_limited", missing_bundle)
        missing_evidence = await client.get(
            f"/competitions/{competition_id}/result", headers=AUTH
        )
        assert missing_evidence.status_code == 409
        assert missing_evidence.json()["detail"]["code"] == "unauditable_result"
        assert missing_digest in missing_evidence.text
