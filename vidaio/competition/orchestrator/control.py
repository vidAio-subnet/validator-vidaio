"""Competition CONTROL API.

Before this, a competition could only be driven by calling Python methods on an
in-process Orchestrator: the live stack started an orchestrator nobody could
reach, commitment anchoring wrote SQLite and never touched the ChainAdapter, and
nothing turned a finished competition into a tokenomics input. The e2e suite
bridged all three by hand, which is exactly the drift the project design record rule 8
forbids: the chainless "report" path and the future real-chain path must exercise
the SAME service code.

Routes (all require auth; JSON in, JSON out):

  POST /competitions
      {"manifest": {...CompetitionManifest...}}
      -> 201 {"competition_id", "status"}
      422 invalid manifest · 409 already exists / another competition is running

  POST /competitions/{competition_id}/contenders
      {"hotkey", "repo_url", "commit_sha", "tree_sha", "stake"}
      -> 201 {"contender_id"}
      404 unknown competition · 409 enrollment refused (window/stake/duplicate)

  POST /competitions/{competition_id}/items
      {"input_name", "reference_name"?, "upscale_factor"?, "item_index",
       "threshold_commitment", "challenge_id"?, "length_seconds"?}
      -> 201 the committed digest binding. Names are basenames under the dedicated
      ``<work_dir>/ingest`` directory, never arbitrary host paths. Files are
      regular/no-symlink, bounded and copied with O_NOFOLLOW. This route is open
      only while SCHEDULED and before anchoring.

  POST /competitions/{competition_id}/anchor
      {"reward_param_digest", "baseline_image_digest"?, "baseline_tree_digest"?}
      -> 200 {"competition_id", "root", "tx_id", "baseline_image_digest",
              "payload_hex", "canonical_json", "anchor_block",
              "anchor_block_hash", "finalized_block", "archive_verified"}
      When the manifest declares a baseline, omitting ``baseline_image_digest`` makes
      the orchestrator build it in this fresh runtime and anchor/return the
      learned exact digest. A supplied value remains a strict assertion.
      The manifest digest and the dataset-selection seed commitment are read from
      the PERSISTED manifest — a caller can never substitute them. The anchoring
      right is CLAIMED in the DB first, the payload is then anchored THROUGH the
      injected ChainAdapter (report mode records it; the real chain submits it),
      and independently verified at its exact finalized archive inclusion block
      before it is marked anchored: one anchor path, no drift, and at most one
      chain write per competition even under ambiguous or concurrent requests
.
      503 no ChainAdapter or no safe commitment capacity · 409 REFUSED BEFORE
      ANY CHAIN WRITE (already
      anchored / not SCHEDULED / another anchor in flight / an ambiguous claim
      from a crashed attempt) · 502 an exact finalized/archive receipt could not
      be proved · 422 missing
      reward digest or an unbuildable manifest with no explicit baseline digest
      A 409 always carries a machine-readable `code`, and it always means NOTHING
      was written to the chain by this request.

  POST /competitions/{competition_id}/anchor/release
      {"operator", "reason"}
      -> 200 {"released": true|false}
      Operator resolution of an AMBIGUOUS anchor claim left by a crashed or
      timed-out attempt: stale identical recovery is read-only; a human uses this
      route only after independently proving that nothing landed. Deliberately
      manual — auto-releasing would restore the double-anchor hazard.

  POST /competitions/{competition_id}/halt/clear
      {"operator", "reason"}
      -> 200 {"cleared": true|false}
      Authenticated operator recovery after fixing a systemic blocker. A
      successful clear records both fields in the append-only event log before
      phase work can resume.

  GET  /competitions/{competition_id}
      -> 200 {competition_id, track, status, halted, commitment_root, timings,
              contenders[], podium[]}

  POST /competitions/{competition_id}/review
      {"contender_id", "action", "reviewer", "reason", "detail"?,
       "supersedes_review_id"?}
      -> 201 {"review_id"}  (re-ranks inside vidaio.competition.review)
      409 outside the review window / invalid action

  GET  /competitions/{competition_id}/result
      -> 200 the tokenomics CompetitionResult payload (see results.py)
      409 the competition has not COMPLETED

AUTH: a single bearer token from config (`orchestrator.control_token`), compared
with hmac.compare_digest, accepted as ``Authorization: Bearer <token>`` or
``X-Control-Token: <token>``. An EMPTY token means the API is NOT SERVED at all
(vidaio.competition.orchestrator.service refuses to start it) — this surface
creates competitions and anchors commitments; unauthenticated is not an option.

CONCURRENCY: handlers run on the orchestrator's own event loop and use its
SQLite connection. That is safe because no transaction in this package is ever
held across an ``await`` (all repository/persistence writes are synchronous
blocks), so a control mutation can never interleave with a phase-stage
transaction. The only awaiting handler is /anchor — and that await is now
BRACKETED by DB work: the anchoring right is claimed in a synchronous BEGIN
IMMEDIATE transaction BEFORE the chain call, and completion is recorded after it.
Two concurrent /anchor requests therefore both run the claim step to completion
before either can suspend, so the second is refused (409) without reaching the
chain.
"""

from __future__ import annotations

import asyncio
import hmac
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vidaio.audit import NotConfiguredError
from vidaio.competition import CompetitionManifest
from vidaio.competition import repository as repo
from vidaio.competition.epoch_evidence import CompetitionEvidenceError
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.orchestrator.results import ResultNotReady, result_payload
from vidaio.competition.review import ReviewError
from vidaio.competition.runners.errors import SandboxRunnerError
from vidaio.services.commitment_capacity import CommitmentCapacityError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vidaio.competition.orchestrator.service import Orchestrator

SHA256_HEX = r"^[0-9a-f]{64}$"


class CreateCompetitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, Any]


class EnrollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotkey: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)
    commit_sha: str = Field(min_length=1)
    tree_sha: str = Field(min_length=1)
    stake: float = Field(ge=0)


class EvaluationItemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_name: str = Field(min_length=1, max_length=255)
    reference_name: str | None = Field(default=None, min_length=1, max_length=255)
    upscale_factor: Literal[2, 4] | None = None
    target_width: int | None = Field(default=None, gt=0)
    target_height: int | None = Field(default=None, gt=0)
    item_index: int = Field(ge=0)
    threshold_commitment: str = Field(pattern=SHA256_HEX)
    challenge_id: str | None = Field(default=None, min_length=1)
    length_seconds: float | None = Field(default=None, gt=0)


class AnchorRequest(BaseModel):
    """Only the digests the competition DB cannot know are accepted from callers.

    manifest_digest and dataset_selection_seed_commitment always come from the
    PERSISTED manifest — a caller can never anchor a commitment over a manifest
    the validator did not store.
    """

    model_config = ConfigDict(extra="forbid")

    # Optional when the manifest declares a baseline: the orchestrator builds it in
    # this same fresh runtime and returns/anchors the learned exact digest. A
    # caller-supplied value remains an assertion and must match. Manifests without
    # a buildable baseline still require it explicitly.
    baseline_image_digest: str | None = Field(default=None, pattern=SHA256_HEX)
    reward_param_digest: str = Field(pattern=SHA256_HEX)
    #: Defaults to sha256(lowercased baseline tree sha) from the manifest's baseline.
    baseline_tree_digest: str | None = Field(default=None, pattern=SHA256_HEX)


class ReleaseAnchorClaimRequest(BaseModel):
    """Operator resolution of an ambiguous anchor claim."""

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1)
    #: What the operator actually checked on chain — recorded verbatim in the
    #: append-only event log, because this is the one place a human overrides a
    #: fail-closed guard.
    reason: str = Field(min_length=1)


class ClearHaltRequest(BaseModel):
    """Audited operator acknowledgement that a systemic blocker was fixed."""

    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=1000)

    @field_validator("operator", "reason")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must contain non-whitespace text")
        return value


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contender_id: int
    action: str
    reviewer: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    detail: dict[str, Any] | None = None
    supersedes_review_id: int | None = None


def create_control_app(
    orch: "Orchestrator", *, token: str, hotkey_guard: object | None = None
) -> FastAPI:
    """Build the control app bound to a live Orchestrator. Requires a token.

    P2: when a ``hotkey_guard`` is supplied, contender ENROLLMENT is self-signed
    — the enrolling miner signs its own request (Scheme A), the signer hotkey
    must equal the enrolled hotkey, the hotkey must be REGISTERED on the subnet,
    and it must clear the configured minimum alpha stake ("registered + signed +
    staked, or no sandbox build" — owner decision 2026-08-26). Operator-only
    routes stay operator-token (operators are not hotkeys)."""
    if not token:
        raise NotConfiguredError(
            "the competition control API refuses to serve without "
            "orchestrator.control_token — it creates competitions and anchors "
            "commitments; an unauthenticated control plane is not an option"
        )
    # Imported lazily: service imports control, control must not import service.
    from vidaio.competition.orchestrator.service import (
        AnchorClaimRefused,
        AnchorError,
        EarningManifestError,
    )

    app = FastAPI(title="vidaio competition control", docs_url=None, redoc_url=None)

    def authenticate(request: Request) -> None:
        presented = request.headers.get("x-control-token") or ""
        header = request.headers.get("authorization") or ""
        if not presented and header.lower().startswith("bearer "):
            presented = header[7:].strip()
        if not presented or not hmac.compare_digest(presented, token):
            raise HTTPException(status_code=401, detail="unauthorized")

    def require_competition(competition_id: str) -> repo.CompetitionRecord:
        comp = repo.get_competition(orch.conn, competition_id)
        if comp is None:
            raise HTTPException(status_code=404, detail="unknown competition")
        return comp

    @app.middleware("http")
    async def require_token(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Authenticate BEFORE routing and body validation.

        A per-handler check runs after FastAPI has already parsed and validated
        the body, which lets an anonymous caller probe the request schema (422 vs
        401). The middleware closes that: everything except /healthz answers 401
        first, and the handlers re-check anyway (defence in depth).
        """
        if request.url.path != "/healthz":
            try:
                authenticate(request)
            except HTTPException as exc:
                return JSONResponse(
                    status_code=exc.status_code, content={"detail": exc.detail}
                )
        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"service": orch.name, "status": "ok"}

    @app.post("/competitions", status_code=201)
    async def create_competition(body: CreateCompetitionRequest, request: Request):
        authenticate(request)
        try:
            manifest = CompetitionManifest.model_validate(body.manifest)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_manifest", "message": str(exc)},
            ) from exc
        try:
            orch.create_competition(manifest, orch.now())
        except EarningManifestError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "earning_manifest_refused", "message": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=409, detail={"code": "not_created", "message": str(exc)}
            ) from exc
        comp = require_competition(manifest.competition_id)
        return {"competition_id": comp.competition_id, "status": comp.status.value}

    @app.post("/competitions/{competition_id}/contenders", status_code=201)
    async def enroll(competition_id: str, body: EnrollRequest, request: Request):
        authenticate(request)
        if hotkey_guard is not None:
            from vidaio.services.hotkey_auth import HotkeyAuthError, HotkeyAuthInvalid

            raw_body = await request.body()
            try:
                verified = hotkey_guard.require(  # type: ignore[attr-defined]
                    dict(request.headers),
                    method=request.method,
                    path=request.url.path,
                    body=raw_body,
                    min_alpha_stake=getattr(
                        getattr(hotkey_guard, "_config", None),
                        "min_enroll_alpha_stake",
                        None,
                    ),
                )
                if verified is not None and verified.hotkey != body.hotkey:
                    if getattr(hotkey_guard, "mode", "enforce") == "enforce":
                        raise HotkeyAuthInvalid(
                            f"enrollment is self-signed: signer {verified.hotkey} "
                            f"may not enroll hotkey {body.hotkey}"
                        )
                    import logging

                    logging.getLogger("hotkey-auth").warning(
                        "enrollment signer/hotkey mismatch (log-only mode): "
                        "signer %s enrolled %s",
                        verified.hotkey,
                        body.hotkey,
                    )
            except HotkeyAuthError as exc:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail={"code": exc.code, "message": str(exc)},
                ) from exc
        require_competition(competition_id)
        try:
            contender_id = orch.enroll_contender(
                competition_id,
                hotkey=body.hotkey,
                repo_url=body.repo_url,
                commit_sha=body.commit_sha,
                tree_sha=body.tree_sha,
                stake=body.stake,
                now=orch.now(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "enrollment_refused", "message": str(exc)},
            ) from exc
        return {"contender_id": contender_id}

    @app.post("/competitions/{competition_id}/items", status_code=201)
    async def add_item(
        competition_id: str, body: EvaluationItemRequest, request: Request
    ):
        authenticate(request)
        require_competition(competition_id)
        try:
            item_id = orch.ingest_evaluation_item(
                competition_id,
                input_name=body.input_name,
                reference_name=body.reference_name,
                upscale_factor=body.upscale_factor,
                target_width=body.target_width,
                target_height=body.target_height,
                item_index=body.item_index,
                threshold_commitment=body.threshold_commitment,
                challenge_id=body.challenge_id,
                length_seconds=body.length_seconds,
                now=orch.now(),
            )
        except (ValueError, OSError, SandboxRunnerError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "item_ingest_refused", "message": str(exc)},
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "item_not_added", "message": str(exc)},
            ) from exc
        row = orch.conn.execute(
            "SELECT * FROM evaluation_items WHERE item_id = ?", (item_id,)
        ).fetchone()
        assert row is not None
        return {
            "item_id": item_id,
            "item_index": int(row["item_index"]),
            "input_sha256": str(row["input_sha256"]),
            "reference_sha256": str(row["reference_sha256"]),
            "upscale_factor": row["upscale_factor"],
            "target_width": row["target_width"],
            "target_height": row["target_height"],
            "item_commitment": row["item_commitment"],
        }

    @app.post("/competitions/{competition_id}/anchor")
    async def anchor(competition_id: str, body: AnchorRequest, request: Request):
        authenticate(request)
        require_competition(competition_id)
        # Preserve useful prerequisite errors: a deployment with no chain is 503,
        # and a manifest with no baseline/tree digest is 422, even if its item matrix is
        # also empty. Any request otherwise capable of touching the chain must pass
        # the complete item-matrix gate first.
        manifest = repo.get_manifest(orch.conn, competition_id)
        can_reach_chain = orch.chain is not None and not (
            body.baseline_tree_digest is None and manifest.baseline is None
        )
        if can_reach_chain:
            try:
                items = repo.validate_evaluation_item_bindings(
                    orch.conn, competition_id
                )
                if not items:
                    raise repo.EvaluationItemBindingError(
                        "at least one evaluation item must be ingested before anchoring"
                    )
            except repo.EvaluationItemBindingError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "items_not_ready", "message": str(exc)},
                ) from exc
        try:
            anchored = await orch.anchor_competition(
                competition_id,
                baseline_image_digest=body.baseline_image_digest,
                reward_param_digest=body.reward_param_digest,
                baseline_tree_digest=body.baseline_tree_digest,
                now=orch.now(),
            )
        except NotConfiguredError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "no_chain_adapter", "message": str(exc)},
            ) from exc
        except CommitmentCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "commitment_capacity_unavailable",
                    "message": str(exc),
                },
            ) from exc
        except AnchorClaimRefused as exc:
            # Refused BEFORE the chain: nothing was written by this request.
            raise HTTPException(
                status_code=409, detail={"code": exc.code, "message": str(exc)}
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_commitment", "message": str(exc)},
            ) from exc
        except AnchorError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "anchor_verification_failed", "message": str(exc)},
            ) from exc
        if not anchored.recorded:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "not_recorded",
                    "message": "the lifecycle engine refused the anchor (already "
                    "anchored, or the competition is past SCHEDULED)",
                    "root": anchored.root,
                    "tx_id": anchored.tx_id,
                },
            )
        return {
            "competition_id": competition_id,
            "root": anchored.root,
            "tx_id": anchored.tx_id,
            "payload_hex": anchored.payload.hex(),
            "canonical_json": anchored.canonical_json.decode("utf-8"),
            "baseline_image_digest": anchored.baseline_image_digest,
            "anchor_block": anchored.anchor_block,
            "anchor_block_hash": anchored.anchor_block_hash,
            "finalized_block": anchored.finalized_block,
            "archive_verified": True,
            "write_response_recovered": anchored.write_response_recovered,
        }

    @app.post("/competitions/{competition_id}/anchor/release")
    async def release_anchor_claim(
        competition_id: str, body: ReleaseAnchorClaimRequest, request: Request
    ):
        authenticate(request)
        require_competition(competition_id)
        released = orch.release_anchor_claim(
            competition_id, body.operator, body.reason, orch.now()
        )
        return {"competition_id": competition_id, "released": released}

    @app.post("/competitions/{competition_id}/halt/clear")
    async def clear_halt(competition_id: str, body: ClearHaltRequest, request: Request):
        authenticate(request)
        require_competition(competition_id)
        cleared = orch.clear_halt(
            competition_id,
            body.operator,
            orch.now(),
            reason=body.reason,
        )
        return {"competition_id": competition_id, "cleared": cleared}

    @app.get("/competitions/{competition_id}")
    async def status(competition_id: str, request: Request):
        authenticate(request)
        comp = require_competition(competition_id)
        contenders = repo.list_contenders(orch.conn, competition_id)
        podium = repo.podium(orch.conn, competition_id)
        return {
            "competition_id": comp.competition_id,
            "track": comp.track,
            "status": comp.status.value,
            "halted": pers.is_halted(orch.conn, competition_id),
            "commitment_root": comp.commitment_root,
            "anchor_receipt": pers.latest_verified_anchor_receipt(
                orch.conn, competition_id
            ),
            #: An unresolved anchor claim (payload digest + root + claimed_at) is
            #: what an operator needs to check the chain against before calling
            #: /anchor/release. None when anchoring is not in an ambiguous state.
            "anchor_claim": pers.open_anchor_claim(orch.conn, competition_id),
            "start_time": comp.start_time.isoformat(),
            "enrollment_deadline": comp.enrollment_deadline.isoformat(),
            "finalization_time": comp.finalization_time.isoformat(),
            "end_time": comp.end_time.isoformat(),
            "human_review_deadline": (
                comp.human_review_deadline.isoformat()
                if comp.human_review_deadline
                else None
            ),
            "failure_reason": comp.failure_reason,
            "ranking_semantics": (
                "operational_human_review_only_non_earning; /result is a packet-"
                "economic preview and finalized epoch evidence is authoritative"
            ),
            "contenders": [
                {
                    "contender_id": c.contender_id,
                    "hotkey": c.hotkey,
                    "is_calibration": c.is_calibration,
                    "status": c.status,
                    "eligible": c.eligible,
                    "manual_disqualified": c.manual_disqualified,
                    "final_rank": c.final_rank,
                    "final_score": c.final_score,
                }
                for c in contenders
            ],
            "podium": [
                {
                    "rank": c.final_rank,
                    "contender_id": c.contender_id,
                    "hotkey": c.hotkey,
                    "final_score": c.final_score,
                }
                for c in podium
            ],
        }

    @app.post("/competitions/{competition_id}/review", status_code=201)
    async def review(competition_id: str, body: ReviewRequest, request: Request):
        authenticate(request)
        require_competition(competition_id)
        try:
            review_id = orch.submit_review(
                competition_id,
                contender_id=body.contender_id,
                action=body.action,
                reviewer=body.reviewer,
                reason=body.reason,
                detail=body.detail,
                supersedes_review_id=body.supersedes_review_id,
                now=orch.now(),
            )
        except (ReviewError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail={"code": "review_refused", "message": str(exc)}
            ) from exc
        return {"review_id": review_id}

    @app.get("/competitions/{competition_id}/result")
    async def result(competition_id: str, request: Request):
        authenticate(request)
        require_competition(competition_id)
        try:
            census_by_hotkey = None
            if orch.chain is not None:
                try:
                    # A production Bittensor adapter starts without a cached
                    # metagraph.  ``neurons()`` deliberately refuses to turn that
                    # UNKNOWN state into an empty census, so the preview endpoint
                    # must refresh before its first read.  Keep the refresh and read
                    # in one worker call: both are synchronous RPC/cache operations
                    # and neither belongs on the control API event loop.
                    def refreshed_neurons():
                        orch.chain.refresh()
                        return orch.chain.neurons()

                    neurons = await asyncio.to_thread(refreshed_neurons)
                except Exception as exc:
                    raise CompetitionEvidenceError(
                        f"current chain census is unavailable: {exc}"
                    ) from exc
                from vidaio.epoch.log import MinerCensusEntry

                census_by_hotkey = {
                    neuron.hotkey: MinerCensusEntry(
                        uid=neuron.uid,
                        hotkey=neuron.hotkey,
                        coldkey=neuron.coldkey,
                        ip=neuron.ip,
                    )
                    for neuron in neurons
                }
            # SQLite remains on the owning event-loop thread. Only the synchronous
            # network RPC above moves to a worker thread.
            built = orch.build_result(competition_id, census_by_hotkey=census_by_hotkey)
        except ResultNotReady as exc:
            raise HTTPException(
                status_code=409, detail={"code": "not_completed", "message": str(exc)}
            ) from exc
        except CompetitionEvidenceError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "unauditable_result", "message": str(exc)},
            ) from exc
        payload = result_payload(built)
        payload["source"] = "packet_mean.current_census_preview.v1"
        payload["identity_snapshot"] = "current_unpinned_chain_head"
        payload["authoritative_emitted_result"] = False
        return payload

    return app
