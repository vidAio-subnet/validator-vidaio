"""Public, token-free competition surface: discovery + self-signed entries.

The operator control API (``control.py``) creates competitions and anchors
commitments; it is loopback-only and bearer-gated. THIS app is what miners reach:

* ``GET  /v1/competitions`` and ``GET /v1/competitions/{id}`` — the anchored manifest
  (times, resources, result rules, stake floor), whether the pipeline is paused,
  what the pre-deadline checks enforce (``entry_checks``) and the enrolled hotkeys,
  each with its ``contender_id``, ``updated_at``, ``entry_commitment`` (see
  ``orchestrator/entries.py``) and a code-only ``checks`` summary. Nothing here is
  secret: the manifest digest is on chain before enrollment opens. Once a
  competition is COMPLETED, ``GET /v1/competitions/{id}`` also carries the
  ``results`` table (``orchestrator/public_results.py``); never earlier.
* ``POST /v1/competitions/{id}/enroll`` — a miner enrolls ITSELF, body
  ``{repo_url, commit_sha, tree_sha, replace=false}``. The request carries the
  Scheme-A signed headers (``vidaio.services.hotkey_auth``); the enrolled hotkey is
  the verified signer, never a body field. The hotkey must be registered on the
  subnet, and its alpha stake — READ FROM THE CHAIN REGISTRY, never supplied by the
  caller — must clear ``max(manifest.minimum_alpha_stake,
  hotkey_auth.min_enroll_alpha_stake)``. A hotkey holds one entry per competition:
  201 for a new entry; with ``replace=true`` an ENROLLED entry is changed IN PLACE
  before the deadline (200, ``replaced``/``unchanged``); without it 409
  ``already_enrolled``. A withdrawn or rejected entry is final (409
  ``entry_not_replaceable``).
* ``POST /v1/competitions/{id}/withdraw`` — body ``{"confirm": true}``: the signer's
  own ENROLLED entry becomes REJECTED ("withdrawn by the miner") before the deadline.
  Final for that hotkey in that competition. Registered hotkey only (no stake floor).
* ``GET /v1/competitions/{id}/entry`` — signed (method GET, this path, empty body):
  the signer's OWN entry with the full latest reason and the latest pre-deadline
  check and trial-build results. 404 ``not_enrolled``. Never another hotkey's.

Signed endpoints are refused (503) unless a registry-backed guard in ``enforce`` mode
is injected: a log-only guard verifies nothing, and an open enrollment endpoint would
let anyone burn sandbox builds. A small in-process limiter bounds abuse per client
address (before verification) and per hotkey (after it); the deployment's reverse
proxy should rate-limit as well.

CONCURRENCY: same single event loop / single thread as the control app — handlers run
between orchestrator ticks and hold no transaction across an await.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from vidaio.competition import repository as repo
from vidaio.competition.orchestrator import entries
from vidaio.competition.orchestrator import persistence as pers
from vidaio.competition.states import Phase

if TYPE_CHECKING:
    from vidaio.competition.orchestrator.service import Orchestrator

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_MAX_BODY_BYTES = 8 * 1024


class PublicEnrollRequest(BaseModel):
    """What a miner submits. Identity and stake are NOT accepted from the body."""

    model_config = ConfigDict(extra="forbid")

    repo_url: str = Field(min_length=1, max_length=512)
    commit_sha: str = Field(pattern=_SHA1.pattern)
    tree_sha: str = Field(pattern=_SHA1.pattern)
    #: Change this hotkey's ENROLLED entry in place (before the deadline).
    replace: bool = False


class PublicWithdrawRequest(BaseModel):
    """Withdrawing is final for the hotkey in this competition, so it is explicit."""

    model_config = ConfigDict(extra="forbid")

    confirm: Literal[True]


class SlidingWindowLimiter:
    """At most ``limit`` events per ``window_seconds`` per key (bounded memory)."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 4096,
    ) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("limiter needs limit >= 1 and a positive window")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self._clock()
        if len(self._events) >= self._max_keys and key not in self._events:
            stale = [
                k for k, q in self._events.items() if not q or now - q[-1] > self._window
            ]
            for k in stale:
                del self._events[k]
            if len(self._events) >= self._max_keys:
                return False  # fail closed under a key-flood
        events = self._events[key]
        while events and now - events[0] > self._window:
            events.popleft()
        if len(events) >= self._limit:
            return False
        events.append(now)
        return True


def _client_address(request: Request, *, trust_forwarded_for: bool) -> str:
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _public_competition(orch: "Orchestrator", comp: repo.CompetitionRecord) -> dict[str, Any]:
    manifest = repo.get_manifest(orch.conn, comp.competition_id)
    contenders = [
        c for c in repo.list_contenders(orch.conn, comp.competition_id)
        if not c.is_calibration
    ]
    reasons = _contender_reasons(orch, comp.competition_id)
    checks = entries.entry_checks(orch.conn, comp.competition_id, contenders)
    trial_enabled = bool(getattr(orch.cfg, "enrollment_trial_build_enabled", False))
    return {
        "competition_id": comp.competition_id,
        "track": comp.track,
        "status": comp.status.value,
        "manifest_digest": comp.manifest_digest,
        "commitment_root": comp.commitment_root,
        "start_time": comp.start_time.isoformat(),
        "enrollment_deadline": comp.enrollment_deadline.isoformat(),
        "finalization_time": comp.finalization_time.isoformat(),
        "end_time": comp.end_time.isoformat(),
        # The phase stays ENROLLING until finalization_time; enrollment itself closes at
        # the deadline (repository.enroll refuses later requests), so say so here too.
        "enrollment_open": comp.status is Phase.ENROLLING
        and orch.now() <= comp.enrollment_deadline,
        #: The pipeline is halted or paused by an operator (the reason stays in the
        #: operator's log; recovery actions are listed under ``amendments``).
        "paused": pers.is_halted(orch.conn, comp.competition_id),
        #: What the pre-deadline fetch check enforces for this competition.
        "entry_checks": {
            "licenses_required": entries.licenses_required(manifest),
            "license_files": list(entries.LICENSE_FILES),
            "licenses_manifest": entries.LICENSES_MANIFEST,
            "max_worktree_bytes": int(orch.cfg.git_checkout_max_bytes),
            "trial_build": trial_enabled,
        },
        "manifest": manifest.model_dump(mode="json", exclude_none=True),
        "enrolled": [
            {
                "contender_id": c.contender_id,
                "hotkey": c.hotkey,
                "status": c.status,
                "updated_at": c.updated_at,
                "entry_commitment": entries.entry_commitment(
                    comp.competition_id,
                    c.hotkey or "",
                    c.repo_url,
                    c.commit_sha,
                    c.tree_sha,
                ),
                "checks": entries.checks_summary(
                    checks.get(c.contender_id), trial_enabled=trial_enabled
                ),
                **(
                    {"reason": reasons[c.contender_id]}
                    if c.contender_id in reasons
                    else {}
                ),
            }
            for c in contenders
        ],
        #: Every recovery action taken on this competition (paused, resumed,
        #: deadlines moved later, sandbox envelope raised, matrix rerun, ...).
        "amendments": [
            {
                "type": entry["type"],
                "at": entry["at"],
                "details": {
                    k: v for k, v in entry["details"].items() if k != "operator"
                },
            }
            for entry in pers.operator_amendments(orch.conn, comp.competition_id)
        ],
        "sandbox_resources": orch.effective_sandbox_resources(comp.competition_id),
    }


def _contender_reasons(
    orch: "Orchestrator", competition_id: str, *, full: bool = False
) -> dict[int, str]:
    """Why a contender stopped (REJECTED / BUILD_FAILED / withdrawn), latest event wins.

    Public callers get the first line only (a build log can quote the private
    tree); ``full=True`` is for the entrant's own view and the operator.
    """
    reasons: dict[int, str] = {}
    rows = orch.conn.execute(
        "SELECT event_type, payload_json FROM events WHERE competition_id = ?"
        " AND event_type IN ('contender_validated', 'contender_build_failed',"
        " 'contender_rejected_by_operator', 'contender_withdrawn') ORDER BY event_id",
        (competition_id,),
    ).fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"] or "{}")
        cid = payload.get("contender_id")
        reason = payload.get("reason")
        if isinstance(cid, int) and isinstance(reason, str) and reason:
            if row["event_type"] == "contender_validated" and payload.get("status") != "REJECTED":
                continue
            if full:
                reasons[cid] = reason.strip()
                continue
            # public: the first line only (a build log can quote the private tree);
            # the full text stays in the operator's control API and event log
            first = reason.strip().splitlines()[0] if reason.strip() else ""
            reasons[cid] = first[:300]
    return reasons


def _entry_view(
    orch: "Orchestrator", comp: repo.CompetitionRecord, record: repo.ContenderRecord
) -> dict[str, Any]:
    """The entrant's own full view of its entry (signed GET /entry only)."""
    checks = entries.entry_checks(orch.conn, comp.competition_id, [record]).get(
        record.contender_id
    )
    trial_enabled = bool(getattr(orch.cfg, "enrollment_trial_build_enabled", False))
    precheck = None
    if checks is not None and checks.precheck is not None:
        precheck = {
            key: checks.precheck.get(key) for key in ("at", "ok", "problems", "detail")
        }
    trial = None
    if checks is not None and checks.trial is not None:
        outcome = checks.trial.get("outcome")
        trial = {
            "at": checks.trial.get("at"),
            "outcome": outcome,
            "ok": outcome == entries.TRIAL_OK,
            "error": checks.trial.get("error")
            if outcome == entries.TRIAL_FAILED
            else (
                "the trial build could not run on our side; it is retried automatically"
                if outcome == entries.TRIAL_UNAVAILABLE
                else None
            ),
        }
    return {
        "competition_id": comp.competition_id,
        "contender_id": record.contender_id,
        "hotkey": record.hotkey,
        "status": record.status,
        "repo_url": record.repo_url,
        "commit_sha": record.commit_sha,
        "tree_sha": record.tree_sha,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "replacements": 0 if checks is None else checks.replacements,
        "entry_commitment": entries.entry_commitment(
            comp.competition_id,
            record.hotkey or "",
            record.repo_url,
            record.commit_sha,
            record.tree_sha,
        ),
        "enrollment_open": comp.status is Phase.ENROLLING
        and orch.now() <= comp.enrollment_deadline,
        "enrollment_deadline": comp.enrollment_deadline.isoformat(),
        "reason": _contender_reasons(orch, comp.competition_id, full=True).get(
            record.contender_id
        ),
        "checks": entries.checks_summary(checks, trial_enabled=trial_enabled),
        "precheck": precheck,
        "trial_build": trial,
    }


def create_public_app(
    orch: "Orchestrator",
    *,
    hotkey_guard: object | None,
    min_enroll_alpha_stake: float | None,
    allowed_repo_hosts: tuple[str, ...],
    enroll_limit_per_hour: int = 12,
    read_limit_per_minute: int = 120,
    entry_limit_per_hour: int = 120,
    trust_forwarded_for: bool = False,
) -> FastAPI:
    app = FastAPI(title="vidaio competitions", docs_url=None, redoc_url=None)
    # enroll (incl. replace) and withdraw share one budget per address and hotkey
    enroll_by_address = SlidingWindowLimiter(enroll_limit_per_hour, 3600.0)
    enroll_by_hotkey = SlidingWindowLimiter(enroll_limit_per_hour, 3600.0)
    entry_by_address = SlidingWindowLimiter(entry_limit_per_hour, 3600.0)
    entry_by_hotkey = SlidingWindowLimiter(entry_limit_per_hour, 3600.0)
    reads = SlidingWindowLimiter(read_limit_per_minute, 60.0)
    #: competition_id -> (monotonic time, results payload); COMPLETED is immutable
    results_cache: dict[str, tuple[float, dict[str, Any]]] = {}
    results_lock = asyncio.Lock()

    def limited(limiter: SlidingWindowLimiter, key: str) -> None:
        if not limiter.allow(key):
            raise HTTPException(
                status_code=429,
                detail={"code": "rate_limited", "message": "too many requests"},
            )

    def require_competition(competition_id: str) -> repo.CompetitionRecord:
        comp = repo.get_competition(orch.conn, competition_id)
        if comp is None:
            raise HTTPException(status_code=404, detail="unknown competition")
        return comp

    def refuse(http_status: int, code: str, message: str, **extra: Any) -> HTTPException:
        return HTTPException(
            status_code=http_status, detail={"code": code, "message": message, **extra}
        )

    async def verified_signer(
        request: Request,
        address_limiter: SlidingWindowLimiter,
        hotkey_limiter: SlidingWindowLimiter,
        *,
        floor: Callable[[repo.CompetitionRecord], float | None],
        competition_id: str,
    ) -> tuple[Any, bytes, repo.CompetitionRecord, float | None]:
        """Shared Scheme-A gate: limiter, enforce-mode guard, body cap, signature,
        registration (+ stake floor), then the per-hotkey limiter."""
        address = _client_address(request, trust_forwarded_for=trust_forwarded_for)
        limited(address_limiter, address)
        if hotkey_guard is None or getattr(hotkey_guard, "mode", "off") != "enforce":
            raise refuse(
                503,
                "enrollment_unavailable",
                "public enrollment requires registry-backed hotkey "
                "authentication in enforce mode",
            )
        raw_body = await request.body()
        if len(raw_body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        comp = require_competition(competition_id)
        min_stake = floor(comp)
        from vidaio.services.hotkey_auth import HotkeyAuthError

        try:
            verified = hotkey_guard.require(  # type: ignore[attr-defined]
                dict(request.headers),
                method=request.method,
                path=request.url.path,
                body=raw_body,
                min_alpha_stake=min_stake,
            )
        except HotkeyAuthError as exc:
            raise refuse(exc.status_code, exc.code, str(exc)) from exc
        if verified is None:  # defensive: enforce mode always returns a caller
            raise HTTPException(status_code=503, detail="hotkey verification unavailable")
        limited(hotkey_limiter, verified.hotkey)
        return verified, raw_body, comp, min_stake

    def enroll_floor(comp: repo.CompetitionRecord) -> float:
        manifest = repo.get_manifest(orch.conn, comp.competition_id)
        return max(
            float(manifest.minimum_alpha_stake), float(min_enroll_alpha_stake or 0.0)
        )

    def require_open(comp: repo.CompetitionRecord) -> None:
        if comp.status is not Phase.ENROLLING:
            raise refuse(
                409,
                "enrollment_closed",
                f"competition is {comp.status.value}, not ENROLLING",
            )
        if orch.now() > comp.enrollment_deadline:
            raise refuse(409, "enrollment_closed", "the enrollment deadline has passed")

    def entry_refusal(exc: Exception) -> HTTPException:
        if isinstance(exc, repo.AlreadyEnrolledError):
            return refuse(
                409,
                "already_enrolled",
                str(exc),
                contender_id=exc.contender_id,
                commit_sha=exc.commit_sha,
            )
        if isinstance(exc, repo.EntryNotReplaceableError):
            return refuse(
                409,
                "entry_not_replaceable",
                str(exc),
                contender_id=exc.contender_id,
                status=exc.status,
            )
        if isinstance(exc, repo.NotEnrolledError):
            return refuse(404, "not_enrolled", str(exc))
        if isinstance(exc, repo.EnrollmentClosedError):
            return refuse(409, "enrollment_closed", str(exc))
        return refuse(409, "enrollment_refused", str(exc))

    async def results_for(comp: repo.CompetitionRecord) -> dict[str, Any]:
        """The COMPLETED results table, computed once per competition. The packet
        reads (scorer gate failures only) run off the event loop."""
        from vidaio.competition.orchestrator import public_results

        cached = results_cache.get(comp.competition_id)
        if cached is not None and (
            "results" in cached[1] or time.monotonic() - cached[0] < 300.0
        ):
            return cached[1]
        async with results_lock:
            cached = results_cache.get(comp.competition_id)
            if cached is not None and "results" in cached[1]:
                return cached[1]
            try:
                plan = public_results.collect(orch.conn, comp.competition_id)
                codes = await asyncio.to_thread(
                    public_results.read_failure_codes, orch.store, plan.packets_to_read
                )
                payload = public_results.assemble(plan, codes)
            except Exception as exc:  # noqa: BLE001 - the view must still render
                payload = {"results_unavailable": f"{type(exc).__name__}: {exc}"[:300]}
            results_cache[comp.competition_id] = (time.monotonic(), payload)
            return payload

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"service": "competitions-public", "status": "ok"}

    @app.get("/v1/competitions")
    async def list_competitions(request: Request) -> dict[str, Any]:
        limited(reads, _client_address(request, trust_forwarded_for=trust_forwarded_for))
        records = repo.list_competitions_in(orch.conn, tuple(Phase))
        records.sort(key=lambda c: c.start_time, reverse=True)
        return {
            "min_enroll_alpha_stake": min_enroll_alpha_stake,
            "allowed_repo_hosts": list(allowed_repo_hosts),
            "competitions": [_public_competition(orch, c) for c in records[:50]],
        }

    @app.get("/v1/competitions/{competition_id}")
    async def get_competition(competition_id: str, request: Request) -> dict[str, Any]:
        limited(reads, _client_address(request, trust_forwarded_for=trust_forwarded_for))
        comp = require_competition(competition_id)
        view = _public_competition(orch, comp)
        if comp.status is Phase.COMPLETED:
            # hidden inputs are public after the result, and so are their scores
            view.update(await results_for(comp))
        return view

    @app.post("/v1/competitions/{competition_id}/enroll", status_code=201)
    async def enroll(
        competition_id: str, request: Request, response: Response
    ) -> dict[str, Any]:
        verified, raw_body, comp, floor = await verified_signer(
            request,
            enroll_by_address,
            enroll_by_hotkey,
            floor=enroll_floor,
            competition_id=competition_id,
        )
        try:
            body = PublicEnrollRequest.model_validate_json(raw_body)
        except Exception as exc:
            raise refuse(422, "invalid_enrollment", str(exc)) from exc
        host = _repo_host(body.repo_url)
        if host is None or host not in allowed_repo_hosts:
            raise refuse(
                422,
                "repo_host_not_allowed",
                "repo_url must be https://<host>/... on one of "
                f"{list(allowed_repo_hosts)}",
            )
        require_open(comp)
        try:
            outcome = orch.submit_entry(
                competition_id,
                hotkey=verified.hotkey,
                repo_url=body.repo_url,
                commit_sha=body.commit_sha,
                tree_sha=body.tree_sha,
                stake=float(verified.alpha_stake),
                now=orch.now(),
                replace=body.replace,
            )
        except Exception as exc:
            raise entry_refusal(exc) from exc
        result: dict[str, Any] = {
            "competition_id": competition_id,
            "contender_id": outcome.contender_id,
            "hotkey": verified.hotkey,
            "alpha_stake": float(verified.alpha_stake),
            "required_alpha_stake": floor,
            "repo_url": body.repo_url,
            "commit_sha": body.commit_sha,
            "tree_sha": body.tree_sha,
            "entry_commitment": entries.entry_commitment(
                competition_id,
                verified.hotkey,
                body.repo_url,
                body.commit_sha,
                body.tree_sha,
            ),
            "replaced": outcome.action == "replaced",
        }
        if outcome.action != "enrolled":
            response.status_code = 200
            result["unchanged"] = outcome.action == "unchanged"
            result["previous_commit_sha"] = outcome.previous_commit_sha
        return result

    @app.post("/v1/competitions/{competition_id}/withdraw")
    async def withdraw(competition_id: str, request: Request) -> dict[str, Any]:
        verified, raw_body, comp, _floor = await verified_signer(
            request,
            enroll_by_address,
            enroll_by_hotkey,
            floor=lambda _comp: None,  # registered hotkey only
            competition_id=competition_id,
        )
        try:
            PublicWithdrawRequest.model_validate_json(raw_body)
        except Exception as exc:
            raise refuse(
                422, "invalid_withdrawal", 'the body must be {"confirm": true}'
            ) from exc
        require_open(comp)
        try:
            contender_id = orch.withdraw_entry(
                competition_id, hotkey=verified.hotkey, now=orch.now()
            )
        except Exception as exc:
            raise entry_refusal(exc) from exc
        return {
            "competition_id": competition_id,
            "contender_id": contender_id,
            "hotkey": verified.hotkey,
            "status": "REJECTED",
            "withdrawn": True,
            "final": True,
            "message": "entry withdrawn. This is final: this hotkey cannot enroll "
            "again in this competition.",
        }

    @app.get("/v1/competitions/{competition_id}/entry")
    async def entry(competition_id: str, request: Request) -> dict[str, Any]:
        verified, _raw, comp, _floor = await verified_signer(
            request,
            entry_by_address,
            entry_by_hotkey,
            floor=lambda _comp: None,  # registered hotkey only
            competition_id=competition_id,
        )
        record = repo.find_contender_by_hotkey(
            orch.conn, competition_id, verified.hotkey
        )
        if record is None or record.is_calibration:
            raise refuse(
                404, "not_enrolled", "this hotkey is not enrolled in this competition"
            )
        return _entry_view(orch, comp, record)

    return app


def _repo_host(repo_url: str) -> str | None:
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(repo_url)
    except ValueError:
        return None
    if parts.scheme != "https" or parts.username or parts.password:
        return None
    if parts.query or parts.fragment or parts.port not in (None, 443):
        return None
    return parts.hostname


__all__ = [
    "PublicEnrollRequest",
    "PublicWithdrawRequest",
    "SlidingWindowLimiter",
    "create_public_app",
]
