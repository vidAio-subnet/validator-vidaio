"""Public, token-free competition surface: discovery + self-signed enrollment.

The operator control API (``control.py``) creates competitions and anchors
commitments; it is loopback-only and bearer-gated. THIS app is what miners reach.
It exposes exactly three things:

* ``GET  /v1/competitions`` and ``GET /v1/competitions/{id}`` — the anchored manifest
  (times, resources, result rules, stake floor) and the enrolled hotkeys. Nothing here
  is secret: the manifest digest is on chain before enrollment opens.
* ``POST /v1/competitions/{id}/enroll`` — a miner enrolls ITSELF. The request carries
  the Scheme-A signed headers (``vidaio.services.hotkey_auth``); the enrolled hotkey is
  the verified signer, never a body field. The hotkey must be registered on the
  subnet, and its alpha stake — READ FROM THE CHAIN REGISTRY, never supplied by the
  caller — must clear ``max(manifest.minimum_alpha_stake,
  hotkey_auth.min_enroll_alpha_stake)``.

Enrollment is refused (503) unless a registry-backed guard in ``enforce`` mode is
injected: a log-only guard verifies nothing, and an open enrollment endpoint would let
anyone burn sandbox builds.  A small in-process limiter bounds abuse per client
address and per hotkey; the deployment's reverse proxy should rate-limit as well.

CONCURRENCY: same single event loop / single thread as the control app — handlers run
between orchestrator ticks and hold no transaction across an await.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from vidaio.competition import repository as repo
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
        "enrollment_open": comp.status is Phase.ENROLLING,
        "manifest": manifest.model_dump(mode="json", exclude_none=True),
        "enrolled": [{"hotkey": c.hotkey, "status": c.status} for c in contenders],
    }


def create_public_app(
    orch: "Orchestrator",
    *,
    hotkey_guard: object | None,
    min_enroll_alpha_stake: float | None,
    allowed_repo_hosts: tuple[str, ...],
    enroll_limit_per_hour: int = 12,
    read_limit_per_minute: int = 120,
    trust_forwarded_for: bool = False,
) -> FastAPI:
    app = FastAPI(title="vidaio competitions", docs_url=None, redoc_url=None)
    enroll_by_address = SlidingWindowLimiter(enroll_limit_per_hour, 3600.0)
    enroll_by_hotkey = SlidingWindowLimiter(enroll_limit_per_hour, 3600.0)
    reads = SlidingWindowLimiter(read_limit_per_minute, 60.0)

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
        return _public_competition(orch, require_competition(competition_id))

    @app.post("/v1/competitions/{competition_id}/enroll", status_code=201)
    async def enroll(competition_id: str, request: Request) -> dict[str, Any]:
        address = _client_address(request, trust_forwarded_for=trust_forwarded_for)
        limited(enroll_by_address, address)
        if hotkey_guard is None or getattr(hotkey_guard, "mode", "off") != "enforce":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "enrollment_unavailable",
                    "message": "public enrollment requires registry-backed hotkey "
                    "authentication in enforce mode",
                },
            )
        raw_body = await request.body()
        if len(raw_body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="request body too large")
        comp = require_competition(competition_id)
        manifest = repo.get_manifest(orch.conn, competition_id)
        floor = max(
            float(manifest.minimum_alpha_stake), float(min_enroll_alpha_stake or 0.0)
        )
        from vidaio.services.hotkey_auth import HotkeyAuthError

        try:
            verified = hotkey_guard.require(  # type: ignore[attr-defined]
                dict(request.headers),
                method=request.method,
                path=request.url.path,
                body=raw_body,
                min_alpha_stake=floor,
            )
        except HotkeyAuthError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        if verified is None:  # defensive: enforce mode always returns a caller
            raise HTTPException(status_code=503, detail="hotkey verification unavailable")
        limited(enroll_by_hotkey, verified.hotkey)
        try:
            body = PublicEnrollRequest.model_validate_json(raw_body)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_enrollment", "message": str(exc)},
            ) from exc
        host = _repo_host(body.repo_url)
        if host is None or host not in allowed_repo_hosts:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "repo_host_not_allowed",
                    "message": "repo_url must be https://<host>/... on one of "
                    f"{list(allowed_repo_hosts)}",
                },
            )
        if comp.status is not Phase.ENROLLING:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "enrollment_closed",
                    "message": f"competition is {comp.status.value}, not ENROLLING",
                },
            )
        try:
            contender_id = orch.enroll_contender(
                competition_id,
                hotkey=verified.hotkey,
                repo_url=body.repo_url,
                commit_sha=body.commit_sha,
                tree_sha=body.tree_sha,
                stake=float(verified.alpha_stake),
                now=orch.now(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "enrollment_refused", "message": str(exc)},
            ) from exc
        return {
            "competition_id": competition_id,
            "contender_id": contender_id,
            "hotkey": verified.hotkey,
            "alpha_stake": float(verified.alpha_stake),
            "required_alpha_stake": floor,
        }

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


__all__ = ["PublicEnrollRequest", "SlidingWindowLimiter", "create_public_app"]
