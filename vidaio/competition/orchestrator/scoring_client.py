"""HTTP CompetitionScoringClient — calls the trusted remote scoring worker.

Spec §05: whoever controls the scoring endpoint controls the numbers, so the
endpoint must be validator-operated.  A self-consistent response digest is not
enough: before returning the worker's verbatim ItemScore bytes, this client binds
their challenge/item/miner/track/output/scorer identity to the exact request and
their COMPLETE backend map to the independently validated ``GET /healthz``
runtime commitment.  Only then may repository.record_item_score derive economic
state from those bytes. Local-first wire contract: artifacts are exchanged as
absolute paths on the shared filesystem plus digests (vidaio.services.protocol).
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import httpx
from pydantic import ValidationError

from vidaio.competition import repository as repo
from vidaio.competition.interfaces import BatchItem, BatchOutput, ScorePacket
from vidaio.scoring.result import ItemScore
from vidaio.services.protocol import (
    SCORER_RUNTIME_BACKEND_KEY,
    ScoreRequest,
    ScoreResponse,
    ScorerIdentityUnavailable,
    ScorerRuntimeContract,
    fetch_scorer_runtime_contract,
    require_matching_scorer_runtime_contract,
)


class ScoringClientError(Exception):
    """Transport/consistency failure talking to the scoring worker.

    ``status_code`` carries the worker's HTTP status when there was one, and
    ``error_code`` / ``error_field`` carry the worker's TYPED rejection
    (``{"detail": {"error": ..., "field": ...}}`` — see
    vidaio.scoring_worker.inputs.ScoreRejected) when the body had one.

    The status alone is not enough to apportion blame (review service-review #14,
    round 2). The worker answers 422 both for "the contender's output is not a
    readable regular file" and for "the reference/miner input YOU named is missing"
    or "YOUR params are invalid". Zeroing a contender for our own bad request would
    silently corrupt the competition, so ``classify_failure`` reads the typed code
    plus the ``field`` it names: only ``field == "output"`` is contender-
    attributable. Everything else — and any 422 we cannot type — is INFRA.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        error_field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        #: the worker's machine-readable rejection reason, e.g. "not_a_regular_file"
        self.error_code = error_code
        #: which of the request's three artifacts it names: reference | miner_input | output
        self.error_field = error_field


def _typed_rejection(response: httpx.Response) -> tuple[str | None, str | None]:
    """(error_code, error_field) out of the worker's typed error body, if any.

    Never raises: a worker that answered with something other than the documented
    ``{"detail": {"error": ...}}`` shape simply yields (None, None), which
    classify_failure treats as an untyped — therefore INFRA — failure.
    """
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - non-JSON error bodies are expected in the wild
        return None, None
    detail = body.get("detail") if isinstance(body, dict) else None
    if not isinstance(detail, dict):
        return None, None
    code = detail.get("error")
    field = detail.get("field")
    return (
        str(code) if isinstance(code, str) and code else None,
        str(field) if isinstance(field, str) and field else None,
    )


class HttpScoringClient:
    """CompetitionScoringClient over HTTP (POST {base_url}/score).

    Reads item/contender identity from the shared competition DB (read-only) and
    resolves artifact paths in the orchestrator's content-addressed pools. For the
    compression track the reference and the miner input are the same sealed
    original. Verifies the worker's own packet digest before handing the packet
    back — a worker whose bytes don't hash to its claimed digest is broken.

    NOTE: the orchestrator invokes score_item via asyncio.to_thread, so `conn`
    must be its OWN connection opened with check_same_thread=False (calls are
    serialized; never share the orchestrator's main connection).
    """

    def __init__(
        self,
        base_url: str,
        conn: sqlite3.Connection,
        *,
        inputs_dir: str | Path,
        outputs_dir: str | Path,
        timeout_seconds: float = 300.0,
        transport: httpx.BaseTransport | None = None,
        expected_runtime_contract: ScorerRuntimeContract | None = None,
        allow_noncanonical_runtime_for_report_or_tests: bool = False,
    ) -> None:
        if expected_runtime_contract is None and not (
            allow_noncanonical_runtime_for_report_or_tests
        ):
            raise ValueError(
                "HttpScoringClient requires a locally derived canonical payout-"
                "runtime contract; only explicit report/test callers may opt out"
            )
        if expected_runtime_contract is not None and (
            allow_noncanonical_runtime_for_report_or_tests
        ):
            raise ValueError(
                "expected_runtime_contract and the report/test opt-out are mutually "
                "exclusive"
            )
        self._conn = conn
        self._inputs_dir = Path(inputs_dir)
        self._outputs_dir = Path(outputs_dir)
        self._base_url = base_url.rstrip("/")
        self._expected_runtime_contract = expected_runtime_contract
        self._allow_noncanonical_runtime = (
            allow_noncanonical_runtime_for_report_or_tests
        )
        self._runtime_contract: ScorerRuntimeContract | None = None
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def scorer_identity(self) -> str:
        """The worker's effective scorer identity from its GET /healthz.

        The orchestrator compares this to the PERSISTED manifest's
        `scoring_version` at competition start and again before SCORING, so a
        scorer disagreement is an explicit early INFRA halt instead of a 409
        surprise halfway through measuring a competition (services.protocol,
        THE SCORER-IDENTITY CONTRACT). Raises ScorerIdentityUnavailable.
        """
        # Never retain a prior worker's runtime contract across a failed refresh.
        # The orchestrator deliberately defers an unreachable health probe, so a
        # stale cache here would otherwise let the subsequent score path proceed
        # under yesterday's runtime expectations.
        self._runtime_contract = None
        contract = self._fetch_runtime_contract()
        self._runtime_contract = contract
        return contract.scorer_version

    def _fetch_runtime_contract(self) -> ScorerRuntimeContract:
        contract = fetch_scorer_runtime_contract(
            self._base_url,
            client=self._client,
            require_canonical=not self._allow_noncanonical_runtime,
        )
        expected = self._expected_runtime_contract
        if expected is not None:
            require_matching_scorer_runtime_contract(
                contract,
                expected,
                context="competition scoring worker healthz",
            )
        return contract

    def expected_backend_versions(self) -> dict[str, str]:
        """The health-bound exact backend map for defense-in-depth persistence.

        ``score_item`` guarantees this contract exists before calling ``/score``.
        The orchestrator re-reads it immediately before artifact/DB persistence so
        a later refactor cannot accidentally drop the HTTP-bound runtime check.
        """

        if self._runtime_contract is None:
            raise ScoringClientError(
                "scoring worker runtime contract has not been established"
            )
        return dict(self._runtime_contract.backend_versions)

    def _contract_for(self, expected_scorer: str) -> ScorerRuntimeContract:
        contract = self._runtime_contract
        if contract is None:
            try:
                contract = self._fetch_runtime_contract()
            except ScorerIdentityUnavailable as exc:
                raise ScoringClientError(
                    f"scoring worker runtime identity is unavailable: {exc}"
                ) from exc
            self._runtime_contract = contract
        if contract.scorer_version != expected_scorer:
            raise ScoringClientError(
                "scoring worker health identity does not match the anchored "
                f"competition manifest: health={contract.scorer_version!r}, "
                f"manifest={expected_scorer!r}"
            )
        return contract

    @staticmethod
    def _binding_problem(packet: ItemScore, request: ScoreRequest) -> str | None:
        expected: dict[str, object] = {
            "track": request.track,
            "challenge_id": request.challenge_id,
            "item_id": request.item_id,
            "miner_hotkey": request.miner_hotkey,
            "content_digest": request.output_digest,
            "scorer_version": request.scorer_version,
        }
        for field, wanted in expected.items():
            if getattr(packet, field, None) != wanted:
                return (
                    f"{field} differs from the committed scoring request "
                    f"(packet={getattr(packet, field, None)!r}, expected={wanted!r})"
                )
        return None

    def score_item(
        self,
        competition_id: str,
        contender_id: int,
        item: BatchItem,
        output: BatchOutput,
    ) -> ScorePacket:
        item_row = self._conn.execute(
            "SELECT * FROM evaluation_items WHERE item_id = ?", (item.item_id,)
        ).fetchone()
        if item_row is None or item_row["competition_id"] != competition_id:
            raise ScoringClientError(
                f"evaluation item {item.item_id} not part of {competition_id}"
            )
        contender = repo.get_contender(self._conn, contender_id)
        if contender is None or contender.competition_id != competition_id:
            raise ScoringClientError(
                f"contender {contender_id} not part of {competition_id}"
            )
        manifest = repo.get_manifest(self._conn, competition_id)
        # Re-derive every row's reference/input/factor binding from the anchored
        # manifest before trusted scoring.  DB metadata alone is never authoritative.
        repo.validate_evaluation_item_bindings(self._conn, competition_id)
        input_path = self._inputs_dir / item.input_sha256
        reference_sha256 = str(item_row["reference_sha256"])
        reference_path = self._inputs_dir / reference_sha256
        output_path = self._outputs_dir / output.output_sha256
        params: dict[str, float | int]
        if manifest.track == "upscaling":
            params = {
                "upscale_factor": int(item_row["upscale_factor"]),
                "target_width": int(item_row["target_width"]),
                "target_height": int(item_row["target_height"]),
            }
        else:
            params = {"vmaf_threshold": manifest.vmaf_threshold}
        request = ScoreRequest(
            track=manifest.track,
            challenge_id=item_row["challenge_id"],
            item_id=item_row["scoring_item_id"],
            miner_hotkey=contender.hotkey,
            reference_path=str(reference_path),
            reference_digest=reference_sha256,
            miner_input_path=str(input_path),
            miner_input_digest=item.input_sha256,
            output_path=str(output_path),
            output_digest=output.output_sha256,
            params=params,
            scorer_version=manifest.scoring_version,
        )
        # Fail closed BEFORE score work unless healthz provides a self-consistent
        # runtime preimage whose identity is exactly the anchored manifest pin.
        contract = self._contract_for(manifest.scoring_version)
        try:
            response = self._client.post("/score", json=request.model_dump())
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_code, error_field = _typed_rejection(exc.response)
            raise ScoringClientError(
                f"scoring worker call failed: {exc}"
                + (f" [{error_code}" + (f" {error_field}]" if error_field else "]")
                   if error_code else ""),
                status_code=exc.response.status_code,
                error_code=error_code,
                error_field=error_field,
            ) from exc
        except httpx.HTTPError as exc:  # transport: connect/read/timeout — infra
            raise ScoringClientError(f"scoring worker call failed: {exc}") from exc
        try:
            parsed = ScoreResponse.model_validate(response.json())
        except ValidationError as exc:
            raise ScoringClientError(
                f"scoring worker returned a malformed response envelope: {exc}"
            ) from exc
        packet_bytes = parsed.item_score_json.encode("utf-8")
        if hashlib.sha256(packet_bytes).hexdigest() != parsed.packet_digest:
            raise ScoringClientError(
                "scoring worker returned a packet whose bytes do not hash to its "
                "claimed packet_digest — refusing to persist"
            )
        try:
            item_score = ItemScore.model_validate_json(packet_bytes)
        except ValidationError as exc:
            raise ScoringClientError(
                f"scoring worker returned a malformed ItemScore packet: {exc}"
            ) from exc
        binding_problem = self._binding_problem(item_score, request)
        if binding_problem is not None:
            raise ScoringClientError(
                "scoring worker returned a packet that is not bound to this request: "
                + binding_problem
            )

        expected_runtime = contract.backend_versions[SCORER_RUNTIME_BACKEND_KEY]
        packet_runtime = item_score.backend_versions.get(SCORER_RUNTIME_BACKEND_KEY)
        if packet_runtime != expected_runtime:
            raise ScoringClientError(
                "scoring worker packet runtime stamp differs from its validated "
                f"health commitment (packet={packet_runtime!r}, "
                f"expected={expected_runtime!r})"
            )
        if item_score.backend_versions != contract.backend_versions:
            changed = sorted(
                key
                for key in (
                    set(item_score.backend_versions) | set(contract.backend_versions)
                )
                if item_score.backend_versions.get(key)
                != contract.backend_versions.get(key)
            )
            raise ScoringClientError(
                "scoring worker packet backend_versions differs from the COMPLETE "
                "validated health runtime map; mismatched key(s): " + ", ".join(changed)
            )
        return ScorePacket(
            item_id=item.item_id, contender_id=contender_id, packet_bytes=packet_bytes
        )
