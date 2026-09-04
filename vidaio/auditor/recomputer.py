"""The REAL ``ScoreRecomputer`` — over the ACTUAL scoring engine.

This is the crux of the auditor's value (the project design record §1c, build
wave 6): given one audit item's integrity-verified artifacts, it RERUNS the honest
scoring pipeline and returns the :class:`~vidaio.audit.recompute.RecomputedScore`
that ``verify_bundle`` compares against the RECORDED score packet. An injected or
substituted score cannot survive it — the fresh measurement over the real bytes is
what fails ``SCORE_MISMATCH``.

How it reuses the scoring engine (does NOT duplicate it): it composes the audit
artifacts back into a :class:`~vidaio.services.protocol.ScoreRequest` and runs the
scoring worker's OWN pipeline function
(:func:`vidaio.scoring_worker.service._score_sync`) verbatim — same verify-then-
snapshot of every input, same canonicalization, same VMAF models, same gate
pipeline, same gates-first composition. So an honest packet recomputes EQUAL within
the audit tolerances, because it is the identical code path that produced it.

Both tracks recompute on CPU: compression uses ffmpeg/libvmaf; upscaling uses the
pinned PIQ/PyTorch CPU PieAPP model; and tone/grayscale/chroma use deterministic
integer reductions over pinned OpenCV decoding. Missing dependencies or model
weights still fail closed as :class:`RecomputeUnavailable`.

Backend-injectable: pass real ffmpeg/libvmaf backends for real audits, or a
``DeterministicFakeBackend`` composition for logic tests.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from vidaio.audit.bundle import LifecycleStage
from vidaio.audit.recompute import (
    _ORCHESTRATOR_ZERO_PREFIX,
    ArtifactPayload,
    RecomputedScore,
    _orchestrator_zero_identity,
)
from vidaio.audit.store import ArtifactKind
from vidaio.challenge.commitment import RevealedCommitment, verify_reveal_deep
from vidaio.challenge.dag import UPSCALE_FACTORS, build_dag, dag_rng_from_seed
from vidaio.competition.item_commitment import evaluation_item_commitment
from vidaio.scoring import (
    TRACK_UPSCALING,
    InvalidDuplicateEvidence,
    ScoringConfig,
    canonical_receipt_digest,
    config_digest,
    duplicate_identity,
    duplicate_order_key,
    duplicate_witness_from_packet,
    is_duplicate_identity,
)
from vidaio.scoring.backends_real import NotConfiguredError
from vidaio.scoring_worker import (
    ScoringBackends,
    ScoringWorkerConfig,
    effective_scorer_version,
    real_backends,
)
from vidaio.scoring_worker.inputs import ScoreRejected
from vidaio.scoring_worker.runtime_identity import (
    payout_runtime_attestation,
    require_attested_backend_versions,
    require_canonical_release_runtime,
    runtime_commitment_digest,
)
from vidaio.scoring_worker.service import _score_sync
from vidaio.services.protocol import ScoreRequest

#: The version string unconfigured backend stubs stamp (see backends_real).
_UNCONFIGURED = "not-configured"
_VALIDATOR_ZERO_SCORER_NAME = "validator-zero/1"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _is_validator_zero_identity(identity: object) -> bool:
    value = str(identity or "")
    return value == _VALIDATOR_ZERO_SCORER_NAME or value.startswith(
        f"{_VALIDATOR_ZERO_SCORER_NAME}+"
    )


class RecomputeUnavailable(Exception):
    """The item cannot be HONESTLY recomputed on this backend composition.

    NOT a verdict: the auditor records SKIP, never PASS/FAIL. Raised when a metric
    the item needs is measured by an unconfigured backend (PieAPP model weights or
    the required CPU perceptual dependencies) —
    the same honesty boundary as the scoring worker's typed 501.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RealScoreRecomputer:
    """``ScoreRecomputer`` implemented over the real scoring worker pipeline.

    Construct with a :class:`ScoringWorkerConfig` and a :class:`ScoringBackends`
    composition; the recomputer stamps and enforces the SAME ``scorer_version`` the
    worker would for that config, so recompute parity is a property of running the
    identical scorer, not a coincidence.
    """

    def __init__(
        self,
        config: ScoringWorkerConfig,
        backends: ScoringBackends,
        *,
        scoring_config: ScoringConfig | None = None,
        allow_noncanonical_pre_marker_build_or_test_runtime: bool = False,
    ) -> None:
        """Build the shipped CPU recomputer over a canonical payout runtime.

        ``allow_noncanonical_pre_marker_build_or_test_runtime`` exists only for
        the dependency smoke that runs before the release marker is created and
        for isolated tests with injected backends.  Production callers and
        independent auditors must keep the fail-closed default.
        """
        self._config = config
        self._backends = backends
        self._scoring_config = scoring_config or ScoringConfig()
        backend_attestation = getattr(backends, "runtime_attestation", None)
        if allow_noncanonical_pre_marker_build_or_test_runtime:
            attestation = backend_attestation
            if attestation is None:
                attestation = payout_runtime_attestation(
                    config, self._scoring_config
                )
        else:
            # Runtime metadata supplied by an injected composition is not proof
            # of where this process is executing. Re-probe locally, require the
            # canonical contract, and bind the actual backends to that result.
            attestation = payout_runtime_attestation(config, self._scoring_config)
            require_canonical_release_runtime(attestation)
            if backend_attestation is None:
                raise RuntimeError(
                    "canonical audit backends must carry the runtime attestation "
                    "produced by real_backends"
                )
            if getattr(backends.pieapp, "device", None) != "cpu":
                raise RuntimeError("canonical audit PieAPP backend must execute on CPU")
            if runtime_commitment_digest(
                backend_attestation
            ) != runtime_commitment_digest(attestation):
                raise RuntimeError(
                    "audit backend runtime attestation differs from the "
                    "independently probed canonical payout runtime"
                )
        if backend_attestation is not None:
            # The build/test marker-policy opt-out never permits an attestation
            # whose concrete packet backend map says something else.
            require_attested_backend_versions(attestation, backends.versions)
        self._scorer_version = effective_scorer_version(
            config,
            self._scoring_config,
            runtime_attestation=attestation,
        )
        self._config.work_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        config: ScoringWorkerConfig,
        *,
        scoring_config: ScoringConfig | None = None,
        allow_noncanonical_pre_marker_build_or_test_runtime: bool = False,
    ) -> "RealScoreRecomputer":
        """Compose the shipped CPU-capable audit backends.

        Identical composition to the scoring worker's ``real_backends``: compression
        Compression and upscaling both recompute for real. The explicit CPU
        override is invariant even when the scoring-worker config names CUDA.
        """
        scoring_cfg = scoring_config or ScoringConfig()
        # Hard invariant: an auditor never selects CUDA, even on a CUDA host.
        # Validators can therefore accept the audit workload on ordinary CPUs.
        return cls(
            config,
            real_backends(config, scoring_config=scoring_cfg, pieapp_device="cpu"),
            scoring_config=scoring_cfg,
            allow_noncanonical_pre_marker_build_or_test_runtime=(
                allow_noncanonical_pre_marker_build_or_test_runtime
            ),
        )

    @property
    def scorer_version(self) -> str:
        return self._scorer_version

    # -- honest-refusal probe ---------------------------------------------------------

    def _pieapp_available(self) -> bool:
        return getattr(self._backends.pieapp, "version", "") != _UNCONFIGURED

    @staticmethod
    def _track(artifacts: Mapping[ArtifactKind, ArtifactPayload]) -> str:
        """The item's declared track, read from the recorded packet (else "")."""
        packet_bytes = artifacts.get(ArtifactKind.SCORE_PACKET)
        if packet_bytes is None or isinstance(packet_bytes, Path):
            return ""
        try:
            payload = json.loads(packet_bytes)
        except (ValueError, TypeError):
            return ""
        return str(payload.get("track", "")) if isinstance(payload, dict) else ""

    def unsupported_reason(
        self,
        bundle: object,
        artifacts: Mapping[ArtifactKind, ArtifactPayload],
        *,
        track: str | None = None,
    ) -> str | None:
        """Why this item cannot be recomputed on THIS backend, or None.

        Cheap: no media is decoded. Recompute-ability is decided from the COMMITTED
        ``track`` when the auditor supplies one (#9) — NOT the authority's
        packet-declared track — so a packet substituting ``track=upscaling`` to force a
        PieAPP-unavailable SKIP over a real (committed-compression) item cannot dodge
        verification. Falls back to the packet's declared track only when no committed
        track is supplied (legacy/report path). The auditor calls this BEFORE
        ``verify_bundle`` so a genuinely un-recomputable item becomes a SKIP, never a
        false CLEAN or a spurious FAIL.
        """
        # A validator-attributed failure packet performs no media measurement and
        # therefore needs neither PieAPP nor ffmpeg. Its strict convention is
        # validated by ``recompute``; do not turn an upscaling timeout into an
        # un-auditable PieAPP SKIP before that validation gets a chance to run.
        try:
            packet_identity = str(self._packet(artifacts).get("scorer_version", ""))
        except Exception:
            packet_identity = ""
        if (
            _is_validator_zero_identity(packet_identity)
            or packet_identity.startswith(_ORCHESTRATOR_ZERO_PREFIX)
            or self._is_duplicate(artifacts)
        ):
            return None
        track = track if track is not None else self._track(artifacts)
        if track == TRACK_UPSCALING and not self._pieapp_available():
            return (
                "upscaling item requires the PieAPP perceptual backend, which is "
                "not installed or has no cached model weights — cannot recompute "
                "(honest refusal, never a false CLEAN; mirrors the worker's 501)"
            )
        return None

    # -- recompute --------------------------------------------------------------------

    def recompute(
        self, bundle: object, artifacts: Mapping[ArtifactKind, ArtifactPayload]
    ) -> RecomputedScore:
        """Rerun the honest pipeline over ``artifacts`` and return the fresh score.

        Reconstructs the ScoreRequest from the item's artifacts + recorded packet
        (track, and the scoring params the packet recorded — the compression VMAF
        threshold from the breakdown, the upscaling content length from the metrics)
        and hands it to the worker's own ``_score_sync``. Raises
        :class:`RecomputeUnavailable` when a required backend is unconfigured (so the
        auditor SKIPs), and lets a genuine engine failure propagate as an error
        ``verify_bundle`` reports as RECOMPUTE_ERROR.
        """
        packet = self._packet(artifacts)
        if _is_validator_zero_identity(packet.get("scorer_version")):
            raise RuntimeError(
                "validator-zero packets are not launch-valid economic evidence"
            )
        if is_duplicate_identity(packet.get("scorer_version")):
            raise RuntimeError(
                "duplicate recompute requires the witness's second output; verify_bundle "
                "must call recompute_duplicate"
            )

        reason = self.unsupported_reason(bundle, artifacts)
        if reason is not None:
            raise RecomputeUnavailable(reason)

        try:
            reference = artifacts[ArtifactKind.REFERENCE_ORIGINAL]
            miner_input = artifacts[ArtifactKind.CHALLENGE_INPUT]
            output = artifacts[ArtifactKind.MINER_OUTPUT]
        except KeyError as exc:
            raise RuntimeError(
                f"recompute requires the {exc} artifact but it was not fetched"
            ) from exc

        track = str(packet.get("track", ""))
        params = self._reconstruct_params(bundle, artifacts, packet)

        with TemporaryDirectory(prefix="auditor-recompute-") as tmp:
            root = Path(tmp)
            ref_path, reference_digest = self._as_path(
                root / "reference_original", reference
            )
            input_path, input_digest = self._as_path(
                root / "challenge_input", miner_input
            )
            output_path, output_digest = self._as_path(root / "miner_output", output)

            request = ScoreRequest(
                track=track,
                challenge_id=getattr(bundle, "challenge_id", ""),
                item_id=getattr(bundle, "item_id", ""),
                miner_hotkey=getattr(bundle, "miner_hotkey", None),
                reference_path=str(ref_path),
                reference_digest=reference_digest,
                miner_input_path=str(input_path),
                miner_input_digest=input_digest,
                output_path=str(output_path),
                output_digest=output_digest,
                params=params,
                scorer_version=None,  # accept the recomputer's own identity
            )
            try:
                item = _score_sync(
                    request,
                    self._config,
                    self._scoring_config,
                    self._backends,
                    self._scorer_version,
                )
            except NotConfiguredError as exc:
                # An unconfigured backend surfaced only once the pipeline ran (e.g.
                # a deliberately injected refusal backend) — the SAME honest
                # refusal as the worker's 501, mapped to a SKIP by the auditor.
                raise RecomputeUnavailable(
                    f"scoring backend not configured: {exc}"
                ) from exc
            except ScoreRejected as exc:
                # The pipeline typed-rejected the reconstructed request (e.g. an
                # unsupported track) — an engine-level recompute failure, surfaced
                # as RECOMPUTE_ERROR by verify_bundle.
                raise RuntimeError(
                    f"recompute rejected the item: {exc.payload}"
                ) from exc

        metrics = {
            k: float(v)
            for k, v in item.metrics.items()
            if isinstance(v, numbers.Real) and not isinstance(v, bool)
        }
        return RecomputedScore(
            metrics=metrics,
            scorer_version=item.scorer_version or self._scorer_version,
            backend_versions=dict(item.backend_versions),
            score=item.score,
            gate_passed=item.gate_passed,
            violations=[
                v.model_dump(mode="json") for v in getattr(item, "violations", ())
            ],
            breakdown=(
                item.breakdown.model_dump(mode="json")
                if getattr(item, "breakdown", None) is not None
                else None
            ),
        )

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _packet(artifacts: Mapping[ArtifactKind, ArtifactPayload]) -> dict:
        raw = artifacts.get(ArtifactKind.SCORE_PACKET)
        if raw is None:
            raise RuntimeError("recompute requires the score_packet artifact")
        if isinstance(raw, Path):
            raise RuntimeError("score_packet metadata was materialized as media")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("score packet is not a JSON object")
        return payload

    @classmethod
    def _is_duplicate(cls, artifacts: Mapping[ArtifactKind, ArtifactPayload]) -> bool:
        try:
            packet = cls._packet(artifacts)
        except Exception:
            return False
        return is_duplicate_identity(packet.get("scorer_version"))

    def recompute_duplicate(
        self,
        bundle: object,
        artifacts: Mapping[ArtifactKind, ArtifactPayload],
        winner_output: ArtifactPayload,
        witness_payload: Mapping[str, Any],
    ) -> RecomputedScore:
        """CPU-recompute both SHA-256 digests and the deterministic exact verdict."""
        packet = self._packet(artifacts)
        witness = duplicate_witness_from_packet(packet)
        if witness.model_dump(mode="json") != dict(witness_payload):
            raise RuntimeError(
                "duplicate witness changed between audit parsing and recompute"
            )
        revealed = self._committed_reveal(bundle, artifacts)
        if (
            revealed.scorer_version != witness.committed_scorer_version
            or revealed.track != witness.track
        ):
            raise RuntimeError(
                "duplicate witness is not bound to the committed scorer and track"
            )
        scoring_digest = config_digest(self._scoring_config)
        expected_identity = duplicate_identity(
            committed_scorer_version=revealed.scorer_version,
            track=revealed.track,
            scoring_config_digest=scoring_digest,
        )
        if (
            packet.get("scoring_config_digest") != scoring_digest
            or packet.get("scorer_version") != expected_identity
            or str(getattr(bundle, "scorer_version", "")) != expected_identity
        ):
            raise RuntimeError(
                "duplicate packet identity/config is not the committed launch convention"
            )
        try:
            loser_output = artifacts[ArtifactKind.MINER_OUTPUT]
            challenge_input = artifacts[ArtifactKind.CHALLENGE_INPUT]
        except KeyError as exc:
            raise RuntimeError(f"duplicate recompute requires {exc} artifact") from exc
        with TemporaryDirectory(prefix="auditor-duplicate-") as tmp:
            root = Path(tmp)
            loser_path, loser_digest = self._as_path(
                root / "loser_output", loser_output
            )
            winner_path, winner_digest = self._as_path(
                root / "winner_output", winner_output
            )
            _, input_digest = self._as_path(root / "challenge_input", challenge_input)
            loser_size = loser_path.stat().st_size
            winner_size = winner_path.stat().st_size

        if (
            loser_digest != witness.loser_output_digest
            or loser_size != witness.loser_output_size
            or winner_digest != witness.winner_output.digest
            or winner_size != witness.winner_output.byte_size
        ):
            raise RuntimeError(
                "duplicate witness output digests/sizes do not match stored bytes"
            )
        if loser_digest != winner_digest or loser_size != winner_size:
            raise RuntimeError(
                "economic duplicate evidence is not byte-exact under independent SHA-256"
            )

        loser_receipt = getattr(bundle, "miner_receipt", None)
        if loser_receipt is None:
            raise RuntimeError(
                "duplicate loser receipt is absent from the audit bundle"
            )
        challenge_id = str(getattr(bundle, "challenge_id", ""))
        if (
            canonical_receipt_digest(loser_receipt) != witness.loser_receipt_digest
            or loser_receipt.metadata.task_id != f"{challenge_id}:{witness.loser_uid}"
            or loser_receipt.miner_hotkey != witness.loser_hotkey
        ):
            raise RuntimeError(
                "duplicate witness does not bind the loser task, hotkey and receipt"
            )
        winner_receipt = witness.winner_receipt
        anchor = getattr(bundle, "challenge_anchor", None)
        if (
            anchor is None
            or anchor.block_hash is None
            or loser_receipt.metadata.commitment_anchor != anchor
            or winner_receipt.metadata.commitment_anchor != anchor
            or winner_receipt.metadata.task_id != f"{challenge_id}:{witness.winner_uid}"
            or winner_receipt.metadata.track != witness.track
            or winner_receipt.metadata.input_digest != input_digest
            or winner_receipt.input_size != self._payload_size(challenge_input)
            or winner_receipt.output_digest != winner_digest
            or winner_receipt.output_size != winner_size
            or winner_receipt.miner_hotkey != witness.winner_hotkey
            or winner_receipt.validator_hotkey != loser_receipt.validator_hotkey
        ):
            raise RuntimeError(
                "duplicate winner receipt is not bound to the same challenge and media"
            )
        try:
            winner_order = duplicate_order_key(anchor.block_hash, witness.winner_hotkey)
            loser_order = duplicate_order_key(anchor.block_hash, witness.loser_hotkey)
        except InvalidDuplicateEvidence as exc:
            raise RuntimeError(
                f"duplicate deterministic ordering is invalid: {exc}"
            ) from exc
        if winner_order >= loser_order:
            raise RuntimeError(
                "duplicate winner violates anchor_hash_hotkey/1 deterministic ordering"
            )
        return RecomputedScore(
            metrics={},
            scorer_version=expected_identity,
            backend_versions={},
            score=0.0,
            gate_passed=False,
        )

    def recompute_orchestrator_zero(
        self,
        bundle: object,
        artifacts: Mapping[ArtifactKind, ArtifactPayload],
        *,
        committed_scorer_version: str,
        committed_track: str,
    ) -> RecomputedScore:
        """CPU-verify the canonical orchestrator-authored economic zero.

        This path invents no media metric. Its independent fact is byte-exact:
        the contender output artifact is empty. The remaining closed packet shape
        derives from the manifest-committed worker/track and this auditor's locked
        scoring config. Any measurement-shaped field or non-empty output is refused.
        """
        packet = self._packet(artifacts)
        scoring_digest = config_digest(self._scoring_config)
        expected_identity = _orchestrator_zero_identity(
            committed_scorer_version=committed_scorer_version,
            track=committed_track,
            scoring_config_digest=scoring_digest,
        )

        violations = packet.get("violations")
        violation = (
            violations[0]
            if isinstance(violations, list) and len(violations) == 1
            else None
        )
        problems: list[str] = []
        if (
            packet.get("scorer_version") != expected_identity
            or str(getattr(bundle, "scorer_version", "")) != expected_identity
        ):
            problems.append(
                "reserved scorer identity is not derived from committed policy"
            )
        if packet.get("scoring_config_digest") != scoring_digest:
            problems.append(
                "scoring_config_digest differs from the CPU auditor's locked config"
            )
        if packet.get("track") != committed_track:
            problems.append("packet track differs from the committed competition track")
        if packet.get("score") != 0.0 or packet.get("gate_passed") is not False:
            problems.append("orchestrator-zero outcome must be gate-failed score zero")
        if packet.get("metrics") != {}:
            problems.append("orchestrator-zero packet must not claim measured metrics")
        if packet.get("backend_versions") != {}:
            problems.append("orchestrator-zero packet must not claim measurement backends")
        if (
            packet.get("breakdown") is not None
            or packet.get("skips") not in (None, [])
            or packet.get("canonicalization_plan_digest") is not None
            or packet.get("pieapp_start_frame") is not None
        ):
            problems.append("orchestrator-zero packet must not claim scoring work")
        if not isinstance(violation, dict) or (
            violation.get("code") != "METRIC_MISSING"
            or not isinstance(violation.get("detail"), str)
            or not violation.get("detail")
            or violation.get("measured") is not None
            or violation.get("limit") is not None
        ):
            problems.append(
                "orchestrator-zero packet has no canonical METRIC_MISSING violation"
            )

        output = artifacts.get(ArtifactKind.MINER_OUTPUT)
        if output is None:
            problems.append("orchestrator-zero output artifact is unavailable")
        else:
            output_size = self._payload_size(output)
            output_digest = (
                self._hash_path(output) if isinstance(output, Path) else _sha256(output)
            )
            output_ref = getattr(bundle, "miner_output", None)
            if (
                output_size != 0
                or output_digest != _EMPTY_SHA256
                or output_ref is None
                or output_ref.byte_size != 0
                or output_ref.digest != _EMPTY_SHA256
                or packet.get("content_digest") != _EMPTY_SHA256
            ):
                problems.append(
                    "orchestrator-zero is not backed by the canonical empty output"
                )
        if problems:
            raise RuntimeError("; ".join(problems))

        return RecomputedScore(
            metrics={},
            scorer_version=expected_identity,
            backend_versions={},
            score=0.0,
            gate_passed=False,
            violations=[dict(violation)],
            breakdown=None,
        )

    @staticmethod
    def _committed_reveal(
        bundle: object, artifacts: Mapping[ArtifactKind, ArtifactPayload]
    ) -> RevealedCommitment:
        """Parse and deeply verify the challenge commitment's canonical preimage.

        ``DAG_REVEAL`` is not an authority-controlled bag of scoring parameters. It
        is the exact canonical preimage whose sha256 is pinned by
        ``bundle.commitment_hash``. Rebuilding the DAG from its seed additionally
        proves that parameters recovered from the DAG were fixed before dispatch,
        rather than selected after the miner's response was known.
        """
        raw = artifacts.get(ArtifactKind.DAG_REVEAL)
        if raw is None:
            raise RuntimeError("recompute requires the committed DAG_REVEAL artifact")
        if isinstance(raw, Path):
            raise RuntimeError("DAG_REVEAL metadata was materialized as media")

        commitment_hash = str(getattr(bundle, "commitment_hash", ""))
        if not commitment_hash:
            raise RuntimeError("audit bundle does not pin a challenge commitment hash")
        if _sha256(raw) != commitment_hash:
            raise RuntimeError(
                "DAG_REVEAL bytes do not match the audit bundle's commitment hash"
            )

        try:
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                raise TypeError("reveal is not a JSON object")
            revealed = RevealedCommitment(
                clean_asset_id=doc["asset_id"],
                dag_digest=doc["dag_digest"],
                seed=doc["seed"],
                scorer_version=doc["scorer_version"],
                track=doc["track"],
                dispatch_ordering_key=doc["dispatch_ordering_key"],
                commit_hash=commitment_hash,
                revealed_at="",
            )
        except Exception as exc:
            raise RuntimeError(
                f"DAG_REVEAL is not a valid challenge commitment preimage: {exc}"
            ) from exc
        if not verify_reveal_deep(revealed):
            raise RuntimeError(
                "DAG_REVEAL does not regenerate its committed DAG from the revealed seed"
            )
        return revealed

    @classmethod
    def _reconstruct_params(
        cls,
        bundle: object,
        artifacts: Mapping[ArtifactKind, ArtifactPayload],
        packet: dict,
    ) -> dict[str, float | int]:
        """The scoring params the recompute uses — DELIBERATELY NOT the packet's (#9).

        The gate parameters must come from a source the AUTHORITY cannot dial, not
        from its own score packet: a real VMAF 86 (which should fail a threshold of
        90) packaged with ``vmaf_threshold=0`` + a positive score would otherwise let
        the auditor reuse 0, pass the gate, and reproduce the misreport. So the recompute
        passes NO ``vmaf_threshold`` (the worker falls back to the auditor's OWN
        locked ``ScoringConfig`` threshold for the track) and NO ``content_length``
        (the worker re-measures it from the committed reference bytes). Both are
        values the auditor controls or measures, so a packet-supplied gate parameter
        can never soften the auditor's recompute. An honest packet — produced with the
        same config threshold and reference duration — still recomputes EQUAL.

        Upscaling's file-size gate is the one exception that needs a per-challenge
        value: its discrete scale factor. That value is reconstructed from the
        seed-derived, commitment-bound degradation DAG, never read from the packet.
        The same preimage also binds the track, so a packet cannot choose which DAG
        interpretation the auditor uses.

        (A legitimate per-competition threshold override, if ever introduced, must
        likewise be carried in committed challenge/manifest evidence.)
        """
        packet_track = str(packet.get("track", ""))
        stage = getattr(bundle, "stage", None)
        if stage in {
            LifecycleStage.PRE_REVEAL,
            LifecycleStage.COMPETITION_SEALED,
        }:
            raw_manifest = artifacts.get(ArtifactKind.MANIFEST)
            if raw_manifest is None or isinstance(raw_manifest, Path):
                raise RuntimeError(
                    "competition recompute requires the committed manifest artifact"
                )
            try:
                manifest = json.loads(raw_manifest)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"competition manifest is not valid JSON: {exc}"
                ) from exc
            if not isinstance(manifest, dict) or manifest.get("track") != packet_track:
                raise RuntimeError(
                    "competition manifest track does not match the score packet"
                )
            if packet_track == TRACK_UPSCALING:
                binding = getattr(bundle, "competition_item", None)
                if binding is None:
                    raise RuntimeError(
                        "upscaling competition bundle has no committed item preimage"
                    )
                derived = evaluation_item_commitment(
                    competition_id=str(manifest.get("competition_id", "")),
                    item_index=binding.item_index,
                    reference_sha256=binding.reference_sha256,
                    input_sha256=binding.input_sha256,
                    upscale_factor=binding.upscale_factor,
                    target_width=binding.target_width,
                    target_height=binding.target_height,
                )
                commitments = manifest.get("evaluation_item_commitments")
                allowed = manifest.get("allowed_upscale_factors")
                if (
                    derived != binding.item_commitment
                    or not isinstance(commitments, list)
                    or binding.item_index >= len(commitments)
                    or commitments[binding.item_index] != derived
                    or not isinstance(allowed, list)
                    or binding.upscale_factor not in allowed
                ):
                    raise RuntimeError(
                        "upscaling factor/reference/input do not match the committed "
                        "manifest item"
                    )
                params = {"upscale_factor": binding.upscale_factor}
                if binding.target_width is not None:
                    params.update(
                        {
                            "target_width": binding.target_width,
                            "target_height": binding.target_height,
                        }
                    )
                return params
            threshold = manifest.get("vmaf_threshold")
            if (
                packet_track != TRACK_UPSCALING
                and isinstance(threshold, numbers.Real)
                and not isinstance(threshold, bool)
            ):
                return {"vmaf_threshold": float(threshold)}
            if packet_track != TRACK_UPSCALING:
                raise RuntimeError(
                    "competition manifest has no numeric VMAF threshold"
                )
        if packet_track != TRACK_UPSCALING:
            return {}

        revealed = cls._committed_reveal(bundle, artifacts)
        if revealed.track != packet_track:
            raise RuntimeError(
                f"score packet track {packet_track!r} does not match the committed "
                f"DAG_REVEAL track {revealed.track!r}"
            )
        dag = build_dag(revealed.track, dag_rng_from_seed(revealed.seed))
        downscales = [op for op in dag.ops if getattr(op, "op", "") == "downscale"]
        if len(downscales) != 1:
            raise RuntimeError(
                "committed upscaling DAG must contain exactly one downscale operation"
            )
        scale_factor = float(downscales[0].scale_factor)
        if scale_factor <= 0.0:
            raise RuntimeError(
                "committed upscaling DAG has a non-positive scale factor"
            )
        upscale_factor = round(1.0 / scale_factor)
        if upscale_factor not in UPSCALE_FACTORS:
            raise RuntimeError(
                f"committed upscaling DAG yields unsupported factor {upscale_factor}; "
                f"supported factors are {list(UPSCALE_FACTORS)}"
            )
        return {"upscale_factor": upscale_factor}

    @staticmethod
    def _write(path: Path, data: bytes) -> Path:
        path.write_bytes(data)
        return path

    @classmethod
    def _as_path(cls, destination: Path, payload: ArtifactPayload) -> tuple[Path, str]:
        """Reuse verified disk media; preserve byte fixtures for unit tests."""
        if isinstance(payload, Path):
            return payload, cls._hash_path(payload)
        return cls._write(destination, payload), _sha256(payload)

    @staticmethod
    def _hash_path(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _payload_size(payload: ArtifactPayload) -> int:
        return payload.stat().st_size if isinstance(payload, Path) else len(payload)
