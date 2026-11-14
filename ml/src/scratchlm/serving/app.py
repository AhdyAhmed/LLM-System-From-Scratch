"""
Day051-Day053: FastAPI internal inference service (`create_inference_app`, `ScratchInferenceEngine`)
loading the frozen Gate G5 `ScratchLmPyfuncAdapter` package with:
  - `X-Internal-Service-Token` authentication (`401 Unauthorized` on missing/invalid token)
  - `/internal/health` and `/internal/readiness` with immutable provenance & GPU lease telemetry
  - Package & tokenizer SHA-256 verification against `frozen_g5_candidate.json`
  - MLflow downtime resilience (active inference continues from verified local cache while new
    promotions requiring MLflow verification are blocked with `503 MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED`)
  - Bounded NDJSON streaming (`POST /internal/v1/generate`) using Day057 `ScratchKVCache`
    and Day058 `ContextPacker`, with per-step deadline, cancellation (`POST /internal/v1/cancel/{run_id}`),
    and fault-injection support.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from scratchlm.inference.context_packer import (
    ContextPacker,
    ContextWindowExceededError,
    ConversationTurn,
)
from scratchlm.inference.kv_cache import ScratchKVCache
from scratchlm.inference.sampling import sample_next_token
from scratchlm.serving.gpu_supervisor import GpuCapacityError, SingleGpuSupervisor
from scratchlm.serving.package import ScratchLmPyfuncAdapter
from scratchlm.tokenizer.bpe_tokenizer import EOS_TOKEN_ID

DEFAULT_INTERNAL_SERVICE_TOKEN = "scratchllm-internal-dev-token-day051"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


class SamplingSpecModel(BaseModel):
    max_new_tokens: int = Field(..., ge=1, le=64)
    temperature: float = Field(..., ge=0.0, le=2.0)
    top_k: int = Field(..., ge=1, le=256)
    top_p: float = Field(..., ge=0.05, le=1.0)
    seed: int | None = Field(default=1337, ge=0)


class FaultInjectionModel(BaseModel):
    chunk_delay_ms: int = Field(default=0, ge=0, le=5000)
    fail_after_chunks: int | None = Field(default=None, ge=0, le=64)
    error_code: str | None = Field(default=None, max_length=64)


class ConversationTurnModel(BaseModel):
    role: str
    content: str


class InternalGenerateRequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    schema_version: int = Field(..., ge=1, le=1)
    run_id: str = Field(..., min_length=1, max_length=64)
    attempt_token: int = Field(..., ge=1)
    model_revision: str = Field(..., min_length=1, max_length=120)
    prompt: str = Field(..., min_length=1, max_length=16384)
    context_limit: int = Field(default=128)
    sampling: SamplingSpecModel
    deadline_epoch_ms: int = Field(..., ge=1)
    fault_injection: FaultInjectionModel | None = None
    history: list[ConversationTurnModel] | None = None
    apply_chat_template: bool = True
    use_kv_cache: bool = True


class PromoteModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    target_version: str
    expected_package_sha256: str
    expected_tokenizer_sha256: str
    package_dir: str | None = None


class DrainGpuRequest(BaseModel):
    training_job_id: str
    training_pid: int | None = None
    training_create_time_utc: float | None = None
    drain_timeout_sec: float = 5.0


class ReleaseGpuRequest(BaseModel):
    training_job_id: str | None = None
    respawn_child_worker: bool = False


@dataclass
class LoadedModelRuntime:
    adapter: ScratchLmPyfuncAdapter
    context_packer: ContextPacker
    model_revision: str
    selected_model_name: str
    registry_version: str
    source_run_id: str
    package_sha256: str
    tokenizer_sha256: str
    device: str
    package_dir: Path


class ScratchInferenceEngine:
    """
    Manages the verified `ScratchLmPyfuncAdapter` runtime, Day058 `ContextPacker`,
    Day056 `SingleGpuSupervisor`, and MLflow promotion readiness checks.
    """

    def __init__(
        self,
        *,
        frozen_config_path: Path | None = None,
        package_dir_override: Path | None = None,
        device_preference: str | None = None,
        mlflow_tracking_uri: str | None = None,
        max_queued_runs: int = 8,
    ) -> None:
        repo_root = _default_repo_root()
        self.frozen_config_path = frozen_config_path or (
            repo_root / "ml" / "configs" / "frozen_g5_candidate.json"
        )
        self.package_dir_override = package_dir_override
        self.device_preference = device_preference or os.environ.get("SCRATCHLM_DEVICE", "auto")
        self.mlflow_tracking_uri = (
            mlflow_tracking_uri
            or os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:50000")
        ).rstrip("/")

        self.readiness_status: str = "UNHEALTHY"
        self.readiness_error: str | None = None
        self.runtime: LoadedModelRuntime | None = None

        resolved_device = self._resolve_device(self.device_preference)
        self.gpu_supervisor = SingleGpuSupervisor(
            max_active_slots=1,
            max_queued_runs=max_queued_runs,
            device=resolved_device,
        )
        self.reload_from_frozen_config()

    @staticmethod
    def _resolve_device(preference: str) -> str:
        pref = preference.strip().lower()
        if pref == "cpu":
            return "cpu"
        if pref.startswith("cuda"):
            return pref if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def reload_from_frozen_config(
        self,
        *,
        expected_package_sha256_override: str | None = None,
        expected_tokenizer_sha256_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Load and cryptographically verify the frozen G5 model package.
        If any file hash, `package_sha256`, or `tokenizer_sha256` fails verification,
        transitions `readiness_status` to `UNHEALTHY` and raises ValueError.
        """
        try:
            frozen = json.loads(self.frozen_config_path.read_text(encoding="utf-8"))
            repo_root = _default_repo_root()
            rel_pkg = str(frozen.get("package_dir", "artifacts/packages/scratchlm_candidate_pkg"))
            pkg_dir = self.package_dir_override or (repo_root / "ml" / rel_pkg)
            expected_pkg_sha = expected_package_sha256_override or str(frozen["package_sha256"])
            expected_tok_sha = expected_tokenizer_sha256_override or str(frozen["tokenizer_sha256"])

            device = self._resolve_device(self.device_preference)
            adapter = ScratchLmPyfuncAdapter.load(pkg_dir, device=device)

            if adapter.manifest.package_sha256 != expected_pkg_sha:
                raise ValueError(
                    f"Package SHA-256 mismatch: manifest={adapter.manifest.package_sha256} "
                    f"vs expected={expected_pkg_sha}"
                )
            actual_tok_sha = adapter.manifest.lineage.tokenizer_sha256
            if actual_tok_sha != expected_tok_sha or adapter.tokenizer.tokenizer_hash() != expected_tok_sha:
                raise ValueError(
                    f"Tokenizer SHA-256 mismatch: actual={actual_tok_sha} vs expected={expected_tok_sha}"
                )

            packer = ContextPacker(
                tokenizer=adapter.tokenizer,
                chat_template=adapter.chat_template,
                default_context_limit=adapter.model_config.max_seq_len,
                max_output_cap=64,
            )

            param_bytes = sum(p.numel() * p.element_size() for p in adapter.model.parameters())
            vram_bytes = (
                int(torch.cuda.memory_allocated())
                if device.startswith("cuda") and torch.cuda.is_available()
                else param_bytes
            )
            self.gpu_supervisor.package_dir = pkg_dir
            self.gpu_supervisor.device = device
            self.gpu_supervisor.set_inference_vram_bytes(vram_bytes)

            self.runtime = LoadedModelRuntime(
                adapter=adapter,
                context_packer=packer,
                model_revision="scratch-tiny-r1",
                selected_model_name=str(frozen["selected_model_name"]),
                registry_version=str(frozen["mlflow_registered_version"]),
                source_run_id=str(frozen.get("sft_run_id") or frozen.get("source_sft_run_id", "")),
                package_sha256=adapter.manifest.package_sha256,
                tokenizer_sha256=actual_tok_sha,
                device=device,
                package_dir=pkg_dir,
            )
            self.readiness_status = "READY"
            self.readiness_error = None
            return self.readiness_payload()
        except Exception as exc:
            self.readiness_status = "UNHEALTHY"
            self.readiness_error = str(exc)
            self.runtime = None
            raise ValueError(f"MODEL_PACKAGE_CORRUPTED: {exc}") from exc

    def check_mlflow_reachable(self, timeout_sec: float = 1.5) -> bool:
        try:
            resp = httpx.get(f"{self.mlflow_tracking_uri}/health", timeout=timeout_sec)
            return resp.status_code == 200
        except Exception:
            return False

    def readiness_payload(self) -> dict[str, Any]:
        sup = self.gpu_supervisor.metrics()
        lease = sup["gpu_lease"]
        effective_status = self.readiness_status
        if effective_status == "READY":
            if lease["holder_kind"] != "INFERENCE" or lease["lease_state"] != "ACTIVE":
                effective_status = "DRAINING"

        if self.runtime is None:
            return {
                "readiness_status": "UNHEALTHY",
                "error_code": "MODEL_PACKAGE_CORRUPTED",
                "error_message": self.readiness_error or "Model runtime is not loaded.",
                "gpu_lease": lease,
            }

        return {
            "readiness_status": effective_status,
            "model_revision": self.runtime.model_revision,
            "selected_model_name": self.runtime.selected_model_name,
            "registry_version": self.runtime.registry_version,
            "source_run_id": self.runtime.source_run_id,
            "package_sha256": self.runtime.package_sha256,
            "tokenizer_sha256": self.runtime.tokenizer_sha256,
            "context_limit": self.runtime.adapter.model_config.max_seq_len,
            "max_output_tokens": 64,
            "device": self.runtime.device,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
            "mlflow_reachable": self.check_mlflow_reachable(),
            "active_runs": sup["active_runs"],
            "queued_runs": sup["queued_runs"],
            "max_queued_runs": sup["max_queued_runs"],
            "gpu_lease": lease,
        }

    def promote_candidate_with_mlflow_guard(self, req: PromoteModelRequest) -> dict[str, Any]:
        """
        Day052: Promoting a new candidate requires live MLflow registry verification AND
        cryptographic package/tokenizer hash verification.
        If MLflow is unreachable, promotion is blocked (`MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED`)
        while the currently loaded model continues serving uninterrupted.
        """
        if not self.check_mlflow_reachable():
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED",
                    "message": (
                        "MLflow registry is unreachable; new model promotion is blocked while "
                        "existing cached model inference remains operational."
                    ),
                },
            )

        pkg_dir = (
            Path(req.package_dir)
            if req.package_dir
            else (self.runtime.package_dir if self.runtime is not None else None)
        )
        if pkg_dir is None or not pkg_dir.exists():
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_PACKAGE_PATH",
                    "message": f"Candidate package_dir does not exist: {pkg_dir}",
                },
            )

        try:
            candidate_adapter = ScratchLmPyfuncAdapter.load(pkg_dir, device="cpu")
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "MODEL_PACKAGE_CORRUPTED",
                    "message": f"Candidate package failed verification: {exc}",
                },
            ) from exc

        if candidate_adapter.manifest.package_sha256 != req.expected_package_sha256:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "PACKAGE_HASH_MISMATCH",
                    "message": (
                        f"Candidate package_sha256 ({candidate_adapter.manifest.package_sha256}) "
                        f"!= expected ({req.expected_package_sha256})"
                    ),
                },
            )
        if candidate_adapter.manifest.lineage.tokenizer_sha256 != req.expected_tokenizer_sha256:
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "TOKENIZER_HASH_MISMATCH",
                    "message": (
                        f"Candidate tokenizer_sha256 ({candidate_adapter.manifest.lineage.tokenizer_sha256}) "
                        f"!= expected ({req.expected_tokenizer_sha256})"
                    ),
                },
            )

        # Swap active runtime atomically
        device = self._resolve_device(self.device_preference)
        if device != "cpu":
            candidate_adapter = ScratchLmPyfuncAdapter.load(pkg_dir, device=device)

        packer = ContextPacker(
            tokenizer=candidate_adapter.tokenizer,
            chat_template=candidate_adapter.chat_template,
            default_context_limit=candidate_adapter.model_config.max_seq_len,
            max_output_cap=64,
        )
        self.runtime = LoadedModelRuntime(
            adapter=candidate_adapter,
            context_packer=packer,
            model_revision=f"scratch-tiny-r{req.target_version}",
            selected_model_name=req.model_name,
            registry_version=req.target_version,
            source_run_id=candidate_adapter.manifest.source_run_id,
            package_sha256=candidate_adapter.manifest.package_sha256,
            tokenizer_sha256=candidate_adapter.manifest.lineage.tokenizer_sha256,
            device=device,
            package_dir=pkg_dir,
        )
        self.readiness_status = "READY"
        self.readiness_error = None
        return self.readiness_payload()


def create_inference_app(
    engine: ScratchInferenceEngine | None = None,
    *,
    internal_token: str | None = None,
) -> FastAPI:
    """Create the FastAPI internal inference application for Day051–Day060."""
    app = FastAPI(
        title="ScratchLLM Internal Inference Service",
        version="0.6.0",
    )
    svc_engine = engine or ScratchInferenceEngine()
    expected_token = internal_token or os.environ.get(
        "INTERNAL_SERVICE_TOKEN", DEFAULT_INTERNAL_SERVICE_TOKEN
    )
    app.state.engine = svc_engine
    app.state.expected_token = expected_token

    def verify_internal_token(header_val: str | None) -> None:
        if not header_val or header_val != app.state.expected_token:
            raise HTTPException(
                status_code=401,
                detail={
                    "error_code": "UNAUTHORIZED_INTERNAL_TOKEN",
                    "message": "Missing or invalid X-Internal-Service-Token header.",
                },
            )

    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": "HTTP_ERROR", "message": str(exc.detail)},
        )

    @app.get("/internal/health")
    def health() -> dict[str, Any]:
        return {
            "status": "UP",
            "service": "scratchlm-inference",
            "readiness_status": svc_engine.readiness_status,
        }

    @app.get("/internal/readiness")
    def readiness(
        x_internal_service_token: str | None = Header(default=None),
    ) -> JSONResponse:
        verify_internal_token(x_internal_service_token)
        payload = svc_engine.readiness_payload()
        status_code = 200 if payload.get("readiness_status") == "READY" else 503
        return JSONResponse(status_code=status_code, content=payload)

    @app.post("/internal/v1/cancel/{run_id}")
    def cancel_run(
        run_id: str,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        verify_internal_token(x_internal_service_token)
        was_active_or_queued = svc_engine.gpu_supervisor.cancel_run(run_id)
        return {
            "run_id": run_id,
            "cancel_signaled": True,
            "was_active_or_queued": was_active_or_queued,
        }

    @app.post("/internal/v1/admin/promote")
    def promote_model(
        req: PromoteModelRequest,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        verify_internal_token(x_internal_service_token)
        return svc_engine.promote_candidate_with_mlflow_guard(req)

    @app.post("/internal/v1/admin/gpu/drain-for-training")
    def drain_gpu_for_training(
        req: DrainGpuRequest,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        verify_internal_token(x_internal_service_token)
        try:
            return svc_engine.gpu_supervisor.drain_and_acquire_training_lease(
                req.training_job_id,
                training_pid=req.training_pid,
                training_create_time_utc=req.training_create_time_utc,
                drain_timeout_sec=req.drain_timeout_sec,
            )
        except GpuCapacityError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": exc.error_code, "message": str(exc)},
            ) from exc

    @app.post("/internal/v1/admin/gpu/release-training")
    def release_training_gpu(
        req: ReleaseGpuRequest,
        x_internal_service_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        verify_internal_token(x_internal_service_token)
        try:
            return svc_engine.gpu_supervisor.release_training_lease_to_inference(
                req.training_job_id,
                respawn_child_worker=req.respawn_child_worker,
            )
        except GpuCapacityError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error_code": exc.error_code, "message": str(exc)},
            ) from exc

    @app.post("/internal/v1/generate")
    def generate(
        req: InternalGenerateRequestModel,
        x_internal_service_token: str | None = Header(default=None),
    ) -> StreamingResponse:
        verify_internal_token(x_internal_service_token)

        if svc_engine.readiness_status != "READY" or svc_engine.runtime is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "MODEL_PACKAGE_CORRUPTED",
                    "message": svc_engine.readiness_error or "Model runtime is not READY.",
                },
            )

        # Prompt marker for immediate upstream 503 fault injection
        if "[FAULT:UPSTREAM_503]" in req.prompt or (
            req.fault_injection is not None
            and req.fault_injection.fail_after_chunks == 0
            and req.fault_injection.error_code == "UPSTREAM_503"
        ):
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "UPSTREAM_503",
                    "message": "Simulated upstream 503 service unavailable before streaming.",
                },
            )

        runtime = svc_engine.runtime
        history_turns = (
            [ConversationTurn(role=t.role, content=t.content) for t in req.history]
            if req.history
            else None
        )
        effective_limit = min(int(req.context_limit), runtime.adapter.model_config.max_seq_len)

        try:
            packed = runtime.context_packer.pack_prompt(
                req.prompt,
                max_new_tokens=req.sampling.max_new_tokens,
                context_limit=effective_limit,
                history=history_turns,
                apply_chat_template=req.apply_chat_template,
            )
        except ContextWindowExceededError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": exc.error_code,
                    "message": str(exc),
                    "input_tokens": exc.input_tokens,
                    "reserved_output_tokens": exc.reserved_output_tokens,
                    "context_limit": exc.context_limit,
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "INVALID_PROMPT",
                    "message": str(exc),
                },
            ) from exc

        try:
            svc_engine.gpu_supervisor.acquire_generation_slot(req.run_id, timeout_sec=15.0)
        except GpuCapacityError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": exc.error_code,
                    "message": str(exc),
                },
            ) from exc

        def ndjson_generator() -> Iterator[bytes]:
            try:
                model = runtime.adapter.model
                tokenizer = runtime.adapter.tokenizer
                device = next(model.parameters()).device

                seed_val = int(req.sampling.seed if req.sampling.seed is not None else 1337)
                gen = torch.Generator(device=device).manual_seed(seed_val)
                stream_decoder = tokenizer.create_stream_decoder()

                kv_cache = (
                    ScratchKVCache(
                        n_layers=model.config.n_layers,
                        max_seq_len=model.config.max_seq_len,
                    )
                    if req.use_kv_cache
                    else None
                )

                seq: list[int] = list(packed.prompt_ids)
                input_tokens = packed.input_tokens
                output_tokens = 0
                chunk_index = 0
                accumulated_text: list[str] = []
                finish_reason = "length"

                fault = req.fault_injection or FaultInjectionModel()
                delay_ms = fault.chunk_delay_ms
                if "[SLOW_REAL]" in req.prompt or "[SLOW_FAKE]" in req.prompt:
                    delay_ms = max(delay_ms, 80)

                with torch.no_grad():
                    for step_idx in range(req.sampling.max_new_tokens):
                        # 1. Check cancellation
                        if svc_engine.gpu_supervisor.is_cancelled(req.run_id):
                            finish_reason = "cancelled"
                            done_record = {
                                "type": "done",
                                "run_id": req.run_id,
                                "attempt_token": req.attempt_token,
                                "model_revision": req.model_revision,
                                "finish_reason": "cancelled",
                                "output_text": "".join(accumulated_text),
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "error_code": "RUN_CANCELLED",
                                "error_message": "Generation cancelled by caller signal.",
                                "registry_version": runtime.registry_version,
                                "package_sha256": runtime.package_sha256,
                                "tokenizer_sha256": runtime.tokenizer_sha256,
                            }
                            yield (json.dumps(done_record) + "\n").encode("utf-8")
                            return

                        # 2. Check execution deadline
                        now_ms = int(time.time() * 1000)
                        if now_ms > req.deadline_epoch_ms:
                            done_record = {
                                "type": "done",
                                "run_id": req.run_id,
                                "attempt_token": req.attempt_token,
                                "model_revision": req.model_revision,
                                "finish_reason": "error",
                                "output_text": "".join(accumulated_text),
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "error_code": "EXECUTION_DEADLINE_EXCEEDED",
                                "error_message": "Generation exceeded deadline_epoch_ms.",
                                "registry_version": runtime.registry_version,
                                "package_sha256": runtime.package_sha256,
                                "tokenizer_sha256": runtime.tokenizer_sha256,
                            }
                            yield (json.dumps(done_record) + "\n").encode("utf-8")
                            return

                        # 3. Check injected mid-stream failure
                        if "[FAULT:MALFORMED_NDJSON]" in req.prompt and step_idx == 1:
                            yield b"{malformed_json_line_day054\n"
                            return

                        if "[FAULT:DISCONNECT]" in req.prompt and step_idx == 1:
                            # Abruptly end stream without emitting a terminal "done" record
                            return

                        if fault.fail_after_chunks is not None and chunk_index >= fault.fail_after_chunks:
                            err_code = fault.error_code or "INJECTED_PROVIDER_FAILURE"
                            if err_code == "STREAM_DISCONNECTED":
                                # Simulate abrupt mid-stream socket close before terminal record
                                return
                            done_record = {
                                "type": "done",
                                "run_id": req.run_id,
                                "attempt_token": req.attempt_token,
                                "model_revision": req.model_revision,
                                "finish_reason": "error",
                                "output_text": "".join(accumulated_text),
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "error_code": err_code,
                                "error_message": f"Fault injected after {chunk_index} chunks.",
                                "registry_version": runtime.registry_version,
                                "package_sha256": runtime.package_sha256,
                                "tokenizer_sha256": runtime.tokenizer_sha256,
                            }
                            yield (json.dumps(done_record) + "\n").encode("utf-8")
                            return

                        if delay_ms > 0:
                            time.sleep(delay_ms / 1000.0)

                        # 4. Forward pass (prefill on step_idx == 0, single-token KV decode on step_idx > 0)
                        if kv_cache is not None:
                            step_tokens = seq if step_idx == 0 else [seq[-1]]
                            inp = torch.tensor([step_tokens], dtype=torch.long, device=device)
                            logits, _ = model(inp, kv_cache=kv_cache)
                        else:
                            inp = torch.tensor([seq], dtype=torch.long, device=device)
                            logits, _ = model(inp)

                        next_id = sample_next_token(
                            logits[0, -1, :],
                            temperature=req.sampling.temperature,
                            top_k=req.sampling.top_k,
                            top_p=req.sampling.top_p,
                            generator=gen,
                        )
                        if next_id in (
                            EOS_TOKEN_ID,
                            runtime.adapter.chat_template.end_turn_token_id,
                        ):
                            finish_reason = "stop"
                            break

                        seq.append(next_id)
                        output_tokens += 1
                        delta_text = stream_decoder.step(next_id)
                        if delta_text:
                            chunk_index += 1
                            accumulated_text.append(delta_text)
                            delta_record = {
                                "type": "delta",
                                "run_id": req.run_id,
                                "attempt_token": req.attempt_token,
                                "chunk_index": chunk_index,
                                "text": delta_text,
                                "emitted_tokens": 1,
                            }
                            yield (json.dumps(delta_record) + "\n").encode("utf-8")

                    tail_text = stream_decoder.flush()
                    if tail_text:
                        chunk_index += 1
                        accumulated_text.append(tail_text)
                        delta_record = {
                            "type": "delta",
                            "run_id": req.run_id,
                            "attempt_token": req.attempt_token,
                            "chunk_index": chunk_index,
                            "text": tail_text,
                            "emitted_tokens": 0,
                        }
                        yield (json.dumps(delta_record) + "\n").encode("utf-8")

                    done_record = {
                        "type": "done",
                        "run_id": req.run_id,
                        "attempt_token": req.attempt_token,
                        "model_revision": req.model_revision,
                        "finish_reason": finish_reason,
                        "output_text": "".join(accumulated_text),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "error_code": None,
                        "error_message": None,
                        "registry_version": runtime.registry_version,
                        "package_sha256": runtime.package_sha256,
                        "tokenizer_sha256": runtime.tokenizer_sha256,
                    }
                    yield (json.dumps(done_record) + "\n").encode("utf-8")
            finally:
                svc_engine.gpu_supervisor.release_generation_slot(req.run_id)

        return StreamingResponse(ndjson_generator(), media_type="application/x-ndjson")

    return app


app = create_inference_app()
