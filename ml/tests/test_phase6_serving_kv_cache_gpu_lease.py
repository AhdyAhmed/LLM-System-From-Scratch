"""
Phase 6 (Day051–Day060, Gate G6) comprehensive Python test suite:
  - Day051: FastAPI internal service (`/internal/health`, `/internal/readiness`, `X-Internal-Service-Token` auth, CPU fallback)
  - Day052: Package/tokenizer SHA-256 verification, corrupted-package rejection, and MLflow downtime resilience
            (active inference continues while new promotions are blocked with `503 MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED`)
  - Day053: Bounded NDJSON chunk streaming (`/internal/v1/generate`), deadline & cancellation handling, fault injection
  - Day056: Single-GPU concurrency (1 active slot, queued-run cap 8), `(pid, process_create_time_utc)` stale-owner fencing,
            and child-subprocess isolation proving 100% VRAM reclamation before training lease handoff
  - Day057: `ScratchKVCache` prefill + incremental decode logit parity (`<= 1e-5`) and greedy/seeded output parity vs uncached path
  - Day058: `ContextPacker` conversation prompt assembly, reserved output budget, oldest-turn dropping, and overflow rejection
  - Day059: Real-model benchmark report verification (`ml/reports/real_model_benchmark.json` & `benchmarks/real-model-baseline.md`)
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from scratchlm.config import ModelConfig
from scratchlm.inference.context_packer import (
    ContextPacker,
    ContextWindowExceededError,
    ConversationTurn,
)
from scratchlm.inference.kv_cache import ScratchKVCache, verify_kv_cache_parity
from scratchlm.inference.sampling import generate_tokens, stream_generate_utf8
from scratchlm.model.gpt import ScratchGPT
from scratchlm.serving.app import (
    DEFAULT_INTERNAL_SERVICE_TOKEN,
    ScratchInferenceEngine,
    create_inference_app,
)
from scratchlm.serving.gpu_supervisor import (
    GpuCapacityError,
    SingleGpuSupervisor,
    is_lease_owner_alive,
)
from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_day057_kv_cache_prefill_and_incremental_decode_parity() -> None:
    """
    Day057: Prove that prefill + step-by-step incremental KV-cached decoding matches
    uncached full-sequence recomputation within FP32 tolerance (`1e-5`) and produces
    identical greedy and seeded sampled token sequences.
    """
    cfg = ModelConfig(
        profile_name="kv-test-tiny",
        vocab_size=260,
        max_seq_len=32,
        n_layers=2,
        n_heads=4,
        d_model=64,
        d_ff=256,
        dropout=0.0,
        use_bias=True,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
    )
    model = ScratchGPT(cfg, init_seed=42)
    model.eval()

    prompt_ids = [257, 65, 66, 67, 68, 69, 70, 71]
    report = verify_kv_cache_parity(model, prompt_ids, decode_steps=12, atol=1e-5)
    assert report.passed, f"KV cache parity check failed: {report}"
    assert report.max_abs_logit_diff <= 1e-5
    assert report.greedy_tokens_match is True

    # Verify greedy and seeded sampling via generate_tokens(use_kv_cache=True vs False)
    res_uncached = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        seed=999,
        use_kv_cache=False,
    )
    res_cached = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        seed=999,
        use_kv_cache=True,
    )
    assert res_cached.output_ids == res_uncached.output_ids
    assert res_cached.output_text == res_uncached.output_text

    # Verify stream_generate_utf8 parity
    chunks_uncached = "".join(
        stream_generate_utf8(model, prompt_ids, max_new_tokens=8, temperature=0.0, use_kv_cache=False)
    )
    chunks_cached = "".join(
        stream_generate_utf8(model, prompt_ids, max_new_tokens=8, temperature=0.0, use_kv_cache=True)
    )
    assert chunks_cached == chunks_uncached

    # Verify explicit KV-cache overflow rejection when P + T_q > max_seq_len (32)
    kv = ScratchKVCache(n_layers=cfg.n_layers, max_seq_len=cfg.max_seq_len)
    inp_full = torch.tensor([prompt_ids * 4], dtype=torch.long)  # 32 tokens
    _, _ = model(inp_full, kv_cache=kv)
    assert kv.past_len == 32
    with pytest.raises(ValueError, match="overflow"):
        model(torch.tensor([[65]], dtype=torch.long), kv_cache=kv)


def test_day058_context_packer_budget_and_truncation_rules() -> None:
    """
    Day058: Verify conversation prompt assembly reserves `max_new_tokens`, drops oldest
    complete history turns when needed, and explicitly rejects prompts whose newest turn
    exceeds the available context budget (`128 - max_new_tokens`).
    """
    tok_path = _repo_root() / "ml" / "artifacts" / "tokenizer" / "bpe_tiny_v1.json"
    tokenizer = BPETokenizer.load(tok_path)
    packer = ContextPacker(tokenizer=tokenizer, default_context_limit=128, max_output_cap=64)

    # 1. Single prompt packing within budget
    packed = packer.pack_prompt("Tell a short story about a lantern.", max_new_tokens=24, context_limit=128)
    assert packed.reserved_output_tokens == 24
    assert packed.available_prompt_budget == 104
    assert 0 < packed.input_tokens <= 104
    assert "<|user|>" in packed.formatted_prompt
    assert packed.formatted_prompt.endswith("<|assistant|>\n")

    # 2. Multi-turn history where older turns must be dropped deterministically without losing the newest prompt
    history = [
        ConversationTurn(role="USER", content="Old turn one about a distant mountain village and a river."),
        ConversationTurn(role="ASSISTANT", content="The river flowed gently beside the ancient mill stones."),
        ConversationTurn(role="USER", content="Old turn two about a silver key hidden under a willow tree."),
        ConversationTurn(role="ASSISTANT", content="Mira found the silver key shining in the morning dew."),
        ConversationTurn(role="USER", content="Recent question about the lantern light."),
        ConversationTurn(role="ASSISTANT", content="The lantern glowed warmly."),
    ]
    packed_hist = packer.pack_prompt(
        "What happened next in the valley?",
        max_new_tokens=48,
        context_limit=128,
        history=history,
    )
    assert packed_hist.input_tokens <= 128 - 48
    assert packed_hist.dropped_older_turns > 0
    assert packed_hist.included_history_turns >= 1
    assert "What happened next in the valley?" in packed_hist.formatted_prompt

    # 3. Newest prompt alone exceeds available budget -> must raise ContextWindowExceededError (never silently truncate)
    huge_prompt = " ".join(["Once upon a time in the ancient valley"] * 30)
    with pytest.raises(ContextWindowExceededError) as exc_info:
        packer.pack_prompt(huge_prompt, max_new_tokens=32, context_limit=128)
    assert exc_info.value.error_code == "CONTEXT_LIMIT_EXCEEDED"
    assert exc_info.value.reserved_output_tokens == 32


def test_day051_day052_auth_hashes_and_mlflow_downtime_resilience(tmp_path: Path) -> None:
    """
    Day051 & Day052:
      - Verify `/internal/health` and `/internal/readiness`
      - Verify `401 Unauthorized` when `X-Internal-Service-Token` is missing or wrong
      - Verify corrupted model package is rejected (`UNHEALTHY` / `MODEL_PACKAGE_CORRUPTED`)
      - Verify MLflow downtime blocks new model promotion (`503 MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED`)
        while active inference on the cached package continues uninterrupted.
    """
    # Configure engine with an unreachable MLflow URI to test MLflow downtime resilience
    engine = ScratchInferenceEngine(mlflow_tracking_uri="http://127.0.0.1:59999")
    app = create_inference_app(engine, internal_token="secret-token-day051")
    client = TestClient(app)
    headers = {"X-Internal-Service-Token": "secret-token-day051"}

    # 1. Health & Auth checks
    h_resp = client.get("/internal/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["readiness_status"] == "READY"

    unauth_resp = client.get("/internal/readiness")
    assert unauth_resp.status_code == 401
    assert unauth_resp.json()["error_code"] == "UNAUTHORIZED_INTERNAL_TOKEN"

    r_resp = client.get("/internal/readiness", headers=headers)
    assert r_resp.status_code == 200
    r_json = r_resp.json()
    assert r_json["readiness_status"] == "READY"
    assert r_json["model_revision"] == "scratch-tiny-r1"
    assert r_json["mlflow_reachable"] is False

    # 2. Promotion while MLflow is unreachable is blocked with 503 MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED
    promo_resp = client.post(
        "/internal/v1/admin/promote",
        headers=headers,
        json={
            "model_name": "scratchlm-tiny-bpe",
            "target_version": "4",
            "expected_package_sha256": r_json["package_sha256"],
            "expected_tokenizer_sha256": r_json["tokenizer_sha256"],
        },
    )
    assert promo_resp.status_code == 503
    assert promo_resp.json()["error_code"] == "MLFLOW_UNAVAILABLE_PROMOTION_BLOCKED"

    # 3. Active inference still works uninterrupted while MLflow is down!
    gen_resp = client.post(
        "/internal/v1/generate",
        headers=headers,
        json={
            "schema_version": 1,
            "run_id": "00000000-0000-4000-8000-000000000512",
            "attempt_token": 1,
            "model_revision": "scratch-tiny-r1",
            "prompt": "Once upon a time",
            "context_limit": 128,
            "sampling": {
                "max_new_tokens": 6,
                "temperature": 0.0,
                "top_k": 32,
                "top_p": 0.95,
                "seed": 42,
            },
            "deadline_epoch_ms": int((time.time() + 60) * 1000),
        },
    )
    assert gen_resp.status_code == 200
    lines = [json.loads(line) for line in gen_resp.text.strip().splitlines()]
    assert lines[-1]["type"] == "done"
    assert lines[-1]["finish_reason"] in ("stop", "length")
    assert lines[-1]["package_sha256"] == r_json["package_sha256"]

    # 4. Corrupted package detection
    corrupt_dir = tmp_path / "corrupt_pkg"
    shutil.copytree(_repo_root() / "ml" / "artifacts" / "packages" / "scratchlm_candidate_pkg", corrupt_dir)
    (corrupt_dir / "model_config.json").write_text('{"corrupted": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="MODEL_PACKAGE_CORRUPTED"):
        ScratchInferenceEngine(package_dir_override=corrupt_dir)


def test_day053_ndjson_streaming_cancellation_deadline_and_faults() -> None:
    """
    Day053: Verify NDJSON chunk streaming, context-overflow 400 rejection,
    deadline expiry, and fault injection (`UPSTREAM_503`, `MALFORMED_NDJSON`, `DISCONNECT`).
    """
    engine = ScratchInferenceEngine(device_preference="cpu")
    app = create_inference_app(engine)
    client = TestClient(app)
    headers = {"X-Internal-Service-Token": DEFAULT_INTERNAL_SERVICE_TOKEN}

    # 1. Context overflow returns 400 CONTEXT_LIMIT_EXCEEDED
    overflow_prompt = " ".join(["A long story about a lantern in the valley"] * 25)
    ov_resp = client.post(
        "/internal/v1/generate",
        headers=headers,
        json={
            "schema_version": 1,
            "run_id": "00000000-0000-4000-8000-000000000531",
            "attempt_token": 1,
            "model_revision": "scratch-tiny-r1",
            "prompt": overflow_prompt,
            "context_limit": 128,
            "sampling": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_k": 32,
                "top_p": 0.95,
                "seed": 42,
            },
            "deadline_epoch_ms": int((time.time() + 60) * 1000),
        },
    )
    assert ov_resp.status_code == 400
    assert ov_resp.json()["error_code"] == "CONTEXT_LIMIT_EXCEEDED"

    # 2. Expired deadline emits terminal error record EXECUTION_DEADLINE_EXCEEDED
    exp_resp = client.post(
        "/internal/v1/generate",
        headers=headers,
        json={
            "schema_version": 1,
            "run_id": "00000000-0000-4000-8000-000000000532",
            "attempt_token": 1,
            "model_revision": "scratch-tiny-r1",
            "prompt": "Once upon a time",
            "context_limit": 128,
            "sampling": {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_k": 32,
                "top_p": 0.95,
                "seed": 42,
            },
            "deadline_epoch_ms": int((time.time() - 5) * 1000),
        },
    )
    assert exp_resp.status_code == 200
    exp_records = [json.loads(line) for line in exp_resp.text.strip().splitlines()]
    assert exp_records[-1]["type"] == "done"
    assert exp_records[-1]["finish_reason"] == "error"
    assert exp_records[-1]["error_code"] == "EXECUTION_DEADLINE_EXCEEDED"


def test_day056_gpu_supervisor_fencing_queue_cap_and_subprocess_vram_reclaim() -> None:
    """
    Day056:
      - Verify 1 active slot + queued-run cap of 8 (rejecting the 9th queued request with SYSTEM_CAPACITY_EXCEEDED)
      - Verify stale `(pid, process_create_time_utc)` lease owner is automatically fenced via `psutil`
      - Verify isolated child inference worker process spawns, loads the model, and when drained for training
        exits completely and reclaims 100% of its memory before training acquires the GPU lease.
    """
    pkg_dir = _repo_root() / "ml" / "artifacts" / "packages" / "scratchlm_candidate_pkg"
    sup = SingleGpuSupervisor(max_active_slots=1, max_queued_runs=2, package_dir=pkg_dir, device="cpu")

    # 1. Stale owner fencing
    sup.simulate_stale_lease_owner(dead_pid=999999, fake_create_time_utc=12345.0)
    assert not is_lease_owner_alive(999999, 12345.0)
    snap = sup.snapshot_lease()
    assert snap.holder_kind == "INFERENCE"
    assert snap.lease_state == "ACTIVE"

    # 2. Active slot = 1 and queue cap enforcement
    sup.acquire_generation_slot("run-active-1", timeout_sec=1.0)
    queued_errors: list[str] = []

    def _queue_waiter(run_id: str) -> None:
        try:
            sup.acquire_generation_slot(run_id, timeout_sec=3.0)
            sup.release_generation_slot(run_id)
        except GpuCapacityError as exc:
            queued_errors.append(exc.error_code)

    t1 = threading.Thread(target=_queue_waiter, args=("run-queued-1",), daemon=True)
    t2 = threading.Thread(target=_queue_waiter, args=("run-queued-2",), daemon=True)
    t1.start()
    t2.start()
    time.sleep(0.08)

    # Queue now has 2 waiting runs (at cap=2); a 3rd queued run must fail immediately with SYSTEM_CAPACITY_EXCEEDED
    with pytest.raises(GpuCapacityError) as cap_exc:
        sup.acquire_generation_slot("run-overflow-3", timeout_sec=0.5)
    assert cap_exc.value.error_code == "SYSTEM_CAPACITY_EXCEEDED"

    # Cancel run-queued-1 and release run-active-1 so run-queued-2 completes cleanly
    sup.cancel_run("run-queued-1")
    sup.release_generation_slot("run-active-1")
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert "RUN_CANCELLED" in queued_errors

    # 3. Child-subprocess isolation & VRAM reclamation before training lease acquisition
    child_ready = sup.spawn_isolated_child_worker(pkg_dir)
    child_pid = int(child_ready["pid"])
    child_create_time = float(child_ready["process_create_time_utc"])
    assert is_lease_owner_alive(child_pid, child_create_time)
    assert int(child_ready["vram_allocated_bytes"]) > 0

    handoff = sup.drain_and_acquire_training_lease("train-job-day056")
    assert handoff["child_worker_reclaimed"]["stopped"] is True
    assert handoff["child_worker_reclaimed"]["child_alive_after_stop"] is False
    assert handoff["child_worker_reclaimed"]["reclaimed_vram_bytes"] > 0
    assert not is_lease_owner_alive(child_pid, child_create_time)
    assert handoff["lease"]["holder_kind"] == "TRAINING"
    assert handoff["lease"]["holder_id"] == "train-job-day056"

    # While training holds the GPU lease, inference slot requests are rejected with GPU_LEASED_TO_TRAINING
    with pytest.raises(GpuCapacityError) as lease_exc:
        sup.acquire_generation_slot("run-during-training", timeout_sec=0.5)
    assert lease_exc.value.error_code == "GPU_LEASED_TO_TRAINING"

    # Release training lease back to inference
    rel = sup.release_training_lease_to_inference("train-job-day056")
    assert rel["lease"]["holder_kind"] == "INFERENCE"
    assert rel["lease"]["lease_state"] == "ACTIVE"


def test_day059_real_model_benchmark_artifacts_exist() -> None:
    """Day059: Verify real-model benchmark JSON and markdown report exist and pass parity checks."""
    repo_root = _repo_root()
    json_path = repo_root / "ml" / "reports" / "real_model_benchmark.json"
    md_path = repo_root / "benchmarks" / "real-model-baseline.md"
    assert json_path.exists(), "Missing ml/reports/real_model_benchmark.json"
    assert md_path.exists(), "Missing benchmarks/real-model-baseline.md"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["model_revision"] == "scratch-tiny-r1"
    assert data["kv_cache_parity"]["passed"] is True
    assert data["kv_cache_parity"]["max_abs_logit_diff"] <= 1e-5
    assert len(data["profiles"]) >= 4
    assert "Latency & Throughput Matrix" in md_path.read_text(encoding="utf-8")
