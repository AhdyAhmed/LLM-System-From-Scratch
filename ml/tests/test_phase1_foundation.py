"""Phase 1 (Day001-Day010) foundation tests for the Python ML workspace and shared contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

from scratchlm.config import ModelConfig, load_model_config
from scratchlm.contracts import validate_against_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
CONTRACTS_DIR = REPO_ROOT / "contracts"


@pytest.mark.parametrize(
    "subpackage",
    [
        "scratchlm",
        "scratchlm.data",
        "scratchlm.tokenizer",
        "scratchlm.model",
        "scratchlm.training",
        "scratchlm.evaluation",
        "scratchlm.inference",
        "scratchlm.serving",
        "scratchlm.retrieval",
    ],
)
def test_all_subpackages_import_cleanly(subpackage: str) -> None:
    module = importlib.import_module(subpackage)
    assert module is not None


def test_tiny_and_local_model_configs_match_exact_parameter_counts() -> None:
    tiny_cfg = load_model_config(ML_ROOT / "configs" / "tiny.json")
    assert tiny_cfg.profile_name == "tiny"
    assert tiny_cfg.d_head == 32
    assert tiny_cfg.exact_parameter_count() == 843_520

    local_cfg = load_model_config(ML_ROOT / "configs" / "local.json")
    assert local_cfg.profile_name == "local"
    assert local_cfg.d_head == 32
    assert local_cfg.exact_parameter_count() == 3_111_936


def test_model_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(
            profile_name="bad",
            vocab_size=264,
            max_seq_len=128,
            n_layers=2,
            n_heads=5,
            d_model=64,
            d_ff=256,
            dropout=0.0,
            use_bias=True,
            tie_word_embeddings=True,
            layer_norm_eps=1e-5,
        )


def test_shared_contracts_and_fixtures_validate_in_python() -> None:
    openapi_path = CONTRACTS_DIR / "openapi.yaml"
    openapi_doc = yaml.safe_load(openapi_path.read_text(encoding="utf-8"))
    assert openapi_doc["openapi"] == "3.1.0"
    assert "/api/v1/conversations/{conversationId}/runs" in openapi_doc["paths"]

    internal_schema = CONTRACTS_DIR / "internal-inference.schema.json"
    internal_req = json.loads(
        (CONTRACTS_DIR / "fixtures" / "internal-generate-request.json").read_text(encoding="utf-8")
    )
    validate_against_schema(internal_req, internal_schema)

    sse_schema = CONTRACTS_DIR / "sse-events.schema.json"
    sse_lines = (
        (CONTRACTS_DIR / "fixtures" / "sse-stream-sample.jsonl")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(sse_lines) == 5
    events = [json.loads(line) for line in sse_lines]
    for ev in events:
        validate_against_schema(ev, sse_schema)

    ids = [ev["id"] for ev in events]
    assert ids == [1, 2, 3, 4, 5]
    assert events[0]["event"] == "run.queued"
    assert events[-1]["event"] == "run.completed"
