"""
Phase 2 (Day011-Day020, Gate G2) tests for the Python httpx client, incremental UTF-8 SSE parser,
Postman collection, and Kafka event schema contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from scratchlm.client import ScratchLlmClient, parse_sse_byte_chunks
from scratchlm.contracts import validate_against_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_DIR = REPO_ROOT / "contracts"
KAFKA_SCHEMA_PATH = CONTRACTS_DIR / "kafka-events.schema.json"
KAFKA_FIXTURES_DIR = CONTRACTS_DIR / "fixtures" / "kafka"
POSTMAN_COLLECTION_PATH = CONTRACTS_DIR / "postman" / "scratchllm-phase2.postman_collection.json"


@pytest.mark.parametrize(
    "fixture_name,expected_type",
    [
        ("generation-requested-v1.json", "generation.requested.v1"),
        ("document-ingestion-requested-v1.json", "document.ingestion.requested.v1"),
        ("model-released-v1.json", "model.released.v1"),
    ],
)
def test_kafka_golden_fixtures_validate_against_schema(
    fixture_name: str, expected_type: str
) -> None:
    payload = json.loads((KAFKA_FIXTURES_DIR / fixture_name).read_text(encoding="utf-8"))
    assert payload["event_type"] == expected_type
    assert payload["schema_version"] == 1
    validate_against_schema(payload, KAFKA_SCHEMA_PATH)


def test_kafka_schema_rejects_heavy_bulk_fields() -> None:
    valid_gen = json.loads(
        (KAFKA_FIXTURES_DIR / "generation-requested-v1.json").read_text(encoding="utf-8")
    )
    invalid_with_prompt = dict(valid_gen)
    invalid_with_prompt["prompt"] = "Forbidden raw prompt text in Kafka"

    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(invalid_with_prompt, KAFKA_SCHEMA_PATH)


def test_postman_collection_is_valid_and_keeps_tokens_out_of_urls() -> None:
    collection = json.loads(POSTMAN_COLLECTION_PATH.read_text(encoding="utf-8"))
    assert collection["info"]["name"] == "ScratchLLM Phase 2 Headless API Collection"
    items = collection["item"]
    assert len(items) >= 10
    for item in items:
        raw_url = item["request"]["url"]["raw"]
        assert "SESSION=" not in raw_url
        assert "csrf" not in raw_url.lower() or raw_url.endswith("/auth/csrf")


def test_incremental_utf8_sse_parser_handles_split_multibyte_codepoints() -> None:
    # Construct an SSE frame with 2-byte Arabic ("سلام") and 4-byte emoji ("🤖")
    frame = (
        'id: 3\n'
        'event: text.delta\n'
        'data: {"run_id":"11111111-2222-4333-8444-555555555555","schema_version":1,"text":"سلام 🤖"}\n\n'
    ).encode("utf-8")

    # Deliberately split the UTF-8 byte sequence mid-character across 3 byte chunks
    split_idx_1 = frame.index("سلام".encode("utf-8")) + 1
    split_idx_2 = frame.index("🤖".encode("utf-8")) + 2
    chunks = [frame[:split_idx_1], frame[split_idx_1:split_idx_2], frame[split_idx_2:]]

    events = list(parse_sse_byte_chunks(iter(chunks)))
    assert len(events) == 1
    assert events[0].id == 3
    assert events[0].event == "text.delta"
    assert events[0].data["text"] == "سلام 🤖"


def test_scratchllm_httpx_client_csrf_conversations_and_sse_reconnect() -> None:
    recorded_requests: list[httpx.Request] = []
    sse_call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sse_call_count
        recorded_requests.append(request)
        path = request.url.path

        # Ensure cookies or tokens never leak into query params
        assert "SESSION" not in str(request.url.query)
        assert "csrf" not in str(request.url.query).lower()

        if path == "/auth/csrf":
            return httpx.Response(
                200,
                headers={"Set-Cookie": "XSRF-TOKEN=csrf-tok-123; Path=/"},
                json={
                    "header_name": "X-CSRF-TOKEN",
                    "parameter_name": "_csrf",
                    "token": "csrf-tok-123",
                },
            )

        if request.method in {"POST", "PATCH", "DELETE"}:
            assert request.headers.get("X-CSRF-TOKEN") == "csrf-tok-123"

        if path == "/auth/login":
            return httpx.Response(
                200,
                headers={"Set-Cookie": "SESSION=sess-abc; Path=/; HttpOnly; SameSite=Lax"},
                json={
                    "user_id": "11111111-1111-4111-8111-111111111111",
                    "email": "alice@example.com",
                    "workspaces": [
                        {
                            "workspace_id": "22222222-2222-4222-8222-222222222222",
                            "name": "Alice Lab",
                            "role": "OWNER",
                        }
                    ],
                },
            )

        if path == "/api/v1/workspaces/22222222-2222-4222-8222-222222222222/conversations" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "conversation_id": "33333333-3333-4333-8333-333333333333",
                    "workspace_id": "22222222-2222-4222-8222-222222222222",
                    "owner_user_id": "11111111-1111-4111-8111-111111111111",
                    "title": "Test Chat",
                    "version": 0,
                    "updated_at": "2026-10-16T05:00:00Z",
                },
            )

        if path == "/api/v1/conversations/33333333-3333-4333-8333-333333333333/runs":
            assert request.headers.get("Idempotency-Key") == "idem-key-0001"
            return httpx.Response(
                202,
                json={
                    "run_id": "44444444-4444-4444-8444-444444444444",
                    "conversation_id": "33333333-3333-4333-8333-333333333333",
                    "status": "QUEUED",
                    "model_revision": "fake-deterministic-r1",
                    "events_url": "/api/v1/runs/44444444-4444-4444-8444-444444444444/events",
                    "status_url": "/api/v1/runs/44444444-4444-4444-8444-444444444444",
                },
            )

        if path == "/api/v1/runs/44444444-4444-4444-8444-444444444444/events":
            sse_call_count += 1
            if sse_call_count == 1:
                # First stream connection drops after event 2 before terminal event
                body = (
                    'id: 1\nevent: run.queued\ndata: {"run_id":"44444444-4444-4444-8444-444444444444","schema_version":1}\n\n'
                    'id: 2\nevent: text.delta\ndata: {"run_id":"44444444-4444-4444-8444-444444444444","schema_version":1,"text":"Hello "}\n\n'
                )
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=body.encode("utf-8"),
                )
            else:
                # Reconnect sends Last-Event-ID: 2 and receives remaining events
                assert request.headers.get("Last-Event-ID") == "2"
                body = (
                    'id: 3\nevent: text.delta\ndata: {"run_id":"44444444-4444-4444-8444-444444444444","schema_version":1,"text":"world!"}\n\n'
                    'id: 4\nevent: run.completed\ndata: {"run_id":"44444444-4444-4444-8444-444444444444","schema_version":1,"finish_reason":"stop","input_tokens":5,"output_tokens":2}\n\n'
                )
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=body.encode("utf-8"),
                )

        if path == "/auth/logout":
            return httpx.Response(204)

        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    deltas: list[str] = []

    with ScratchLlmClient(transport=transport) as client:
        profile = client.login("alice@example.com", "StrongPass#2026!")
        ws_id = profile["workspaces"][0]["workspace_id"]
        conv = client.create_conversation(ws_id, "Test Chat")
        run = client.submit_run(
            conv["conversation_id"],
            prompt="Say hello",
            idempotency_key="idem-key-0001",
        )
        events = client.stream_run_events(
            run["run_id"],
            max_reconnects=2,
            on_delta=deltas.append,
        )
        client.logout()

    assert "".join(deltas) == "Hello world!"
    assert [e.id for e in events] == [1, 2, 3, 4]
    assert events[-1].event == "run.completed"
    assert sse_call_count == 2
