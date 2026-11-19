"""
Day066, Day067, Day068 & Day069: Phase 7 Python Client & Reliability Tests.
Verifies:
- `ScratchLlmClient.stream_run_events` and `stream_run_with_resync_fallback` handling of
  `resync.required` control frames (both `CURSOR_EXPIRED` and `SLOW_READER_BUFFER_OVERFLOW`),
  closing the SSE stream cleanly and fetching the authoritative durable run status snapshot
  from `GET /api/v1/runs/{run_id}`.
- Client methods for durable model promotion (`POST /api/v1/models/promotions`) and
  DLQ inspection/manual replay (`GET /api/v1/admin/dlq`, `POST /api/v1/admin/dlq/{event_id}/replay`).
"""

from __future__ import annotations

import json
import httpx
from scratchlm.client import ScratchLlmClient


def test_client_handles_resync_required_expired_cursor_by_fetching_snapshot() -> None:
    run_id = "d1111111-2222-4333-8444-555555555599"
    snapshot_calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/csrf":
            return httpx.Response(
                200,
                json={"header_name": "X-CSRF-TOKEN", "token": "csrf-phase7-token"},
            )
        if request.url.path == f"/api/v1/runs/{run_id}/events":
            assert request.headers.get("Last-Event-ID") == "4"
            sse_payload = (
                "event: resync.required\n"
                + "data: "
                + json.dumps(
                    {
                        "run_id": run_id,
                        "schema_version": 1,
                        "reason": "CURSOR_EXPIRED",
                        "last_event_id": 4,
                        "snapshot_url": f"/api/v1/runs/{run_id}",
                    }
                )
                + "\n\n"
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_payload.encode("utf-8"),
            )
        if request.url.path == f"/api/v1/runs/{run_id}":
            snapshot_calls.append(request.url.path)
            return httpx.Response(
                200,
                json={
                    "run_id": run_id,
                    "conversation_id": "c1111111-2222-4333-8444-555555555599",
                    "status": "COMPLETED",
                    "model_revision": "scratch-tiny-r1",
                    "finish_reason": "stop",
                    "output_text": "Recovered durable output from PostgreSQL snapshot.",
                    "reserved_tokens": 20,
                    "input_tokens": 4,
                    "output_tokens": 8,
                    "error_code": None,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with ScratchLlmClient(transport=transport) as client:
        res = client.stream_run_with_resync_fallback(run_id=run_id, last_event_id=4)
        assert res["resynced"] is True
        assert res["reason"] == "CURSOR_EXPIRED"
        assert res["snapshot"]["status"] == "COMPLETED"
        assert (
            res["snapshot"]["output_text"]
            == "Recovered durable output from PostgreSQL snapshot."
        )
        assert len(snapshot_calls) == 1


def test_client_handles_slow_reader_buffer_overflow_resync_event() -> None:
    run_id = "d1111111-2222-4333-8444-555555555588"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/api/v1/runs/{run_id}/events":
            sse_payload = (
                "id: 1\n"
                "event: run.queued\n"
                f'data: {{"run_id":"{run_id}","schema_version":1}}\n\n'
                "event: resync.required\n"
                f'data: {{"run_id":"{run_id}","schema_version":1,"reason":"SLOW_READER_BUFFER_OVERFLOW","last_event_id":1,"snapshot_url":"/api/v1/runs/{run_id}"}}\n\n'
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_payload.encode("utf-8"),
            )
        if request.url.path == f"/api/v1/runs/{run_id}":
            return httpx.Response(
                200,
                json={
                    "run_id": run_id,
                    "status": "RUNNING",
                    "model_revision": "scratch-tiny-r1",
                    "output_text": "Partial coalesced text",
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with ScratchLlmClient(transport=transport) as client:
        res = client.stream_run_with_resync_fallback(run_id=run_id)
        assert res["resynced"] is True
        assert res["reason"] == "SLOW_READER_BUFFER_OVERFLOW"
        assert res["snapshot"]["status"] == "RUNNING"
        assert len(res["events"]) == 2
        assert res["events"][-1].event == "resync.required"


def test_client_model_promotion_and_dlq_replay_contracts() -> None:
    event_id = "e1111111-2222-4333-8444-555555555577"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/csrf":
            return httpx.Response(
                200,
                json={"header_name": "X-CSRF-TOKEN", "token": "csrf-admin-token"},
            )
        if request.url.path == "/api/v1/models/promotions" and request.method == "POST":
            body = json.loads(request.content.decode("utf-8"))
            assert body["model_alias"] == "scratch-tiny"
            assert body["target_revision_tag"] == "scratch-tiny-r2"
            assert body["release_generation"] == 2
            return httpx.Response(
                200,
                json={
                    "promotion_id": "a1111111-2222-4333-8444-555555555566",
                    "model_alias": "scratch-tiny",
                    "release_generation": 2,
                    "target_revision_tag": "scratch-tiny-r2",
                    "previous_revision_tag": "scratch-tiny-r1",
                    "status": "COMMITTED",
                },
            )
        if request.url.path == "/api/v1/admin/dlq" and request.method == "GET":
            return httpx.Response(
                200,
                json={"items": [{"eventId": event_id, "status": "QUARANTINED"}]},
            )
        if (
            request.url.path == f"/api/v1/admin/dlq/{event_id}/replay"
            and request.method == "POST"
        ):
            return httpx.Response(
                200,
                json={
                    "event_id": event_id,
                    "outcome": "SKIPPED_TERMINAL_RUN",
                    "run_status": "COMPLETED",
                    "work_item_created": False,
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    with ScratchLlmClient(transport=transport) as client:
        promo = client.promote_model(
            model_alias="scratch-tiny",
            target_revision_tag="scratch-tiny-r2",
            release_generation=2,
        )
        assert promo["status"] == "COMMITTED"
        assert promo["release_generation"] == 2

        dlq_list = client.list_dlq_events()
        assert len(dlq_list["items"]) == 1

        replay = client.replay_dlq_event(event_id)
        assert replay["outcome"] == "SKIPPED_TERMINAL_RUN"
        assert replay["work_item_created"] is False
