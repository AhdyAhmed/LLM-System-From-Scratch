"""
Day015 & Day018: Headless Python httpx client for the ScratchLLM API.
Manages session cookies, X-CSRF-TOKEN headers, conversation CRUD, keyset pagination,
idempotent run submission, incremental UTF-8 SSE streaming, event IDs, cancellation,
and bounded reconnects without ever placing cookies or tokens in URLs.
"""

from __future__ import annotations

import codecs
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import httpx

TERMINAL_SSE_EVENTS = frozenset({"run.completed", "run.cancelled", "run.failed"})


@dataclass(frozen=True)
class SseEvent:
    """Parsed Server-Sent Event frame with sequence ID, event name, and JSON payload."""

    id: int | None
    event: str
    data: dict[str, Any]


def parse_sse_byte_chunks(byte_chunks: Iterator[bytes]) -> Iterator[SseEvent]:
    """
    Incrementally decode raw byte chunks into UTF-8 using a stateful incremental decoder
    (so multi-byte UTF-8 characters split across TCP frames are never corrupted)
    and yield structured SseEvent objects.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buffer = ""

    for chunk in byte_chunks:
        buffer += decoder.decode(chunk, final=False)
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            parsed = _parse_single_sse_frame(frame)
            if parsed is not None:
                yield parsed

    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        parsed = _parse_single_sse_frame(buffer)
        if parsed is not None:
            yield parsed


def _parse_single_sse_frame(frame: str) -> SseEvent | None:
    event_id: int | None = None
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in frame.splitlines():
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            # Heartbeat comment or blank line
            continue
        if line.startswith("id:"):
            raw_id = line[3:].strip()
            if raw_id.isdigit():
                event_id = int(raw_id)
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].strip())

    if not data_lines:
        return None

    joined_data = "\n".join(data_lines)
    payload = json.loads(joined_data)
    return SseEvent(id=event_id, event=event_name, data=payload)


class ScratchLlmClient:
    """
    Cookie-jar + CSRF-aware headless API client for ScratchLLM.
    Cookies remain strictly in the HTTP cookie jar and are never appended to URLs.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )
        self._csrf_header_name = "X-CSRF-TOKEN"
        self._csrf_token: str | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ScratchLlmClient:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def bootstrap_csrf(self) -> str:
        resp = self._client.get("/auth/csrf")
        resp.raise_for_status()
        body = resp.json()
        self._csrf_header_name = str(body.get("header_name", "X-CSRF-TOKEN"))
        self._csrf_token = str(body["token"])
        return self._csrf_token

    def _csrf_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        if not self._csrf_token:
            self.bootstrap_csrf()
        headers = {self._csrf_header_name: str(self._csrf_token)}
        if extra:
            headers.update(extra)
        return headers

    def register(self, email: str, password: str, workspace_name: str) -> dict[str, Any]:
        resp = self._client.post(
            "/auth/register",
            headers=self._csrf_headers(),
            json={
                "email": email,
                "password": password,
                "workspace_name": workspace_name,
            },
        )
        resp.raise_for_status()
        return dict(resp.json())

    def login(self, email: str, password: str) -> dict[str, Any]:
        resp = self._client.post(
            "/auth/login",
            headers=self._csrf_headers(),
            json={"email": email, "password": password},
        )
        resp.raise_for_status()
        # Refresh CSRF token after session fixation rotation
        self.bootstrap_csrf()
        return dict(resp.json())

    def logout(self) -> None:
        resp = self._client.post("/auth/logout", headers=self._csrf_headers())
        resp.raise_for_status()
        self._csrf_token = None

    def me(self) -> dict[str, Any]:
        resp = self._client.get("/api/v1/me")
        resp.raise_for_status()
        return dict(resp.json())

    def list_models(self) -> dict[str, Any]:
        resp = self._client.get("/api/v1/models")
        resp.raise_for_status()
        return dict(resp.json())


    def create_conversation(self, workspace_id: str, title: str) -> dict[str, Any]:
        resp = self._client.post(
            f"/api/v1/workspaces/{workspace_id}/conversations",
            headers=self._csrf_headers(),
            json={"title": title},
        )
        resp.raise_for_status()
        return dict(resp.json())

    def list_conversations(
        self, workspace_id: str, cursor: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = self._client.get(
            f"/api/v1/workspaces/{workspace_id}/conversations",
            params=params,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def rename_conversation(
        self, conversation_id: str, title: str, version: int
    ) -> dict[str, Any]:
        resp = self._client.patch(
            f"/api/v1/conversations/{conversation_id}",
            headers=self._csrf_headers(),
            json={"title": title, "version": version},
        )
        resp.raise_for_status()
        return dict(resp.json())

    def delete_conversation(self, conversation_id: str) -> None:
        resp = self._client.delete(
            f"/api/v1/conversations/{conversation_id}",
            headers=self._csrf_headers(),
        )
        resp.raise_for_status()

    def list_messages(
        self, conversation_id: str, cursor: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = self._client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            params=params,
        )
        resp.raise_for_status()
        return dict(resp.json())

    def submit_run(
        self,
        conversation_id: str,
        prompt: str,
        idempotency_key: str,
        model_alias: str = "fake-deterministic",
        max_new_tokens: int = 24,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        resp = self._client.post(
            f"/api/v1/conversations/{conversation_id}/runs",
            headers=self._csrf_headers({"Idempotency-Key": idempotency_key}),
            json={
                "prompt": prompt,
                "model_alias": model_alias,
                "sampling": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "seed": seed,
                },
            },
        )
        resp.raise_for_status()
        return dict(resp.json())

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/api/v1/runs/{run_id}")
        resp.raise_for_status()
        return dict(resp.json())

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        resp = self._client.post(
            f"/api/v1/runs/{run_id}/cancel",
            headers=self._csrf_headers(),
        )
        resp.raise_for_status()
        return dict(resp.json())

    def stream_run_events(
        self,
        run_id: str,
        last_event_id: int | None = None,
        max_reconnects: int = 2,
        on_delta: Callable[[str], None] | None = None,
    ) -> list[SseEvent]:
        """
        Stream SSE events for a run with incremental UTF-8 decoding, Last-Event-ID tracking,
        and bounded reconnect attempts until a terminal event is received.
        """
        collected: list[SseEvent] = []
        cursor = last_event_id
        attempts = 0

        while attempts <= max_reconnects:
            headers = {"Accept": "text/event-stream"}
            if cursor is not None and cursor >= 0:
                headers["Last-Event-ID"] = str(cursor)

            try:
                with self._client.stream(
                    "GET", f"/api/v1/runs/{run_id}/events", headers=headers
                ) as response:
                    response.raise_for_status()
                    for ev in parse_sse_byte_chunks(response.iter_bytes()):
                        if cursor is not None and ev.id is not None and ev.id <= cursor:
                            continue
                        if ev.id is not None:
                            cursor = ev.id
                        collected.append(ev)
                        if ev.event == "text.delta" and on_delta is not None:
                            on_delta(str(ev.data.get("text", "")))
                        if ev.event in TERMINAL_SSE_EVENTS:
                            return collected
                # Stream ended cleanly
                if collected and collected[-1].event in TERMINAL_SSE_EVENTS:
                    return collected
                attempts += 1
            except httpx.TransportError:
                attempts += 1
                if attempts > max_reconnects:
                    raise

        return collected
