"""
Day037, Day039 & Day040: Pluggable ExperimentTracker interface, InMemoryExperimentTracker,
and SpoolingMlflowRestTracker communicating with MLflow (--serve-artifacts over mlflow-artifacts:/)
with bounded local JSONL spooling during tracking-service outages.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import httpx


class TrackingSpoolOverflowError(RuntimeError):
    """Raised when the local tracking outage spool exceeds its configured byte cap."""


def validate_logical_artifact_uri(uri: str) -> None:
    """
    Cross-OS boundary rule (design.md Section 3):
    Artifact URIs must use the logical `mlflow-artifacts:/` scheme and never leak
    raw Windows drive paths (`C:\\`, `E:\\`) or container filesystem paths (`/opt/`, `/mlartifacts`).
    """
    if not uri.startswith("mlflow-artifacts:/"):
        raise ValueError(
            f"Invalid artifact URI '{uri}': must start with 'mlflow-artifacts:/' (host-path leakage is forbidden)"
        )
    if "\\" in uri or ":" in uri[len("mlflow-artifacts:") :]:
        raise ValueError(f"Artifact URI '{uri}' contains forbidden host path characters")


class ExperimentTracker(Protocol):
    """Small explicit tracking interface decoupling model/training math from MLflow transport."""

    def start_run(self, experiment_name: str, run_name: str, tags: dict[str, str] | None = None) -> str: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def log_artifact(self, local_path: Path, artifact_subpath: str = "") -> str: ...
    def end_run(self, status: str = "FINISHED") -> None: ...


class InMemoryExperimentTracker:
    """Deterministic in-memory tracker for unit testing training math without network calls."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.experiment_name: str | None = None
        self.run_name: str | None = None
        self.tags: dict[str, str] = {}
        self.params: dict[str, str] = {}
        self.metrics_history: list[tuple[int, dict[str, float]]] = []
        self.artifacts: list[tuple[str, str, str]] = []  # (logical_uri, local_name, sha256)
        self.status: str = "UNSTARTED"

    def start_run(self, experiment_name: str, run_name: str, tags: dict[str, str] | None = None) -> str:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = dict(tags or {})
        self.run_id = hashlib.sha256(f"{experiment_name}:{run_name}".encode("utf-8")).hexdigest()[:32]
        self.status = "RUNNING"
        return self.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            self.params[str(k)] = str(v)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics_history.append((step, {k: float(v) for k, v in metrics.items()}))

    def log_artifact(self, local_path: Path, artifact_subpath: str = "") -> str:
        if not self.run_id:
            raise RuntimeError("start_run must be called before log_artifact")
        sha = hashlib.sha256(local_path.read_bytes()).hexdigest()
        sub = f"{artifact_subpath.strip('/')}/{local_path.name}" if artifact_subpath else local_path.name
        logical_uri = f"mlflow-artifacts:/0/{self.run_id}/artifacts/{sub}"
        validate_logical_artifact_uri(logical_uri)
        self.artifacts.append((logical_uri, local_path.name, sha))
        return logical_uri

    def end_run(self, status: str = "FINISHED") -> None:
        self.status = status


@dataclass(frozen=True)
class SpooledTrackingEvent:
    """Record written to the bounded local spool when the MLflow server is temporarily unreachable."""

    timestamp_ms: int
    event_type: str  # "params", "metrics", "artifact", "end_run"
    run_id: str
    payload: dict[str, Any]


class SpoolingMlflowRestTracker:
    """
    Production MLflow 3.x REST & Artifact-Proxy Tracker (Day039-Day040):
    - Communicates with MLflow over HTTP (`/api/2.0/mlflow/...` and `/api/2.0/mlflow-artifacts/artifacts/...`).
    - Enforces `mlflow-artifacts:/` logical URIs so Windows native training and Linux Airflow tasks
      resolve the exact same artifact paths without host-path leakage.
    - Buffers tracking events to a bounded local JSONL spool (`max_spool_bytes`) if MLflow experiences
      a transient outage, and reconciles/flushes them automatically when connectivity returns.
    """

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5001",
        spool_dir: Path | None = None,
        max_spool_bytes: int = 4 * 1024 * 1024,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        self.spool_dir = spool_dir
        self.max_spool_bytes = max_spool_bytes
        self.timeout_seconds = timeout_seconds

        self.experiment_id: str | None = None
        self.run_id: str | None = None
        self.run_artifact_uri: str | None = None
        self.simulate_outage: bool = False

        if self.spool_dir is not None:
            self.spool_dir.mkdir(parents=True, exist_ok=True)
            self.spool_path: Path | None = self.spool_dir / "mlflow_tracking_spool.jsonl"
        else:
            self.spool_path = None

    def _append_to_spool(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.spool_path is None:
            raise RuntimeError(f"MLflow unreachable for {event_type} and no local spool_dir configured")
        event = SpooledTrackingEvent(
            timestamp_ms=int(time.time() * 1000),
            event_type=event_type,
            run_id=self.run_id or "pending",
            payload=payload,
        )
        line = json.dumps(asdict(event), sort_keys=True) + "\n"
        line_bytes = len(line.encode("utf-8"))
        current_size = self.spool_path.stat().st_size if self.spool_path.exists() else 0
        if current_size + line_bytes > self.max_spool_bytes:
            raise TrackingSpoolOverflowError(
                f"Local MLflow spool {self.spool_path} exceeded max_spool_bytes={self.max_spool_bytes}"
            )
        with self.spool_path.open("a", encoding="utf-8") as f:
            f.write(line)

    def pending_spool_count(self) -> int:
        if self.spool_path is None or not self.spool_path.exists():
            return 0
        content = self.spool_path.read_text(encoding="utf-8").strip()
        return len(content.splitlines()) if content else 0

    def _request(self, method: str, path: str, *, json_body: dict | None = None, content: bytes | None = None) -> httpx.Response:
        if self.simulate_outage:
            raise httpx.ConnectError("Simulated MLflow tracking service outage")
        url = f"{self.tracking_uri}{path}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.request(method, url, json=json_body, content=content)
            resp.raise_for_status()
            return resp

    def start_run(self, experiment_name: str, run_name: str, tags: dict[str, str] | None = None) -> str:
        # 1. Get or create experiment with mlflow-artifacts:/ root
        try:
            resp = self._request(
                "GET",
                f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={quote(experiment_name)}",
            )
            data = resp.json() if resp.text else {}
            self.experiment_id = str(data["experiment"]["experiment_id"])
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            create_resp = self._request(
                "POST",
                "/api/2.0/mlflow/experiments/create",
                json_body={
                    "name": experiment_name,
                    "artifact_location": f"mlflow-artifacts:/{quote(experiment_name)}",
                },
            )
            self.experiment_id = str(create_resp.json()["experiment_id"])

        tag_list = [{"key": str(k), "value": str(v)} for k, v in (tags or {}).items()]
        run_resp = self._request(
            "POST",
            "/api/2.0/mlflow/runs/create",
            json_body={
                "experiment_id": self.experiment_id,
                "run_name": run_name,
                "start_time": int(time.time() * 1000),
                "tags": tag_list,
            },
        )
        run_info = run_resp.json()["run"]["info"]
        self.run_id = str(run_info["run_id"])
        self.run_artifact_uri = str(run_info["artifact_uri"])
        validate_logical_artifact_uri(self.run_artifact_uri)
        return self.run_id

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.run_id:
            raise RuntimeError("start_run must be called before log_params")
        param_list = [{"key": str(k), "value": str(v)[:500]} for k, v in sorted(params.items())]
        try:
            self._request(
                "POST",
                "/api/2.0/mlflow/runs/log-batch",
                json_body={"run_id": self.run_id, "params": param_list},
            )
        except httpx.HTTPError:
            self._append_to_spool("params", {"params": param_list})

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        if not self.run_id:
            raise RuntimeError("start_run must be called before log_metrics")
        now_ms = int(time.time() * 1000)
        metric_list = [
            {"key": str(k), "value": float(v), "timestamp": now_ms, "step": int(step)}
            for k, v in sorted(metrics.items())
        ]
        try:
            self._request(
                "POST",
                "/api/2.0/mlflow/runs/log-batch",
                json_body={"run_id": self.run_id, "metrics": metric_list},
            )
        except httpx.HTTPError:
            self._append_to_spool("metrics", {"metrics": metric_list})

    def _artifact_proxy_relative_path(self, artifact_subpath: str, filename: str) -> tuple[str, str]:
        if not self.run_artifact_uri:
            raise RuntimeError("run_artifact_uri is not initialized")
        validate_logical_artifact_uri(self.run_artifact_uri)
        # Strip 'mlflow-artifacts:/' prefix to obtain the server-side proxy path
        prefix = self.run_artifact_uri[len("mlflow-artifacts:/") :].strip("/")
        sub = f"{artifact_subpath.strip('/')}/{filename}" if artifact_subpath else filename
        full_rel = f"{prefix}/{sub}" if prefix else sub
        logical_uri = f"mlflow-artifacts:/{full_rel}"
        validate_logical_artifact_uri(logical_uri)
        return full_rel, logical_uri

    def log_artifact(self, local_path: Path, artifact_subpath: str = "") -> str:
        if not self.run_id:
            raise RuntimeError("start_run must be called before log_artifact")
        rel_path, logical_uri = self._artifact_proxy_relative_path(artifact_subpath, local_path.name)
        raw_bytes = local_path.read_bytes()
        try:
            self._request(
                "PUT",
                f"/api/2.0/mlflow-artifacts/artifacts/{rel_path}",
                content=raw_bytes,
            )
        except httpx.HTTPError:
            self._append_to_spool(
                "artifact",
                {
                    "local_path": str(local_path.resolve()),
                    "rel_path": rel_path,
                    "logical_uri": logical_uri,
                },
            )
        return logical_uri

    def download_artifact_bytes(self, logical_uri: str) -> bytes:
        """
        Resolve and download an `mlflow-artifacts:/` URI over HTTP without any host-path dependency.
        Works identically from native Windows Python and Linux containers.
        """
        validate_logical_artifact_uri(logical_uri)
        rel_path = logical_uri[len("mlflow-artifacts:/") :].lstrip("/")
        resp = self._request("GET", f"/api/2.0/mlflow-artifacts/artifacts/{rel_path}")
        return resp.content

    def get_run(self, run_id: str | None = None) -> dict[str, Any]:
        target_id = run_id or self.run_id
        if not target_id:
            raise ValueError("run_id is required")
        resp = self._request("GET", f"/api/2.0/mlflow/runs/get?run_id={quote(target_id)}")
        return resp.json()["run"]

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.run_id:
            return
        try:
            self.flush_spool()
            self._request(
                "POST",
                "/api/2.0/mlflow/runs/update",
                json_body={
                    "run_id": self.run_id,
                    "status": status,
                    "end_time": int(time.time() * 1000),
                },
            )
        except httpx.HTTPError:
            self._append_to_spool("end_run", {"status": status, "end_time": int(time.time() * 1000)})

    def flush_spool(self) -> int:
        """Replay all locally spooled tracking events to MLflow once connectivity is restored."""
        if self.spool_path is None or not self.spool_path.exists():
            return 0
        lines = [ln for ln in self.spool_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return 0

        flushed = 0
        remaining: list[str] = []
        for idx, line in enumerate(lines):
            rec = SpooledTrackingEvent(**json.loads(line))
            try:
                if rec.event_type == "params":
                    self._request(
                        "POST",
                        "/api/2.0/mlflow/runs/log-batch",
                        json_body={"run_id": rec.run_id, "params": rec.payload["params"]},
                    )
                elif rec.event_type == "metrics":
                    self._request(
                        "POST",
                        "/api/2.0/mlflow/runs/log-batch",
                        json_body={"run_id": rec.run_id, "metrics": rec.payload["metrics"]},
                    )
                elif rec.event_type == "artifact":
                    raw_bytes = Path(rec.payload["local_path"]).read_bytes()
                    self._request(
                        "PUT",
                        f"/api/2.0/mlflow-artifacts/artifacts/{rec.payload['rel_path']}",
                        content=raw_bytes,
                    )
                elif rec.event_type == "end_run":
                    self._request(
                        "POST",
                        "/api/2.0/mlflow/runs/update",
                        json_body={
                            "run_id": rec.run_id,
                            "status": rec.payload["status"],
                            "end_time": rec.payload["end_time"],
                        },
                    )
                flushed += 1
            except httpx.HTTPError:
                remaining = lines[idx:]
                break

        if remaining:
            self.spool_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            self.spool_path.unlink(missing_ok=True)
        return flushed
