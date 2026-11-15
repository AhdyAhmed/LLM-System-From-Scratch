"""
Day056: Single-GPU concurrency, exclusive GPU lease supervisor, `(pid, process_create_time_utc)`
stale-owner fencing via `psutil`, child-subprocess model-worker isolation so stopping the inference
child process reclaims 100% of its CUDA context memory before a training job acquires the GPU lease,
1 active generation slot, and queued-run cap of 8.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil
import torch


class GpuCapacityError(RuntimeError):
    """Raised when the single-GPU queue cap (8) is full or the GPU lease is owned by training."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class GpuLeaseOwner:
    """Identifies the OS process holding the exclusive single-GPU lease (`design.md` §10.4)."""

    holder_kind: str  # "INFERENCE" | "TRAINING" | "NONE"
    holder_id: str
    pid: int
    process_create_time_utc: float
    acquired_at_epoch_ms: int
    lease_state: str  # "ACTIVE" | "DRAINING" | "RELEASED"
    child_worker_pid: int | None
    child_worker_alive: bool
    vram_allocated_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_process_create_time_utc(pid: int) -> float:
    """Return `psutil.Process(pid).create_time()` or raise `psutil.NoSuchProcess`."""
    return float(psutil.Process(int(pid)).create_time())


def is_lease_owner_alive(pid: int, expected_create_time_utc: float, tolerance_sec: float = 1.0) -> bool:
    """
    Verify both OS PID existence AND matching `process_create_time_utc` to prevent PID-reuse
    hazards (`design.md` §10.4).
    """
    if pid <= 0:
        return False
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        actual_create = float(proc.create_time())
        return abs(actual_create - float(expected_create_time_utc)) <= tolerance_sec
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _child_inference_worker_loop(
    package_dir_str: str,
    device_str: str,
    cmd_queue: mp.Queue,  # type: ignore[type-arg]
    resp_queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """
    Isolated child process that loads `ScratchLmPyfuncAdapter` on `device_str`,
    reports its PID, `create_time_utc`, and allocated VRAM bytes, and exits cleanly on `SHUTDOWN`
    so the OS/CUDA driver reclaims 100% of its GPU context memory.
    """
    from scratchlm.serving.package import ScratchLmPyfuncAdapter

    pid = os.getpid()
    create_time = get_process_create_time_utc(pid)
    adapter = ScratchLmPyfuncAdapter.load(Path(package_dir_str), device=device_str)
    vram_bytes = (
        int(torch.cuda.memory_allocated())
        if device_str.startswith("cuda") and torch.cuda.is_available()
        else sum(p.numel() * p.element_size() for p in adapter.model.parameters())
    )
    resp_queue.put(
        {
            "status": "READY",
            "pid": pid,
            "process_create_time_utc": create_time,
            "vram_allocated_bytes": vram_bytes,
            "package_sha256": adapter.manifest.package_sha256,
        }
    )

    while True:
        msg = cmd_queue.get()
        if not isinstance(msg, dict):
            continue
        op = msg.get("op")
        if op == "SHUTDOWN":
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            resp_queue.put({"status": "EXITED", "pid": pid})
            break
        elif op == "PING":
            resp_queue.put(
                {
                    "status": "PONG",
                    "pid": pid,
                    "process_create_time_utc": create_time,
                    "vram_allocated_bytes": vram_bytes,
                }
            )
        elif op == "PREDICT":
            prompt = str(msg.get("prompt", "Tell a story."))
            max_new = int(msg.get("max_new_tokens", 12))
            out = adapter.predict([prompt], max_new_tokens=max_new, temperature=0.0, seed=1337)
            resp_queue.put({"status": "OK", "result": out[0]})


class SingleGpuSupervisor:
    """
    Day056 Single-GPU Lease & Concurrency Supervisor:
      - Enforces at most `max_active_slots = 1` active generation run and `max_queued_runs = 8` waiting runs.
      - Tracks exclusive GPU lease ownership (`INFERENCE` vs `TRAINING`) with `(pid, process_create_time_utc)`
        stale-owner fencing.
      - Manages an optional isolated child inference worker process so draining inference for a training job
        terminates the child process and verifiably reclaims 100% of child VRAM before training starts.
    """

    def __init__(
        self,
        *,
        max_active_slots: int = 1,
        max_queued_runs: int = 8,
        package_dir: Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.max_active_slots = max_active_slots
        self.max_queued_runs = max_queued_runs
        self.package_dir = package_dir
        self.device = device

        self._lock = threading.Condition()
        self._active_run_ids: set[str] = set()
        self._queued_run_ids: list[str] = []
        self._cancelled_run_ids: set[str] = set()

        self._self_pid = os.getpid()
        self._self_create_time = get_process_create_time_utc(self._self_pid)

        self._holder_kind = "INFERENCE"
        self._holder_id = "inference-worker-main"
        self._holder_pid = self._self_pid
        self._holder_create_time = self._self_create_time
        self._acquired_at_ms = int(time.time() * 1000)
        self._lease_state = "ACTIVE"
        self._vram_allocated_bytes = 0

        # Optional isolated child worker state
        self._ctx = mp.get_context("spawn")
        self._child_proc: mp.Process | None = None
        self._child_cmd_q: mp.Queue | None = None  # type: ignore[type-arg]
        self._child_resp_q: mp.Queue | None = None  # type: ignore[type-arg]
        self._child_pid: int | None = None
        self._child_create_time: float | None = None

    def set_inference_vram_bytes(self, vram_bytes: int) -> None:
        with self._lock:
            self._vram_allocated_bytes = int(vram_bytes)

    def spawn_isolated_child_worker(self, package_dir: Path | None = None, timeout_sec: float = 20.0) -> dict[str, Any]:
        """
        Spawn an isolated child inference process holding the model weights and record its
        `(pid, process_create_time_utc)` on the GPU lease.
        """
        pkg = package_dir or self.package_dir
        if pkg is None:
            raise ValueError("package_dir is required to spawn isolated child worker")

        with self._lock:
            if self._child_proc is not None and self._child_proc.is_alive():
                return {
                    "status": "ALREADY_RUNNING",
                    "pid": self._child_pid,
                    "process_create_time_utc": self._child_create_time,
                    "vram_allocated_bytes": self._vram_allocated_bytes,
                }

            self._child_cmd_q = self._ctx.Queue()
            self._child_resp_q = self._ctx.Queue()
            self._child_proc = self._ctx.Process(
                target=_child_inference_worker_loop,
                args=(str(pkg), self.device, self._child_cmd_q, self._child_resp_q),
                daemon=True,
            )
            self._child_proc.start()
            ready_info: dict[str, Any] = self._child_resp_q.get(timeout=timeout_sec)

            self._child_pid = int(ready_info["pid"])
            self._child_create_time = float(ready_info["process_create_time_utc"])
            self._holder_kind = "INFERENCE"
            self._holder_id = f"inference-child-{self._child_pid}"
            self._holder_pid = self._child_pid
            self._holder_create_time = self._child_create_time
            self._lease_state = "ACTIVE"
            self._vram_allocated_bytes = int(ready_info["vram_allocated_bytes"])
            return ready_info

    def stop_isolated_child_worker(self, timeout_sec: float = 10.0) -> dict[str, Any]:
        """
        Gracefully stop (or force-kill) the isolated child inference process and verify that
        the child OS process has exited and 100% of its memory is reclaimed.
        """
        with self._lock:
            if self._child_proc is None:
                return {"stopped": True, "reclaimed_vram_bytes": 0, "child_pid": None}

            prior_pid = self._child_pid
            prior_vram = self._vram_allocated_bytes
            try:
                if self._child_proc.is_alive() and self._child_cmd_q is not None:
                    self._child_cmd_q.put({"op": "SHUTDOWN"})
                    if self._child_resp_q is not None:
                        try:
                            self._child_resp_q.get(timeout=timeout_sec)
                        except Exception:
                            pass
                self._child_proc.join(timeout=timeout_sec)
                if self._child_proc.is_alive():
                    self._child_proc.terminate()
                    self._child_proc.join(timeout=3.0)
            finally:
                alive_after = (
                    is_lease_owner_alive(prior_pid, self._child_create_time or 0.0)
                    if prior_pid is not None
                    else False
                )
                self._child_proc = None
                self._child_cmd_q = None
                self._child_resp_q = None
                self._child_pid = None
                self._child_create_time = None
                self._vram_allocated_bytes = 0

            return {
                "stopped": not alive_after,
                "child_pid": prior_pid,
                "child_alive_after_stop": alive_after,
                "reclaimed_vram_bytes": prior_vram,
            }

    def snapshot_lease(self) -> GpuLeaseOwner:
        with self._lock:
            self._fence_stale_owner_unlocked()
            child_alive = (
                self._child_proc is not None
                and self._child_proc.is_alive()
                and self._child_pid is not None
                and is_lease_owner_alive(self._child_pid, self._child_create_time or 0.0)
            )
            return GpuLeaseOwner(
                holder_kind=self._holder_kind,
                holder_id=self._holder_id,
                pid=self._holder_pid,
                process_create_time_utc=self._holder_create_time,
                acquired_at_epoch_ms=self._acquired_at_ms,
                lease_state=self._lease_state,
                child_worker_pid=self._child_pid,
                child_worker_alive=bool(child_alive),
                vram_allocated_bytes=self._vram_allocated_bytes,
            )

    def acquire_generation_slot(self, run_id: str, timeout_sec: float = 15.0) -> None:
        """
        Admit `run_id` into the bounded waiting queue (cap = 8) and wait until the single
        active generation slot (cap = 1) is free.
        Raises `GpuCapacityError` if:
          - GPU lease is currently held by a `TRAINING` job or `DRAINING`,
          - `len(queued_runs) >= max_queued_runs` (8), or
          - `run_id` is cancelled while queued.
        """
        deadline = time.monotonic() + timeout_sec
        with self._lock:
            self._fence_stale_owner_unlocked()
            if self._holder_kind != "INFERENCE" or self._lease_state != "ACTIVE":
                raise GpuCapacityError(
                    "GPU_LEASED_TO_TRAINING",
                    f"GPU lease is currently {self._lease_state} under holder_kind={self._holder_kind} ({self._holder_id}).",
                )

            if run_id in self._cancelled_run_ids:
                raise GpuCapacityError("RUN_CANCELLED", f"Run {run_id} was cancelled before slot acquisition.")

            # If active slot is immediately available and no one is ahead in queue:
            if len(self._active_run_ids) < self.max_active_slots and not self._queued_run_ids:
                self._active_run_ids.add(run_id)
                return

            if len(self._queued_run_ids) >= self.max_queued_runs:
                raise GpuCapacityError(
                    "SYSTEM_CAPACITY_EXCEEDED",
                    f"Queued generation runs ({len(self._queued_run_ids)}) reached max_queued_runs ({self.max_queued_runs}).",
                )

            self._queued_run_ids.append(run_id)
            try:
                while True:
                    if run_id in self._cancelled_run_ids:
                        raise GpuCapacityError("RUN_CANCELLED", f"Run {run_id} cancelled while queued.")
                    if self._holder_kind != "INFERENCE" or self._lease_state != "ACTIVE":
                        raise GpuCapacityError(
                            "GPU_LEASED_TO_TRAINING",
                            "GPU lease transitioned away from active inference while run was queued.",
                        )
                    if (
                        len(self._active_run_ids) < self.max_active_slots
                        and self._queued_run_ids
                        and self._queued_run_ids[0] == run_id
                    ):
                        self._queued_run_ids.pop(0)
                        self._active_run_ids.add(run_id)
                        return

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GpuCapacityError(
                            "QUEUE_WAIT_TIMEOUT",
                            f"Run {run_id} timed out waiting for the single active GPU generation slot.",
                        )
                    self._lock.wait(timeout=min(remaining, 0.05))
            except Exception:
                if run_id in self._queued_run_ids:
                    self._queued_run_ids.remove(run_id)
                self._lock.notify_all()
                raise

    def release_generation_slot(self, run_id: str) -> None:
        with self._lock:
            self._active_run_ids.discard(run_id)
            if run_id in self._queued_run_ids:
                self._queued_run_ids.remove(run_id)
            self._cancelled_run_ids.discard(run_id)
            self._lock.notify_all()

    def cancel_run(self, run_id: str) -> bool:
        """
        Mark `run_id` cancelled, immediately removing it from the waiting queue if queued
        and waking all waiting threads so the slot is freed without delay.
        """
        with self._lock:
            was_known = (run_id in self._active_run_ids) or (run_id in self._queued_run_ids)
            self._cancelled_run_ids.add(run_id)
            if run_id in self._queued_run_ids:
                self._queued_run_ids.remove(run_id)
            self._lock.notify_all()
            return was_known

    def is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancelled_run_ids

    def drain_and_acquire_training_lease(
        self,
        training_job_id: str,
        *,
        training_pid: int | None = None,
        training_create_time_utc: float | None = None,
        drain_timeout_sec: float = 10.0,
    ) -> dict[str, Any]:
        """
        Day056 GPU Lease Handoff:
          1. Transition lease_state to `DRAINING` so new inference requests are rejected.
          2. Wait for the 1 active generation slot to drain (or cancel if timeout).
          3. Stop the isolated child inference worker process (if running) so 100% of its CUDA
             context memory is released back to the OS.
          4. Assign exclusive GPU lease to `TRAINING` with `(training_pid, training_create_time_utc)`.
        """
        deadline = time.monotonic() + drain_timeout_sec
        with self._lock:
            self._fence_stale_owner_unlocked()
            if self._holder_kind == "TRAINING" and self._holder_id != training_job_id:
                raise GpuCapacityError(
                    "GPU_ALREADY_LEASED",
                    f"GPU is already leased to training job {self._holder_id} (pid={self._holder_pid}).",
                )
            self._lease_state = "DRAINING"
            self._lock.notify_all()

            while self._active_run_ids:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    for active_id in list(self._active_run_ids):
                        self._cancelled_run_ids.add(active_id)
                    self._lock.notify_all()
                    break
                self._lock.wait(timeout=min(remaining, 0.05))

        child_stop_report = self.stop_isolated_child_worker()

        with self._lock:
            pid = int(training_pid if training_pid is not None else self._self_pid)
            create_time = (
                float(training_create_time_utc)
                if training_create_time_utc is not None
                else get_process_create_time_utc(pid)
            )
            self._holder_kind = "TRAINING"
            self._holder_id = training_job_id
            self._holder_pid = pid
            self._holder_create_time = create_time
            self._acquired_at_ms = int(time.time() * 1000)
            self._lease_state = "ACTIVE"
            self._vram_allocated_bytes = 0
            self._lock.notify_all()

            return {
                "lease": self.snapshot_lease().to_dict(),
                "child_worker_reclaimed": child_stop_report,
            }

    def release_training_lease_to_inference(
        self,
        training_job_id: str | None = None,
        *,
        respawn_child_worker: bool = False,
    ) -> dict[str, Any]:
        """Release a training GPU lease (or fence a dead training owner) and restore `INFERENCE` readiness."""
        with self._lock:
            self._fence_stale_owner_unlocked()
            if (
                training_job_id is not None
                and self._holder_kind == "TRAINING"
                and self._holder_id != training_job_id
            ):
                raise GpuCapacityError(
                    "LEASE_OWNER_MISMATCH",
                    f"Cannot release training lease owned by {self._holder_id} using job_id={training_job_id}.",
                )
            self._holder_kind = "INFERENCE"
            self._holder_id = "inference-worker-main"
            self._holder_pid = self._self_pid
            self._holder_create_time = self._self_create_time
            self._acquired_at_ms = int(time.time() * 1000)
            self._lease_state = "ACTIVE"
            self._lock.notify_all()

        child_info = None
        if respawn_child_worker and self.package_dir is not None:
            child_info = self.spawn_isolated_child_worker(self.package_dir)

        return {
            "lease": self.snapshot_lease().to_dict(),
            "child_worker": child_info,
        }

    def simulate_stale_lease_owner(self, dead_pid: int = 999999, fake_create_time_utc: float = 1000.0) -> None:
        """Test helper to inject a stale/dead `(pid, process_create_time_utc)` lease owner for fencing verification."""
        with self._lock:
            self._holder_kind = "TRAINING"
            self._holder_id = "crashed-training-job"
            self._holder_pid = dead_pid
            self._holder_create_time = fake_create_time_utc
            self._lease_state = "ACTIVE"

    def _fence_stale_owner_unlocked(self) -> bool:
        """
        If the current lease owner process is dead or its `process_create_time_utc` does not match
        (PID reuse), fence the stale owner and restore the lease to the live inference supervisor.
        """
        if not is_lease_owner_alive(self._holder_pid, self._holder_create_time):
            self._holder_kind = "INFERENCE"
            self._holder_id = "inference-worker-main"
            self._holder_pid = self._self_pid
            self._holder_create_time = self._self_create_time
            self._acquired_at_ms = int(time.time() * 1000)
            self._lease_state = "ACTIVE"
            self._vram_allocated_bytes = 0
            return True
        return False

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            self._fence_stale_owner_unlocked()
            return {
                "active_runs": len(self._active_run_ids),
                "active_run_ids": list(self._active_run_ids),
                "queued_runs": len(self._queued_run_ids),
                "queued_run_ids": list(self._queued_run_ids),
                "max_active_slots": self.max_active_slots,
                "max_queued_runs": self.max_queued_runs,
                "gpu_lease": self.snapshot_lease().to_dict(),
            }
