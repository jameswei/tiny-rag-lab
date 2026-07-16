"""Small, inspectable persistence for local background work.

Jobs are JSON files because the lab is single-user and local. Atomic replace
keeps browser polling from observing partial JSON, while explicit states make
restart and cooperative cancellation behavior visible to learners.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ACTIVE_STATUSES = {"queued", "running", "cancel_requested", "publishing"}
TERMINAL_STATUSES = {"complete", "failed", "cancelled"}


class JobConflictError(RuntimeError):
    """Another resource-heavy local job is still active."""


class JobNotFoundError(FileNotFoundError):
    """The requested persisted job does not exist."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict) -> None:
    """Write JSON completely, then atomically replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class LocalJobStore:
    """Persist and coordinate one local resource-heavy job at a time."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.results_dir = self.root / "results"
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def _result_path(self, job_id: str) -> Path:
        return self.results_dir / f"{job_id}.json"

    def read(self, job_id: str) -> dict:
        path = self._path(job_id)
        if not path.exists():
            raise JobNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def active(self, *, kind: str | None = None) -> list[dict]:
        items = []
        for path in sorted(self.root.glob("*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in ACTIVE_STATUSES and (kind is None or job.get("kind") == kind):
                items.append(job)
        return items

    def all(self) -> list[dict]:
        items = []
        for path in sorted(self.root.glob("*.json")):
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return items

    def admit(self, kind: str, **fields) -> dict:
        with self._lock:
            active = self.active()
            if active:
                current = active[0]
                raise JobConflictError(
                    f"Local job {current.get('id', 'unknown')} is {current['status']}. "
                    "Wait for it or cancel it before starting another job."
                )
            job_id = f"{kind}-{uuid4().hex[:12]}"
            job = {
                "id": job_id, "kind": kind, "status": "queued",
                "created_at": utc_now(),
                "progress": {"current": 0, "total": None, "message": "Queued"},
                **fields,
            }
            atomic_write_json(self._path(job_id), job)
            return job

    def update(self, job_id: str, **changes) -> dict:
        with self._lock:
            job = self.read(job_id)
            job.update(changes)
            atomic_write_json(self._path(job_id), job)
            return job

    def start(self, job_id: str, *, total: int | None = None, message: str = "Running") -> bool:
        with self._lock:
            job = self.read(job_id)
            if job["status"] == "cancel_requested":
                job.update(status="cancelled", completed_at=utc_now())
                atomic_write_json(self._path(job_id), job)
                return False
            if job["status"] != "queued":
                return False
            job.update(
                status="running", started_at=utc_now(),
                progress={"current": 0, "total": total, "message": message},
            )
            atomic_write_json(self._path(job_id), job)
            return True

    def progress(
        self, job_id: str, current: int, *, total: int | None = None,
        message: str,
    ) -> bool:
        """Publish progress and return False when cancellation was accepted."""
        with self._lock:
            job = self.read(job_id)
            if job["status"] == "cancel_requested":
                job.update(status="cancelled", completed_at=utc_now())
                atomic_write_json(self._path(job_id), job)
                return False
            if job["status"] != "running":
                return False
            previous = job.get("progress", {})
            job["progress"] = {
                "current": current,
                "total": total if total is not None else previous.get("total"),
                "message": message,
            }
            atomic_write_json(self._path(job_id), job)
            return True

    def request_cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self.read(job_id)
            if job["status"] in TERMINAL_STATUSES or job["status"] == "publishing":
                return job
            job["status"] = "cancel_requested"
            job["cancel_requested_at"] = utc_now()
            atomic_write_json(self._path(job_id), job)
            return job

    def begin_publish(self, job_id: str, *, message: str = "Publishing result") -> bool:
        """Claim the short, non-cancellable final publication boundary."""
        with self._lock:
            job = self.read(job_id)
            if job["status"] == "cancel_requested":
                job.update(status="cancelled", completed_at=utc_now())
                atomic_write_json(self._path(job_id), job)
                return False
            if job["status"] != "running":
                return False
            job["status"] = "publishing"
            progress = job.get("progress", {})
            job["progress"] = {**progress, "message": message}
            atomic_write_json(self._path(job_id), job)
            return True

    def complete(self, job_id: str, *, result: dict | None = None, **fields) -> dict:
        """Publish a result first, then expose the terminal complete state."""
        with self._lock:
            job = self.read(job_id)
            if job["status"] == "cancel_requested":
                job.update(status="cancelled", completed_at=utc_now())
                atomic_write_json(self._path(job_id), job)
                return job
            if job["status"] not in {"running", "publishing"}:
                return job
            if result is not None:
                atomic_write_json(self._result_path(job_id), result)
                fields["result_available"] = True
            job.update(
                status="complete", completed_at=utc_now(),
                progress={
                    "current": job.get("progress", {}).get("total"),
                    "total": job.get("progress", {}).get("total"),
                    "message": "Complete",
                },
                **fields,
            )
            atomic_write_json(self._path(job_id), job)
            return job

    def fail(self, job_id: str, error: str) -> dict:
        with self._lock:
            job = self.read(job_id)
            if job["status"] == "cancel_requested":
                job.update(status="cancelled", completed_at=utc_now())
            elif job["status"] not in TERMINAL_STATUSES:
                job.update(status="failed", error=error, completed_at=utc_now())
            atomic_write_json(self._path(job_id), job)
            return job

    def result(self, job_id: str) -> dict:
        job = self.read(job_id)
        path = self._result_path(job_id)
        if job.get("status") != "complete" or not job.get("result_available") or not path.exists():
            raise JobNotFoundError(f"No complete result for {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def recover_interrupted(self) -> None:
        """Make process-bound work terminal after a local server restart."""
        for job in self.active():
            if job["status"] == "cancel_requested":
                self.update(job["id"], status="cancelled", completed_at=utc_now())
            else:
                self.update(
                    job["id"], status="failed", completed_at=utc_now(),
                    error="The local server restarted before this job completed. Please start it again.",
                )
