import json

import pytest

from tiny_rag_lab.jobs import (
    JobConflictError,
    JobNotFoundError,
    LocalJobStore,
    atomic_write_json,
)


def test_atomic_json_replace_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "job.json"
    atomic_write_json(path, {"status": "queued"})
    atomic_write_json(path, {"status": "running", "progress": {"current": 1}})

    assert json.loads(path.read_text()) == {
        "status": "running", "progress": {"current": 1},
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_store_admits_only_one_active_job_and_discovers_it(tmp_path):
    store = LocalJobStore(tmp_path)
    job = store.admit("evaluation", preset="dense-vs-hybrid")

    assert store.active() == [job]
    assert store.active(kind="evaluation") == [job]
    assert store.active(kind="index") == []
    with pytest.raises(JobConflictError, match=job["id"]):
        store.admit("index")


def test_queued_and_running_cancellation_become_terminal_at_checkpoint(tmp_path):
    queued_store = LocalJobStore(tmp_path / "queued")
    queued = queued_store.admit("evaluation")
    assert queued_store.request_cancel(queued["id"])["status"] == "cancel_requested"
    assert queued_store.start(queued["id"], total=16) is False
    assert queued_store.read(queued["id"])["status"] == "cancelled"

    running_store = LocalJobStore(tmp_path / "running")
    running = running_store.admit("evaluation")
    assert running_store.start(running["id"], total=16)
    running_store.request_cancel(running["id"])
    assert running_store.progress(
        running["id"], 1, total=16, message="Finished current question",
    ) is False
    assert running_store.read(running["id"])["status"] == "cancelled"


def test_only_complete_job_publishes_a_readable_result(tmp_path):
    store = LocalJobStore(tmp_path)
    job = store.admit("evaluation")
    store.start(job["id"], total=2)
    store.progress(job["id"], 1, total=2, message="Question 1 of 2")
    completed = store.complete(job["id"], result={"metrics": {"hit_rate": 1.0}})

    assert completed["status"] == "complete"
    assert completed["result_available"] is True
    assert store.result(job["id"]) == {"metrics": {"hit_rate": 1.0}}

    failed = store.admit("evaluation")
    store.start(failed["id"])
    store.fail(failed["id"], "failed safely")
    with pytest.raises(JobNotFoundError):
        store.result(failed["id"])


def test_publication_boundary_rejects_late_cancellation(tmp_path):
    store = LocalJobStore(tmp_path)
    job = store.admit("evaluation")
    store.start(job["id"], total=1)
    store.progress(job["id"], 1, total=1, message="Work complete")

    assert store.begin_publish(job["id"], message="Publishing comparison") is True
    assert store.request_cancel(job["id"])["status"] == "publishing"
    completed = store.complete(job["id"], result={"questions": []})

    assert completed["status"] == "complete"
    assert store.result(job["id"]) == {"questions": []}


def test_restart_recovery_fails_work_but_honors_requested_cancellation(tmp_path):
    store = LocalJobStore(tmp_path)
    running = store.admit("index")
    store.start(running["id"], total=5)
    # Seed a second record directly to simulate state recovered from another
    # process; admission correctly prevents creating it through the API.
    cancelled = {
        "id": "evaluation-cancel", "kind": "evaluation",
        "status": "cancel_requested",
    }
    atomic_write_json(tmp_path / "evaluation-cancel.json", cancelled)

    store.recover_interrupted()

    assert store.read(running["id"])["status"] == "failed"
    assert "restarted" in store.read(running["id"])["error"]
    assert store.read("evaluation-cancel")["status"] == "cancelled"
