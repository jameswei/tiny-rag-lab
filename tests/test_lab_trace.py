import json

from tiny_rag_lab.lab_trace import (
    EvidenceSnapshot,
    build_lab_run,
    load_lab_run,
    write_lab_run,
)
from tiny_rag_lab.trace import ChunkTrace, RetrieveTrace


def _trace() -> RetrieveTrace:
    return RetrieveTrace(
        query="Where is the runbook?",
        retriever="dense",
        top_k=1,
        chunks=[
            ChunkTrace(
                rank=1,
                chunk_id="chunk-1",
                doc_id="runbook.md",
                title="Runbook",
                path="/corpus/runbook.md",
                score=0.8,
                text_preview="Use the runbook.",
            )
        ],
    )


def test_lab_run_is_json_safe_and_keeps_full_evidence(tmp_path):
    run = build_lab_run(
        _trace(),
        index_id="demo-index",
        manifest={"chunk_count": 1, "index_backend": "numpy"},
        document_count=1,
        query_vector=[0.1, 0.2],
        evidence=[
            EvidenceSnapshot(
                chunk_id="chunk-1",
                doc_id="runbook.md",
                title="Runbook",
                path="/corpus/runbook.md",
                text="Use the full runbook text.",
                rank=1,
                score=0.8,
                score_semantics="cosine_similarity[-1,1]",
            )
        ],
    )

    path = tmp_path / "run.json"
    write_lab_run(run, path)

    raw = json.loads(path.read_text())
    assert raw["operation"] == "retrieve"
    assert raw["query_vector"] == [0.1, 0.2]
    assert raw["evidence"][0]["text"] == "Use the full runbook text."
    assert "api_key" not in json.dumps(raw).lower()
    assert load_lab_run(path)["run_id"] == run.run_id
