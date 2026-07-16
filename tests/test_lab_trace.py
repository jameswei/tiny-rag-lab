import json

from tiny_rag_lab.lab_trace import (
    LAB_TRACE_SCHEMA_VERSION,
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
        candidates=[
            EvidenceSnapshot(
                chunk_id="chunk-2", doc_id="other.md", title="Other",
                path="/corpus/other.md", text="A pre-rerank candidate.",
                rank=2, score=0.4, score_semantics="cosine_similarity[-1,1]",
            )
        ],
        explanations={"dense": {"dimension": 2}},
    )

    path = tmp_path / "run.json"
    write_lab_run(run, path)

    raw = json.loads(path.read_text())
    assert raw["operation"] == "retrieve"
    assert raw["query_vector"] == [0.1, 0.2]
    assert raw["evidence"][0]["text"] == "Use the full runbook text."
    assert raw["candidates"][0]["chunk_id"] == "chunk-2"
    assert raw["explanations"]["dense"]["dimension"] == 2
    assert raw["schema_version"] == LAB_TRACE_SCHEMA_VERSION == "1.1"
    assert "api_key" not in json.dumps(raw).lower()
    assert load_lab_run(path)["run_id"] == run.run_id


def test_schema_1_0_run_loads_without_phase_3_4_fields(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "run_id": "legacy-run",
        "operation": "retrieve",
        "evidence": [],
    }), encoding="utf-8")

    loaded = load_lab_run(path)

    assert loaded["schema_version"] == "1.0"
    assert "candidates" not in loaded
    assert "explanations" not in loaded
