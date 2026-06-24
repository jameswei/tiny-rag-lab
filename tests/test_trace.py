"""Tests for tiny_rag_lab/trace.py.

Test naming convention:
  *_dataclass_* — dataclass round-trip and JSON serialization (T01)
  *_serial_*    — write_trace_json and trace_to_dict (T02)
  *_format_*    — format_retrieve_trace and format_ask_trace (T03)
"""
import dataclasses
import json

import pytest

from tiny_rag_lab.trace import (
    AskTrace,
    ChunkTrace,
    RetrieveTrace,
    format_ask_trace,
    format_retrieve_trace,
    trace_to_dict,
    write_trace_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _chunk_trace(rank: int = 1) -> ChunkTrace:
    return ChunkTrace(
        rank=rank,
        chunk_id="abc123def456",
        doc_id="docs/deploy.md",
        title="Deploying Models",
        path="/corpus/docs/deploy.md",
        score=0.8432,
        text_preview="First 120 chars of chunk text.",
    )


def _retrieve_trace() -> RetrieveTrace:
    return RetrieveTrace(
        query="how to deploy a model",
        retriever="dense",
        top_k=5,
        chunks=[_chunk_trace(1), _chunk_trace(2)],
        latency_by_stage={"load": 0.051, "embed": 0.012, "retrieve": 0.003},
    )


def _ask_trace() -> AskTrace:
    return AskTrace(
        query="how to deploy a model",
        retriever="dense",
        top_k=5,
        chunks=[_chunk_trace(1)],
        prompt="Context:\n...\nQuestion: how to deploy a model",
        answer="You can deploy using the CLI. [Source: docs/deploy.md]",
        citations=["docs/deploy.md"],
        latency_by_stage={
            "load": 0.051,
            "embed": 0.012,
            "retrieve": 0.003,
            "prompt_assembly": 0.001,
            "generate": 1.234,
        },
    )


# ---------------------------------------------------------------------------
# T01: Dataclass tests
# ---------------------------------------------------------------------------

def test_chunk_trace_dataclass_fields():
    c = _chunk_trace()
    assert c.rank == 1
    assert c.chunk_id == "abc123def456"
    assert c.doc_id == "docs/deploy.md"
    assert c.title == "Deploying Models"
    assert c.path == "/corpus/docs/deploy.md"
    assert c.score == 0.8432
    assert c.text_preview == "First 120 chars of chunk text."


def test_chunk_trace_dataclass_asdict_roundtrip():
    c = _chunk_trace()
    d = dataclasses.asdict(c)
    assert d["rank"] == 1
    assert d["chunk_id"] == "abc123def456"
    assert d["doc_id"] == "docs/deploy.md"
    assert d["score"] == 0.8432


def test_chunk_trace_dataclass_json_serializable():
    c = _chunk_trace()
    serialized = json.dumps(dataclasses.asdict(c))
    loaded = json.loads(serialized)
    assert loaded["rank"] == 1
    assert loaded["doc_id"] == "docs/deploy.md"


def test_retrieve_trace_dataclass_fields():
    t = _retrieve_trace()
    assert t.query == "how to deploy a model"
    assert t.retriever == "dense"
    assert t.top_k == 5
    assert len(t.chunks) == 2
    assert t.latency_by_stage["load"] == 0.051
    assert t.latency_by_stage["embed"] == 0.012
    assert t.latency_by_stage["retrieve"] == 0.003


def test_retrieve_trace_dataclass_asdict_contains_chunks():
    t = _retrieve_trace()
    d = dataclasses.asdict(t)
    assert d["retriever"] == "dense"
    assert d["top_k"] == 5
    assert isinstance(d["chunks"], list)
    assert d["chunks"][0]["chunk_id"] == "abc123def456"


def test_retrieve_trace_dataclass_json_serializable():
    t = _retrieve_trace()
    serialized = json.dumps(dataclasses.asdict(t))
    loaded = json.loads(serialized)
    assert loaded["retriever"] == "dense"
    assert loaded["top_k"] == 5
    assert loaded["chunks"][0]["doc_id"] == "docs/deploy.md"


def test_retrieve_trace_dataclass_bm25_omits_embed():
    # BM25 runs should not include "embed" in latency_by_stage
    t = RetrieveTrace(
        query="q",
        retriever="bm25",
        top_k=3,
        chunks=[],
        latency_by_stage={"load": 0.01, "retrieve": 0.002},
    )
    d = dataclasses.asdict(t)
    assert "embed" not in d["latency_by_stage"]
    assert "load" in d["latency_by_stage"]
    assert "retrieve" in d["latency_by_stage"]


def test_ask_trace_dataclass_fields():
    t = _ask_trace()
    assert t.query == "how to deploy a model"
    assert t.retriever == "dense"
    assert t.top_k == 5
    assert t.prompt.startswith("Context:")
    assert "deploy" in t.answer
    assert t.citations == ["docs/deploy.md"]
    assert set(t.latency_by_stage) == {
        "load", "embed", "retrieve", "prompt_assembly", "generate"
    }


def test_ask_trace_dataclass_asdict_contains_all_required_keys():
    t = _ask_trace()
    d = dataclasses.asdict(t)
    for key in ("query", "retriever", "top_k", "chunks", "prompt", "answer",
                "citations", "latency_by_stage"):
        assert key in d, f"missing key: {key}"


def test_ask_trace_dataclass_json_serializable():
    t = _ask_trace()
    serialized = json.dumps(dataclasses.asdict(t))
    loaded = json.loads(serialized)
    assert loaded["prompt"] != ""
    assert loaded["answer"] != ""
    assert loaded["citations"] == ["docs/deploy.md"]


def test_ask_trace_dataclass_latency_keys():
    t = _ask_trace()
    d = dataclasses.asdict(t)
    assert set(d["latency_by_stage"]) == {
        "load", "embed", "retrieve", "prompt_assembly", "generate"
    }


def test_retrieve_trace_dataclass_empty_chunks_default():
    t = RetrieveTrace(query="q", retriever="dense", top_k=5)
    assert t.chunks == []
    assert t.latency_by_stage == {}


def test_ask_trace_dataclass_empty_defaults():
    t = AskTrace(query="q", retriever="dense", top_k=5)
    assert t.chunks == []
    assert t.prompt == ""
    assert t.answer == ""
    assert t.citations == []
    assert t.latency_by_stage == {}


# ---------------------------------------------------------------------------
# T02: Serialization tests (serial)
# ---------------------------------------------------------------------------

def test_serial_trace_to_dict_retrieve(tmp_path):
    t = _retrieve_trace()
    d = trace_to_dict(t)
    assert isinstance(d, dict)
    assert d["retriever"] == "dense"
    assert isinstance(d["chunks"], list)
    assert d["chunks"][0]["rank"] == 1


def test_serial_trace_to_dict_ask(tmp_path):
    t = _ask_trace()
    d = trace_to_dict(t)
    assert d["prompt"] != ""
    assert d["answer"] != ""
    assert d["citations"] == ["docs/deploy.md"]


def test_serial_write_trace_json_creates_file(tmp_path):
    t = _retrieve_trace()
    out = tmp_path / "trace.json"
    write_trace_json(t, out)
    assert out.exists()


def test_serial_write_trace_json_valid_json(tmp_path):
    t = _retrieve_trace()
    out = tmp_path / "trace.json"
    write_trace_json(t, out)
    loaded = json.loads(out.read_text())
    assert loaded["retriever"] == "dense"
    assert loaded["top_k"] == 5


def test_serial_write_trace_json_chunk_fields(tmp_path):
    t = _retrieve_trace()
    out = tmp_path / "trace.json"
    write_trace_json(t, out)
    loaded = json.loads(out.read_text())
    chunk = loaded["chunks"][0]
    for key in ("rank", "score", "chunk_id", "doc_id", "title", "path"):
        assert key in chunk, f"missing key in chunk: {key}"


def test_serial_write_trace_json_creates_parent_dirs(tmp_path):
    t = _ask_trace()
    out = tmp_path / "nested" / "dir" / "trace.json"
    write_trace_json(t, out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["answer"] != ""


def test_serial_write_trace_json_ask_latency_keys(tmp_path):
    t = _ask_trace()
    out = tmp_path / "ask_trace.json"
    write_trace_json(t, out)
    loaded = json.loads(out.read_text())
    assert set(loaded["latency_by_stage"]) == {
        "load", "embed", "retrieve", "prompt_assembly", "generate"
    }


# ---------------------------------------------------------------------------
# T03: Formatter tests (format)
# ---------------------------------------------------------------------------

def test_format_retrieve_trace_contains_query():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "how to deploy a model" in out


def test_format_retrieve_trace_contains_retriever():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "dense" in out


def test_format_retrieve_trace_contains_top_k():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "5" in out


def test_format_retrieve_trace_contains_latency_keys():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "load" in out
    assert "retrieve" in out


def test_format_retrieve_trace_contains_rank_and_score():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "Rank 1" in out
    assert "0.8432" in out


def test_format_retrieve_trace_contains_doc_id():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "docs/deploy.md" in out


def test_format_retrieve_trace_no_ansi_codes():
    t = _retrieve_trace()
    out = format_retrieve_trace(t)
    assert "\x1b" not in out


def test_format_retrieve_trace_empty_chunks():
    t = RetrieveTrace(
        query="q", retriever="bm25", top_k=3,
        chunks=[],
        latency_by_stage={"load": 0.01, "retrieve": 0.002},
    )
    out = format_retrieve_trace(t)
    assert "No results" in out


def test_format_ask_trace_contains_query():
    t = _ask_trace()
    out = format_ask_trace(t)
    assert "how to deploy a model" in out


def test_format_ask_trace_contains_answer():
    t = _ask_trace()
    out = format_ask_trace(t)
    assert "You can deploy using the CLI" in out


def test_format_ask_trace_contains_citations():
    t = _ask_trace()
    out = format_ask_trace(t)
    assert "docs/deploy.md" in out


def test_format_ask_trace_contains_latency():
    t = _ask_trace()
    out = format_ask_trace(t)
    assert "generate" in out
    assert "load" in out


def test_format_ask_trace_no_ansi_codes():
    t = _ask_trace()
    out = format_ask_trace(t)


# ---------------------------------------------------------------------------
# P2.0-T04 — AskTrace.verdict field and format_ask_trace verdict block
# ---------------------------------------------------------------------------

from tiny_rag_lab.judge import JudgeVerdict


def _verdict(**kwargs) -> JudgeVerdict:
    defaults = dict(
        faithfulness=0.9, answer_relevance=0.8, citation_support=0.7,
        answer_correctness=None, judge_name="fake", latency=0.0,
    )
    defaults.update(kwargs)
    return JudgeVerdict(**defaults)


def test_ask_trace_verdict_defaults_to_none():
    t = AskTrace(query="q", retriever="dense", top_k=3)
    assert t.verdict is None


def test_ask_trace_verdict_none_serializes_as_null():
    t = AskTrace(query="q", retriever="dense", top_k=3)
    d = dataclasses.asdict(t)
    assert d["verdict"] is None
    assert json.dumps(d)  # no encoder error


def test_ask_trace_verdict_populated_serializes_all_fields():
    t = AskTrace(query="q", retriever="dense", top_k=3, verdict=_verdict())
    d = dataclasses.asdict(t)
    v = d["verdict"]
    assert v["faithfulness"] == 0.9
    assert v["answer_relevance"] == 0.8
    assert v["citation_support"] == 0.7
    assert v["answer_correctness"] is None
    assert v["judge_name"] == "fake"
    assert json.dumps(d)  # no encoder error


def test_ask_trace_verdict_answer_correctness_float_serializes():
    t = AskTrace(query="q", retriever="dense", top_k=3,
                 verdict=_verdict(answer_correctness=0.65))
    d = dataclasses.asdict(t)
    assert d["verdict"]["answer_correctness"] == pytest.approx(0.65)


def test_format_ask_trace_verdict_none_has_no_verdict_block():
    t = _ask_trace()
    assert t.verdict is None
    out = format_ask_trace(t)
    assert "Judge verdict" not in out


def test_format_ask_trace_verdict_populated_shows_header():
    t = _ask_trace()
    t.verdict = _verdict()
    out = format_ask_trace(t)
    assert "Judge verdict" in out
    assert "judge=fake" in out


def test_format_ask_trace_verdict_shows_all_three_base_scores():
    t = _ask_trace()
    t.verdict = _verdict(faithfulness=0.900, answer_relevance=0.800,
                         citation_support=0.700)
    out = format_ask_trace(t)
    assert "Faithfulness" in out
    assert "Answer Relevance" in out
    assert "Citation Support" in out


def test_format_ask_trace_verdict_omits_correctness_when_none():
    t = _ask_trace()
    t.verdict = _verdict(answer_correctness=None)
    out = format_ask_trace(t)
    assert "Answer Correct" not in out


def test_format_ask_trace_verdict_shows_correctness_when_set():
    t = _ask_trace()
    t.verdict = _verdict(answer_correctness=0.75)
    out = format_ask_trace(t)
    assert "Answer Correct" in out
    assert "0.750" in out


def test_format_ask_trace_verdict_shows_notes_when_set():
    t = _ask_trace()
    t.verdict = _verdict(notes="One claim lacks citation.")
    out = format_ask_trace(t)
    assert "One claim lacks citation." in out


def test_format_ask_trace_verdict_omits_notes_when_empty():
    t = _ask_trace()
    t.verdict = _verdict(notes="")
    out = format_ask_trace(t)
    assert "Notes" not in out
    assert "\x1b" not in out


# ---------------------------------------------------------------------------
# P2.1-T02 — AskTrace.context_pack field and format_ask_trace packing block
# ---------------------------------------------------------------------------

from tiny_rag_lab.context import ContextPackResult


def _pack_result(
    selected: list[str] | None = None,
    omitted: list[str] | None = None,
    estimated_tokens: int = 120,
    budget: int = 8192,
    counter_name: str = "char",
) -> ContextPackResult:
    return ContextPackResult(
        selected=selected or ["chunk0000000001", "chunk0000000002"],
        omitted=omitted or [],
        estimated_tokens=estimated_tokens,
        budget=budget,
        counter_name=counter_name,
    )


def test_ask_trace_context_pack_defaults_to_none():
    t = AskTrace(query="q", retriever="dense", top_k=3)
    assert t.context_pack is None


def test_ask_trace_context_pack_none_serializes_as_null():
    t = AskTrace(query="q", retriever="dense", top_k=3)
    d = dataclasses.asdict(t)
    assert d["context_pack"] is None
    assert json.dumps(d)  # no encoder error


def test_ask_trace_context_pack_populated_serializes_all_fields():
    t = AskTrace(query="q", retriever="dense", top_k=3,
                 context_pack=_pack_result(
                     selected=["aaa", "bbb"], omitted=["ccc"],
                     estimated_tokens=200, budget=500, counter_name="char",
                 ))
    d = dataclasses.asdict(t)
    cp = d["context_pack"]
    assert cp["selected"] == ["aaa", "bbb"]
    assert cp["omitted"] == ["ccc"]
    assert cp["estimated_tokens"] == 200
    assert cp["budget"] == 500
    assert cp["counter_name"] == "char"
    assert json.dumps(d)  # no encoder error


def test_ask_trace_context_pack_all_fields_json_native():
    t = AskTrace(query="q", retriever="dense", top_k=3,
                 context_pack=_pack_result())
    serialized = json.dumps(dataclasses.asdict(t))
    parsed = json.loads(serialized)
    assert isinstance(parsed["context_pack"]["selected"], list)
    assert isinstance(parsed["context_pack"]["estimated_tokens"], int)
    assert isinstance(parsed["context_pack"]["budget"], int)
    assert isinstance(parsed["context_pack"]["counter_name"], str)


def test_format_ask_trace_context_pack_none_has_no_packing_block():
    t = _ask_trace()
    assert t.context_pack is None
    out = format_ask_trace(t)
    assert "Context packing" not in out


def test_format_ask_trace_context_pack_shows_header():
    t = _ask_trace()
    t.context_pack = _pack_result(budget=8192, counter_name="tiktoken-gpt-4o-mini")
    out = format_ask_trace(t)
    assert "Context packing" in out
    assert "budget=8192" in out
    assert "tiktoken-gpt-4o-mini" in out


def test_format_ask_trace_context_pack_shows_selected_count():
    t = _ask_trace()
    t.context_pack = _pack_result(selected=["a", "b", "c"], estimated_tokens=300)
    out = format_ask_trace(t)
    assert "Selected" in out
    assert "3 chunks" in out
    assert "300" in out


def test_format_ask_trace_context_pack_zero_omitted():
    t = _ask_trace()
    t.context_pack = _pack_result(omitted=[])
    out = format_ask_trace(t)
    assert "Omitted" in out
    assert "0 chunks" in out


def test_format_ask_trace_context_pack_lists_omitted_chunk_ids():
    t = _ask_trace()
    t.context_pack = _pack_result(
        selected=["chunk0000000001"],
        omitted=["chunk0000000002", "chunk0000000003"],
    )
    out = format_ask_trace(t)
    assert "2 chunks" in out
    assert "chunk0000000002" in out
    assert "chunk0000000003" in out


def test_format_ask_trace_context_pack_no_ansi_codes():
    t = _ask_trace()
    t.context_pack = _pack_result()
    out = format_ask_trace(t)
    assert "\x1b" not in out


def test_format_ask_trace_context_pack_and_verdict_both_render():
    t = _ask_trace()
    t.context_pack = _pack_result()
    t.verdict = _verdict()
    out = format_ask_trace(t)
    assert "Context packing" in out
    assert "Judge verdict" in out


def test_format_ask_trace_context_pack_appears_before_answer():
    """Context packing block must appear after chunks, before the answer separator."""
    t = _ask_trace()
    t.context_pack = _pack_result()
    out = format_ask_trace(t)
    pack_pos = out.index("Context packing")
    answer_pos = out.index("Answer:")
    assert pack_pos < answer_pos, (
        "Context packing block must appear before 'Answer:' section"
    )
