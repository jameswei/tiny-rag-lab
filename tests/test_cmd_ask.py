"""Tests for T18 — rag ask CLI command, and T05 --trace-out (P1.7).

Patches both _make_embedder and _make_generator with fake backends so no
model download or API credentials are needed.
"""
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from tiny_rag_lab.cli import build_parser, cmd_ask, cmd_index
from tiny_rag_lab.embeddings import FakeEmbedder
from tiny_rag_lab.generation import FakeGenerator
from tiny_rag_lab.reranker import FakeReranker

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedder_factory(dim: int = 8):
    def _make(model_name=None):
        return FakeEmbedder(dim=dim)
    return _make


def _fake_generator_factory():
    def _make(args):
        return FakeGenerator()
    return _make


def _index_args(corpus, index_dir):
    return build_parser().parse_args([
        "index", "--corpus", str(corpus),
        "--index-dir", str(index_dir),
        "--chunk-size", "200", "--chunk-overlap", "20",
    ])


def _ask_args(query, index_dir, top_k=3, trace_out=None):
    argv = ["ask", query, "--index-dir", str(index_dir), "--top-k", str(top_k)]
    if trace_out is not None:
        argv += ["--trace-out", str(trace_out)]
    return build_parser().parse_args(argv)


@pytest.fixture()
def ask_setup(tmp_path):
    """Build an index with fake embedder, return index_dir."""
    index_dir = tmp_path / "index"
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()):
        cmd_index(_index_args(FIXTURE_CORPUS, index_dir))
    return index_dir


# ---------------------------------------------------------------------------
# T18 — cmd_ask output
# ---------------------------------------------------------------------------

def test_ask_prints_answer(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert len(out.strip()) > 0


def test_ask_prints_retrieved_chunks(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Rank 1" in out
    assert "doc_id" in out


def test_ask_prints_latency(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "latency" in out
    assert "embed=" in out
    assert "retrieve=" in out
    assert "generate=" in out


def test_ask_output_contains_source_markers(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "[Source:" in out


def test_ask_respects_top_k(ask_setup, capsys):
    # top_k=1 means exactly one retrieved chunk and one source marker in the answer.
    args = _ask_args("sample document", ask_setup, top_k=1)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert len(re.findall(r"Rank \d+", out)) == 1
    assert len(re.findall(r"\[Source:", out)) == 1


# ---------------------------------------------------------------------------
# Parser — new flags on rag ask
# ---------------------------------------------------------------------------

def test_ask_parser_has_model_flag():
    args = build_parser().parse_args(["ask", "Q?", "--model", "gpt-4o"])
    assert args.model == "gpt-4o"


def test_ask_parser_has_api_key_flag():
    args = build_parser().parse_args(["ask", "Q?", "--api-key", "sk-test"])
    assert args.api_key == "sk-test"


def test_ask_parser_has_base_url_flag():
    args = build_parser().parse_args(["ask", "Q?", "--base-url", "http://localhost/v1"])
    assert args.base_url == "http://localhost/v1"


def test_ask_parser_defaults_are_none():
    args = build_parser().parse_args(["ask", "Q?"])
    assert args.model is None
    assert args.api_key is None
    assert args.base_url is None


# ---------------------------------------------------------------------------
# Existing test still passes (rag ask --help)
# ---------------------------------------------------------------------------

def test_ask_help():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["ask", "--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# P1.7-T05 — --trace-out flag
# ---------------------------------------------------------------------------

def test_ask_help_shows_trace_out_flag(capsys):
    try:
        build_parser().parse_args(["ask", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--trace-out" in out


def test_ask_trace_out_creates_file(tmp_path, ask_setup):
    out_path = tmp_path / "ask_trace.json"
    args = _ask_args("sample document", ask_setup, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    assert out_path.exists()


def test_ask_trace_out_valid_json(tmp_path, ask_setup):
    out_path = tmp_path / "ask_trace.json"
    args = _ask_args("sample document", ask_setup, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    loaded = json.loads(out_path.read_text())
    assert isinstance(loaded, dict)


def test_ask_trace_out_top_level_fields(tmp_path, ask_setup):
    out_path = tmp_path / "ask_trace.json"
    args = _ask_args("sample document", ask_setup, top_k=2, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    loaded = json.loads(out_path.read_text())
    assert loaded["retriever"] == "dense"
    assert loaded["top_k"] == 2
    assert loaded["query"] == "sample document"
    assert loaded["prompt"] != ""
    assert loaded["answer"] != ""
    assert isinstance(loaded["citations"], list)
    assert isinstance(loaded["chunks"], list)


def test_ask_trace_out_latency_keys(tmp_path, ask_setup):
    out_path = tmp_path / "ask_trace.json"
    args = _ask_args("sample document", ask_setup, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    loaded = json.loads(out_path.read_text())
    assert set(loaded["latency_by_stage"]) == {
        "load", "embed", "retrieve", "prompt_assembly", "generate"
    }


def test_ask_trace_out_chunk_fields(tmp_path, ask_setup):
    out_path = tmp_path / "ask_trace.json"
    args = _ask_args("sample document", ask_setup, top_k=1, trace_out=out_path)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    loaded = json.loads(out_path.read_text())
    chunk = loaded["chunks"][0]
    for key in ("rank", "score", "chunk_id", "doc_id", "title", "path"):
        assert key in chunk, f"missing key: {key}"


def test_ask_without_trace_out_still_prints(ask_setup, capsys):
    args = _ask_args("sample document", ask_setup)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert len(out.strip()) > 0


# ---------------------------------------------------------------------------
# P1.9-T05 — reranker flags and trace (cmd_ask)
# ---------------------------------------------------------------------------


def test_ask_help_shows_reranker_flags(capsys):
    parser = build_parser()
    try:
        parser.parse_args(["ask", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--reranker" in out
    assert "--rerank-top-n" in out
    assert "--reranker-model" in out


def test_ask_reranker_none_is_noop(tmp_path, ask_setup):
    """--reranker none produces trace with defaults and no rerank latency."""
    out_path = tmp_path / "trace.json"
    parser = build_parser()
    args = parser.parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--top-k", "2",
        "--reranker", "none",
        "--trace-out", str(out_path),
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()):
        cmd_ask(args)
    loaded = json.loads(out_path.read_text())
    assert loaded["reranker"] == "none"
    assert loaded["rerank_top_n"] is None
    assert "rerank" not in loaded["latency_by_stage"]
    for c in loaded["chunks"]:
        assert c["pre_rerank_rank"] is None
        assert c["pre_rerank_score"] is None


def test_ask_reranker_cross_encoder_with_fake(tmp_path, ask_setup):
    """Patched FakeReranker produces rerank trace fields in ask."""
    out_path = tmp_path / "trace.json"
    parser = build_parser()
    args = parser.parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--top-k", "2",
        "--reranker", "cross-encoder",
        "--rerank-top-n", "5",
        "--trace-out", str(out_path),
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker(name="cross-encoder")):
        cmd_ask(args)
    loaded = json.loads(out_path.read_text())
    assert loaded["reranker"] == "cross-encoder"
    assert loaded["rerank_top_n"] == 5
    assert "rerank" in loaded["latency_by_stage"]
    for c in loaded["chunks"]:
        assert c["pre_rerank_rank"] is not None
        assert c["pre_rerank_score"] is not None


def test_ask_reranker_cross_encoder_human_output(tmp_path, ask_setup, capsys):
    """Human-readable ask output shows reranker info when active."""
    parser = build_parser()
    args = parser.parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--top-k", "2",
        "--reranker", "cross-encoder",
        "--rerank-top-n", "5",
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()), \
         patch("tiny_rag_lab.cli._make_reranker", return_value=FakeReranker(name="cross-encoder")):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "reranker" in out
    assert "cross-encoder" in out


def test_ask_rerank_top_n_lt_top_k_exits_nonzero(ask_setup):
    """rerank_top_n < top_k with active reranker raises ValueError."""
    parser = build_parser()
    args = parser.parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--top-k", "5",
        "--reranker", "cross-encoder",
        "--rerank-top-n", "3",
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()), \
         pytest.raises(ValueError, match="rerank-top-n"):
        cmd_ask(args)


def test_ask_reranker_model_with_none_exits_nonzero(ask_setup):
    """--reranker-model with --reranker none raises ValueError."""
    parser = build_parser()
    args = parser.parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--reranker", "none",
        "--reranker-model", "some-model",
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator", side_effect=_fake_generator_factory()), \
         pytest.raises(ValueError, match="reranker-model"):
        cmd_ask(args)


# ---------------------------------------------------------------------------
# P2.0-T04 — cmd_ask --judge and --generator flags
# ---------------------------------------------------------------------------

from tiny_rag_lab.judge import FakeJudge


def _ask_args_with_judge(index_dir, judge="fake", generator="fake"):
    return build_parser().parse_args([
        "ask", "sample document",
        "--index-dir", str(index_dir),
        "--judge", judge,
        "--generator", generator,
    ])


def test_ask_parser_judge_default_is_none():
    args = build_parser().parse_args(["ask", "q", "--index-dir", "d"])
    assert args.judge == "none"


def test_ask_parser_generator_default_is_openai():
    args = build_parser().parse_args(["ask", "q", "--index-dir", "d"])
    assert args.generator == "openai"


def test_ask_parser_judge_choices():
    for choice in ("none", "fake", "openai"):
        args = build_parser().parse_args(["ask", "q", "--index-dir", "d", "--judge", choice])
        assert args.judge == choice


def test_ask_judge_none_output_has_no_verdict_block(ask_setup, capsys):
    """--judge none produces no Judge verdict block (identical to Phase 1.9)."""
    args = build_parser().parse_args([
        "ask", "sample document", "--index-dir", str(ask_setup),
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Judge verdict" not in out
    assert "Ask trace" in out


def test_ask_judge_fake_shows_verdict_block(ask_setup, capsys):
    """--judge fake --generator fake prints Judge verdict block."""
    args = _ask_args_with_judge(ask_setup)
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
        patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
    ):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Judge verdict" in out
    assert "Faithfulness" in out
    assert "Answer Relevance" in out
    assert "Citation Support" in out


def test_ask_judge_fake_verdict_block_after_answer(ask_setup, capsys):
    """Verdict block appears after the answer section."""
    args = _ask_args_with_judge(ask_setup)
    with (
        patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()),
        patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()),
        patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()),
    ):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert out.index("Answer:") < out.index("Judge verdict")


# ---------------------------------------------------------------------------
# P2.1-T03 — --context-budget and --output-format flags on rag ask
# ---------------------------------------------------------------------------

def _ask_args_with_budget(index_dir, budget, output_format="text", top_k=3, generator="fake"):
    argv = [
        "ask", "sample document",
        "--index-dir", str(index_dir),
        "--top-k", str(top_k),
        "--generator", generator,
        "--context-budget", str(budget),
        "--output-format", output_format,
    ]
    return build_parser().parse_args(argv)


def test_ask_parser_context_budget_default_is_zero():
    args = build_parser().parse_args(["ask", "q", "--index-dir", "d"])
    assert args.context_budget == 0


def test_ask_parser_output_format_default_is_text():
    args = build_parser().parse_args(["ask", "q", "--index-dir", "d"])
    assert args.output_format == "text"


def test_ask_parser_context_budget_flag():
    args = build_parser().parse_args(["ask", "q", "--index-dir", "d", "--context-budget", "8192"])
    assert args.context_budget == 8192


def test_ask_parser_output_format_json():
    args = build_parser().parse_args(["ask", "q", "--index-dir", "d", "--output-format", "json"])
    assert args.output_format == "json"


def test_ask_help_shows_context_budget_flag(capsys):
    try:
        build_parser().parse_args(["ask", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--context-budget" in out


def test_ask_help_shows_output_format_flag(capsys):
    try:
        build_parser().parse_args(["ask", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--output-format" in out


def test_ask_context_budget_zero_has_no_packing_block(ask_setup, capsys):
    """--context-budget 0 (default) produces no Context packing line."""
    args = _ask_args_with_budget(ask_setup, budget=0)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Context packing" not in out


def test_ask_context_budget_nonzero_shows_packing_block(ask_setup, capsys):
    """--context-budget 8192 shows Context packing block in text output."""
    args = _ask_args_with_budget(ask_setup, budget=8192)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert "Context packing" in out
    assert "budget=8192" in out


def test_ask_context_budget_packing_block_before_answer(ask_setup, capsys):
    """Context packing block appears before the answer section."""
    args = _ask_args_with_budget(ask_setup, budget=8192)
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    assert out.index("Context packing") < out.index("Answer:")


def test_ask_output_format_json_is_valid_json(ask_setup, capsys):
    """--output-format json prints valid JSON to stdout."""
    args = _ask_args_with_budget(ask_setup, budget=8192, output_format="json")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, dict)


def test_ask_output_format_json_has_answer_and_context_pack(ask_setup, capsys):
    """JSON output includes answer and context_pack keys when budget active."""
    args = _ask_args_with_budget(ask_setup, budget=8192, output_format="json")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "answer" in parsed
    assert "context_pack" in parsed
    assert parsed["context_pack"] is not None


def test_ask_output_format_json_budget_zero_context_pack_null(ask_setup, capsys):
    """JSON output has context_pack=null when budget=0."""
    args = _ask_args_with_budget(ask_setup, budget=0, output_format="json")
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["context_pack"] is None


def test_ask_negative_context_budget_raises(ask_setup):
    """--context-budget -1 raises ValueError."""
    args = build_parser().parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--context-budget", "-1",
        "--generator", "fake",
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()), \
         pytest.raises(ValueError, match="context-budget"):
        cmd_ask(args)


def test_ask_output_format_json_with_judge_includes_verdict(ask_setup, capsys):
    """--output-format json --judge fake includes verdict field in JSON."""
    args = build_parser().parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--generator", "fake",
        "--judge", "fake",
        "--context-budget", "8192",
        "--output-format", "json",
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_judge", return_value=FakeJudge()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "verdict" in parsed
    assert parsed["verdict"] is not None


def test_ask_trace_out_still_writes_json_with_json_output_format(tmp_path, ask_setup):
    """--trace-out writes JSON file even when --output-format json."""
    out_path = tmp_path / "trace.json"
    args = build_parser().parse_args([
        "ask", "sample document",
        "--index-dir", str(ask_setup),
        "--generator", "fake",
        "--context-budget", "8192",
        "--output-format", "json",
        "--trace-out", str(out_path),
    ])
    with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
         patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()):
        cmd_ask(args)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert "answer" in loaded


def test_ask_tight_budget_filters_chunks_vs_large_budget(ask_setup):
    """Tight budget produces fewer source markers than large budget (proves filtering)."""
    def _run(budget):
        args = build_parser().parse_args([
            "ask", "sample document",
            "--index-dir", str(ask_setup),
            "--top-k", "3",
            "--generator", "fake",
            "--context-budget", str(budget),
            "--output-format", "json",
        ])
        import io, sys
        buf = io.StringIO()
        with patch("tiny_rag_lab.cli._make_embedder", side_effect=_fake_embedder_factory()), \
             patch("tiny_rag_lab.cli._make_generator_from_flag", return_value=FakeGenerator()), \
             patch("sys.stdout", buf):
            cmd_ask(args)
        return json.loads(buf.getvalue())

    large = _run(8192)
    tight = _run(1)  # budget so small even overhead exceeds it → all omitted

    large_sources = len(re.findall(r"\[Source:", large["answer"]))
    tight_sources = len(re.findall(r"\[Source:", tight["answer"]))
    assert tight_sources < large_sources, (
        f"Tight budget should produce fewer source markers: {tight_sources} vs {large_sources}"
    )
