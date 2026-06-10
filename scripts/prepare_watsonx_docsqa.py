#!/usr/bin/env python3
"""Prepare the watsonxDocsQA dataset as a local tiny-rag-lab corpus.

Real dataset: ibm-research/watsonxDocsQA on HuggingFace Hub.

Corpus split schema (fields this script reads):
  doc_id, title, md_document (preferred), document (fallback), url

QA split schema (fields this script reads):
  question_id, question, correct_answer, correct_answer_document_ids
  (older/cache variants may expose ground_truths_contexts_ids)

Produces:
  <output-dir>/
    docs/
      <slug>.md           one file per unique document
    dataset-manifest.json
    qa.jsonl              one QA record per line

The prepared doc_id in manifest and qa.jsonl gold_doc_ids are identical
corpus-relative paths, e.g. "docs/some-slug.md", so Phase 1.6 evaluation
can match retrieved Document.doc_id directly against qa.jsonl gold labels.

Usage — download from HuggingFace Hub:
  uv run python scripts/prepare_watsonx_docsqa.py \\
      --output-dir corpus/watsonx-docsqa

Usage — with a local HuggingFace cache:
  uv run python scripts/prepare_watsonx_docsqa.py \\
      --output-dir corpus/watsonx-docsqa \\
      --cache-dir /path/to/hf_cache

Usage — inspect dataset fields without writing files:
  uv run python scripts/prepare_watsonx_docsqa.py --inspect
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Real dataset identifier and schema
# ---------------------------------------------------------------------------

DATASET_NAME = "ibm-research/watsonxDocsQA"

# Corpus split: rows that have document content
_CORPUS_ID_FIELD = "doc_id"
_CORPUS_TITLE_FIELD = "title"
# md_document is the Markdown version; fall back to plain document
_CORPUS_TEXT_FIELDS = ("md_document", "document")
_CORPUS_URL_FIELD = "url"

# QA splits: rows that have questions
_QA_ID_FIELD = "question_id"
_QA_QUESTION_FIELD = "question"
_QA_ANSWER_FIELD = "correct_answer"
_QA_GOLD_ID_FIELDS = (
    "correct_answer_document_ids",  # current cached Parquet schema
    "ground_truths_contexts_ids",   # earlier schema name used in Phase 1 tests
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert a doc_id or title to a safe filename stem (max 80 chars)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text[:80].strip("-") or "doc"


def _pick_text(row: dict) -> str:
    """Return the best available document text from a corpus row."""
    for field in _CORPUS_TEXT_FIELDS:
        val = row.get(field, "")
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _is_corpus_row(row: dict) -> bool:
    """True if the row looks like a document (has a text field)."""
    return any(row.get(f, "") for f in _CORPUS_TEXT_FIELDS)


def _is_qa_row(row: dict) -> bool:
    """True if the row looks like a QA pair (has a question field)."""
    return bool(row.get(_QA_QUESTION_FIELD, ""))


def _cached_snapshot_path(dataset_name: str, cache_dir: str | None) -> Path | None:
    """Return a local hub snapshot path for dataset_name when cache_dir has one."""
    if not cache_dir or "/" not in dataset_name:
        return None

    owner, name = dataset_name.split("/", 1)
    repo_dir = Path(cache_dir) / "hub" / f"datasets--{owner}--{name}"
    ref_path = repo_dir / "refs" / "main"
    if not ref_path.exists():
        return None

    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    return snapshot if snapshot.exists() else None


def _extract_gold_ids(row: dict) -> list[str]:
    """Return raw gold document IDs from the first supported QA schema field."""
    for field in _QA_GOLD_ID_FIELDS:
        raw_gold = row.get(field)
        if raw_gold:
            break
    else:
        return []

    if isinstance(raw_gold, str):
        return [gid.strip() for gid in raw_gold.split(",") if gid.strip()]
    try:
        iterator = iter(raw_gold)
    except TypeError:
        gid = str(raw_gold).strip()
        return [gid] if gid else []
    return [str(gid).strip() for gid in iterator if str(gid).strip()]


# ---------------------------------------------------------------------------
# Conversion functions (pure — testable without HuggingFace)
# ---------------------------------------------------------------------------

def doc_to_markdown(title: str, text: str) -> str:
    """Render a document as Markdown with a top-level H1 title."""
    title = title.strip() or "Untitled"
    text = text.strip()
    return f"# {title}\n\n{text}\n"


def extract_documents(
    rows: list[dict],
) -> tuple[dict[str, dict], dict[str, str]]:
    """Extract unique documents from corpus rows.

    Returns:
        docs:    {original_doc_id: {title, text, prepared_id, slug}}
        id_map:  {original_doc_id: prepared_doc_id}

    prepared_doc_id is the corpus-relative path used as Document.doc_id in
    the RAG index, e.g. "docs/some-slug.md". id_map lets extract_qa_pairs
    translate gold doc IDs to the same prepared IDs.
    """
    docs: dict[str, dict] = {}
    id_map: dict[str, str] = {}

    for row in rows:
        if not _is_corpus_row(row):
            continue
        original_id = str(row.get(_CORPUS_ID_FIELD, "")).strip()
        if not original_id or original_id in docs:
            continue
        slug = _slugify(original_id)
        prepared_id = f"docs/{slug}.md"
        docs[original_id] = {
            "title": str(row.get(_CORPUS_TITLE_FIELD, original_id)).strip(),
            "text": _pick_text(row),
            "url": str(row.get(_CORPUS_URL_FIELD, "")),
            "slug": slug,
            "prepared_id": prepared_id,
        }
        id_map[original_id] = prepared_id

    return docs, id_map


def extract_qa_pairs(
    rows: list[dict],
    id_map: dict[str, str],
) -> list[dict]:
    """Extract QA pairs from QA split rows.

    gold_doc_ids in the output are translated through id_map so they match
    the prepared_id values used as Document.doc_id in the RAG index.
    Unknown IDs are preserved as-is with a warning.
    """
    pairs = []
    unknown_ids: set[str] = set()

    for row in rows:
        if not _is_qa_row(row):
            continue
        question = str(row.get(_QA_QUESTION_FIELD, "")).strip()
        if not question:
            continue

        gold_doc_ids = []
        for gid in _extract_gold_ids(row):
            if gid in id_map:
                gold_doc_ids.append(id_map[gid])
            else:
                gold_doc_ids.append(gid)
                unknown_ids.add(gid)

        pairs.append(
            {
                "question_id": str(row.get(_QA_ID_FIELD, "")).strip(),
                "question": question,
                "answer": str(row.get(_QA_ANSWER_FIELD, "")).strip(),
                "gold_doc_ids": gold_doc_ids,
            }
        )

    if unknown_ids:
        print(
            f"  Warning: {len(unknown_ids)} gold doc ID(s) not found in corpus "
            f"(first few: {sorted(unknown_ids)[:3]})"
        )

    return pairs


def build_manifest(
    dataset_name: str,
    doc_records: list[dict],
    source_url: str = "",
    license_str: str = "see dataset source",
) -> dict:
    """Return a manifest dict (not yet written to disk)."""
    return {
        "dataset_name": dataset_name,
        "source_url": source_url,
        "license": license_str,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(doc_records),
        "documents": doc_records,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_corpus(output_dir: Path, docs: dict[str, dict]) -> list[dict]:
    """Write docs/ Markdown files and return manifest document records."""
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for original_id, info in docs.items():
        filepath = docs_dir / f"{info['slug']}.md"
        filepath.write_text(
            doc_to_markdown(info["title"], info["text"]), encoding="utf-8"
        )
        records.append(
            {
                "doc_id": info["prepared_id"],       # corpus-relative, matches qa.jsonl
                "original_doc_id": original_id,      # original dataset ID for traceability
                "path": str(filepath),
                "title": info["title"],
                "url": info["url"],
            }
        )

    return records


def write_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "dataset-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  manifest  → {path}")


def write_qa_jsonl(output_dir: Path, pairs: list[dict]) -> None:
    path = output_dir / "qa.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"  qa.jsonl  → {path}  ({len(pairs)} pairs)")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_splits(dataset_name: str, cache_dir: str | None) -> tuple[list[dict], list[dict]]:
    """Load and separate corpus rows from QA rows.

    Returns (corpus_rows, qa_rows).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Error: 'datasets' package not found. Run: uv sync")

    dataset_path = _cached_snapshot_path(dataset_name, cache_dir) or dataset_name
    print(f"Loading '{dataset_path}' from HuggingFace Hub/cache...")
    kwargs: dict = {}
    if cache_dir and dataset_path == dataset_name:
        kwargs["cache_dir"] = cache_dir
    try:
        ds = load_dataset(str(dataset_path), **kwargs)
    except ValueError as exc:
        if "Config name is missing" not in str(exc):
            raise
        corpus_ds = load_dataset(str(dataset_path), "corpus", **kwargs)
        qa_ds = load_dataset(str(dataset_path), "question_answers", **kwargs)
        ds = {"corpus": corpus_ds["train"], **{f"question_answers/{k}": v for k, v in qa_ds.items()}}

    corpus_rows: list[dict] = []
    qa_rows: list[dict] = []

    for split_name, split in ds.items():
        rows = split.to_list()
        print(f"  split '{split_name}': {len(rows)} rows")
        for row in rows:
            if _is_corpus_row(row):
                corpus_rows.append(row)
            elif _is_qa_row(row):
                qa_rows.append(row)

    return corpus_rows, qa_rows


def inspect_dataset(dataset_name: str, cache_dir: str | None) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Error: 'datasets' package not found. Run: uv sync")

    dataset_path = _cached_snapshot_path(dataset_name, cache_dir) or dataset_name
    kwargs = {"cache_dir": cache_dir} if cache_dir and dataset_path == dataset_name else {}
    try:
        ds = load_dataset(str(dataset_path), **kwargs)
    except ValueError as exc:
        if "Config name is missing" not in str(exc):
            raise
        corpus_ds = load_dataset(str(dataset_path), "corpus", **kwargs)
        qa_ds = load_dataset(str(dataset_path), "question_answers", **kwargs)
        ds = {"corpus": corpus_ds["train"], **{f"question_answers/{k}": v for k, v in qa_ds.items()}}
    print(f"\nDataset: {dataset_name}")
    print(f"Splits:  {list(ds.keys())}")
    for split_name, split in ds.items():
        print(f"\n--- Split '{split_name}' ({len(split)} rows) ---")
        print("Features:", {k: str(v) for k, v in split.features.items()})
        if len(split) > 0:
            print("Sample row:")
            for k, v in split[0].items():
                print(f"  {k}: {str(v)[:120]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prepare_watsonx_docsqa",
        description="Prepare ibm-research/watsonxDocsQA as a local corpus for tiny-rag-lab.",
    )
    p.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        metavar="NAME",
        help=f"HuggingFace dataset identifier (default: {DATASET_NAME})",
    )
    p.add_argument(
        "--output-dir",
        default="corpus/watsonx-docsqa",
        metavar="PATH",
        help="where to write prepared corpus (default: corpus/watsonx-docsqa)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        metavar="PATH",
        help="HuggingFace cache directory (optional)",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="print dataset structure and exit without writing files",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.inspect:
        inspect_dataset(args.dataset_name, args.cache_dir)
        return

    corpus_rows, qa_rows = load_dataset_splits(args.dataset_name, args.cache_dir)

    if not corpus_rows:
        sys.exit("Error: no corpus rows found. Run with --inspect to check split structure.")

    print(f"\nCorpus rows: {len(corpus_rows)}, QA rows: {len(qa_rows)}")

    docs, id_map = extract_documents(corpus_rows)
    qa_pairs = extract_qa_pairs(qa_rows, id_map)

    print(f"Unique documents: {len(docs)}, QA pairs: {len(qa_pairs)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting corpus to {output_dir}/")
    doc_records = write_corpus(output_dir, docs)
    manifest = build_manifest(args.dataset_name, doc_records)
    write_manifest(output_dir, manifest)
    write_qa_jsonl(output_dir, qa_pairs)

    print(f"\nDone. {len(doc_records)} documents, {len(qa_pairs)} QA pairs.")


if __name__ == "__main__":
    main()
