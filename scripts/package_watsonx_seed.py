"""Package an already prepared watsonxDocsQA snapshot as a lab seed asset."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


CORPUS_ID = "watsonxdocsqa-v1"


def package(source_dir: Path, output_dir: Path) -> None:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    docs = source_dir / "docs"
    qa = source_dir / "qa.jsonl"
    dataset = source_dir / "dataset-manifest.json"
    if not docs.is_dir() or not qa.is_file() or not dataset.is_file():
        raise ValueError("source_dir must be a prepared watsonxDocsQA corpus")
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.copytree(docs, output_dir / "files" / "docs")
    shutil.copy2(qa, output_dir / "questions.jsonl")
    shutil.copy2(dataset, output_dir / "dataset-manifest.json")
    file_count = len(list((output_dir / "files").rglob("*.md")))
    corpus = {"id": CORPUS_ID, "name": "watsonxDocsQA", "kind": "catalog", "file_count": file_count}
    (output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("corpus/watsonx-docsqa"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/seed/v1/corpora/watsonxdocsqa-v1"))
    args = parser.parse_args()
    package(args.source_dir, args.output_dir)
    print(f"Packaged watsonxDocsQA at {args.output_dir}")


if __name__ == "__main__":
    main()
