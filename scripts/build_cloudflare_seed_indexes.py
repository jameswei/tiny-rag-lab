"""Build the two reviewed Cloudflare NumPy indexes for image seed assets."""
from __future__ import annotations

import argparse
from pathlib import Path

from tiny_rag_lab.chunking import chunk_documents_with_strategy
from tiny_rag_lab.documents import load_documents
from tiny_rag_lab.embeddings import SentenceTransformerEmbedder
from tiny_rag_lab.index_writer import write_index


CORPUS_ID = "cloudflare-state-v1"
INDEXES = (
    ("cloudflare-state-structural-v1", "structural", 800, 120),
    ("cloudflare-state-fixed-v1", "fixed_character", 800, 120),
)


def build(corpus_dir: Path, indexes_dir: Path) -> None:
    corpus_dir, indexes_dir = Path(corpus_dir), Path(indexes_dir)
    docs = load_documents(corpus_dir / "files")
    if len(docs) != 40:
        raise ValueError(f"Expected 40 Cloudflare documents, found {len(docs)}")
    embedder = SentenceTransformerEmbedder(local_files_only=True)
    # Index artifacts are copied from the immutable image tree to `/data` at
    # startup. Persist the future runtime location, not the build-tree path,
    # so source metadata and citations remain valid after promotion.
    runtime_corpus_root = Path("/data/corpora") / CORPUS_ID / "files"
    for doc in docs:
        doc.path = str(runtime_corpus_root / doc.doc_id)
    for index_id, strategy, chunk_size, overlap in INDEXES:
        chunks = chunk_documents_with_strategy(
            docs, strategy=strategy, chunk_size=chunk_size, chunk_overlap=overlap,
        )
        embeddings = embedder.embed([chunk.text for chunk in chunks])
        target = indexes_dir / index_id
        if target.exists():
            import shutil
            shutil.rmtree(target)
        write_index(
            target, docs, chunks, embeddings, corpus_root=runtime_corpus_root,
            embedding_backend=type(embedder).__name__, embedding_model=embedder.model_name,
            embedding_revision=embedder.revision,
            embedding_dim=embedder.dim, chunk_size=chunk_size, chunk_overlap=overlap,
            chunking_strategy=strategy, index_backend="numpy", source_corpus_id=CORPUS_ID,
        )
        print(f"Built {index_id}: {len(docs)} documents, {len(chunks)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, default=Path("assets/seed/v2/corpora/cloudflare-state-v1"))
    parser.add_argument("--indexes-dir", type=Path, default=Path("assets/seed/v2/indexes"))
    args = parser.parse_args()
    build(args.corpus_dir, args.indexes_dir)


if __name__ == "__main__":
    main()
