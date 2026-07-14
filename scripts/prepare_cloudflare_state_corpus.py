"""Fetch the reviewed Cloudflare documentation slice into a reproducible seed.

The source revision and paths are deliberately listed here rather than hidden
behind a site crawler.  Re-running the script at the same revision produces a
small Markdown snapshot whose provenance can be inspected before indexing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REVISION = "3dcb728cb29f4239e08ba894f0a40650d51ba4f6"
REPOSITORY = "cloudflare/cloudflare-docs"
SOURCE_PREFIX = "src/content/docs"
CORPUS_ID = "cloudflare-state-v1"

SOURCE_PATHS = [
    "workers/index.mdx", "workers/get-started/guide.mdx",
    "workers/runtime-apis/bindings/durable-objects.mdx",
    "workers/runtime-apis/bindings/queues.mdx", "workers/runtime-apis/bindings/kv.mdx",
    "workers/runtime-apis/bindings/R2.mdx",
    "durable-objects/concepts/what-are-durable-objects.mdx",
    "durable-objects/concepts/durable-object-lifecycle.mdx",
    "durable-objects/platform/storage-options.mdx",
    "durable-objects/best-practices/rules-of-durable-objects.mdx",
    "durable-objects/best-practices/create-durable-object-stubs-and-send-requests.mdx",
    "durable-objects/api/state.mdx", "durable-objects/api/alarms.mdx",
    "durable-objects/examples/build-a-counter.mdx",
    "queues/index.mdx", "queues/get-started.mdx", "queues/reference/how-queues-works.mdx",
    "queues/reference/delivery-guarantees.mdx", "queues/configuration/batching-retries.mdx",
    "queues/configuration/consumer-concurrency.mdx",
    "queues/examples/use-queues-with-durable-objects.mdx",
    "kv/index.mdx", "kv/concepts/how-kv-works.mdx", "kv/concepts/kv-bindings.mdx",
    "kv/concepts/kv-namespaces.mdx", "kv/examples/distributed-configuration-with-workers-kv.mdx",
    "kv/examples/cache-data-with-workers-kv.mdx",
    "r2/index.mdx", "r2/how-r2-works.mdx", "r2/reference/consistency.mdx",
    "r2/reference/durability.mdx", "r2/api/workers/workers-api-reference.mdx",
    "r2/buckets/create-buckets.mdx",
    "workflows/index.mdx", "workflows/get-started/guide.mdx",
    "workflows/build/rules-of-workflows.mdx", "workflows/build/sleeping-and-retrying.mdx",
    "workflows/build/step-context.mdx", "workflows/build/trigger-workflows.mdx",
    "workflows/build/workers-api.mdx",
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare(output_dir: Path, source_root: Path | None = None) -> dict:
    output_dir = Path(output_dir)
    files_dir = output_dir / "files"
    documents = []
    for source_path in SOURCE_PATHS:
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{SOURCE_PREFIX}/{source_path}"
        if source_root is None:
            with urlopen(url, timeout=30) as response:  # nosec B310 - pinned public source
                text = response.read().decode("utf-8")
        else:
            text = (Path(source_root) / SOURCE_PREFIX / source_path).read_text(encoding="utf-8")
        destination = files_dir / Path(source_path).with_suffix(".md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        documents.append({
            "source_path": source_path,
            "path": destination.relative_to(output_dir).as_posix(),
            "source_url": f"https://github.com/{REPOSITORY}/blob/{REVISION}/{SOURCE_PREFIX}/{source_path}",
            "sha256": _sha256(text),
        })
    corpus = {"id": CORPUS_ID, "name": "Cloudflare State & Coordination", "kind": "catalog", "file_count": len(documents)}
    provenance = {
        "schema_version": 1, "repository": REPOSITORY, "revision": REVISION,
        "license": "CC BY 4.0", "documents": documents,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    (output_dir / "source-manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("assets/seed/v1/corpora/cloudflare-state-v1"))
    parser.add_argument("--source-root", type=Path, help="unpacked cloudflare-docs revision; avoids network fetches")
    args = parser.parse_args()
    result = prepare(args.output_dir, args.source_root)
    print(f"Prepared {len(result['documents'])} Cloudflare documents at {args.output_dir}")


if __name__ == "__main__":
    main()
