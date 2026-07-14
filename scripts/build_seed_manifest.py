"""Write file-level SHA-256 metadata for immutable image seed assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ASSETS = (
    ("cloudflare-state-v1", "corpora/cloudflare-state-v1", "corpora/cloudflare-state-v1"),
    ("watsonxdocsqa-v1", "corpora/watsonxdocsqa-v1", "corpora/watsonxdocsqa-v1"),
    ("cloudflare-state-structural-v1", "indexes/cloudflare-state-structural-v1", "indexes/cloudflare-state-structural-v1"),
    ("cloudflare-state-fixed-v1", "indexes/cloudflare-state-fixed-v1", "indexes/cloudflare-state-fixed-v1"),
    ("cloudflare-state-coordination-v1", "lessons/cloudflare-state-coordination-v1", "lessons/cloudflare-state-coordination-v1"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(seed_root: Path, seed_version: str) -> dict:
    seed_root = Path(seed_root)
    assets = []
    for asset_id, relative_path, target in ASSETS:
        root = seed_root / relative_path
        if not root.is_dir():
            raise ValueError(f"Seed asset directory missing: {root}")
        files = [
            {"path": path.relative_to(root).as_posix(), "sha256": _sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file()
        ]
        assets.append({"id": asset_id, "path": relative_path, "target": target, "files": files})
    manifest = {"schema_version": 1, "seed_version": seed_version, "assets": assets}
    (seed_root / "seed-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", type=Path, default=Path("assets/seed/v1"))
    parser.add_argument("--seed-version", default="v1")
    args = parser.parse_args()
    manifest = build(args.seed_root, args.seed_version)
    print(f"Wrote {len(manifest['assets'])} seed assets to {args.seed_root / 'seed-manifest.json'}")


if __name__ == "__main__":
    main()
