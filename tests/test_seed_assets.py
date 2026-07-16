import hashlib
import json
from pathlib import Path

import pytest

from tiny_rag_lab.browser_eval import validate_evaluation_bundle
from tiny_rag_lab.seed_assets import SeedAssetError, load_seed_manifest, seed_bundled_assets, verify_asset


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(tmp_path: Path, *, version: str = "v1", text: str = "seed") -> Path:
    root = tmp_path / "image-seed"
    asset_root = root / "corpora" / "cloudflare-state-v1"
    asset_root.mkdir(parents=True, exist_ok=True)
    source = asset_root / "source.md"
    source.write_text(text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "seed_version": version,
        "assets": [{
            "id": "cloudflare-state-v1",
            "path": "corpora/cloudflare-state-v1",
            "target": "corpora/cloudflare-state-v1",
            "files": [{"path": "source.md", "sha256": _digest(source)}],
        }],
    }
    (root / "seed-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_seed_copies_missing_asset_and_is_idempotent(tmp_path: Path):
    root = _seed(tmp_path)
    data = tmp_path / "data"

    assert [result.status for result in seed_bundled_assets(data, root)] == ["seeded"]
    assert (data / "corpora/cloudflare-state-v1/source.md").read_text() == "seed"
    assert [result.status for result in seed_bundled_assets(data, root)] == ["ready"]


def test_seed_repairs_a_partial_managed_asset(tmp_path: Path):
    root = _seed(tmp_path)
    data = tmp_path / "data"
    seed_bundled_assets(data, root)
    (data / "corpora/cloudflare-state-v1/source.md").unlink()

    assert [result.status for result in seed_bundled_assets(data, root)] == ["repaired"]
    assert (data / "corpora/cloudflare-state-v1/source.md").read_text() == "seed"


def test_seed_never_overwrites_an_unmanaged_or_modified_target(tmp_path: Path):
    root = _seed(tmp_path)
    data = tmp_path / "data"
    target = data / "corpora/cloudflare-state-v1"
    target.mkdir(parents=True)
    (target / "source.md").write_text("user", encoding="utf-8")

    assert [result.status for result in seed_bundled_assets(data, root)] == ["conflict"]
    assert (target / "source.md").read_text() == "user"


def test_seed_rejects_a_corrupt_image_asset_before_copying(tmp_path: Path):
    root = _seed(tmp_path)
    (root / "corpora/cloudflare-state-v1/source.md").write_text("changed", encoding="utf-8")

    with pytest.raises(SeedAssetError, match="digest mismatch"):
        seed_bundled_assets(tmp_path / "data", root)


def test_seed_discards_stale_staging_before_promotion(tmp_path: Path):
    root = _seed(tmp_path)
    data = tmp_path / "data"
    stale = data / ".seed-staging" / "interrupted"
    stale.mkdir(parents=True)
    (stale / "partial.md").write_text("partial", encoding="utf-8")

    seed_bundled_assets(data, root)

    assert not (data / ".seed-staging").exists()


def test_seed_upgrades_a_matching_prior_managed_version(tmp_path: Path):
    root = _seed(tmp_path, version="v1", text="v1")
    data = tmp_path / "data"
    seed_bundled_assets(data, root)
    _seed(tmp_path, version="v2", text="v2")

    assert [result.status for result in seed_bundled_assets(data, root)] == ["upgraded"]
    assert (data / "corpora/cloudflare-state-v1/source.md").read_text() == "v2"


def test_seed_recovers_when_publish_finished_before_state_update(tmp_path: Path):
    root = _seed(tmp_path, version="v1", text="v1")
    data = tmp_path / "data"
    seed_bundled_assets(data, root)
    _seed(tmp_path, version="v2", text="v2")

    # Reproduce the durable state immediately after an interrupted promotion:
    # the v2 directory is published, but .seed-state.json still owns v1.
    target = data / "corpora/cloudflare-state-v1"
    (target / "source.md").write_text("v2", encoding="utf-8")
    backup = target.with_name(".cloudflare-state-v1.seed-backup")
    backup.mkdir()
    (backup / "source.md").write_text("v1", encoding="utf-8")

    result = seed_bundled_assets(data, root)[0]

    assert result.status == "recovered"
    assert not backup.exists()
    state = json.loads((data / ".seed-state.json").read_text())
    assert state["assets"]["cloudflare-state-v1"]["seed_version"] == "v2"
    assert seed_bundled_assets(data, root)[0].status == "ready"


def test_seed_preserves_modified_managed_target_and_extra_user_file(tmp_path: Path):
    root = _seed(tmp_path)
    data = tmp_path / "data"
    seed_bundled_assets(data, root)
    target = data / "corpora/cloudflare-state-v1"
    (target / "user.md").write_text("keep", encoding="utf-8")
    _seed(tmp_path, version="v2", text="v2")

    result = seed_bundled_assets(data, root)[0]
    assert result.status == "conflict"
    assert result.detail == "unexpected_files"
    assert (target / "user.md").read_text() == "keep"
    assert (target / "source.md").read_text() == "seed"


def test_seed_preserves_a_modified_declared_managed_file(tmp_path: Path):
    root = _seed(tmp_path)
    data = tmp_path / "data"
    seed_bundled_assets(data, root)
    target = data / "corpora/cloudflare-state-v1/source.md"
    target.write_text("edited", encoding="utf-8")

    result = seed_bundled_assets(data, root)[0]
    assert result.status == "conflict"
    assert result.detail == "digest_mismatch"
    assert target.read_text() == "edited"


def test_bundled_cloudflare_indexes_reference_promoted_data_paths():
    seed_root = Path(__file__).parents[1] / "assets" / "seed" / "v1"
    for index_id in ("cloudflare-state-structural-v1", "cloudflare-state-fixed-v1"):
        manifest = json.loads((seed_root / "indexes" / index_id / "manifest.json").read_text())
        assert manifest["source_corpus_id"] == "cloudflare-state-v1"
        assert manifest["corpus_root"] == "/data/corpora/cloudflare-state-v1/files"
        assert all(entry["path"].startswith(manifest["corpus_root"] + "/") for entry in manifest["corpus_files"])
        first_chunk = json.loads((seed_root / "indexes" / index_id / "chunks.jsonl").read_text().splitlines()[0])
        assert first_chunk["metadata"]["path"].startswith(manifest["corpus_root"] + "/")


def test_v2_seed_manifest_verifies_and_promotes_reviewed_evaluation_bundle(tmp_path):
    seed_root = Path(__file__).parents[1] / "assets" / "seed" / "v2"
    manifest = load_seed_manifest(seed_root)
    assert manifest["seed_version"] == "v2"
    for asset in manifest["assets"]:
        verify_asset(seed_root / asset["path"], asset)

    results = seed_bundled_assets(tmp_path / "data", seed_root)
    assert all(result.status == "seeded" for result in results)
    index, questions, evaluation = validate_evaluation_bundle(
        tmp_path / "data/indexes/cloudflare-state-structural-v1",
        tmp_path / "data/corpora/cloudflare-state-v1/retrieval-questions.jsonl",
        tmp_path / "data/corpora/cloudflare-state-v1/evaluation-manifest.json",
    )
    assert len(index.chunks) == 537
    assert len(questions) == 16
    assert evaluation["question_count"] == 16
    assert index.manifest["embedding_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
