"""Versioned, inspectable seed assets for the local learning lab.

The container image owns a read-only seed directory.  A user's named `/data`
volume owns the working copies.  This module deliberately keeps that boundary
small and visible: every copied file has a digest in ``seed-manifest.json`` and
promotion happens only after a staged copy verifies successfully.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED_MANIFEST_NAME = "seed-manifest.json"
SEED_STATE_NAME = ".seed-state.json"
SEED_STAGING_NAME = ".seed-staging"


class SeedAssetError(RuntimeError):
    """A bundled seed is malformed, corrupt, or conflicts with local data."""


@dataclass(frozen=True)
class SeedResult:
    asset_id: str
    status: str
    detail: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_seed_manifest(seed_root: Path) -> dict[str, Any]:
    path = Path(seed_root) / SEED_MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SeedAssetError(f"Seed manifest not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SeedAssetError(f"Seed manifest is invalid JSON: {path}") from exc
    if not isinstance(manifest.get("seed_version"), str) or not isinstance(manifest.get("assets"), list):
        raise SeedAssetError("Seed manifest requires seed_version and assets")
    return manifest


def verify_asset(root: Path, asset: dict[str, Any]) -> None:
    """Verify every declared asset file and reject undeclared path traversal."""
    for entry in asset.get("files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SeedAssetError(f"Invalid seed file path: {entry['path']}")
        candidate = Path(root) / relative
        if not candidate.is_file():
            raise SeedAssetError(f"Seed file missing: {candidate}")
        if sha256_file(candidate) != entry.get("sha256"):
            raise SeedAssetError(f"Seed digest mismatch: {candidate}")


def seed_bundled_assets(data_root: Path, seed_root: Path | None) -> list[SeedResult]:
    """Promote verified immutable assets into an empty or managed local volume.

    A missing seed directory is normal for native development.  In that case
    the lab remains usable with custom uploads and returns no seed results.
    """
    if seed_root is None or not Path(seed_root).exists():
        return []
    seed_root = Path(seed_root)
    manifest = load_seed_manifest(seed_root)
    data_root = Path(data_root)
    state_path = data_root / SEED_STATE_NAME
    state = _load_state(state_path)
    results: list[SeedResult] = []
    staging_root = data_root / SEED_STAGING_NAME
    shutil.rmtree(staging_root, ignore_errors=True)

    for asset in manifest["assets"]:
        _validate_asset(asset)
        source = seed_root / asset["path"]
        verify_asset(source, asset)
        target = data_root / asset["target"]
        prior = state.get("assets", {}).get(asset["id"])

        if not target.exists():
            _promote(source, target, asset, staging_root)
            state.setdefault("assets", {})[asset["id"]] = _state_entry(manifest, asset)
            _write_state(state_path, state)
            results.append(SeedResult(asset["id"], "seeded"))
            continue

        # A process can stop after the verified directory is published but
        # before its ownership state is replaced.  Only recover a target that
        # was already managed and exactly matches the incoming manifest; an
        # unmanaged lookalike remains a conflict by design.
        if prior and _files_status(target, asset["files"]) == "matches":
            incoming = _state_entry(manifest, asset)
            if prior == incoming:
                results.append(SeedResult(asset["id"], "ready"))
            else:
                state.setdefault("assets", {})[asset["id"]] = incoming
                _write_state(state_path, state)
                shutil.rmtree(target.with_name(f".{target.name}.seed-backup"), ignore_errors=True)
                results.append(SeedResult(asset["id"], "recovered"))
            continue

        local_status = _managed_status(target, prior)
        if local_status == "matches":
            _promote(source, target, asset, staging_root)
            state.setdefault("assets", {})[asset["id"]] = _state_entry(manifest, asset)
            _write_state(state_path, state)
            results.append(SeedResult(asset["id"], "upgraded"))
            continue

        if local_status == "missing_files":
            _promote(source, target, asset, staging_root)
            state.setdefault("assets", {})[asset["id"]] = _state_entry(manifest, asset)
            _write_state(state_path, state)
            results.append(SeedResult(asset["id"], "repaired"))
            continue

        results.append(SeedResult(asset["id"], "conflict", local_status))

    shutil.rmtree(staging_root, ignore_errors=True)
    _write_state(state_path, state)
    return results


def _validate_asset(asset: dict[str, Any]) -> None:
    required = ("id", "path", "target", "files")
    if any(not asset.get(key) for key in required):
        raise SeedAssetError("Each seed asset requires id, path, target, and files")
    for key in ("path", "target"):
        candidate = Path(asset[key])
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SeedAssetError(f"Invalid seed asset {key}: {asset[key]}")


def _managed_status(target: Path, prior: dict[str, Any] | None) -> str:
    if not prior:
        return "unmanaged_target"
    return _files_status(target, prior.get("files", []))


def _files_status(target: Path, expected: list[dict[str, Any]]) -> str:
    """Compare one directory with an explicit manifest file set."""
    expected_paths = {entry["path"] for entry in expected}
    actual_paths = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*") if path.is_file()
    }
    if actual_paths - expected_paths:
        return "unexpected_files"
    if any(not (target / entry["path"]).is_file() for entry in expected):
        return "missing_files"
    if any(sha256_file(target / entry["path"]) != entry.get("sha256") for entry in expected):
        return "digest_mismatch"
    return "matches"


def _promote(source: Path, target: Path, asset: dict[str, Any], staging_root: Path) -> None:
    staging = staging_root / asset["id"]
    shutil.copytree(source, staging)
    verify_asset(staging, asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.seed-backup")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    os.replace(staging, target)
    shutil.rmtree(backup, ignore_errors=True)


def _state_entry(manifest: dict[str, Any], asset: dict[str, Any]) -> dict[str, Any]:
    return {"seed_version": manifest["seed_version"], "files": asset["files"]}


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "assets": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema_version": 1, "assets": {}}
    return state if isinstance(state.get("assets"), dict) else {"schema_version": 1, "assets": {}}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
