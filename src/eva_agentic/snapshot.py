"""Fingerprinting for immutable experiment inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Return the size and SHA-256 digest of a file."""
    source = Path(path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {
        "path": source.name,
        "size": size,
        "algorithm": "sha256",
        "sha256": digest.hexdigest(),
    }


def create_input_manifest(inputs_dir: str | Path) -> dict[str, Any]:
    """Fingerprint every input file except the manifest itself."""
    directory = Path(inputs_dir)
    files = [
        fingerprint_file(path)
        for path in sorted(directory.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != MANIFEST_NAME
    ]
    if not files:
        raise ValueError(f"input directory is empty: {directory}")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": files,
    }


def verify_input_manifest(inputs_dir: str | Path, manifest: Mapping[str, Any]) -> None:
    """Verify the exact input file set, sizes, and content digests."""
    directory = Path(inputs_dir)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported input manifest schema_version")
    if manifest.get("algorithm") not in {None, "sha256"}:
        raise ValueError("unsupported input manifest algorithm")

    expected_items = manifest.get("files")
    if not isinstance(expected_items, list) or not expected_items:
        raise ValueError("input manifest must contain a non-empty files list")

    expected = {Path(str(item["path"])).name: item for item in expected_items}
    actual_names = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name != MANIFEST_NAME
    }
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"input files changed: missing={missing}, extra={extra}")

    for name, expected_item in expected.items():
        actual = fingerprint_file(directory / name)
        if actual["size"] != expected_item.get("size"):
            raise ValueError(f"input file size changed: {name}")
        if actual["sha256"] != expected_item.get("sha256"):
            raise ValueError(f"input file digest changed: {name}")
