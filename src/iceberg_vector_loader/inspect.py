from __future__ import annotations

import json
from pathlib import Path

from iceberg_vector_loader.paths import has_file_scheme, resolve_local_path


def table_dir(warehouse: str | Path, namespace: str, table: str) -> Path:
    return resolve_local_path(warehouse) / namespace / table


def table_exists_on_fs(warehouse: str | Path, namespace: str, table: str) -> bool:
    meta = table_dir(warehouse, namespace, table) / "metadata"
    return meta.is_dir() and any(meta.glob("*.metadata.json"))


def latest_metadata_path(warehouse: str | Path, namespace: str, table: str) -> Path:
    metadata_dir = table_dir(warehouse, namespace, table) / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"no Iceberg metadata directory at {metadata_dir}")
    hint = metadata_dir / "version-hint.text"
    files = sorted(metadata_dir.glob("*.metadata.json"))
    if not files:
        raise FileNotFoundError(f"no metadata.json under {metadata_dir}")
    if hint.exists():
        raw = hint.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            version = int(raw)
            candidates = [
                metadata_dir / f"v{version}.metadata.json",
                *[path for path in files if path.name.startswith(f"{version:05d}-")],
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        hinted = metadata_dir / raw
        if hinted.exists():
            return hinted
    return max(files, key=lambda path: path.stat().st_mtime)


def load_metadata(warehouse: str | Path, namespace: str, table: str) -> dict:
    path = latest_metadata_path(warehouse, namespace, table)
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_total_records(metadata: dict) -> int:
    current = metadata.get("current-snapshot-id")
    for snapshot in metadata.get("snapshots") or []:
        if snapshot.get("snapshot-id") == current:
            summary = snapshot.get("summary") or {}
            if "total-records" in summary:
                return int(summary["total-records"])
    return 0


def assert_path_style(metadata: dict, want_uri: bool) -> None:
    location = metadata.get("location") or ""
    if want_uri:
        if not has_file_scheme(location):
            raise RuntimeError(f"expected file:// table location, got {location}")
        return
    if has_file_scheme(location):
        raise RuntimeError(f"table location unexpectedly uses a file URI: {location}")
