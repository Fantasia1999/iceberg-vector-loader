from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from iceberg_vector_loader.paths import PathStyle, format_location, has_file_scheme


def strip_file_scheme(location: str) -> str:
    if not has_file_scheme(location):
        return location
    parsed = urlparse(location)
    return unquote(parsed.path)


def rewrite_metadata_locations(metadata_path: Path, style: PathStyle) -> None:
    """Normalize location strings already written by Spark.

    Hadoop FileIO often persists file:// even when the warehouse was a bare path.
    For path-style we strip the scheme from metadata.json location fields.
    Manifest avro paths are left as Spark wrote them; Spark still reads them.
    """
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    changed = False

    def convert(value: str) -> str:
        if style is PathStyle.PATH:
            return strip_file_scheme(value)
        if has_file_scheme(value):
            return value
        return format_location(value, PathStyle.URI)

    if isinstance(data.get("location"), str):
        new = convert(data["location"])
        if new != data["location"]:
            data["location"] = new
            changed = True

    for snapshot in data.get("snapshots") or []:
        for key in ("manifest-list",):
            if isinstance(snapshot.get(key), str):
                new = convert(snapshot[key])
                if new != snapshot[key]:
                    snapshot[key] = new
                    changed = True

    for entry in data.get("metadata-log") or []:
        if isinstance(entry.get("metadata-file"), str):
            new = convert(entry["metadata-file"])
            if new != entry["metadata-file"]:
                entry["metadata-file"] = new
                changed = True

    if changed:
        metadata_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
