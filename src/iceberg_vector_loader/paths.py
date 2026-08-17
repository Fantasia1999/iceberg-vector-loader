from __future__ import annotations

from enum import Enum
from pathlib import Path
from urllib.parse import urlparse, unquote


class PathStyle(str, Enum):
    PATH = "path"
    URI = "uri"


def resolve_local_path(raw: str | Path) -> Path:
    """Turn a filesystem path or file:// URI into an absolute Path."""
    text = str(raw)
    if text.startswith("file:"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
    return Path(text).expanduser().resolve()


def format_location(path: str | Path, style: PathStyle | str) -> str:
    """Format an absolute location for Iceberg metadata."""
    resolved = resolve_local_path(path)
    style = PathStyle(style)
    if style is PathStyle.URI:
        return resolved.as_uri()
    return str(resolved)


def has_file_scheme(location: str) -> bool:
    return location.startswith("file:")
