from pathlib import Path

from iceberg_vector_loader.paths import PathStyle, format_location, has_file_scheme, resolve_local_path


def test_resolve_plain_and_uri(tmp_path: Path) -> None:
    target = tmp_path / "warehouse"
    target.mkdir()
    assert resolve_local_path(target) == target.resolve()
    assert resolve_local_path(target.resolve().as_uri()) == target.resolve()


def test_format_location_path_has_no_scheme(tmp_path: Path) -> None:
    location = format_location(tmp_path, PathStyle.PATH)
    assert location == str(tmp_path.resolve())
    assert not has_file_scheme(location)


def test_format_location_uri(tmp_path: Path) -> None:
    location = format_location(tmp_path, PathStyle.URI)
    assert location.startswith("file://")
    assert location.endswith(str(tmp_path.resolve()))
