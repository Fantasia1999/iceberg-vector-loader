from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from iceberg_vector_loader.fvecs import convert_fvecs_to_parquet
from iceberg_vector_loader.paths import resolve_local_path
from iceberg_vector_loader.schema import (
    EMBEDDING_FIELD,
    ColumnMapping,
    identity_mapping,
    infer_column_mapping,
    normalize_table,
    remapped_schema,
)

DEFAULT_BATCH_SIZE = 250_000
FVECS_SUFFIXES = {".fvecs"}
PARQUET_SUFFIXES = {".parquet", ".parq"}
_GLOB_METACHARS = set("*?[]")
_SHARD_NAME = re.compile(
    r"^(?P<prefix>.+-)(?P<index>\d+)-of-(?P<total>\d+)(?P<suffix>\.parq(?:uet)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedInput:
    path: Path
    dimension: int
    schema: pa.Schema
    mapping: ColumnMapping


def is_fvecs(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in FVECS_SUFFIXES


def is_parquet_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in PARQUET_SUFFIXES


def is_parquet_input(path: Path) -> bool:
    if path.is_dir():
        return True
    return is_parquet_file(path)


def looks_like_glob(text: str) -> bool:
    return any(char in text for char in _GLOB_METACHARS)


def absolute_glob(pattern: str) -> Path:
    raw = Path(pattern).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return Path(os.path.normpath(str(raw)))


def expand_parquet_glob(pattern: str) -> list[Path]:
    absolute = absolute_glob(pattern)
    matches = [Path(item).resolve() for item in sorted(glob.glob(str(absolute), recursive=True))]
    files = [path for path in matches if is_parquet_file(path)]
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern}")
    return files


def list_parquet_files(directory: Path) -> list[Path]:
    files = sorted(
        path
        for path in directory.rglob("*")
        if is_parquet_file(path) and not path.name.startswith(".") and not path.name.startswith("_")
    )
    if not files:
        raise ValueError(f"no parquet files in directory {directory}")
    return files


def parquet_schema(source: Path | list[Path]) -> pa.Schema:
    if isinstance(source, list):
        return assert_same_schema(source)
    if source.is_dir():
        return assert_same_schema(list_parquet_files(source))
    return pq.read_schema(source)


def assert_same_schema(files: list[Path]) -> pa.Schema:
    if not files:
        raise ValueError("no parquet files to read")
    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    schemas: dict[tuple[tuple[str, str], ...], pa.Schema] = {}
    for path in files:
        schema = pq.read_schema(path)
        key = tuple((field.name, str(field.type)) for field in schema)
        groups.setdefault(key, []).append(path.name)
        schemas[key] = schema
    if len(groups) == 1:
        return next(iter(schemas.values()))
    lines = []
    for key, names in groups.items():
        columns = ", ".join(f"{name}: {data_type}" for name, data_type in key)
        shown = ", ".join(names[:3])
        if len(names) > 3:
            shown += f", ... ({len(names)} files)"
        lines.append(f"  {columns} -> {shown}")
    raise ValueError(
        "parquet inputs have mixed schemas:\n"
        + "\n".join(lines)
        + "\nPass a glob that selects one logical split, e.g. shuffle_train-*-of-10.parquet"
    )


def warn_if_partial_shard(path: Path) -> None:
    match = _SHARD_NAME.match(path.name)
    if match is None:
        return
    total = int(match.group("total"))
    if total <= 1:
        return
    pattern = f"{match.group('prefix')}*-of-{match.group('total')}{match.group('suffix')}"
    siblings = [item for item in sorted(path.parent.glob(pattern)) if is_parquet_file(item)]
    if len(siblings) <= 1:
        return
    glob_path = path.parent / pattern
    warnings.warn(
        f"{path.name} looks like shard {match.group('index')} of {total} "
        f"({len(siblings)} files match). Only this file will be loaded. "
        f"To load all shards, pass --input '{glob_path}'",
        UserWarning,
        stacklevel=2,
    )


def _iter_source_tables(source: Path | list[Path], batch_size: int):
    files = source if isinstance(source, list) else list_parquet_files(source) if source.is_dir() else [source]
    for file in files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield pa.Table.from_batches([batch])


def rewrite_normalized_parquet(
    source: Path | list[Path],
    dest: Path,
    id_column: str | None,
    embedding_column: str | None,
    batch_size: int,
) -> tuple[Path, int, pa.Schema]:
    files = source if isinstance(source, list) else [source]
    mapping = infer_column_mapping(parquet_schema(files), id_column=id_column, embedding_column=embedding_column)
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    dimension: int | None = None
    schema: pa.Schema | None = None
    try:
        for raw in _iter_source_tables(files, batch_size):
            if raw.num_rows == 0:
                continue
            table, dim = normalize_table(raw, mapping, expected_dim=dimension)
            dimension = dim
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema)
                schema = table.schema
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if dimension is None or schema is None:
        raise ValueError("input contained no rows; cannot infer embedding dimension")
    return dest, dimension, schema


def infer_dimension(source: Path | list[Path], embedding_column: str) -> int:
    files = source if isinstance(source, list) else list_parquet_files(source) if source.is_dir() else [source]
    from iceberg_vector_loader.schema import embedding_dimension

    for file in files:
        parquet_file = pq.ParquetFile(file)
        for batch in parquet_file.iter_batches(batch_size=1024, columns=[embedding_column]):
            col = batch.column(embedding_column)
            if hasattr(col, "combine_chunks"):
                col = col.combine_chunks()
            return embedding_dimension(col)
    raise ValueError("input contained no rows; cannot infer embedding dimension")


def _rewrite_destination(
    files: list[Path],
    spark_path: Path,
    parquet_output: str | Path | None,
    staging_dir: str | Path | None,
) -> Path:
    if parquet_output:
        return resolve_local_path(parquet_output)
    if staging_dir:
        return resolve_local_path(staging_dir) / "prepared.parquet"
    if spark_path.is_file():
        return spark_path.with_name(f"{spark_path.stem}.prepared.parquet")
    return files[0].parent / "_prepared.parquet"


def _prepare_parquet_files(
    files: list[Path],
    spark_path: Path,
    parquet_output: str | Path | None,
    staging_dir: str | Path | None,
    id_column: str | None,
    embedding_column: str | None,
    batch_size: int,
) -> PreparedInput:
    schema = parquet_schema(files)
    mapping = infer_column_mapping(schema, id_column=id_column, embedding_column=embedding_column)
    if parquet_output:
        dest = _rewrite_destination(files, spark_path, parquet_output, staging_dir)
        dest, dim, written = rewrite_normalized_parquet(
            files,
            dest,
            id_column=id_column,
            embedding_column=embedding_column,
            batch_size=batch_size,
        )
        return PreparedInput(path=dest, dimension=dim, schema=written, mapping=identity_mapping(written))

    dim = infer_dimension(files, mapping.embedding_column)
    return PreparedInput(path=spark_path, dimension=dim, schema=remapped_schema(schema, mapping), mapping=mapping)


def prepare_input(
    input_path: str | Path,
    parquet_output: str | Path | None = None,
    staging_dir: str | Path | None = None,
    id_column: str | None = None,
    embedding_column: str | None = None,
    id_offset: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> PreparedInput:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    text = str(input_path)
    if looks_like_glob(text):
        files = expand_parquet_glob(text)
        return _prepare_parquet_files(
            files,
            absolute_glob(text),
            parquet_output=parquet_output,
            staging_dir=staging_dir,
            id_column=id_column,
            embedding_column=embedding_column,
            batch_size=batch_size,
        )

    source = resolve_local_path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)

    if is_fvecs(source):
        dest = resolve_local_path(parquet_output) if parquet_output else source.with_suffix(".parquet")
        convert_fvecs_to_parquet(source, dest, batch_size=batch_size, id_offset=id_offset)
        schema = parquet_schema(dest)
        dim = infer_dimension(dest, EMBEDDING_FIELD)
        return PreparedInput(path=dest, dimension=dim, schema=schema, mapping=identity_mapping(schema))

    if source.is_dir():
        files = list_parquet_files(source)
        return _prepare_parquet_files(
            files,
            source,
            parquet_output=parquet_output,
            staging_dir=staging_dir,
            id_column=id_column,
            embedding_column=embedding_column,
            batch_size=batch_size,
        )

    if is_parquet_file(source):
        warn_if_partial_shard(source)
        return _prepare_parquet_files(
            [source],
            source,
            parquet_output=parquet_output,
            staging_dir=staging_dir,
            id_column=id_column,
            embedding_column=embedding_column,
            batch_size=batch_size,
        )

    raise ValueError(f"unsupported input {source}; expected .fvecs, .parquet, a parquet directory, or a parquet glob")
