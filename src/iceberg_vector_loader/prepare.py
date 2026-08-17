from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from iceberg_vector_loader.fvecs import convert_fvecs_to_parquet
from iceberg_vector_loader.paths import resolve_local_path
from iceberg_vector_loader.schema import (
    EMBEDDING_FIELD,
    ID_FIELD,
    infer_column_mapping,
    is_float_list_type,
    is_integer_type,
    normalize_table,
)

DEFAULT_BATCH_SIZE = 250_000
FVECS_SUFFIXES = {".fvecs"}
PARQUET_SUFFIXES = {".parquet", ".parq"}


def is_fvecs(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in FVECS_SUFFIXES


def is_parquet_input(path: Path) -> bool:
    if path.is_dir():
        return True
    return path.is_file() and path.suffix.lower() in PARQUET_SUFFIXES


def parquet_schema(source: Path) -> pa.Schema:
    if source.is_dir():
        return pq.ParquetDataset(source).schema
    return pq.read_schema(source)


def already_normalized(schema: pa.Schema, id_column: str | None, embedding_column: str | None) -> bool:
    mapping = infer_column_mapping(schema, id_column=id_column, embedding_column=embedding_column)
    if mapping.id_column != ID_FIELD or mapping.embedding_column != EMBEDDING_FIELD:
        return False
    id_type = schema.field(ID_FIELD).type
    emb_type = schema.field(EMBEDDING_FIELD).type
    return is_integer_type(id_type) and is_float_list_type(emb_type)


def _iter_source_tables(source: Path, batch_size: int):
    if source.is_dir():
        dataset = pq.ParquetDataset(source)
        for fragment in dataset.fragments:
            for batch in fragment.to_table().to_batches(max_chunksize=batch_size):
                yield pa.Table.from_batches([batch])
        return
    parquet_file = pq.ParquetFile(source)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield pa.Table.from_batches([batch])


def rewrite_normalized_parquet(
    source: Path,
    dest: Path,
    id_column: str | None,
    embedding_column: str | None,
    batch_size: int,
) -> tuple[Path, int, pa.Schema]:
    mapping = infer_column_mapping(parquet_schema(source), id_column=id_column, embedding_column=embedding_column)
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    dimension: int | None = None
    schema: pa.Schema | None = None
    try:
        for raw in _iter_source_tables(source, batch_size):
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


def infer_dimension(source: Path, embedding_column: str) -> int:
    if source.is_dir():
        dataset = pq.ParquetDataset(source)
        for fragment in dataset.fragments:
            table = fragment.to_table(columns=[embedding_column])
            if table.num_rows:
                from iceberg_vector_loader.schema import embedding_dimension

                col = table[embedding_column]
                if hasattr(col, "combine_chunks"):
                    col = col.combine_chunks()
                return embedding_dimension(col)
        raise ValueError("input contained no rows; cannot infer embedding dimension")
    parquet_file = pq.ParquetFile(source)
    for batch in parquet_file.iter_batches(batch_size=1024, columns=[embedding_column]):
        from iceberg_vector_loader.schema import embedding_dimension

        col = batch.column(embedding_column)
        if hasattr(col, "combine_chunks"):
            col = col.combine_chunks()
        return embedding_dimension(col)
    raise ValueError("input contained no rows; cannot infer embedding dimension")


def prepare_input(
    input_path: str | Path,
    parquet_output: str | Path | None = None,
    staging_dir: str | Path | None = None,
    id_column: str | None = None,
    embedding_column: str | None = None,
    id_offset: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[Path, int, pa.Schema]:
    source = resolve_local_path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if is_fvecs(source):
        dest = resolve_local_path(parquet_output) if parquet_output else source.with_suffix(".parquet")
        convert_fvecs_to_parquet(source, dest, batch_size=batch_size, id_offset=id_offset)
        schema = parquet_schema(dest)
        dim = infer_dimension(dest, EMBEDDING_FIELD)
        return dest, dim, schema

    if not is_parquet_input(source):
        raise ValueError(f"unsupported input {source}; expected .fvecs, .parquet, or a parquet directory")

    schema = parquet_schema(source)
    if already_normalized(schema, id_column, embedding_column):
        dim = infer_dimension(source, EMBEDDING_FIELD)
        return source, dim, schema

    if parquet_output:
        dest = resolve_local_path(parquet_output)
    elif staging_dir:
        dest = resolve_local_path(staging_dir) / "prepared.parquet"
    else:
        dest = source.with_name(f"{source.stem}.prepared.parquet") if source.is_file() else source / "_prepared.parquet"
    return rewrite_normalized_parquet(
        source,
        dest,
        id_column=id_column,
        embedding_column=embedding_column,
        batch_size=batch_size,
    )
