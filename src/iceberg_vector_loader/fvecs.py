from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from iceberg_vector_loader.schema import (
    ID_FIELD,
    NEIGHBORS_FIELD,
    embedding_field,
    neighbors_arrow_schema,
    vector_arrow_schema,
)

FVECS_DTYPE_HEADER = np.dtype("<i4")
FVECS_PAYLOAD = np.dtype("<f4")
IVECS_PAYLOAD = np.dtype("<i4")
FVECS_SUFFIXES = {".fvecs"}
IVECS_SUFFIXES = {".ivecs"}


def _record_size(dim: int, item_size: int = 4) -> int:
    return 4 + dim * item_size


def read_texmex_header(path: str | Path, *, kind: str = "texmex", item_size: int = 4) -> tuple[int, int]:
    """Return (dimension, vector_count) for a TexMex fvecs/ivecs file."""
    file_path = Path(path)
    size = file_path.stat().st_size
    if size == 0:
        return 0, 0
    if size < 4:
        raise ValueError(f"{file_path} is too small to be an {kind} file")
    with file_path.open("rb") as handle:
        dim = int(np.frombuffer(handle.read(4), dtype=FVECS_DTYPE_HEADER)[0])
    if dim <= 0:
        raise ValueError(f"{file_path} has invalid dimension {dim}")
    rec = _record_size(dim, item_size)
    if size % rec != 0:
        raise ValueError(f"{file_path} size {size} is not a multiple of record size {rec} (dim={dim})")
    return dim, size // rec


def read_fvecs_header(path: str | Path) -> tuple[int, int]:
    """Return (dimension, vector_count) for a TexMex fvecs file."""
    return read_texmex_header(path, kind="fvecs")


def read_ivecs_header(path: str | Path) -> tuple[int, int]:
    """Return (dimension, vector_count) for a TexMex ivecs file."""
    return read_texmex_header(path, kind="ivecs")


def iter_texmex(
    path: str | Path,
    *,
    dtype: np.dtype,
    kind: str,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (ids, values) batches from a TexMex file.

    Each record is little-endian int32 dimension followed by `dim` payload values.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    payload = np.dtype(dtype)
    file_path = Path(path)
    dim, count = read_texmex_header(file_path, kind=kind, item_size=payload.itemsize)
    if count == 0:
        return
    rec = _record_size(dim, payload.itemsize)
    next_id = id_offset
    with file_path.open("rb") as handle:
        remaining = count
        while remaining > 0:
            take = min(batch_size, remaining)
            chunk = handle.read(rec * take)
            if len(chunk) != rec * take:
                raise ValueError(f"{file_path} ended early while reading {kind} records")
            raw = np.frombuffer(chunk, dtype=np.uint8).reshape(take, rec)
            dims = raw[:, :4].view(FVECS_DTYPE_HEADER).reshape(take)
            if not np.all(dims == dim):
                raise ValueError(f"{file_path} contains mixed dimensions (expected {dim})")
            values = np.ascontiguousarray(raw[:, 4:].view(payload).reshape(take, dim))
            ids = np.arange(next_id, next_id + take, dtype=np.int64)
            next_id += take
            remaining -= take
            yield ids, values


def iter_fvecs(
    path: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (ids, vectors) batches from a TexMex fvecs file.

    Each record is little-endian int32 dimension followed by `dim` float32s.
    """
    yield from iter_texmex(path, dtype=FVECS_PAYLOAD, kind="fvecs", batch_size=batch_size, id_offset=id_offset)


def iter_ivecs(
    path: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (ids, neighbors) batches from a TexMex ivecs file.

    Each record is little-endian int32 dimension followed by `dim` int32s.
    Typical SIFT ground-truth files store 100 neighbor ids per query.
    """
    yield from iter_texmex(path, dtype=IVECS_PAYLOAD, kind="ivecs", batch_size=batch_size, id_offset=id_offset)


def matrix_to_list_array(matrix: np.ndarray, *, numpy_dtype, arrow_type: pa.DataType) -> pa.Array:
    if matrix.ndim != 2:
        raise ValueError(f"expected 2-D matrix, got shape {matrix.shape}")
    rows, dim = matrix.shape
    values = pa.array(np.ascontiguousarray(matrix, dtype=numpy_dtype).reshape(-1), type=arrow_type)
    offsets = np.arange(0, (rows + 1) * dim, dim, dtype=np.int32)
    return pa.ListArray.from_arrays(offsets, values)


def embeddings_to_list_array(vectors: np.ndarray) -> pa.Array:
    return matrix_to_list_array(vectors, numpy_dtype=np.float32, arrow_type=pa.float32())


def neighbors_to_list_array(values: np.ndarray) -> pa.Array:
    return matrix_to_list_array(values, numpy_dtype=np.int32, arrow_type=pa.int32())


def batch_to_arrow(ids: np.ndarray, vectors: np.ndarray) -> pa.Table:
    schema = vector_arrow_schema()
    return pa.table(
        {
            ID_FIELD: pa.array(ids, type=pa.int64()),
            embedding_field(): embeddings_to_list_array(vectors),
        },
        schema=schema,
    )


def ivecs_batch_to_arrow(ids: np.ndarray, values: np.ndarray) -> pa.Table:
    schema = neighbors_arrow_schema()
    return pa.table(
        {
            ID_FIELD: pa.array(ids, type=pa.int64()),
            NEIGHBORS_FIELD: neighbors_to_list_array(values),
        },
        schema=schema,
    )


def _write_parquet_from_tables(dest: Path, tables: Iterator[pa.Table], empty: pa.Table) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for table in tables:
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema)
            writer.write_table(table)
        if writer is None:
            pq.write_table(empty, dest)
    finally:
        if writer is not None:
            writer.close()
    return dest


def convert_fvecs_to_parquet(
    source: str | Path,
    dest: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Path:
    """Convert an fvecs file to a parquet file with id + embedding columns."""
    empty = pa.table(
        {ID_FIELD: pa.array([], type=pa.int64()), embedding_field(): pa.array([], type=pa.list_(pa.float32()))},
        schema=vector_arrow_schema(),
    )
    tables = (batch_to_arrow(ids, vectors) for ids, vectors in iter_fvecs(source, batch_size=batch_size, id_offset=id_offset))
    return _write_parquet_from_tables(Path(dest), tables, empty)


def convert_ivecs_to_parquet(
    source: str | Path,
    dest: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Path:
    """Convert an ivecs file to a parquet file with id + neighbors columns."""
    empty = pa.table(
        {ID_FIELD: pa.array([], type=pa.int64()), NEIGHBORS_FIELD: pa.array([], type=pa.list_(pa.int32()))},
        schema=neighbors_arrow_schema(),
    )
    tables = (
        ivecs_batch_to_arrow(ids, values) for ids, values in iter_ivecs(source, batch_size=batch_size, id_offset=id_offset)
    )
    return _write_parquet_from_tables(Path(dest), tables, empty)


def convert_texmex_to_parquet(
    source: str | Path,
    dest: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Path:
    """Convert a TexMex .fvecs or .ivecs file to parquet."""
    suffix = Path(source).suffix.lower()
    if suffix in FVECS_SUFFIXES:
        return convert_fvecs_to_parquet(source, dest, batch_size=batch_size, id_offset=id_offset)
    if suffix in IVECS_SUFFIXES:
        return convert_ivecs_to_parquet(source, dest, batch_size=batch_size, id_offset=id_offset)
    raise ValueError(f"unsupported TexMex input {source}; expected .fvecs or .ivecs")
