from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from iceberg_vector_loader.schema import ID_FIELD, embedding_field, vector_arrow_schema

FVECS_DTYPE_HEADER = np.dtype("<i4")


def _record_size(dim: int) -> int:
    return 4 + dim * 4


def read_fvecs_header(path: str | Path) -> tuple[int, int]:
    """Return (dimension, vector_count) for a TexMex fvecs file."""
    file_path = Path(path)
    size = file_path.stat().st_size
    if size == 0:
        return 0, 0
    if size < 4:
        raise ValueError(f"{file_path} is too small to be an fvecs file")
    with file_path.open("rb") as handle:
        dim = int(np.frombuffer(handle.read(4), dtype=FVECS_DTYPE_HEADER)[0])
    if dim <= 0:
        raise ValueError(f"{file_path} has invalid dimension {dim}")
    rec = _record_size(dim)
    if size % rec != 0:
        raise ValueError(f"{file_path} size {size} is not a multiple of record size {rec} (dim={dim})")
    return dim, size // rec


def iter_fvecs(
    path: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (ids, vectors) batches from a TexMex fvecs file.

    Each record is little-endian int32 dimension followed by `dim` float32s.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    file_path = Path(path)
    dim, count = read_fvecs_header(file_path)
    if count == 0:
        return
    rec = _record_size(dim)
    next_id = id_offset
    with file_path.open("rb") as handle:
        remaining = count
        while remaining > 0:
            take = min(batch_size, remaining)
            chunk = handle.read(rec * take)
            if len(chunk) != rec * take:
                raise ValueError(f"{file_path} ended early while reading fvecs records")
            raw = np.frombuffer(chunk, dtype=np.uint8).reshape(take, rec)
            dims = raw[:, :4].view(FVECS_DTYPE_HEADER).reshape(take)
            if not np.all(dims == dim):
                raise ValueError(f"{file_path} contains mixed dimensions (expected {dim})")
            vectors = np.ascontiguousarray(raw[:, 4:].view(np.float32).reshape(take, dim))
            ids = np.arange(next_id, next_id + take, dtype=np.int64)
            next_id += take
            remaining -= take
            yield ids, vectors


def embeddings_to_list_array(vectors: np.ndarray) -> pa.Array:
    if vectors.ndim != 2:
        raise ValueError(f"expected 2-D embedding matrix, got shape {vectors.shape}")
    rows, dim = vectors.shape
    values = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1), type=pa.float32())
    offsets = np.arange(0, (rows + 1) * dim, dim, dtype=np.int32)
    return pa.ListArray.from_arrays(offsets, values)


def batch_to_arrow(ids: np.ndarray, vectors: np.ndarray) -> pa.Table:
    schema = vector_arrow_schema()
    return pa.table(
        {
            ID_FIELD: pa.array(ids, type=pa.int64()),
            embedding_field(): embeddings_to_list_array(vectors),
        },
        schema=schema,
    )


def convert_fvecs_to_parquet(
    source: str | Path,
    dest: str | Path,
    batch_size: int = 250_000,
    id_offset: int = 0,
) -> Path:
    """Convert an fvecs file to a parquet file with id + embedding columns."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for ids, vectors in iter_fvecs(source, batch_size=batch_size, id_offset=id_offset):
            table = batch_to_arrow(ids, vectors)
            if writer is None:
                writer = pq.ParquetWriter(dest_path, table.schema)
            writer.write_table(table)
        if writer is None:
            empty = pa.table(
                {ID_FIELD: pa.array([], type=pa.int64()), embedding_field(): pa.array([], type=pa.list_(pa.float32()))},
                schema=vector_arrow_schema(),
            )
            pq.write_table(empty, dest_path)
    finally:
        if writer is not None:
            writer.close()
    return dest_path
