from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from iceberg_vector_loader.fvecs import (
    convert_fvecs_to_parquet,
    convert_texmex_to_parquet,
    iter_fvecs,
    iter_ivecs,
    read_fvecs_header,
    read_ivecs_header,
)


def write_fvecs(path: Path, vectors: np.ndarray) -> None:
    dim = vectors.shape[1]
    with path.open("wb") as handle:
        for row in vectors:
            handle.write(np.int32(dim).tobytes())
            handle.write(np.asarray(row, dtype=np.float32).tobytes())


def test_iter_fvecs_roundtrip(tmp_path: Path) -> None:
    vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "sample.fvecs"
    write_fvecs(path, vectors)

    dim, count = read_fvecs_header(path)
    assert (dim, count) == (4, 3)

    batches = list(iter_fvecs(path, batch_size=2, id_offset=10))
    assert len(batches) == 2
    ids0, vecs0 = batches[0]
    ids1, vecs1 = batches[1]
    assert ids0.tolist() == [10, 11]
    assert ids1.tolist() == [12]
    np.testing.assert_array_equal(np.vstack([vecs0, vecs1]), vectors)


def test_convert_fvecs_to_parquet(tmp_path: Path) -> None:
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src = tmp_path / "base.fvecs"
    dest = tmp_path / "base.parquet"
    write_fvecs(src, vectors)
    convert_fvecs_to_parquet(src, dest, batch_size=1)

    table = pq.read_table(dest)
    assert table.column_names == ["id", "embedding"]
    assert table["id"].to_pylist() == [0, 1]
    assert table["embedding"].to_pylist() == [[1.0, 2.0], [3.0, 4.0]]


def write_ivecs(path: Path, values: np.ndarray) -> None:
    dim = values.shape[1]
    with path.open("wb") as handle:
        for row in values:
            handle.write(np.int32(dim).tobytes())
            handle.write(np.asarray(row, dtype=np.int32).tobytes())


def test_iter_ivecs_roundtrip(tmp_path: Path) -> None:
    values = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    path = tmp_path / "gt.ivecs"
    write_ivecs(path, values)

    dim, count = read_ivecs_header(path)
    assert (dim, count) == (3, 2)
    ids, neighbors = next(iter_ivecs(path, batch_size=10, id_offset=5))
    assert ids.tolist() == [5, 6]
    np.testing.assert_array_equal(neighbors, values)


def test_convert_ivecs_to_parquet(tmp_path: Path) -> None:
    values = np.array([[1, 2], [3, 4]], dtype=np.int32)
    src = tmp_path / "gt.ivecs"
    dest = tmp_path / "gt.parquet"
    write_ivecs(src, values)
    convert_texmex_to_parquet(src, dest)

    table = pq.read_table(dest)
    assert table.column_names == ["id", "neighbors"]
    assert table["id"].to_pylist() == [0, 1]
    assert table["neighbors"].to_pylist() == [[1, 2], [3, 4]]
