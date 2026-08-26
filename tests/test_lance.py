from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

lance = pytest.importorskip("lance")

from iceberg_vector_loader.lance_io import convert_to_lance, format_query_vector, parse_query_vector, query_lance
from iceberg_vector_loader.schema import EMBEDDING_FIELD, NEIGHBORS_FIELD


def _write_vectors(path: Path, ids: list[int], column: str = "embedding", extra: dict | None = None) -> None:
    data = {
        "id": pa.array(ids, type=pa.int64()),
        column: pa.array([[float(i), 1.0] for i in ids], type=pa.list_(pa.float32())),
    }
    if extra:
        data.update(extra)
    pq.write_table(pa.table(data), path)


def test_convert_parquet_to_lance_and_knn(tmp_path: Path) -> None:
    src = tmp_path / "vecs.parquet"
    dest = tmp_path / "vecs.lance"
    _write_vectors(src, [10, 11, 12], extra={"label": pa.array(["a", "b", "a"])})

    result = convert_to_lance(src, dest)
    assert result.rows == 3
    assert result.dimension == 2
    assert result.index_type is None
    dataset = lance.dataset(dest)
    assert pa.types.is_fixed_size_list(dataset.schema.field(EMBEDDING_FIELD).type)
    assert dataset.schema.field(EMBEDDING_FIELD).type.list_size == 2
    assert dataset.schema.names == ["id", "embedding", "label"]

    knn = query_lance(dest, query_id=10, k=2)
    assert knn.query_id == 10
    assert knn.table.num_rows == 2
    assert knn.table["id"].to_pylist()[0] == 10
    assert knn.table["_distance"].to_pylist()[0] == pytest.approx(0.0)
    assert "embedding" not in knn.table.column_names
    assert "label" in knn.table.column_names


def test_convert_emb_alias_and_glob(tmp_path: Path) -> None:
    for index in range(2):
        _write_vectors(tmp_path / f"shuffle_train-{index:02d}-of-02.parquet", [index], column="emb")
    dest = tmp_path / "train.lance"
    result = convert_to_lance(str(tmp_path / "shuffle_train-*-of-02.parquet"), dest)
    assert result.rows == 2
    table = lance.dataset(dest).to_table(columns=["id"])
    assert sorted(table["id"].to_pylist()) == [0, 1]


def test_query_vector_filter_and_sql(tmp_path: Path) -> None:
    src = tmp_path / "vecs.parquet"
    dest = tmp_path / "vecs.lance"
    _write_vectors(src, [0, 1, 2], extra={"label": pa.array(["keep", "drop", "keep"])})
    convert_to_lance(src, dest)

    knn = query_lance(dest, query_vector=[0.0, 1.0], k=3, filter_expr="label = 'keep'")
    assert knn.table["id"].to_pylist() == [0, 2]
    assert knn.table["label"].to_pylist() == ["keep", "keep"]

    sql = query_lance(dest, sql="SELECT id, label FROM dataset WHERE label = 'drop'")
    assert sql.table["id"].to_pylist() == [1]
    assert sql.k is None


def test_overwrite_required(tmp_path: Path) -> None:
    src = tmp_path / "vecs.parquet"
    dest = tmp_path / "vecs.lance"
    _write_vectors(src, [0])
    convert_to_lance(src, dest)
    with pytest.raises(FileExistsError, match="already exists"):
        convert_to_lance(src, dest)
    again = convert_to_lance(src, dest, overwrite=True)
    assert again.rows == 1


def test_parse_query_vector() -> None:
    assert parse_query_vector("1, 2.5, 3") == [1.0, 2.5, 3.0]
    assert parse_query_vector("[1, 2, 3]") == [1.0, 2.0, 3.0]


def test_format_query_vector_preview_and_verbose() -> None:
    values = tuple(float(i) for i in range(12))
    preview = format_query_vector(values)
    assert "dim 12" in preview
    assert "..." in preview
    assert "11" not in preview.split("...")[0]
    full = format_query_vector(values, verbose=True)
    assert "..." not in full
    assert full.endswith("(dim 12)")
    assert "11" in full


def test_self_search_first_row(tmp_path: Path) -> None:
    src = tmp_path / "vecs.parquet"
    dest = tmp_path / "vecs.lance"
    _write_vectors(src, [7, 8, 9])
    convert_to_lance(src, dest)
    knn = query_lance(dest, k=1)
    assert knn.query_id == 7
    assert knn.table["id"].to_pylist() == [7]


def test_include_embedding_column(tmp_path: Path) -> None:
    src = tmp_path / "vecs.parquet"
    dest = tmp_path / "vecs.lance"
    _write_vectors(src, [0, 1])
    convert_to_lance(src, dest)
    knn = query_lance(dest, query_id=0, k=1, include_embedding=True)
    assert EMBEDDING_FIELD in knn.table.column_names
    np.testing.assert_allclose(knn.table[EMBEDDING_FIELD][0].as_py(), [0.0, 1.0])
    assert knn.query_vector == (0.0, 1.0)


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    dim = vectors.shape[1]
    with path.open("wb") as handle:
        for row in vectors:
            handle.write(np.int32(dim).tobytes())
            handle.write(np.asarray(row, dtype=np.float32).tobytes())


def _write_ivecs(path: Path, values: np.ndarray) -> None:
    dim = values.shape[1]
    with path.open("wb") as handle:
        for row in values:
            handle.write(np.int32(dim).tobytes())
            handle.write(np.asarray(row, dtype=np.int32).tobytes())


def test_to_lance_fvecs_goes_through_parquet(tmp_path: Path) -> None:
    src = tmp_path / "base.fvecs"
    dest = tmp_path / "base.lance"
    _write_fvecs(src, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    result = convert_to_lance(src, dest)
    parquet = dest.with_suffix(".parquet")
    assert parquet.exists()
    assert result.rows == 2
    assert result.dimension == 2
    dataset = lance.dataset(dest)
    assert dataset.schema.names == ["id", "embedding"]
    knn = query_lance(dest, query_id=0, k=1)
    assert knn.table["id"].to_pylist() == [0]


def test_to_lance_ivecs_goes_through_parquet(tmp_path: Path) -> None:
    src = tmp_path / "gt.ivecs"
    dest = tmp_path / "gt.lance"
    _write_ivecs(src, np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32))
    result = convert_to_lance(src, dest)
    parquet = dest.with_suffix(".parquet")
    assert parquet.exists()
    table = pq.read_table(parquet)
    assert table.column_names == ["id", "neighbors"]
    assert result.rows == 2
    assert result.dimension == 3
    assert result.index_type is None
    dataset = lance.dataset(dest)
    assert dataset.schema.names == ["id", "neighbors"]
    assert pa.types.is_fixed_size_list(dataset.schema.field(NEIGHBORS_FIELD).type)
    assert dataset.schema.field(NEIGHBORS_FIELD).type.list_size == 3
    sql = query_lance(dest, sql="SELECT id, neighbors FROM dataset WHERE id = 1")
    assert sql.table["id"].to_pylist() == [1]
    assert sql.table["neighbors"].to_pylist() == [[40, 50, 60]]


def test_to_lance_ivecs_rejects_index(tmp_path: Path) -> None:
    src = tmp_path / "gt.ivecs"
    dest = tmp_path / "gt.lance"
    _write_ivecs(src, np.array([[1, 2], [3, 4]], dtype=np.int32))
    with pytest.raises(ValueError, match="vector index requires float embeddings"):
        convert_to_lance(src, dest, create_index=True)
