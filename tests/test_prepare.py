from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iceberg_vector_loader.prepare import (
    expand_parquet_glob,
    prepare_input,
    warn_if_partial_shard,
)


def _write_vectors(path: Path, ids: list[int], column: str = "embedding") -> None:
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                column: pa.array([[float(i), 0.0] for i in ids], type=pa.list_(pa.float32())),
            }
        ),
        path,
    )


def test_expand_parquet_glob(tmp_path: Path) -> None:
    for index in range(3):
        _write_vectors(tmp_path / f"shuffle_train-{index:02d}-of-03.parquet", [index], column="emb")
    _write_vectors(tmp_path / "test.parquet", [99], column="emb")
    pq.write_table(
        pa.table({"id": pa.array([0], type=pa.int64()), "labels": pa.array(["x"])}),
        tmp_path / "scalar_labels.parquet",
    )

    files = expand_parquet_glob(str(tmp_path / "shuffle_train-*-of-03.parquet"))
    assert [path.name for path in files] == [
        "shuffle_train-00-of-03.parquet",
        "shuffle_train-01-of-03.parquet",
        "shuffle_train-02-of-03.parquet",
    ]


def test_prepare_glob_aliases_emb_without_rewrite(tmp_path: Path) -> None:
    for index in range(2):
        _write_vectors(tmp_path / f"shuffle_train-{index:02d}-of-02.parquet", [index], column="emb")
    pq.write_table(
        pa.table({"id": pa.array([0], type=pa.int64()), "neighbors_id": pa.array([[1]], type=pa.list_(pa.int64()))}),
        tmp_path / "neighbors.parquet",
    )

    prepared = prepare_input(str(tmp_path / "shuffle_train-*-of-02.parquet"))
    assert prepared.path.name == "shuffle_train-*-of-02.parquet"
    assert prepared.mapping.embedding_column == "emb"
    assert prepared.schema.names == ["id", "embedding"]
    assert prepared.dimension == 2
    assert not (tmp_path / "prepared.parquet").exists()


def test_mixed_directory_lists_schemas(tmp_path: Path) -> None:
    _write_vectors(tmp_path / "shuffle_train-00-of-01.parquet", [0], column="emb")
    pq.write_table(
        pa.table({"id": pa.array([0], type=pa.int64()), "labels": pa.array(["x"])}),
        tmp_path / "scalar_labels.parquet",
    )
    with pytest.raises(ValueError, match="mixed schemas"):
        prepare_input(tmp_path)


def test_glob_with_no_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no parquet files match"):
        expand_parquet_glob(str(tmp_path / "missing-*-of-10.parquet"))


def test_single_shard_warns(tmp_path: Path) -> None:
    first = tmp_path / "shuffle_train-00-of-02.parquet"
    second = tmp_path / "shuffle_train-01-of-02.parquet"
    _write_vectors(first, [0], column="emb")
    _write_vectors(second, [1], column="emb")
    with pytest.warns(UserWarning, match="shard 00 of 2"):
        warn_if_partial_shard(first)
    with pytest.warns(UserWarning, match="Only this file will be loaded"):
        prepare_input(first)
