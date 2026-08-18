from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iceberg_vector_loader.inspect import load_metadata, snapshot_total_records
from iceberg_vector_loader.loader import load_vectors, read_table_parquet
from iceberg_vector_loader.paths import PathStyle, has_file_scheme
from iceberg_vector_loader.spark_env import runtime_available
from iceberg_vector_loader.schema import infer_column_mapping
from iceberg_vector_loader.spark_sql import build_load_sql, build_select_exprs, normalize_compression_codec
from iceberg_vector_loader.spark_types import column_ddl, spark_type

spark_only = pytest.mark.skipif(not runtime_available(), reason="Spark 3.5.9 / JDK 21 / Iceberg 1.11.0 jar not available")


def write_fvecs(path: Path, vectors: np.ndarray) -> None:
    dim = vectors.shape[1]
    with path.open("wb") as handle:
        for row in vectors:
            handle.write(np.int32(dim).tobytes())
            handle.write(np.asarray(row, dtype=np.float32).tobytes())


def test_spark_type_array_float() -> None:
    assert spark_type(pa.list_(pa.float32())) == "ARRAY<FLOAT>"
    assert spark_type(pa.int64()) == "BIGINT"


def test_build_load_sql_is_v3() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32()), nullable=False),
        ]
    )
    sql = build_load_sql(
        namespace="bench",
        table="sift1m",
        parquet_path=Path("/tmp/sift.parquet"),
        schema_ddl=column_ddl(schema),
        overwrite=True,
        dimension=128,
        select_columns=["id", "embedding"],
    )
    assert "format-version'='3'" in sql
    assert "ARRAY<FLOAT>" in sql
    assert "USING iceberg" in sql
    assert "DROP TABLE IF EXISTS" in sql
    assert "parquet.`/tmp/sift.parquet`" in sql
    assert "write.parquet.compression-codec'='zstd'" in sql

    snappy = build_load_sql(
        namespace="bench",
        table="sift1m",
        parquet_path=Path("/tmp/sift.parquet"),
        schema_ddl=column_ddl(schema),
        overwrite=False,
        dimension=128,
        select_columns=["id", "embedding"],
        compression_codec="SNAPPY",
    )
    assert "write.parquet.compression-codec'='snappy'" in snappy


def test_build_select_exprs_aliases_emb() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("emb", pa.list_(pa.float32())),
        ]
    )
    exprs = build_select_exprs(infer_column_mapping(schema))
    assert exprs == ["id", "emb AS embedding"]
    sql = build_load_sql(
        namespace="bench",
        table="bio",
        parquet_path=Path("/data/shuffle_train-*-of-10.parquet"),
        schema_ddl=column_ddl(schema),
        overwrite=False,
        dimension=1024,
        select_columns=["id", "embedding"],
        select_exprs=exprs,
    )
    assert "SELECT id, emb AS embedding" in sql
    assert "FROM parquet.`/data/shuffle_train-*-of-10.parquet`" in sql


def test_compression_codec_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported compression codec"):
        normalize_compression_codec("lz4hc")


@spark_only
def test_load_fvecs_v3_path_style(tmp_path: Path) -> None:
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    src = tmp_path / "sift.fvecs"
    write_fvecs(src, vectors)
    warehouse = tmp_path / "warehouse"

    result = load_vectors(
        namespace="bench",
        table="sift1m",
        input_path=src,
        warehouse=warehouse,
        path_style=PathStyle.PATH,
        batch_size=2,
    )

    assert result.format_version == 3
    assert result.rows_written == 3
    assert result.dimension == 3
    assert not has_file_scheme(result.location)
    assert not has_file_scheme(result.metadata_location)
    assert result.location.rstrip("/").endswith("bench/sift1m")
    assert Path(result.metadata_location).parent.joinpath("version-hint.text").exists()
    assert not (warehouse / "catalog.db").exists()

    data = read_table_parquet(warehouse, "bench", "sift1m").sort_by("id")
    assert data["id"].to_pylist() == [0, 1, 2]
    assert data["embedding"].to_pylist() == vectors.tolist()
    metadata = load_metadata(warehouse, "bench", "sift1m")
    assert int(metadata["format-version"]) == 3


@spark_only
def test_load_parquet_uri_style_and_extra_column(tmp_path: Path) -> None:
    parquet_path = tmp_path / "vectors.parquet"
    table = pa.table(
        {
            "id": pa.array([10, 11], type=pa.int64()),
            "embedding": pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32())),
            "label": pa.array(["q", "w"], type=pa.string()),
        }
    )
    pq.write_table(table, parquet_path)
    warehouse = tmp_path / "wh"

    result = load_vectors(
        namespace="ds",
        table="extra",
        input_path=parquet_path,
        warehouse=warehouse,
        path_style=PathStyle.URI,
    )
    assert result.format_version == 3
    assert has_file_scheme(result.location)
    assert result.location.startswith("file://")

    loaded = read_table_parquet(warehouse, "ds", "extra").sort_by("id")
    assert "label" in loaded.column_names
    assert loaded["label"].to_pylist() == ["q", "w"]


@spark_only
def test_load_parquet_directory(tmp_path: Path) -> None:
    directory = tmp_path / "parts"
    directory.mkdir()
    for index, rows in enumerate(([0], [1, 2])):
        pq.write_table(
            pa.table(
                {
                    "id": pa.array(rows, type=pa.int64()),
                    "embedding": pa.array([[float(i), 0.0] for i in rows], type=pa.list_(pa.float32())),
                }
            ),
            directory / f"part-{index}.parquet",
        )
    result = load_vectors(namespace="dir", table="parts", input_path=directory, warehouse=tmp_path / "wh")
    assert result.rows_written == 3
    ids = read_table_parquet(tmp_path / "wh", "dir", "parts")["id"].to_pylist()
    assert sorted(ids) == [0, 1, 2]


@spark_only
def test_load_parquet_glob_skips_other_schemas(tmp_path: Path) -> None:
    directory = tmp_path / "bioasq"
    directory.mkdir()
    for index, rows in enumerate(([0], [1, 2])):
        pq.write_table(
            pa.table(
                {
                    "id": pa.array(rows, type=pa.int64()),
                    "emb": pa.array([[float(i), 1.0] for i in rows], type=pa.list_(pa.float32())),
                }
            ),
            directory / f"shuffle_train-{index:02d}-of-02.parquet",
        )
    pq.write_table(
        pa.table({"id": pa.array([99], type=pa.int64()), "labels": pa.array(["skip"])}),
        directory / "scalar_labels.parquet",
    )
    result = load_vectors(
        namespace="bio",
        table="train",
        input_path=str(directory / "shuffle_train-*-of-02.parquet"),
        warehouse=tmp_path / "wh",
    )
    assert result.rows_written == 3
    assert result.dimension == 2
    assert result.parquet_source.endswith("shuffle_train-*-of-02.parquet")
    loaded = read_table_parquet(tmp_path / "wh", "bio", "train").sort_by("id")
    assert loaded["id"].to_pylist() == [0, 1, 2]
    assert loaded.column_names[1] == "embedding"


@spark_only
def test_refuse_existing_table_unless_overwrite(tmp_path: Path) -> None:
    warehouse = tmp_path / "wh"
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([0], type=pa.int64()),
                "embedding": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
            }
        ),
        first,
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array([1], type=pa.int64()),
                "embedding": pa.array([[3.0, 4.0]], type=pa.list_(pa.float32())),
            }
        ),
        second,
    )

    load_vectors(namespace="n", table="t", input_path=first, warehouse=warehouse)
    with pytest.raises(FileExistsError, match="already exists"):
        load_vectors(namespace="n", table="t", input_path=second, warehouse=warehouse)
    assert read_table_parquet(warehouse, "n", "t")["id"].to_pylist() == [0]

    load_vectors(namespace="n", table="t", input_path=second, warehouse=warehouse, overwrite=True)
    ids = read_table_parquet(warehouse, "n", "t")["id"].to_pylist()
    assert ids == [1]


@spark_only
def test_cli_load(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from iceberg_vector_loader.cli import main

    vectors = np.array([[9.0, 8.0]], dtype=np.float32)
    src = tmp_path / "one.fvecs"
    write_fvecs(src, vectors)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "load",
            "--namespace",
            "cli",
            "--table",
            "demo",
            "--input",
            str(src),
            "--warehouse",
            str(tmp_path / "wh"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "format-version  3" in result.output
