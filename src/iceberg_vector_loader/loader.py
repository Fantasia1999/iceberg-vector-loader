from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq

from iceberg_vector_loader.inspect import (
    assert_path_style,
    latest_metadata_path,
    load_metadata,
    snapshot_total_records,
    table_dir,
    table_exists_on_fs,
)
from iceberg_vector_loader.locations import rewrite_metadata_locations
from iceberg_vector_loader.paths import PathStyle, resolve_local_path
from iceberg_vector_loader.prepare import DEFAULT_BATCH_SIZE, prepare_input
from iceberg_vector_loader.spark_env import resolve_runtime
from iceberg_vector_loader.spark_sql import (
    DEFAULT_COMPRESSION_CODEC,
    build_load_sql,
    build_select_exprs,
    normalize_compression_codec,
    run_spark_sql,
    warehouse_location,
)
from iceberg_vector_loader.spark_types import column_ddl

DEFAULT_WAREHOUSE = "warehouse"
DEFAULT_DRIVER_MEMORY = "4g"


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


@dataclass(frozen=True)
class LoadResult:
    namespace: str
    table: str
    warehouse: str
    location: str
    metadata_location: str
    format_version: int
    rows_written: int
    files_written: int
    dimension: int
    overwritten: bool
    parquet_source: str
    compression_codec: str
    elapsed_seconds: float
    prepare_seconds: float
    spark_seconds: float


def load_vectors(
    namespace: str,
    table: str,
    input_path: str | Path,
    warehouse: str | Path = DEFAULT_WAREHOUSE,
    path_style: PathStyle | str = PathStyle.PATH,
    overwrite: bool = False,
    id_column: str | None = None,
    embedding_column: str | None = None,
    id_offset: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    parquet_output: str | Path | None = None,
    java_home: str | Path | None = None,
    spark_home: str | Path | None = None,
    iceberg_jar: str | Path | None = None,
    driver_memory: str = DEFAULT_DRIVER_MEMORY,
    compression_codec: str = DEFAULT_COMPRESSION_CODEC,
) -> LoadResult:
    if not namespace or not table:
        raise ValueError("namespace and table are required")
    if "." in table:
        raise ValueError(f"table name must not contain '.': {table}")
    codec = normalize_compression_codec(compression_codec)
    started = perf_counter()

    warehouse_dir = resolve_local_path(warehouse)
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    style = PathStyle(path_style)
    warehouse_loc = warehouse_location(warehouse_dir, style)

    existed = table_exists_on_fs(warehouse_dir, namespace, table)
    if existed and not overwrite:
        raise FileExistsError(
            f"table {namespace}.{table} already exists at {table_dir(warehouse_dir, namespace, table)}; "
            "pass --overwrite to drop and recreate it"
        )

    staging = warehouse_dir / "_staging" / namespace / table
    prepare_started = perf_counter()
    prepared = prepare_input(
        input_path,
        parquet_output=parquet_output,
        staging_dir=staging,
        id_column=id_column,
        embedding_column=embedding_column,
        id_offset=id_offset,
        batch_size=batch_size,
    )
    prepare_seconds = perf_counter() - prepare_started

    runtime = resolve_runtime(java_home=java_home, spark_home=spark_home, iceberg_jar=iceberg_jar)
    sql = build_load_sql(
        namespace=namespace,
        table=table,
        parquet_path=prepared.path,
        schema_ddl=column_ddl(prepared.schema),
        overwrite=overwrite,
        dimension=prepared.dimension,
        select_columns=list(prepared.schema.names),
        compression_codec=codec,
        select_exprs=build_select_exprs(prepared.mapping),
    )
    work_dir = staging / "spark"
    spark_started = perf_counter()
    run_spark_sql(
        runtime,
        sql=sql,
        warehouse=warehouse_loc,
        work_dir=work_dir,
        driver_memory=driver_memory,
    )
    spark_seconds = perf_counter() - spark_started

    metadata_path = latest_metadata_path(warehouse_dir, namespace, table)
    rewrite_metadata_locations(metadata_path, style)
    metadata = load_metadata(warehouse_dir, namespace, table)
    assert_path_style(metadata, want_uri=(style is PathStyle.URI))
    format_version = int(metadata.get("format-version", 0))
    if format_version != 3:
        raise RuntimeError(f"expected Iceberg format-version 3, got {format_version}")

    data_dir = table_dir(warehouse_dir, namespace, table) / "data"
    files_after = len(list(data_dir.rglob("*.parquet"))) if data_dir.exists() else 0

    return LoadResult(
        namespace=namespace,
        table=table,
        warehouse=str(warehouse_dir),
        location=str(metadata.get("location")),
        metadata_location=str(latest_metadata_path(warehouse_dir, namespace, table)),
        format_version=format_version,
        rows_written=snapshot_total_records(metadata),
        files_written=files_after,
        dimension=prepared.dimension,
        overwritten=existed and overwrite,
        parquet_source=str(prepared.path),
        compression_codec=codec,
        elapsed_seconds=perf_counter() - started,
        prepare_seconds=prepare_seconds,
        spark_seconds=spark_seconds,
    )


def read_table_parquet(warehouse: str | Path, namespace: str, table: str):
    data_dir = table_dir(warehouse, namespace, table) / "data"
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {data_dir}")
    return pq.read_table(files)
