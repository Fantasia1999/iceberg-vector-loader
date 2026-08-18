from __future__ import annotations

import os
import subprocess
from pathlib import Path

from iceberg_vector_loader.paths import PathStyle, format_location
from iceberg_vector_loader.schema import EMBEDDING_FIELD, ID_FIELD, ColumnMapping
from iceberg_vector_loader.spark_env import CATALOG_NAME, SparkRuntime
from iceberg_vector_loader.spark_types import quote_ident

DEFAULT_DRIVER_MEMORY = "4g"
DEFAULT_COMPRESSION_CODEC = "zstd"
COMPRESSION_CODECS = ("zstd", "snappy", "gzip", "lz4", "brotli", "uncompressed")


def normalize_compression_codec(codec: str) -> str:
    name = codec.strip().lower()
    if name not in COMPRESSION_CODECS:
        allowed = ", ".join(COMPRESSION_CODECS)
        raise ValueError(f"unsupported compression codec {codec!r}; choose one of: {allowed}")
    return name


def warehouse_location(warehouse: str | Path, path_style: PathStyle | str) -> str:
    return format_location(warehouse, path_style)


def table_ident(namespace: str, table: str) -> str:
    return f"{CATALOG_NAME}.{quote_ident(namespace)}.{quote_ident(table)}"


def spark_conf(warehouse: str, iceberg_jar: Path) -> list[str]:
    return [
        "--jars",
        str(iceberg_jar),
        "--conf",
        "spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        "--conf",
        f"spark.sql.catalog.{CATALOG_NAME}=org.apache.iceberg.spark.SparkCatalog",
        "--conf",
        f"spark.sql.catalog.{CATALOG_NAME}.type=hadoop",
        "--conf",
        f"spark.sql.catalog.{CATALOG_NAME}.warehouse={warehouse}",
        "--conf",
        f"spark.sql.catalog.{CATALOG_NAME}.io-impl=org.apache.iceberg.hadoop.HadoopFileIO",
        "--conf",
        f"spark.sql.defaultCatalog={CATALOG_NAME}",
    ]


def build_select_exprs(mapping: ColumnMapping) -> list[str]:
    def aliased(source: str, dest: str) -> str:
        if source == dest:
            return quote_ident(source)
        return f"{quote_ident(source)} AS {quote_ident(dest)}"

    return [
        aliased(mapping.id_column, ID_FIELD),
        aliased(mapping.embedding_column, EMBEDDING_FIELD),
        *[quote_ident(name) for name in mapping.extra_columns],
    ]


def build_load_sql(
    namespace: str,
    table: str,
    parquet_path: Path,
    schema_ddl: str,
    overwrite: bool,
    dimension: int,
    select_columns: list[str],
    compression_codec: str = DEFAULT_COMPRESSION_CODEC,
    select_exprs: list[str] | None = None,
) -> str:
    ident = table_ident(namespace, table)
    parquet_uri = str(parquet_path)
    cols = ", ".join(quote_ident(name) for name in select_columns)
    selected = ", ".join(select_exprs) if select_exprs is not None else cols
    codec = normalize_compression_codec(compression_codec)
    props = ",\n  ".join(
        [
            "'format-version'='3'",
            "'write.format.default'='parquet'",
            f"'write.parquet.compression-codec'='{codec}'",
            f"'iceberg.vector.dimension'='{dimension}'",
        ]
    )
    statements = [
        f"CREATE NAMESPACE IF NOT EXISTS {CATALOG_NAME}.{quote_ident(namespace)};",
    ]
    if overwrite:
        statements.append(f"DROP TABLE IF EXISTS {ident} PURGE;")
    statements.append(
        f"CREATE TABLE IF NOT EXISTS {ident} (\n  {schema_ddl}\n) USING iceberg\nTBLPROPERTIES (\n  {props}\n);"
    )
    statements.append(
        f"INSERT INTO {ident} ({cols})\nSELECT {selected}\nFROM parquet.`{parquet_uri}`;"
    )
    return "\n".join(statements) + "\n"


def run_spark_sql(
    runtime: SparkRuntime,
    sql: str,
    warehouse: str,
    work_dir: Path,
    driver_memory: str = DEFAULT_DRIVER_MEMORY,
    extra_conf: list[str] | None = None,
) -> str:
    work_dir.mkdir(parents=True, exist_ok=True)
    sql_path = work_dir / "job.sql"
    sql_path.write_text(sql, encoding="utf-8")
    cmd = [
        str(runtime.spark_sql),
        "--master",
        "local[*]",
        "--driver-memory",
        driver_memory,
        *spark_conf(warehouse, runtime.iceberg_jar),
        *(extra_conf or []),
        "-f",
        str(sql_path),
    ]
    env = os.environ.copy()
    env["JAVA_HOME"] = str(runtime.java_home)
    env["SPARK_HOME"] = str(runtime.spark_home)
    env["PATH"] = f"{runtime.java_home / 'bin'}:{runtime.spark_home / 'bin'}:{env.get('PATH', '')}"
    # Hadoop local FS: keep paths as given when warehouse has no scheme.
    env.setdefault("HADOOP_CONF_DIR", "")
    completed = subprocess.run(
        cmd,
        env=env,
        cwd=str(work_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    log = work_dir / "spark-sql.log"
    log.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"spark-sql failed (exit {completed.returncode}). See {log}\n{tail[-4000:]}")
    return completed.stdout or ""
