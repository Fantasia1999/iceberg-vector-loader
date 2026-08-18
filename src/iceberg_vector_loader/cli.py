from __future__ import annotations

import click

from iceberg_vector_loader.bootstrap import bootstrap_tools
from iceberg_vector_loader.fvecs import convert_fvecs_to_parquet
from iceberg_vector_loader.loader import DEFAULT_DRIVER_MEMORY, DEFAULT_WAREHOUSE, load_vectors
from iceberg_vector_loader.paths import PathStyle
from iceberg_vector_loader.prepare import DEFAULT_BATCH_SIZE
from iceberg_vector_loader.spark_sql import COMPRESSION_CODECS, DEFAULT_COMPRESSION_CODEC


@click.group()
def main() -> None:
    """Load local fvecs / parquet embeddings into a Spark Iceberg v3 table."""


@main.command("bootstrap")
def bootstrap_cmd() -> None:
    """Resolve JDK 21 and Spark 3.5.9 (same as scripts/prepare.sh). Uses JAVA_HOME / SPARK_HOME or .tools/; downloads only if missing."""
    tools = bootstrap_tools()
    click.echo(f"tools ready under {tools}")


@main.command("load")
@click.option("--namespace", required=True, help="Iceberg namespace.")
@click.option("--table", "table_name", required=True, help="Iceberg table name.")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="fvecs file, parquet file, or parquet directory.")
@click.option("--warehouse", default=DEFAULT_WAREHOUSE, show_default=True, help="Local Hadoop warehouse directory.")
@click.option(
    "--path-style",
    type=click.Choice([PathStyle.PATH.value, PathStyle.URI.value]),
    default=PathStyle.PATH.value,
    show_default=True,
    help="How to write locations into Iceberg metadata. 'path' is a bare filesystem path; 'uri' uses file://.",
)
@click.option("--overwrite", is_flag=True, help="Drop and recreate the table if it already exists. Without this flag, load refuses if the table exists.")
@click.option("--id-column", default=None, help="Parquet integer id column. Inferred when omitted.")
@click.option("--embedding-column", default=None, help="Parquet list<float> embedding column. Inferred when omitted.")
@click.option("--id-offset", type=int, default=0, show_default=True, help="Starting id when converting fvecs (parquet must already have an id column).")
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True, help="Rows per fvecs/normalize batch.")
@click.option("--output", "parquet_output", default=None, type=click.Path(), help="Where to write converted/normalized parquet.")
@click.option("--java-home", default=None, type=click.Path(), help="JDK 21 home. Defaults to JAVA_HOME or .tools/jdk21.")
@click.option("--spark-home", default=None, type=click.Path(), help="Spark 3.5.9 home. Defaults to SPARK_HOME or .tools/spark-3.5.9-bin-hadoop3.")
@click.option("--iceberg-jar", default=None, type=click.Path(), help="iceberg-spark-runtime-3.5_2.12-1.11.0.jar")
@click.option(
    "--driver-memory",
    default=DEFAULT_DRIVER_MEMORY,
    show_default=True,
    help="spark-sql driver heap (local[*]). Default 4g fits SIFT1M; SIFT10M about 16g. See README.",
)
@click.option(
    "--compression-codec",
    type=click.Choice(COMPRESSION_CODECS, case_sensitive=False),
    default=DEFAULT_COMPRESSION_CODEC,
    show_default=True,
    help="Iceberg Parquet compression: write.parquet.compression-codec.",
)
def load_cmd(
    namespace: str,
    table_name: str,
    input_path: str,
    warehouse: str,
    path_style: str,
    overwrite: bool,
    id_column: str | None,
    embedding_column: str | None,
    id_offset: int,
    batch_size: int,
    parquet_output: str | None,
    java_home: str | None,
    spark_home: str | None,
    iceberg_jar: str | None,
    driver_memory: str,
    compression_codec: str,
) -> None:
    try:
        result = load_vectors(
            namespace=namespace,
            table=table_name,
            input_path=input_path,
            warehouse=warehouse,
            path_style=path_style,
            overwrite=overwrite,
            id_column=id_column,
            embedding_column=embedding_column,
            id_offset=id_offset,
            batch_size=batch_size,
            parquet_output=parquet_output,
            java_home=java_home,
            spark_home=spark_home,
            iceberg_jar=iceberg_jar,
            driver_memory=driver_memory,
            compression_codec=compression_codec,
        )
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        "\n".join(
            [
                f"table           {result.namespace}.{result.table}",
                f"format-version  {result.format_version}",
                f"location        {result.location}",
                f"metadata        {result.metadata_location}",
                f"dimension       {result.dimension}",
                f"rows            {result.rows_written}",
                f"data files      {result.files_written}",
                f"warehouse       {result.warehouse}",
                f"source parquet  {result.parquet_source}",
                f"compression     {result.compression_codec}",
            ]
        )
    )


@main.command("convert")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="TexMex .fvecs file.")
@click.option("--output", "output_path", required=True, type=click.Path(), help="Destination parquet file.")
@click.option("--id-offset", type=int, default=0, show_default=True)
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True)
def convert_cmd(input_path: str, output_path: str, id_offset: int, batch_size: int) -> None:
    dest = convert_fvecs_to_parquet(input_path, output_path, batch_size=batch_size, id_offset=id_offset)
    click.echo(str(dest))
