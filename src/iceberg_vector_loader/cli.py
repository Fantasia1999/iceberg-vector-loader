from __future__ import annotations

import click

from iceberg_vector_loader.bootstrap import bootstrap_tools
from iceberg_vector_loader.fvecs import convert_texmex_to_parquet
from iceberg_vector_loader.lance_io import (
    INDEX_TYPES,
    METRICS,
    LanceUnavailableError,
    convert_to_lance,
    format_arrow_table,
    format_query_vector,
    parse_columns,
    parse_query_vector,
    query_lance,
)
from iceberg_vector_loader.loader import DEFAULT_DRIVER_MEMORY, DEFAULT_WAREHOUSE, format_elapsed, load_vectors
from iceberg_vector_loader.paths import PathStyle
from iceberg_vector_loader.prepare import DEFAULT_BATCH_SIZE
from iceberg_vector_loader.spark_sql import COMPRESSION_CODECS, DEFAULT_COMPRESSION_CODEC


@click.group()
def main() -> None:
    """Load local fvecs / parquet embeddings into Iceberg v3, or convert fvecs / ivecs / parquet to Lance."""


@main.command("bootstrap")
def bootstrap_cmd() -> None:
    """Resolve JDK 21 and Spark 3.5.9 (same as scripts/prepare.sh). Uses JAVA_HOME / SPARK_HOME or .tools/; downloads only if missing."""
    tools = bootstrap_tools()
    click.echo(f"tools ready under {tools}")


@main.command("load")
@click.option("--namespace", required=True, help="Iceberg namespace.")
@click.option("--table", "table_name", required=True, help="Iceberg table name.")
@click.option(
    "--input",
    "input_path",
    required=True,
    help="fvecs file, parquet file, parquet directory, or a glob of parquet files (quote the pattern).",
)
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
    except (FileNotFoundError, ValueError) as exc:
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
                f"elapsed         {format_elapsed(result.elapsed_seconds)} "
                f"(prepare {format_elapsed(result.prepare_seconds)}, "
                f"spark {format_elapsed(result.spark_seconds)})",
            ]
        )
    )


@main.command("convert")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="TexMex .fvecs or .ivecs file.")
@click.option("--output", "output_path", required=True, type=click.Path(), help="Destination parquet file.")
@click.option("--id-offset", type=int, default=0, show_default=True)
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True)
def convert_cmd(input_path: str, output_path: str, id_offset: int, batch_size: int) -> None:
    dest = convert_texmex_to_parquet(input_path, output_path, batch_size=batch_size, id_offset=id_offset)
    click.echo(str(dest))


@main.command("to-lance")
@click.option(
    "--input",
    "input_path",
    required=True,
    help="fvecs, ivecs, parquet file, parquet directory, or a glob of parquet files (quote the pattern).",
)
@click.option("--output", "output_path", required=True, type=click.Path(), help="Destination Lance dataset directory (usually *.lance).")
@click.option("--overwrite", is_flag=True, help="Replace the dataset if it already exists.")
@click.option("--id-column", default=None, help="Parquet integer id column. Inferred when omitted.")
@click.option("--embedding-column", default=None, help="Parquet list<float> embedding column. Inferred when omitted.")
@click.option("--id-offset", type=int, default=0, show_default=True, help="Starting id when converting fvecs / ivecs.")
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True, help="Rows per convert batch.")
@click.option("--index", "create_index", is_flag=True, help="Build a vector index after writing (IVF_PQ by default).")
@click.option(
    "--index-type",
    type=click.Choice(INDEX_TYPES, case_sensitive=False),
    default="IVF_PQ",
    show_default=True,
    help="Vector index type. Only used with --index.",
)
@click.option(
    "--metric",
    type=click.Choice(METRICS, case_sensitive=False),
    default="L2",
    show_default=True,
    help="Distance metric for the vector index.",
)
@click.option("--num-partitions", type=int, default=None, help="IVF partitions. Default is sqrt(rows).")
@click.option("--num-sub-vectors", type=int, default=None, help="PQ sub-vectors. Default is dim/8 when dim is divisible by 8.")
def to_lance_cmd(
    input_path: str,
    output_path: str,
    overwrite: bool,
    id_column: str | None,
    embedding_column: str | None,
    id_offset: int,
    batch_size: int,
    create_index: bool,
    index_type: str,
    metric: str,
    num_partitions: int | None,
    num_sub_vectors: int | None,
) -> None:
    """Convert parquet / fvecs / ivecs into a Lance dataset. TexMex inputs go through parquet first."""
    try:
        result = convert_to_lance(
            input_path,
            output_path,
            id_column=id_column,
            embedding_column=embedding_column,
            id_offset=id_offset,
            batch_size=batch_size,
            overwrite=overwrite,
            create_index=create_index,
            index_type=index_type,
            metric=metric,
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
        )
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except (FileNotFoundError, ValueError, LanceUnavailableError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        "\n".join(
            [
                f"dataset         {result.path}",
                f"dimension       {result.dimension}",
                f"rows            {result.rows}",
                f"columns         {', '.join(result.schema.names)}",
                f"index           {result.index_type or '(none)'}",
                f"metric          {result.metric or '-'}",
                f"elapsed         {format_elapsed(result.elapsed_seconds)}",
            ]
        )
    )


@main.command("query-lance")
@click.option("--dataset", "dataset_path", required=True, type=click.Path(), help="Lance dataset to search (e.g. sift_base.lance).")
@click.option(
    "--queries",
    "queries_path",
    default=None,
    type=click.Path(),
    help="Query set: parquet / fvecs / Lance (e.g. sift_query.parquet). Required with --query-id.",
)
@click.option("--query-id", type=int, default=None, help="Row id in --queries. Required with --queries unless --sample.")
@click.option("--query-vector", default=None, help="Query vector as comma-separated floats or a JSON list.")
@click.option("--sample", is_flag=True, help="Pick a random row from --queries instead of --query-id.")
@click.option("--seed", type=int, default=None, help="RNG seed for --sample.")
@click.option("--k", type=int, default=10, show_default=True, help="Number of nearest neighbors.")
@click.option("--columns", default=None, help="Comma-separated columns to return. Default: all except embedding.")
@click.option("--include-embedding", is_flag=True, help="Include the embedding column in the result.")
@click.option("--filter", "filter_expr", default=None, help="SQL predicate applied as a prefilter, e.g. id < 1000.")
@click.option("--sql", default=None, help="Run a SQL SELECT instead of vector search. FROM dataset.")
@click.option(
    "--metric",
    type=click.Choice(METRICS, case_sensitive=False),
    default="L2",
    show_default=True,
    help="Distance metric. Should match the index metric if one exists.",
)
@click.option("--nprobes", type=int, default=None, help="IVF partitions to probe (indexed search).")
@click.option("--refine-factor", type=int, default=None, help="Re-rank this many extra ANN candidates.")
@click.option("--no-index", is_flag=True, help="Ignore an existing vector index and brute-force scan.")
@click.option("--verbose", is_flag=True, help="Print the full query vector instead of a truncated preview.")
def query_lance_cmd(
    dataset_path: str,
    queries_path: str | None,
    query_id: int | None,
    query_vector: str | None,
    sample: bool,
    seed: int | None,
    k: int,
    columns: str | None,
    include_embedding: bool,
    filter_expr: str | None,
    sql: str | None,
    metric: str,
    nprobes: int | None,
    refine_factor: int | None,
    no_index: bool,
    verbose: bool,
) -> None:
    """Run KNN / SQL queries against a Lance dataset written by to-lance."""
    vector = None
    if query_vector is not None:
        try:
            vector = parse_query_vector(query_vector)
        except ValueError as exc:
            raise click.ClickException(f"invalid --query-vector: {exc}") from exc
    try:
        result = query_lance(
            dataset_path,
            queries_path=queries_path,
            query_vector=vector,
            query_id=query_id,
            sample=sample,
            seed=seed,
            k=k,
            columns=parse_columns(columns),
            include_embedding=include_embedding,
            filter_expr=filter_expr,
            sql=sql,
            nprobes=nprobes,
            refine_factor=refine_factor,
            metric=metric,
            use_index=not no_index,
        )
    except (FileNotFoundError, ValueError, LanceUnavailableError) as exc:
        raise click.ClickException(str(exc)) from exc
    header = [
        f"dataset         {dataset_path}",
        f"rows            {result.table.num_rows}",
        f"elapsed         {format_elapsed(result.elapsed_seconds)}",
    ]
    if result.sql:
        header.append(f"sql             {result.sql}")
    else:
        header.append(f"k               {result.k}")
        if result.queries_path is not None:
            header.append(f"queries         {result.queries_path}")
        if result.query_id is not None:
            header.append(f"query-id        {result.query_id}")
        if result.query_vector is not None:
            header.append(f"query-vector    {format_query_vector(result.query_vector, verbose=verbose)}")
    click.echo("\n".join(header))
    click.echo()
    click.echo(format_arrow_table(result.table, max_list_items=None if verbose else 8))
