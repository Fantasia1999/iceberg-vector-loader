from __future__ import annotations

import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from iceberg_vector_loader.fvecs import convert_texmex_to_parquet
from iceberg_vector_loader.paths import resolve_local_path
from iceberg_vector_loader.prepare import (
    DEFAULT_BATCH_SIZE,
    _iter_source_tables,
    expand_parquet_glob,
    infer_dimension,
    is_fvecs,
    is_ivecs,
    is_parquet_file,
    list_parquet_files,
    looks_like_glob,
    parquet_schema,
    warn_if_partial_shard,
)
from iceberg_vector_loader.schema import (
    EMBEDDING_FIELD,
    ID_FIELD,
    NEIGHBORS_FIELD,
    ColumnMapping,
    NeighborsMapping,
    has_float_list_column,
    infer_column_mapping,
    infer_neighbors_mapping,
    normalize_neighbors_table,
    normalize_table,
)

DEFAULT_MAX_ROWS_PER_GROUP = 8192
DEFAULT_MAX_ROWS_PER_FILE = 1024 * 1024
INDEX_TYPES = ("IVF_PQ", "IVF_HNSW_PQ", "IVF_HNSW_SQ")
METRICS = ("L2", "cosine", "dot")
PQ_INDEX_TYPES = {"IVF_PQ", "IVF_HNSW_PQ"}
_METRIC_LOOKUP = {name.lower(): name for name in METRICS}
_INDEX_LOOKUP = {name.lower(): name for name in INDEX_TYPES}


def normalize_metric(metric: str) -> str:
    key = metric.lower()
    if key not in _METRIC_LOOKUP:
        raise ValueError(f"unsupported metric {metric!r}; choose one of {list(METRICS)}")
    return _METRIC_LOOKUP[key]


def normalize_index_type(index_type: str) -> str:
    key = index_type.lower()
    if key not in _INDEX_LOOKUP:
        raise ValueError(f"unsupported index type {index_type!r}; choose one of {list(INDEX_TYPES)}")
    return _INDEX_LOOKUP[key]


class LanceUnavailableError(ImportError):
    """Raised when pylance is not installed."""


def _require_lance():
    try:
        import lance
    except ImportError as exc:
        raise LanceUnavailableError(
            "pylance is required for Lance commands. Reinstall with: uv pip install -e ."
        ) from exc
    return lance


def to_fixed_size_list(column: pa.Array, dim: int, value_type: pa.DataType) -> pa.Array:
    """Cast a list/large_list/fixed_size_list column to fixed_size_list[value_type][dim]."""
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    target = pa.list_(value_type, dim)
    if len(column) == 0:
        return pa.array([], type=target)
    if pa.types.is_fixed_size_list(column.type):
        if column.type.list_size != dim:
            raise ValueError(f"list size {column.type.list_size} does not match dim {dim}")
        if column.type.value_type == value_type:
            return column
        return pc.cast(column, target)

    flattened = pc.list_flatten(column)
    if flattened.type != value_type:
        flattened = pc.cast(flattened, value_type)
    expected = len(column) * dim
    if len(flattened) != expected:
        raise ValueError(f"list values {len(flattened)} do not match {len(column)} rows * dim {dim}")
    return pa.FixedSizeListArray.from_arrays(flattened, dim)


def embedding_to_fixed_size_list(column: pa.Array, dim: int) -> pa.Array:
    return to_fixed_size_list(column, dim, pa.float32())


def with_fixed_size_list_column(table: pa.Table, name: str, dim: int, value_type: pa.DataType) -> pa.Table:
    values = to_fixed_size_list(table[name], dim, value_type)
    field = pa.field(name, pa.list_(value_type, dim), nullable=False)
    return table.set_column(table.schema.get_field_index(name), field, values)


def with_fixed_size_embedding(table: pa.Table, dim: int) -> pa.Table:
    return with_fixed_size_list_column(table, EMBEDDING_FIELD, dim, pa.float32())


def with_fixed_size_neighbors(table: pa.Table, dim: int) -> pa.Table:
    return with_fixed_size_list_column(table, NEIGHBORS_FIELD, dim, pa.int32())


def lance_vector_schema(source_schema: pa.Schema | None, mapping: ColumnMapping | None, dim: int) -> pa.Schema:
    fields = [
        pa.field(ID_FIELD, pa.int64(), nullable=False),
        pa.field(EMBEDDING_FIELD, pa.list_(pa.float32(), dim), nullable=False),
    ]
    if mapping is not None and source_schema is not None:
        for name in mapping.extra_columns:
            fields.append(source_schema.field(name))
    return pa.schema(fields)


def lance_neighbors_schema(source_schema: pa.Schema | None, mapping: NeighborsMapping | None, dim: int) -> pa.Schema:
    fields = [
        pa.field(ID_FIELD, pa.int64(), nullable=False),
        pa.field(NEIGHBORS_FIELD, pa.list_(pa.int32(), dim), nullable=False),
    ]
    if mapping is not None and source_schema is not None:
        for name in mapping.extra_columns:
            fields.append(source_schema.field(name))
    return pa.schema(fields)


def default_num_sub_vectors(dim: int) -> int:
    for width in (8, 4, 2):
        if dim % width == 0:
            return dim // width
    return dim


def default_num_partitions(rows: int) -> int:
    if rows <= 1:
        return 1
    return max(1, min(rows, int(rows**0.5)))


def parse_query_vector(text: str) -> list[float]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("query vector is empty")
    if stripped[0] in "[{":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON query vector: {exc}") from exc
        if isinstance(parsed, dict):
            raise ValueError("query vector JSON must be a list of floats")
        return [float(value) for value in parsed]
    return [float(part) for part in stripped.split(",") if part.strip()]


def parse_columns(text: str | None) -> list[str] | None:
    if text is None:
        return None
    columns = [part.strip() for part in text.split(",") if part.strip()]
    return columns or None


@dataclass(frozen=True)
class LanceConvertResult:
    path: Path
    dimension: int
    rows: int
    schema: pa.Schema
    index_type: str | None
    metric: str | None
    elapsed_seconds: float


@dataclass(frozen=True)
class LanceQueryResult:
    table: pa.Table
    query_id: int | None
    k: int | None
    sql: str | None
    elapsed_seconds: float


def _assert_output_path(dest: Path, overwrite: bool) -> str:
    if dest.exists() and dest.is_file():
        raise ValueError(f"{dest} is a file; Lance dataset path must be a directory (usually *.lance)")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} already exists; pass --overwrite to replace it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    return "overwrite" if dest.exists() else "create"


def _write_batches(
    batches: Iterator[pa.RecordBatch],
    dest: Path,
    schema: pa.Schema,
    mode: str,
    max_rows_per_group: int,
    max_rows_per_file: int,
):
    lance = _require_lance()
    first = next(batches, None)
    if first is None:
        empty = pa.table({name: pa.array([], type=schema.field(name).type) for name in schema.names}, schema=schema)
        return lance.write_dataset(
            empty,
            dest,
            schema=schema,
            mode=mode,
            max_rows_per_group=max_rows_per_group,
            max_rows_per_file=max_rows_per_file,
        )

    def chained() -> Iterator[pa.RecordBatch]:
        yield first
        yield from batches

    return lance.write_dataset(
        chained(),
        dest,
        schema=first.schema,
        mode=mode,
        max_rows_per_group=max_rows_per_group,
        max_rows_per_file=max_rows_per_file,
    )


def _parquet_batches(
    files: list[Path],
    mapping: ColumnMapping,
    dim: int,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    for raw in _iter_source_tables(files, batch_size):
        if raw.num_rows == 0:
            continue
        table, got = normalize_table(raw, mapping, expected_dim=dim)
        yield from with_fixed_size_embedding(table, got).to_batches()


def _neighbors_batches(
    files: list[Path],
    mapping: NeighborsMapping,
    dim: int,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    for raw in _iter_source_tables(files, batch_size):
        if raw.num_rows == 0:
            continue
        table, got = normalize_neighbors_table(raw, mapping, expected_dim=dim)
        yield from with_fixed_size_neighbors(table, got).to_batches()


def _maybe_create_index(
    dataset,
    *,
    create_index: bool,
    index_type: str,
    metric: str,
    num_partitions: int | None,
    num_sub_vectors: int | None,
    dim: int,
    rows: int,
):
    if not create_index:
        return dataset, None, None
    if rows < 2:
        raise ValueError("not enough rows to build a vector index")
    kind = normalize_index_type(index_type)
    distance = normalize_metric(metric)
    kwargs: dict = {
        "metric": distance,
        "num_partitions": num_partitions if num_partitions is not None else default_num_partitions(rows),
    }
    if kind in PQ_INDEX_TYPES:
        kwargs["num_sub_vectors"] = num_sub_vectors if num_sub_vectors is not None else default_num_sub_vectors(dim)
    dataset = dataset.create_index(EMBEDDING_FIELD, index_type=kind, **kwargs)
    return dataset, kind, distance


def convert_to_lance(
    input_path: str | Path,
    output_path: str | Path,
    *,
    id_column: str | None = None,
    embedding_column: str | None = None,
    id_offset: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    create_index: bool = False,
    index_type: str = "IVF_PQ",
    metric: str = "L2",
    num_partitions: int | None = None,
    num_sub_vectors: int | None = None,
    max_rows_per_group: int = DEFAULT_MAX_ROWS_PER_GROUP,
    max_rows_per_file: int = DEFAULT_MAX_ROWS_PER_FILE,
) -> LanceConvertResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    started = perf_counter()
    dest = resolve_local_path(output_path)
    mode = _assert_output_path(dest, overwrite)

    text = str(input_path)
    if looks_like_glob(text):
        files = expand_parquet_glob(text)
    else:
        source = resolve_local_path(input_path)
        if not source.exists():
            raise FileNotFoundError(source)
        if is_fvecs(source) or is_ivecs(source):
            parquet_path = dest.with_suffix(".parquet")
            convert_texmex_to_parquet(source, parquet_path, batch_size=batch_size, id_offset=id_offset)
            files = [parquet_path]
        elif source.is_dir():
            files = list_parquet_files(source)
        elif is_parquet_file(source):
            warn_if_partial_shard(source)
            files = [source]
        else:
            raise ValueError(
                f"unsupported input {source}; expected .fvecs, .ivecs, .parquet, a parquet directory, or a parquet glob"
            )

    schema = parquet_schema(files)
    if has_float_list_column(schema, embedding_column):
        mapping = infer_column_mapping(schema, id_column=id_column, embedding_column=embedding_column)
        dim = infer_dimension(files, mapping.embedding_column)
        dataset = _write_batches(
            _parquet_batches(files, mapping, dim, batch_size),
            dest,
            lance_vector_schema(schema, mapping, dim),
            mode,
            max_rows_per_group,
            max_rows_per_file,
        )
        indexable = True
    else:
        if create_index:
            raise ValueError(
                "vector index requires float embeddings; this input has integer lists "
                "(ivecs / neighbors). Omit --index and query with --sql."
            )
        neighbors = infer_neighbors_mapping(schema, id_column=id_column)
        dim = infer_dimension(files, neighbors.neighbors_column)
        dataset = _write_batches(
            _neighbors_batches(files, neighbors, dim, batch_size),
            dest,
            lance_neighbors_schema(schema, neighbors, dim),
            mode,
            max_rows_per_group,
            max_rows_per_file,
        )
        indexable = False

    rows = dataset.count_rows()
    if not indexable:
        return LanceConvertResult(
            path=dest,
            dimension=dim,
            rows=rows,
            schema=dataset.schema,
            index_type=None,
            metric=None,
            elapsed_seconds=perf_counter() - started,
        )
    dataset, built_index, built_metric = _maybe_create_index(
        dataset,
        create_index=create_index,
        index_type=index_type,
        metric=metric,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        dim=dim,
        rows=rows,
    )
    return LanceConvertResult(
        path=dest,
        dimension=dim,
        rows=rows,
        schema=dataset.schema,
        index_type=built_index,
        metric=built_metric,
        elapsed_seconds=perf_counter() - started,
    )


def _open_dataset(dataset_path: str | Path):
    lance = _require_lance()
    path = resolve_local_path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return lance.dataset(path)


def _row_embedding(dataset, query_id: int | None, sample: bool, seed: int | None) -> tuple[int, np.ndarray]:
    if query_id is not None and sample:
        raise ValueError("pass only one of query_id or sample")
    columns = [ID_FIELD, EMBEDDING_FIELD]
    if query_id is not None:
        table = dataset.to_table(columns=columns, filter=f"{ID_FIELD} = {int(query_id)}")
        if table.num_rows == 0:
            raise ValueError(f"no row with {ID_FIELD}={query_id}")
        row = table.slice(0, 1)
    elif sample:
        total = dataset.count_rows()
        if total == 0:
            raise ValueError("dataset is empty")
        rng = random.Random(seed)
        offset = rng.randrange(total)
        row = dataset.to_table(columns=columns, offset=offset, limit=1)
    else:
        row = dataset.to_table(columns=columns, limit=1)
        if row.num_rows == 0:
            raise ValueError("dataset is empty")
    chosen_id = int(row[ID_FIELD][0].as_py())
    vector = np.asarray(row[EMBEDDING_FIELD][0].as_py(), dtype=np.float32)
    return chosen_id, vector


def _select_columns(dataset, columns: list[str] | None, include_embedding: bool, nearest: bool) -> list[str]:
    names = list(dataset.schema.names)
    if columns is None:
        selected = [name for name in names if include_embedding or name != EMBEDDING_FIELD]
    else:
        missing = [name for name in columns if name not in names and name != "_distance"]
        if missing:
            raise ValueError(f"unknown columns {missing}; available: {names}")
        selected = list(columns)
    if nearest and "_distance" not in selected:
        selected.append("_distance")
    if not selected:
        selected = [ID_FIELD]
        if nearest:
            selected.append("_distance")
    return selected


def query_lance(
    dataset_path: str | Path,
    *,
    query_vector: list[float] | np.ndarray | None = None,
    query_id: int | None = None,
    sample: bool = False,
    seed: int | None = None,
    k: int = 10,
    columns: list[str] | None = None,
    include_embedding: bool = False,
    filter_expr: str | None = None,
    sql: str | None = None,
    nprobes: int | None = None,
    refine_factor: int | None = None,
    metric: str = "L2",
    use_index: bool = True,
) -> LanceQueryResult:
    if k <= 0:
        raise ValueError("k must be positive")
    started = perf_counter()
    dataset = _open_dataset(dataset_path)

    if sql:
        if query_vector is not None or query_id is not None:
            raise ValueError("--sql cannot be combined with --query-vector / --query-id")
        batches = dataset.sql(sql).build().to_batch_records()
        table = pa.Table.from_batches(batches) if batches else dataset.head(0)
        return LanceQueryResult(
            table=table,
            query_id=None,
            k=None,
            sql=sql,
            elapsed_seconds=perf_counter() - started,
        )

    if EMBEDDING_FIELD not in dataset.schema.names:
        raise ValueError(
            f"dataset has no {EMBEDDING_FIELD!r} column "
            f"(ivecs tables use {NEIGHBORS_FIELD!r}). Pass --sql to inspect them."
        )

    chosen_id = None
    if query_vector is not None:
        if query_id is not None or sample:
            raise ValueError("pass only one of query_vector, query_id, or sample")
        vector = np.asarray(query_vector, dtype=np.float32)
    else:
        chosen_id, vector = _row_embedding(dataset, query_id=query_id, sample=sample, seed=seed)

    dim = vector.shape[-1]
    field = dataset.schema.field(EMBEDDING_FIELD)
    if pa.types.is_fixed_size_list(field.type) and field.type.list_size != dim:
        raise ValueError(f"query vector dim {dim} does not match dataset embedding dim {field.type.list_size}")

    nearest: dict = {
        "column": EMBEDDING_FIELD,
        "q": vector,
        "k": k,
        "metric": normalize_metric(metric),
        "use_index": use_index,
    }
    if nprobes is not None:
        nearest["nprobes"] = nprobes
    if refine_factor is not None:
        nearest["refine_factor"] = refine_factor

    selected = _select_columns(dataset, columns, include_embedding=include_embedding, nearest=True)
    kwargs: dict = {"columns": selected, "nearest": nearest}
    if filter_expr:
        kwargs["filter"] = filter_expr
        kwargs["prefilter"] = True
    table = dataset.to_table(**kwargs)
    return LanceQueryResult(
        table=table,
        query_id=chosen_id,
        k=k,
        sql=None,
        elapsed_seconds=perf_counter() - started,
    )


def _format_cell(value, max_list_items: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if len(value) > max_list_items:
            head = ", ".join(_format_cell(item, max_list_items) for item in value[:max_list_items])
            return f"[{head}, ...]"
        inner = ", ".join(_format_cell(item, max_list_items) for item in value)
        return f"[{inner}]"
    return str(value)


def format_arrow_table(table: pa.Table, *, max_list_items: int = 8) -> str:
    if table.num_rows == 0:
        return "(0 rows)"
    names = list(table.column_names)
    formatted_cols = []
    widths = []
    for name in names:
        cells = [_format_cell(value, max_list_items) for value in table[name].to_pylist()]
        formatted_cols.append(cells)
        widths.append(max(len(name), max((len(cell) for cell in cells), default=0)))
    header = "  ".join(name.ljust(width) for name, width in zip(names, widths))
    rule = "  ".join("-" * width for width in widths)
    lines = [header, rule]
    for row in range(table.num_rows):
        lines.append("  ".join(formatted_cols[col][row].ljust(widths[col]) for col in range(len(names))))
    return "\n".join(lines)
