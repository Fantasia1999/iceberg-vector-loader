from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc

ID_FIELD = "id"
EMBEDDING_FIELD = "embedding"
ID_NAME_CANDIDATES = ("id", "index", "idx")
EMBEDDING_NAME_CANDIDATES = ("embedding", "vector", "features", "emb")


def embedding_field() -> str:
    return EMBEDDING_FIELD


def vector_arrow_schema(extra: list[pa.Field] | None = None) -> pa.Schema:
    fields = [
        pa.field(ID_FIELD, pa.int64(), nullable=False),
        pa.field(EMBEDDING_FIELD, pa.list_(pa.float32()), nullable=False),
    ]
    if extra:
        fields.extend(extra)
    return pa.schema(fields)


def is_integer_type(data_type: pa.DataType) -> bool:
    return pa.types.is_integer(data_type)


def is_float_list_type(data_type: pa.DataType) -> bool:
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        return pa.types.is_floating(data_type.value_type)
    return False


def _pick_named(names: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


@dataclass(frozen=True)
class ColumnMapping:
    id_column: str
    embedding_column: str
    extra_columns: tuple[str, ...]


def infer_column_mapping(
    schema: pa.Schema,
    id_column: str | None = None,
    embedding_column: str | None = None,
) -> ColumnMapping:
    names = list(schema.names)
    if not names:
        raise ValueError("parquet schema is empty")

    if embedding_column is not None:
        if embedding_column not in names:
            raise ValueError(f"embedding column {embedding_column!r} not in parquet: {names}")
        if not is_float_list_type(schema.field(embedding_column).type):
            raise ValueError(
                f"column {embedding_column!r} is {schema.field(embedding_column).type}, expected list/array of float"
            )
        chosen_embedding = embedding_column
    else:
        float_lists = [name for name in names if is_float_list_type(schema.field(name).type)]
        named = _pick_named(float_lists, EMBEDDING_NAME_CANDIDATES)
        if named is not None:
            chosen_embedding = named
        elif len(float_lists) == 1:
            chosen_embedding = float_lists[0]
        elif not float_lists:
            raise ValueError(f"no list<float> embedding column found in parquet schema: {schema}")
        else:
            raise ValueError(
                f"multiple list<float> columns {float_lists}; pass --embedding-column to choose one"
            )

    if id_column is not None:
        if id_column not in names:
            raise ValueError(f"id column {id_column!r} not in parquet: {names}")
        if not is_integer_type(schema.field(id_column).type):
            raise ValueError(f"column {id_column!r} is {schema.field(id_column).type}, expected integer")
        chosen_id = id_column
    else:
        integers = [name for name in names if name != chosen_embedding and is_integer_type(schema.field(name).type)]
        named_id = _pick_named(integers, ID_NAME_CANDIDATES)
        if named_id is not None:
            chosen_id = named_id
        elif len(integers) == 1:
            chosen_id = integers[0]
        elif not integers:
            raise ValueError(
                "parquet has no integer id column; add one or pass --id-column. "
                f"columns: {names}"
            )
        else:
            raise ValueError(f"multiple integer columns {integers}; pass --id-column to choose one")

    extras = tuple(name for name in names if name not in {chosen_id, chosen_embedding})
    return ColumnMapping(
        id_column=chosen_id,
        embedding_column=chosen_embedding,
        extra_columns=extras,
    )


def remapped_schema(schema: pa.Schema, mapping: ColumnMapping) -> pa.Schema:
    """Same types as the source, with id/embedding renamed to the table names."""
    fields = [
        schema.field(mapping.id_column).with_name(ID_FIELD),
        schema.field(mapping.embedding_column).with_name(EMBEDDING_FIELD),
    ]
    for name in mapping.extra_columns:
        fields.append(schema.field(name))
    return pa.schema(fields)


def identity_mapping(schema: pa.Schema) -> ColumnMapping:
    extras = tuple(name for name in schema.names if name not in {ID_FIELD, EMBEDDING_FIELD})
    return ColumnMapping(id_column=ID_FIELD, embedding_column=EMBEDDING_FIELD, extra_columns=extras)


def _cast_embedding(column: pa.Array) -> pa.Array:
    target = pa.list_(pa.float32())
    if pa.types.is_fixed_size_list(column.type) or pa.types.is_large_list(column.type):
        return pc.cast(column, target)
    if pa.types.is_list(column.type):
        if pa.types.is_float32(column.type.value_type):
            return column
        return pc.cast(column, target)
    raise ValueError(f"cannot cast {column.type} to list<float32>")


def list_widths(column: pa.Array) -> pa.Array:
    if pa.types.is_fixed_size_list(column.type):
        return pa.array([column.type.list_size] * len(column), type=pa.int32())
    if pa.types.is_large_list(column.type) or pa.types.is_list(column.type):
        offsets = column.offsets
        return pc.subtract(offsets[1:], offsets[:-1])
    raise ValueError(f"not a list array: {column.type}")


def embedding_dimension(column: pa.Array) -> int:
    if len(column) == 0:
        raise ValueError("cannot infer embedding dimension from empty data")
    widths = list_widths(column)
    unique = pc.unique(widths)
    if len(unique) != 1:
        raise ValueError(f"embedding dimension is not constant: {unique.to_pylist()}")
    dim = int(unique[0].as_py())
    if dim <= 0:
        raise ValueError(f"invalid embedding dimension {dim}")
    return dim


def normalize_table(
    table: pa.Table,
    mapping: ColumnMapping,
    expected_dim: int | None = None,
) -> tuple[pa.Table, int]:
    embedding = _cast_embedding(table[mapping.embedding_column].combine_chunks())
    dim = embedding_dimension(embedding)
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"embedding dimension {dim} does not match expected {expected_dim}")

    if mapping.id_column not in table.schema.names:
        raise ValueError(f"parquet has no id column {mapping.id_column!r}")
    ids = table[mapping.id_column].combine_chunks().cast(pa.int64())

    arrays: list[pa.Array] = [ids, embedding]
    fields: list[pa.Field] = [
        pa.field(ID_FIELD, pa.int64(), nullable=False),
        pa.field(EMBEDDING_FIELD, pa.list_(pa.float32()), nullable=False),
    ]
    for name in mapping.extra_columns:
        extra = table[name].combine_chunks()
        arrays.append(extra)
        fields.append(table.schema.field(name).with_name(name))

    return pa.table(arrays, schema=pa.schema(fields)), dim
