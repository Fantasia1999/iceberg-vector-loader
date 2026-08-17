from __future__ import annotations

import re

import pyarrow as pa

from iceberg_vector_loader.schema import ID_FIELD, EMBEDDING_FIELD

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    if not _IDENT.match(name):
        escaped = name.replace("`", "``")
        return f"`{escaped}`"
    return name


def spark_type(data_type: pa.DataType) -> str:
    if pa.types.is_int8(data_type) or pa.types.is_int16(data_type) or pa.types.is_int32(data_type):
        return "INT"
    if pa.types.is_int64(data_type) or pa.types.is_uint32(data_type) or pa.types.is_uint64(data_type):
        return "BIGINT"
    if pa.types.is_uint8(data_type) or pa.types.is_uint16(data_type):
        return "INT"
    if pa.types.is_float32(data_type):
        return "FLOAT"
    if pa.types.is_float64(data_type):
        return "DOUBLE"
    if pa.types.is_boolean(data_type):
        return "BOOLEAN"
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return "STRING"
    if pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        return "BINARY"
    if pa.types.is_timestamp(data_type):
        return "TIMESTAMP"
    if pa.types.is_date(data_type):
        return "DATE"
    if pa.types.is_fixed_size_list(data_type) or pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return f"ARRAY<{spark_type(data_type.value_type)}>"
    raise ValueError(f"no Spark SQL mapping for Arrow type {data_type}")


def column_ddl(schema: pa.Schema) -> str:
    parts: list[str] = []
    for field in schema:
        not_null = ""
        if field.name in {ID_FIELD, EMBEDDING_FIELD} or not field.nullable:
            not_null = " NOT NULL"
        parts.append(f"{quote_ident(field.name)} {spark_type(field.type)}{not_null}")
    return ",\n  ".join(parts)
