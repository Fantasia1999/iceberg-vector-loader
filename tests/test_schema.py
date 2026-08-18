import pyarrow as pa
import pytest

from iceberg_vector_loader.schema import infer_column_mapping, normalize_table, remapped_schema


def test_infer_standard_id_embedding() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )
    mapping = infer_column_mapping(schema)
    assert mapping.id_column == "id"
    assert mapping.embedding_column == "embedding"
    assert mapping.extra_columns == ()


def test_infer_missing_id_errors() -> None:
    schema = pa.schema(
        [
            pa.field("embedding", pa.list_(pa.float32())),
            pa.field("label", pa.string()),
        ]
    )
    with pytest.raises(ValueError, match="no integer id column"):
        infer_column_mapping(schema)


def test_infer_ambiguous_embeddings() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("a", pa.list_(pa.float32())),
            pa.field("b", pa.list_(pa.float32())),
        ]
    )
    with pytest.raises(ValueError, match="multiple list"):
        infer_column_mapping(schema)
    mapping = infer_column_mapping(schema, embedding_column="b")
    assert mapping.embedding_column == "b"
    assert mapping.id_column == "id"


def test_normalize_casts_and_keeps_extras() -> None:
    table = pa.table(
        {
            "idx": [10, 11],
            "vector": [[1.0, 2.0], [3.0, 4.0]],
            "label": ["a", "b"],
        }
    )
    mapping = infer_column_mapping(table.schema)
    out, dim = normalize_table(table, mapping)
    assert dim == 2
    assert out.column_names == ["id", "embedding", "label"]
    assert out["id"].to_pylist() == [10, 11]
    assert out["embedding"].to_pylist() == [[1.0, 2.0], [3.0, 4.0]]
    assert out["label"].to_pylist() == ["a", "b"]


def test_remapped_schema_renames_only() -> None:
    schema = pa.schema(
        [
            pa.field("idx", pa.int32()),
            pa.field("emb", pa.list_(pa.float32())),
            pa.field("label", pa.string()),
        ]
    )
    mapping = infer_column_mapping(schema)
    remapped = remapped_schema(schema, mapping)
    assert remapped.names == ["id", "embedding", "label"]
    assert remapped.field("id").type == pa.int32()
    assert remapped.field("embedding").type == pa.list_(pa.float32())
