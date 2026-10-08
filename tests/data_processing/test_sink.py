# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

import pytest

from lerobot.data_processing.sinks.parquet import materialize_patch, patch_table

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def source_table():
    return pa.table(
        {
            "episode_index": [0, 0, 1],
            "frame_index": [0, 1, 0],
            "timestamp": [0.0, 0.1, 0.0],
            "action": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "label": ["original", "untouched", "other_episode"],
        }
    ).replace_schema_metadata({b"source": b"publisher"})


def test_partial_patch_preserves_all_other_rows_signals_and_metadata(tmp_path):
    source = source_table()
    patch = pa.table({"episode_index": [0], "frame_index": [0], "label": ["new"]})
    result = patch_table(source, patch, columns=("label",))
    assert result["label"].to_pylist() == ["new", "untouched", "other_episode"]
    assert result.schema.metadata == source.schema.metadata
    for column in ("action", "timestamp", "frame_index", "episode_index"):
        assert result[column].equals(source[column])
    original = tmp_path / "original.parquet"
    output = tmp_path / "new" / "output.parquet"
    pq.write_table(source, original)
    before = original.read_bytes()
    materialize_patch(original, patch, output, columns=("label",))
    assert pq.read_table(output).equals(result)
    assert original.read_bytes() == before
    with pytest.raises(ValueError, match="separate version"):
        materialize_patch(original, patch, original, columns=("label",))


def test_typed_empty_nested_annotation_keeps_schema():
    dtype = pa.list_(pa.struct([("text", pa.string()), ("valid", pa.bool_())]))
    patch = pa.table({"episode_index": [0], "frame_index": [0], "annotations": pa.array([[]], type=dtype)})
    result = patch_table(source_table(), patch, columns=("annotations",))
    assert result.schema.field("annotations").type == dtype
    assert result["annotations"].to_pylist() == [[], None, None]


def test_patch_rejects_null_in_nonnullable_owned_column():
    source = source_table()
    label_field = pa.field("label", pa.string(), nullable=False)
    source = source.set_column(source.column_names.index("label"), label_field, source["label"])
    patch = pa.table({"episode_index": [0], "frame_index": [0], "label": pa.array([None], type=pa.string())})
    with pytest.raises(ValueError, match="nonnullable"):
        patch_table(source, patch, columns=("label",))


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "timeline", "extra", "type"])
def test_invalid_patch_fails_without_mutation(kind):
    source = source_table()
    if kind == "duplicate":
        patch = pa.table({"episode_index": [0, 0], "frame_index": [0, 0], "label": ["a", "b"]})
        columns = ("label",)
    elif kind == "unknown":
        patch = pa.table({"episode_index": [99], "frame_index": [0], "label": ["a"]})
        columns = ("label",)
    elif kind == "timeline":
        patch = pa.table({"episode_index": [0], "frame_index": [0], "timestamp": [3.0]})
        columns = ("timestamp",)
    elif kind == "extra":
        patch = pa.table({"episode_index": [0], "frame_index": [0], "label": ["a"], "extra": [0]})
        columns = ("label",)
    else:
        patch = pa.table({"episode_index": [0], "frame_index": [0], "label": [42]})
        columns = ("label",)
    with pytest.raises(ValueError):
        patch_table(source, patch, columns=columns)
    assert source["label"].to_pylist() == ["original", "untouched", "other_episode"]
