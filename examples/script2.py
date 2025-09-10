#!/usr/bin/env python3
import os
import pyarrow.parquet as pq
import tempfile
import shutil

FILES_TO_PATCH = [
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000021.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000022.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000023.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000024.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000025.parquet",
]

# Column renaming map
rename_map = {
    "wrist_image": "observation.images.image2",
    "actions": "action",
}

# Hugging Face metadata
HF_METADATA = {
    b"huggingface": b'{"info": {"features": {'
    b'"observation.images.image": {"_type": "Image"}, '
    b'"observation.images.image2": {"_type": "Image"}, '
    b'"state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 8, "_type": "Sequence"}, '
    b'"action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "Sequence"}, '
    b'"timestamp": {"dtype": "float32", "_type": "Value"}, '
    b'"frame_index": {"dtype": "int64", "_type": "Value"}, '
    b'"episode_index": {"dtype": "int64", "_type": "Value"}, '
    b'"index": {"dtype": "int64", "_type": "Value"}, '
    b'"task_index": {"dtype": "int64", "_type": "Value"}}}}'
}

def patch_parquet(parquet_path, hf_metadata, rename_map):
    try:
        table = pq.read_table(parquet_path)

        # Apply column renames if needed
        if rename_map:
            schema = table.schema
            new_names = [rename_map.get(name, name) for name in schema.names]
            table = table.rename_columns(new_names)

        # Merge schema metadata
        new_meta = dict(table.schema.metadata or {})
        new_meta.update(hf_metadata)

        # Replace metadata
        table = table.replace_schema_metadata(new_meta)

        # Write via temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(tmp_fd)
        pq.write_table(table, tmp_path)

        shutil.move(tmp_path, parquet_path)
        print(f"✅ Patched: {parquet_path}")
        return True
    except Exception as e:
        print(f"❌ Failed on {parquet_path}: {e}")
        return False


if __name__ == "__main__":
    for fpath in FILES_TO_PATCH:
        if os.path.exists(fpath):
            patch_parquet(fpath, HF_METADATA, rename_map)
        else:
            print(f"⚠️ File not found: {fpath}")
