#!/usr/bin/env python3
import os
import pyarrow.parquet as pq
import tempfile
import shutil

# Root directory containing all parquet files
ROOT_DIR = "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data"

# Column renaming map (normalize schema to what training expects)
rename_map = {
    "state": "observation.state",
}

# Hugging Face metadata (aligned with expected feature names)
HF_METADATA = {
    b"huggingface": b'{"info": {"features": {'
    b'"observation.images.image": {"_type": "Image"}, '
    b'"observation.images.image2": {"_type": "Image"}, '
    b'"observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 8, "_type": "Sequence"}, '
    b'"action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "Sequence"}, '
    b'"timestamp": {"dtype": "float32", "_type": "Value"}, '
    b'"frame_index": {"dtype": "int64", "_type": "Value"}, '
    b'"episode_index": {"dtype": "int64", "_type": "Value"}, '
    b'"index": {"dtype": "int64", "_type": "Value"}, '
    b'"task_index": {"dtype": "int64", "_type": "Value"}}}}'
}

def patch_parquet(parquet_path, hf_metadata, rename_map):
    try:
        # Read the parquet table
        table = pq.read_table(parquet_path)

        # Apply renames if necessary
        if rename_map:
            new_names = [rename_map.get(name, name) for name in table.schema.names]
            if new_names != table.schema.names:
                table = table.rename_columns(new_names)

        # Update metadata
        new_meta = dict(table.schema.metadata or {})
        new_meta.update(hf_metadata)
        table = table.replace_schema_metadata(new_meta)

        # Write to temp file then atomically move back
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(tmp_fd)
        pq.write_table(table, tmp_path)
        shutil.move(tmp_path, parquet_path)

        # Debug print
        print(f"✅ Patched: {parquet_path}")
        print("   Columns:", table.schema.names)
        return True
    except Exception as e:
        print(f"❌ Failed on {parquet_path}: {e}")
        return False

if __name__ == "__main__":
    for dirpath, _, filenames in os.walk(ROOT_DIR):
        for fname in filenames:
            if fname.endswith(".parquet"):
                fpath = os.path.join(dirpath, fname)
                patch_parquet(fpath, HF_METADATA, rename_map)
