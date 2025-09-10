import os
import pyarrow.parquet as pq
import tempfile
import shutil

# Root directory of converted data
root_dir = "/raid/jade/libero_converted"

# No renaming
rename_map = {
    
}

# Hugging Face features metadata (constant across all files)
HF_METADATA = {
    b"huggingface": b'{"info": {"features": {"observation.images.image": {"_type": "Image"}, "observation.images.image2": {"_type": "Image"}, "state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 8, "_type": "Sequence"}, "actions": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "Sequence"}, "timestamp": {"dtype": "float32", "_type": "Value"}, "frame_index": {"dtype": "int64", "_type": "Value"}, "episode_index": {"dtype": "int64", "_type": "Value"}, "index": {"dtype": "int64", "_type": "Value"}, "task_index": {"dtype": "int64", "_type": "Value"}}}}'
}

def patch_parquet(parquet_path, hf_metadata):
    try:
        table = pq.read_table(parquet_path)

        # Merge metadata
        new_meta = dict(table.schema.metadata or {})
        new_meta.update(hf_metadata)

        # Apply metadata to table
        table = table.replace_schema_metadata(new_meta)

        # Write safely via temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(tmp_fd)
        pq.write_table(table, tmp_path)
        shutil.move(tmp_path, parquet_path)

        print(f"✅ Patched: {parquet_path}")
        return True
    except Exception as e:
        print(f"❌ Failed on {parquet_path}: {e}")
        return False

# Walk through all chunk dirs and patch parquet files
for dirpath, _, filenames in os.walk(root_dir):
    for fname in filenames:
        if fname.endswith(".parquet"):
            fpath = os.path.join(dirpath, fname)
            patch_parquet(fpath, HF_METADATA)#!/usr/bin/env python3

#!/usr/bin/env python3
import os
import pyarrow.parquet as pq
import tempfile
import shutil

# Explicit list of files to patch
FILES_TO_PATCH = [
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000021.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000022.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000023.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000024.parquet",
    "/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000025.parquet",
]

# Optional renaming map (fill in as needed)
rename_map = {
    # "old_column_name": "new_column_name",
    "image": "observation.images.image",
    "image2": "observation.images.image2",
    "actions": "action",
}

# Hugging Face features metadata (constant across all files)
HF_METADATA = {
    b"huggingface": b'{"info": {"features": {'
    b'"observation.images.image": {"_type": "Image"}, '
    b'"observation.images.image2": {"_type": "Image"}, '
    b'"state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 8, "_type": "Sequence"}, '
    b'"actions": {"feature": {"dtype": "float32", "_type": "Value"}, "length": 7, "_type": "Sequence"}, '
    b'"timestamp": {"dtype": "float32", "_type": "Value"}, '
    b'"frame_index": {"dtype": "int64", "_type": "Value"}, '
    b'"episode_index": {"dtype": "int64", "_type": "Value"}, '
    b'"index": {"dtype": "int64", "_type": "Value"}, '
    b'"task_index": {"dtype": "int64", "_type": "Value"}}}}'
}

def patch_parquet(parquet_path, hf_metadata, rename_map):
    try:
        # Load parquet table
        table = pq.read_table(parquet_path)

        # If renaming is needed
        if rename_map:
            schema = table.schema
            new_names = [
                rename_map.get(name, name) for name in schema.names
            ]
            table = table.rename_columns(new_names)

        # Merge schema metadata
        new_meta = dict(table.schema.metadata or {})
        new_meta.update(hf_metadata)

        # Replace metadata in table
        table = table.replace_schema_metadata(new_meta)

        # Write safely via temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
        os.close(tmp_fd)
        pq.write_table(table, tmp_path)

        # Replace original file
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
