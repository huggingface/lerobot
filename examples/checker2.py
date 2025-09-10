import pyarrow.parquet as pq

# # First parquet (cached HF version)
meta1 = pq.read_metadata("/raid/jade/.cache/huggingface/datasets/data/chunk-000/episode_000000.parquet")
meta1 = pq.read_metadata("//raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000019.parquet")
print("First parquet key_value_metadata:")
print(meta1.metadata)  # low-level file metadata
# print()
print("Second")
# Second parquet (your converted version)
meta2 = pq.read_metadata("//raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/episode_000019.parquet")
print("\nSecond parquet key_value_metadata:")
# print(meta2.metadata)

# from datasets import load_dataset
# root_dir = "/raid/jade/libero_converted"

# # Load all parquet files under the root_dir recursively
# ds = load_dataset("parquet", data_files=f"{root_dir}/**/*.parquet")

# print(ds)                 # prints split info
# print(ds["train"].features)  # check schema/features

# # Peek at one row
# example = ds["train"][0]
# print(example.keys())
# print(type(example["observation.images.image"]))
# print(type(example["observation.images.image2"]))

import pyarrow.parquet as pq

for ep in ["episode_000019.parquet", "episode_000021.parquet", "episode_000026.parquet"]:
    path = f"/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero/data/chunk-000/{ep}"
    schema = pq.read_schema(path)
    print(ep, schema.names)
