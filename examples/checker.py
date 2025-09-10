from huggingface_hub import HfApi
api = HfApi()
# api.upload_large_folder(
#     repo_id="HuggingFaceVLA/libero",
#     repo_type="dataset",
#     folder_path="/raid/jade/.cache/huggingface/lerobot/HuggingFaceVLA/libero",
# )
api.upload_large_folder(
    repo_id="HuggingFaceVLA/metaworld_mt50",
    repo_type="dataset",
    folder_path="/raid/jade/.cache/huggingface/lerobot/metaworld_mt50",
)
# repo_id="HuggingFaceVLA/libero"
# # Upload extra files
# api.upload_file(
#     repo_id=repo_id,
#     repo_type="dataset",
#     path_or_fileobj="/raid/jade/libero_converted/README.md",
#     path_in_repo="README.md"
# )

# api.upload_folder(
#     repo_id=repo_id,
#     repo_type="dataset",
#     folder_path="/raid/jade/libero_converted/meta",
#     path_in_repo="meta"
# )
