from huggingface_hub import HfApi
hub_api = HfApi()
hub_api.create_tag("HuggingFaceVLA/libero", tag="v2.1", repo_type="dataset")
