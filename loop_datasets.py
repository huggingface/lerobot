from huggingface_hub import HfApi, list_datasets

api = HfApi()
datasets = list_datasets(author="lerobot-data-collection")
for dataset in datasets:
    if "test" in dataset.id:
        print("'" + dataset.id + "',", end=" ")