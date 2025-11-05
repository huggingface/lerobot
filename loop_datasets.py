from huggingface_hub import HfApi, list_datasets

api = HfApi()
datasets = list_datasets(author="lerobot-data-collection")
for dataset in datasets:
    if "two-folds-dataset" in dataset.id:
        print("'" + dataset.id + "',", end=" ")
print("\n")