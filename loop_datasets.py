from huggingface_hub import HfApi, list_datasets

api = HfApi()
datasets = list_datasets(author="lerobot-data-collection")
print('"[', end="")
i=0
for dataset in datasets:
    if "three-folds-dataset" in dataset.id:
        print("'" + dataset.id + "',", end="")
print(']"',)
