import json
from pathlib import Path

from huggingface_hub import HfApi

from lerobot import available_datasets
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

api = HfApi()
LOCAL_DIR = Path("outputs/test_artifacts/")
# LOCAL_DIR = Path("tests/data/")

datasets_info = api.list_datasets(author="lerobot")
hub_available_datasets = [info.id for info in datasets_info if info.id in available_datasets]

for repo_id in hub_available_datasets:
    print(repo_id)

    dataset_info = api.list_repo_refs(repo_id, repo_type="dataset")
    branches = [b.name for b in dataset_info.branches]
    if CODEBASE_VERSION in branches:
        # if "_image" not in repo_id:
        # print(f"{repo_id} already @{CODEBASE_VERSION}, skipping.")
        continue
    else:
        # Check if meta_data/info.json exists in the main branch
        files = api.list_repo_files(repo_id, repo_type="dataset", revision="main")
        info_file_path = "meta_data/info.json"

        if info_file_path in files:
            local_dir = LOCAL_DIR / repo_id
            local_dir.mkdir(exist_ok=True, parents=True)
            # Download the meta_data/info.json file from the main branch
            local_info_file_path = api.hf_hub_download(
                repo_id=repo_id,
                filename=info_file_path,
                revision="main",
                repo_type="dataset",
                local_dir=local_dir,
            )
        else:
            continue

        with open(local_info_file_path) as f:
            info_data = json.load(f)

        # Update the JSON data
        new_info_data = {}
        new_info_data["codebase_version"] = CODEBASE_VERSION
        for k, v in info_data.items():
            if k != "codebase_version":
                new_info_data[k] = v

        # Save the updated JSON file
        with open(local_info_file_path, "w") as f:
            json.dump(new_info_data, f, indent=4)

        # Upload the modified file to the new branch
        api.upload_file(
            path_or_fileobj=local_info_file_path,
            path_in_repo=info_file_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update meta_data/info.json for {CODEBASE_VERSION}",
            revision="main",
        )
        print(f"{repo_id} meta_data/info.json updated with new codebase version")

        # Now create a branch named after the new version by branching out from "main"
        # which is expected to be the preceding version
        api.create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION, revision="main")
        print(f"{repo_id} successfully updated @{CODEBASE_VERSION}")


def main():
    # TODO: from list of repos, download:
    #     - data/
    #     - meta_data/
    #     - video/{key}_episode_000001.mp4
    ...


if __name__ == "__main__":
    main()
