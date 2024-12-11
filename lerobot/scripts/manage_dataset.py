"""
Utilities to manage a dataset.

Examples of usage:

- Consolidate a dataset, by encoding images into videos and computing statistics:
```bash
python lerobot/scripts/manage_dataset.py consolidate \
    --repo-id $USER/koch_test
```

- Consolidate a dataset which is not uploaded on the hub yet:
```bash
python lerobot/scripts/manage_dataset.py consolidate \
    --repo-id $USER/koch_test \
    --local-files-only 1
```

- Upload a dataset on the hub:
```bash
python lerobot/scripts/manage_dataset.py push_to_hub \
    --repo-id $USER/koch_test
```
"""

import argparse
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory where the dataset will be stored (e.g. 'dataset/path').",
    )
    base_parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    base_parser.add_argument(
        "--local-files-only",
        type=int,
        default=0,
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )


    ############################################################################
    # consolidate
    parser_conso = subparsers.add_parser("consolidate", parents=[base_parser])
    parser_conso.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser_conso.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )

    ############################################################################
    # push_to_hub
    parser_push = subparsers.add_parser("push_to_hub", parents=[base_parser])
    parser_push.add_argument(
        "--tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional additional tags to categorize the dataset on the Hugging Face Hub. Use space-separated values (e.g. 'so100 indoor'). The tag 'LeRobot' will always be added.",
    )
    parser_push.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="Repo license. Must be one of https://huggingface.co/docs/hub/repositories-licenses. Defaults to mit.",
    )
    parser_push.add_argument(
        "--private",
        type=int,
        default=0,
        help="Create a private dataset repository on the Hugging Face Hub. Push publicly by default.",
    )

    args = parser.parse_args()
    kwargs = vars(args)

    mode = kwargs.pop("mode")
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    local_files_only = kwargs.pop("local_files_only")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        local_files_only=local_files_only,
    )

    if mode == "consolidate":
        dataset.consolidate(**kwargs)

    elif mode == "push_to_hub":
        private = kwargs.pop("private") == 1
        dataset.push_to_hub(private=private, **kwargs)

    elif mode == "remove_episode":
        remove_episode(**kwargs)

    elif mode == "delete_dataset":
        delete_dataset()
    
    elif mode == "_episode":




    