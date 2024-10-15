## Using / Updating `CODEBASE_VERSION` (for maintainers)

Since our dataset pushed to the hub are decoupled with the evolution of this repo, we ensure compatibility of
the datasets with our code, we use a `CODEBASE_VERSION` (defined in
lerobot/common/datasets/lerobot_dataset.py) variable.

For instance, [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) has many versions to maintain backward compatibility between LeRobot codebase versions:
- [v1.0](https://huggingface.co/datasets/lerobot/pusht/tree/v1.0)
- [v1.1](https://huggingface.co/datasets/lerobot/pusht/tree/v1.1)
- [v1.2](https://huggingface.co/datasets/lerobot/pusht/tree/v1.2)
- [v1.3](https://huggingface.co/datasets/lerobot/pusht/tree/v1.3)
- [v1.4](https://huggingface.co/datasets/lerobot/pusht/tree/v1.4)
- [v1.5](https://huggingface.co/datasets/lerobot/pusht/tree/v1.5)
- [v1.6](https://huggingface.co/datasets/lerobot/pusht/tree/v1.6) <-- last version
- [main](https://huggingface.co/datasets/lerobot/pusht/tree/main) <-- points to the last version

Starting with v1.6, every dataset pushed to the hub or saved locally also have this version number in their
`info.json` metadata.

### Uploading a new dataset
If you are pushing a new dataset, you don't need to worry about any of the instructions below, nor to be
compatible with previous codebase versions. The `push_dataset_to_hub.py` script will automatically tag your
dataset with the current `CODEBASE_VERSION`.

### Updating an existing dataset
If you want to update an existing dataset, you need to change the `CODEBASE_VERSION` from `lerobot_dataset.py`
before running `push_dataset_to_hub.py`. This is especially useful if you introduce a breaking change
intentionally or not (i.e. something not backward compatible such as modifying the reward functions used,
deleting some frames at the end of an episode, etc.). That way, people running a previous version of the
codebase won't be affected by your change and backward compatibility is maintained.

However, you will need to update the version of ALL the other datasets so that they have the new
`CODEBASE_VERSION` as a branch in their hugging face dataset repository. Don't worry, there is an easy way
that doesn't require to run `push_dataset_to_hub.py`. You can just "branch-out" from the `main` branch on HF
dataset repo by running this script which corresponds to a `git checkout -b` (so no copy or upload needed):

```python
from huggingface_hub import HfApi

from lerobot import available_datasets
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

api = HfApi()

for repo_id in available_datasets:
    dataset_info = api.list_repo_refs(repo_id, repo_type="dataset")
    branches = [b.name for b in dataset_info.branches]
    if CODEBASE_VERSION in branches:
        print(f"{repo_id} already @{CODEBASE_VERSION}, skipping.")
        continue
    else:
        # Now create a branch named after the new version by branching out from "main"
        # which is expected to be the preceding version
        api.create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION, revision="main")
        print(f"{repo_id} successfully updated @{CODEBASE_VERSION}")
```
