import packaging.version

V2_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v2.0 which is not backward compatible with v1.x.
Please, use our conversion script. Modify the following command with your own task description:
```
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \\
    --repo-id {repo_id} \\
    --single-task "TASK DESCRIPTION."  # <---- /!\\ Replace TASK DESCRIPTION /!\\
```

A few examples to replace TASK DESCRIPTION: "Pick up the blue cube and place it into the bin.", "Insert the
peg into the socket.", "Slide open the ziploc bag.", "Take the elevator to the 1st floor.", "Open the top
cabinet, store the pot inside it then close the cabinet.", "Push the T-shaped block onto the T-shaped
target.", "Grab the spray paint on the shelf and place it in the bin on top of the robot dog.", "Fold the
sweatshirt.", ...

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

V21_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.
While current version of LeRobot is backward-compatible with it, the version of your dataset still uses global
stats instead of per-episode stats. Update your dataset stats to the new format using this command:
```
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py --repo-id={repo_id}
```

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = V2_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)
