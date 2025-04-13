from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/eval_3Feb25",
    # root="data",
    # local_files_only=True,
)
dataset.push_to_hub()
