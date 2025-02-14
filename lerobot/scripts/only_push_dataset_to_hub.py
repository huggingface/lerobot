from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/agilex_take_off_lid_of_ice_box",
    root="data",
    local_files_only=True,
)
dataset.push_to_hub()
