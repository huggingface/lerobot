from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/pack_easter_eggs_into_basket",
    root='data/villekuosmanen/pack_easter_eggs_into_basket',
)
dataset.push_to_hub()
