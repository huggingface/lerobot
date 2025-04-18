from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "villekuosmanen/dAgger_pack_easter_eggs_into_basket_2.0.1",
    # root='data/villekuosmanen/pack_easter_eggs_into_basket',
)
dataset.push_to_hub()
