import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

dataset = LeRobotDataset(repo_id="lerobot/libero")

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=4,
        shuffle=True,
)
batch = next(iter(dataloader))
print(batch.keys())

breakpoint()
