import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

dataset = LeRobotDataset(repo_id="local", root="/fsx/jade_choghari/outputs/pgen_annotations1")

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=2,
        shuffle=True,
)

batch = next(iter(dataloader))
print(batch.keys())
print(batch['task_index_high_level'].shape)
print(batch['task_index_high_level'])
print(batch['user_prompt'][0])
print(batch['robot_utterance'][0])
print(batch['task'][0])
breakpoint()