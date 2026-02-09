import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

# /fsx/jade_choghari/data/libero_10_subtasks_kw_converted
dataset = LeRobotDataset(repo_id="lerobot/libero_10_image_subtask")

dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=2,
        shuffle=True,
)

cfg = PreTrainedConfig.from_pretrained(
    pretrained_name_or_path="/fsx/jade_choghari/models/pi05-base",
)
cfg.dtype = "bfloat16"

pre_processor, post_processor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path="/fsx/jade_choghari/models/pi05-base",
)
batch = next(iter(dataloader))
breakpoint()
batch1 = pre_processor(batch)
breakpoint()
print(batch.keys())
# print(batch['task_index_high_level'].shape)
# print(batch['task_index_high_level'])
# print(batch['user_prompt'][0])
# print(batch['robot_utterance'][0])
# print(batch['task'][0])

valid_episode_list = []
for episode_idx in range(len(dataset.meta.episodes)):
        subtask_index = dataset[episode_idx]["subtask_index"]
        valid_episode_list.append(episode_idx)

print(len(valid_episode_list))

# read this parquet /fsx/jade_choghari/outputs/pgen_annotations1/meta/tasks.parquett
# import pandas as pd
# tasks_df = pd.read_parquet('/fsx/jade_choghari/outputs/pgen_annotations1/meta/tasks.parquet')

# # print all
# print(tasks_df.columns)
# breakpoint()