import os
from pathlib import Path

import lerobot
from lerobot.common.datasets.aloha import AlohaDataset
from lerobot.scripts.visualize_dataset import render_dataset

print(lerobot.available_datasets)
# >>> ['aloha_sim_insertion_human', 'aloha_sim_insertion_scripted', 'aloha_sim_transfer_cube_human', 'aloha_sim_transfer_cube_scripted', 'pusht', 'xarm_lift_medium']

# TODO(rcadene): remove DATA_DIR
dataset = AlohaDataset("aloha_sim_transfer_cube_human", root=Path(os.environ.get("DATA_DIR")))

video_paths = render_dataset(
    dataset,
    out_dir="outputs/visualize_dataset/example",
    max_num_episodes=1,
)
print(video_paths)
# ['outputs/visualize_dataset/example/episode_0_top.mp4']
