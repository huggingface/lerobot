import os

from torchrl.data.replay_buffers import SamplerWithoutReplacement

import lerobot
from lerobot.common.datasets.aloha import AlohaDataset
from lerobot.scripts.visualize_dataset import render_dataset

print(lerobot.available_datasets)
# >>> ['aloha_sim_insertion_human', 'aloha_sim_insertion_scripted', 'aloha_sim_transfer_cube_human', 'aloha_sim_transfer_cube_scripted', 'pusht', 'xarm_lift_medium']

# we use this sampler to sample 1 frame after the other
sampler = SamplerWithoutReplacement(shuffle=False)

dataset = AlohaDataset("aloha_sim_transfer_cube_human", sampler=sampler, root=os.environ.get("DATA_DIR"))

video_paths = render_dataset(
    dataset,
    out_dir="outputs/visualize_dataset/example",
    max_num_samples=300,
    fps=50,
)
print(video_paths)
# ['outputs/visualize_dataset/example/episode_0.mp4']
