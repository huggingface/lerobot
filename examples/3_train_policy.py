"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 5000
device = torch.device("cuda")
log_freq = 250

# Load the dataset from Hugging Face hub (or from the local cache).
dataset = LeRobotDataset("lerobot/pusht")

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
cfg = DiffusionConfig()
policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)
policy.train()
policy.to(device)

# This policy makes use of past observations and a horizon of actions for training. The dataset needs to know
# about this. For that, the policy has a method that takes as input a frames-per-second (fps) argument and
# returns a dictionary mapping each data key to a list of relative timestamps. For example. If we need a
# horizon of 4 actions, starting from the "previous" frame and at 10 FPS, the dictionary would contain
# {"action": [-0.1, 0.0, 0.1, 0.2]}. The policy is equipped with a method to produce this dictionary.
# Here, we know that for the PushT simulation environment is 10.
dataset.delta_timestamps = policy.make_delta_timestamps(fps=10)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Run training loop.
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save a policy checkpoint.
policy.save_pretrained(output_directory)
