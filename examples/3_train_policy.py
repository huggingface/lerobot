"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import init_hydra_config

output_directory = Path("outputs/train/example_pusht_diffusion")
os.makedirs(output_directory, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 5000
device = torch.device("cuda")
log_freq = 250

# Set up the dataset.
hydra_cfg = init_hydra_config("lerobot/configs/default.yaml", overrides=["env=pusht"])
dataset = make_dataset(hydra_cfg)

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
cfg = DiffusionConfig()
# TODO(alexander-soare): Remove LR scheduler from the policy.
policy = DiffusionPolicy(cfg, lr_scheduler_num_training_steps=training_steps, dataset_stats=dataset.stats)
policy.train()
policy.to(device)

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

# Save the policy and configuration for later use.
policy.save(output_directory / "model.pt")
OmegaConf.save(hydra_cfg, output_directory / "config.yaml")
