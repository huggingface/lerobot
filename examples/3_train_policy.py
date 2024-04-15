"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils import init_hydra_config

output_directory = Path("outputs/train/example_pusht_diffusion")
os.makedirs(output_directory, exist_ok=True)

overrides = [
    "env=pusht",
    "policy=diffusion",
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    "offline_steps=5000",
    "log_freq=250",
    "device=cuda",
]

cfg = init_hydra_config("lerobot/configs/default.yaml", overrides)

policy = DiffusionPolicy(
    cfg=cfg.policy,
    cfg_device=cfg.device,
    cfg_noise_scheduler=cfg.noise_scheduler,
    cfg_optimizer=cfg.optimizer,
    cfg_ema=cfg.ema,
    **cfg.policy,
)
policy.train()

dataset = make_dataset(cfg)

# create dataloader for offline training
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=cfg.policy.batch_size,
    shuffle=True,
    pin_memory=cfg.device != "cpu",
    drop_last=True,
)

for step, batch in enumerate(dataloader):
    info = policy(batch, step)

    if step % cfg.log_freq == 0:
        num_samples = (step + 1) * cfg.policy.batch_size
        loss = info["loss"]
        update_s = info["update_s"]
        print(f"step:{step} samples:{num_samples} loss:{loss:.3f} update_time:{update_s:.3f}(seconds)")


# Save the policy, configuration, and normalization stats for later use.
policy.save(output_directory / "model.pt")
OmegaConf.save(cfg, output_directory / "config.yaml")
torch.save(dataset.transform.transforms[-1].stats, output_directory / "stats.pth")
