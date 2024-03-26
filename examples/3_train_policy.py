"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import trange

from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.policies.diffusion.policy import DiffusionPolicy
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
    cfg_rgb_model=cfg.rgb_model,
    cfg_obs_encoder=cfg.obs_encoder,
    cfg_optimizer=cfg.optimizer,
    cfg_ema=cfg.ema,
    n_action_steps=cfg.n_action_steps + cfg.n_latency_steps,
    **cfg.policy,
)
policy.train()

offline_buffer = make_offline_buffer(cfg)

for offline_step in trange(cfg.offline_steps):
    train_info = policy.update(offline_buffer, offline_step)
    if offline_step % cfg.log_freq == 0:
        print(train_info)

# Save the policy, configuration, and normalization stats for later use.
policy.save(output_directory / "model.pt")
OmegaConf.save(cfg, output_directory / "config.yaml")
torch.save(offline_buffer.transform[-1].stats, output_directory / "stats.pth")
