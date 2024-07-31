#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, set_global_seed
from lerobot.scripts.train import make_optimizer_and_scheduler
from tests.utils import DEFAULT_CONFIG_PATH


def get_policy_stats(env_name, policy_name, extra_overrides):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            "device=cpu",
        ]
        + extra_overrides,
    )
    set_global_seed(1337)
    dataset = make_dataset(cfg)
    policy = make_policy(cfg, dataset_stats=dataset.stats)
    policy.train()
    optimizer, _ = make_optimizer_and_scheduler(cfg, policy)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    output_dict = policy.forward(batch)
    output_dict = {k: v for k, v in output_dict.items() if isinstance(v, torch.Tensor)}
    loss = output_dict["loss"]

    loss.backward()
    grad_stats = {}
    for key, param in policy.named_parameters():
        if param.requires_grad:
            grad_stats[f"{key}_mean"] = param.grad.mean()
            grad_stats[f"{key}_std"] = (
                param.grad.std() if param.grad.numel() > 1 else torch.tensor(float(0.0))
            )

    optimizer.step()
    param_stats = {}
    for key, param in policy.named_parameters():
        param_stats[f"{key}_mean"] = param.mean()
        param_stats[f"{key}_std"] = param.std() if param.numel() > 1 else torch.tensor(float(0.0))

    optimizer.zero_grad()
    policy.reset()

    # HACK: We reload a batch with no delta_timestamps as `select_action` won't expect a timestamps dimension
    dataset.delta_timestamps = None
    batch = next(iter(dataloader))
    obs = {}
    for k in batch:
        if k.startswith("observation"):
            obs[k] = batch[k]

    if "n_action_steps" in cfg.policy:
        actions_queue = cfg.policy.n_action_steps
    else:
        actions_queue = cfg.policy.n_action_repeats

    actions = {str(i): policy.select_action(obs).contiguous() for i in range(actions_queue)}
    return output_dict, grad_stats, param_stats, actions


def save_policy_to_safetensors(output_dir, env_name, policy_name, extra_overrides, file_name_extra):
    env_policy_dir = Path(output_dir) / f"{env_name}_{policy_name}{file_name_extra}"

    if env_policy_dir.exists():
        print(f"Overwrite existing safetensors in '{env_policy_dir}':")
        print(f" - Validate with: `git add {env_policy_dir}`")
        print(f" - Revert with: `git checkout -- {env_policy_dir}`")
        shutil.rmtree(env_policy_dir)

    env_policy_dir.mkdir(parents=True, exist_ok=True)
    output_dict, grad_stats, param_stats, actions = get_policy_stats(env_name, policy_name, extra_overrides)
    save_file(output_dict, env_policy_dir / "output_dict.safetensors")
    save_file(grad_stats, env_policy_dir / "grad_stats.safetensors")
    save_file(param_stats, env_policy_dir / "param_stats.safetensors")
    save_file(actions, env_policy_dir / "actions.safetensors")


if __name__ == "__main__":
    env_policies = [
        # ("xarm", "tdmpc", ["policy.use_mpc=false"], "use_policy"),
        # ("xarm", "tdmpc", ["policy.use_mpc=true"], "use_mpc"),
        # (
        #     "pusht",
        #     "diffusion",
        #     [
        #         "policy.n_action_steps=8",
        #         "policy.num_inference_steps=10",
        #         "policy.down_dims=[128, 256, 512]",
        #     ],
        #     "",
        # ),
        # ("aloha", "act", ["policy.n_action_steps=10"], ""),
        # ("aloha", "act", ["policy.n_action_steps=1000", "policy.chunk_size=1000"], "_1000_steps"),
        # ("dora_aloha_real", "act_real", ["policy.n_action_steps=10"], ""),
        # ("dora_aloha_real", "act_real_no_state", ["policy.n_action_steps=10"], ""),
    ]
    if len(env_policies) == 0:
        raise RuntimeError("No policies were provided!")
    for env, policy, extra_overrides, file_name_extra in env_policies:
        save_policy_to_safetensors(
            "tests/data/save_policy_to_safetensors", env, policy, extra_overrides, file_name_extra
        )
