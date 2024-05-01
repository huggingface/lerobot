import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, set_global_seed
from lerobot.scripts.train import make_optimizer
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE


def get_policy_stats(env_name, policy_name):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ],
    )
    set_global_seed(1337)
    dataset = make_dataset(cfg)
    policy = make_policy(cfg, dataset_stats=dataset.stats)
    policy.train()
    policy.to(DEVICE)
    optimizer, _ = make_optimizer(cfg, policy)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    batch = next(iter(dataloader))
    output_dict = policy.forward(batch)
    loss = output_dict["loss"]
    # TODO Check output dict values

    loss.backward()
    grad_stats = {}
    for key, param in policy.named_parameters():
        if param.requires_grad:
            grad_stats[f"{key}_mean"] = param.grad.mean()
            grad_stats[f"{key}_std"] = param.grad.std()

    optimizer.step()
    param_stats = {}
    for key, param in policy.named_parameters():
        param_stats[f"{key}_mean"] = param.mean()
        param_stats[f"{key}_std"] = param.std()

    optimizer.zero_grad()
    policy.reset()

    dataset.delta_timestamps = None
    batch = next(iter(dataloader))
    obs = {
        k: batch[k]
        for k in batch
        if k in ["observation.image", "observation.images.top", "observation.state"]
    }

    # TODO(aliberts): refacor `select_action` methods so that it expects `obs` instead of `batch`
    # if policy_name == "diffusion":
    #     obs = {k: batch[k] for k in ["observation.image", "observation.state"]}

    actions = {str(i): policy.select_action(obs).contiguous() for i in range(cfg.policy.n_action_steps)}
    return grad_stats, param_stats, actions


def save_policy_to_safetensors(output_dir, env_name, policy_name):
    env_policy_dir = Path(output_dir) / f"{env_name}_{policy_name}"

    if env_policy_dir.exists():
        shutil.rmtree(env_policy_dir)

    env_policy_dir.mkdir(parents=True, exist_ok=True)
    grad_stats, param_stats, actions = get_policy_stats(env_name, policy_name)
    save_file(grad_stats, env_policy_dir / "grad_stats")
    save_file(param_stats, env_policy_dir / "param_stats")
    save_file(actions, env_policy_dir / "actions")


if __name__ == "__main__":
    env_policies = [
        # ("xarm", "tdmpc", ["policy.mpc=true"]),
        # ("pusht", "tdmpc", ["policy.mpc=false"]),
        (
            "pusht",
            "diffusion",
        ),
        # ("aloha", "act"),
    ]
    for env, policy in env_policies:
        save_policy_to_safetensors("tests/data/save_policy_to_safetensors", env, policy)
