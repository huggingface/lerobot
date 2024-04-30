import pytest
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, set_global_seed
from lerobot.scripts.train import make_optimizer
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, require_env


@pytest.mark.parametrize(
    "env_name,policy_name,extra_overrides",
    [
        # ("xarm", "tdmpc", ["policy.mpc=true"]),
        # ("pusht", "tdmpc", ["policy.mpc=false"]),
        ("pusht", "diffusion", []),
        ("aloha", "act"),
    ],
)
@require_env
def test_backward_compatibility(env_name, policy_name):
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

    # TODO(aliberts): refacor `select_action` methods so that it expects `obs` instead of `batch`
    if policy_name == "diffusion":
        batch = {k: batch[k] for k in ["observation.image", "observation.state"]}

    actions = {i: policy.select_action(batch) for i in range(cfg.policy.n_action_steps)}

    print(len(actions))


if __name__ == "__main__":
    # test_backward_compatibility("aloha", "act")
    test_backward_compatibility("pusht", "diffusion")
