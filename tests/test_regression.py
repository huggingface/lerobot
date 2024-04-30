import pytest
import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.train import make_optimizer
from tests.utils import DEFAULT_CONFIG_PATH, DEVICE, require_env


@pytest.mark.parametrize(
    "env_name,policy_name,extra_overrides",
    [
        # ("xarm", "tdmpc", ["policy.mpc=true"]),
        # ("pusht", "tdmpc", ["policy.mpc=false"]),
        ("pusht", "diffusion", []),
        ("aloha", "act", ["env.task=AlohaInsertion-v0", "dataset.repo_id=lerobot/aloha_sim_insertion_human"]),
        (
            "aloha",
            "act",
            ["env.task=AlohaInsertion-v0", "dataset.repo_id=lerobot/aloha_sim_insertion_scripted"],
        ),
        (
            "aloha",
            "act",
            ["env.task=AlohaTransferCube-v0", "dataset.repo_id=lerobot/aloha_sim_transfer_cube_human"],
        ),
        (
            "aloha",
            "act",
            ["env.task=AlohaTransferCube-v0", "dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted"],
        ),
    ],
)
@require_env
def test_backward(env_name, policy_name, extra_overrides):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
        + extra_overrides,
    )
    dataset = make_dataset(cfg)
    policy = make_policy(cfg, dataset_stats=dataset.stats)
    policy.train()
    policy.to(DEVICE)
    optimizer, lr_scheduler = make_optimizer(cfg, policy)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=cfg.policy.batch_size,
        shuffle=True,
        pin_memory=torch.device("cpu") != DEVICE,
        drop_last=True,
    )

    step = 0
    done = False
    training_steps = 1
    while not done:
        for batch in dataloader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step >= training_steps:
                done = True
                break


if __name__ == "__main__":
    test_backward(
        "aloha", "act", ["env.task=AlohaInsertion-v0", "dataset.repo_id=lerobot/aloha_sim_insertion_scripted"]
    )
