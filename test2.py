import time

import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config, set_global_seed
from tests.utils import DEFAULT_CONFIG_PATH


def main(env_name, policy_name, extra_overrides):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            "device=mps",
        ]
        + extra_overrides,
    )
    set_global_seed(1337)
    dataset = make_dataset(cfg)
    policy = make_policy(cfg, dataset_stats=dataset.stats)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    obs = {}
    for k in batch:
        if k.startswith("observation"):
            obs[k] = batch[k].to("mps")

    # actions = policy.inference(obs)

    fps = 30

    for i in range(400):
        start_loop_t = time.perf_counter()

        next_action = policy.select_action(obs)  # noqa: F841

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        print(
            f"{i=}, {dt_s * 1000:5.2f} ({1/ dt_s:3.1f}hz) \t{policy._present_timestamp}\t{policy._present_action_timestamp}"
        )  # , {next_action.mean().item()}")

        # time.sleep(1/30)  # frequency at which we receive a new observation (30 Hz = 0.03 s)
        # time.sleep(0.5)  # frequency at which we receive a new observation (5 Hz = 0.2 s)


if __name__ == "__main__":
    main("aloha", "act", ["policy.n_action_steps=100"])
