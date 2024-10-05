import torch

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config, set_global_seed
from tests.utils import DEFAULT_CONFIG_PATH


def main(env_name, policy_name, extra_overrides):
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
            obs[k] = batch[k]

    actions = policy.inference(obs)

    action, timestamp = policy.select_action(obs)

    print(actions[0])
    print(action)


if __name__ == "__main__":
    main("aloha", "act", ["policy.n_action_steps=10"])
