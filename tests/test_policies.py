import pytest
import torch

from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.utils import postprocess_action, preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.envs.factory import make_env
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.utils import init_hydra_config
from .utils import DEVICE, DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "env_name,policy_name,extra_overrides",
    [
        ("xarm", "tdmpc", ["policy.mpc=true"]),
        ("pusht", "tdmpc", ["policy.mpc=false"]),
        ("pusht", "diffusion", []),
        ("aloha", "act", ["env.task=AlohaInsertion-v0", "dataset_id=aloha_sim_insertion_human"]),
        ("aloha", "act", ["env.task=AlohaInsertion-v0", "dataset_id=aloha_sim_insertion_scripted"]),
        ("aloha", "act", ["env.task=AlohaTransferCube-v0", "dataset_id=aloha_sim_transfer_cube_human"]),
        ("aloha", "act", ["env.task=AlohaTransferCube-v0", "dataset_id=aloha_sim_transfer_cube_scripted"]),
        # TODO(aliberts): xarm not working with diffusion
        # ("xarm", "diffusion", []),
    ],
)
def test_policy(env_name, policy_name, extra_overrides):
    """
    Tests:
        - Making the policy object.
        - Checking that the policy follows the correct protocol.
        - Updating the policy.
        - Using the policy to select actions at inference time.
        - Test the action can be applied to the policy
    """
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
        + extra_overrides,
    )
    # Check that we can make the policy object.
    policy = make_policy(cfg)
    # Check that the policy follows the required protocol.
    assert isinstance(
        policy, Policy
    ), f"The policy does not follow the required protocol. Please see {Policy.__module__}.{Policy.__name__}."
    # Check that we run select_actions and get the appropriate output.
    dataset = make_dataset(cfg)
    env = make_env(cfg, num_parallel_envs=2)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=2,
        shuffle=True,
        pin_memory=DEVICE != "cpu",
        drop_last=True,
    )
    dl_iter = cycle(dataloader)

    batch = next(dl_iter)

    for key in batch:
        batch[key] = batch[key].to(DEVICE, non_blocking=True)

    # Test updating the policy
    policy.update(batch, step=0)

    # reset the policy and environment
    policy.reset()
    observation, _ = env.reset(seed=cfg.seed)

    # apply transform to normalize the observations
    observation = preprocess_observation(observation, dataset.transform)

    # send observation to device/gpu
    observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}

    # get the next action for the environment
    with torch.inference_mode():
        action = policy.select_action(observation, step=0)

    # apply inverse transform to unnormalize the action
    action = postprocess_action(action, dataset.transform)

    # Test step through policy
    env.step(action)
