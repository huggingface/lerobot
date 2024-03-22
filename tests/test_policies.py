from omegaconf import open_dict
import pytest
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase

from lerobot.common.policies.factory import make_policy
from lerobot.common.envs.factory import make_env
from lerobot.common.datasets.factory import make_offline_buffer
from lerobot.common.policies.abstract import AbstractPolicy

from .utils import DEVICE, init_config

@pytest.mark.parametrize(
    "env_name,policy_name,extra_overrides",
    [
        ("simxarm", "tdmpc", ["policy.mpc=true"]),
        ("pusht", "tdmpc", ["policy.mpc=false"]),
        ("pusht", "diffusion", []),
        ("aloha", "act", ["env.task=sim_insertion", "dataset_id=aloha_sim_insertion_human"]),
        ("aloha", "act", ["env.task=sim_insertion", "dataset_id=aloha_sim_insertion_scripted"]),
        ("aloha", "act", ["env.task=sim_transfer_cube", "dataset_id=aloha_sim_transfer_cube_human"]),
        ("aloha", "act", ["env.task=sim_transfer_cube", "dataset_id=aloha_sim_transfer_cube_scripted"]),
        # TODO(aliberts): simxarm not working with diffusion
        # ("simxarm", "diffusion", []),
    ],
)
def test_concrete_policy(env_name, policy_name, extra_overrides):
    """
    Tests:
        - Making the policy object.
        - Updating the policy.
        - Using the policy to select actions at inference time.
    """
    cfg = init_config(
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
        + extra_overrides
    )
    # Check that we can make the policy object.
    policy = make_policy(cfg)
    # Check that we run select_actions and get the appropriate output.
    offline_buffer = make_offline_buffer(cfg)
    env = make_env(cfg, transform=offline_buffer.transform)

    if env_name != "aloha":
        # TODO(alexander-soare): Fix this part of the test. PrioritizedSliceSampler raises NotImplementedError:
        # seq_length as a list is not supported for now.
        policy.update(offline_buffer, torch.tensor(0, device=DEVICE))

    action = policy(
        env.observation_spec.rand()["observation"].to(DEVICE),
        torch.tensor(0, device=DEVICE),
    )
    assert action.shape == env.action_spec.shape


def test_abstract_policy_forward():
    """
    Given an underlying policy that produces an action trajectory with n_action_steps actions, checks that:
        - The policy is invoked the expected number of times during a rollout.
        - The environment's termination condition is respected even when part way through an action trajectory.
        - The observations are returned correctly.
    """

    n_action_steps = 8  # our test policy will output 8 action step horizons
    terminate_at = 10  # some number that is more than n_action_steps but not a multiple
    rollout_max_steps = terminate_at + 1  # some number greater than terminate_at

    # A minimal environment for testing.
    class StubEnv(EnvBase):

        def __init__(self):
            super().__init__()
            self.action_spec = UnboundedContinuousTensorSpec(shape=(1,))
            self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))

        def _step(self, tensordict: TensorDict) -> TensorDict:
            self.invocation_count += 1
            return TensorDict(
                {
                    "observation": torch.tensor([self.invocation_count]),
                    "reward": torch.tensor([self.invocation_count]),
                    "terminated": torch.tensor(
                        tensordict["action"].item() == terminate_at
                    ),
                }
            )

        def _reset(self, tensordict: TensorDict) -> TensorDict:
            self.invocation_count = 0
            return TensorDict(
                {
                    "observation": torch.tensor([self.invocation_count]),
                    "reward": torch.tensor([self.invocation_count]),
                }
            )

        def _set_seed(self, seed: int | None):
            return

    class StubPolicy(AbstractPolicy):
        name = "stub"

        def __init__(self):
            super().__init__(n_action_steps)
            self.n_policy_invocations = 0

        def update(self):
            pass

        def select_actions(self):
            self.n_policy_invocations += 1
            return torch.stack(
                [torch.tensor([i]) for i in range(self.n_action_steps)]
            ).unsqueeze(0)

    env = StubEnv()
    policy = StubPolicy()
    policy = TensorDictModule(
        policy,
        in_keys=[],
        out_keys=["action"],
    )

    # Keep track to make sure the policy is called the expected number of times
    rollout = env.rollout(rollout_max_steps, policy)

    assert len(rollout) == terminate_at + 1  # +1 for the reset observation
    assert policy.n_policy_invocations == (terminate_at // n_action_steps) + 1
    assert torch.equal(rollout["observation"].flatten(), torch.arange(terminate_at + 1))
