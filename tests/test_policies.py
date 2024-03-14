
import pytest
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase

from lerobot.common.policies.factory import make_policy

from lerobot.common.policies.abstract import AbstractPolicy

from .utils import DEVICE, init_config


@pytest.mark.parametrize(
    "env_name,policy_name",
    [
        ("simxarm", "tdmpc"),
        ("pusht", "tdmpc"),
        ("simxarm", "diffusion"),
        ("pusht", "diffusion"),
    ],
)
def test_factory(env_name, policy_name):
    cfg = init_config(
        overrides=[
            f"env={env_name}",
            f"policy={policy_name}",
            f"device={DEVICE}",
        ]
    )
    policy = make_policy(cfg)


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
        def __init__(self):
            super().__init__()
            self.n_action_steps = n_action_steps
            self.n_policy_invocations = 0

        def select_action(self):
            self.n_policy_invocations += 1
            return torch.stack([torch.tensor([i]) for i in range(self.n_action_steps)]).unsqueeze(0)


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
    assert torch.equal(rollout['observation'].flatten(), torch.arange(terminate_at + 1))
