import pytest
from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs, step_mdp

from lerobot.lib.envs import SimxarmEnv


@pytest.mark.parametrize(
    "task,from_pixels,pixels_only",
    [
        ("lift", False, False),
        ("lift", True, False),
        ("lift", True, True),
        ("reach", False, False),
        ("reach", True, False),
        ("push", False, False),
        ("push", True, False),
        ("peg_in_box", False, False),
        ("peg_in_box", True, False),
    ],
)
def test_simxarm(task, from_pixels, pixels_only):
    env = SimxarmEnv(
        task,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        image_size=84 if from_pixels else None,
    )
    check_env_specs(env)

    print("observation_spec:", env.observation_spec)
    print("action_spec:", env.action_spec)
    print("reward_spec:", env.reward_spec)
    print("done_spec:", env.done_spec)
    print("success_spec:", env.success_spec)

    td = env.reset()
    print("reset tensordict", td)

    td = env.rand_step(td)
    print("random step tensordict", td)

    def simple_rollout(steps=100):
        # preallocate:
        data = TensorDict({}, [steps])
        # reset
        _data = env.reset()
        for i in range(steps):
            _data["action"] = env.action_spec.rand()
            _data = env.step(_data)
            data[i] = _data
            _data = step_mdp(_data, keep_other=True)
        return data

    print("data from rollout:", simple_rollout(100))
