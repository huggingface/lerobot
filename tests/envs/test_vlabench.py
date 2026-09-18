"""Real-simulator regression tests for `VLABenchEnv`'s seed plumbing (`reset(seed=...)`).

These build and reset the actual VLABench/MuJoCo simulator (no mocking) because the bug being
guarded against -- `env.reset(seed=X)` silently not affecting the sampled scene at all -- can only
be caught by observing genuine simulator state, not a stand-in. Skipped when VLABench isn't
installed (it's a heavy optional dependency, not part of the base test extras).
"""

import numpy as np

from lerobot.envs.configs import VLABenchEnv
from lerobot.envs import make_env
from tests.utils import skip_if_package_missing

TASK = "select_poker"


def _build_env():
    envs = make_env(VLABenchEnv(task=TASK), n_envs=1, use_async_envs=False)
    return envs[TASK][0]


@skip_if_package_missing("VLABench")
def test_same_seed_reset_reproduces_initial_scene():
    """Resetting the same live env twice with the same seed must reproduce the same initial
    object layout/target and the same rendered frame -- the ability to replay a specific eval
    episode depends on this."""
    env = _build_env()
    raw_env = env.envs[0]
    try:
        obs_a, _ = env.reset(seed=[1000])
        frame_a = raw_env.render().copy()
        target_a = raw_env._env.task.target_entity

        obs_b, _ = env.reset(seed=[1000])
        frame_b = raw_env.render().copy()
        target_b = raw_env._env.task.target_entity

        np.testing.assert_array_equal(obs_a["agent_pos"], obs_b["agent_pos"])
        np.testing.assert_array_equal(frame_a, frame_b)
        assert target_a == target_b
    finally:
        env.close()


@skip_if_package_missing("VLABench")
def test_different_seed_reset_changes_scene():
    """Different seeds must produce a genuinely different sampled scene -- at least the
    rendered frame and the sampled target entity, both of which the previous (no-op)
    `_seed_inner_env` implementation left completely unaffected by `seed`."""
    env = _build_env()
    raw_env = env.envs[0]
    try:
        env.reset(seed=[1000])
        frame_1000 = raw_env.render().copy()
        target_1000 = raw_env._env.task.target_entity

        env.reset(seed=[1001])
        frame_1001 = raw_env.render().copy()
        target_1001 = raw_env._env.task.target_entity

        assert not np.array_equal(frame_1000, frame_1001), (
            "rendered frame is identical across different seeds -- scene randomization is not "
            "actually seed-controlled"
        )
        assert target_1000 != target_1001, (
            "sampled target entity is identical across different seeds -- scene randomization is "
            "not actually seed-controlled"
        )
    finally:
        env.close()
