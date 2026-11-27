# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO integration observations."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.envs.configs import LiberoEnvConfig
from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from tests.utils import import_without_simulator, resolve_without_simulator


def test_libero_numpy_adapter_parity(monkeypatch):
    canonical_observation = import_without_simulator("libero", monkeypatch).canonical_observation

    native = {
        "pixels": {"camera1": np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)},
        "robot_state": {
            "eef": {"pos": np.array([1, 2, 3], np.float32), "quat": np.array([0, 0, 0.6, 0.8], np.float32)},
            "gripper": {"qpos": np.array([0.1, 0.2], np.float32)},
        },
    }

    def batch(value):
        return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else value[None]

    expected = LiberoProcessorStep().observation(preprocess_observation(batch(native)))
    actual = canonical_observation(native)
    np.testing.assert_allclose(
        actual["observation.state"], expected["observation.state"][0].numpy(), atol=1e-6
    )
    np.testing.assert_array_equal(
        actual["observation.images.camera1"],
        (expected["observation.images.camera1"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
    )


def test_libero_declares_the_control_mode_it_runs(monkeypatch):
    relative = resolve_without_simulator(LiberoEnvConfig(task="libero_spatial").sim_settings(), monkeypatch)
    assert (relative.control_mode, relative.semantics) == ("delta", "libero-delta-v1")
    assert relative.action_names[0] == "eef.delta.x"
    # LIBERO's absolute mode commands end-effector poses, not deltas.
    absolute = resolve_without_simulator(
        LiberoEnvConfig(task="libero_spatial", control_mode="absolute").sim_settings(), monkeypatch
    )
    assert (absolute.control_mode, absolute.semantics) == ("eef_pose", "libero-eef-pose-v1")
    assert absolute.action_names[0] == "eef.x"


@pytest.mark.parametrize("init_states", [True, False])
def test_libero_init_state_metrics(monkeypatch, tmp_path, init_states):
    libero = import_without_simulator("libero", monkeypatch)

    states = np.arange(10).reshape(5, 2)

    monkeypatch.setattr(libero, "get_task_init_states", lambda *args, **kwargs: states)
    monkeypatch.setattr(libero, "get_libero_path", lambda *args: str(tmp_path))

    task = SimpleNamespace(
        name="test_task",
        language="Test task",
        problem_folder="test",
        bddl_file="test.bddl",
    )
    suite = SimpleNamespace(get_task=lambda task_id: task)

    env = libero.LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_spatial",
        camera_name="agentview_image",
        observation_width=2,
        observation_height=2,
        init_states=init_states,
        episode_index=1,
        n_envs=2,
        num_steps_wait=0,
    )

    raw_obs = {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

    simulator = MagicMock()
    simulator.reset.return_value = raw_obs
    simulator.set_init_state.return_value = raw_obs
    simulator.check_success.return_value = False
    env._env = simulator

    action = np.zeros(7, dtype=np.float32)

    for expected_index in (1, 3, 0):
        env.reset(seed=42)

        if init_states:
            np.testing.assert_array_equal(
                simulator.set_init_state.call_args.args[0],
                states[expected_index],
            )

        for step in range(2):
            terminal = step == 1
            simulator.check_success.return_value = terminal
            simulator.step.side_effect = lambda action, terminal=terminal: (
                raw_obs,
                0.0,
                False,
                {"metrics": {"existing_metric": 1.0}} if terminal else {},
            )
            _, _, terminated, _, info = env.step(action)
            assert terminated == terminal
            if terminal:
                assert info["metrics"]["existing_metric"] == 1.0

            if init_states:
                assert info["metrics"]["init_state_id"] == float(expected_index)
            else:
                assert "init_state_id" not in info.get("metrics", {})

    assert simulator.set_init_state.call_count == (3 if init_states else 0)
