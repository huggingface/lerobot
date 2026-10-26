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

import numpy as np


def test_libero_numpy_adapter_parity(monkeypatch):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep
    from tests.utils import import_without_simulator

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
