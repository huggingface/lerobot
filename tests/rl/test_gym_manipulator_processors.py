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

from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch
from torchvision.transforms import functional

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig, ImagePreprocessingConfig
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import VanillaObservationProcessorStep, create_transition
from lerobot.rl.gym_manipulator import make_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@pytest.mark.parametrize("name", ["real_robot", "gym_hil"])
@pytest.mark.parametrize("image_device", [None, "cpu", "cuda:1"])
def test_image_device_reaches_both_environment_pipelines(name, image_device):
    raw_config = {"name": name}
    if image_device is not None:
        raw_config["processor"] = {"image_preprocessing": {"image_device": image_device}}
    config = draccus.decode(HILSerlRobotEnvConfig, raw_config)
    env = SimpleNamespace(robot=SimpleNamespace(bus=SimpleNamespace(motors={})))
    teleop = SimpleNamespace(get_teleop_events=lambda: {}, get_action=lambda: {})

    env_processor, _ = make_processors(env, teleop, config)

    image_step = next(
        step for step in env_processor.steps if isinstance(step, VanillaObservationProcessorStep)
    )
    assert image_step.get_config()["image_device"] == (image_device or "cpu")


@pytest.mark.parametrize("image_device", ["cpu", "cuda", "mps"])
def test_real_robot_image_pipeline_matches_cpu_crop_resize(image_device):
    if image_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if image_device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    image_key = f"{OBS_IMAGES}.top"
    crop = (1, 2, 8, 12)
    config = HILSerlRobotEnvConfig(
        processor=HILSerlProcessorConfig(
            image_preprocessing=ImagePreprocessingConfig(
                image_device=image_device, crop_params_dict={image_key: crop}, resize_size=(4, 6)
            )
        )
    )
    env = SimpleNamespace(robot=SimpleNamespace(bus=SimpleNamespace(motors={})))
    teleop = SimpleNamespace(get_teleop_events=lambda: {}, get_action=lambda: {})
    env_processor, _ = make_processors(env, teleop, config, device=image_device)
    image = np.arange(11 * 17 * 3, dtype=np.uint8).reshape(11, 17, 3)
    state = np.array([1.0, 2.0], dtype=np.float32)
    original_image = image.copy()

    output = env_processor(create_transition(observation={"pixels": {"top": image}, "agent_pos": state}))
    observation = output[TransitionKey.OBSERVATION]

    expected = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    expected = functional.resize(functional.crop(expected, *crop), (4, 6)).clamp(0.0, 1.0)
    torch.testing.assert_close(observation[image_key].cpu(), expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(observation[OBS_STATE].cpu(), torch.from_numpy(state).unsqueeze(0))
    assert observation[image_key].device.type == image_device
    np.testing.assert_array_equal(image, original_image)
