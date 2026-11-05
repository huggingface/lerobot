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

"""Opt-in original-checkpoint parity test for RynnValue.

Set both checkpoint variables to compare the released implementation with a
converted LeRobot checkpoint. The original model runs in a clean subprocess so
its Transformers registrations cannot resolve to LeRobot's ported classes.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")

from lerobot.configs.rewards import RewardModelConfig  # noqa: E402
from lerobot.rewards import make_reward_model, make_reward_pre_post_processors  # noqa: E402
from tests.utils import require_cuda  # noqa: E402

ORIGINAL_CHECKPOINT = os.environ.get("LEROBOT_RYNNVALUE_ORIGINAL_CHECKPOINT")
CONVERTED_CHECKPOINT = os.environ.get("LEROBOT_RYNNVALUE_CONVERTED_CHECKPOINT")

pytestmark = pytest.mark.skipif(
    not ORIGINAL_CHECKPOINT or not CONVERTED_CHECKPOINT,
    reason=(
        "Set LEROBOT_RYNNVALUE_ORIGINAL_CHECKPOINT and "
        "LEROBOT_RYNNVALUE_CONVERTED_CHECKPOINT to run RynnValue parity"
    ),
)


def _deterministic_frames() -> tuple[list[Image.Image], torch.Tensor]:
    arrays = []
    for frame_index in range(3):
        y, x = np.indices((64, 64), dtype=np.uint16)
        array = np.stack(
            (
                (x + frame_index * 17) % 256,
                (y * 2 + frame_index * 31) % 256,
                (x + y + frame_index * 47) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        arrays.append(array)
    images = [Image.fromarray(array) for array in arrays]
    frames = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    return images, frames


@require_cuda
def test_original_and_lerobot_checkpoints_predict_same_remaining_time(tmp_path):
    instruction = "put the cube in the drawer"
    robot_description = "a single-arm robot"
    camera_description = "a fixed third-person camera"
    images, frames = _deterministic_frames()
    frames_path = tmp_path / "frames.npy"
    np.save(frames_path, frames.permute(0, 2, 3, 1).numpy())
    original_script = r"""
import json
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

checkpoint, frames_path, instruction, robot_description, camera_description = sys.argv[1:]
images = [Image.fromarray(frame) for frame in np.load(frames_path)]
config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
config._attn_implementation = "pred_slot_isolated_eager"
model = AutoModel.from_pretrained(
    checkpoint,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
batch = processor.process_episode(
    instruction=instruction,
    images=images,
    robot_description=robot_description,
    camera_description=camera_description,
)
inputs = {
    key: value.to("cuda") if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}
with torch.inference_mode():
    values = model(**inputs).value.pred_value.float()
if values.ndim == 1:
    flattened = values
elif values.ndim == 2:
    flattened = values.mean(dim=0)
else:
    raise ValueError(f"Unexpected original RynnValue prediction shape: {tuple(values.shape)}")
remaining_time = flattened.view(1, -1)[:, -1].cpu().tolist()
print("RYNNVALUE_PARITY_RESULT=" + json.dumps(remaining_time))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            original_script,
            ORIGINAL_CHECKPOINT,
            str(frames_path),
            instruction,
            robot_description,
            camera_description,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("RYNNVALUE_PARITY_RESULT=")
    )
    expected = torch.tensor(json.loads(result_line.partition("=")[2]), dtype=torch.float32)

    config = RewardModelConfig.from_pretrained(CONVERTED_CHECKPOINT)
    config.pretrained_path = CONVERTED_CHECKPOINT
    config.device = "cuda"
    config.max_frames = None
    config.robot_description = robot_description
    config.camera_description = camera_description
    model = make_reward_model(config).eval()
    preprocessor, _ = make_reward_pre_post_processors(config)
    encoded = preprocessor(
        {
            config.image_key: frames,
            config.task_key: instruction,
        }
    )
    actual = model.predict_remaining_time(encoded).remaining_time_s.detach().cpu()

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
