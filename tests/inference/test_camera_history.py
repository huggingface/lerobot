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

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets")

from lerobot.inference import RTCInferenceEngine
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep
from tests.inference.rtc_helpers import ControlledPolicy, Pipeline
from tests.utils import wait_for

pytestmark = pytest.mark.dataset


class MemoryPolicy(ControlledPolicy):
    def __init__(self):
        super().__init__()
        self.batches = []

    def chunk_inference_spec(self):
        return replace(
            super().chunk_inference_spec(),
            current_observation_only=False,
            memory_image_key="observation.images.cam_high",
            memory_frames=1,
            memory_stride=25,
        )

    def predict_action_chunk(self, batch, **kwargs):
        self.batches.append(batch)
        return torch.ones(1, 20, 2)


@pytest.mark.parametrize("frame", [0, 25, 625])
@pytest.mark.parametrize("renamed", [False, True])
def test_local_chunk_history_uses_hardware_keys_until_preprocessing(frame, renamed):
    hardware_key = "observation.images.top" if renamed else "observation.images.cam_high"
    policy_key = "observation.images.cam_high"
    pre = PolicyProcessorPipeline(
        steps=[RenameObservationsProcessorStep(rename_map={hardware_key: policy_key} if renamed else {})]
    )
    policy = MemoryPolicy()
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=pre,
        postprocessor=Pipeline(width=2),
        robot_wrapper=SimpleNamespace(robot_type="test", action_features={}, supports_hold=True),
        rtc_config=RTCConfig(enabled=False),
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a.pos", "b.pos"]},
            hardware_key: {"dtype": "image", "shape": (2, 2, 3), "names": ["height", "width", "channels"]},
        },
        task="pick",
        fps=25,
        device="cpu",
    )
    engine.start()
    for index in range(frame + 1):
        engine.notify_observation(
            {
                "a.pos": 0.0,
                "b.pos": 0.0,
                hardware_key.removeprefix("observation.images."): np.full(
                    (2, 2, 3), index % 256, dtype=np.uint8
                ),
            }
        )
    engine.resume()
    try:
        assert wait_for(lambda: bool(policy.batches)), engine.failure_traceback
        batch = policy.batches[0]
        assert batch[policy_key].shape == (1, 2, 3, 2, 2)
        torch.testing.assert_close(
            batch[policy_key][0, 0], torch.full((3, 2, 2), max(0, frame - 25) % 256 / 255)
        )
        torch.testing.assert_close(batch[policy_key][0, 1], torch.full((3, 2, 2), frame % 256 / 255))
        assert batch[f"{policy_key}_is_pad"].tolist() == [[frame < 25, False]]
        if renamed:
            assert hardware_key not in batch and f"{hardware_key}_is_pad" not in batch
        engine.reset()
        assert not engine._memory_images
        engine.notify_observation(
            {
                "a.pos": 0.0,
                "b.pos": 0.0,
                hardware_key.removeprefix("observation.images."): np.zeros((2, 2, 3), dtype=np.uint8),
            }
        )
        assert engine._obs_holder["history"][1] == (True,)
    finally:
        engine.stop()
