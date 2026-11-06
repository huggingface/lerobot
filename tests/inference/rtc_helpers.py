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

"""Gated local policies and engine setup for worker transition tests."""

from __future__ import annotations

from threading import Semaphore
from types import SimpleNamespace

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference import RTCInferenceEngine
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig


class Pipeline:
    steps: tuple = ()

    def __init__(self, width: int | None = None):
        self.width = width

    def __call__(self, value):
        return value if self.width is None else value[..., : self.width]

    def reset(self):
        pass


class ControlledPolicy:
    chunk_inference_spec = PreTrainedPolicy.chunk_inference_spec
    drop_queued_actions = PreTrainedPolicy.drop_queued_actions
    _action_queue_attrs = PreTrainedPolicy._action_queue_attrs

    def __init__(self, width: int = 2):
        self.width = width
        self.config = SimpleNamespace(
            n_obs_steps=1,
            chunk_size=20,
            n_action_steps=20,
            rtc_training_max_delay=4,
            action_feature=PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        )
        self.release = Semaphore(0)
        self.calls: list[tuple[torch.Tensor | None, int]] = []
        self.action_observations: list[torch.Tensor] = []
        self.text_observations: list[torch.Tensor] = []

    def predict_action_chunk(self, batch, *, inference_delay=0, prev_chunk_left_over=None):
        self.calls.append((prev_chunk_left_over, inference_delay))
        self.action_observations.append(batch["observation.state"].clone())
        assert self.release.acquire(timeout=3), "test must release the model call"
        return torch.ones(1, 20, self.width)

    def generate_text(self, batch):
        self.text_observations.append(batch["observation.state"].clone())
        return "answer"

    def reset(self):
        pass

    def supports_rtc(self):
        return True

    def supports_text_generation(self):
        return True


def make_engine(policy=None, robot=None, postprocessor=None, **kwargs):
    policy = policy or ControlledPolicy()
    robot = robot or SimpleNamespace(robot_type="mock", action_features={}, supports_hold=True)
    return RTCInferenceEngine(
        policy=policy,
        preprocessor=Pipeline(),
        postprocessor=postprocessor or Pipeline(width=2),
        robot_wrapper=robot,
        rtc_config=kwargs.pop("rtc_config", RTCConfig(enabled=True, execution_horizon=4)),
        dataset_features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["j1.pos", "j2.pos"]},
        },
        task="pick",
        fps=30,
        device="cpu",
        **kwargs,
    )


def observe(engine, value=0.0):
    engine.notify_observation({"j1.pos": value, "j2.pos": value})


def stop(engine):
    engine._policy.release.release(20)
    engine.stop()
