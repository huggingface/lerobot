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

"""Small policies and real processor pipelines for inference contract tests."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, PolicyRunner
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.evo1.processor_evo1 import Evo1ActionProcessorStep
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE, QUERY_KIND, QUERY_TEXT


def tiny_config(**kwargs):
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ENV: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        device="cpu",
        chunk_size=8,
        n_action_steps=3,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        **kwargs,
    )


def processors(config, relative=False):
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.ones(3) * 2},
    }
    pre, post = make_act_pre_post_processors(config, stats)
    if relative:
        step = RelativeActionsProcessorStep(enabled=True)
        pre.steps.insert(3, step)
        post.steps.insert(1, AbsoluteActionsProcessorStep(enabled=True, relative_step=step))
    return pre, post


def runner_for(policy, *, relative=False, modes=(ExecutionMode.CHUNK,), **kwargs):
    return PolicyRunner(
        policy,
        *processors(policy.config, relative=relative),
        action_interval=1 / 30,
        features=tuple(
            FeatureSpec(key, (3,), "float32", semantics="joint positions in radians")
            for key in (OBS_STATE, OBS_ENV_STATE)
        ),
        action_feature=FeatureSpec(ACTION, (3,), "float32", semantics="joint positions in radians"),
        modes=modes,
        **kwargs,
    )


def local_engine_for(policy, *, relative=False, rtc_config=None, queue_threshold=4):
    pytest.importorskip("datasets")
    from lerobot.inference import RTCInferenceEngine

    state_names = ("a.pos", "b.pos", "c.pos")
    env_names = ("env_a", "env_b", "env_c")
    return RTCInferenceEngine(
        policy,
        *processors(policy.config, relative=relative),
        robot_wrapper=SimpleNamespace(robot_type="test", action_features=dict.fromkeys(state_names, float)),
        rtc_config=rtc_config or policy.config.rtc_config,
        dataset_features={
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": state_names},
            OBS_ENV_STATE: {"dtype": "float32", "shape": (3,), "names": env_names},
        },
        task="pick up the cube",
        fps=30,
        device="cpu",
        rtc_queue_threshold=queue_threshold,
    )


def observation(value=1.0):
    return ObservationSnapshot(
        {OBS_STATE: np.full(3, value, dtype=np.float32), OBS_ENV_STATE: np.zeros(3, dtype=np.float32)},
        capture_time=10.0,
        task="pick up the cube",
        task_version=4,
        observation_id="obs-1",
    )


class ConformingPolicy(PreTrainedPolicy):
    """A new family exercises the public contract without a serving allowlist."""

    config_class = ACTConfig
    name = "conformance"

    def __init__(self, config):
        super().__init__(config)
        self.resets = 0
        self.last_kwargs = {}
        self.reset()

    def get_optim_params(self):
        return {}

    def reset(self):
        self.resets += 1
        self._action_queue = deque()

    def forward(self, batch):
        raise NotImplementedError

    def select_action(self, batch, **kwargs):
        raise AssertionError("The runner must not call select_action")

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        return torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)

    def supports_rtc(self):
        return True

    def supports_text_generation(self):
        return True

    def generate_text(self, batch):
        assert batch[QUERY_KIND] == "vqa"
        assert batch[QUERY_TEXT] == "What is visible?"
        return "a cube"


NAMES = ("shoulder.pos", "elbow.pos", "gripper.pos")


def make_runner(policy, *, state_names=NAMES, action_names=NAMES, padded=False, relative=False, **kwargs):
    pre, post = processors(policy.config, relative=relative)
    if padded:
        post.steps.insert(0, Evo1ActionProcessorStep(action_dim=3))
    return PolicyRunner(
        policy,
        pre,
        post,
        action_interval=1 / 30,
        features=(
            FeatureSpec(OBS_STATE, (3,), "float32", names=state_names, semantics="radians"),
            FeatureSpec(OBS_ENV_STATE, (3,), "float32", semantics="radians"),
        ),
        action_feature=FeatureSpec(ACTION, (3,), "float32", names=action_names, semantics="radians"),
        **kwargs,
    )


class PaddedPolicy(ConformingPolicy):
    name = "padded_conformance"

    def __init__(self, config):
        super().__init__(config)
        self.width = 6

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        return torch.arange(8 * self.width, dtype=torch.float32).reshape(1, 8, self.width)
