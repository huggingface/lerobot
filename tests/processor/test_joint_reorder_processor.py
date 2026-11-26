#!/usr/bin/env python

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

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import DataProcessorPipeline, JointReorderProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import ACTION

STEP = JointReorderProcessorStep(observation_indices={"q": [2, 0, 1]}, action_indices=[1, 0])


def test_reorders_listed_observations_and_the_action():
    q = torch.tensor([[0.0, 1.0, 2.0]])
    out = STEP(create_transition(observation={"q": q, "other": q}, action=torch.tensor([5.0, 6.0])))

    torch.testing.assert_close(out[TransitionKey.OBSERVATION]["q"], torch.tensor([[2.0, 0.0, 1.0]]))
    torch.testing.assert_close(out[TransitionKey.OBSERVATION]["other"], q)
    torch.testing.assert_close(out[TransitionKey.ACTION], torch.tensor([6.0, 5.0]))


def test_transform_features_uses_the_index_count():
    features = {
        PipelineFeatureType.OBSERVATION: {"q": PolicyFeature(type=FeatureType.STATE, shape=(29,))},
        PipelineFeatureType.ACTION: {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(29,))},
    }
    out = STEP.transform_features(features)

    assert out[PipelineFeatureType.OBSERVATION]["q"].shape == (3,)
    assert out[PipelineFeatureType.ACTION][ACTION].shape == (2,)


def test_save_and_load(tmp_path):
    DataProcessorPipeline(steps=[STEP], name="pre").save_pretrained(tmp_path)
    loaded = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="pre.json").steps[0]

    assert loaded.observation_indices == STEP.observation_indices
    assert loaded.action_indices == STEP.action_indices
