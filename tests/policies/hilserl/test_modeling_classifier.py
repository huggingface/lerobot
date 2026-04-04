# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.sac.reward_model.modeling_classifier import ClassifierOutput
from lerobot.utils.constants import OBS_IMAGE, REWARD
from tests.utils import require_package


def test_classifier_output():
    output = ClassifierOutput(
        logits=torch.tensor([1, 2, 3]),
        probabilities=torch.tensor([0.1, 0.2, 0.3]),
        hidden_states=None,
    )

    assert (
        f"{output}"
        == "ClassifierOutput(logits=tensor([1, 2, 3]), probabilities=tensor([0.1000, 0.2000, 0.3000]), hidden_states=None)"
    )


@require_package("transformers")
def test_binary_classifier_with_default_params():
    from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

    config = RewardClassifierConfig()
    config.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,)),
    }
    config.normalization_mapping = {
        "VISUAL": NormalizationMode.IDENTITY,
        "REWARD": NormalizationMode.IDENTITY,
    }
    config.num_cameras = 1
    classifier = Classifier(config)

    batch_size = 10

    input = {
        OBS_IMAGE: torch.rand((batch_size, 3, 128, 128)),
        REWARD: torch.randint(low=0, high=2, size=(batch_size,)).float(),
    }

    images, labels = classifier.extract_images_and_labels(input)
    assert len(images) == 1
    assert images[0].shape == torch.Size([batch_size, 3, 128, 128])
    assert labels.shape == torch.Size([batch_size])

    output = classifier.predict(images)

    assert output is not None
    assert output.logits.size() == torch.Size([batch_size])
    assert not torch.isnan(output.logits).any(), "Tensor contains NaN values"
    assert output.probabilities.shape == torch.Size([batch_size])
    assert not torch.isnan(output.probabilities).any(), "Tensor contains NaN values"
    assert output.hidden_states.shape == torch.Size([batch_size, 256])
    assert not torch.isnan(output.hidden_states).any(), "Tensor contains NaN values"


@require_package("transformers")
def test_multiclass_classifier():
    from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

    num_classes = 5
    config = RewardClassifierConfig()
    config.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(num_classes,)),
    }
    config.num_cameras = 1
    config.num_classes = num_classes
    classifier = Classifier(config)

    batch_size = 10

    input = {
        OBS_IMAGE: torch.rand((batch_size, 3, 128, 128)),
        REWARD: torch.rand((batch_size, num_classes)),
    }

    images, labels = classifier.extract_images_and_labels(input)
    assert len(images) == 1
    assert images[0].shape == torch.Size([batch_size, 3, 128, 128])
    assert labels.shape == torch.Size([batch_size, num_classes])

    output = classifier.predict(images)

    assert output is not None
    assert output.logits.shape == torch.Size([batch_size, num_classes])
    assert not torch.isnan(output.logits).any(), "Tensor contains NaN values"
    assert output.probabilities.shape == torch.Size([batch_size, num_classes])
    assert not torch.isnan(output.probabilities).any(), "Tensor contains NaN values"
    assert output.hidden_states.shape == torch.Size([batch_size, 256])
    assert not torch.isnan(output.hidden_states).any(), "Tensor contains NaN values"


@require_package("transformers")
def test_default_device():
    from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

    config = RewardClassifierConfig()
    assert config.device == "cpu"

    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")


@require_package("transformers")
def test_explicit_device_setup():
    from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

    config = RewardClassifierConfig(device="cpu")
    assert config.device == "cpu"

    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")
