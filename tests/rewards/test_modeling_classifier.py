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
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig
from lerobot.rewards.classifier.modeling_classifier import Classifier, ClassifierOutput
from lerobot.utils.constants import OBS_IMAGE, REWARD
from tests.utils import skip_if_package_missing


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


@skip_if_package_missing("transformers")
def test_binary_classifier_with_default_params():
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


@skip_if_package_missing("transformers")
def test_multiclass_classifier():
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


@skip_if_package_missing("transformers")
def test_default_device():
    config = RewardClassifierConfig()
    assert config.device is None or config.device == "cpu"

    config.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.num_cameras = 1
    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")


@skip_if_package_missing("transformers")
def test_explicit_device_setup():
    config = RewardClassifierConfig(device="cpu")
    assert config.device == "cpu"

    config.input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.num_cameras = 1
    classifier = Classifier(config)
    for p in classifier.parameters():
        assert p.device == torch.device("cpu")


def _make_tiny_classifier(num_cameras: int = 1) -> Classifier:
    config = RewardClassifierConfig(num_cameras=num_cameras)
    config.input_features = {
        f"{OBS_IMAGE}s.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
        for i in range(num_cameras)
    }
    config.output_features = {REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,))}
    return Classifier(config)


def test_classifier_trains_layers_on_top_of_frozen_backbone(tiny_classifier_backbone):
    classifier = _make_tiny_classifier()
    classifier.train()
    encoder = classifier.encoders[classifier.image_keys[0]]
    backbone, spatial_embeddings, projection, layer_norm = encoder[0], encoder[1], encoder[3], encoder[4]
    assert not any(p.requires_grad for p in backbone.parameters())
    # The frozen backbone stays in eval mode (BatchNorm running stats, no dropout), the rest trains.
    assert not any(module.training for module in backbone.modules())
    assert spatial_embeddings.training and classifier.classifier_head.training

    def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
        tensors = [*module.named_parameters(), *module.named_buffers()]
        return {name: tensor.detach().clone() for name, tensor in tensors}

    trainable = {"spatial_embeddings": spatial_embeddings, "projection": projection, "layer_norm": layer_norm}
    trainable_before = {module_name: snapshot(module) for module_name, module in trainable.items()}
    backbone_before = snapshot(backbone)

    batch = {
        f"{OBS_IMAGE}s.cam0": torch.rand(4, 3, 32, 32),
        REWARD: torch.tensor([0.0, 1.0, 0.0, 1.0]),
    }
    optimizer = torch.optim.AdamW([p for p in classifier.parameters() if p.requires_grad], lr=1e-2)
    loss, _ = classifier(batch)
    loss.backward()
    optimizer.step()

    for module_name, module in trainable.items():
        for name, param in module.named_parameters():
            assert param.grad is not None, f"{module_name}.{name} got no gradient"
            assert not torch.equal(param, trainable_before[module_name][name]), (
                f"{module_name}.{name} was not updated"
            )
    assert all(param.grad is None for param in backbone.parameters())
    assert any("running_mean" in name for name in backbone_before)
    backbone_after = snapshot(backbone)
    for name, tensor in backbone_before.items():
        assert torch.equal(backbone_after[name], tensor), f"frozen backbone tensor {name} changed"


def test_classifier_inference_does_not_track_gradients(tiny_classifier_backbone):
    classifier = _make_tiny_classifier()
    batch = {f"{OBS_IMAGE}s.cam0": torch.rand(2, 3, 32, 32)}

    # The rewards are thresholded, hence never require grad: check that no graph is built on the way.
    head_grad_enabled: list[bool] = []
    classifier.classifier_head.register_forward_hook(
        lambda *_: head_grad_enabled.append(torch.is_grad_enabled())
    )

    assert torch.is_grad_enabled()
    classifier.predict_reward(batch)
    classifier.compute_reward(batch)
    assert head_grad_enabled == [False, False]

    # Training still gets gradients.
    loss, _ = classifier({**batch, REWARD: torch.tensor([0.0, 1.0])})
    assert loss.requires_grad


def test_classifier_state_dict_keys_unchanged(tiny_classifier_backbone, tmp_path):
    classifier = _make_tiny_classifier(num_cameras=2)
    keys = set(classifier.state_dict())

    # Existing checkpoints store the backbone once under `encoder.` and once per camera under
    # `encoders.<camera>.0.`, followed by the per-camera layers at their `nn.Sequential` indices.
    backbone_keys = {key.removeprefix("encoder.") for key in keys if key.startswith("encoder.")}
    assert backbone_keys
    expected = {f"encoder.{key}" for key in backbone_keys}
    for camera in ("observation_images_cam0", "observation_images_cam1"):
        expected |= {f"encoders.{camera}.0.{key}" for key in backbone_keys}
        expected |= {
            f"encoders.{camera}.1.kernel",
            f"encoders.{camera}.3.weight",
            f"encoders.{camera}.3.bias",
            f"encoders.{camera}.4.weight",
            f"encoders.{camera}.4.bias",
        }
    expected |= {
        "classifier_head.0.weight",
        "classifier_head.0.bias",
        "classifier_head.2.weight",
        "classifier_head.2.bias",
        "classifier_head.4.weight",
        "classifier_head.4.bias",
    }
    assert keys == expected

    # A saved checkpoint loads back with the same weights and predictions. The weights are perturbed
    # first so that they differ from a fresh initialization.
    classifier.eval()
    with torch.no_grad():
        for param in classifier.parameters():
            param.add_(torch.randn_like(param))
    classifier.save_pretrained(tmp_path)
    loaded = Classifier.from_pretrained(tmp_path)
    loaded_state_dict = loaded.state_dict()
    for key, value in classifier.state_dict().items():
        torch.testing.assert_close(loaded_state_dict[key], value, msg=key)
    images = [torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)]
    torch.testing.assert_close(loaded.predict(images).logits, classifier.predict(images).logits)
