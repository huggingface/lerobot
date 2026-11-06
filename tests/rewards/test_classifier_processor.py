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
"""Tests for Reward Classifier processor."""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    DataProcessorPipeline,
    DeviceProcessorStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RewardClassifierProcessorStep,
    TransitionKey,
)
from lerobot.processor.converters import create_transition, transition_to_batch
from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig
from lerobot.rewards.classifier.modeling_classifier import Classifier
from lerobot.rewards.classifier.processor_classifier import make_classifier_processor
from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES, OBS_STATE, REWARD


def create_default_config():
    """Create a default Reward Classifier configuration for testing."""
    config = RewardClassifierConfig()
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "reward": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    config.normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY,
        FeatureType.ACTION: NormalizationMode.IDENTITY,  # No normalization for classifier output
    }
    config.device = "cpu"
    return config


def create_default_stats():
    """Create default dataset statistics for testing."""
    return {
        OBS_STATE: {"mean": torch.zeros(10), "std": torch.ones(10)},
        OBS_IMAGE: {},  # No normalization for images
        "reward": {},  # No normalization for classifier output
    }


def test_make_classifier_processor_basic():
    """Test basic creation of Classifier processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Check processor names
    assert preprocessor.name == "classifier_preprocessor"
    assert postprocessor.name == "classifier_postprocessor"

    # Check steps in preprocessor
    assert len(preprocessor.steps) == 3
    assert isinstance(preprocessor.steps[0], NormalizerProcessorStep)  # For input features
    assert isinstance(preprocessor.steps[1], NormalizerProcessorStep)  # For output features
    assert isinstance(preprocessor.steps[2], DeviceProcessorStep)

    # Check steps in postprocessor
    assert len(postprocessor.steps) == 2
    assert isinstance(postprocessor.steps[0], DeviceProcessorStep)
    assert isinstance(postprocessor.steps[1], IdentityProcessorStep)


def test_classifier_processor_normalization():
    """Test that Classifier processor correctly normalizes data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Create test data
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor
    processed = preprocessor(batch)

    # Check that data is processed
    assert processed[OBS_STATE].shape == (10,)
    assert processed[OBS_IMAGE].shape == (3, 224, 224)
    assert processed[TransitionKey.ACTION].shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classifier_processor_cuda():
    """Test Classifier processor with CUDA device."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Create CPU data
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is on CUDA
    assert processed[OBS_STATE].device.type == "cuda"
    assert processed[OBS_IMAGE].device.type == "cuda"
    assert processed[TransitionKey.ACTION].device.type == "cuda"

    # Process through postprocessor
    postprocessed = postprocessor(processed[TransitionKey.ACTION])

    # Check that output is back on CPU
    assert postprocessed.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classifier_processor_accelerate_scenario():
    """Test Classifier processor in simulated Accelerate scenario."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Simulate Accelerate: data already on GPU
    device = torch.device("cuda:0")
    observation = {
        OBS_STATE: torch.randn(10).to(device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(device),
    }
    action = torch.randn(1).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on same GPU
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION].device == device


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
def test_classifier_processor_multi_gpu():
    """Test Classifier processor with multi-GPU setup."""
    config = create_default_config()
    config.device = "cuda:0"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Simulate data on different GPU
    device = torch.device("cuda:1")
    observation = {
        OBS_STATE: torch.randn(10).to(device),
        OBS_IMAGE: torch.randn(3, 224, 224).to(device),
    }
    action = torch.randn(1).to(device)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data stays on cuda:1
    assert processed[OBS_STATE].device == device
    assert processed[OBS_IMAGE].device == device
    assert processed[TransitionKey.ACTION].device == device


def test_classifier_processor_without_stats():
    """Test Classifier processor creation without dataset statistics."""
    config = create_default_config()

    preprocessor, postprocessor = make_classifier_processor(config, dataset_stats=None)

    # Should still create processors
    assert preprocessor is not None
    assert postprocessor is not None

    # Process should still work
    observation = {
        OBS_STATE: torch.randn(10),
        OBS_IMAGE: torch.randn(3, 224, 224),
    }
    action = torch.randn(1)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    processed = preprocessor(batch)
    assert processed is not None


def test_classifier_processor_save_and_load():
    """Test saving and loading Classifier processor."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save preprocessor
        preprocessor.save_pretrained(tmpdir)

        # Load preprocessor
        loaded_preprocessor = DataProcessorPipeline.from_pretrained(
            tmpdir, config_filename="classifier_preprocessor.json"
        )

        # Test that loaded processor works
        observation = {
            OBS_STATE: torch.randn(10),
            OBS_IMAGE: torch.randn(3, 224, 224),
        }
        action = torch.randn(1)
        transition = create_transition(observation, action)
        batch = transition_to_batch(transition)

        processed = loaded_preprocessor(batch)
        assert processed[OBS_STATE].shape == (10,)
        assert processed[OBS_IMAGE].shape == (3, 224, 224)
        assert processed[TransitionKey.ACTION].shape == (1,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_classifier_processor_mixed_precision():
    """Test Classifier processor with mixed precision."""
    config = create_default_config()
    config.device = "cuda"
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Replace DeviceProcessorStep with one that uses float16
    modified_steps = []
    for step in preprocessor.steps:
        if isinstance(step, DeviceProcessorStep):
            modified_steps.append(DeviceProcessorStep(device=config.device, float_dtype="float16"))
        else:
            modified_steps.append(step)
    preprocessor.steps = modified_steps

    # Create test data
    observation = {
        OBS_STATE: torch.randn(10, dtype=torch.float32),
        OBS_IMAGE: torch.randn(3, 224, 224, dtype=torch.float32),
    }
    action = torch.randn(1, dtype=torch.float32)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that data is converted to float16
    assert processed[OBS_STATE].dtype == torch.float16
    assert processed[OBS_IMAGE].dtype == torch.float16
    assert processed[TransitionKey.ACTION].dtype == torch.float16


def test_classifier_processor_batch_data():
    """Test Classifier processor with batched data."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Test with batched data
    batch_size = 16
    observation = {
        OBS_STATE: torch.randn(batch_size, 10),
        OBS_IMAGE: torch.randn(batch_size, 3, 224, 224),
    }
    action = torch.randn(batch_size, 1)
    transition = create_transition(observation, action)
    batch = transition_to_batch(transition)

    # Process through preprocessor

    processed = preprocessor(batch)

    # Check that batch dimension is preserved
    assert processed[OBS_STATE].shape == (batch_size, 10)
    assert processed[OBS_IMAGE].shape == (batch_size, 3, 224, 224)
    assert processed[TransitionKey.ACTION].shape == (batch_size, 1)


def test_classifier_processor_postprocessor_identity():
    """Test that Classifier postprocessor uses IdentityProcessor correctly."""
    config = create_default_config()
    stats = create_default_stats()

    preprocessor, postprocessor = make_classifier_processor(config, stats)

    # Create test data for postprocessor
    reward = torch.tensor([[0.8], [0.3], [0.9]])
    transition = create_transition(action=reward)
    _ = transition_to_batch(transition)

    # Process through postprocessor
    processed = postprocessor(reward)

    # IdentityProcessor should leave values unchanged (except device)
    assert torch.allclose(processed.cpu(), reward.cpu())
    assert processed.device.type == "cpu"


CAMERA_KEY = f"{OBS_IMAGES}.front"


def _save_trained_classifier(
    path: Path,
    save_preprocessor: bool = True,
    image_stats: bool = True,
    edit_normalizer: Callable[[NormalizerProcessorStep], None] | None = None,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """Save a classifier and its preprocessor the way `lerobot-train` does; return the preprocessor.

    `edit_normalizer` modifies the preprocessor's normalizer of the input features before it is saved.
    """
    # Default normalization mapping: VISUAL -> MEAN_STD.
    config = RewardClassifierConfig(num_cameras=1)
    config.input_features = {
        CAMERA_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {REWARD: PolicyFeature(type=FeatureType.REWARD, shape=(1,))}
    stats = {OBS_STATE: {"mean": torch.zeros(6), "std": torch.ones(6)}}
    if image_stats:
        stats[CAMERA_KEY] = {
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        }
    preprocessor, _ = make_classifier_processor(config, dataset_stats=stats)
    if edit_normalizer is not None:
        input_normalizer = preprocessor.steps[0]
        assert isinstance(input_normalizer, NormalizerProcessorStep)
        edit_normalizer(input_normalizer)

    Classifier(config).save_pretrained(path)
    if save_preprocessor:
        preprocessor.save_pretrained(path)
    return preprocessor


def test_reward_classifier_step_preprocesses_images_as_in_training(
    tiny_classifier_backbone, tmp_path, caplog
):
    training_preprocessor = _save_trained_classifier(tmp_path)
    with caplog.at_level(logging.WARNING):
        step = RewardClassifierProcessorStep(pretrained_path=str(tmp_path), device="cpu")
    assert step.reward_preprocessor is not None
    assert "will NOT be normalized" not in caplog.text
    classifier = step.reward_classifier

    classifier_inputs: dict[str, torch.Tensor] = {}
    predict_reward = classifier.predict_reward

    def spy_predict_reward(batch: dict[str, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        classifier_inputs.update(batch)
        return predict_reward(batch, threshold=threshold)

    classifier.predict_reward = spy_predict_reward

    # What the env pipeline hands the step: float images in [0, 1], batched, not normalized.
    image = torch.rand(1, 3, 32, 32)
    observation = {CAMERA_KEY: image.clone(), OBS_STATE: torch.rand(1, 6)}
    result = step(create_transition(observation=observation, reward=0.0, done=False))

    # The classifier gets the images exactly as the training preprocessor produces them.
    trained_on = training_preprocessor({CAMERA_KEY: image.clone()})[CAMERA_KEY]
    torch.testing.assert_close(classifier_inputs[CAMERA_KEY], trained_on)
    assert not torch.allclose(trained_on, image)

    # Hence the same probabilities, and the reward they imply.
    with torch.inference_mode():
        online_probs = classifier.predict([classifier_inputs[CAMERA_KEY]]).probabilities
        training_probs = classifier.predict([trained_on]).probabilities
        raw_probs = classifier.predict([image]).probabilities
    torch.testing.assert_close(online_probs, training_probs)
    assert not torch.allclose(online_probs, raw_probs)
    success = training_probs.item() > step.success_threshold
    assert result[TransitionKey.REWARD] == (step.success_reward if success else 0.0)

    # Only the classifier's inputs are normalized, not the observation passed down the pipeline.
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][CAMERA_KEY], image)


def test_reward_classifier_step_without_saved_preprocessor_warns(tiny_classifier_backbone, tmp_path, caplog):
    _save_trained_classifier(tmp_path, save_preprocessor=False)

    with caplog.at_level(logging.WARNING):
        step = RewardClassifierProcessorStep(pretrained_path=str(tmp_path), device="cpu")

    assert step.reward_preprocessor is None
    assert "has no saved preprocessor ('classifier_preprocessor.json')" in caplog.text
    assert "will NOT be normalized" in caplog.text
    # No advice to migrate the checkpoint, which does not apply to reward classifiers.
    assert "migrat" not in caplog.text

    # The images are passed to the classifier as they are, as before.
    observation = {CAMERA_KEY: torch.rand(1, 3, 32, 32)}
    result = step(create_transition(observation=observation, reward=0.0, done=False))
    with torch.inference_mode():
        probs = step.reward_classifier.predict([observation[CAMERA_KEY]]).probabilities
    success = probs.item() > step.success_threshold
    assert result[TransitionKey.REWARD] == (step.success_reward if success else 0.0)


# These replace the normalizer's dicts rather than modify them in place, as they are shared with the config.
def _drop_image_feature(normalizer: NormalizerProcessorStep) -> None:
    normalizer.features = {key: ft for key, ft in normalizer.features.items() if key != CAMERA_KEY}


def _map_visual_to_identity(normalizer: NormalizerProcessorStep) -> None:
    normalizer.norm_map = {**normalizer.norm_map, FeatureType.VISUAL: NormalizationMode.IDENTITY}


def _only_normalize_state(normalizer: NormalizerProcessorStep) -> None:
    normalizer.normalize_observation_keys = {OBS_STATE}


@pytest.mark.parametrize(
    "image_stats, edit_normalizer",
    [
        (False, None),
        (True, _drop_image_feature),
        (True, _map_visual_to_identity),
        (True, _only_normalize_state),
    ],
    ids=["no_image_stats", "image_not_in_features", "visual_mapped_to_identity", "image_key_excluded"],
)
def test_reward_classifier_step_warns_when_preprocessor_does_not_normalize_images(
    tiny_classifier_backbone, tmp_path, caplog, image_stats, edit_normalizer
):
    training_preprocessor = _save_trained_classifier(
        tmp_path, image_stats=image_stats, edit_normalizer=edit_normalizer
    )
    # The saved preprocessor indeed passes the images through unchanged.
    image = torch.rand(1, 3, 32, 32)
    torch.testing.assert_close(training_preprocessor({CAMERA_KEY: image.clone()})[CAMERA_KEY], image)

    with caplog.at_level(logging.WARNING):
        step = RewardClassifierProcessorStep(pretrained_path=str(tmp_path), device="cpu")

    assert step.reward_preprocessor is not None
    assert "does not normalize" in caplog.text
    assert CAMERA_KEY in caplog.text


def _hub_error(error_cls: type[Exception], status_code: int) -> Exception:
    request = httpx.Request("GET", "https://huggingface.co/user/reward-classifier")
    return error_cls("Hub error", response=httpx.Response(status_code, request=request))


def test_reward_classifier_step_hub_repo_without_preprocessor_warns(monkeypatch, caplog):
    def hf_hub_download(**kwargs: Any) -> str:
        raise _hub_error(RemoteEntryNotFoundError, 404)

    monkeypatch.setattr("lerobot.processor.hil_processor.hf_hub_download", hf_hub_download)
    step = RewardClassifierProcessorStep()

    with caplog.at_level(logging.WARNING):
        assert step._load_reward_preprocessor("user/reward-classifier") is None
    assert "has no saved preprocessor" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        _hub_error(RepositoryNotFoundError, 401),
        ConnectionError("Network unreachable"),
    ],
    ids=["unauthorized", "network"],
)
def test_reward_classifier_step_raises_when_preprocessor_lookup_fails(monkeypatch, error):
    """Only a preprocessor file that does not exist falls back to unnormalized inputs."""

    def hf_hub_download(**kwargs: Any) -> str:
        raise error

    monkeypatch.setattr("lerobot.processor.hil_processor.hf_hub_download", hf_hub_download)
    step = RewardClassifierProcessorStep()

    with pytest.raises(type(error)):
        step._load_reward_preprocessor("user/reward-classifier")


HUB_REPO_ID = "user/reward-classifier"


@pytest.fixture
def offline_hub_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run with `HF_HUB_OFFLINE=1` on an empty Hub cache; return the cache folder of `HUB_REPO_ID`."""
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    repo_cache = tmp_path / "models--user--reward-classifier"
    # What loading the classifier online caches: the commit of `main` and the classifier files.
    commit = "0" * 40
    (repo_cache / "refs").mkdir(parents=True)
    (repo_cache / "refs" / "main").write_text(commit)
    (repo_cache / "snapshots" / commit).mkdir(parents=True)
    (repo_cache / "snapshots" / commit / "config.json").write_text("{}")
    return repo_cache


def test_reward_classifier_step_offline_with_cached_missing_preprocessor_warns(offline_hub_cache, caplog):
    # An earlier online lookup found that the repository has no preprocessor, and cached that.
    no_exist = offline_hub_cache / ".no_exist" / ("0" * 40) / "classifier_preprocessor.json"
    no_exist.parent.mkdir(parents=True)
    no_exist.touch()
    step = RewardClassifierProcessorStep()

    with caplog.at_level(logging.WARNING):
        assert step._load_reward_preprocessor(HUB_REPO_ID) is None
    assert "has no saved preprocessor" in caplog.text


def test_reward_classifier_step_offline_with_cached_preprocessor_finds_it(offline_hub_cache):
    (offline_hub_cache / "snapshots" / ("0" * 40) / "classifier_preprocessor.json").write_text("{}")

    assert RewardClassifierProcessorStep._has_saved_preprocessor(HUB_REPO_ID) is True


def test_reward_classifier_step_offline_with_unknown_preprocessor_raises(offline_hub_cache):
    """Offline, a preprocessor that was never looked up may well exist: do not run without it."""
    step = RewardClassifierProcessorStep()

    with pytest.raises(LocalEntryNotFoundError) as exc_info:
        step._load_reward_preprocessor(HUB_REPO_ID)
    message = str(exc_info.value)
    assert f"Cannot tell whether the reward classifier '{HUB_REPO_ID}' has a saved preprocessor" in message
    assert f"`hf download {HUB_REPO_ID} classifier_preprocessor.json`" in message
