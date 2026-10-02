"""LaWAM serving contracts without downloading the model or tokenizer."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.inference.contracts import ExecutionMode, ObservationSnapshot
from lerobot.inference.policy_runner import PolicyRunner
from lerobot.policies.lawam.configuration_lawam import LaWAMConfig
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy
from lerobot.policies.lawam.processor_lawam import LaWAMResizeImagesProcessorStep
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import make_policy_processor_pipelines
from lerobot.utils.constants import OBS_STATE
from tests.inference.fixtures import omx_contract


class FixedBackend(nn.Module):
    def predict_action(self, batch):
        return torch.arange(50 * 32, dtype=torch.float32).reshape(1, 50, 32)


@pytest.fixture
def setup():
    server = omx_contract((("image", "rgb-front-v1"), ("image2", "rgb-wrist-v1")))
    config = LaWAMConfig(
        device="cpu",
        chunk_size=50,
        action_horizon=24,
        n_action_steps=24,
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        primary_image_features=["observation.images.image"],
        wrist_image_features=["observation.images.image2"],
        flow_use_state=False,
    )
    policy = LaWAMPolicy.__new__(LaWAMPolicy)
    PreTrainedPolicy.__init__(policy, config)
    policy.model = FixedBackend()
    policy.reset()
    pre, post = make_policy_processor_pipelines(
        input_steps=[
            LaWAMResizeImagesProcessorStep(image_features=list(config.image_features), image_hw=(256, 256))
        ],
        output_steps=[],
    )
    runner = PolicyRunner(
        policy,
        pre,
        post,
        action_interval=1 / 30,
        features=tuple(server.features),
        action_feature=server.action_feature,
    )
    return server, policy, runner


def test_lawam_serves_cropped_horizon_from_current_images(setup):
    server, policy, runner = setup
    assert policy.config.image_observation_delta_indices == [0, 23]  # Training teacher targets.
    assert runner.capabilities.prediction_steps == 24
    assert runner.capabilities.execution_steps == 24
    assert runner.capabilities.modes == (ExecutionMode.CHUNK,)
    arrays = {
        feature.name: np.full(feature.shape, 64 + i, dtype=feature.dtype)
        for i, feature in enumerate(server.features)
    }
    observation = ObservationSnapshot(arrays, 0.0, "pick all the cubes")
    remote = runner.preprocessor(runner._batch(observation))
    local = runner.preprocessor(
        prepare_observation_for_inference(arrays.copy(), torch.device("cpu"), observation.task)
    )
    for name in policy.config.image_features:
        assert remote[name].shape == (1, 3, 256, 256)
        torch.testing.assert_close(remote[name], local[name], rtol=0, atol=0)
    result = runner.predict(observation)
    expected = torch.stack([policy.select_action(local) for _ in range(24)])[:, 0]
    torch.testing.assert_close(result.canonical_actions, expected, rtol=0, atol=0)
    assert result.canonical_actions.shape == (24, 6)


def test_lawam_rejects_missing_unknown_or_nonrgb_cameras(setup):
    server, policy, _ = setup
    features = tuple(server.features)
    with pytest.raises(ValueError, match="all configured inputs"):
        policy.validate_chunk_input_features(features[:-1])
    with pytest.raises(ValueError, match="known features"):
        policy.validate_chunk_input_features((*features, replace(features[-1], name="unknown")))
    with pytest.raises(ValueError, match="RGB"):
        policy.validate_chunk_input_features((*features[:-1], replace(features[-1], kind="tensor")))


def test_lawam_state_conditioning_requires_exact_state_schema(setup):
    server, policy, _ = setup
    policy.config.flow_use_state = True
    with pytest.raises(ValueError, match="checkpoint state feature"):
        policy.validate_chunk_input_features(tuple(server.features))
    policy.config.input_features[OBS_STATE] = PolicyFeature(FeatureType.STATE, (7,))
    with pytest.raises(ValueError, match="Tensor feature shape differs"):
        policy.validate_chunk_input_features(tuple(server.features))
    policy.config.input_features[OBS_STATE] = PolicyFeature(FeatureType.STATE, (6,))
    policy.validate_chunk_input_features(tuple(server.features))


def test_lawam_rejects_temporal_observation_and_rtc(setup):
    server, policy, runner = setup
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        PolicyRunner(
            policy,
            runner.preprocessor,
            runner.postprocessor,
            action_interval=1 / 30,
            features=tuple(server.features),
            action_feature=server.action_feature,
            modes=(ExecutionMode.RTC_GUIDED,),
        )
    policy.config.n_obs_steps = 2
    with pytest.raises(ValueError, match="n_obs_steps=1"):
        policy.chunk_inference_spec()
