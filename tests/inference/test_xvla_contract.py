"""XVLA input compatibility and canonical preparation without model downloads."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference.contracts import ExecutionMode, ObservationSnapshot
from lerobot.inference.policy_runner import PolicyRunner
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
)
from lerobot.processor import NormalizerProcessorStep, make_policy_processor_pipelines
from lerobot.remote_inference.configs import ServerConfig
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_STATE


@pytest.fixture
def setup():
    path = Path(__file__).resolve().parents[2] / "examples/remote_inference/omx_xvla_local.yaml"
    server = draccus.parse(ServerConfig, config_path=path, args=[])
    config = XVLAConfig(
        device="cpu",
        chunk_size=30,
        n_action_steps=30,
        action_mode="auto",
        max_state_dim=20,
        num_image_views=3,
        resize_imgs_with_padding=(224, 224),
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image3": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    config.validate_features()
    policy = XVLAPolicy.__new__(XVLAPolicy)
    PreTrainedPolicy.__init__(policy, config)
    # The preparation path only needs the model's declared state width.
    policy.model = SimpleNamespace(dim_proprio=config.max_state_dim)
    pre, post = make_policy_processor_pipelines(
        input_steps=[
            XVLAAddDomainIdProcessorStep(domain_id=0),
            XVLAImageToFloatProcessorStep(),
            XVLAImageNetNormalizeProcessorStep(),
            NormalizerProcessorStep(features=config.input_features, norm_map=config.normalization_mapping),
        ],
        output_steps=[],
    )
    return server, policy, pre, post


def test_xvla_remote_preserves_local_preparation_for_six_joints_and_two_cameras(setup):
    server, policy, pre, post = setup
    runner = PolicyRunner(
        policy,
        pre,
        post,
        action_interval=1 / 30,
        features=tuple(server.features),
        action_feature=server.action_feature,
    )
    assert runner.capabilities.execution_steps == 30
    assert runner.capabilities.modes == (ExecutionMode.CHUNK,)
    arrays = {
        feature.name: np.full(feature.shape, 64 + i, dtype=feature.dtype)
        for i, feature in enumerate(server.features)
    }
    arrays[OBS_STATE] = np.arange(6, dtype=np.float32)
    observation = ObservationSnapshot(arrays, 0.0, "pick up the cube")
    remote = pre(runner._batch(observation))
    local = pre(prepare_observation_for_inference(arrays.copy(), torch.device("cpu"), observation.task))
    # Tokenization is independent of transport; avoid downloading its vocabulary.
    local[OBS_LANGUAGE_TOKENS] = remote[OBS_LANGUAGE_TOKENS] = torch.ones((1, 4), dtype=torch.long)
    actual = policy._build_model_inputs(remote)
    expected = policy._build_model_inputs(local)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["image_input"].shape == (1, 3, 3, 224, 224)
    assert actual["image_mask"].tolist() == [[True, True, False]]
    assert actual["domain_id"].tolist() == [0]
    assert actual["proprio"].shape == (1, 20)
    torch.testing.assert_close(actual["proprio"][0, :6], torch.arange(6, dtype=torch.float32))
    assert not actual["proprio"][0, 6:].any()


def test_xvla_rejects_state_truncation_and_nonidentity_shape_changes(setup):
    server, policy, _, _ = setup
    features = tuple(server.features)
    with pytest.raises(ValueError, match="truncate"):
        policy.validate_chunk_input_features((replace(features[0], shape=(21,), names=()), *features[1:]))
    policy.config.normalization_mapping["STATE"] = NormalizationMode.MEAN_STD
    with pytest.raises(ValueError, match="identity normalization"):
        policy.validate_chunk_input_features(features)


def test_xvla_rejects_unknown_or_missing_inputs_and_unsupported_rtc(setup):
    server, policy, pre, post = setup
    features = tuple(server.features)
    with pytest.raises(ValueError, match="nonvisual"):
        policy.validate_chunk_input_features(features[1:])
    with pytest.raises(ValueError, match="at least one camera"):
        policy.validate_chunk_input_features(features[:1])
    with pytest.raises(ValueError, match="known camera"):
        policy.validate_chunk_input_features((*features, replace(features[1], name="unknown")))
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        PolicyRunner(
            policy,
            pre,
            post,
            action_interval=1 / 30,
            features=features,
            action_feature=server.action_feature,
            modes=(ExecutionMode.RTC_GUIDED,),
        )
    policy.config.resize_imgs_with_padding = None
    with pytest.raises(ValueError, match="without policy resizing"):
        policy.validate_chunk_input_features(features)
