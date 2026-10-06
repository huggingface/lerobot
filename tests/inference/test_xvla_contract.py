"""XVLA input compatibility and canonical preparation without model downloads."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference import ExecutionMode
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
)
from lerobot.processor import NormalizerProcessorStep, make_policy_processor_pipelines
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.inference.fixtures import omx_contract, preparation_batches


@pytest.fixture
def setup():
    server = omx_contract((("image", "rgb-front-v1"), ("image2", "rgb-wrist-v1")))
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
    runner, _, remote, local = preparation_batches(policy, server, pre, post)
    assert runner.capabilities.execution_steps == 30
    assert runner.capabilities.modes == (ExecutionMode.CHUNK,)
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
