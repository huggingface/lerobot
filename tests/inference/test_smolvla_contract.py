"""SmolVLA's real camera preparation contract, without downloading model weights."""

from dataclasses import replace
from pathlib import Path

import draccus
import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.inference.contracts import ObservationSnapshot
from lerobot.inference.policy_runner import PolicyRunner
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.remote_inference.configs import ServerConfig


@pytest.fixture
def setup():
    path = Path(__file__).resolve().parents[2] / "examples/remote_inference/omx_smolvla_local.yaml"
    server = draccus.parse(ServerConfig, config_path=path, args=[])
    config = SmolVLAConfig(
        device="cpu",
        empty_cameras=1,
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            **{
                f"observation.images.camera{i}": PolicyFeature(FeatureType.VISUAL, (3, 256, 256))
                for i in range(1, 4)
            },
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
    )
    config.validate_features()
    # Exercise real validation and image preparation without instantiating a VLM.
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    PreTrainedPolicy.__init__(policy, config)
    return server, policy


def test_two_camera_deployment_preserves_local_smolvla_images_and_masks(setup):
    server, policy = setup
    runner = PolicyRunner(
        policy,
        PolicyProcessorPipeline(steps=[]),
        PolicyProcessorPipeline(steps=[]),
        action_interval=1 / 30,
        features=tuple(server.features),
        action_feature=server.action_feature,
    )
    arrays = {
        feature.name: np.full(feature.shape, 64 + i, dtype=feature.dtype)
        for i, feature in enumerate(server.features)
    }
    remote_batch = runner._batch(ObservationSnapshot(arrays, 0.0, "pick up the cube"))
    local_batch = {
        feature.name: torch.from_numpy(arrays[feature.name]).permute(2, 0, 1).float().div(255)[None]
        for feature in server.features
        if feature.kind == "rgb"
    }
    actual_images, actual_masks = policy.prepare_images(remote_batch)
    expected_images, expected_masks = policy.prepare_images(local_batch)
    assert len(actual_images) == 3  # Two real cameras, one masked empty camera.
    assert [mask.item() for mask in actual_masks] == [True, True, False]
    for actual, expected in zip(actual_images + actual_masks, expected_images + expected_masks, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_images[0].shape == (1, 3, 512, 512)
    # The default remains strict for policies without this declared exception.
    with pytest.raises(ValueError, match="exactly match"):
        PreTrainedPolicy.validate_chunk_input_features(policy, tuple(server.features))


def test_smolvla_rejects_missing_state_unknown_cameras_and_unresized_shapes(setup):
    server, policy = setup
    features = tuple(server.features)
    with pytest.raises(ValueError, match="nonvisual"):
        policy.validate_chunk_input_features(features[1:])
    with pytest.raises(ValueError, match="at least one camera"):
        policy.validate_chunk_input_features(features[:1])
    with pytest.raises(ValueError, match="known camera"):
        policy.validate_chunk_input_features((*features, replace(features[1], name="unknown")))
    policy.config.resize_imgs_with_padding = None
    with pytest.raises(ValueError, match="without policy resizing"):
        policy.validate_chunk_input_features(features)
