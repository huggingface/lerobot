"""SmolVLA's real camera preparation contract, without downloading model weights."""

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tests.inference.fixtures import omx_contract, preparation_batches


@pytest.fixture
def setup():
    server = omx_contract((("camera1", "rgb-wrist-v1"), ("camera2", "rgb-front-v1")))
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
    _, _, remote_batch, local_batch = preparation_batches(policy, server)
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
