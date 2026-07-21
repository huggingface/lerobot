import math

import pytest
import torch

from lerobot.processor.relative_action_processor import (
    to_absolute_actions,
    to_absolute_se3_pose,
    to_relative_actions,
    to_relative_se3_pose,
)

POSE_GROUP = [list(range(6))]


def test_se3_translation_is_expressed_in_reference_frame():
    reference = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2]])
    target = torch.tensor([[1.0, 3.0, 3.0, 0.0, 0.0, math.pi / 2]])

    relative = to_relative_se3_pose(target, reference)

    torch.testing.assert_close(relative, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]), atol=1e-6, rtol=1e-6)


def test_se3_pose_roundtrip_for_batched_chunks():
    torch.manual_seed(0)
    reference = torch.randn(4, 6)
    reference[:, 3:] *= 0.8
    target = torch.randn(4, 11, 6)
    target[..., 3:] *= 0.8

    relative = to_relative_se3_pose(target, reference.unsqueeze(1))
    recovered = to_absolute_se3_pose(relative, reference.unsqueeze(1))

    torch.testing.assert_close(recovered, target, atol=2e-5, rtol=2e-5)


def test_mixed_se3_pose_and_absolute_gripper_roundtrip():
    reference = torch.tensor([[0.2, -0.1, 0.4, 0.1, 0.2, -0.3, 0.06]])
    target = torch.tensor([[[0.3, 0.2, 0.5, -0.2, 0.1, 0.4, 0.03], [0.1, -0.3, 0.2, 0.5, -0.1, 0.2, 0.05]]])
    mask = [True, True, True, True, True, True, False]

    relative = to_relative_actions(
        target,
        reference,
        mask,
        pose_representation="se3",
        se3_pose_groups=POSE_GROUP,
    )
    recovered = to_absolute_actions(
        relative,
        reference,
        mask,
        pose_representation="se3",
        se3_pose_groups=POSE_GROUP,
    )

    torch.testing.assert_close(relative[..., 6], target[..., 6])
    torch.testing.assert_close(recovered, target, atol=2e-5, rtol=2e-5)


def test_se3_pose_group_cannot_be_partially_relative():
    with pytest.raises(ValueError, match="wholly relative or wholly absolute"):
        to_relative_actions(
            torch.zeros(1, 7),
            torch.zeros(1, 7),
            [True, True, True, False, False, False, False],
            pose_representation="se3",
            se3_pose_groups=POSE_GROUP,
        )
