import torch

from lerobot.policies.cig_vla.flow_controller import FlowMatchingController
from lerobot.policies.cig_vla.interaction_bottleneck import InteractionGeometryBottleneck


def _bottleneck():
    return InteractionGeometryBottleneck(
        translation_goal=torch.tensor([[0.1, 0.2, 0.3]]),
        approach_direction=torch.tensor([[1.0, 0.0, 0.0]]),
        translation_magnitude=torch.tensor([[0.1]]),
        rotation_goal=torch.zeros(1, 3),
        gripper_transition=torch.zeros(1, 1),
        confidence_logit=torch.zeros(1, 1),
        valid_mask=torch.ones(1, 1, dtype=torch.bool),
    )


def test_controller_condition_contract_and_field_order():
    bottleneck = _bottleneck()
    tensor = bottleneck.as_controller_tensor()
    assert tensor.shape == (1, 13)
    torch.testing.assert_close(tensor[:, :3], bottleneck.translation_goal)
    torch.testing.assert_close(tensor[:, 3:6], bottleneck.approach_direction)
    torch.testing.assert_close(tensor[:, 6:7], bottleneck.translation_magnitude)
    torch.testing.assert_close(tensor[:, 7:10], bottleneck.rotation_goal)
    torch.testing.assert_close(tensor[:, 10:11], bottleneck.gripper_transition)
    controller = FlowMatchingController(8, 7, hidden_dim=16, num_layers=1, num_heads=4)
    assert controller.condition.in_features == 13 + 8


def test_translation_interventions_cover_signed_xyz_without_mutation():
    original = _bottleneck()
    original_goal = original.translation_goal.clone()
    for axis in range(3):
        for sign in (-1.0, 1.0):
            offset = torch.zeros_like(original.translation_goal)
            offset[:, axis] = sign * 0.01
            changed = original.with_translation_offset(offset)
            torch.testing.assert_close(changed.translation_goal, original_goal + offset)
            assert changed is not original
    torch.testing.assert_close(original.translation_goal, original_goal)
