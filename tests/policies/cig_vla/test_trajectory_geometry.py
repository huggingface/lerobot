import torch

from lerobot.policies.cig_vla.trajectory_geometry import TrajectoryGeometryTargetBuilder


def stats():
    return {"actions": {"mean": torch.zeros(7), "std": torch.tensor([2.0, 1.0, 0.5, 1, 1, 1, 1])}}


def test_physical_translation_padding_direction_and_gripper_transition():
    actions = torch.zeros(1, 4, 7)
    actions[0, :, 0] = 1
    actions[0, 0, 6], actions[0, 1, 6] = -1, 1
    padding = torch.tensor([[False, False, True, True]])
    target = TrajectoryGeometryTargetBuilder().build(actions, torch.zeros(1, 8), stats(), padding)
    torch.testing.assert_close(target.translation_goal, torch.tensor([[4.0, 0.0, 0.0]]))
    torch.testing.assert_close(target.approach_direction, torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(target.translation_magnitude, torch.tensor([[4.0]]))
    torch.testing.assert_close(target.gripper_transition, torch.tensor([[2.0]]))
    assert target.valid_mask.item()
    assert target.rotation_goal is None


def test_missing_stats_disables_metric_geometry_supervision():
    actions = torch.ones(2, 3, 7)
    target = TrajectoryGeometryTargetBuilder().build(actions, torch.zeros(2, 8), None, None)
    assert not target.valid_mask.any()
    assert torch.isfinite(target.translation_goal).all()


def test_all_padding_is_finite_and_invalid():
    target = TrajectoryGeometryTargetBuilder().build(
        torch.randn(1, 3, 7), torch.zeros(1, 8), stats(), torch.ones(1, 3, dtype=torch.bool)
    )
    assert not target.valid_mask.item()
    assert torch.isfinite(target.approach_direction).all()


def test_gripper_only_transition_is_meaningful_and_rotation_is_disabled():
    actions = torch.zeros(1, 3, 7)
    actions[0, 0, 6], actions[0, 2, 6] = -1, 1
    target = TrajectoryGeometryTargetBuilder().build(
        actions, torch.zeros(1, 8), stats(), torch.zeros(1, 3, dtype=torch.bool)
    )
    assert target.valid_mask.item()
    torch.testing.assert_close(target.gripper_transition, torch.tensor([[2.0]]))
    assert target.rotation_goal is None
