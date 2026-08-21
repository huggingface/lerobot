import torch

from lerobot.policies.cig_vla.action_semantics import LiberoSafetyDeltaOSCActionSemantics


def test_libero_osc_contract_and_scale_correction():
    adapter = LiberoSafetyDeltaOSCActionSemantics()
    actions = torch.ones(2, 4, 7)
    assert adapter.cartesian_translation(actions).shape == (2, 4, 3)
    assert adapter.controller == "OSC_POSE" and adapter.control_mode == "delta"
    assert adapter.translation_delta(actions).shape == (2, 4, 3)
    assert adapter.rotation_delta(actions).shape == (2, 4, 3)
    assert adapter.gripper_command(actions).shape == (2, 4)
    assert adapter.motion_action(actions).shape[-1] == 6
    torch.testing.assert_close(adapter.aggregate_translation(actions, 2), torch.full((2, 3), 2.0))
    stats = {"action": {"mean": torch.zeros(7), "std": torch.arange(1, 8)}}
    physical = adapter.denormalize_actions(actions, stats)
    torch.testing.assert_close(physical[0, 0], torch.arange(1, 8, dtype=torch.float32))
    assert adapter.safe_action(torch.zeros(1, 3), 7, torch.device("cpu"), torch.float32) is None
