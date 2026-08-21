import torch
from conftest import make_batch, make_policy


def test_full_policy_forward_backward_and_missing_labels():
    policy = make_policy(detach_main=False, detach_causal=False)
    loss, metrics = policy(make_batch())
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert {
        "loss",
        "action_loss",
        "geometry_loss",
        "translation_goal_loss",
        "gripper_transition_loss",
    } <= metrics.keys()
    assert all(torch.isfinite(value).all() for value in metrics.values())
    loss.backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in policy.controller.parameters())
    loss_without_object_labels, metrics = policy(make_batch(labels=False))
    assert torch.isfinite(loss_without_object_labels)
    assert metrics["trajectory_geometry_valid_count"] > 0


def test_future_action_changes_target_but_not_stage_a_prediction():
    policy = make_policy().eval()
    first = make_batch()
    second = {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value) for key, value in first.items()
    }
    second["action"] = first["action"] + 3.0
    with torch.no_grad():
        predicted_first = policy._predict(first)
        predicted_second = policy._predict(second)
    for name in predicted_first.__dataclass_fields__:
        torch.testing.assert_close(getattr(predicted_first, name), getattr(predicted_second, name))
    target_first = policy.target_builder.build(
        first["action"], first["observation.state"], policy.dataset_stats, first["action_is_pad"]
    )
    target_second = policy.target_builder.build(
        second["action"], second["observation.state"], policy.dataset_stats, second["action_is_pad"]
    )
    assert not torch.equal(target_first.translation_goal, target_second.translation_goal)
