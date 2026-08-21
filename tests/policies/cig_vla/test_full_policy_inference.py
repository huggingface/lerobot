import copy

import torch
from conftest import make_batch, make_policy


def test_inference_override_queue_reset_and_no_label_leakage():
    policy = make_policy()
    policy.eval()
    batch = make_batch(labels=False)
    batch = {key: value[:1] if isinstance(value, torch.Tensor) else value[:1] for key, value in batch.items()}
    bottleneck = policy.predict_geometric_bottleneck(batch)
    torch.manual_seed(7)
    chunk = policy.predict_action_chunk(batch)
    assert bottleneck.translation_goal.shape == (1, 3)
    assert chunk.shape == (1, 4, 7)
    policy.reset()
    torch.manual_seed(9)
    first, second = policy.select_action(batch), policy.select_action(batch)
    policy.reset()
    torch.manual_seed(9)
    expected = policy.predict_action_chunk(batch)
    torch.testing.assert_close(first, expected[:, 0])
    torch.testing.assert_close(second, expected[:, 1])
    policy.reset()
    assert not policy._action_queue

    override = bottleneck.with_translation_offset(torch.ones_like(bottleneck.translation_goal))
    torch.manual_seed(11)
    original = policy.predict_action_chunk(batch)
    torch.manual_seed(11)
    changed = policy.predict_action_chunk(batch, override)
    assert not torch.allclose(original, changed)

    labelled_a, labelled_b = copy.deepcopy(batch), copy.deepcopy(batch)
    labelled_a["unused.object_pose"] = torch.zeros(1, 3)
    labelled_b["unused.object_pose"] = torch.full((1, 3), 999.0)
    torch.manual_seed(13)
    action_a = policy.predict_action_chunk(labelled_a)
    torch.manual_seed(13)
    action_b = policy.predict_action_chunk(labelled_b)
    torch.testing.assert_close(action_a, action_b)
