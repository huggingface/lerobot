"""PI052 text generation uses main's shared runtime, not the removed adapter CLI."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi052.modeling_pi052 import PI052Policy


def _policy(**kwargs):
    policy = PI052Policy.__new__(PI052Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(text_loss_weight=1.0, memory_scratchpad=False, **kwargs)
    return policy


def test_shared_generation_uses_the_preprocessed_batch_without_reprompting():
    policy = _policy()
    batch = {"observation.state": torch.zeros(1, 14)}
    seen = []
    policy.select_message = lambda value: seen.append(value) or "pick plate"
    assert policy.supports_text_generation()
    assert policy.generate_text(batch) == "pick plate"
    assert seen == [batch]


def test_action_only_checkpoint_does_not_advertise_a_trained_text_head():
    policy = _policy()
    policy.config.text_loss_weight = 0
    assert not policy.supports_text_generation()


def test_text_queries_reject_multi_observation_batches():
    policy = _policy()
    with pytest.raises(ValueError, match="one observation"):
        policy.generate_text({"observation.state": torch.zeros(2, 14)})


def test_scratchpad_response_cannot_be_applied_as_an_ordinary_subtask():
    policy = _policy()
    policy.config.memory_scratchpad = True
    with pytest.raises(ValueError, match="scratchpad-aware controller"):
        policy.generate_text({"observation.state": torch.zeros(1, 14)})
