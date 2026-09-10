"""PI052 text generation uses main's shared runtime, not the removed adapter CLI."""

import json
from collections import deque
from dataclasses import fields
from types import SimpleNamespace

import draccus
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.modeling_pi052 import PI052Policy, _last_valid_prefix_hidden


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


def test_generation_reads_last_valid_prompt_token_not_right_padding():
    hidden = torch.arange(12.0).reshape(2, 6, 1)
    mask = torch.tensor([[True, False, True, True, False, False], [False, True, True, True, True, False]])
    assert _last_valid_prefix_hidden(hidden, mask).flatten().tolist() == [3.0, 10.0]


def test_obsolete_runtime_options_are_not_policy_fields():
    names = {field.name for field in fields(PI052Config)}
    assert not names & {"subtask_replan_steps", "joint_subtask_conditioning", "apply_chat_template"}


def test_tagged_config_roundtrip_uses_policy_local_decoder():
    config = PI052Config(device="cpu")
    encoded = draccus.encode(config, PreTrainedConfig)
    assert draccus.decode(PI052Config, encoded).recipe == config.recipe
    assert encoded["type"] == "pi052"


def test_legacy_checkpoint_defaults_load_without_rewriting_source(tmp_path, caplog):
    config = PI052Config(device="cpu", enable_fast_action_loss=False)
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw.update(subtask_replan_steps=90, joint_subtask_conditioning=False, apply_chat_template=False)
    path.write_text(json.dumps(raw))
    original = path.read_bytes()
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert restored.recipe == config.recipe
    assert "subtask_replan_steps" in caplog.text
    assert path.read_bytes() == original
    assert "subtask_replan_steps" not in draccus.encode(restored)


def test_legacy_joint_prompt_request_is_not_silently_ignored():
    with pytest.raises(ValueError, match="legacy joint-subtask prompt layout"):
        draccus.decode(PI052Config, {"device": "cpu", "joint_subtask_conditioning": True})


def test_single_action_calls_do_not_generate_or_rewrite_runtime_subtask():
    policy = _policy()
    policy.config = PI052Config(device="cpu", n_action_steps=2)
    policy._action_queue = deque()
    batch = {"observation.state": torch.zeros(1, 14)}
    seen = []
    policy.select_message = lambda *args, **kwargs: pytest.fail("Only the runtime may request text")
    policy.predict_action_chunk = lambda value: seen.append(value) or torch.ones(1, 2, 14)
    assert policy.select_action(batch).shape == (1, 14)
    assert policy.select_action(batch).shape == (1, 14)
    assert len(seen) == 1
    assert seen[0] is batch
