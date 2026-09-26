"""FineARTVLA text generation uses main's shared runtime, not the removed adapter CLI."""

import json
from collections import deque
from dataclasses import dataclass, fields
from types import SimpleNamespace

import draccus
import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy, _last_valid_prefix_hidden


def _policy(**kwargs):
    policy = FineARTVLAPolicy.__new__(FineARTVLAPolicy)
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
    names = {field.name for field in fields(FineARTVLAConfig)}
    assert not names & {"subtask_replan_steps", "joint_subtask_conditioning", "apply_chat_template"}


def test_tagged_config_roundtrip_uses_standard_choice_decoder():
    config = FineARTVLAConfig(device="cpu")
    encoded = draccus.encode(config, PreTrainedConfig)
    assert draccus.decode(PreTrainedConfig, dict(encoded)).recipe == config.recipe
    assert encoded["type"] == "fineart_vla"


def test_legacy_checkpoint_defaults_load_without_rewriting_source(tmp_path, caplog):
    config = FineARTVLAConfig(device="cpu", enable_fast_action_loss=False)
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


def test_legacy_joint_prompt_request_is_not_silently_ignored(tmp_path):
    config = FineARTVLAConfig(device="cpu")
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["joint_subtask_conditioning"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="legacy joint-subtask prompt layout"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_fineart_vla_can_be_used_in_the_training_cli_policy_choice():
    from draccus.argparsing import ArgumentParser

    @dataclass
    class Pipeline:
        policy: PreTrainedConfig | None = None

    parser = ArgumentParser(config_class=Pipeline)
    config = parser.parse_args(["--policy.type=fineart_vla", "--policy.device=cpu"])
    assert isinstance(config.policy, FineARTVLAConfig)


def test_single_action_calls_do_not_generate_or_rewrite_runtime_subtask():
    policy = _policy()
    policy.config = FineARTVLAConfig(device="cpu", n_action_steps=2)
    policy._action_queue = deque()
    batch = {"observation.state": torch.zeros(1, 14)}
    seen = []
    policy.select_message = lambda *args, **kwargs: pytest.fail("Only the runtime may request text")
    policy.predict_action_chunk = lambda value: seen.append(value) or torch.ones(1, 2, 14)
    assert policy.select_action(batch).shape == (1, 14)
    assert policy.select_action(batch).shape == (1, 14)
    assert len(seen) == 1
    assert seen[0] is batch


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_shared_action_sampler_forwards_optional_state_prefix(with_history, causal, monkeypatch):
    """Exercise the real parent sampler and both prefix overrides, without model weights."""
    from torch import nn

    from lerobot.policies.fineart_vla.modeling_fineart_vla import PI05Pytorch
    from lerobot.policies.pi05 import modeling_pi05

    integrate = modeling_pi05.euler_integrate
    integration_options = {}

    def checked_integrate(*args, **kwargs):
        integration_options.update(kwargs)
        return integrate(*args, **kwargs)

    monkeypatch.setattr(modeling_pi05, "euler_integrate", checked_integrate)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma = SimpleNamespace(
                model=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))
            )
            self.seen = None

        def embed_image(self, image):
            return torch.ones(image.shape[0], 2, 4)

        def embed_language_tokens(self, tokens):
            return tokens[..., None].float().expand(-1, -1, 4)

        def forward(self, **kwargs):
            self.seen = kwargs
            return (None, None), None

    core = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(num_inference_steps=1, rtc_config=None)
    core.rtc_processor = None
    core.gradient_checkpointing_enabled = False
    core.paligemma_with_expert = Backbone()
    core.proprio_history_proj = nn.Linear(2, 4) if with_history else None
    core.denoise_step = lambda **kwargs: torch.zeros_like(kwargs["x_t"])
    states = torch.ones(1, 2, 2) if with_history else None
    state_masks = torch.tensor([[True, False]]) if with_history else None
    tokens = torch.tensor([[1, 2, 3]])
    marks = torch.tensor([[False, True, True]]) if causal else None
    noise = torch.ones(1, 2, 14)

    actual = core.sample_actions(
        [torch.zeros(1, 3, 2, 2)],
        [torch.tensor([True])],
        tokens,
        torch.ones_like(tokens, dtype=torch.bool),
        states=states,
        state_masks=state_masks,
        noise=noise,
        lang_causal_marks=marks,
    )
    torch.testing.assert_close(actual, noise)
    assert integration_options["precompute_times"] is True
    seen = core.paligemma_with_expert.seen
    prefix = seen["inputs_embeds"][0]
    assert prefix.shape == (1, 7 if with_history else 5, 4)
    if with_history:
        torch.testing.assert_close(prefix[:, 2:4], core.proprio_history_proj(states))
        assert seen["attention_mask"][0, 0, -1, 3] < -1e10
    assert bool(seen["attention_mask"][0, 0, 0, -1] < -1e10) == causal
    assert core._lang_causal_marks is None
