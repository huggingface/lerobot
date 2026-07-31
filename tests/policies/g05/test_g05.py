# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

import json

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.g05.action_tokenizer import (
    G05ActionCodecConfig,
    G05ActionCodecModel,
    G05ActionTokenizer,
)
from lerobot.policies.g05.action_tokenizer.modular_actioncodec import (
    ActionTimeContrastiveLoss,
    time_shift_positive,
)
from lerobot.policies.g05.configuration_g05 import G05Config
from lerobot.policies.g05.convert_g05_checkpoint import (
    _action_head_flags,
    _action_tokenizer_source,
    _action_tokens,
    _camera_layout,
    _canonical_shape_meta,
    _cot_prompt,
    _exported_action_tokens,
    _joint_frame_transform,
    _normalization_config,
    _normalization_specs,
    _processor_contract,
    _processor_source,
)
from lerobot.policies.g05.modeling_g05 import G05Policy
from lerobot.policies.g05.processor_g05 import (
    G05ActionFrameTransformStep,
    G05LiberoActionStep,
    G05LiberoObservationStep,
    G05PrepareInputsStep,
    G05StateFrameTransformStep,
    G05StepwiseActionUnnormalizerStep,
    G05StepwiseNormalizerStep,
    _apply_normalization,
    make_g05_pre_post_processors,
    make_g05_pre_post_processors_from_pretrained,
)
from lerobot.processor import (
    LiberoProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
)
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def _features() -> tuple[dict, dict]:
    return (
        {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def _tiny_config() -> G05Config:
    input_features, output_features = _features()
    return G05Config(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        camera_keys=["observation.images.cam"],
        image_size=(32, 32),
        internal_action_dim=4,
        internal_state_dim=4,
        action_indices=[1, 3],
        state_indices=[0, 1, 3],
        vocab_size=100,
        image_token_id=2,
        state_token_id=3,
        eov_token_id=4,
        eos_token_id=5,
        text_hidden_size=32,
        text_intermediate_size=64,
        text_num_layers=4,
        text_num_heads=2,
        text_num_kv_heads=1,
        text_head_dim=16,
        text_layer_types=["linear_attention"] * 3 + ["full_attention"],
        mrope_section=(2, 1, 1),
        vision_depth=1,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_heads=2,
        expert_hidden_size=32,
        expert_intermediate_size=64,
        expert_num_layers=4,
        expert_num_heads=2,
        expert_num_kv_heads=1,
        expert_head_dim=16,
        chunk_size=2,
        n_action_steps=1,
        num_inference_steps=2,
        dtype="float32",
    )


def test_action_steps_default_to_prediction_horizon() -> None:
    config = _tiny_config()
    config.chunk_size = 7
    config.n_action_steps = None

    config.__post_init__()

    assert config.n_action_steps == config.chunk_size == 7


def test_config_delta_indices_and_validation() -> None:
    config = _tiny_config()
    assert config.observation_delta_indices == [0]
    assert config.action_delta_indices == [0, 1]
    config.validate_features()

    config.n_action_steps = 3
    with pytest.raises(ValueError, match="cannot exceed"):
        config.__post_init__()

    config.n_action_steps = 1
    config.continuous_action = False
    config.discrete_action = False
    with pytest.raises(ValueError, match="discrete_action"):
        config.__post_init__()

    config.continuous_action = True
    config.predict_cot = True
    config.cot_prompt = ""
    with pytest.raises(ValueError, match="cot_prompt|CoT prompt"):
        config.__post_init__()


def test_legacy_action_attend_cot_config_is_ignored(tmp_path) -> None:
    config = _tiny_config()
    config.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    serialized = json.loads(config_path.read_text())
    assert "action_attend_cot" not in serialized

    serialized["action_attend_cot"] = False
    config_path.write_text(json.dumps(serialized))

    reloaded = G05Config.from_pretrained(tmp_path)
    assert reloaded.predict_cot is config.predict_cot
    assert not hasattr(reloaded, "action_attend_cot")


def test_inference_action_heads_are_exclusive(monkeypatch) -> None:
    policy = G05Policy(_tiny_config())
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(1, 1, dtype=torch.long),
        OBS_STATE: torch.zeros(1, 4),
    }

    def fail(*args, **kwargs):
        raise AssertionError("disabled inference head was called")

    monkeypatch.setattr(policy.model, "sample_action_tokens", fail)
    monkeypatch.setattr(
        policy.model,
        "sample_actions",
        lambda batch, noise=None: torch.tensor([[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]]),
    )
    assert policy.predict_action_chunk(batch).tolist() == [[[11.0, 13.0], [21.0, 23.0]]]

    class FakeActionTokenizer:
        def decode(self, rows):
            assert rows.tolist() == [[10, 11]]
            return torch.tensor([[[30.0, 31.0, 32.0, 33.0], [40.0, 41.0, 42.0, 43.0]]])

    policy.config.discrete_action = True
    policy.config.continuous_action = False
    policy._action_tokenizer = FakeActionTokenizer()
    monkeypatch.setattr(policy.model, "sample_actions", fail)
    monkeypatch.setattr(
        policy.model,
        "sample_action_tokens",
        lambda batch: [torch.tensor([10, 11])],
    )
    assert policy.predict_action_chunk(batch).tolist() == [[[31.0, 33.0], [41.0, 43.0]]]


def test_model_forward_backward_and_action_queue() -> None:
    policy = G05Policy(_tiny_config())
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.randn(1, 2, 4),
    }
    loss, logs = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(logs) == {
        "fm_loss",
        "action_token_loss",
        "cot_loss",
        "action_token_accuracy",
        "cot_accuracy",
    }
    assert all(isinstance(value, float) for value in logs.values())
    assert any(parameter.grad is not None for parameter in policy.parameters())
    noise = torch.zeros(1, 2, 4)
    assert policy.predict_action_chunk(batch, noise=noise).shape == (1, 2, 2)
    padded_actions = policy.model.sample_actions(batch, noise=torch.ones_like(noise))
    torch.testing.assert_close(padded_actions[..., [0, 2]], torch.zeros(1, 2, 2))
    assert policy.select_action(batch, noise=noise).shape == (1, 2)
    policy.reset()
    assert not policy._action_queue


def test_sdpa_forward_backward_uses_native_qwen_attention() -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    config = _tiny_config()
    config.attn_implementation = "sdpa"
    policy = G05Policy(config)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.randn(1, 2, 4),
    }

    loss, _ = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert isinstance(policy.model.vlm.layers[-1].self_attn, Qwen3_5Attention)
    assert policy.model.action_expert.layers[0].self_attn.config._attn_implementation == "sdpa"


def test_missing_flash_attention_dependency_fails_clearly(monkeypatch) -> None:
    from transformers import utils as transformers_utils

    monkeypatch.setattr(transformers_utils, "is_flash_attn_4_available", lambda: False)
    config = _tiny_config()
    config.attn_implementation = "flash_attention_4"

    with pytest.raises(ImportError, match="local package is unavailable"):
        G05Policy(config)


def test_predict_cot_inference_runs_before_flow() -> None:
    class FixedNextToken(torch.nn.Module):
        def forward(self, hidden_states):
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[0, 4] = 1  # First row emits EOV immediately.
            logits[1, 6] = 1
            return logits

    config = _tiny_config()
    config.predict_cot = True
    config.cot_prompt = "predict subtask"
    config.max_cot_tokens = 1
    policy = G05Policy(config)
    policy.model.output_proj = FixedNextToken()
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6], [2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        "pixel_values": torch.randn(2, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]]),
    }

    action = policy.predict_action_chunk(batch, noise=torch.zeros(2, 2, 4))

    assert action.shape == (2, 2, 2)


def test_predict_cot_ar_inference_consumes_eov_before_action_tokens() -> None:
    class SequencedNextToken(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            sequence = (4, 4, 7, 5)  # CoT stop, AR transition EOV, action token, EOS.
            token = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[:, token] = 1
            return logits

    config = _tiny_config()
    config.predict_cot = True
    config.cot_prompt = "predict subtask"
    config.discrete_action = True
    config.action_token_start_id = 6
    config.action_token_end_id = 10
    policy = G05Policy(config)
    policy.model.output_proj = SequencedNextToken()
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
    }

    rows = policy.model.sample_action_tokens(batch)

    assert [row.tolist() for row in rows] == [[7]]


def _cot_ar_policy() -> G05Config:
    config = _tiny_config()
    config.predict_cot = True
    config.cot_prompt = "predict subtask"
    config.discrete_action = True
    config.continuous_action = False
    config.action_token_start_id = 6
    config.action_token_end_id = 10
    return config


def test_ar_transition_commits_the_cot_stop_token_instead_of_redecoding() -> None:
    """The AR stage must reuse the CoT stop token, not re-decode the frozen state.

    Under ``ar_do_sample`` a second decode of the same hidden state draws a fresh
    sample, so the transition token would usually not come back as EOV and the row
    would terminate with an empty ActionCodec payload.
    """

    class RedecodeDiverges(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            # Call 1 ends the CoT with EOV. Call 2 is the AR stage's first decode
            # and deliberately returns a token that is neither EOV nor an action
            # token, standing in for a diverging sample.
            sequence = (4, 50, 7, 5)
            token = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[:, token] = 1
            return logits

    policy = G05Policy(_cot_ar_policy())
    policy.model.output_proj = RedecodeDiverges()
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
    }

    rows = policy.model.sample_action_tokens(batch)

    assert [row.tolist() for row in rows] == [[7]]


def test_generate_cot_reports_stop_tokens_and_history() -> None:
    class TwoTokenCot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            sequence = (9, 4)  # One CoT token, then EOV.
            token = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[:, token] = 1
            return logits

    policy = G05Policy(_cot_ar_policy())
    policy.model.output_proj = TwoTokenCot()
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
    }

    _, cot = policy.model.generate_cot(batch, policy.model.prefill(batch))

    assert cot.stop_tokens.tolist() == [policy.config.eov_token_id]
    # The CoT tokens stay visible so repetition penalties span both AR stages.
    assert cot.history.tolist() == [[9]]
    assert cot.history_mask is None


def test_batched_cot_history_excludes_tokens_after_a_row_stops() -> None:
    policy = G05Policy(_cot_ar_policy())
    policy.config.max_cot_tokens = 3
    sequence = [
        torch.tensor([policy.config.eov_token_id, 9]),
        torch.tensor([42, 9]),
        torch.tensor([43, policy.config.eov_token_id]),
    ]
    original_sample = policy.model.sample_next_token
    calls = 0

    def scripted_sample(logits, history=None, history_mask=None):
        nonlocal calls
        token = sequence[calls].to(logits.device)
        calls += 1
        return token

    policy.model.sample_next_token = scripted_sample
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6], [2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        "pixel_values": torch.randn(2, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]]),
    }

    _, cot = policy.model.generate_cot(batch, policy.model.prefill(batch))

    assert cot.history.tolist() == [
        [policy.config.pad_token_id, policy.config.pad_token_id],
        [9, 9],
    ]
    assert cot.history_mask.tolist() == [[False, False], [True, True]]
    assert [row[mask].tolist() for row, mask in zip(cot.history, cot.history_mask, strict=True)] == [
        [],
        [9, 9],
    ]

    policy.model.sample_next_token = original_sample
    policy.config.ar_do_sample = True
    policy.config.ar_temperature = 1.0
    policy.config.ar_top_k = 1
    policy.config.ar_top_p = 1.0
    policy.config.ar_repetition_penalty = 2.0
    policy.config.ar_no_repeat_ngram_size = 0
    logits = torch.zeros(2, policy.config.vocab_size)
    logits[0, policy.config.pad_token_id] = 10
    logits[0, 7] = 9
    logits[1, 9] = 10
    logits[1, 8] = 9

    sampled = policy.model.sample_next_token(logits, cot.history, cot.history_mask)
    assert sampled.tolist() == [policy.config.pad_token_id, 8]


def test_so100_batched_cot_preserves_hardware_reference_history() -> None:
    config = _cot_ar_policy()
    config.embodiment = "so100"
    config.max_cot_tokens = 3
    policy = G05Policy(config)
    sequence = [
        torch.tensor([policy.config.eov_token_id, 9]),
        torch.tensor([42, 9]),
        torch.tensor([43, policy.config.eov_token_id]),
    ]
    calls = 0

    def scripted_sample(logits, history=None, history_mask=None):
        nonlocal calls
        token = sequence[calls].to(logits.device)
        calls += 1
        return token

    policy.model.sample_next_token = scripted_sample
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6], [2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        "pixel_values": torch.randn(2, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0], [1.0, 2.0, 0.0, 3.0]]),
    }

    _, cot = policy.model.generate_cot(batch, policy.model.prefill(batch))

    assert cot.history.tolist() == [
        [policy.config.eov_token_id, 42],
        [9, 9],
    ]
    assert cot.history_mask is None


def test_ar_stage_seeds_repetition_history_from_the_cot_stage() -> None:
    """no_repeat_ngram / repetition_penalty must see the CoT tokens too.

    Both processors operate over the complete CoT and action-token sequence.
    """

    class TwoTokenCot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            sequence = (9, 4, 7, 5)  # CoT token, EOV, action token, EOS.
            token = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[:, token] = 1
            return logits

    policy = G05Policy(_cot_ar_policy())
    policy.model.output_proj = TwoTokenCot()
    seen: list[torch.Tensor | None] = []
    original = policy.model.sample_next_token

    def record(logits, history=None):
        seen.append(None if history is None else history.clone())
        return original(logits, history)

    policy.model.sample_next_token = record
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
    }

    policy.model.sample_action_tokens(batch)

    # Calls: [0] first CoT token (no history), [1] CoT stop decode, then the AR
    # stage, which must already carry the CoT token rather than restarting empty.
    assert seen[0] is None
    assert seen[2] is not None and seen[2].tolist() == [[9]]


def test_ar_inference_without_an_eov_token_does_not_crash() -> None:
    class ActionThenEos(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            sequence = (7, 5)
            token = sequence[min(self.calls, len(sequence) - 1)]
            self.calls += 1
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[:, token] = 1
            return logits

    config = _tiny_config()
    config.discrete_action = True
    config.continuous_action = False
    config.action_token_start_id = 6
    config.action_token_end_id = 10
    config.eov_token_id = None
    policy = G05Policy(config)
    policy.model.output_proj = ActionThenEos()
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
    }

    rows = policy.model.sample_action_tokens(batch)

    assert [row.tolist() for row in rows] == [[7]]


def test_empty_action_token_row_warns_before_zero_chunk(caplog) -> None:
    config = _tiny_config()
    config.discrete_action = True
    config.continuous_action = False
    config.action_token_start_id = 6
    config.action_token_end_id = 10
    policy = G05Policy(config)

    class FakeActionTokenizer:
        def decode(self, rows):
            raise AssertionError("an empty row must not reach the codec")

    policy._action_tokenizer = FakeActionTokenizer()
    policy.model.sample_action_tokens = lambda batch: [torch.empty(0, dtype=torch.long)]
    batch = {
        OBS_LANGUAGE_TOKENS: torch.ones(1, 1, dtype=torch.long),
        OBS_STATE: torch.zeros(1, 4),
    }

    with caplog.at_level("WARNING"):
        actions = policy.predict_action_chunk(batch)

    assert torch.count_nonzero(actions) == 0
    assert "no ActionCodec tokens" in caplog.text


def test_discrete_action_training_loss_backpropagates() -> None:
    config = _tiny_config()
    config.discrete_action = True
    config.action_token_loss_weight = 1.0
    config.action_token_start_id = 6
    config.action_token_end_id = 10
    policy = G05Policy(config)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        ACTION_TOKENS: torch.tensor([[7, 5]]),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.randn(1, 2, 4),
    }

    loss, logs = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert logs["action_token_loss"] > 0


def test_frozen_vlm_training_detaches_only_kv_cache_layers() -> None:
    config = _tiny_config()
    config.flow_joint_training = False
    policy = G05Policy(config)
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.randn(1, 2, 4),
    }

    loss, logs = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert logs["fm_loss"] > 0
    assert all(parameter.grad is None for parameter in policy.model.vlm.parameters())


def test_prepare_inputs_builds_camera_slots_and_prompts(monkeypatch) -> None:
    class FakeTokenizer:
        bos_token = None
        encoded_texts = []

        @classmethod
        def encode(cls, text, add_special_tokens=False):
            del add_special_tokens
            cls.encoded_texts.append(text)
            return [10 + ord(char) % 10 for char in text]

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    step = G05PrepareInputsStep(
        tokenizer_path="artifact/processor",
        action_tokenizer_path="",
        camera_keys=["observation.images.cam", "observation.images.optional"],
        dummy_camera_keys=[],
        image_size=(32, 32),
        patch_size=16,
        spatial_merge_size=2,
        n_obs_steps=1,
        internal_state_dim=4,
        internal_action_dim=4,
        state_indices=[0, 1, 3],
        action_indices=[1, 3],
        embodiment="test",
        max_task_tokens=200,
        max_prompt_length=512,
        image_token_id=2,
        vision_start_token_id=7,
        vision_end_token_id=8,
        state_token_id=3,
        eov_token_id=4,
        pad_token_id=0,
        eos_token_id=5,
        optional_camera_keys=["observation.images.optional"],
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[1.0, 2.0, 3.0]]),
            "observation.images.cam": torch.zeros(1, 3, 24, 24),
        },
        TransitionKey.ACTION: torch.zeros(1, 2, 2),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["test task"]},
    }
    output = step(transition)
    complementary = output[TransitionKey.COMPLEMENTARY_DATA]
    assert "<|im_start|>user\n" not in FakeTokenizer.encoded_texts
    assert ";Action: " in FakeTokenizer.encoded_texts
    assert output[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 4)
    assert output[TransitionKey.ACTION].shape == (1, 2, 4)
    assert complementary["pixel_values"].shape == (1, 2, 1, 3, 32, 32)
    assert complementary[OBS_LANGUAGE_TOKENS].eq(2).sum() == 2

    exterior = "observation.images.exterior_rgb"
    left = "observation.images.left_wrist_rgb"
    right = "observation.images.right_wrist_rgb"
    step.camera_keys = [exterior, left, right]
    step.dummy_camera_keys = []
    step.optional_camera_keys = [left, right]
    step.camera_order = [exterior, left, right]
    state = torch.zeros(1, 3)
    exterior_image = torch.zeros(1, 3, 32, 32)
    optional_image = torch.ones(1, 3, 32, 32)

    for provided_key, provided_index, omitted_index in ((left, 1, 2), (right, 2, 1)):
        images = step._prepare_images(
            {exterior: exterior_image, provided_key: optional_image},
            state,
        )
        assert images[:, provided_index].eq(1).all()
        assert images[:, omitted_index].eq(-1).all()

    exterior_only = step._prepare_images({exterior: exterior_image}, state)
    assert exterior_only[:, 1:].eq(-1).all()

    step.camera_keys = ["observation.images.cam", "observation.images.optional"]
    step.optional_camera_keys = ["observation.images.optional"]
    step.camera_order = None

    ids, mask = step._prompt_ids(["a", "longer task"], num_images=2)
    assert not mask[0, 0]
    assert mask[1, 0]
    assert ids[0, 0] == step.pad_token_id
    assert ids[0, -1] == step.eov_token_id

    step.append_eov = False
    FakeTokenizer.encoded_texts.clear()
    cot_ids, _ = step._prompt_ids(["task"], num_images=2)
    assert step.eov_token_id not in cot_ids
    assert ";Action: " in FakeTokenizer.encoded_texts

    step.predict_cot = True
    step.cot_prompt = "predict bbox, subtask and action"
    FakeTokenizer.encoded_texts.clear()
    cot_ids, _ = step._prompt_ids(["task"], num_images=2)
    assert step.eov_token_id not in cot_ids
    assert ";predict bbox, subtask and action\n" in FakeTokenizer.encoded_texts

    training = step(transition)
    assert training[TransitionKey.COMPLEMENTARY_DATA][OBS_LANGUAGE_TOKENS][0, -1] == step.eov_token_id


def test_so100_joint_frame_transform_is_invertible() -> None:
    signs = [1, -1, 1, 1, 1, 1]
    offsets = [0, 90, 90, 0, 0, 0]
    arm_state = torch.tensor([[3.1, -34.3, 31.5, 55.9, -12.3, 13.4]])
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: arm_state},
        TransitionKey.ACTION: None,
    }

    state_step = G05StateFrameTransformStep(joint_signs=signs, joint_offsets=offsets)
    model_transition = state_step(transition)
    model_state = model_transition[TransitionKey.OBSERVATION][OBS_STATE]
    assert torch.allclose(
        model_state,
        torch.tensor([[3.1, 124.3, 121.5, 55.9, -12.3, 13.4]]),
    )

    action_step = G05ActionFrameTransformStep(joint_signs=signs, joint_offsets=offsets)
    arm_action = action_step({TransitionKey.ACTION: model_state})[TransitionKey.ACTION]
    assert torch.allclose(arm_action, arm_state)

    # Training actions are recorded in the arm frame and need the same forward
    # transform as the state, otherwise the relative-action step differences two
    # different coordinate frames.
    forward_step = G05ActionFrameTransformStep(joint_signs=signs, joint_offsets=offsets, inverse=False)
    model_action = forward_step({TransitionKey.ACTION: arm_state})[TransitionKey.ACTION]
    assert torch.allclose(model_action, model_state)
    round_tripped = action_step({TransitionKey.ACTION: model_action})[TransitionKey.ACTION]
    assert torch.allclose(round_tripped, arm_state)


def test_joint_signs_must_stay_invertible() -> None:
    config = _tiny_config()
    config.joint_signs = [0.5, 1.0]
    config.joint_offsets = [0.0, 0.0]
    with pytest.raises(ValueError, match="invertible"):
        config.__post_init__()

    config.joint_signs = [1.0, -1.0]
    config.__post_init__()


def test_training_pipeline_transforms_state_and_action_into_one_frame(monkeypatch) -> None:
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    config = _tiny_config()
    config.joint_signs = [1.0, -1.0]
    config.joint_offsets = [0.0, 90.0]
    config.__post_init__()

    preprocessor, _ = make_g05_pre_post_processors(
        config,
        {
            OBS_STATE: {"q01": torch.full((3,), -1.0), "q99": torch.full((3,), 1.0)},
            ACTION: {"q01": torch.full((2,), -1.0), "q99": torch.full((2,), 1.0)},
        },
        tokenizer_path="artifact/processor",
    )
    frame_steps = [step for step in preprocessor.steps if isinstance(step, G05ActionFrameTransformStep)]
    assert [step.inverse for step in frame_steps] == [False]


def test_processor_factory_separates_base_and_stepwise_normalization(monkeypatch) -> None:
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    stats = {
        OBS_STATE: {"q01": torch.full((3,), -1.0), "q99": torch.full((3,), 1.0)},
        ACTION: {"q01": torch.full((2,), -1.0), "q99": torch.full((2,), 1.0)},
    }
    base = _tiny_config()
    preprocessor, postprocessor = make_g05_pre_post_processors(
        base, stats, tokenizer_path="artifact/processor"
    )
    assert base.normalization_mapping["STATE"] is NormalizationMode.QUANTILES
    assert any(isinstance(step, NormalizerProcessorStep) for step in preprocessor.steps)
    assert any(isinstance(step, UnnormalizerProcessorStep) for step in postprocessor.steps)

    finetuned = _tiny_config()
    finetuned.normalization_strategy = "g05_stepwise"
    finetuned.state_normalization = [
        {"mode": "q01/q99", "width": 3, "stats": {"q01": [-1.0] * 3, "q99": [1.0] * 3}}
    ]
    finetuned.action_normalization = [
        {
            "mode": "q01/q99",
            "width": 2,
            "stats": {"q01": [[-1.0] * 2] * 2, "q99": [[1.0] * 2] * 2},
        }
    ]
    preprocessor, postprocessor = make_g05_pre_post_processors(finetuned, tokenizer_path="artifact/processor")
    assert any(isinstance(step, G05StepwiseNormalizerStep) for step in preprocessor.steps)
    assert any(isinstance(step, G05StepwiseActionUnnormalizerStep) for step in postprocessor.steps)


def test_libero_policy_steps_transform_features_and_round_trip(monkeypatch, tmp_path) -> None:
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    config = _tiny_config()
    config.embodiment = "libero"
    preprocessor, postprocessor = make_g05_pre_post_processors(
        config,
        tokenizer_path="artifact/processor",
    )

    camera_feature = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
    transformed = G05LiberoObservationStep().transform_features(
        {
            PipelineFeatureType.OBSERVATION: {
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
                "observation.images.cam": camera_feature,
            },
            PipelineFeatureType.ACTION: {},
        }
    )
    assert transformed[PipelineFeatureType.OBSERVATION][OBS_STATE].shape == (7,)
    assert transformed[PipelineFeatureType.OBSERVATION]["observation.images.cam"] is camera_feature

    (tmp_path / config.tokenizer_subdir).mkdir()
    (tmp_path / config.action_tokenizer_subdir).mkdir()
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    reloaded_preprocessor, reloaded_postprocessor = make_pre_post_processors(
        config, pretrained_path=str(tmp_path)
    )

    assert isinstance(reloaded_preprocessor.steps[0], G05LiberoObservationStep)
    assert isinstance(reloaded_postprocessor.steps[-1], G05LiberoActionStep)


def test_pretrained_stepwise_processor_applies_runtime_overrides_and_libero_steps(
    monkeypatch, tmp_path
) -> None:
    config = _tiny_config()
    config.normalization_strategy = "g05_stepwise"
    config.n_action_steps = 1
    config.embodiment = "libero"
    (tmp_path / config.tokenizer_subdir).mkdir()
    (tmp_path / config.action_tokenizer_subdir).mkdir()
    load_calls = []

    def fake_from_pretrained(**kwargs):
        load_calls.append(kwargs)
        return type("Pipeline", (), {"steps": []})()

    monkeypatch.setattr(PolicyProcessorPipeline, "from_pretrained", fake_from_pretrained)

    preprocessor, postprocessor = make_g05_pre_post_processors_from_pretrained(config, str(tmp_path))

    assert load_calls[1]["overrides"] == {
        "g05_stepwise_action_unnormalizer": {"action_horizon": config.n_action_steps}
    }
    assert isinstance(preprocessor.steps[0], G05LiberoObservationStep)
    assert isinstance(postprocessor.steps[-1], G05LiberoActionStep)


@pytest.mark.parametrize("assets_in_parent", [False, True])
def test_converter_discovers_shared_assets(tmp_path, assets_in_parent: bool) -> None:
    input_dir = tmp_path / "g05-libero"
    input_dir.mkdir()
    asset_dir = tmp_path if assets_in_parent else input_dir
    processor_dir = asset_dir / ("qwen3_5_2b_base_processor" if assets_in_parent else "hf_processor")
    processor_dir.mkdir()
    (processor_dir / "tokenizer_config.json").write_text("{}")
    action_tokenizer = asset_dir / "action_tokenizer.pt"
    action_tokenizer.touch()

    assert _processor_source(input_dir) == processor_dir
    assert _action_tokenizer_source(input_dir) == action_tokenizer


def test_conversion_resolves_normalization_from_original_config() -> None:
    normalization = _normalization_config(
        {"use_stepwise_action_norm": True},
        {
            "norm_default_mode": "z-score-tail",
            "norm_exception_mode": {"state": {"gripper": "q01/q99"}},
            "norm_tail_scale": 0.125,
        },
    )

    assert normalization == {
        "default_mode": "z-score-tail",
        "exception_modes": {"state": {"gripper": "q01/q99"}},
        "use_stepwise_action_norm": True,
    }


def test_conversion_resolves_cot_prompt_and_action_head_from_checkpoint() -> None:
    processor = {
        "samples_builder": {
            "_target_": "g05.data_processor.processor.samples_builder.MixedSamplesBuilder",
            "eval_builder": {
                "_target_": "g05.data_processor.processor.samples_builder.BBoxSubtaskCoTBuilder"
            },
        }
    }

    assert _cot_prompt(processor) == "predict bbox, subtask and action"
    # (continuous_action, discrete_action); a dual-head checkpoint collapses to
    # the head named by its own return_continuous_action flag.
    assert _action_head_flags(
        {"continuous_action": False, "discrete_action": True, "return_continuous_action": False}
    ) == (False, True)
    assert _action_head_flags(
        {"continuous_action": True, "discrete_action": True, "return_continuous_action": True}
    ) == (True, False)
    assert _action_head_flags(
        {"continuous_action": True, "discrete_action": True, "return_continuous_action": False}
    ) == (False, True)
    assert _joint_frame_transform("so100", 6, 6) == (
        [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 90.0, 90.0, 0.0, 0.0, 0.0],
    )


def test_conversion_model_normalization_override_is_atomic() -> None:
    normalization = _normalization_config(
        {
            "norm_default_mode": "q01/q99",
            "use_stepwise_action_norm": False,
        },
        {
            "norm_default_mode": "z-score-tail",
            "norm_exception_mode": {"action": {"gripper": "dummy"}},
            "use_stepwise_action_norm": True,
        },
    )

    assert normalization["default_mode"] == "q01/q99"
    assert normalization["exception_modes"] == {}
    assert normalization["use_stepwise_action_norm"] is False


def test_conversion_rejects_missing_normalization_mode() -> None:
    with pytest.raises(ValueError, match="norm_default_mode"):
        _normalization_config({"use_stepwise_action_norm": True}, {})


def test_conversion_keeps_only_zscore_and_q01_stats() -> None:
    stats = {
        "arm": {
            "global_mean": [1.0],
            "global_std": [2.0],
            "global_q01": [-1.0],
            "global_q99": [3.0],
        }
    }
    items = [{"key": "arm", "shape": 1}]
    tail_spec = _normalization_specs(
        stats,
        items,
        stepwise=False,
        default_mode="z-score-tail",
        exception_modes={},
        section="action",
    )
    assert tail_spec == [{"mode": "z-score", "width": 1, "stats": {"mean": [1.0], "std": [2.0]}}]
    with pytest.raises(ValueError, match="not supported"):
        _normalization_specs(
            stats,
            items,
            stepwise=False,
            default_mode="q0001/q9999",
            exception_modes={},
            section="action",
        )


def test_conversion_supports_resolved_training_export_schema(tmp_path) -> None:
    shape_meta = _canonical_shape_meta(
        {
            "action": [
                {
                    "key": "right_control",
                    "sources": [{"lerobot_key": "action", "start_index": 0, "raw_shape": 6}],
                },
                {
                    "key": "right_gripper",
                    "sources": [{"lerobot_key": "action", "start_index": 6, "raw_shape": 1}],
                },
            ],
            "proprio": [
                {
                    "key": "right_control",
                    "sources": [{"lerobot_key": OBS_STATE, "start_index": 0, "raw_shape": 6}],
                },
                {
                    "key": "right_gripper",
                    "sources": [{"lerobot_key": OBS_STATE, "start_index": 6, "raw_shape": 1}],
                },
            ],
            "images": [
                {
                    "key": "exterior_rgb",
                    "sources": [{"lerobot_key": "observation.images.random_slot_0"}],
                }
            ],
        }
    )
    assert shape_meta["action"] == [
        {"key": "right_control", "lerobot_key": ACTION, "shape": 6},
        {"key": "right_gripper", "lerobot_key": ACTION, "shape": 1},
    ]
    assert shape_meta["state"] == [
        {"key": "right_control", "lerobot_key": OBS_STATE, "shape": 6},
        {"key": "right_gripper", "lerobot_key": OBS_STATE, "shape": 1},
    ]
    assert shape_meta["images"] == [{"key": "exterior_rgb", "lerobot_key": "observation.images.exterior_rgb"}]

    processor = _processor_contract(
        {
            "steps": [
                {
                    "_target_": "package.LinearNormalizer",
                    "default_mode": "q01/q99",
                    "exception_mode": {},
                    "use_stepwise_action_norm": True,
                },
                {"_target_": "package.PaddingActionMerger", "merge_spec": None},
            ]
        }
    )
    assert processor["norm_default_mode"] == "q01/q99"
    assert processor["use_stepwise_action_norm"] is True
    assert processor["action_state_merger"] == {"merge_spec": None}

    input_processor = tmp_path / "input_processor"
    input_processor.mkdir()
    (input_processor / "input_processor_config.json").write_text(
        '{"added_tokens":{"<action0000>":10,"<bos_blk>":11,"<EOV>":12,"<state>":13}}'
    )
    assert _exported_action_tokens(tmp_path) == ["<action0000>", "<bos_blk>"]


def test_conversion_maps_so101_camera_placeholders_to_lerobot_features() -> None:
    camera_keys, dummy_camera_keys, camera_order = _camera_layout(
        [
            {"key": "exterior", "lerobot_key": "__so100_exterior__"},
            {"key": "wrist_left", "lerobot_key": "__so100_wrist_left__"},
            {"key": "wrist_right", "lerobot_key": "__so100_wrist_right__"},
        ],
        output_camera_count=3,
    )

    assert camera_keys == [
        "observation.images.exterior_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ]
    assert dummy_camera_keys == []
    assert camera_order == [
        "observation.images.exterior_rgb",
        "observation.images.left_wrist_rgb",
        "observation.images.right_wrist_rgb",
    ]


def test_mrope_positions_use_image_grid_and_ignore_left_padding() -> None:
    config = _tiny_config()
    config.image_size = (64, 64)
    policy = G05Policy(config)
    input_ids = torch.tensor([[0, 7, 2, 2, 2, 2, 8]])
    attention_mask = input_ids.ne(0)

    positions = policy.model.build_mrope_position_ids(input_ids, attention_mask)

    torch.testing.assert_close(positions[:, 0, 0], torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(positions[:, 0, 1], torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(
        positions[:, 0, 2:6],
        torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]),
    )
    torch.testing.assert_close(positions[:, 0, 6], torch.full((3,), 3, dtype=torch.long))


def test_action_token_order_matches_g05_vocab_layout() -> None:
    model_config = {
        "AT_CONFIG": {
            "parts_meta": {
                "left_control": 9,
                "left_gripper": 1,
                "right_control": 9,
                "right_gripper": 1,
            },
            "rule_based_key_patterns": ["gripper"],
            "use_group_markers": True,
            "model_arch": {"codebook_size": 4096, "n_codebooks": 4},
        }
    }
    tokens = _action_tokens(model_config)
    assert len(tokens) == 4106
    assert tokens[0] == "<action0000>"
    assert tokens[4096:4100] == [
        "<left_control_0>",
        "<right_control_0>",
        "<left_control_1>",
        "<right_control_1>",
    ]
    assert tokens[-2:] == ["<left_gripper>", "<right_gripper>"]


def test_action_codec_grouped_token_roundtrip() -> None:
    codec_config = G05ActionCodecConfig(
        max_component_dim=3,
        horizon=4,
        horizon_patch_size=2,
        conv_in_action_kernel=2,
        encoder_channels=8,
        latent_dim=4,
        c_mults=[1],
        strides=[[1, 1]],
        transformer_depths=[1],
        num_heads=1,
        dim_heads=32,
        use_block_dct=True,
        block_dct_block_size=2,
        n_codebooks=1,
        codebook_size=16,
        codebook_dim=2,
        parts_meta={"control": 2, "gripper": 1},
        num_residuals=1,
    )
    tokenizer = G05ActionTokenizer(G05ActionCodecModel(codec_config))
    action = torch.tensor([[[0.1, 0.2, -1.0], [0.3, 0.4, -1.0], [0.5, 0.6, 1.0], [0.7, 0.8, 1.0]]])

    token_ids = tokenizer.encode_action_indices(action)
    decoded = tokenizer.decode_action_indices(token_ids)

    assert token_ids.shape == (1, 7)
    assert token_ids[0, 0] == tokenizer.marker_indices["<control_0>"]
    assert token_ids[0, -2] == tokenizer.marker_indices["<gripper>"]
    torch.testing.assert_close(decoded[..., -1], action[..., -1])


def _codec_config(**overrides) -> G05ActionCodecConfig:
    base = {
        "max_component_dim": 3,
        "horizon": 4,
        "horizon_patch_size": 2,
        "conv_in_action_kernel": 2,
        "encoder_channels": 8,
        "latent_dim": 4,
        "c_mults": [1],
        "strides": [[1, 1]],
        "transformer_depths": [1],
        "num_heads": 1,
        "dim_heads": 32,
        "use_block_dct": True,
        "block_dct_block_size": 2,
        "n_codebooks": 1,
        "codebook_size": 16,
        "codebook_dim": 2,
        "num_residuals": 1,
    }
    base.update(overrides)
    return G05ActionCodecConfig(**base)


def _initialize_codec_codebooks(model: G05ActionCodecModel) -> None:
    """Give a tiny codec a stable, non-degenerate initialized EMA state."""
    with torch.no_grad():
        for quantizer in model.rvq.quantizers:
            quantizer.codebook.normal_()
            quantizer.embed_avg.copy_(quantizer.codebook)
            quantizer.cluster_size.fill_(1)
            quantizer.inited.fill_(True)


def test_action_codec_forward_uses_full_training_objective() -> None:
    torch.manual_seed(0)
    config = _codec_config(
        quantizer_dropout=0.0,
        threshold_ema_dead=0.0,
        consistency_loss_weight=0.0,
    )
    model = G05ActionCodecModel(config).train()
    _initialize_codec_codebooks(model)
    cluster_size_before = model.rvq.quantizers[0].cluster_size.clone()
    assert model.action_time_contrastive_loss is None
    assert not any(key.startswith("action_time_contrastive_loss.") for key in model.state_dict())

    output = model({"control": torch.randn(2, config.horizon, 2)})
    loss_dict = output["loss_dict"]

    assert output["loss"] is loss_dict["loss"]
    assert loss_dict["commitment_loss"] > 0
    assert all(quantizer.commitment == 1.0 for quantizer in model.rvq.quantizers)
    expected = (
        config.reconstruction_loss_weight * loss_dict["reconstruction_loss"]
        + config.commitment_loss_weight * loss_dict["commitment_loss"]
    )
    torch.testing.assert_close(output["loss"].detach(), expected)
    assert not torch.equal(model.rvq.quantizers[0].cluster_size, cluster_size_before)
    assert any(key.startswith("codebook/perplexity_l") for key in loss_dict)

    output["loss"].backward()
    assert model.encoder.out_proj.weight.grad is not None
    assert torch.isfinite(model.encoder.out_proj.weight.grad).all()


def test_action_codec_forward_adds_optional_consistency_loss() -> None:
    torch.manual_seed(0)
    config = _codec_config(
        n_codebooks=2,
        quantizer_dropout=0.0,
        threshold_ema_dead=0.0,
        consistency_loss_weight=0.5,
        consistency_loss_type="token_residual",
    )
    model = G05ActionCodecModel(config).eval()
    _initialize_codec_codebooks(model)
    components = {"control": torch.randn(2, config.horizon, 2)}
    assert model.action_time_contrastive_loss is None
    positive_components = {"control": components["control"].roll(1, dims=1) * 1.05}

    output = model(
        components,
        x_pos_dict=positive_components,
        layer_weights=[0.5, 1.0],
    )
    loss_dict = output["loss_dict"]

    assert "consist/loss" in loss_dict
    assert "contrastive/loss" not in loss_dict
    expected = (
        config.reconstruction_loss_weight * loss_dict["reconstruction_loss"]
        + config.commitment_loss_weight * loss_dict["commitment_loss"]
        + config.consistency_loss_weight * loss_dict["consist/loss"]
    )
    torch.testing.assert_close(output["loss"].detach(), expected)


def test_action_codec_forward_adds_time_contrastive_loss() -> None:
    torch.manual_seed(0)
    config = _codec_config(
        n_codebooks=2,
        quantizer_dropout=0.0,
        threshold_ema_dead=0.0,
        consistency_loss_weight=0.5,
        consistency_loss_type="action_time_contrastive",
        action_time_contrastive_mode="siglip",
    )
    model = G05ActionCodecModel(config).eval()
    _initialize_codec_codebooks(model)
    components = {"control": torch.randn(2, config.horizon, 2)}
    assert {
        "action_time_contrastive_loss.logit_scale",
        "action_time_contrastive_loss.logit_bias",
    } <= set(model.state_dict())

    shifted = time_shift_positive(components["control"])
    torch.testing.assert_close(shifted[:, :1], components["control"][:, :1])
    torch.testing.assert_close(shifted[:, 1:], components["control"][:, :-1])

    # Construct the delayed positive view automatically when none is supplied.
    output = model(components)
    loss_dict = output["loss_dict"]

    assert "contrastive/loss" in loss_dict
    assert "consist/loss_layer_0" not in loss_dict
    expected = (
        config.reconstruction_loss_weight * loss_dict["reconstruction_loss"]
        + config.commitment_loss_weight * loss_dict["commitment_loss"]
        + config.consistency_loss_weight * loss_dict["contrastive/loss"]
    )
    torch.testing.assert_close(output["loss"].detach(), expected)

    output["loss"].backward()
    assert model.action_time_contrastive_loss is not None
    assert model.action_time_contrastive_loss.logit_scale.grad is not None
    assert torch.isfinite(model.action_time_contrastive_loss.logit_scale.grad)


def test_action_time_siglip_batch_one_metrics_are_finite() -> None:
    loss_function = ActionTimeContrastiveLoss(mode="siglip")
    hidden_states = torch.randn(1, 2, 3)

    loss, metrics = loss_function(hidden_states, hidden_states.roll(1, dims=-1))

    assert torch.isfinite(loss)
    assert all(torch.isfinite(metric).all() for metric in metrics.values())


def test_parts_order_survives_config_serialisation(tmp_path) -> None:
    """parts_meta's key order is the action layout, and saved JSON sorts keys.

    A non-alphabetical layout must survive a configuration round trip without
    permuting decoded action dimensions.
    """
    canonical = ["right_control", "right_gripper", "lower_body"]
    config = _codec_config(
        parts_meta={"right_control": 2, "right_gripper": 1, "lower_body": 3},
        parts_order=canonical,
    )
    config.save_pretrained(tmp_path)

    reloaded = G05ActionCodecConfig.from_pretrained(tmp_path)
    assert list(reloaded.parts_meta) == sorted(reloaded.parts_meta)
    assert reloaded.parts_order == canonical
    assert list(reloaded.ordered_parts_meta) == canonical


def test_tokenizer_lays_out_parts_in_canonical_order() -> None:
    """The decoded chunk follows parts_order, not the alphabetised parts_meta."""
    canonical = ["right_control", "gripper", "lower_body"]
    config = _codec_config(
        parts_meta={"right_control": 2, "gripper": 1, "lower_body": 3},
        parts_order=canonical,
    )
    tokenizer = G05ActionTokenizer(G05ActionCodecModel(config))
    assert list(tokenizer.parts_meta) == canonical

    # The gripper column is rule-coded and round-trips exactly, so it pins the
    # slot the layout assigns it: index 2, straight after right_control's width.
    action = torch.zeros(1, 4, 6)
    action[..., 2] = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    decoded = tokenizer.decode_action_indices(tokenizer.encode_action_indices(action))
    assert decoded.shape == (1, 4, 6)
    torch.testing.assert_close(decoded[..., 2], action[..., 2])


def test_greedy_decoding_ignores_repetition_penalties() -> None:
    """Greedy AR must be a plain argmax.

    When sampling is disabled, repetition penalties and n-gram constraints
    must not alter the selected token.
    """
    config = _tiny_config()
    config.ar_do_sample = False
    config.ar_repetition_penalty = 1.2
    config.ar_no_repeat_ngram_size = 3
    model = G05Policy(config).model

    logits = torch.zeros(1, 100)
    logits[0, 7] = 5.0
    logits[0, 8] = 4.0
    history = torch.tensor([[7, 7, 7]])

    # 7 already dominates and is heavily repeated; a penalised argmax picks 8.
    assert model.sample_next_token(logits, history).tolist() == [7]


def test_stepwise_normalization_and_restore() -> None:
    specs = [
        {
            "mode": "z-score",
            "width": 1,
            "stats": {
                "mean": [0.0],
                "std": [2.0],
            },
        },
        {"mode": "q01/q99", "width": 1, "stats": {"q01": [0.0], "q99": [4.0]}},
    ]
    values = torch.tensor([[[-1.0, 1.0], [1.0, 3.0]]])
    normalized = _apply_normalization(values, specs, inverse=False)
    torch.testing.assert_close(normalized, torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]]))
    torch.testing.assert_close(_apply_normalization(normalized, specs, inverse=True), values)

    restore = G05StepwiseActionUnnormalizerStep(
        action_horizon=2,
        action_normalization=[
            {
                "mode": "q01/q99",
                "width": 1,
                "stats": {"q01": [[0.0], [2.0]], "q99": [[2.0], [6.0]]},
            }
        ],
    )
    first = restore({TransitionKey.ACTION: torch.zeros(1)})[TransitionKey.ACTION]
    second = restore({TransitionKey.ACTION: torch.zeros(1)})[TransitionKey.ACTION]
    third = restore({TransitionKey.ACTION: torch.zeros(1)})[TransitionKey.ACTION]
    torch.testing.assert_close(first, torch.tensor([1.0]))
    torch.testing.assert_close(second, torch.tensor([4.0]))
    torch.testing.assert_close(third, first)


def test_libero_boundary_matches_g05_state_and_gripper_contract() -> None:
    observation = {
        "observation.images.image": torch.arange(12).reshape(1, 1, 3, 4),
        "observation.robot_state": {
            "eef": {
                "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            "gripper": {"qpos": torch.tensor([[0.03, -0.03]])},
        },
    }
    config = _tiny_config()
    config.embodiment = "libero"
    env_preprocessor, _ = make_env_pre_post_processors(LiberoEnv(), config)
    assert isinstance(env_preprocessor.steps[0], LiberoProcessorStep)
    env_processed = env_preprocessor.process_observation(observation)
    processed = G05LiberoObservationStep().observation(env_processed)

    torch.testing.assert_close(processed[OBS_STATE], torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.03]]))
    torch.testing.assert_close(
        processed["observation.images.image"],
        torch.flip(observation["observation.images.image"], dims=(-2, -1)),
    )

    transition = {TransitionKey.ACTION: torch.tensor([[0.0, 0.49], [0.0, 0.51]])}
    action = G05LiberoActionStep()(transition)[TransitionKey.ACTION]
    torch.testing.assert_close(action[..., -1], torch.tensor([1.0, -1.0]))


def test_chain_of_thought_is_logged_by_the_policy_not_the_runner(caplog) -> None:
    """Any caller of predict_action_chunk sees the CoT, without runner support."""
    config = _cot_ar_policy()
    policy = G05Policy(config)

    class FixedToken(torch.nn.Module):
        def forward(self, hidden_states):
            logits = hidden_states.new_zeros(hidden_states.shape[0], 100)
            logits[:, 4] = 1  # EOV, ending the CoT immediately
            return logits

    policy.model.output_proj = FixedToken()
    policy._text_tokenizer = type(
        "_Tok", (), {"decode": staticmethod(lambda ids, **kw: "Subtask: lift the block|Action: ")}
    )()
    policy._action_tokenizer = type(
        "_Codec", (), {"decode": staticmethod(lambda rows: torch.zeros(1, 2, 4))}
    )()
    batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[2, 3, 6]]),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        "pixel_values": torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
    }

    with caplog.at_level("INFO"):
        policy.predict_action_chunk(batch)

    assert "Chain of Thought" in caplog.text
    assert "lift the block" in caplog.text
