# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the TOPReward reward model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.topreward import TOPRewardConfig
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX, TOPREWARD_INPUT_KEYS
from tests.utils import skip_if_package_missing


class _FakeQwenModel(torch.nn.Module):
    """Stand-in for ``Qwen3VLForConditionalGeneration``.

    Returns a ``SimpleNamespace`` with ``logits`` of a controlled shape so
    the log-prob extraction path in ``compute_reward`` can be exercised
    without downloading real VLM weights.
    """

    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._reward_value: float = -1.5

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()

    def forward(  # noqa: ARG002
        self, input_ids, attention_mask=None, labels=None, logits_to_keep=0, **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        vocab_size = 1000
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        # Place a controlled log-prob at the target token position so the
        # model returns a predictable reward value.
        # The label-masked suffix is the last token.
        # After the causal-LM shift (logits[:, :-1], labels[:, 1:]) the scored
        # position is logits[:, -2, :] predicting labels[:, -1].
        # We set logits so that log_softmax at the target token ≈ _reward_value.
        for i in range(batch_size):
            target_idx = int(input_ids[i, -1].item())
            logits[i, -2, target_idx] = self._reward_value * -10  # high logit -> high log-prob
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def _patch_build(monkeypatch) -> None:
    """Stub out HF AutoX so TOPReward construction is cheap and offline."""
    from lerobot.rewards.topreward import modeling_topreward

    monkeypatch.setattr(modeling_topreward, "Qwen3VLForConditionalGeneration", _FakeQwenModel)


def _make_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    *,
    omit: str | None = None,
) -> dict[str, torch.Tensor]:
    """Build a ``compute_reward``-ready batch using TOPReward's namespaced keys."""
    batch_size, seq_len = input_ids.shape
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    batch: dict[str, torch.Tensor] = {}
    if labels is not None:
        batch[f"{TOPREWARD_FEATURE_PREFIX}labels"] = labels
    batch.update(
        {
            f"{TOPREWARD_FEATURE_PREFIX}input_ids": input_ids,
            f"{TOPREWARD_FEATURE_PREFIX}attention_mask": attention_mask,
            f"{TOPREWARD_FEATURE_PREFIX}pixel_values_videos": torch.zeros(
                batch_size, 1536, dtype=torch.float32
            ),
            f"{TOPREWARD_FEATURE_PREFIX}video_grid_thw": torch.ones(batch_size, 3, dtype=torch.long),
            f"{TOPREWARD_FEATURE_PREFIX}mm_token_type_ids": torch.zeros_like(input_ids),
        }
    )
    if omit is not None:
        batch.pop(f"{TOPREWARD_FEATURE_PREFIX}{omit}", None)
    return batch


def _terminal_labels(input_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    labels[:, -1] = input_ids[:, -1]
    return labels


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------


def test_topreward_config_registered():
    assert "topreward" in RewardModelConfig.get_known_choices()
    assert RewardModelConfig.get_choice_class("topreward") is TOPRewardConfig
    assert isinstance(make_reward_model_config("topreward", device="cpu"), TOPRewardConfig)


def test_topreward_factory_returns_in_tree_class():
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    assert get_reward_model_class("topreward") is TOPRewardModel


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_topreward_config_rejects_zero_max_frames():
    with pytest.raises(ValueError, match="max_frames must be >= 1"):
        TOPRewardConfig(device="cpu", max_frames=0)


def test_topreward_config_rejects_non_positive_fps():
    with pytest.raises(ValueError, match="fps must be > 0"):
        TOPRewardConfig(device="cpu", fps=0.0)


def test_topreward_config_rejects_suffix_without_instruction_placeholder():
    with pytest.raises(ValueError, match=r"\{instruction\}"):
        TOPRewardConfig(device="cpu", prompt_suffix_template="no placeholder here")


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_returns_one_scalar_per_sample(monkeypatch):
    """``compute_reward`` must return a ``(B,)`` float32 tensor with one
    log-prob reward per sample, consuming pre-encoded Qwen-VL tensors."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    labels = _terminal_labels(input_ids)

    batch = _make_batch(input_ids, attention_mask, labels)
    rewards = model.compute_reward(batch)

    assert rewards.shape == (2,)
    assert rewards.dtype == torch.float32


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_applies_success_threshold(monkeypatch):
    """When ``success_threshold`` is finite, the model returns binary success."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu", success_threshold=0.0)
    model = TOPRewardModel(cfg)

    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)
    labels = _terminal_labels(input_ids)

    batch = _make_batch(input_ids, attention_mask, labels)
    rewards = model.compute_reward(batch)

    assert rewards.shape == (2,)
    assert set(rewards.tolist()).issubset({0.0, 1.0})


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_errors_when_inputs_missing(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    with pytest.raises(KeyError, match=r"observation\.topreward\.input_ids"):
        model.compute_reward(_make_batch(torch.randint(0, 100, (1, 10)), omit="input_ids"))


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_errors_when_labels_missing(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    input_ids = torch.randint(0, 100, (1, 10))
    with pytest.raises(KeyError, match=r"observation\.topreward\.labels"):
        model.compute_reward(_make_batch(input_ids, labels=None))


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_requires_all_encoder_keys(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    input_ids = torch.randint(0, 100, (1, 10))
    labels = _terminal_labels(input_ids)
    required_encoder_keys = set(TOPREWARD_INPUT_KEYS) - {"input_ids", "labels"}

    for key in required_encoder_keys:
        with pytest.raises(KeyError, match=rf"observation\.topreward\.{key}"):
            model.compute_reward(_make_batch(input_ids, labels=labels, omit=key))


# ---------------------------------------------------------------------------
# Save / load — config-only checkpoint
# ---------------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_topreward_save_pretrained_writes_only_config_json(monkeypatch, tmp_path):
    from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(
        device="cpu",
        vlm_name="Qwen/Qwen3-VL-8B-Instruct",
        fps=4.0,
        image_key="observation.images.front",
    )
    model = TOPRewardModel(cfg)
    model.save_pretrained(str(tmp_path))

    assert (tmp_path / CONFIG_NAME).exists()
    assert not (tmp_path / SAFETENSORS_SINGLE_FILE).exists()


@skip_if_package_missing("transformers")
def test_topreward_from_pretrained_local_dir_roundtrips_config(monkeypatch, tmp_path):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(
        device="cpu",
        vlm_name="Qwen/Qwen3-VL-8B-Instruct",
        fps=4.0,
        image_key="observation.images.front",
        add_chat_template=True,
        success_threshold=-1.5,
    )
    TOPRewardModel(cfg).save_pretrained(str(tmp_path))

    reloaded = TOPRewardModel.from_pretrained(str(tmp_path))

    assert isinstance(reloaded.config, TOPRewardConfig)
    assert reloaded.config.vlm_name == "Qwen/Qwen3-VL-8B-Instruct"
    assert reloaded.config.fps == 4.0
    assert reloaded.config.image_key == "observation.images.front"
    assert reloaded.config.add_chat_template is True
    assert reloaded.config.success_threshold == -1.5


@skip_if_package_missing("transformers")
def test_topreward_is_not_trainable(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    assert model.is_trainable is False
    with pytest.raises(NotImplementedError, match="not trainable"):
        model.forward({"x": torch.zeros(1)})
