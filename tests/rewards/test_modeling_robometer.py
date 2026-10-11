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

"""Tests for Robometer reward model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.robometer import RobometerConfig
from lerobot.rewards.robometer.configuration_robometer import ROBOMETER_SPECIAL_TOKENS
from lerobot.rewards.robometer.modeling_robometer import (
    ROBOMETER_FEATURE_PREFIX,
    convert_bins_to_continuous,
    decode_progress_outputs,
)
from tests.utils import skip_if_package_missing

# Length of the fake tokenizer used in `_patch_build`. The deterministic
# resize target derived in ``RobometerConfig.__post_init__`` is therefore
# ``_FAKE_TOKENIZER_LEN + len(ROBOMETER_SPECIAL_TOKENS)``.
_FAKE_TOKENIZER_LEN = 100
_EXPECTED_RESIZED_VOCAB = _FAKE_TOKENIZER_LEN + len(ROBOMETER_SPECIAL_TOKENS)


class _FakeQwenConfig:
    """Stand-in for a Qwen3-VL config (the `model.config` attribute).

    ``to_dict`` matches HF's ``PretrainedConfig.to_dict`` closely enough for
    ``RobometerConfig.__post_init__`` to snapshot a meaningful ``vlm_config``
    into the saved ``config.json`` and for the reload path to round-trip
    through ``AutoConfig.for_model``.
    """

    def __init__(self, hidden_dim: int = 8, vocab_size: int = _FAKE_TOKENIZER_LEN) -> None:
        # `vocab_size` here is the *pre-resize* value the fake backbone advertises.
        # `__post_init__` is expected to overwrite it with `len(tokenizer) + 5`.
        self.text_config = SimpleNamespace(hidden_size=hidden_dim, vocab_size=vocab_size)
        self._hidden_dim = hidden_dim
        self._vocab_size = vocab_size

    def to_dict(self) -> dict:
        return {
            "model_type": "fake_qwen",
            "text_config": {
                "hidden_size": self._hidden_dim,
                "vocab_size": self._vocab_size,
            },
        }


class _FakeEmbeddings(torch.nn.Module):
    def __init__(self, num_embeddings: int = _FAKE_TOKENIZER_LEN) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings


class _FakeBaseModel(torch.nn.Module):
    """Stand-in for the Qwen3-VL backbone during tests.

    Provides the minimum surface `RobometerRewardModel.__init__` and
    `_compute_rbm_logits` rely on: a `parameters()` iterator (for dtype +
    device), a `config.text_config.hidden_size`, a `config.to_dict()` so
    `_save_pretrained` can snapshot `vlm_config`,
    `get_input_embeddings()` / `resize_token_embeddings()` so the fresh-init
    embed resize is a no-op, and a forward that returns a `SimpleNamespace`
    with a `hidden_states` tuple.
    """

    def __init__(self, hidden_dim: int = 8) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.hidden_dim = hidden_dim
        self.config = _FakeQwenConfig(hidden_dim)
        self._embeddings = _FakeEmbeddings()

    def get_input_embeddings(self) -> _FakeEmbeddings:
        return self._embeddings

    def resize_token_embeddings(self, new_size: int) -> None:
        self._embeddings.num_embeddings = new_size

    def forward(self, **kwargs):  # noqa: ARG002 - intentional kwargs sink
        input_ids = kwargs["input_ids"]
        return SimpleNamespace(
            hidden_states=(torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden_dim),),
            last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden_dim),
        )


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer.

    ``RobometerConfig.__post_init__`` uses ``len(tokenizer)`` to compute the
    deterministic resize target ``len(tokenizer) + len(ROBOMETER_SPECIAL_TOKENS)``,
    so a working ``__len__`` is all we need.
    """

    def __init__(self, length: int = _FAKE_TOKENIZER_LEN) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


def _patch_build(monkeypatch) -> None:
    """Stub out the HF AutoX calls so Robometer construction stays cheap in tests.

    Covers (EO-1 style — no model-side override hooks):
    * ``AutoConfig.from_pretrained`` (config side) — used by
      ``RobometerConfig.__post_init__`` to snapshot the backbone config.
    * ``AutoTokenizer.from_pretrained`` (config side) — used by
      ``__post_init__`` to compute ``len(tokenizer) + 5``.
    * ``AutoConfig.for_model``                       — used by
      ``RobometerConfig.vlm_backbone_config`` when rebuilding for ``from_config``.
    * ``AutoModelForImageTextToText.from_pretrained`` — fresh-training path
      (``pretrained_path is None``).
    * ``AutoModelForImageTextToText.from_config``    — checkpoint-reload path
      (``pretrained_path`` is set).
    """
    from lerobot.rewards.robometer import configuration_robometer, modeling_robometer

    monkeypatch.setattr(
        modeling_robometer.AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: _FakeBaseModel(hidden_dim=8),
    )
    monkeypatch.setattr(
        modeling_robometer.AutoModelForImageTextToText,
        "from_config",
        lambda *args, **kwargs: _FakeBaseModel(hidden_dim=8),
    )
    monkeypatch.setattr(
        configuration_robometer.AutoConfig,
        "for_model",
        lambda *args, **kwargs: _FakeQwenConfig(hidden_dim=8),
    )
    monkeypatch.setattr(
        configuration_robometer.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: _FakeQwenConfig(hidden_dim=8),
    )
    monkeypatch.setattr(
        configuration_robometer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(length=_FAKE_TOKENIZER_LEN),
    )


def _make_batch(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build a `compute_reward`-ready batch using Robometer's namespaced keys."""
    return {f"{ROBOMETER_FEATURE_PREFIX}{key}": value for key, value in features.items()}


@skip_if_package_missing("transformers")
def test_robometer_config_registered(monkeypatch):
    _patch_build(monkeypatch)
    assert "robometer" in RewardModelConfig.get_known_choices()
    assert RewardModelConfig.get_choice_class("robometer") is RobometerConfig
    assert isinstance(make_reward_model_config("robometer", device="cpu"), RobometerConfig)


def test_robometer_factory_returns_in_tree_class():
    from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel

    assert get_reward_model_class("robometer") is RobometerRewardModel


def test_convert_bins_to_continuous_returns_expected_values():
    # Two frames: first peaks at bin 0 (center 0.0), second peaks at bin 9 (center 1.0).
    bin_logits = torch.full((2, 10), -10.0)
    bin_logits[0, 0] = 10.0
    bin_logits[1, -1] = 10.0
    values = convert_bins_to_continuous(bin_logits)
    assert values.shape == (2,)
    assert torch.allclose(values, torch.tensor([0.0, 1.0]), atol=1e-3)


def test_decode_progress_outputs_returns_last_frame_values():
    progress = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    success_logits = torch.tensor([[0.0, 5.0], [0.0, -5.0]])

    outputs = decode_progress_outputs(progress, success_logits, is_discrete_mode=False)

    assert outputs["progress_pred"] == [pytest.approx([0.1, 0.9]), pytest.approx([0.4, 0.6])]
    assert outputs["success_probs"][0][-1] == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item(), abs=1e-3)
    assert outputs["success_probs"][1][-1] == pytest.approx(
        torch.sigmoid(torch.tensor(-5.0)).item(), abs=1e-3
    )


def test_decode_progress_outputs_discrete_mode_softmaxes_over_bins():
    # 2 frames, peaks at bin 0 and bin 9 → continuous predictions 0.0 and 1.0
    bin_logits = torch.full((1, 2, 10), -10.0)
    bin_logits[0, 0, 0] = 10.0
    bin_logits[0, 1, -1] = 10.0

    outputs = decode_progress_outputs(bin_logits, success_logits=None, is_discrete_mode=True)

    assert outputs["success_probs"] == []
    assert outputs["progress_pred"][0] == pytest.approx([0.0, 1.0], abs=1e-3)


@skip_if_package_missing("transformers")
def test_robometer_post_init_overwrites_vocab_size_with_tokenizer_length(monkeypatch):
    """``RobometerConfig.__post_init__`` must overwrite the backbone's stale
    ``text_config.vocab_size`` (which on the real Qwen3-VL config is the
    padded embedding size, ``151,936``) with ``len(tokenizer) + 5``. This is
    the contract that makes the published ``Robometer-4B`` checkpoint load
    byte-equivalently."""
    _patch_build(monkeypatch)

    cfg = RobometerConfig(device="cpu", progress_loss_type="l2")

    assert cfg.vlm_config["text_config"]["vocab_size"] == _EXPECTED_RESIZED_VOCAB


@skip_if_package_missing("transformers")
def test_robometer_compute_reward_reads_pre_encoded_inputs(monkeypatch):
    from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel

    progress = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    success_logits = torch.tensor([[0.0, 5.0], [0.0, -5.0]])
    _patch_build(monkeypatch)

    cfg = RobometerConfig(device="cpu", reward_output="progress", progress_loss_type="l2")
    model = RobometerRewardModel(cfg)
    # Bypass the Qwen3-VL forward + head extraction with deterministic logits.
    monkeypatch.setattr(model, "_compute_rbm_logits", lambda _inputs: (progress, success_logits))

    batch = _make_batch({"input_ids": torch.zeros(2, 2, dtype=torch.long)})
    rewards = model.compute_reward(batch)

    assert torch.allclose(rewards, torch.tensor([0.9, 0.6]))


@skip_if_package_missing("transformers")
def test_robometer_compute_reward_can_return_binary_success(monkeypatch):
    from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel

    progress = torch.tensor([[0.1, 0.9], [0.4, 0.6]])
    success_logits = torch.tensor([[0.0, 5.0], [0.0, -5.0]])  # sigmoid(5) > 0.5; sigmoid(-5) < 0.5
    _patch_build(monkeypatch)

    cfg = RobometerConfig(
        device="cpu",
        reward_output="success",
        success_threshold=0.5,
        progress_loss_type="l2",
    )
    model = RobometerRewardModel(cfg)
    monkeypatch.setattr(model, "_compute_rbm_logits", lambda _inputs: (progress, success_logits))

    batch = _make_batch({"input_ids": torch.zeros(2, 2, dtype=torch.long)})
    rewards = model.compute_reward(batch)

    assert torch.equal(rewards, torch.tensor([1.0, 0.0]))


@skip_if_package_missing("transformers")
def test_robometer_compute_reward_errors_when_inputs_missing(monkeypatch):
    from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel

    _patch_build(monkeypatch)

    cfg = RobometerConfig(device="cpu", progress_loss_type="l2")
    model = RobometerRewardModel(cfg)

    with pytest.raises(KeyError, match=r"observation\.robometer\.input_ids"):
        model.compute_reward({})


@skip_if_package_missing("transformers")
def test_robometer_save_pretrained_roundtrips(monkeypatch, tmp_path):
    """Saving and reloading a Robometer model in LeRobot HF format must produce
    a single ``model.safetensors`` + ``config.json`` (no Hydra ``config.yaml``),
    must round-trip user-tunable config fields, and must persist all three
    prediction heads (``progress_head``, ``success_head``, ``preference_head``)
    so the published ``Robometer-4B`` checkpoint loads byte-equivalently.
    """
    from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
    from safetensors.torch import load_file

    from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel

    _patch_build(monkeypatch)
    cfg = RobometerConfig(
        device="cpu",
        pretrained_path="robometer/Robometer-4B",
        # Knobs the user might tweak — must survive the round-trip.
        image_key="observation.images.cam_top",
        task_key="task",
        reward_output="success",
        success_threshold=0.7,
        progress_loss_type="l2",
    )
    model = RobometerRewardModel(cfg)
    model.save_pretrained(str(tmp_path))

    # Exactly the files LeRobot's HubMixin promises.
    assert (tmp_path / CONFIG_NAME).exists()
    assert (tmp_path / SAFETENSORS_SINGLE_FILE).exists()
    assert not (tmp_path / "config.yaml").exists()  # we want HF-style, not Hydra

    # All three heads must be present in the saved safetensors. The preference
    # head is unused at inference but the published checkpoint expects its
    # rows — losing it would silently break weight loading.
    state = load_file(str(tmp_path / SAFETENSORS_SINGLE_FILE))
    assert any(k.startswith("progress_head.") for k in state), "progress_head weights missing"
    assert any(k.startswith("success_head.") for k in state), "success_head weights missing"
    assert any(k.startswith("preference_head.") for k in state), "preference_head weights missing"

    # Reload from the local directory: no Hub fetch, no YAML overlay. The
    # base class drives subclass dispatch via the `type` field in config.json.
    reloaded_cfg = RewardModelConfig.from_pretrained(str(tmp_path))
    assert isinstance(reloaded_cfg, RobometerConfig)
    reloaded_cfg.pretrained_path = str(tmp_path)  # mimic lerobot-train's `validate()`
    reloaded = RobometerRewardModel.from_pretrained(str(tmp_path), config=reloaded_cfg)

    assert reloaded.config.image_key == "observation.images.cam_top"
    assert reloaded.config.task_key == "task"
    assert reloaded.config.reward_output == "success"
    assert reloaded.config.success_threshold == 0.7
    assert reloaded.config.progress_loss_type == "l2"  # came back from config.json
