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

import numpy as np
import pytest
import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.rewards.factory import get_reward_model_class, make_reward_model_config
from lerobot.rewards.topreward import TOPRewardConfig
from lerobot.rewards.topreward.modeling_topreward import minmax_normalize_rewards
from lerobot.rewards.topreward.processor_topreward import TOPREWARD_FEATURE_PREFIX
from tests.utils import skip_if_package_missing


class _FakeTokenizer:
    """Minimal tokenizer surface used by ``TOPRewardModel._compute_log_prob_reward``."""

    eos_token = "<|endoftext|>"


class _FakeProcessor:
    """Stand-in for the Qwen ``AutoProcessor`` returned by ``from_pretrained``."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()


class _FakeQwenModel(torch.nn.Module):
    """Stand-in for ``Qwen3VLForConditionalGeneration``.

    Provides the minimum surface ``TOPRewardModel`` touches at construction
    time (a ``parameters()`` iterator for device inference). Actual
    ``_compute_log_prob_reward`` calls are bypassed by monkey-patching the
    method directly in the tests, so we never invoke ``self.model(...)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()


def _patch_build(monkeypatch) -> None:
    """Stub out HF AutoX so TOPReward construction is cheap and offline."""
    from lerobot.rewards.topreward import modeling_topreward

    monkeypatch.setattr(modeling_topreward, "Qwen3VLForConditionalGeneration", _FakeQwenModel)
    monkeypatch.setattr(modeling_topreward, "AutoProcessor", _FakeProcessor)


def _make_batch(frames: list[np.ndarray], tasks: list[str]) -> dict[str, list]:
    """Build a ``compute_reward``-ready batch using TOPReward's namespaced keys."""
    return {
        f"{TOPREWARD_FEATURE_PREFIX}frames": frames,
        f"{TOPREWARD_FEATURE_PREFIX}task": tasks,
    }


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


def test_topreward_config_rejects_bad_reduction():
    with pytest.raises(ValueError, match="reduction must be"):
        TOPRewardConfig(device="cpu", reduction="median")


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
# minmax_normalize_rewards — pure math helper
# ---------------------------------------------------------------------------


def test_minmax_normalize_rewards_maps_min_and_max_to_zero_and_one():
    values = minmax_normalize_rewards([-3.0, -1.0, 0.0, -2.0])
    assert values.shape == (4,)
    assert values[0] == pytest.approx(0.0)
    assert values[2] == pytest.approx(1.0)
    # Monotonicity preserved within the input range.
    assert values[3] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_minmax_normalize_rewards_handles_singleton_and_flat_inputs():
    # Single element -> mapped to 1.0 (no information to scale).
    assert minmax_normalize_rewards([42.0]).tolist() == [1.0]
    # All-equal values -> all ones (avoid divide-by-zero).
    assert minmax_normalize_rewards([0.5, 0.5, 0.5]).tolist() == [1.0, 1.0, 1.0]


def test_minmax_normalize_rewards_empty_input_returns_empty_array():
    out = minmax_normalize_rewards([])
    assert out.shape == (0,)


# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_returns_one_scalar_per_sample(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    captured = []

    def fake_log_prob(self, frames, instruction):  # noqa: ARG002
        captured.append((frames.shape, instruction))
        return -1.5

    monkeypatch.setattr(TOPRewardModel, "_compute_log_prob_reward", fake_log_prob)

    frames_a = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    frames_b = np.zeros((6, 8, 8, 3), dtype=np.uint8)
    batch = _make_batch([frames_a, frames_b], ["pick the cube", "open the drawer"])

    rewards = model.compute_reward(batch)

    assert rewards.shape == (2,)
    assert rewards.dtype == torch.float32
    assert torch.allclose(rewards, torch.tensor([-1.5, -1.5]))
    # `_compute_log_prob_reward` was called once per sample with the right tasks.
    assert [task for _, task in captured] == ["pick the cube", "open the drawer"]
    assert [shape[0] for shape, _ in captured] == [4, 6]


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_applies_success_threshold(monkeypatch):
    """When ``success_threshold`` is finite, the model returns binary success
    instead of the raw log-prob — useful as a drop-in success detector."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu", success_threshold=-2.0)
    model = TOPRewardModel(cfg)

    rewards_in = iter([-1.5, -3.0])  # first above threshold, second below
    monkeypatch.setattr(
        TOPRewardModel,
        "_compute_log_prob_reward",
        lambda _self, _frames, _instr: next(rewards_in),
    )

    frames = [np.zeros((2, 8, 8, 3), dtype=np.uint8), np.zeros((2, 8, 8, 3), dtype=np.uint8)]
    rewards = model.compute_reward(_make_batch(frames, ["task", "task"]))

    assert torch.equal(rewards, torch.tensor([1.0, 0.0]))


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_errors_when_inputs_missing(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    with pytest.raises(KeyError, match=r"observation\.topreward\."):
        model.compute_reward({})


@skip_if_package_missing("transformers")
def test_topreward_compute_reward_errors_when_batch_sizes_mismatch(monkeypatch):
    """frames and task lists must have matching lengths — a stale processor
    that produces only one task for a multi-sample batch should surface as
    an explicit error, not a silent zip truncation."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)
    monkeypatch.setattr(
        TOPRewardModel,
        "_compute_log_prob_reward",
        lambda _self, _frames, _instr: 0.0,
    )

    frames = [np.zeros((2, 8, 8, 3), dtype=np.uint8), np.zeros((2, 8, 8, 3), dtype=np.uint8)]
    with pytest.raises(ValueError, match="task batch size"):
        model.compute_reward(_make_batch(frames, ["only one task"]))


# ---------------------------------------------------------------------------
# predict_curves
# ---------------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_topreward_predict_curves_runs_one_forward_per_prefix(monkeypatch):
    """``predict_curves`` must call the VLM once per prefix length per
    trajectory and write min-max-normalised values back into the curve."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    # Simulate a strictly increasing log-prob curve as the prefix grows.
    call_log: list[int] = []

    def fake_log_prob(self, frames, instruction):  # noqa: ARG002
        call_log.append(int(frames.shape[0]))
        return float(frames.shape[0])  # log-prob = prefix length

    monkeypatch.setattr(TOPRewardModel, "_compute_log_prob_reward", fake_log_prob)

    frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    batch = _make_batch([frames], ["lift the cup"])
    out = model.predict_curves(batch)

    # One forward per prefix length, in order.
    assert call_log == [1, 2, 3, 4, 5]
    # (B, T_max) shape, padded with NaN beyond each trajectory's length.
    assert out["progress"].shape == (1, 5)
    # Strictly increasing raw rewards -> min-max-normalised to [0, 1] linearly.
    expected = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
    assert torch.allclose(out["progress"], expected, atol=1e-6)


@skip_if_package_missing("transformers")
def test_topreward_predict_curves_sparse_dense_interpolates_to_full_resolution(monkeypatch):
    """With ``num_prefixes < N`` the model should score only the requested
    number of anchor prefixes and linearly interpolate between them — the
    upstream sparse-dense pattern (``num_samples=15``)."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    call_log: list[int] = []

    def fake_log_prob(self, frames, instruction):  # noqa: ARG002
        call_log.append(int(frames.shape[0]))
        return float(frames.shape[0])

    monkeypatch.setattr(TOPRewardModel, "_compute_log_prob_reward", fake_log_prob)

    frames = np.zeros((9, 8, 8, 3), dtype=np.uint8)
    out = model.predict_curves(_make_batch([frames], ["lift the cup"]), num_prefixes=3)

    # 3 anchors at linspace(1, 9, 3) -> [1, 5, 9] -> 3 VLM forwards instead of 9.
    assert call_log == [1, 5, 9]
    # Returned curve is full resolution (9 frames) and monotone in [0, 1].
    assert out["progress"].shape == (1, 9)
    curve = out["progress"][0].numpy()
    assert curve[0] == pytest.approx(0.0)
    assert curve[-1] == pytest.approx(1.0)
    assert np.all(np.diff(curve) >= 0)


@skip_if_package_missing("transformers")
def test_topreward_predict_curves_rejects_invalid_num_prefixes(monkeypatch):
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    model = TOPRewardModel(TOPRewardConfig(device="cpu"))
    batch = _make_batch([np.zeros((3, 8, 8, 3), dtype=np.uint8)], ["task"])
    with pytest.raises(ValueError, match="num_prefixes must be"):
        model.predict_curves(batch, num_prefixes=0)


@skip_if_package_missing("transformers")
def test_topreward_predict_curves_right_pads_with_nan_for_variable_lengths(monkeypatch):
    """Trajectories of different lengths in the same batch are right-padded
    with ``NaN`` so the output is a regular ``(B, T_max)`` tensor."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)
    monkeypatch.setattr(
        TOPRewardModel,
        "_compute_log_prob_reward",
        lambda _self, frames, _instr: float(frames.shape[0]),
    )

    frames_short = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames_long = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    out = model.predict_curves(_make_batch([frames_short, frames_long], ["a", "b"]))

    assert out["progress"].shape == (2, 4)
    # Trailing entries for the shorter trajectory are NaN.
    assert torch.isnan(out["progress"][0, 2:]).all()
    # The longer trajectory has no NaNs.
    assert not torch.isnan(out["progress"][1]).any()


# ---------------------------------------------------------------------------
# Save / load — config-only checkpoint
# ---------------------------------------------------------------------------


@skip_if_package_missing("transformers")
def test_topreward_save_pretrained_writes_only_config_json(monkeypatch, tmp_path):
    """A TOPReward "checkpoint" is just ``config.json``. Writing
    ``model.safetensors`` would only duplicate ~16 GB of Qwen weights for
    no benefit, so :meth:`_save_pretrained` must skip it entirely.
    """
    from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE

    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(
        device="cpu",
        vlm_name="Qwen/Qwen3-VL-8B-Instruct",
        reduction="sum",
        fps=4.0,
        image_key="observation.images.front",
    )
    model = TOPRewardModel(cfg)
    model.save_pretrained(str(tmp_path))

    assert (tmp_path / CONFIG_NAME).exists()
    # Zero-shot model: no safetensors written by `_save_pretrained`.
    assert not (tmp_path / SAFETENSORS_SINGLE_FILE).exists()


@skip_if_package_missing("transformers")
def test_topreward_from_pretrained_local_dir_roundtrips_config(monkeypatch, tmp_path):
    """Save a TOPRewardConfig locally and reload it — user knobs must survive."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(
        device="cpu",
        vlm_name="Qwen/Qwen3-VL-8B-Instruct",
        reduction="sum",
        fps=4.0,
        image_key="observation.images.front",
        use_video_description=True,
        add_chat_template=True,
        success_threshold=-1.5,
    )
    TOPRewardModel(cfg).save_pretrained(str(tmp_path))

    reloaded = TOPRewardModel.from_pretrained(str(tmp_path))

    assert isinstance(reloaded.config, TOPRewardConfig)
    assert reloaded.config.vlm_name == "Qwen/Qwen3-VL-8B-Instruct"
    assert reloaded.config.reduction == "sum"
    assert reloaded.config.fps == 4.0
    assert reloaded.config.image_key == "observation.images.front"
    assert reloaded.config.use_video_description is True
    assert reloaded.config.add_chat_template is True
    assert reloaded.config.success_threshold == -1.5


@skip_if_package_missing("transformers")
def test_topreward_is_not_trainable(monkeypatch):
    """The whole point of TOPReward is that it is zero-shot.
    ``is_trainable`` must therefore be ``False`` and ``forward(...)`` must
    raise the base-class ``NotImplementedError``."""
    from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel

    _patch_build(monkeypatch)
    cfg = TOPRewardConfig(device="cpu")
    model = TOPRewardModel(cfg)

    assert model.is_trainable is False
    with pytest.raises(NotImplementedError, match="not trainable"):
        model.forward({"x": torch.zeros(1)})
