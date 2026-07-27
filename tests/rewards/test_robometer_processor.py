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

"""Tests for Robometer's pre-processing helpers and encoder step.

Covers the pure helpers (``_video_to_numpy`` and ``_expand_tasks``) directly,
and exercises :class:`RobometerEncoderProcessorStep` with a stubbed
``AutoProcessor`` so we don't need to download Qwen-VL just to test the
dataclass plumbing (``transform_features`` / ``get_config``).

The full ``__call__`` path that runs ``process_vision_info`` + the Qwen
processor is intentionally *not* covered here — it is essentially HF glue
that's exercised by the integration / parity scripts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.rewards.robometer.processor_robometer import (
    PROGRESS_PROMPT,
    _expand_tasks,
    _frames_to_pil,
    _video_to_numpy,
)
from tests.utils import skip_if_package_missing


def _skip_if_robometer_extras_missing(func):
    """Apply both optional-dependency guards in one shot.

    ``RobometerEncoderProcessorStep.__post_init__`` calls
    ``require_package("transformers", ...)`` *and*
    ``require_package("qwen-vl-utils", ...)``, so both need to be present
    before we can instantiate the step.
    """
    func = skip_if_package_missing("qwen-vl-utils", import_name="qwen_vl_utils")(func)
    func = skip_if_package_missing("transformers")(func)
    return func


# ---------------------------------------------------------------------------
# _video_to_numpy — pure tensor → uint8 (T, H, W, C) conversion
# ---------------------------------------------------------------------------


def test_video_to_numpy_chw_float_is_converted_to_thwc_uint8():
    video = torch.rand(4, 3, 8, 8)  # (T, C, H, W) floats in [0, 1]
    array = _video_to_numpy(video, max_frames=None)

    assert array.shape == (4, 8, 8, 3)
    assert array.dtype == np.uint8
    assert array.min() >= 0 and array.max() <= 255


def test_video_to_numpy_already_thwc_uint8_passes_through():
    video = torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8)  # (T, H, W, C)
    array = _video_to_numpy(video, max_frames=None)

    assert array.shape == (3, 8, 8, 3)
    assert array.dtype == np.uint8


def test_video_to_numpy_max_frames_tail_crops_recent_frames():
    """``max_frames`` should keep the **last** K frames (most recent)."""
    video = torch.zeros(10, 3, 4, 4)
    for t in range(10):
        video[t] = t / 9.0  # marker: 0 at t=0, ≈1 at t=9

    array = _video_to_numpy(video, max_frames=3)

    assert array.shape == (3, 4, 4, 3)
    # The first kept frame is t=7 → marker ≈ 7/9 → uint8 ≈ 198
    assert int(array[0, 0, 0, 0]) == int(round(7 / 9 * 255))
    # The last kept frame is t=9 → marker = 1.0 → uint8 = 255
    assert int(array[-1, 0, 0, 0]) == 255


def test_video_to_numpy_rejects_3d_input():
    with pytest.raises(ValueError, match="Expected channel dim"):
        _video_to_numpy(torch.zeros(4, 8, 8), max_frames=None)


def test_video_to_numpy_floats_above_one_pass_through_without_rescaling():
    """If ``array.max() > 1`` the helper assumes the tensor is already in the
    [0, 255] range (uint8-as-float), so values pass through unchanged."""
    video = torch.full((1, 3, 2, 2), 5.0)
    array = _video_to_numpy(video, max_frames=None)

    assert array.shape == (1, 2, 2, 3)
    assert int(array.max()) == 5


def test_video_to_numpy_clips_very_large_floats_to_uint8_max():
    """Out-of-uint8-range floats are clipped at 255 before the cast."""
    video = torch.full((1, 3, 2, 2), 300.0)
    array = _video_to_numpy(video, max_frames=None)

    assert int(array.max()) == 255


# ---------------------------------------------------------------------------
# _expand_tasks — string / list / tuple broadcasting to batch size
# ---------------------------------------------------------------------------


def test_expand_tasks_string_is_broadcast_to_batch_size():
    assert _expand_tasks("pick up", batch_size=3, default=None) == ["pick up", "pick up", "pick up"]


def test_expand_tasks_list_of_matching_size_passes_through():
    assert _expand_tasks(["a", "b", "c"], batch_size=3, default=None) == ["a", "b", "c"]


def test_expand_tasks_tuple_is_normalised_to_list():
    assert _expand_tasks(("a", "b"), batch_size=2, default=None) == ["a", "b"]


def test_expand_tasks_single_element_list_is_broadcast():
    assert _expand_tasks(["only one"], batch_size=3, default=None) == ["only one"] * 3


def test_expand_tasks_size_mismatch_raises():
    with pytest.raises(ValueError, match="Expected 3 tasks"):
        _expand_tasks(["a", "b"], batch_size=3, default=None)


def test_expand_tasks_missing_uses_default():
    assert _expand_tasks(None, batch_size=2, default="fallback") == ["fallback", "fallback"]


def test_expand_tasks_missing_without_default_raises():
    with pytest.raises(KeyError, match="task description"):
        _expand_tasks(None, batch_size=1, default=None)


def test_expand_tasks_wrong_type_raises():
    with pytest.raises(TypeError, match="must be a string or list"):
        _expand_tasks(42, batch_size=1, default=None)


# ---------------------------------------------------------------------------
# _frames_to_pil — uint8 (T, H, W, C) → list[PIL.Image]
# ---------------------------------------------------------------------------


def test_frames_to_pil_returns_one_image_per_frame():
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    images = _frames_to_pil(frames)

    assert len(images) == 4
    assert all(img.size == (8, 8) for img in images)


def test_frames_to_pil_casts_floats_to_uint8():
    frames = np.full((2, 4, 4, 3), 200.0, dtype=np.float32)
    images = _frames_to_pil(frames)

    assert len(images) == 2
    # PIL converted from clipped uint8 - sanity check pixel values come through.
    assert np.asarray(images[0]).dtype == np.uint8


def test_frames_to_pil_rejects_non_4d_input():
    with pytest.raises(ValueError, match=r"\(T,H,W,C\)"):
        _frames_to_pil(np.zeros((4, 8, 8), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Encoder step plumbing — exercise dataclass surface with a stubbed AutoProcessor
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Tokenizer surface the encoder step touches in ``__post_init__``."""

    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.eos_token = "<|endoftext|>"
        self._vocab: dict[str, int] = {"<|endoftext|>": 0}
        self.added: list[str] = []

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def add_special_tokens(self, payload: dict[str, Any]) -> int:
        for token in payload.get("additional_special_tokens", []):
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                self.added.append(token)
        return len(self.added)


class _FakeAutoProcessor:
    """Stand-in returned by ``AutoProcessor.from_pretrained`` during tests."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.image_processor = None
        self.video_processor = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return cls()


def _build_step(monkeypatch, **overrides):
    from lerobot.rewards.robometer import processor_robometer

    monkeypatch.setattr(processor_robometer, "AutoProcessor", _FakeAutoProcessor)

    return processor_robometer.RobometerEncoderProcessorStep(**overrides)


@_skip_if_robometer_extras_missing
def test_encoder_step_registers_special_tokens_on_tokenizer(monkeypatch):
    """``__post_init__`` must register Robometer's five special tokens on the
    tokenizer that ships with the chosen Qwen-VL checkpoint."""
    from lerobot.rewards.robometer.configuration_robometer import ROBOMETER_SPECIAL_TOKENS

    step = _build_step(monkeypatch)

    vocab = step._processor.tokenizer.get_vocab()
    for token in ROBOMETER_SPECIAL_TOKENS:
        assert token in vocab, f"{token} not registered on the tokenizer"


@_skip_if_robometer_extras_missing
def test_encoder_step_sets_pad_token_to_eos_when_missing(monkeypatch):
    """Qwen tokenizers ship without a pad token; the step must reuse EOS so
    batched processing doesn't crash on padding."""
    step = _build_step(monkeypatch)

    assert step._processor.tokenizer.pad_token == "<|endoftext|>"


@_skip_if_robometer_extras_missing
def test_encoder_step_get_config_roundtrips_user_fields(monkeypatch):
    """``get_config`` must serialise every user-tunable field — these are what
    the processor pipeline saves under ``preprocessor_config.json``."""
    step = _build_step(
        monkeypatch,
        base_model_id="Qwen/Qwen3-VL-4B-Instruct",
        image_key="observation.images.cam_top",
        task_key="task",
        default_task="do the thing",
        max_frames=12,
        use_multi_image=True,
        use_per_frame_progress_token=True,
        max_length=2048,
    )

    cfg = step.get_config()
    assert cfg == {
        "base_model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "image_key": "observation.images.cam_top",
        "task_key": "task",
        "default_task": "do the thing",
        "max_frames": 12,
        "use_multi_image": True,
        "use_per_frame_progress_token": True,
        "max_length": 2048,
    }


@_skip_if_robometer_extras_missing
def test_encoder_step_transform_features_is_identity(monkeypatch):
    """The encoder step writes Qwen tensors into ``observation`` at call time,
    but it does **not** advertise new typed features at pipeline-build time —
    the downstream model consumes them via the ``ROBOMETER_FEATURE_PREFIX``
    namespace, not via the typed feature map.
    """
    step = _build_step(monkeypatch)

    features = {
        PipelineFeatureType.OBSERVATION: {
            "observation.images.top": PolicyFeature(shape=(3, 224, 224), type=FeatureType.VISUAL),
        }
    }
    assert step.transform_features(features) == features


@_skip_if_robometer_extras_missing
def test_encoder_step_build_conversation_inserts_prog_token_per_frame(monkeypatch):
    """In multi-image mode with per-frame progress tokens, the conversation
    must alternate ``image`` and ``<|prog_token|>`` text entries, one pair
    per frame, after the task prompt."""
    step = _build_step(
        monkeypatch,
        use_multi_image=True,
        use_per_frame_progress_token=True,
    )

    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    conversation = step._build_conversation(frames, task="pick up the cube")

    assert len(conversation) == 1 and conversation[0]["role"] == "user"
    content = conversation[0]["content"]

    # First entry is the task prompt.
    assert content[0] == {"type": "text", "text": PROGRESS_PROMPT.format(task="pick up the cube")}

    # Then 3 (image, <|prog_token|>) pairs.
    expected_tail = [
        item
        for _ in range(3)
        for item in (
            {"type": "image"},  # value asserted below
            {"type": "text", "text": "<|prog_token|>"},
        )
    ]
    assert len(content) == 1 + len(expected_tail)
    for got, exp in zip(content[1:], expected_tail, strict=True):
        assert got["type"] == exp["type"]
        if exp["type"] == "text":
            assert got["text"] == exp["text"]


@_skip_if_robometer_extras_missing
def test_encoder_step_build_conversation_video_mode_uses_single_video_entry(monkeypatch):
    """When ``use_multi_image=False``, frames are bundled into a single
    ``video`` content entry instead of individual ``image`` entries."""
    step = _build_step(
        monkeypatch,
        use_multi_image=False,
        use_per_frame_progress_token=False,
    )

    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    conversation = step._build_conversation(frames, task="pour the water")

    content = conversation[0]["content"]
    # Exactly two entries: the prompt and one video entry.
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "video"
    # The video entry carries all four frames.
    assert len(content[1]["video"]) == 4
