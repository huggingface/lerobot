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

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.rewards.rynnvalue.configuration_rynnvalue import (  # noqa: E402
    RYNNVALUE_FEATURE_PREFIX,
    RynnValueConfig,
)
from lerobot.rewards.rynnvalue.processor_rynnvalue import (  # noqa: E402
    RynnValueEncoderProcessorStep,
    _uniform_subsample,
    _video_to_pil,
    make_rynnvalue_pre_post_processors,
)
from lerobot.rewards.rynnvalue.rynn_value_lang.processing_rynn_value_lang import (  # noqa: E402
    RynnValueLangProcessor,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.use_meta = False
        self.calls: list[dict] = []
        self.refresh_count = 0

    def refresh_conversation_builder(self):
        self.refresh_count += 1

    def process_episode(self, instruction, images, robot_description=None, camera_description=None):
        self.calls.append(
            {
                "instruction": instruction,
                "images": images,
                "robot_description": robot_description,
                "camera_description": camera_description,
            }
        )
        length = 4 + len(instruction.split())
        return {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "pixel_values": torch.zeros(1, len(images), 6),
            "image_grid_thw": torch.ones(1, len(images), 3, dtype=torch.long),
            "mm_token_type_ids": torch.zeros(1, length, dtype=torch.long),
        }


class _FakeConversationBuilder:
    def build_progress(self, **kwargs):  # noqa: ARG002
        return []


class _FakeNativeProcessor:
    conversation_builder = _FakeConversationBuilder()

    class _Tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):  # noqa: ARG004
            return 99

    tokenizer = _Tokenizer()

    @staticmethod
    def apply_chat_template(conversation):  # noqa: ARG004
        return "prompt"

    def __call__(self, text, images):  # noqa: ARG002
        return {
            "input_ids": torch.tensor([[1, 99, 2, 99, 3]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "mm_token_type_ids": torch.arange(5).unsqueeze(0),
            "pixel_values": torch.zeros(2, 6),
            "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
        }


def _patch_processor(monkeypatch) -> _FakeProcessor:
    from lerobot.rewards.rynnvalue import processor_rynnvalue

    fake = _FakeProcessor()
    monkeypatch.setattr(
        processor_rynnvalue.RynnValueLangProcessor,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: fake),
    )
    return fake


def test_uniform_subsample_keeps_history_boundaries():
    video = torch.arange(10)
    assert _uniform_subsample(video, 4).tolist() == [0, 3, 6, 9]
    assert torch.equal(_uniform_subsample(video, None), video)


def test_video_to_pil_accepts_float_chw_frames():
    frames = torch.zeros(5, 3, 8, 8)
    frames[-1, 0].fill_(1.0)
    images = _video_to_pil(frames, max_frames=3)
    assert len(images) == 3
    assert images[0].size == (8, 8)
    assert images[-1].getpixel((0, 0))[0] == 255


def test_native_processor_truncates_all_sequence_tensors_together():
    outputs = RynnValueLangProcessor.process_episode(
        _FakeNativeProcessor(),
        instruction="pick cube",
        images=[object(), object()],
    )
    assert outputs["input_ids"].shape == (1, 3)
    assert outputs["attention_mask"].shape == (1, 3)
    assert outputs["mm_token_type_ids"].shape == (1, 3)


def test_encoder_batches_and_pads_native_processor_outputs(monkeypatch):
    fake = _patch_processor(monkeypatch)
    encoder = RynnValueEncoderProcessorStep(
        model_id="test/rynnvalue",
        max_frames=3,
        robot_description="a test robot",
        camera_description="a front camera",
    )
    frames = torch.zeros(2, 5, 3, 8, 8)
    transition = {
        TransitionKey.OBSERVATION: {"observation.images.top": frames},
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick cube", "place the cube carefully"],
        },
    }
    encoded = encoder(transition)[TransitionKey.OBSERVATION]

    assert encoded[f"{RYNNVALUE_FEATURE_PREFIX}input_ids"].shape == (2, 8)
    assert encoded[f"{RYNNVALUE_FEATURE_PREFIX}attention_mask"].shape == (2, 8)
    assert encoded[f"{RYNNVALUE_FEATURE_PREFIX}mm_token_type_ids"].shape == (2, 8)
    assert encoded[f"{RYNNVALUE_FEATURE_PREFIX}pixel_values"].shape == (6, 6)
    assert encoded[f"{RYNNVALUE_FEATURE_PREFIX}image_grid_thw"].shape == (6, 3)
    assert [len(call["images"]) for call in fake.calls] == [3, 3]
    assert fake.calls[0]["robot_description"] == "a test robot"
    assert fake.calls[0]["camera_description"] == "a front camera"


def test_encoder_can_override_checkpoint_meta_setting(monkeypatch):
    fake = _patch_processor(monkeypatch)
    RynnValueEncoderProcessorStep(use_meta=True)
    assert fake.use_meta is True
    assert fake.refresh_count == 1


def test_converted_checkpoint_loads_processor_assets_from_lerobot_path(monkeypatch):
    from lerobot.rewards.rynnvalue import processor_rynnvalue

    captured = {}
    fake = _FakeProcessor()

    def from_pretrained(cls, model_id, **kwargs):  # noqa: ARG001
        captured["model_id"] = model_id
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(
        processor_rynnvalue.RynnValueLangProcessor,
        "from_pretrained",
        classmethod(from_pretrained),
    )
    config = RynnValueConfig(
        device="cpu",
        pretrained_path="lerobot/converted-rynnvalue",
        pretrained_revision="converted-revision",
        model_config={"model_type": "rynn_value_lang"},
    )
    make_rynnvalue_pre_post_processors(config)
    assert captured["model_id"] == "lerobot/converted-rynnvalue"
    assert captured["revision"] == "converted-revision"


def test_encoder_requires_task_or_default(monkeypatch):
    _patch_processor(monkeypatch)
    encoder = RynnValueEncoderProcessorStep()
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.top": torch.zeros(1, 2, 3, 8, 8),
        },
        TransitionKey.COMPLEMENTARY_DATA: {},
    }
    with pytest.raises(KeyError, match="task"):
        encoder(transition)
