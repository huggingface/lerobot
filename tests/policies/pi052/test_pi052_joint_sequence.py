#!/usr/bin/env python

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

"""Tests for PI052 joint-sequence (paper-style) subtask conditioning.

Joint recipes train the subtask text and the action losses in one sequence,
with the supervised subtask span attended causally. At inference the same
layout is rebuilt around the *generated* subtask, so these tests pin:

- the inference-side encoder produces the same token ids and target positions
  as the training-time tokenizer step for the same messages;
- OR-ing causal marks into a prefix reproduces the training-time attention
  pattern (prompt cannot see the subtask; subtask is causal over itself);
- the joint recipe file stays a valid message recipe;
- the FAST id mapping with the default ``fast_skip_tokens`` stays clear of
  PaliGemma's ``<loc>`` range so VQA and FAST supervision never collide.
"""

from pathlib import Path

import torch

from lerobot.configs.recipe import TrainingRecipe
from lerobot.policies.pi052.text_processor_pi052 import (
    PI052TextTokenizerStep,
    encode_prompt_with_targets,
)


class _CharTokenizer:
    """Char-level stub: 1 char = 1 token, so offsets are trivially aligned."""

    pad_token_id = 0
    eos_token = "\x1f"  # unit separator — a 1-char "EOS" for testing

    def __call__(self, text, max_length=None, padding=None, return_tensors=None, **kwargs):
        limit = max_length if max_length is not None else len(text)
        ids = [ord(c) % 251 + 1 for c in text[:limit]]
        offsets = [(i, i + 1) for i in range(len(ids))]
        attention = [1] * len(ids)
        if padding == "max_length" and max_length is not None and len(ids) < max_length:
            pad = max_length - len(ids)
            ids += [self.pad_token_id] * pad
            offsets += [(0, 0)] * pad
            attention += [0] * pad
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention], dtype=torch.long),
            "offset_mapping": torch.tensor([offsets], dtype=torch.long),
        }


_MESSAGES = [
    {"role": "user", "content": "fold the towel"},
    {"role": "assistant", "content": "grab the near corner"},
]


def test_encode_prompt_with_targets_matches_training_labels():
    tokenizer = _CharTokenizer()

    step = PI052TextTokenizerStep(max_length=120)
    step._tokenizer = tokenizer
    train_ids, train_attn, labels, predict_actions, _prompt = step._encode_messages(
        tokenizer,
        [dict(m) for m in _MESSAGES],
        message_streams=["low_level", "low_level"],
        target_indices=[1],
        complementary={},
    )
    assert bool(predict_actions)

    ids, attn, marks = encode_prompt_with_targets(tokenizer, [dict(m) for m in _MESSAGES], [1])

    n = int(attn.sum())
    assert n == int(train_attn.sum())
    assert torch.equal(ids[0, :n], train_ids[:n])
    # Causal marks at inference must cover exactly the supervised label span.
    assert torch.equal(marks[0, :n], labels[:n] != -100)
    assert marks.any(), "the assistant target span must be marked"
    # The user turn must stay unmarked (bidirectional prompt).
    user_len = len("User: fold the towel\n")
    assert not marks[0, :user_len].any()


def test_apply_causal_language_marks_reproduces_training_mask():
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from lerobot.policies.pi052.modeling_pi052 import _apply_causal_language_marks

    n_img, n_lang = 4, 8
    prefix_len = n_img + n_lang
    pad = torch.ones((1, prefix_len), dtype=torch.bool)
    att = torch.zeros((1, prefix_len), dtype=torch.bool)
    # Subtask span = language positions 5..7 (prefix positions 9..11).
    marks = torch.zeros((1, n_lang), dtype=torch.bool)
    marks[0, 5:8] = True

    att_marked = _apply_causal_language_marks(att, marks)
    att_2d = make_att_2d_masks(pad, att_marked)[0]

    subtask = [n_img + 5, n_img + 6, n_img + 7]
    # Prompt and images never see the subtask.
    for q in range(n_img + 5):
        for k in subtask:
            assert not att_2d[q, k], f"prompt position {q} must not attend subtask position {k}"
    # Subtask tokens see the full prompt and earlier subtask tokens only.
    for qi, q in enumerate(subtask):
        for k in range(n_img + 5):
            assert att_2d[q, k]
        for ki, k in enumerate(subtask):
            assert bool(att_2d[q, k]) == (ki <= qi)


def test_joint_recipe_is_a_valid_message_recipe():
    recipe_path = Path(__file__).parents[3] / "src" / "lerobot" / "configs" / "recipes" / "subtask_joint.yaml"
    recipe = TrainingRecipe.from_yaml(recipe_path)
    assert recipe.messages is not None and len(recipe.messages) == 2
    assert all(turn.stream == "low_level" for turn in recipe.messages)
    assert not recipe.messages[0].target
    assert recipe.messages[1].target
    assert recipe.messages[1].if_present == "subtask"


def test_default_fast_mapping_clears_loc_and_seg_ranges():
    from lerobot.policies.pi052.configuration_pi052 import PI052Config
    from lerobot.policies.pi052.modeling_pi052 import _FAST_ACTION_VOCAB_SIZE

    skip = PI052Config.__dataclass_fields__["fast_skip_tokens"].default
    assert skip == 1152

    paligemma_vocab = 257152
    fast_ids = paligemma_vocab - 1 - skip - torch.arange(_FAST_ACTION_VOCAB_SIZE)
    # Below the <loc> range [256000, 257024) and the <seg> range [257024, 257152).
    assert int(fast_ids.max()) < 256000
    assert int(fast_ids.min()) >= 0
