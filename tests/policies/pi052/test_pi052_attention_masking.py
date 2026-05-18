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

"""Attention-masking tests for the PI052 (π0.5 v2) text head.

Regression coverage for the text-CE collapse bug: PaliGemma's
``embed_prefix`` flags every language token ``att=0``, which
``make_att_2d_masks`` turns into one fully *bidirectional* block. Under
that mask the text cross-entropy degenerates into a copy task — a
supervised target token attends to the tokens it is trained to predict —
and the LM head never learns causal generation, so ``select_message``
collapses at inference.

``_mark_target_span_causal`` sets ``att=1`` on the supervised target
language positions so each target token attends causally among the
targets while staying bidirectional to images + the user prompt. These
tests pin that behaviour for the PaliGemma prefix layout.
"""

import pytest
import torch

# modeling_pi052 / modeling_pi05 import transformers transitively.
pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks  # noqa: E402
from lerobot.policies.pi052.modeling_pi052 import _mark_target_span_causal  # noqa: E402

# ---------------------------------------------------------------------------
# A synthetic PI052 prefix layout: [images, prompt-lang, target-lang]
#
#   indices 0-1  : 2 image tokens          (att = 0)
#   indices 2-4  : 3 user-prompt lang      (att = 0)
#   indices 5-8  : 4 supervised target lang(att = 0 from embed_prefix)
#
# ``text_labels`` covers the 7 language tokens; -100 on the prompt span,
# real ids on the 4-token target span. PaliGemma's prefix has no state
# token (unlike SmolVLA), so the lang span ends at the prefix end.
# ---------------------------------------------------------------------------
N_IMAGE = 2
N_PROMPT = 3
N_TARGET = 4
LANG_START = N_IMAGE
LANG_END = N_IMAGE + N_PROMPT + N_TARGET  # = prefix length
PREFIX_LEN = LANG_END


def _embed_prefix_att_masks() -> torch.Tensor:
    """Mimic PaliGemma ``embed_prefix``: images + lang all att=0."""
    return torch.zeros(1, PREFIX_LEN, dtype=torch.bool)


def _text_labels() -> torch.Tensor:
    """-100 over the prompt span, real ids over the target span."""
    labels = torch.full((1, N_PROMPT + N_TARGET), -100, dtype=torch.long)
    labels[0, N_PROMPT:] = torch.arange(10, 10 + N_TARGET)
    return labels


def _attends(prefix_att_masks: torch.Tensor) -> torch.Tensor:
    """2D boolean attendance matrix; ``[i, j]`` True ⇒ i attends to j."""
    pad = torch.ones(1, PREFIX_LEN, dtype=torch.bool)
    return make_att_2d_masks(pad, prefix_att_masks)[0]


def test_mark_sets_att_on_targets_only():
    """Only the supervised target language positions flip to att=1."""
    marked = _mark_target_span_causal(
        _embed_prefix_att_masks(), _text_labels(), LANG_START, LANG_END
    )
    expected = [False] * PREFIX_LEN
    for i in range(LANG_START + N_PROMPT, LANG_END):  # target span
        expected[i] = True
    assert marked[0].tolist() == expected


def test_target_tokens_attend_causally_among_themselves():
    """A target token must NOT attend to later targets, but must attend
    to earlier ones — genuine causal next-token prediction."""
    marked = _mark_target_span_causal(
        _embed_prefix_att_masks(), _text_labels(), LANG_START, LANG_END
    )
    attends = _attends(marked)
    tgt = range(LANG_START + N_PROMPT, LANG_END)
    for i in tgt:
        for j in tgt:
            if j > i:
                assert not attends[i, j], f"target {i} must not see future target {j}"
            else:
                assert attends[i, j], f"target {i} must see earlier/self target {j}"


def test_target_tokens_attend_prompt_and_images_bidirectionally():
    """Targets keep full visibility of images + the user prompt."""
    marked = _mark_target_span_causal(
        _embed_prefix_att_masks(), _text_labels(), LANG_START, LANG_END
    )
    attends = _attends(marked)
    context = list(range(0, LANG_START + N_PROMPT))  # images + prompt
    for i in range(LANG_START + N_PROMPT, LANG_END):
        for j in context:
            assert attends[i, j], f"target {i} must attend context {j}"


def test_non_target_subtask_stays_bidirectional():
    """A flow-only / non-target language span (all -100 labels) leaves the
    mask untouched — the action expert reads it bidirectionally."""
    all_ignored = torch.full((1, N_PROMPT + N_TARGET), -100, dtype=torch.long)
    marked = _mark_target_span_causal(
        _embed_prefix_att_masks(), all_ignored, LANG_START, LANG_END
    )
    assert torch.equal(marked, _embed_prefix_att_masks())


def test_unmarked_mask_is_bidirectional_the_bug():
    """Documents the bug the fix prevents: without ``_mark_target_span_causal``
    a target token attends *bidirectionally* to later targets — the
    text-CE can copy the answer it is trained to predict."""
    attends = _attends(_embed_prefix_att_masks())
    first_tgt = LANG_START + N_PROMPT
    last_tgt = LANG_END - 1
    assert attends[first_tgt, last_tgt], (
        "raw embed_prefix mask is bidirectional over language — the first "
        "target token can see the last, which is the collapse bug"
    )
