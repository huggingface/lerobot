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

"""π0.5 v2 text-tokenisation step.

PaliGemma is *not* chat-pretrained, so unlike SmolVLA2 we can't lean on
``tokenizer.apply_chat_template``. Instead we concatenate the rendered
messages as plain text with simple ``User: ... Assistant: ...`` role
delimiters — matching the prompt format π0.5 uses in the paper
(``Task: ... State: ... Action: ...``).

Outputs:

* ``OBS_LANGUAGE_TOKENS`` / ``OBS_LANGUAGE_ATTENTION_MASK`` — the
  concatenated prompt tokenised by the PaliGemma tokenizer (the same
  one ``processor_pi05`` already uses).
* ``text_labels`` — same shape as token ids, ``-100`` everywhere except
  positions belonging to messages whose index is in
  ``target_message_indices``. ``modeling_pi052`` runs cross-entropy on
  those positions via the PaliGemma ``lm_head``.
* ``predict_actions`` — bool tensor, ``True`` iff any of the rendered
  target messages has ``message_streams[i] == "low_level"``. Same
  semantics as the SmolVLA2 step.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

logger = logging.getLogger(__name__)


def _strip_blocks(message: dict[str, Any]) -> dict[str, Any]:
    """Normalise a message's content to a plain string.

    The recipe renderer can emit ``content`` as a string OR as a list
    of HF-style multimodal blocks (``{type: text, text: ...}``,
    ``{type: image, feature: ...}``). PaliGemma's text tokenizer can
    only consume strings, so we flatten: drop image blocks (cameras
    flow through ``observation.images.*`` separately) and join text
    block texts.
    """
    new = dict(message)
    new.pop("stream", None)
    new.pop("target", None)
    content = new.get("content")
    if content is None:
        new["content"] = ""
    elif isinstance(content, str):
        pass
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                t = block.get("text", "")
                if isinstance(t, str):
                    parts.append(t)
        new["content"] = "\n".join(parts)
    else:
        new["content"] = str(content)
    return new


def _format_messages(messages: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate messages into the π0.5-style flat prompt.

    Returns:
        prompt:       the full text the tokenizer will consume.
        msg_spans:    list of ``(char_start, char_end)`` covering each
                      message's content within ``prompt``. The
                      target-mask builder uses this to find the
                      character ranges belonging to the supervised
                      messages.
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        # Role tag + newline. The model has to learn to emit the same
        # role tokens at generation time, which is fine for greedy
        # decoding because the chat template is implicit in the
        # supervised target span.
        header = f"{role.capitalize()}: "
        # span covers ONLY the content portion (so labels are computed
        # over the supervised payload, not the role tag).
        full = header + content + "\n"
        start = cursor + len(header)
        end = start + len(content)
        parts.append(full)
        spans.append((start, end))
        cursor += len(full)
    return "".join(parts), spans


@dataclass
@ProcessorStepRegistry.register(name="pi052_text_tokenizer")
class PI052TextTokenizerStep(ProcessorStep):
    """Render messages → token ids + label mask + predict_actions flag.

    π0.5 analogue of ``SmolVLA2ChatTokenizerStep``. No chat template;
    concatenates messages as ``User: ... \\nAssistant: ...`` text.
    """

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    max_length: int = 200
    padding: str = "max_length"
    padding_side: str = "right"
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0
    interjection_dropout_prob: float = 0.0
    dropout_seed: int | None = None

    def __post_init__(self) -> None:
        self._tokenizer: Any = None

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer

    # ------------------------------------------------------------------
    # Pipeline step
    # ------------------------------------------------------------------

    def __call__(self, transition: EnvTransition) -> EnvTransition | None:
        transition = transition.copy()
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        messages = complementary.get("messages") or []
        target_indices = list(complementary.get("target_message_indices") or [])
        message_streams = list(complementary.get("message_streams") or [])

        if not messages:
            # No recipe was rendered — caller will fall back to the
            # plain Pi0.5 prompt path. We pass the transition through
            # unmodified.
            return transition

        # Optional: drop non-target messages per the dropout config.
        # Keeps the supervised-target indices stable by re-mapping
        # after removal.
        if (
            self.plan_dropout_prob
            or self.memory_dropout_prob
            or self.subtask_dropout_prob
            or self.interjection_dropout_prob
        ):
            messages, target_indices = self._apply_prompt_dropout(
                messages,
                target_indices,
                complementary,
            )

        messages = [_strip_blocks(m) for m in messages]
        prompt, spans = _format_messages(messages)

        tokenizer = self._ensure_tokenizer()
        encoded = tokenizer(
            prompt,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding_side=self.padding_side,
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].bool()
        offsets = encoded["offset_mapping"][0]  # (seq, 2), char (start,end)

        # Build label mask: -100 everywhere except over supervised
        # target message char ranges.
        labels = torch.full_like(input_ids, fill_value=-100)
        for idx in target_indices:
            if idx >= len(spans):
                continue
            char_start, char_end = spans[idx]
            for token_pos in range(input_ids.shape[0]):
                if not attention_mask[token_pos]:
                    continue
                tok_start, tok_end = int(offsets[token_pos, 0]), int(offsets[token_pos, 1])
                if tok_end <= char_start or tok_start >= char_end:
                    continue
                labels[token_pos] = input_ids[token_pos]

        predict_actions = torch.tensor(
            bool(any(message_streams[i] == "low_level" for i in target_indices if i < len(message_streams))),
            dtype=torch.bool,
        )

        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        obs[OBS_LANGUAGE_TOKENS] = input_ids.unsqueeze(0)
        obs[OBS_LANGUAGE_ATTENTION_MASK] = attention_mask.unsqueeze(0)
        transition[TransitionKey.OBSERVATION] = obs

        transition[TransitionKey.COMPLEMENTARY_DATA] = {
            **complementary,
            "text_labels": labels.unsqueeze(0),
            "predict_actions": predict_actions.unsqueeze(0),
        }
        return transition

    # ------------------------------------------------------------------
    # Per-component prompt dropout (Pi0.7 §V.E)
    # ------------------------------------------------------------------

    def _apply_prompt_dropout(
        self,
        messages: list[dict[str, Any]],
        target_indices: list[int],
        complementary: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Drop messages classified as plan/memory/subtask context.

        Targets are *never* dropped (they're the supervised payload).
        Re-maps target_indices to the new positions after drops.
        """
        import random  # noqa: PLC0415

        seed = self.dropout_seed
        if seed is None:
            seed_src = complementary.get("dataset_index") or complementary.get("frame_index") or 0
            try:
                seed = int(seed_src)
            except (TypeError, ValueError):
                seed = 0
        rng = random.Random(seed)

        keep_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if idx in target_indices:
                keep_indices.append(idx)
                continue
            kind = _classify_for_dropout(msg)
            prob = {
                "plan": self.plan_dropout_prob,
                "memory": self.memory_dropout_prob,
                "subtask": self.subtask_dropout_prob,
                "interjection": self.interjection_dropout_prob,
            }.get(kind, 0.0)
            if prob > 0.0 and rng.random() < prob:
                continue
            keep_indices.append(idx)

        # Build remap and apply
        new_messages = [messages[i] for i in keep_indices]
        old_to_new = {old: new for new, old in enumerate(keep_indices)}
        new_targets = [old_to_new[t] for t in target_indices if t in old_to_new]
        return new_messages, new_targets

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _classify_for_dropout(message: dict[str, Any]) -> str | None:
    """Heuristic content-prefix classifier — mirrors SmolVLA2's."""
    content = message.get("content")
    if isinstance(content, list):
        text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        content = " ".join(text_parts)
    elif content is None:
        return None
    elif not isinstance(content, str):
        return None
    s = content.strip()
    if s.startswith("Plan:") or s.startswith("Previous plan"):
        return "plan"
    if s.startswith("Memory:") or s.startswith("Previous memory"):
        return "memory"
    if s.startswith("Current subtask") or s.startswith("Completed subtask"):
        return "subtask"
    return None
