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

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debug helper — see ``chat_processor_smolvla2._dump_recipe_sample`` for the
# matching SmolVLA2 implementation. Behaviour: when
# ``LEROBOT_DUMP_RECIPE_SAMPLES=N`` is set, the next N samples processed (on
# rank 0) are pretty-printed with ``[TGT]...[/TGT]`` markers over the spans
# the LM head will be supervised on.
# ---------------------------------------------------------------------------

_DUMP_BUDGET = int(os.environ.get("LEROBOT_DUMP_RECIPE_SAMPLES", "0"))
_DUMPED_SO_FAR = 0


def _is_dump_rank() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _dump_recipe_sample(
    *,
    messages: list[dict[str, Any]],
    prompt_text: str,
    token_ids: list[int],
    labels: list[int],
    predict_actions: bool,
    tokenizer: Any,
) -> None:
    """Pretty-print one rendered sample. Stops once the global budget is hit."""
    global _DUMPED_SO_FAR
    if _DUMPED_SO_FAR >= _DUMP_BUDGET or not _is_dump_rank():
        return
    _DUMPED_SO_FAR += 1

    parts: list[str] = []
    i = 0
    while i < len(labels):
        if labels[i] == -100:
            j = i
            while j < len(labels) and labels[j] == -100:
                j += 1
            parts.append(tokenizer.decode(token_ids[i:j], skip_special_tokens=False))
            i = j
        else:
            j = i
            while j < len(labels) and labels[j] != -100:
                j += 1
            tgt_text = tokenizer.decode(token_ids[i:j], skip_special_tokens=False)
            parts.append(f"[TGT]{tgt_text}[/TGT]")
            i = j
    annotated = "".join(parts)

    n_tgt = sum(1 for l in labels if l != -100)
    print(
        "\n========== RECIPE SAMPLE DUMP "
        f"({_DUMPED_SO_FAR}/{_DUMP_BUDGET}) ==========",
        flush=True,
    )
    print(f"  predict_actions: {predict_actions}", flush=True)
    print(f"  rendered messages ({len(messages)}):", flush=True)
    for m in messages:
        stream = m.get("stream")
        target = m.get("target")
        role = m.get("role")
        content = m.get("content")
        print(f"    - role={role} stream={stream} target={target}", flush=True)
        print(f"      content: {content!r}", flush=True)
    print(f"  rendered prompt:\n    {prompt_text!r}", flush=True)
    print(f"  token count: {len(token_ids)} (target tokens: {n_tgt})", flush=True)
    print(f"  decoded (with target markers):\n    {annotated}", flush=True)
    print("==============================================\n", flush=True)


def _content_to_text(content: Any) -> str:
    """Collapse a message's ``content`` (string or multimodal blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        return "\n".join(parts)
    return ""


def _flatten_say_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Serialize assistant ``say`` tool calls into a ``<say>...</say>`` marker.

    PaliGemma's flat text prompt has no notion of structured tool calls,
    and ``_format_messages`` only reads ``role`` / ``content`` — so
    without this a ``say`` tool call is dropped entirely and never
    supervised. Rewriting it into the content text as a ``<say>...</say>``
    marker lets the LM head learn to emit it; the runtime parses it back
    via ``_split_plan_and_say``. Messages without ``say`` tool calls are
    returned unchanged (the structured calls, if any, are still dropped).
    """
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return message
    say_texts: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if fn.get("name") != "say":
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                import json  # noqa: PLC0415

                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        text = args.get("text", "") if isinstance(args, dict) else ""
        if text:
            say_texts.append(str(text))
    new = dict(message)
    new.pop("tool_calls", None)
    if not say_texts:
        return new
    base = _content_to_text(new.get("content")).strip()
    marker = "".join(f"<say>{t}</say>" for t in say_texts)
    new["content"] = f"{base}\n{marker}" if base else marker
    return new


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

        # Flatten ``say`` tool calls into ``<say>...</say>`` text before
        # stripping, so the spoken reply is actually tokenized and
        # supervised (PaliGemma's flat prompt has no structured calls).
        messages = [_strip_blocks(_flatten_say_tool_calls(m)) for m in messages]
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

        # Scan ALL message streams (not just targets) — see
        # ``chat_processor_smolvla2.py`` for rationale: the v2
        # ``low_level_execution`` recipe drops ``target: true`` on
        # the assistant to avoid trivial copy-from-user text-CE; the
        # flow loss still needs to fire, gated by ``stream: low_level``.
        predict_actions = torch.tensor(
            bool(any(s == "low_level" for s in message_streams)),
            dtype=torch.bool,
        )

        if _DUMP_BUDGET > 0:
            # Stream / target metadata live in parallel arrays; zip them
            # back into the dicts so the dump shows them per message.
            target_set = {int(i) for i in target_indices}
            annotated_msgs = [
                {
                    **m,
                    "stream": message_streams[i] if i < len(message_streams) else None,
                    "target": True if i in target_set else None,
                }
                for i, m in enumerate(messages)
            ]
            _dump_recipe_sample(
                messages=annotated_msgs,
                prompt_text=prompt,
                token_ids=input_ids.tolist(),
                labels=labels.tolist(),
                predict_actions=bool(predict_actions.item()),
                tokenizer=tokenizer,
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
            # Canonical row-index key set by ``BatchProcessor`` /
            # ``render_messages_processor``. Falling back to other
            # keys silently gave every sample seed=0 → identical
            # dropout pattern across the whole epoch.
            seed_src = complementary.get("index", 0)
            try:
                if hasattr(seed_src, "item"):
                    seed_src = seed_src.item()
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
