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

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

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


def _is_batched_messages(messages: Any) -> bool:
    return isinstance(messages, list) and bool(messages) and isinstance(messages[0], list)


def _sample_indices(value: Any, batch_size: int) -> list[int | None]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return [int(value.item())] * batch_size
        values = value.reshape(-1).tolist()
        return [int(v) for v in values[:batch_size]]
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _sample_indices(value[0], batch_size)
        return [int(v.item() if hasattr(v, "item") else v) for v in value[:batch_size]]
    return [int(value)] * batch_size


# ---------------------------------------------------------------------------
# VQA spatial answers → PaliGemma <loc> format (PI052 only)
#
# PaliGemma is pre-trained on detection / pointing with a ``<locNNNN>``
# vocabulary (normalized [0, 1023]). The recipe's bbox / keypoint VQA
# answers are stored as JSON in Qwen2.5-VL's grounding convention:
# **0–1000 normalized coordinates**, NOT pixels. (Verified empirically
# on the published datasets: x and y both span 0..1000 with ~30% of
# values exceeding the camera's pixel dimensions — they're not pixels.)
# Converting to ``<loc>`` is therefore camera-resolution-independent:
# ``loc_idx = round(coord / 1000 * 1023)``. We do the conversion here —
# not in the dataset — so the dataset stays backbone-agnostic (SmolVLA2
# keeps the JSON).
# ---------------------------------------------------------------------------

# The 0–1000 scale Qwen2.5-VL emits for grounding coordinates.
_VQA_COORD_SCALE = 1000.0


def register_paligemma_loc_tokens(tokenizer: Any) -> Any:
    """Make PaliGemma's ``<locDDDD>`` ids match on raw text — single tokens.

    PaliGemma reserves vocab ids [256000, 257023] for ``<locDDDD>``
    (detection / pointing) tokens, but the *stock* tokenizer does NOT
    match them when encoding raw text — it BPE-splits ``<loc0162>`` into
    7 pieces (``<``, ``loc``, ``0``, ``1``, ``6``, ``2``, ``>``). Training
    the LM head on a ``<loc>`` target then supervises those 7 generic
    BPE pieces instead of one detection-vocab id, the LM head learns to
    emit the *character sequence*, and those pieces' logits dominate
    other turns (the ``<loc>``-salad on subtasks). Registering the loc
    tokens once makes them tokenize as their single ids (256000+idx),
    leveraging PaliGemma's detection prior properly. Idempotent.
    """
    if "<loc0000>" in getattr(tokenizer, "added_tokens_encoder", {}):
        return tokenizer
    tokenizer.add_tokens([f"<loc{i:04d}>" for i in range(1024)])
    return tokenizer


def _loc_token(coord: float, scale: float = _VQA_COORD_SCALE) -> str:
    """PaliGemma ``<locNNNN>`` for a coord on a ``[0, scale]`` axis."""
    idx = round(float(coord) / scale * 1023) if scale > 0 else 0
    return f"<loc{max(0, min(1023, idx)):04d}>"


def _vqa_answer_to_loc(answer: dict[str, Any]) -> str | None:
    """Convert a bbox / keypoint VQA answer dict to PaliGemma ``<loc>`` text.

    Input coordinates are in Qwen2.5-VL's 0–1000 normalized space (see
    module-level note). PaliGemma convention: a point is
    ``<locY><locX> label``; a box is ``<locY0><locX0><locY1><locX1> label``
    (y before x, each index in [0, 1023]). Returns ``None`` for
    non-spatial answers (count / attribute / spatial-relation) — those
    keep their JSON form.
    """
    point = answer.get("point")
    if isinstance(point, list | tuple) and len(point) == 2 and "point_format" in answer:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        label = str(answer.get("label", "")).strip()
        return f"{_loc_token(y)}{_loc_token(x)} {label}".strip()

    detections = answer.get("detections")
    if isinstance(detections, list) and detections:
        parts: list[str] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            box = det.get("bbox")
            if not (isinstance(box, list | tuple) and len(box) == 4):
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            label = str(det.get("label", "")).strip()
            toks = (
                f"{_loc_token(y1)}{_loc_token(x1)}"
                f"{_loc_token(y2)}{_loc_token(x2)}"
            )
            parts.append(f"{toks} {label}".strip())
        return " ; ".join(parts) if parts else None
    return None


def _messages_vqa_to_loc(
    messages: list[dict[str, Any]],
    target_indices: list[int],
) -> list[dict[str, Any]]:
    """Rewrite bbox / keypoint VQA *target* answers from JSON to ``<loc>`` text.

    Each target turn whose content parses as a spatial VQA answer is
    converted. Non-spatial answers and subtask / memory targets (plain
    text → not JSON) are left untouched. Camera-independent: VQA coords
    are 0–1000 normalized, so no observation lookup is needed.
    """
    if not target_indices:
        return messages
    out = list(messages)
    for idx in target_indices:
        if not (0 <= idx < len(out)):
            continue
        content = out[idx].get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            answer = json.loads(content)
        except (ValueError, TypeError):
            continue  # subtask / memory targets are plain text — skip
        if not isinstance(answer, dict):
            continue
        loc_text = _vqa_answer_to_loc(answer)
        if loc_text is not None:
            out[idx] = {**out[idx], "content": loc_text}
    return out


def _format_messages(
    messages: list[dict[str, Any]],
    target_indices: list[int] | None = None,
    eos_token: str | None = None,
) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate messages into the π0.5-style flat prompt.

    When both ``target_indices`` and ``eos_token`` are given, the EOS
    string is appended to each supervised target turn's content and the
    returned span covers it — so the label builder marks the EOS token
    as a supervised label. That teaches the LM head where the answer
    *ends*: without an EOS in the target span the model is never given a
    stop signal and rambles to ``max_length`` at inference. Inference
    callers omit both args (no EOS baked into the prompt — the model
    generates it and ``select_message`` stops on it).

    Returns:
        prompt:       the full text the tokenizer will consume.
        msg_spans:    list of ``(char_start, char_end)`` covering each
                      message's supervised payload (content, plus the
                      appended EOS for target turns) within ``prompt``.
    """
    targets = set(target_indices or [])
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, m in enumerate(messages):
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        # Role tag + newline. The model has to learn to emit the same
        # role tokens at generation time, which is fine for greedy
        # decoding because the chat template is implicit in the
        # supervised target span.
        header = f"{role.capitalize()}: "
        # A supervised target turn ends with EOS so the model learns to
        # terminate; the span below covers content + EOS. Non-target
        # turns (and inference) carry no EOS.
        body = content + eos_token if (eos_token and i in targets) else content
        # span covers the content (+ EOS) portion only — never the role
        # tag — so labels are computed over the supervised payload.
        full = header + body + "\n"
        start = cursor + len(header)
        end = start + len(body)
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

        self._tokenizer = register_paligemma_loc_tokens(
            AutoTokenizer.from_pretrained(self.tokenizer_name)
        )
        return self._tokenizer

    # ------------------------------------------------------------------
    # Pipeline step
    # ------------------------------------------------------------------

    def __call__(self, transition: EnvTransition) -> EnvTransition | None:
        transition = transition.copy()
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        messages = complementary.get("messages") or []

        if not messages:
            # No recipe was rendered — caller will fall back to the
            # plain Pi0.5 prompt path. We pass the transition through
            # unmodified.
            return transition

        tokenizer = self._ensure_tokenizer()
        # VQA coords are 0–1000 normalized (Qwen2.5-VL convention) — the
        # <loc> conversion is camera-resolution-independent and needs no
        # observation lookup here.
        if _is_batched_messages(messages):
            indices_iter = _sample_indices(complementary.get("index"), len(messages))
            encoded = [
                self._encode_messages(
                    tokenizer,
                    msg,
                    list(streams),
                    list(tgt_indices),
                    complementary,
                    sample_idx=int(s_idx) if s_idx is not None else None,
                )
                for msg, streams, tgt_indices, s_idx in zip(
                    messages,
                    complementary.get("message_streams") or [[] for _ in messages],
                    complementary.get("target_message_indices") or [[] for _ in messages],
                    indices_iter,
                    strict=False,
                )
            ]
        else:
            sample_idx = _sample_indices(complementary.get("index"), 1)[0]
            encoded = [
                self._encode_messages(
                    tokenizer,
                    messages,
                    list(complementary.get("message_streams") or []),
                    list(complementary.get("target_message_indices") or []),
                    complementary,
                    sample_idx=sample_idx,
                )
            ]

        if _DUMP_BUDGET > 0:
            if _is_batched_messages(messages):
                msgs_iter = messages
                streams_iter = complementary.get("message_streams") or [[] for _ in messages]
                targets_iter = complementary.get("target_message_indices") or [[] for _ in messages]
            else:
                msgs_iter = [messages]
                streams_iter = [list(complementary.get("message_streams") or [])]
                targets_iter = [list(complementary.get("target_message_indices") or [])]
            for msg, streams, targets, (ids, attn, labels, predict_action, prompt) in zip(
                msgs_iter, streams_iter, targets_iter, encoded, strict=False
            ):
                target_set = {int(i) for i in targets}
                annotated_msgs = [
                    {
                        **m,
                        "stream": streams[i] if i < len(streams) else None,
                        "target": True if i in target_set else None,
                    }
                    for i, m in enumerate(msg)
                ]
                _dump_recipe_sample(
                    messages=annotated_msgs,
                    prompt_text=prompt,
                    token_ids=ids.tolist(),
                    labels=labels.tolist(),
                    predict_actions=bool(predict_action.item()),
                    tokenizer=tokenizer,
                )

        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        obs[OBS_LANGUAGE_TOKENS] = torch.stack([ids for ids, _, _, _, _ in encoded])
        obs[OBS_LANGUAGE_ATTENTION_MASK] = torch.stack([attn for _, attn, _, _, _ in encoded])
        transition[TransitionKey.OBSERVATION] = obs

        transition[TransitionKey.COMPLEMENTARY_DATA] = {
            **complementary,
            "text_labels": torch.stack([labels for _, _, labels, _, _ in encoded]),
            "predict_actions": torch.stack([pred for _, _, _, pred, _ in encoded]),
        }
        return transition

    def _encode_messages(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        message_streams: list[str | None],
        target_indices: list[int],
        complementary: dict[str, Any],
        sample_idx: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, str]:
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
                sample_idx=sample_idx,
            )

        # Rewrite bbox / keypoint VQA target answers from JSON to
        # PaliGemma <loc> text. Coords are 0–1000 normalized so this is
        # camera-independent.
        messages = _messages_vqa_to_loc(messages, target_indices)

        # Flatten ``say`` tool calls into ``<say>...</say>`` text before
        # stripping, so the spoken reply is actually tokenized and
        # supervised (PaliGemma's flat prompt has no structured calls).
        messages = [_strip_blocks(_flatten_say_tool_calls(m)) for m in messages]
        # Append EOS to supervised target turns so the LM head learns to
        # stop (the span covers it → it becomes a supervised label).
        prompt, spans = _format_messages(
            messages, target_indices, getattr(tokenizer, "eos_token", None)
        )

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
        return input_ids, attention_mask, labels, predict_actions, prompt

    # ------------------------------------------------------------------
    # Per-component prompt dropout (Pi0.7 §V.E)
    # ------------------------------------------------------------------

    def _apply_prompt_dropout(
        self,
        messages: list[dict[str, Any]],
        target_indices: list[int],
        complementary: dict[str, Any],
        sample_idx: int | None = None,
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
            seed_src = sample_idx if sample_idx is not None else complementary.get("index", 0)
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
