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
"""SmolVLA2's chat-template tokenization step.

Replaces SmolVLA's plain ``TokenizerProcessorStep`` for SmolVLA2 when a
``recipe_path`` is set. Reads the rendered messages produced by
``RenderMessagesStep`` (PR 1) and produces:

* ``OBS_LANGUAGE_TOKENS`` / ``OBS_LANGUAGE_ATTENTION_MASK`` —
  the chat-templated prompt tokenized by SmolVLM's tokenizer, with
  ``tools=meta.tools`` (PR 1's catalog).
* ``text_labels`` — same shape as token ids, ``-100`` everywhere except
  the positions belonging to messages whose index is in
  ``target_message_indices``. The next commit's modeling forward path
  applies cross-entropy on those positions via the SmolVLM ``lm_head``.
* ``predict_actions`` — bool tensor, ``True`` iff any of the rendered
  target messages has ``message_streams[i] == "low_level"``. The
  modeling forward uses this to gate the flow head.

Image / video content blocks in the rendered messages are dropped
before tokenization — the chat template only handles text, and SmolVLA
already passes camera tensors out-of-band via the standard
``OBS_IMAGES_*`` features. This keeps the prefix layout unchanged
(``embed_prefix`` puts image embeddings before language embeddings,
matching the chat-template-stripped text order).
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
# Debug helper: dump the first N rendered samples to stdout so you can sanity-
# check what the model actually sees before kicking off a long training run.
#
#   LEROBOT_DUMP_RECIPE_SAMPLES=5 lerobot-train ...
#
# Prints the recipe-rendered messages, the chat-templated text (decoded back
# from token ids), and inline ``[TGT]...[/TGT]`` markers showing which spans
# are supervised by text-CE. Stops after N total dumps to keep training logs
# tractable. Rank-0 only when accelerate sets ``RANK``.
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

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
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
    print(f"  token count: {len(token_ids)} (target tokens: {n_tgt})", flush=True)
    print(f"  decoded (raw):\n    {decoded}", flush=True)
    print(f"  decoded (with target markers):\n    {annotated}", flush=True)
    print("==============================================\n", flush=True)


@dataclass
@ProcessorStepRegistry.register(name="smolvla2_chat_tokenizer")
class SmolVLA2ChatTokenizerStep(ProcessorStep):
    """Render messages → token ids + label mask + predict_actions flag.

    This is the bridge between the recipe stack (PR 1's
    ``RenderMessagesStep`` outputs) and the SmolVLA2 modeling forward
    (next commit, which reads ``text_labels`` / ``predict_actions``).
    Pure-text turns and multi-stream targets are both handled.
    """

    tokenizer_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    max_length: int = 2048
    padding: str = "longest"
    padding_side: str = "right"
    tools: list[dict[str, Any]] | None = None
    # --- Per-component prompt dropout (Pi0.7 §V.E, plan follow-up
    # ``feat/pi05-prompt-dropout``). At training, drop non-target
    # messages whose content was substituted from the named recipe
    # binding with the given probability. Forces the model to handle
    # missing context at inference — directly attacks the memorisation
    # collapse where ``current_subtask=""`` puts the prompt OOD. All
    # default to 0.0 (no dropout) so behaviour is identical until
    # explicitly opted in via the training config.
    plan_dropout_prob: float = 0.0
    memory_dropout_prob: float = 0.0
    subtask_dropout_prob: float = 0.0
    interjection_dropout_prob: float = 0.0
    # Optional seed for the per-sample RNG. ``None`` ⇒ use
    # ``sample_idx`` derived from the transition (when present), so
    # dropout is reproducible across runs but varies per sample.
    dropout_seed: int | None = None

    def __post_init__(self) -> None:
        # Lazy: don't load the tokenizer until the step actually runs,
        # so unit tests that import the module without transformers
        # installed still pass.
        self._tokenizer: Any = None
        if self.tools is None:
            # Default: no tools rendered into the system prompt. The
            # ``say()`` tool was only used by the now-removed
            # ``user_interjection_response`` recipe; including its
            # schema on every sample adds a long system message to
            # the action expert's prefix and creates a train/inference
            # mismatch (the inference low-level loop doesn't pass
            # tools=, so the chat template doesn't render them).
            # Users who actually need tools can set them via
            # ``with_tools(meta.tools)``.
            self.tools = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def with_tools(self, tools: list[dict[str, Any]]) -> "SmolVLA2ChatTokenizerStep":
        """Override the tools catalog rendered into the system prompt."""
        self.tools = list(tools)
        return self

    def __call__(self, transition: EnvTransition) -> EnvTransition | None:
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        messages = comp.get("messages")
        if not messages:
            # No recipe rendering happened — nothing to do; downstream
            # falls back to whatever ``task`` is in the transition.
            return transition

        tokenizer = self._get_tokenizer()

        # Pull a sample_idx for the dropout RNG. ``index`` is the
        # canonical per-frame key on ``LeRobotDataset`` samples and
        # flows through into ``COMPLEMENTARY_DATA`` unchanged. When
        # absent (e.g. inference) we fall back to 0 which is harmless
        # because the dropout probs are also 0 at inference time.
        if _is_batched_messages(messages):
            indices_iter = _sample_indices(comp.get("index"), len(messages))
            encoded = [
                self._encode_messages(
                    tokenizer,
                    msg,
                    list(streams),
                    sorted(int(i) for i in tgt_indices),
                    sample_idx=int(s_idx) if s_idx is not None else None,
                )
                for msg, streams, tgt_indices, s_idx in zip(
                    messages,
                    comp.get("message_streams") or [[] for _ in messages],
                    comp.get("target_message_indices") or [[] for _ in messages],
                    indices_iter,
                    strict=False,
                )
            ]
        else:
            sample_idx = _sample_indices(comp.get("index"), 1)[0]
            encoded = [
                self._encode_messages(
                    tokenizer,
                    messages,
                    list(comp.get("message_streams") or []),
                    sorted(int(i) for i in (comp.get("target_message_indices") or [])),
                    sample_idx=sample_idx,
                )
            ]

        # Optional first-N-samples debug dump for sanity-checking what the
        # model actually sees. No-op unless ``LEROBOT_DUMP_RECIPE_SAMPLES``
        # is set; stops globally after the budget is exhausted.
        if _DUMP_BUDGET > 0:
            # Stream / target metadata live in parallel arrays in
            # COMPLEMENTARY_DATA, not on the message dicts themselves
            # (the recipe renderer keeps them separate so the chat
            # template doesn't choke on unknown keys). Zip them back
            # together for the dumper so each printed message shows
            # its actual stream + target flag.
            if _is_batched_messages(messages):
                msgs_iter = messages
                streams_iter = comp.get("message_streams") or [[] for _ in messages]
                targets_iter = comp.get("target_message_indices") or [[] for _ in messages]
            else:
                msgs_iter = [messages]
                streams_iter = [list(comp.get("message_streams") or [])]
                targets_iter = [list(comp.get("target_message_indices") or [])]
            for msg, streams, targets, (ids, labels, predict_action) in zip(
                msgs_iter, streams_iter, targets_iter, encoded, strict=False
            ):
                target_set = {int(i) for i in targets}
                annotated_msgs = []
                for i, m in enumerate(msg):
                    annotated_msgs.append(
                        {
                            **m,
                            "stream": streams[i] if i < len(streams) else None,
                            "target": True if i in target_set else None,
                        }
                    )
                _dump_recipe_sample(
                    messages=annotated_msgs,
                    token_ids=ids,
                    labels=labels,
                    predict_actions=predict_action,
                    tokenizer=tokenizer,
                )

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        target_length = self.max_length if self.padding == "max_length" else max(
            len(ids) for ids, _, _ in encoded
        )
        target_length = min(target_length, self.max_length)

        ids_batch = []
        attn_batch = []
        labels_batch = []
        predict_actions = []
        for ids, labels, predict_action in encoded:
            ids = ids[:target_length]
            labels = labels[:target_length]
            attn = [1] * len(ids)
            if len(ids) < target_length:
                n_pad = target_length - len(ids)
                ids = ids + [pad_id] * n_pad
                labels = labels + [-100] * n_pad
                attn = attn + [0] * n_pad
            ids_batch.append(ids)
            attn_batch.append(attn)
            labels_batch.append(labels)
            predict_actions.append(predict_action)

        ids_t = torch.tensor(ids_batch, dtype=torch.long)
        attn_t = torch.tensor(attn_batch, dtype=torch.bool)
        labels_t = torch.tensor(labels_batch, dtype=torch.long)
        predict_actions_t = torch.tensor(predict_actions, dtype=torch.bool)

        if not _is_batched_messages(messages):
            ids_t = ids_t.squeeze(0)
            attn_t = attn_t.squeeze(0)
            labels_t = labels_t.squeeze(0)
            predict_actions_t = predict_actions_t.squeeze(0)

        new_complementary = dict(comp)
        # Drop the per-recipe sidecar keys; everything downstream needs
        # is now in the tokenized form.
        new_complementary.pop("messages", None)
        new_complementary.pop("message_streams", None)
        new_complementary.pop("target_message_indices", None)
        # SmolVLA's pipeline expects ``OBS_LANGUAGE_TOKENS`` /
        # ``OBS_LANGUAGE_ATTENTION_MASK`` on the OBSERVATION key. Place
        # them there — and drop ``task`` so the upstream
        # ``TokenizerProcessorStep`` (which we replace) doesn't double-
        # tokenize.
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        observation[OBS_LANGUAGE_TOKENS] = ids_t
        observation[OBS_LANGUAGE_ATTENTION_MASK] = attn_t
        new_complementary["text_labels"] = labels_t
        new_complementary["predict_actions"] = predict_actions_t
        new_complementary.pop("task", None)

        new_transition = dict(transition)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary
        new_transition[TransitionKey.OBSERVATION] = observation
        return new_transition

    def _encode_messages(
        self,
        tokenizer: Any,
        messages: list[dict[str, Any]],
        message_streams: list[str | None],
        target_indices: list[int],
        sample_idx: int | None = None,
    ) -> tuple[list[int], list[int], bool]:
        # Apply per-component prompt dropout *before* tokenisation, so
        # the dropped messages don't contribute tokens or label-mask
        # positions at all. Re-maps ``target_indices`` to account for
        # removed messages.
        messages, target_indices = self._apply_prompt_dropout(
            messages, target_indices, sample_idx
        )
        # Flatten ``tool_calls`` into a textual ``<say>...</say>`` marker
        # *before* the chat template sees them, so the model is trained
        # to emit the same marker the inference parser
        # (``_split_plan_and_say``) reads back. See ``_flatten_say_tool_calls``.
        messages = [_flatten_say_tool_calls(m) for m in messages]
        text_messages = [_strip_lerobot_blocks(m) for m in messages]

        full_ids = tokenizer.apply_chat_template(
            text_messages,
            tools=self.tools,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
        )
        full_ids = _as_token_ids(full_ids)

        labels = [-100] * len(full_ids)
        for tgt in target_indices:
            prefix_ids = tokenizer.apply_chat_template(
                text_messages[:tgt],
                tools=self.tools,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors=None,
            )
            full_through_target = tokenizer.apply_chat_template(
                text_messages[: tgt + 1],
                tools=self.tools,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors=None,
            )
            prefix_ids = _as_token_ids(prefix_ids)
            full_through_target = _as_token_ids(full_through_target)
            start = len(prefix_ids)
            end = min(len(full_through_target), len(full_ids))
            for pos in range(start, end):
                labels[pos] = int(full_ids[pos])

        # ``predict_actions`` is True iff this sample's recipe declares
        # at least one ``low_level`` message — regardless of whether
        # it's a target. The ``low_level_execution`` recipe in v2 uses
        # ``stream: low_level`` on both user and assistant turns but
        # only renders the *user* subtask (no text-CE target on the
        # assistant) to avoid trivial "copy previous turn" supervision.
        # Scanning targets alone would miss this sample's action loss.
        predict_actions = any(s == "low_level" for s in message_streams)
        return [int(i) for i in full_ids], labels, predict_actions

    def _apply_prompt_dropout(
        self,
        messages: list[dict[str, Any]],
        target_indices: list[int],
        sample_idx: int | None,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Probabilistically drop non-target context messages.

        Heuristic content sniffing — matches the prefix strings that
        ``subtask_mem_vqa_speech.yaml``'s recipes use when injecting plan /
        memory / subtask / interjection content. Anything else is
        kept unchanged. Target messages are never dropped (we still
        need their tokens for supervision).

        Returns ``(new_messages, new_target_indices)`` where the
        indices are re-mapped to point at the same target turns in
        the trimmed list.
        """
        probs = {
            "plan": float(self.plan_dropout_prob or 0.0),
            "memory": float(self.memory_dropout_prob or 0.0),
            "subtask": float(self.subtask_dropout_prob or 0.0),
            "interjection": float(self.interjection_dropout_prob or 0.0),
        }
        if not any(p > 0.0 for p in probs.values()):
            return messages, target_indices

        # Deterministic per-sample RNG so dropout is reproducible
        # across runs (matters for debugging / repro) but varies
        # frame-to-frame.
        import random  # noqa: PLC0415

        seed_int = self.dropout_seed if self.dropout_seed is not None else (sample_idx or 0)
        rng = random.Random(int(seed_int) & 0xFFFFFFFF)

        target_set = set(target_indices)
        keep_flags: list[bool] = []
        for i, msg in enumerate(messages):
            if i in target_set:
                keep_flags.append(True)
                continue
            kind = _classify_message_for_dropout(msg)
            if kind and rng.random() < probs.get(kind, 0.0):
                keep_flags.append(False)
            else:
                keep_flags.append(True)

        new_messages = [m for m, keep in zip(messages, keep_flags) if keep]
        # Re-map target_indices: each old index drops by the count of
        # falsy flags before it.
        new_target_indices: list[int] = []
        for old_idx in target_indices:
            dropped_before = sum(1 for k in keep_flags[:old_idx] if not k)
            new_target_indices.append(old_idx - dropped_before)
        return new_messages, sorted(new_target_indices)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass-through; this step writes runtime tensors not features."""
        return features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self):  # noqa: ANN202
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SmolVLA2ChatTokenizerStep requires transformers. "
                "`pip install lerobot[transformers-dep]`."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer


def _strip_lerobot_blocks(message: dict[str, Any]) -> dict[str, Any]:
    """Remove LeRobot-specific multimodal blocks from ``message`` content.

    The recipe DSL allows authors to write multimodal content like
    ``{"type": "image", "feature": "observation.images.top"}``. SmolVLM's
    tokenizer doesn't know that ``feature`` key (it expects ``url`` or
    ``path``). The actual image tensor flows through SmolVLA's
    ``OBS_IMAGES_*`` channels separately; the chat template only needs
    the text. So we strip non-text blocks before tokenizing.
    """
    new = dict(message)
    content = new.get("content")
    if isinstance(content, list):
        text_parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append({"type": "text", "text": str(block.get("text", ""))})
        new["content"] = text_parts or [{"type": "text", "text": ""}]
    elif content is None:
        new["content"] = [{"type": "text", "text": ""}]
    else:
        new["content"] = [{"type": "text", "text": str(content)}]
    if "tool_calls" in new and not new["tool_calls"]:
        # Drop empty tool_calls — some templates render them as a
        # spurious empty marker.
        new.pop("tool_calls")
    # ``stream`` and ``target`` were recipe metadata; templates don't
    # know them and may warn or crash.
    new.pop("stream", None)
    new.pop("target", None)
    return new


def _content_to_text(content: Any) -> str:
    """Collapse a message's ``content`` (string or multimodal blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _flatten_say_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Serialize assistant ``say`` tool calls into a textual ``<say>...</say>``
    marker inside the message content (Pi 0.5-style flat tool-call
    serialization).

    SmolVLM's chat template would otherwise render ``tool_calls`` as a
    structured JSON ``<tool_call>`` block, so the LM head learns to emit
    JSON — but the inference parser ``_split_plan_and_say`` looks for a
    ``<say>...</say>`` marker (``_SAY_RE``). Rewriting the call into the
    content text *before* ``apply_chat_template`` aligns the two: the
    template only ever tokenizes plain text, and the supervised target
    span trains the model to produce the exact marker the runtime reads.

    Messages without ``say`` tool calls are returned unchanged.
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

    if not say_texts:
        # No ``say`` calls (or empty text) — drop the structured calls so
        # the template doesn't render a stray JSON block, but leave the
        # content alone.
        new = dict(message)
        new.pop("tool_calls", None)
        return new

    new = dict(message)
    base = _content_to_text(new.get("content")).strip()
    marker = "".join(f"<say>{t}</say>" for t in say_texts)
    new["content"] = f"{base}\n{marker}" if base else marker
    new.pop("tool_calls", None)
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


def _classify_message_for_dropout(message: dict[str, Any]) -> str | None:
    """Best-effort classification of which recipe binding contributed
    to this message, used for per-component dropout.

    The canonical recipe authors plan/memory/subtask injections with
    distinctive prefix strings in the rendered content. Matching on
    those prefixes is brittle if a future recipe author uses
    different wording — but it's also localised to one place and
    only affects the dropout fraction (never the actual semantics).
    Returns ``None`` for messages we don't recognise; those are
    always kept.
    """
    content = message.get("content")
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        content = "\n".join(text_parts)
    if not isinstance(content, str):
        return None
    head = content.lstrip().lower()
    if head.startswith("plan:") or head.startswith("previous plan"):
        return "plan"
    if head.startswith("memory:") or head.startswith("previous memory"):
        return "memory"
    if head.startswith("current subtask") or head.startswith("completed subtask"):
        return "subtask"
    return None


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, dict) or (hasattr(value, "keys") and "input_ids" in value.keys()):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(i) for i in value]


# Re-export for tests / introspection
strip_lerobot_blocks = _strip_lerobot_blocks
flatten_say_tool_calls = _flatten_say_tool_calls
