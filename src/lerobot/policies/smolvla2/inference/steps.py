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
"""Inference steps for the SmolVLA2 multi-rate runtime.

Each step is a tiny class with a ``trigger`` and an ``__call__(state)``;
the runtime applies them in order each tick. When a step's trigger
doesn't fire, the step is a no-op and the runtime moves on.

Stream-to-step mapping mirrors the ``smolvla2_hirobot.yaml`` recipe:

* ``LowLevelForward``        — calls ``policy.select_action`` for the
                                action chunk; trained by
                                ``low_level_execution``
* ``EnqueueChunk``           — pushes the chunk to ``action_queue``
* ``DispatchAction``         — pops one action per control tick and
                                forwards to the robot
* ``HighLevelSubtaskFwd``    — calls ``policy.select_message`` for the
                                next subtask; trained by
                                ``high_level_subtask``
* ``MemoryUpdateFwd``        — fires on subtask boundary; trained by
                                ``memory_update``
* ``UserInterjectionFwd``    — fires on stdin interjection; trained by
                                ``user_interjection_response``
* ``AskVQAFwd``              — fires on stdin question; trained by
                                ``ask_vqa_*``
* ``DispatchToolCalls``      — pops ``tool_calls_pending`` and calls
                                the matching ``Tool`` instance
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .runtime_state import push_log, set_if_changed, take_event
from .triggers import EventTrigger, HzTrigger, Trigger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step base + runner
# ---------------------------------------------------------------------------


@dataclass
class InferenceStep:
    """A trigger-gated callable. Subclasses override :meth:`run`."""

    trigger: Trigger

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.trigger.should_fire(state["_tick"], state):
            return state
        return self.run(state) or state

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Low-level (action) path
# ---------------------------------------------------------------------------


@dataclass
class LowLevelForward(InferenceStep):
    """Run the policy's action head and produce one action chunk."""

    policy: Any = None
    observation_provider: Any = None
    """Callable ``() -> dict``: returns the current observation batch
    (already preprocessed). Typically wraps the robot's camera /
    proprio reads. ``None`` in dry-run mode → step skips."""

    trigger: Trigger = field(default_factory=lambda: HzTrigger(hz=4.0))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if self.policy is None or self.observation_provider is None:
            return None
        if not state.get("task"):
            # No task yet → nothing useful to condition on.
            return None
        observation = self.observation_provider()
        if observation is None:
            return None
        # SmolVLA's ``select_action`` expects the full preprocessed
        # batch, including ``OBS_LANGUAGE_TOKENS`` /
        # ``OBS_LANGUAGE_ATTENTION_MASK``. The observation provider
        # only returns image / state features (the runtime drives
        # messages itself), so build a low-level prompt from current
        # runtime state and tokenize it inline.
        ctx = _control_context_messages(state)
        if state.get("current_subtask"):
            ctx = ctx + [{"role": "assistant", "content": state["current_subtask"]}]
        text_batch = _build_text_batch(self.policy, ctx)
        from lerobot.utils.constants import (  # noqa: PLC0415
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        observation = dict(observation)
        observation[OBS_LANGUAGE_TOKENS] = text_batch["lang_tokens"]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = text_batch["lang_masks"]
        try:
            action = self.policy.select_action(observation)
        except Exception as exc:  # noqa: BLE001
            logger.warning("select_action failed: %s", exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            push_log(state, f"  [warn] select_action failed: {type(exc).__name__}: {exc}")
            return None
        # SmolVLA returns a single action; if the underlying policy
        # streams chunks, split per-step here. For v1 we just enqueue
        # the result.
        state.setdefault("action_queue", []).append(action)
        return None


@dataclass
class DispatchAction(InferenceStep):
    """Pop one action per tick and hand it to the robot.

    In dry-run mode (``robot_executor=None``) the step still pops the
    queue so it doesn't grow unbounded — the popped tensor is logged
    instead of executed.
    """

    robot_executor: Any = None
    trigger: Trigger = field(default_factory=lambda: HzTrigger(hz=50.0))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        queue = state.get("action_queue")
        if not queue:
            return None
        action = queue.popleft() if hasattr(queue, "popleft") else queue.pop(0)
        if self.robot_executor is not None:
            self.robot_executor(action)
        return None


# ---------------------------------------------------------------------------
# High-level (text) paths — all use policy.select_message
# ---------------------------------------------------------------------------


def _build_text_batch(policy: Any, prompt_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokenize a list of chat messages into the batch shape
    ``select_message`` expects.

    Lazy fallback: re-uses the policy's preprocessor by piggy-backing
    on the chat tokenizer step. Production use should construct the
    batch from a real observation; here we focus on the *language*
    path which is independent of camera observations.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    text_messages = [_strip_recipe_keys(m) for m in prompt_messages]
    # SmolVLM's chat template iterates ``message['content']`` expecting
    # a list of typed blocks (``[{type: 'text', text: ...}, ...]``).
    # When ``content`` is a plain ``str`` it silently iterates characters,
    # no branch matches, and *no content tokens are emitted* — the model
    # receives only role markers and starts hallucinating ``Assistant:``
    # fragments. Coerce string content to the list-of-blocks form the
    # template expects.
    for _m in text_messages:
        _c = _m.get("content")
        if isinstance(_c, str):
            _m["content"] = [{"type": "text", "text": _c}]
    encoded = tokenizer.apply_chat_template(
        text_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    # ``apply_chat_template`` can return any of:
    #   - a Tensor of shape ``(seq,)`` or ``(1, seq)`` (older transformers),
    #   - a list[int] / list[list[int]] (when ``return_tensors`` is ignored),
    #   - a ``BatchEncoding`` dict-like with ``input_ids`` / ``attention_mask``
    #     (newer transformers, especially via processor.apply_chat_template
    #     forwarding through here).
    # Normalise to ``ids: Tensor[1, seq]`` and grab the encoder's
    # attention mask when available so we don't have to re-derive it
    # from ``pad_token_id`` (which can be ``None`` for SmolVLM).
    attn: Any = None
    if hasattr(encoded, "input_ids"):
        ids = encoded.input_ids
        attn = getattr(encoded, "attention_mask", None)
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        ids = encoded["input_ids"]
        attn = encoded.get("attention_mask")
    else:
        ids = encoded
    if isinstance(ids, list):
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        import torch  # noqa: PLC0415

        ids = torch.tensor(ids, dtype=torch.long)
    if hasattr(ids, "ndim") and ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if attn is None and tokenizer.pad_token_id is not None:
        attn = ids != tokenizer.pad_token_id
    elif isinstance(attn, list):
        import torch  # noqa: PLC0415

        attn = torch.tensor(attn, dtype=torch.long)
        if attn.ndim == 1:
            attn = attn.unsqueeze(0)
    # SmolVLA's ``eager_attention_forward`` does
    # ``torch.where(attention_mask[..., None, :, :], ...)`` which
    # requires a *bool* condition tensor; ``BatchEncoding``'s
    # attention_mask is typically Long (0/1). Cast so the prefix
    # forward doesn't blow up with ``where expected condition to be a
    # boolean tensor, but got a tensor with dtype Long``.
    if attn is not None and hasattr(attn, "dtype"):
        import torch as _torch  # noqa: PLC0415

        if attn.dtype != _torch.bool:
            attn = attn.bool()
    # Move tokens onto the policy's device — otherwise prefix embedding
    # raises a device-mismatch on every forward (CPU tensor vs MPS / CUDA
    # model), which the caller's broad except would swallow silently.
    device = getattr(getattr(policy, "config", None), "device", None)
    if device is not None:
        try:
            ids = ids.to(device)
            if attn is not None and hasattr(attn, "to"):
                attn = attn.to(device)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not move lang tokens to %s: %s", device, exc)
    return {"lang_tokens": ids, "lang_masks": attn, "tokenizer": tokenizer}


def _strip_recipe_keys(m: dict[str, Any]) -> dict[str, Any]:
    new = dict(m)
    new.pop("stream", None)
    new.pop("target", None)
    return new


@dataclass
class HighLevelSubtaskFwd(InferenceStep):
    """At ~1 Hz, ask the policy for the next subtask.

    Mirrors the ``high_level_subtask`` recipe layout exactly:

        user:   "${task}\\nPlan: ${plan}\\nMemory: ${memory}"
        user:   "Current subtask: ${subtask}"        (if subtask present)
        ↓ generate ↓
        assistant: <next subtask>
    """

    policy: Any = None
    observation_provider: Any = None
    """Same shape as ``LowLevelForward.observation_provider``. When
    set, the resulting observation is merged into ``select_message``'s
    batch so text generation runs against real video + state."""

    trigger: Trigger = field(default_factory=lambda: HzTrigger(hz=1.0))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if self.policy is None or not state.get("task"):
            return None
        ctx = _msgs_for_subtask(state)
        observation = _maybe_observation(self.observation_provider)
        msg = _generate_with_policy(
            self.policy, ctx, observation=observation, state=state, label="subtask gen"
        )
        if msg and _looks_like_gibberish(msg):
            push_log(state, f"  [info] subtask gen rejected (gibberish): {msg[:60]!r}")
            return None
        if msg:
            changed = set_if_changed(state, "current_subtask", msg, label="subtask")
            if changed:
                # Subtask change is a downstream trigger.
                state.setdefault("events_this_tick", []).append("subtask_change")
        else:
            push_log(state, "  [info] subtask gen produced no text this tick")
        return None


@dataclass
class MemoryUpdateFwd(InferenceStep):
    """On subtask boundary, refresh the compressed memory.

    Mirrors the ``memory_update`` recipe layout exactly:

        user:      "${task}"
        assistant: "Previous memory: ${prior_memory}"   (if prior memory)
        user:      "Completed subtask: ${completed_subtask}"  (if subtask)
        ↓ generate ↓
        assistant: <new memory>
    """

    policy: Any = None
    observation_provider: Any = None
    trigger: Trigger = field(default_factory=lambda: EventTrigger("subtask_change"))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        # Don't consume the event — multiple steps may want to react.
        if self.policy is None:
            return None
        ctx = _msgs_for_memory(state)
        observation = _maybe_observation(self.observation_provider)
        new_memory = _generate_with_policy(
            self.policy, ctx, observation=observation, state=state, label="memory gen"
        )
        if new_memory and _looks_like_gibberish(new_memory):
            push_log(state, f"  [info] memory gen rejected (gibberish): {new_memory[:60]!r}")
            return None
        if new_memory:
            set_if_changed(state, "current_memory", new_memory, label="memory")
        return None


@dataclass
class UserInterjectionFwd(InferenceStep):
    """On stdin interjection, refresh the plan + emit a paired ``say``.

    Mirrors the ``user_interjection_response`` recipe layout exactly:

        user:      "${task}"
        assistant: "Previous plan:\\n${prior_plan}"   (if prior plan)
        user:      "${interjection}"                  (the new utterance)
        ↓ generate ↓
        assistant: <plan + <say>...</say>>
    """

    policy: Any = None
    observation_provider: Any = None
    trigger: Trigger = field(default_factory=lambda: EventTrigger("user_interjection"))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if self.policy is None or not take_event(state, "user_interjection"):
            return None
        ctx = _msgs_for_interjection(state)
        observation = _maybe_observation(self.observation_provider)
        out = _generate_with_policy(
            self.policy, ctx, observation=observation, state=state, label="plan/say gen"
        )
        if not out:
            push_log(state, "  [info] plan/say gen produced no text this tick")
            return None
        if _looks_like_gibberish(out):
            push_log(state, f"  [info] plan/say gen rejected (gibberish): {out[:60]!r}")
            return None
        # Heuristic split: model is trained to emit one assistant turn
        # carrying both plan text AND a `say` tool call. Look for a
        # "<say>...</say>" or "say(...)" marker; fall back to whole
        # text → plan, no speech.
        plan_text, speech_text = _split_plan_and_say(out)
        if plan_text and _looks_like_gibberish(plan_text):
            plan_text = ""
        if plan_text:
            set_if_changed(state, "current_plan", plan_text, label="plan")
        if speech_text:
            push_log(state, f"  speech: {speech_text}")
            state.setdefault("tool_calls_pending", []).append(
                {
                    "type": "function",
                    "function": {"name": "say", "arguments": {"text": speech_text}},
                }
            )
            state.setdefault("events_this_tick", []).append("tool_call_pending")
        # Mark interjection consumed.
        state["recent_interjection"] = None
        return None


@dataclass
class AskVQAFwd(InferenceStep):
    """On stdin question, answer a frame-grounded VQA.

    Mirrors the ``ask_vqa_*`` recipe layout exactly: a single user
    turn carrying just the VQA question, plus the camera image block
    in training (we drop the image at inference because the dataset's
    image preprocessing doesn't match SmolVLM's vision tower input).

        user:   <question>
        ↓ generate ↓
        assistant: <vqa answer>
    """

    policy: Any = None
    observation_provider: Any = None
    trigger: Trigger = field(default_factory=lambda: EventTrigger("user_vqa_query"))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if self.policy is None or not take_event(state, "user_vqa_query"):
            return None
        question = state.get("recent_vqa_query")
        if not question:
            return None
        ctx = _msgs_for_vqa(question)
        observation = _maybe_observation(self.observation_provider)
        answer = _generate_with_policy(
            self.policy, ctx, observation=observation, state=state, label="vqa gen"
        )
        # VQA answers are intentionally JSON-like during training, so
        # ``_looks_like_gibberish`` would false-positive on them. Keep
        # the answer as-is — the VQA panel line lets the user judge.
        if answer:
            push_log(state, f"  vqa: {answer}")
        state["recent_vqa_query"] = None
        return None


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


@dataclass
class DispatchToolCalls(InferenceStep):
    """Pop ``tool_calls_pending`` and execute them via :data:`TOOL_REGISTRY`."""

    tools: dict[str, Any] = field(default_factory=dict)
    trigger: Trigger = field(default_factory=lambda: EventTrigger("tool_call_pending"))

    def run(self, state: dict[str, Any]) -> dict[str, Any] | None:
        take_event(state, "tool_call_pending")
        pending = state.get("tool_calls_pending") or []
        for call in pending:
            try:
                fn = (call or {}).get("function") or {}
                name = fn.get("name")
                args = fn.get("arguments") or {}
                tool = self.tools.get(name)
                if tool is None:
                    push_log(state, f"  [warn] tool {name!r} not registered — skipping call")
                    continue
                tool.call(args)
            except Exception as exc:  # noqa: BLE001
                push_log(state, f"  [error] tool dispatch failed: {exc}")
        state["tool_calls_pending"] = []
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_gibberish(text: str) -> bool:
    """Heuristically detect generation that's clearly off the rails.

    Memorised models can collapse to dominant-mode outputs (often the
    JSON-token salad ``":":":":...`` from VQA training) when the prompt
    drifts even slightly from training distribution. If we accept those
    as new state, they pollute the next tick's prompt and cascade into
    worse outputs. Reject anything that looks pathological:

    * empty / whitespace-only
    * mostly punctuation (``"``, ``:``, ``,``)
    * a single character repeated past the threshold
    * starts with ``":"`` and contains no letters

    The thresholds are intentionally lenient — a real subtask like
    ``"close the gripper"`` has ~70%+ alpha characters, while gibberish
    like ``":":":"`` has ~0%.
    """
    if not text or not text.strip():
        return True
    stripped = text.strip()
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha < max(3, len(stripped) // 8):
        return True
    if stripped.startswith('":') and stripped.count('"') > stripped.count(" "):
        return True
    # Single repeating char: e.g. ``""""""``
    if len(set(stripped)) <= 2 and len(stripped) > 4:
        return True
    return False


def _control_context_messages(
    state: dict[str, Any],
    *,
    include_completed: bool = False,
    extra_user: str | None = None,
) -> list[dict[str, Any]]:
    """Build a chat-template-ready prompt from current runtime state.

    Mirrors what ``smolvla2_hirobot.yaml`` renders into ``${task}\nPlan:
    ${plan}\nMemory: ${memory}`` for the high-level branches.
    """
    parts: list[str] = []
    task = state.get("task") or ""
    parts.append(task)
    if state.get("current_plan"):
        parts.append(f"Plan: {state['current_plan']}")
    if state.get("current_memory"):
        parts.append(f"Memory: {state['current_memory']}")
    if include_completed and state.get("current_subtask"):
        parts.append(f"Completed subtask: {state['current_subtask']}")
    head = "\n".join(parts)
    msgs: list[dict[str, Any]] = [{"role": "user", "content": head}]
    if extra_user:
        msgs.append({"role": "user", "content": extra_user})
    return msgs


# ---------------------------------------------------------------------------
# Per-recipe prompt builders. Each one mirrors a single sub-recipe's
# message layout in ``smolvla2_hirobot.yaml`` so the chat-templated
# prompt at inference matches what the model saw during training.
# Generic ``_control_context_messages`` is kept around as a fallback
# for ad-hoc callers but the four high-level steps now use these.
# ---------------------------------------------------------------------------


def _msgs_for_subtask(state: dict[str, Any]) -> list[dict[str, Any]]:
    """``high_level_subtask`` recipe layout."""
    head_parts = [state.get("task") or ""]
    if state.get("current_plan"):
        head_parts.append(f"Plan: {state['current_plan']}")
    if state.get("current_memory"):
        head_parts.append(f"Memory: {state['current_memory']}")
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": "\n".join(head_parts)}
    ]
    if state.get("current_subtask"):
        msgs.append(
            {"role": "user", "content": f"Current subtask: {state['current_subtask']}"}
        )
    return msgs


def _msgs_for_memory(state: dict[str, Any]) -> list[dict[str, Any]]:
    """``memory_update`` recipe layout."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": state.get("task") or ""}
    ]
    if state.get("current_memory"):
        msgs.append(
            {
                "role": "assistant",
                "content": f"Previous memory: {state['current_memory']}",
            }
        )
    if state.get("current_subtask"):
        msgs.append(
            {
                "role": "user",
                "content": f"Completed subtask: {state['current_subtask']}",
            }
        )
    return msgs


def _msgs_for_interjection(state: dict[str, Any]) -> list[dict[str, Any]]:
    """``user_interjection_response`` recipe layout."""
    msgs: list[dict[str, Any]] = [
        {"role": "user", "content": state.get("task") or ""}
    ]
    if state.get("current_plan"):
        msgs.append(
            {"role": "assistant", "content": f"Previous plan:\n{state['current_plan']}"}
        )
    interjection = state.get("recent_interjection")
    if interjection:
        msgs.append({"role": "user", "content": interjection})
    return msgs


def _msgs_for_vqa(question: str) -> list[dict[str, Any]]:
    """``ask_vqa_*`` recipe layout (text-only at inference)."""
    return [{"role": "user", "content": question}]


def _maybe_observation(provider: Any) -> dict | None:
    """Pull one observation from ``provider`` if it's set, else ``None``.

    Errors from the provider are logged at debug level and swallowed —
    text generation still runs (in text-only mode) so a flaky frame
    source doesn't kill the REPL.
    """
    if provider is None:
        return None
    try:
        return provider()
    except Exception as exc:  # noqa: BLE001
        logger.debug("observation_provider raised %s — falling back to text-only", exc)
        return None


def _generate_with_policy(
    policy: Any,
    messages: list[dict[str, Any]],
    *,
    observation: dict | None = None,
    state: dict[str, Any] | None = None,
    label: str = "select_message",
) -> str:
    """Drive ``policy.select_message`` with a chat batch (and optional obs).

    When ``observation`` carries ``observation.images.*`` and
    ``observation.state``, those are merged into the batch so
    ``select_message`` runs the same VLM prefix the policy was trained
    on. Without an observation the runtime falls back to a text-only
    prompt — the text head still runs, but generations may drift from
    the training distribution.

    Failures are surfaced both to the module logger (``warning``) and,
    when ``state`` is given, to the runtime's user-visible log via
    :func:`push_log`, so the REPL no longer "looks dead" when
    something goes wrong inside generation.
    """
    if not hasattr(policy, "select_message"):
        if state is not None:
            push_log(state, f"  [warn] policy has no select_message — skipping {label}")
        return ""
    text_batch = _build_text_batch(policy, messages)
    try:
        from lerobot.utils.constants import (  # noqa: PLC0415
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        batch: dict[str, Any] = {
            OBS_LANGUAGE_TOKENS: text_batch["lang_tokens"],
            OBS_LANGUAGE_ATTENTION_MASK: text_batch["lang_masks"],
        }
        if observation:
            for k, v in observation.items():
                if isinstance(k, str) and k.startswith("observation.") and k not in batch:
                    batch[k] = v
        return policy.select_message(batch, tokenizer=text_batch["tokenizer"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s failed: %s", label, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        if state is not None:
            push_log(state, f"  [warn] {label} failed: {type(exc).__name__}: {exc}")
        return ""


_SAY_RE = re.compile(r"<\s*say\s*>(.*?)<\s*/\s*say\s*>", re.IGNORECASE | re.DOTALL)


def _split_plan_and_say(text: str) -> tuple[str, str]:
    """Pull a ``<say>...</say>`` snippet out of ``text``; remainder is plan.

    The training-time tool-call serializer wraps ``say(text="…")`` in a
    deterministic textual marker so prefix-LM-style training learns to
    emit it. The runtime parses it back here. If no marker is present,
    the entire text is treated as plan with no speech.
    """
    if not text:
        return "", ""
    match = _SAY_RE.search(text)
    if not match:
        return text.strip(), ""
    speech = match.group(1).strip().strip('"').strip("'")
    plan = (text[: match.start()] + text[match.end() :]).strip()
    return plan, speech
