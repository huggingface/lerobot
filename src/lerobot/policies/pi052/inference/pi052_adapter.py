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

"""PI052 adapter for the generic language-conditioned runtime."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from lerobot.runtime import RuntimeState

logger = logging.getLogger(__name__)

_LOC_TOKENIZER_CACHE: dict[str, Any] = {}
_SAY_RE = re.compile(r"<\s*say\s*>(.*?)<\s*/\s*say\s*>", re.IGNORECASE | re.DOTALL)


@dataclass
class PI052PolicyAdapter:
    """Runtime bridge for PI052 policies."""

    policy: Any

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        subtask = state.language_context.get("subtask") or state.task or ""
        text_batch = _build_text_batch(
            self.policy,
            [{"role": "user", "content": subtask}],
            add_generation_prompt=False,
        )
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: PLC0415

        batch = dict(observation)
        batch[OBS_LANGUAGE_TOKENS] = text_batch["lang_tokens"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = text_batch["lang_masks"]
        return self.policy.predict_action_chunk(batch)

    def select_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        messages = self.messages_for(kind, state, user_text=user_text)
        return _generate_with_policy(
            self.policy,
            messages,
            observation=observation,
            state=state,
            label=f"{kind} gen",
            min_new_tokens=int(state.extra.get("text_gen_min_new_tokens") or 0),
            temperature=float(state.extra.get("text_gen_temperature") or 0.0),
            top_p=float(state.extra.get("text_top_p") or 1.0),
            suppress_loc_tokens=kind in {"subtask", "memory", "interjection"},
        )

    def plan_from_text(self, text: str) -> str:
        plan, _speech = split_plan_and_say(text)
        return "" if looks_like_gibberish(plan) else plan

    def update_language_state(self, observation: dict[str, Any] | None, state: RuntimeState) -> None:
        chunks_per_gen = max(1, int(state.extra.get("subtask_chunks_per_gen", 1) or 1))
        if "_hl_chunks_until_gen" not in state.extra:
            state.extra["_hl_chunks_until_gen"] = 0
        if state.extra["_hl_chunks_until_gen"] > 0:
            state.extra["_hl_chunks_until_gen"] -= 1
            return
        state.extra["_hl_chunks_until_gen"] = chunks_per_gen - 1

        msg = self.select_text("subtask", observation, state)
        state.extra["last_subtask_raw"] = msg or ""
        if not msg:
            empties = int(state.extra.get("subtask_empty_count") or 0) + 1
            state.extra["subtask_empty_count"] = empties
            if empties == 1 or empties % 5 == 0:
                debug = getattr(self.policy, "_last_select_message_debug", "") or ""
                state.log(
                    f"  [info] subtask gen empty (x{empties}); {debug}"
                    if debug
                    else f"  [info] subtask gen returned empty (x{empties})"
                )
            return
        if looks_like_gibberish(msg):
            count = int(state.extra.get("subtask_gibberish_count") or 0) + 1
            state.extra["subtask_gibberish_count"] = count
            if count == 1 or count % 30 == 0:
                state.log(f"  [info] subtask gen rejected (gibberish x{count}): {msg[:60]!r}")
            return

        previous = state.language_context.get("subtask")
        changed = state.set_context("subtask", msg, label="subtask")
        if not changed:
            state.extra["subtask_repeat_count"] = int(state.extra.get("subtask_repeat_count") or 0) + 1
            return

        state.extra["subtask_repeat_count"] = 0
        if previous:
            state.extra["prior_subtask"] = previous
        self._update_memory(observation, state)

    def _update_memory(self, observation: dict[str, Any] | None, state: RuntimeState) -> None:
        new_memory = self.select_text("memory", observation, state)
        state.extra["last_memory_raw"] = new_memory or ""
        if not new_memory:
            return
        if looks_like_gibberish(new_memory):
            count = int(state.extra.get("memory_gibberish_count") or 0) + 1
            state.extra["memory_gibberish_count"] = count
            state.log(f"  [info] memory gen rejected (gibberish x{count}): {new_memory[:60]!r}")
            return
        state.set_context("memory", new_memory, label="memory")

    def messages_for(
        self,
        kind: str,
        state: RuntimeState,
        *,
        user_text: str | None = None,
    ) -> list[dict[str, Any]]:
        if kind == "subtask":
            return [{"role": "user", "content": state.task or ""}]
        if kind == "memory":
            messages = [{"role": "user", "content": state.task or ""}]
            if state.language_context.get("memory"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"Previous memory: {state.language_context['memory']}",
                    }
                )
            if state.extra.get("prior_subtask"):
                messages.append(
                    {"role": "user", "content": f"Completed subtask: {state.extra['prior_subtask']}"}
                )
            return messages
        if kind == "interjection":
            messages = [{"role": "user", "content": state.task or ""}]
            if state.language_context.get("plan"):
                messages.append(
                    {"role": "assistant", "content": f"Previous plan:\n{state.language_context['plan']}"}
                )
            if user_text:
                messages.append({"role": "user", "content": user_text})
            return messages
        if kind == "plan":
            return [{"role": "user", "content": state.task or ""}]
        raise ValueError(f"Unknown PI052 text kind: {kind}")


def _get_loc_tokenizer(tok_name: str, auto_tokenizer_cls: Any, register_loc_fn: Any) -> Any:
    tokenizer = _LOC_TOKENIZER_CACHE.get(tok_name)
    if tokenizer is None:
        tokenizer = register_loc_fn(auto_tokenizer_cls.from_pretrained(tok_name))
        _LOC_TOKENIZER_CACHE[tok_name] = tokenizer
    return tokenizer


def _build_text_batch(
    policy: Any,
    prompt_messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from lerobot.policies.pi052.text_processor_pi052 import (  # noqa: PLC0415
        _flatten_say_tool_calls,
        _format_messages,
        _strip_blocks,
        register_paligemma_loc_tokens,
    )

    tok_name = getattr(policy.config, "tokenizer_name", None) or "google/paligemma-3b-pt-224"
    tokenizer = _get_loc_tokenizer(tok_name, AutoTokenizer, register_paligemma_loc_tokens)

    messages = [_strip_blocks(_flatten_say_tool_calls(m)) for m in prompt_messages]
    prompt, _spans = _format_messages(messages)
    if add_generation_prompt:
        prompt = prompt + "Assistant: "

    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"]
    attn = encoded.get("attention_mask")
    if attn is None and tokenizer.pad_token_id is not None:
        attn = ids != tokenizer.pad_token_id
    if attn is not None and hasattr(attn, "dtype") and attn.dtype != torch.bool:
        attn = attn.bool()

    device = getattr(getattr(policy, "config", None), "device", None)
    if device is not None:
        try:
            ids = ids.to(device)
            if attn is not None and hasattr(attn, "to"):
                attn = attn.to(device)
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not move pi052 lang tokens to %s: %s", device, exc)
    return {"lang_tokens": ids, "lang_masks": attn, "tokenizer": tokenizer}


def _generate_with_policy(
    policy: Any,
    messages: list[dict[str, Any]],
    *,
    observation: dict[str, Any] | None = None,
    state: RuntimeState | None = None,
    label: str = "select_message",
    min_new_tokens: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    suppress_loc_tokens: bool = False,
) -> str:
    if not hasattr(policy, "select_message"):
        if state is not None:
            state.log(f"  [warn] policy has no select_message — skipping {label}")
        return ""
    text_batch = _build_text_batch(policy, messages)
    try:
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: PLC0415

        batch: dict[str, Any] = {
            OBS_LANGUAGE_TOKENS: text_batch["lang_tokens"],
            OBS_LANGUAGE_ATTENTION_MASK: text_batch["lang_masks"],
        }
        if observation:
            for k, v in observation.items():
                if isinstance(k, str) and k.startswith("observation.") and k not in batch:
                    batch[k] = v
        return policy.select_message(
            batch,
            tokenizer=text_batch["tokenizer"],
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            suppress_loc_tokens=suppress_loc_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s failed: %s", label, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
        if state is not None:
            state.log(f"  [warn] {label} failed: {type(exc).__name__}: {exc}")
        return ""


def looks_like_gibberish(text: str) -> bool:
    if not text or not text.strip():
        return True
    stripped = text.strip()
    alpha = sum(1 for c in stripped if c.isalpha())
    if alpha < max(3, len(stripped) // 8):
        return True
    if stripped.startswith('":') and stripped.count('"') > stripped.count(" "):
        return True
    if len(set(stripped)) <= 2 and len(stripped) > 4:
        return True
    cleaned = stripped.replace("\n", " ").replace(":", " ")
    for marker in ("Assistant", "User", "Ass "):
        if marker in cleaned and len(cleaned.split()) < 4:
            return True
    tokens = [t for t in cleaned.split() if any(c.isalpha() for c in t)]
    unique_alpha = {t.lower() for t in tokens}
    if len(unique_alpha) < 3 and len(stripped) < 80:
        return True
    return len(tokens) >= 8 and len(unique_alpha) <= max(3, len(tokens) // 10)


def split_plan_and_say(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    match = _SAY_RE.search(text)
    if not match:
        return text.strip(), ""
    speech = match.group(1).strip().strip('"').strip("'")
    plan = (text[: match.start()] + text[match.end() :]).strip()
    return plan, speech
