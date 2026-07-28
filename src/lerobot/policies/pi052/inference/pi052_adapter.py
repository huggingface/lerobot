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

"""PI052 actions and text generation for the generic language runtime."""

from __future__ import annotations

import logging
from typing import Any

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter

logger = logging.getLogger(__name__)

_LOC_TOKENIZER_CACHE: dict[str, Any] = {}


class PI052PolicyAdapter(BaseLanguageAdapter):
    """Runtime bridge for PI052 policies."""

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        import torch  # noqa: PLC0415

        from lerobot.utils.constants import (  # noqa: PLC0415
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
            OBS_STATE,
        )

        subtask = state.language_context.get("subtask") or state.task or ""
        # Match the training prompt by conditioning on both subtask and discretized state.
        state_str = None
        obs_state = observation.get(OBS_STATE)
        if isinstance(obs_state, torch.Tensor) and obs_state.numel() > 0:
            from lerobot.policies.pi052.text_processor_pi052 import discretize_state_str  # noqa: PLC0415

            state_row = obs_state[0] if obs_state.ndim > 1 else obs_state
            state_str = discretize_state_str(state_row)

        batch = dict(observation)
        if getattr(self.policy.config, "joint_subtask_conditioning", False):
            # Joint sequences keep the task turn (with state) and render the
            # subtask as a causal assistant turn, exactly as trained.
            from transformers import AutoTokenizer  # noqa: PLC0415

            from lerobot.policies.pi052.text_processor_pi052 import (  # noqa: PLC0415
                encode_prompt_with_targets,
                register_paligemma_loc_tokens,
            )
            from lerobot.utils.constants import OBS_LANGUAGE_CAUSAL_MARKS  # noqa: PLC0415

            task = state.task or ""
            task_content = task if state_str is None else f"{task}, State: {state_str};"
            tok_name = getattr(self.policy.config, "tokenizer_name", None) or "google/paligemma-3b-pt-224"
            tokenizer = _get_loc_tokenizer(tok_name, AutoTokenizer, register_paligemma_loc_tokens)
            ids, attn, marks = encode_prompt_with_targets(
                tokenizer,
                [
                    {"role": "user", "content": task_content},
                    {"role": "assistant", "content": subtask},
                ],
                target_indices=[1],
            )
            device = getattr(self.policy.config, "device", None)
            if device is not None:
                ids, attn, marks = ids.to(device), attn.to(device), marks.to(device)
            batch[OBS_LANGUAGE_TOKENS] = ids
            batch[OBS_LANGUAGE_ATTENTION_MASK] = attn
            batch[OBS_LANGUAGE_CAUSAL_MARKS] = marks
        else:
            content = subtask if state_str is None else f"{subtask}, State: {state_str};"
            text_batch = _build_text_batch(
                self.policy,
                [{"role": "user", "content": content}],
                add_generation_prompt=False,
            )
            batch[OBS_LANGUAGE_TOKENS] = text_batch["lang_tokens"]
            batch[OBS_LANGUAGE_ATTENTION_MASK] = text_batch["lang_masks"]
        return self.policy.predict_action_chunk(batch)

    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        messages = self.build_messages(kind, state, user_text=user_text)
        if kind == "subtask" and getattr(self.policy.config, "joint_subtask_conditioning", False):
            # Joint samples carry state on the task turn, so the subtask must be
            # generated from the same state-bearing prompt.
            import torch  # noqa: PLC0415

            from lerobot.policies.pi052.text_processor_pi052 import discretize_state_str  # noqa: PLC0415
            from lerobot.utils.constants import OBS_STATE  # noqa: PLC0415

            obs_state = (observation or {}).get(OBS_STATE)
            if isinstance(obs_state, torch.Tensor) and obs_state.numel() > 0:
                state_row = obs_state[0] if obs_state.ndim > 1 else obs_state
                for m in reversed(messages):
                    if m.get("role") == "user":
                        m["content"] = f"{m.get('content', '')}, State: {discretize_state_str(state_row)};"
                        break
        return _generate_with_policy(
            self.policy,
            messages,
            observation=observation,
            state=state,
            label=f"{kind} gen",
            min_new_tokens=self.gen.min_new_tokens,
            temperature=self.gen.temperature,
            top_p=self.gen.top_p,
            suppress_loc_tokens=True,  # all runtime text is prose; never emit <loc>
        )

    def build_messages(
        self,
        kind: str,
        state: RuntimeState,
        *,
        user_text: str | None = None,
    ) -> list[dict[str, Any]]:
        if kind in ("subtask", "plan"):
            return [{"role": "user", "content": state.task or ""}]
        if kind == "memory":
            messages = [{"role": "user", "content": state.task or ""}]
            if state.language_context.get("memory"):
                messages.append(
                    {"role": "assistant", "content": f"Previous memory: {state.language_context['memory']}"}
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
        # No trailing space: SentencePiece folds it into the first target token
        # ("▁move"), so a space-suffixed prefill ends in a lone "▁" the model
        # never saw at this position during training.
        prompt = prompt + "Assistant:"

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
