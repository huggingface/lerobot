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

"""OpenGalaxea G0.5 bridge for the generic language runtime.

G0.5 generates optional reasoning and an action from one shared inference
stream.  This adapter deliberately does not run a second planner: System 2 is
the ``cot_text`` produced by the same call whose action chunk System 1
executes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter, GenerationConfig

_SUBTASK_RE = re.compile(
    r"(?:^|[\n|])\s*Subtask\s*:\s*(.+?)(?=(?:[\n|]\s*(?:Updated\s+Memory|Memory|Action|Plan)\s*:)|$)",
    re.IGNORECASE | re.DOTALL,
)
_MEMORY_RE = re.compile(
    r"(?:^|[\n|])\s*(?:Updated\s+Memory|Memory)\s*:\s*(.+?)(?=(?:[\n|]\s*(?:Subtask|Action|Plan)\s*:)|$)",
    re.IGNORECASE | re.DOTALL,
)
_MODE_ALIASES = {
    "auto": "auto",
    "system1": "system1",
    "system_1": "system1",
    "direct": "system1",
    "system2": "system2",
    "system_2": "system2",
    "hierarchical": "system2",
}
_TEXT_KEYS = ("cot_text", "generated_cot", "reasoning")


class G05PolicyAdapter(BaseLanguageAdapter):
    """Execute G0.5's unified reasoning/action stream in the language runtime.

    The preferred policy hook is::

        predict_action_chunk_with_runtime(observation, *, task, system_mode) ->
            (action_chunk, {"cot_text": ...})

    ``task`` is exactly :attr:`RuntimeState.task`, including whitespace and
    Unicode.  The policy must pass it to the author prompt builder unchanged
    before applying checkpoint-specific formatting. ``system_mode`` selects
    action-only System 1 or unified CoT-plus-action System 2 for that call. A
    structured return is required for batch-safe CoT handling;
    ``predict_action_chunk`` remains a compatibility fallback for System 1.
    """

    def __init__(
        self,
        policy: Any,
        gen: GenerationConfig | None = None,
        *,
        system_mode: str | None = None,
    ) -> None:
        super().__init__(policy, gen)
        self.system_mode = self._resolve_system_mode(system_mode)
        self._validate_checkpoint_capabilities()

    def _resolve_system_mode(self, requested: str | None) -> str:
        config = getattr(self.policy, "config", None)
        raw_mode = requested
        if raw_mode is None and not self.gen.enable_subtask:
            # The shared CLI maps --direct_subtask to enable_subtask=False.
            # For G0.5 this explicitly selects the action-only System 1 path.
            raw_mode = "system1"
        if raw_mode is None:
            raw_mode = _read_config(config, "runtime_system_mode")
        if raw_mode is None:
            raw_mode = _read_config(config, "runtime_system")
        if raw_mode is None:
            raw_mode = "auto"
        normalized = _MODE_ALIASES.get(str(raw_mode).strip().lower())
        if normalized is None:
            raise ValueError(
                f"Unknown G0.5 runtime system mode {raw_mode!r}; expected one of {sorted(_MODE_ALIASES)}."
            )
        if normalized == "auto":
            return "system2" if _predict_cot(self.policy) is True else "system1"
        return normalized

    def _validate_checkpoint_capabilities(self) -> None:
        config = getattr(self.policy, "config", None)
        discrete = _read_config(config, "discrete_action")
        continuous = _read_config(config, "continuous_action")
        if discrete is False and continuous is False:
            raise ValueError(
                "G0.5 System 1 requires an enabled action head, but this checkpoint/config "
                "has both discrete_action=False and continuous_action=False."
            )
        if self.system_mode == "system2" and _predict_cot(self.policy) is not True:
            raise ValueError(
                "G0.5 System 2 requires a checkpoint/config with predict_cot=True; "
                "use runtime_system_mode='system1' for direct task execution."
            )

    def update_language_state(self, observation: dict[str, Any] | None, state: RuntimeState) -> None:
        """Do not invent an independent planner.

        G0.5 System 2 reasoning is collected by :meth:`select_action` from the
        same inference pass as the action.
        """

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        # A new mapping avoids mutating the provider's cached/preprocessed
        # observation.  The raw operator task is authoritative for G0.5; a
        # previously generated subtask remains runtime state, not a silent task
        # replacement before the author prompt path.
        batch = dict(observation)
        batch["task"] = state.task

        runtime_hook = getattr(self.policy, "predict_action_chunk_with_runtime", None)
        if callable(runtime_hook):
            output = runtime_hook(batch, task=state.task, system_mode=self.system_mode)
        else:
            if self.system_mode == "system2":
                raise RuntimeError(
                    "G0.5 System 2 requires policy.predict_action_chunk_with_runtime() "
                    "so cot_text and the matching action chunk come from one structured inference result."
                )
            output = self.policy.predict_action_chunk(batch)

        action_chunk, metadata = _split_runtime_output(output)
        if self.system_mode == "system2":
            self._publish_reasoning(metadata, state)
        return action_chunk

    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        # The author implementation does not expose a separate policy-agnostic
        # planner.  CoT is emitted in-stream by select_action.
        return ""

    def handle_interjection(
        self, user_text: str, observation: dict[str, Any] | None, state: RuntimeState
    ) -> None:
        # Preserve the operator update verbatim and invalidate the displayed
        # generated subtask. The runtime will use set_task for task replacement;
        # this method only records a mid-rollout plan/update.
        if user_text:
            state.set_context("plan", user_text, label="plan")
            state.set_context("subtask", None)

    def _publish_reasoning(self, metadata: Mapping[str, Any], state: RuntimeState) -> None:
        text = _first_text(metadata, _TEXT_KEYS)
        if text:
            _set_generated_context(state, "cot_text", text, label="reasoning")

        explicit_subtask = _as_text(metadata.get("subtask"))
        subtask = explicit_subtask or _extract_labeled_text(_SUBTASK_RE, text)
        if subtask:
            # PR #4183's rollout observation provider treats language_context["subtask"]
            # as the next policy command. G0.5 CoT must not replace the operator's task:
            # it is telemetry from the same unified stream, not a separately supervised
            # low-level prompt. Keep the parsed value visible without changing prompt routing.
            state.extra["g05_subtask"] = subtask
            state.log(f"  subtask: {subtask}")

        explicit_memory = _as_text(metadata.get("memory"))
        memory = explicit_memory or _extract_labeled_text(_MEMORY_RE, text)
        if memory:
            _set_generated_context(state, "memory", memory, label="memory")

        plan = _as_text(metadata.get("plan"))
        if plan:
            _set_generated_context(state, "plan", plan, label="plan")


def _read_config(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _predict_cot(policy: Any) -> bool | None:
    config_value = _read_config(getattr(policy, "config", None), "predict_cot")
    if config_value is not None:
        return bool(config_value)
    policy_value = getattr(policy, "predict_cot", None)
    return None if policy_value is None else bool(policy_value)


def _split_runtime_output(output: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(output, tuple) and len(output) == 2:
        action_chunk, raw_metadata = output
        if isinstance(raw_metadata, Mapping):
            return action_chunk, raw_metadata
        if isinstance(raw_metadata, str):
            return action_chunk, {"cot_text": raw_metadata}
        raise TypeError(
            "G0.5 runtime tuple output must contain metadata mapping or cot_text string as its second item."
        )

    if isinstance(output, Mapping) and (
        "action_chunk" in output or any(key in output for key in (*_TEXT_KEYS, "subtask", "memory", "plan"))
    ):
        action_key = "action_chunk" if "action_chunk" in output else "action"
        if action_key not in output:
            raise ValueError("G0.5 structured runtime output is missing 'action_chunk' (or 'action').")
        return output[action_key], output

    return output, {}


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list | tuple) and len(value) == 1 and isinstance(value[0], str):
        return value[0].strip()
    return ""


def _first_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _as_text(metadata.get(key))
        if text:
            return text
    return ""


def _extract_labeled_text(pattern: re.Pattern[str], text: str) -> str:
    if not text:
        return ""
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _set_generated_context(state: RuntimeState, key: str, value: str, *, label: str) -> bool:
    """Publish metadata from the current inference without invalidating its action.

    ``RuntimeState.revision`` guards against operator changes while inference is
    in flight. G0.5 reasoning belongs to that same inference result, so bumping
    the revision here would make the generic runtime discard the matching
    System 1 action chunk.
    """
    with state.lock:
        if state.language_context.get(key) == value:
            return False
        state.language_context[key] = value
        state.log(f"  {label}: {value}")
        return True
