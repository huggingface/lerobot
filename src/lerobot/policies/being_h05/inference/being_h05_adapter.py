#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Being-H0.5 bridge for the generic language runtime."""

from __future__ import annotations

import threading
from typing import Any

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter, GenerationConfig

_MAX_NEW_TOKENS = {
    "subtask": 32,
    "memory": 48,
    "interjection": 64,
    "vqa": 96,
}


class BeingH05PolicyAdapter(BaseLanguageAdapter):
    """Expose Being-H0.5 action prediction and native VLM text decoding."""

    def __init__(self, policy: Any, gen: GenerationConfig | None = None) -> None:
        super().__init__(policy, gen)
        if not callable(getattr(policy, "generate_text", None)):
            raise TypeError("Being-H0.5 language runtime requires policy.generate_text().")
        model_id = str(getattr(getattr(policy, "config", None), "author_model_id", ""))
        self.supports_text = "robocasa" not in model_id.lower()
        self._inference_lock = threading.RLock()

    def update_language_state(self, observation: dict[str, Any] | None, state: RuntimeState) -> None:
        if self.supports_text:
            super().update_language_state(observation, state)

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        with self._inference_lock:
            return self.policy.predict_action_chunk(observation)

    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        if observation is None:
            return ""
        if not self.supports_text:
            raise RuntimeError(
                "The released Being-H0.5 RoboCasa action fine-tune does not retain usable text generation; "
                "use lerobot/being_h05_base for VQA."
            )
        prompt = _runtime_prompt(kind, state, user_text)
        with self._inference_lock:
            generated = self.policy.generate_text(
                observation,
                prompt,
                max_new_tokens=_MAX_NEW_TOKENS.get(kind, 64),
                min_new_tokens=self.gen.min_new_tokens,
                temperature=self.gen.temperature,
                top_p=self.gen.top_p,
            )
        if isinstance(generated, str):
            return generated.strip()
        return generated[0].strip() if generated else ""

    def handle_interjection(
        self,
        user_text: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
    ) -> None:
        if self.supports_text:
            super().handle_interjection(user_text, observation, state)
        elif user_text:
            state.set_context("subtask", user_text, label="subtask")


def _runtime_prompt(kind: str, state: RuntimeState, user_text: str | None) -> str:
    if kind == "vqa":
        return (user_text or state.task).strip()
    if kind == "subtask":
        return (
            f"The robot's task is: {state.task}\n"
            "What is the next concise, executable subtask? Answer with only the subtask."
        )
    if kind == "memory":
        prior = state.extra.get("prior_subtask", "")
        current = state.language_context.get("subtask", "")
        return (
            f"The robot's task is: {state.task}\n"
            f"Previous subtask: {prior}\nCurrent subtask: {current}\n"
            "Briefly summarize the progress that should be remembered for the next step."
        )
    if kind == "interjection":
        return (
            f"The robot's task is: {state.task}\n"
            f"The operator added: {user_text or ''}\n"
            "Give a concise updated plan."
        )
    raise ValueError(f"Unsupported Being-H0.5 text kind: {kind!r}.")
