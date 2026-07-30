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

"""EO-1 bridge for the generic language runtime."""

from __future__ import annotations

from typing import Any

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter


class EO1PolicyAdapter(BaseLanguageAdapter):
    """Expose EO-1's native Qwen text head and flow-action path."""

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        instruction = state.language_context.get("subtask") or state.task or ""
        batch = self.policy.prepare_runtime_action_batch(observation, instruction)
        return self.policy.predict_action_chunk(batch)

    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        if observation is None:
            return ""
        batch = dict(observation)
        batch["task"] = state.task or ""
        outputs = self.policy.generate_text(
            batch,
            kind=kind,
            user_text=user_text,
            min_new_tokens=self.gen.min_new_tokens,
            temperature=self.gen.temperature,
            top_p=self.gen.top_p,
        )
        if len(outputs) != 1:
            raise ValueError(f"The interactive runtime expected one EO-1 text output, got {len(outputs)}.")
        return outputs[0]
