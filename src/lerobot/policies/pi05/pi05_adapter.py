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

"""PI05 adapter for the language-conditioned runtime.

PI05 is a flat VLA: it conditions the action expert directly on the task text,
which its preprocessor tokenizes into ``observation.language.tokens``. It has no
subtask/memory generation head, so the runtime simply predicts an action chunk
from the already-tokenized observation. Text generation is unsupported — run
with ``--sim.direct_subtask`` so the runtime doesn't attempt subtask/memory
generation (what you type becomes the task the preprocessor tokenizes).
"""

from __future__ import annotations

from typing import Any

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter


class PI05PolicyAdapter(BaseLanguageAdapter):
    """Runtime bridge for flat PI05 policies (direct task-text conditioning)."""

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        # The task text was tokenized into observation.language.* by the policy
        # preprocessor (fed the current task by the observation provider), so we
        # just predict the action chunk from it.
        return self.policy.predict_action_chunk(observation)

    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        # PI05 has no text-generation head; direct-subtask mode skips this path.
        return ""
