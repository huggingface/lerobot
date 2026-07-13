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

"""MolmoAct2 adapter for the language-conditioned runtime.

MolmoAct2 is a flat VLA: it conditions on a single natural-language ``task``
string (``"The task is to {task}. ..."``) that its processor packs — together
with the images and discretized state — into model inputs (``input_ids`` /
``pixel_values`` / ...). It has no subtask/memory generation head, so the runtime
just predicts an action chunk from the already-packed observation.

Run with ``--direct_subtask`` (robot) or ``--sim.direct_subtask`` (sim): what you
type becomes the ``task`` the processor packs, and the runtime does not attempt
subtask/memory generation. The observation provider re-packs on every frame with
the live task (see ``_build_robot_observation_provider`` / the dynamic task
getter in ``runtime.cli``), so typing a new command switches the instruction
immediately.
"""

from __future__ import annotations

from typing import Any

from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import BaseLanguageAdapter


class MolmoAct2PolicyAdapter(BaseLanguageAdapter):
    """Runtime bridge for flat MolmoAct2 policies (direct task-text conditioning)."""

    def select_action(self, observation: dict[str, Any], state: RuntimeState) -> Any:
        # The current task/subtask was packed into the model inputs (input_ids,
        # pixel_values, ...) by the policy processor, fed the live task by the
        # observation provider. ``predict_action_chunk`` resolves the action mode
        # from the checkpoint config (``inference_action_mode`` must be set to
        # "continuous" or "discrete").
        return self.policy.predict_action_chunk(observation)

    def generate_text(
        self,
        kind: str,
        observation: dict[str, Any] | None,
        state: RuntimeState,
        user_text: str | None = None,
    ) -> str:
        # MolmoAct2 has no text-generation head; direct-subtask mode skips this.
        return ""
