# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Agentic flow: observe (stereo + VLM + 3D) → reason (LLM) → act.

Use the stereo camera and VLM to build a scene, then an LLM reasons over
the scene and task to decide the next action (e.g. which object to pick).
"""

from .agentic_flow import (
    AgentAction,
    ReasoningAgent,
    SceneObject,
    SceneObservation,
    build_scene_summary,
)
from .manipulation_skills import (
    MANIPULATION_SKILL_METAS,
    build_skills_system_appendix,
    list_skills_json,
)

__all__ = [
    "AgentAction",
    "MANIPULATION_SKILL_METAS",
    "ReasoningAgent",
    "SceneObject",
    "SceneObservation",
    "build_scene_summary",
    "build_skills_system_appendix",
    "list_skills_json",
]
