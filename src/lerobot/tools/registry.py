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
"""Tool registry — name → implementation class.

Adding a new tool:

1. Drop a file under ``src/lerobot/tools/`` that defines a class
   conforming to :class:`lerobot.tools.base.Tool` (must expose ``name``,
   ``schema``, ``call(arguments)``).
2. Register the class here under :data:`TOOL_REGISTRY`.
3. (Optional) Pre-populate ``meta/info.json["tools"]`` on your dataset
   to advertise the schema to the chat-template + policy. The PR 2
   annotation pipeline preserves anything you put there.

See ``docs/source/tools.mdx`` for the full authoring guide.
"""

from __future__ import annotations

from typing import Any

from .base import Tool
from .say import SayTool

#: Map from ``function.name`` to a class implementing :class:`Tool`.
#: The runtime instantiates entries lazily — registering a tool here is
#: essentially free (no model load happens until ``call`` runs).
TOOL_REGISTRY: dict[str, type] = {
    "say": SayTool,
}


def get_tools(meta: Any, **kwargs: Any) -> dict[str, Tool]:
    """Build name → tool-instance dict from a dataset's declared catalog.

    ``meta`` is anything with a ``.tools`` attribute returning the
    OpenAI-style schema list — typically a
    :class:`lerobot.datasets.dataset_metadata.LeRobotDatasetMetadata`.
    Each entry whose ``function.name`` is registered here is
    instantiated with the schema dict; tools whose name is unknown to
    the registry are skipped (the schema still rides through the chat
    template, the model just can't actually invoke that tool at
    inference).

    Extra keyword arguments are forwarded to every constructor — useful
    for runtime defaults like ``output_dir=Path("./tts_log")``.
    """
    declared = list(meta.tools)
    instances: dict[str, Tool] = {}
    for schema in declared:
        try:
            name = schema["function"]["name"]
        except (KeyError, TypeError):
            continue
        cls = TOOL_REGISTRY.get(name)
        if cls is None:
            continue
        instances[name] = cls(schema=schema, **kwargs)
    return instances
