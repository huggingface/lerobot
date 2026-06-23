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
"""Tool protocol — the contract every runnable tool implementation honors.

Tools are the executable side of the OpenAI-style function-calling
abstraction the v3.1 language schema (PR 1) carries on assistant
messages: the schema describes *what can be called*, the tool
implementation describes *how to call it*.

Implementations live one-per-file under :mod:`lerobot.tools` (e.g.
``say.py`` for ``SayTool``) and are registered in
:mod:`lerobot.tools.registry`. The runtime instantiates them lazily so
heavy dependencies (torch models, audio backends, network clients,
hardware drivers) only load when the dataset actually declares the tool.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    """Minimum surface every tool must expose."""

    #: Name matching ``schema["function"]["name"]``. The runtime dispatcher
    #: routes incoming ``tool_calls`` to the implementation by this key.
    name: str

    #: OpenAI-style function-call schema. Same dict the dataset stores in
    #: ``meta/info.json["tools"]`` and the chat template renders into the
    #: prompt.
    schema: dict[str, Any]

    def call(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool with the model-provided arguments.

        ``arguments`` is the parsed dict from
        ``tool_calls[i]["function"]["arguments"]`` (already JSON-decoded
        when the model emits a JSON-string by the chat-template
        convention). Implementations validate the dict against their own
        schema; the runtime only routes by name.

        Return value is implementation-defined — typically a tensor
        (TTS audio), a Path (saved file), a dict (structured result), or
        ``None`` (side-effect-only call).
        """
