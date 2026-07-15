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
"""Small event helper shared by the language-runtime command loops."""

from __future__ import annotations

from typing import Any


def _emit(state: Any, event_name: str) -> None:
    if hasattr(state, "emit"):
        state.emit(event_name)
    else:
        state.setdefault("events_this_tick", []).append(event_name)
