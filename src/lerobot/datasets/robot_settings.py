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

"""The ``robot_settings`` text column on episode metadata.

The column is written only for a new dataset whose robot returns settings.
It is one JSON string per episode, stored as an Arrow string.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pyarrow as pa

    from lerobot.robots.robot import Robot

ROBOT_SETTINGS_COLUMN = "robot_settings"


class _ColumnTable(Protocol):
    column_names: list[str]


class _SettingsMetadata(Protocol):
    episodes: _ColumnTable | None

    @property
    def total_episodes(self) -> int: ...


def episodes_have_robot_settings(meta: _SettingsMetadata) -> bool:
    """True when this metadata already has the column and at least one episode."""
    episodes = meta.episodes
    if episodes is None or meta.total_episodes == 0:
        return False
    return ROBOT_SETTINGS_COLUMN in episodes.column_names


def settings_text(robot: Robot) -> str | None:
    """JSON for one episode, or None when this robot does not record settings.

    A failed read still returns text, so the column does not change type.
    """
    try:
        value = robot.read_settings()
    except Exception as exc:
        value = {"error": str(exc)}
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cast_robot_settings_table(table: pa.Table) -> pa.Table:
    """Store ``robot_settings`` as an Arrow string."""
    if ROBOT_SETTINGS_COLUMN not in table.column_names:
        return table
    import pyarrow as pa

    index = table.schema.get_field_index(ROBOT_SETTINGS_COLUMN)
    return table.set_column(
        index, ROBOT_SETTINGS_COLUMN, table.column(ROBOT_SETTINGS_COLUMN).cast(pa.string())
    )
