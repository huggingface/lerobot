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

from __future__ import annotations

from typing import Literal

import datasets
import pyarrow as pa

LANGUAGE_PERSISTENT = "language_persistent"
LANGUAGE_EVENTS = "language_events"
LANGUAGE_COLUMNS = (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS)
PERSISTENT_ROW_FIELDS = ("role", "content", "style", "timestamp", "camera", "tool_calls")
EVENT_ROW_FIELDS = ("role", "content", "style", "camera", "tool_calls")

CORE_STYLES = {
    "subtask",
    "plan",
    "memory",
    "motion",
    "interjection",
    "vqa",
    "trace",
    "task_aug",
}
# Project-local styles can be registered at import time by appending to
# ``EXTENDED_STYLES`` before ``column_for_style`` is called. Anything added
# here is treated as a known style alongside ``CORE_STYLES`` for resolver
# validation. Empty by default — populate from a downstream module that
# also extends ``PERSISTENT_STYLES`` or ``EVENT_ONLY_STYLES`` to declare
# the new style's column.
EXTENDED_STYLES: set[str] = set()
STYLE_REGISTRY = CORE_STYLES | EXTENDED_STYLES

PERSISTENT_STYLES = {"subtask", "plan", "memory", "motion", "task_aug"}
EVENT_ONLY_STYLES = {"interjection", "vqa", "trace"}

# Styles whose ``content`` is grounded in a specific camera view. Rows of these
# styles MUST carry a non-null ``camera`` referencing an ``observation.images.*``
# feature key. Rows of every other style MUST have ``camera=None``. ``motion``
# is intentionally NOT in this set: motion primitives are described in
# robot-frame (joint / Cartesian) terms, not pixel space, so they are
# camera-agnostic. ``trace`` is the pixel-trajectory event style and IS
# view-dependent. The ``camera`` field nevertheless lives on
# ``PERSISTENT_ROW_FIELDS`` too so the schema, validator, and resolver
# behave symmetrically across the two columns; persistent rows simply
# always have ``camera=None`` in practice today.
VIEW_DEPENDENT_STYLES = {"vqa", "trace"}

LanguageColumn = Literal["language_persistent", "language_events"]


def _json_arrow_type() -> pa.DataType:
    """Return the Arrow JSON type, falling back to ``string`` on older pyarrow."""
    return pa.json_() if hasattr(pa, "json_") else pa.string()


def _json_feature() -> object:
    """Return the HF ``datasets`` JSON feature, falling back to a string value."""
    return datasets.Json() if hasattr(datasets, "Json") else datasets.Value("string")


def language_persistent_row_arrow_type() -> pa.StructType:
    """Return the Arrow struct type for a single persistent language row.

    Persistent rows carry their own ``timestamp`` because they represent a state
    that became active at a specific moment and remains active until superseded.
    ``timestamp`` is ``float32`` to match the timestamp dtype LeRobotDataset
    uses for frame data.
    """
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )


def language_event_row_arrow_type() -> pa.StructType:
    """Return the Arrow struct type for a single event language row.

    Event rows have no ``timestamp`` field: each event is stored on the dataset
    row whose frame timestamp is the event's firing time.
    """
    return pa.struct(
        [
            pa.field("role", pa.string(), nullable=False),
            pa.field("content", pa.string(), nullable=True),
            pa.field("style", pa.string(), nullable=True),
            pa.field("camera", pa.string(), nullable=True),
            pa.field("tool_calls", pa.list_(_json_arrow_type()), nullable=True),
        ]
    )


def language_persistent_arrow_type() -> pa.ListType:
    """Return the Arrow list type for the ``language_persistent`` column."""
    return pa.list_(language_persistent_row_arrow_type())


def language_events_arrow_type() -> pa.ListType:
    """Return the Arrow list type for the ``language_events`` column."""
    return pa.list_(language_event_row_arrow_type())


def language_persistent_row_feature() -> dict[str, object]:
    """Return the HF ``datasets`` feature mapping for a persistent language row."""
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "timestamp": datasets.Value("float32"),
        "camera": datasets.Value("string"),
        "tool_calls": datasets.List(_json_feature()),
    }


def language_event_row_feature() -> dict[str, object]:
    """Return the HF ``datasets`` feature mapping for an event language row."""
    return {
        "role": datasets.Value("string"),
        "content": datasets.Value("string"),
        "style": datasets.Value("string"),
        "camera": datasets.Value("string"),
        "tool_calls": datasets.List(_json_feature()),
    }


def language_persistent_column_feature() -> datasets.List:
    """Return the HF ``datasets`` feature for the ``language_persistent`` column."""
    return datasets.List(language_persistent_row_feature())


def language_events_column_feature() -> datasets.List:
    """Return the HF ``datasets`` feature for the ``language_events`` column."""
    return datasets.List(language_event_row_feature())


def language_feature_info() -> dict[str, dict]:
    """Return the ``info["features"]`` entries for both language columns."""
    return {
        LANGUAGE_PERSISTENT: {"dtype": "language", "shape": (1,), "names": None},
        LANGUAGE_EVENTS: {"dtype": "language", "shape": (1,), "names": None},
    }


def is_language_column(key: str) -> bool:
    """Return ``True`` if ``key`` is one of the dataset's language column names."""
    return key in LANGUAGE_COLUMNS


def is_view_dependent_style(style: str | None) -> bool:
    """Return ``True`` if rows of ``style`` must be tagged with a ``camera`` key."""
    return style in VIEW_DEPENDENT_STYLES


def validate_camera_field(style: str | None, camera: str | None) -> None:
    """Enforce the ``camera`` invariant: required iff ``style`` is view-dependent.

    Raises ``ValueError`` if a view-dependent style is missing ``camera`` or if
    a non-view-dependent style carries one. Pipeline writers and the validator
    should call this on every emitted row.
    """
    if is_view_dependent_style(style):
        if not camera:
            raise ValueError(
                f"Rows of view-dependent style {style!r} require a non-empty 'camera' "
                f"field referencing an 'observation.images.*' feature key."
            )
    elif camera is not None:
        raise ValueError(f"Rows of style {style!r} must have camera=None; got camera={camera!r}.")


# --- Tool registry --------------------------------------------------------
# Tools declared on a dataset live in ``meta/info.json["tools"]`` as a list
# of OpenAI-style function schemas. The runtime / training stack reads them
# through :class:`LeRobotDatasetMetadata.tools` (with these constants as
# fallback when the dataset doesn't declare any). Implementations live
# under :mod:`lerobot.tools` (one file per tool); see
# ``docs/source/tools.mdx`` for the authoring guide.

SAY_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "say",
        "description": "Speak a short utterance to the user via the TTS executor.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The verbatim text to speak.",
                }
            },
            "required": ["text"],
        },
    },
}
"""Canonical schema for the ``say`` tool emitted by the steerable
annotation pipeline (PR 2 Module 2). Single source of truth — PR 2's
writer, PR 3's runtime tool registry, and the dataset visualizer all
import this constant rather than duplicating the dict."""

DEFAULT_TOOLS: list[dict] = [SAY_TOOL_SCHEMA]
"""Fallback tools list. Returned by ``LeRobotDatasetMetadata.tools``
when ``meta/info.json["tools"]`` is unset, so unannotated datasets and
chat-template consumers (``apply_chat_template(messages, tools=...)``)
keep working out of the box."""


def column_for_style(style: str | None) -> LanguageColumn:
    """Map a language style to the column where rows of that style are stored.

    Styles in :data:`PERSISTENT_STYLES` route to :data:`LANGUAGE_PERSISTENT`.
    Styles in :data:`EVENT_ONLY_STYLES` and the implicit ``None`` style route
    to :data:`LANGUAGE_EVENTS`.
    """
    if style is None:
        return LANGUAGE_EVENTS
    if style in PERSISTENT_STYLES:
        return LANGUAGE_PERSISTENT
    if style in EVENT_ONLY_STYLES:
        return LANGUAGE_EVENTS
    raise ValueError(f"Unknown language style: {style!r}")
