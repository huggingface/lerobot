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

import copy
import hashlib
import re
from collections.abc import Sequence
from typing import Any

from lerobot.configs.recipe import DEFAULT_BINDINGS, TrainingRecipe

from .language import (
    EVENT_ONLY_STYLES,
    LANGUAGE_PERSISTENT,
    PERSISTENT_STYLES,
    column_for_style,
)

LanguageRow = dict[str, Any]
RenderedMessages = dict[str, list[Any]]

_RESOLVER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def active_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow] | None = None,
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """Return the persistent row of ``style`` that is active at time ``t``.

    A persistent row is "active" at ``t`` when its own ``timestamp`` is the
    most recent one ``<= t`` for the given ``style``/``role``/``tool_name``/
    ``camera`` selector. ``events`` is accepted for resolver-signature
    uniformity but is not consulted: only persistent styles are valid here.
    """
    _validate_persistent_resolver("active_at", style)
    matches = _matching_rows(
        persistent, style=style, role=role, tool_name=tool_name, camera=camera
    )
    matches = [row for row in matches if _timestamp(row) <= t]
    return _select_latest(
        matches, style=style, role=role, tool_name=tool_name, camera=camera
    )


def emitted_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """Return the row of ``style`` emitted at exactly time ``t``.

    For persistent styles, this matches persistent rows whose own ``timestamp``
    equals ``t``. For event styles, the ``events`` list is assumed to come from
    the dataset row at frame ``t`` (event rows carry no timestamp of their own),
    so all matching event rows are considered emitted at ``t``. ``camera``
    filters by the row's ``camera`` field — required to disambiguate when
    multiple view-dependent rows share ``(t, role)`` across cameras.
    """
    column = column_for_style(style)
    if column == LANGUAGE_PERSISTENT:
        matches = [
            row
            for row in _matching_rows(
                persistent, style=style, role=role, tool_name=tool_name, camera=camera
            )
            if _timestamp(row) == t
        ]
        return _select_one(
            matches,
            style=style,
            role=role,
            tool_name=tool_name,
            camera=camera,
            sort_key=_persistent_sort_key,
        )
    matches = _matching_rows(
        events, style=style, role=role, tool_name=tool_name, camera=camera
    )
    return _select_one(
        matches,
        style=style,
        role=role,
        tool_name=tool_name,
        camera=camera,
        sort_key=_event_sort_key,
    )


def nth_prev(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow] | None = None,
    style: str | None = None,
    offset: int = 1,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """Return the persistent row that was active ``offset`` steps before ``t``.

    Walks back through chronologically sorted persistent rows of ``style``
    (filtered by optional ``role``/``tool_name``/``camera``) and returns the
    one ``offset`` positions before the row active at ``t``. Only valid for
    persistent styles.
    """
    return _nth_relative(
        t,
        persistent=persistent,
        style=style,
        offset=-offset,
        role=role,
        tool_name=tool_name,
        camera=camera,
        resolver_name="nth_prev",
    )


def nth_next(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow] | None = None,
    style: str | None = None,
    offset: int = 1,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """Return the persistent row that becomes active ``offset`` steps after ``t``.

    Walks forward through chronologically sorted persistent rows of ``style``
    (filtered by optional ``role``/``tool_name``/``camera``) and returns the
    one ``offset`` positions after the row active at ``t``. Only valid for
    persistent styles.
    """
    return _nth_relative(
        t,
        persistent=persistent,
        style=style,
        offset=offset,
        role=role,
        tool_name=tool_name,
        camera=camera,
        resolver_name="nth_next",
    )


def render_sample(
    *,
    recipe: TrainingRecipe,
    persistent: Sequence[LanguageRow] | None,
    events: Sequence[LanguageRow] | None,
    t: float,
    sample_idx: int,
    task: str | None = None,
    dataset_ctx: Any | None = None,
) -> RenderedMessages | None:
    """Render the chat-style messages for a single dataset sample.

    Resolves the recipe's bindings against ``persistent`` and ``events`` rows
    at frame timestamp ``t``, then expands the recipe's message templates.
    Returns ``None`` if the resolved sample contains no target message.
    """
    persistent_rows = _normalize_rows(persistent or [])
    event_rows = _normalize_rows(events or [])
    selected_recipe = _select_recipe(recipe, sample_idx)
    bindings = _resolve_bindings(
        selected_recipe,
        persistent=persistent_rows,
        events=event_rows,
        t=t,
        sample_idx=sample_idx,
        task=task,
        dataset_ctx=dataset_ctx,
    )
    return _render_message_recipe(selected_recipe, bindings)


def _select_recipe(recipe: TrainingRecipe, sample_idx: int) -> TrainingRecipe:
    """Pick a deterministic blend component for ``sample_idx`` (or return ``recipe``)."""
    if recipe.blend is None:
        return recipe

    total_weight = sum(component.weight or 0.0 for component in recipe.blend.values())
    if total_weight <= 0:
        raise ValueError("Blend weights must sum to a positive value.")

    digest = hashlib.blake2b(str(sample_idx).encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / 2**64 * total_weight
    cumulative = 0.0
    last_component: TrainingRecipe | None = None
    for component in recipe.blend.values():
        last_component = component
        cumulative += component.weight or 0.0
        if draw < cumulative:
            return component
    assert last_component is not None
    return last_component


def _resolve_bindings(
    recipe: TrainingRecipe,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
    sample_idx: int,
    task: str | None,
    dataset_ctx: Any | None,
) -> dict[str, LanguageRow | str | None]:
    """Resolve every binding in ``recipe`` (plus ``task``) at time ``t``."""
    bindings: dict[str, LanguageRow | str | None] = {
        "task": _resolve_task(
            task, dataset_ctx, persistent=persistent, sample_idx=sample_idx
        ),
    }
    specs = {**DEFAULT_BINDINGS, **(recipe.bindings or {})}
    for name, spec in specs.items():
        bindings[name] = _resolve_spec(spec, persistent=persistent, events=events, t=t)
    return bindings


def _resolve_task(
    task: str | None,
    dataset_ctx: Any | None,
    *,
    persistent: Sequence[LanguageRow] = (),
    sample_idx: int = 0,
) -> str | None:
    """Return the task string for ``sample_idx``.

    Resolution order:

    1. Explicit ``task`` override (caller-supplied) wins.
    2. If ``persistent`` contains rows of style ``task_aug`` (role=user),
       deterministically pick one by ``sample_idx`` so each frame of an
       episode rotates through the available rephrasings across an epoch.
       This realizes Xiao 2022 / CAST-style task-prompt diversity without
       changing ``meta/tasks.parquet`` and without forcing recipes to opt
       in: ``${task}`` automatically picks a rephrasing when one exists,
       and falls back to the canonical task otherwise. Recipes that want
       the literal canonical task can override the binding.
    3. Otherwise read the canonical task from ``dataset_ctx`` (which is
       backed by ``meta/tasks.parquet``).
    """
    if task is not None:
        return task

    aug_rows = [
        r
        for r in persistent
        if r.get("style") == "task_aug" and r.get("role") == "user"
    ]
    if aug_rows:
        # Deterministic, blake2b-based pick keyed on sample_idx so the
        # rotation is reproducible across runs (Python's built-in ``hash``
        # is process-randomized).
        digest = hashlib.blake2b(
            f"task_aug:{sample_idx}".encode(), digest_size=8
        ).digest()
        idx = int.from_bytes(digest, "big") % len(aug_rows)
        chosen = aug_rows[idx].get("content")
        if chosen:
            return str(chosen)

    if dataset_ctx is None:
        return None
    if isinstance(dataset_ctx, dict):
        return dataset_ctx.get("task")
    return getattr(dataset_ctx, "task", None)


def _resolve_spec(
    spec: str,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
) -> LanguageRow | None:
    """Parse a single binding's resolver expression and dispatch to its function."""
    match = _RESOLVER_RE.match(spec.strip())
    if match is None:
        raise ValueError(f"Invalid resolver expression: {spec!r}")
    name = match.group("name")
    kwargs = _parse_resolver_args(match.group("args"))
    kwargs.pop("t_arg", None)

    resolvers = {
        "active_at": active_at,
        "emitted_at": emitted_at,
        "nth_prev": nth_prev,
        "nth_next": nth_next,
    }
    if name not in resolvers:
        raise ValueError(f"Unknown language resolver: {name!r}")
    return resolvers[name](t, persistent=persistent, events=events, **kwargs)


def _parse_resolver_args(args: str) -> dict[str, Any]:
    """Parse a comma-separated resolver argument list into a kwargs dict."""
    kwargs: dict[str, Any] = {}
    if not args.strip():
        return kwargs

    parts = [part.strip() for part in args.split(",") if part.strip()]
    for part in parts:
        if part == "t":
            kwargs["t_arg"] = True
            continue
        if "=" not in part:
            raise ValueError(f"Invalid resolver argument: {part!r}")
        key, value = (item.strip() for item in part.split("=", 1))
        if key == "offset":
            kwargs[key] = int(value)
        else:
            kwargs[key] = value.strip("\"'")
    return kwargs


def _render_message_recipe(
    recipe: TrainingRecipe,
    bindings: dict[str, LanguageRow | str | None],
) -> RenderedMessages | None:
    """Expand ``recipe.messages`` into rendered chat messages using ``bindings``."""
    assert recipe.messages is not None
    messages: list[dict[str, Any]] = []
    streams: list[str | None] = []
    target_indices: list[int] = []

    for turn in recipe.messages:
        if turn.if_present is not None and bindings.get(turn.if_present) is None:
            continue

        message = {"role": turn.role}
        if turn.content is not None:
            message["content"] = _render_content(turn.content, bindings)

        if turn.tool_calls_from is not None:
            row = bindings.get(turn.tool_calls_from)
            tool_calls = row.get("tool_calls") if isinstance(row, dict) else None
            if tool_calls:
                message["tool_calls"] = copy.deepcopy(tool_calls)

        message_idx = len(messages)
        messages.append(message)
        streams.append(turn.stream)
        if turn.target:
            target_indices.append(message_idx)

    # A render is meaningful if it supervises *something*: either a
    # text-CE target turn, or a ``low_level`` stream turn (flow / action
    # supervision — e.g. the flow-only ``low_level_execution`` recipe,
    # ``user(${subtask})`` with ``stream: low_level`` and no target).
    # Without this, a flow-only recipe renders to ``None`` every time
    # the blend draws it → ``predict_actions`` is never True → the
    # action expert never receives a flow loss.
    has_low_level = any(stream == "low_level" for stream in streams)
    if not target_indices and not has_low_level:
        return None

    rendered = {
        "messages": messages,
        "message_streams": streams,
        "target_message_indices": target_indices,
    }
    _validate_rendered(rendered)
    return rendered


def _render_content(
    content: str | list[dict[str, Any]],
    bindings: dict[str, LanguageRow | str | None],
) -> str | list[dict[str, Any]]:
    """Substitute bindings into a string or each string field of multimodal blocks."""
    if isinstance(content, str):
        return _substitute(content, bindings)

    rendered_blocks = []
    for block in content:
        rendered_block = copy.deepcopy(block)
        for key, value in rendered_block.items():
            if isinstance(value, str):
                rendered_block[key] = _substitute(value, bindings)
        rendered_blocks.append(rendered_block)
    return rendered_blocks


def _substitute(template: str, bindings: dict[str, LanguageRow | str | None]) -> str:
    """Replace ``${name}`` placeholders in ``template`` with their bound values."""

    def replace(match: re.Match[str]) -> str:
        """Resolve a single ``${name}`` match to its bound string value."""
        name = match.group(1)
        if name not in bindings:
            raise ValueError(f"Unknown template binding: {name!r}")
        value = bindings[name]
        if value is None:
            return ""
        if isinstance(value, dict):
            content = value.get("content")
            return "" if content is None else str(content)
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, template)


def _validate_rendered(rendered: RenderedMessages) -> None:
    """Sanity-check the rendered output for stream/target alignment."""
    messages = rendered["messages"]
    streams = rendered["message_streams"]
    target_indices = rendered["target_message_indices"]

    if len(streams) != len(messages):
        raise ValueError("message_streams must be aligned with messages.")
    # Valid iff it supervises something: a text-CE target turn OR a
    # ``low_level`` stream turn (flow / action supervision).
    if not target_indices and not any(s == "low_level" for s in streams):
        raise ValueError(
            "Rendered samples must contain a target message or a "
            "low_level-stream message."
        )
    for idx in target_indices:
        if idx < 0 or idx >= len(messages):
            raise ValueError(f"Target message index {idx} is out of bounds.")
    for idx, stream in enumerate(streams):
        if stream is None:
            raise ValueError(f"Rendered message {idx} has no stream.")


def _nth_relative(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    style: str | None,
    offset: int,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
    resolver_name: str,
) -> LanguageRow | None:
    """Shared body for ``nth_prev`` / ``nth_next`` with signed ``offset``."""
    _validate_persistent_resolver(resolver_name, style)
    if abs(offset) < 1:
        raise ValueError(f"{resolver_name} offset must be non-zero.")

    rows = sorted(
        _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera),
        key=_persistent_sort_key,
    )
    if not rows:
        return None

    anchor_idx = None
    for idx, row in enumerate(rows):
        if _timestamp(row) <= t:
            anchor_idx = idx
        else:
            break

    target_idx = (offset - 1 if offset > 0 else None) if anchor_idx is None else anchor_idx + offset

    if target_idx is None or target_idx < 0 or target_idx >= len(rows):
        return None
    return rows[target_idx]


def _validate_persistent_resolver(resolver_name: str, style: str | None) -> None:
    """Reject calls with missing or event-only ``style`` for persistent resolvers."""
    if style is None:
        raise ValueError(f"{resolver_name} requires a persistent style.")
    if style in EVENT_ONLY_STYLES:
        raise ValueError(f"{resolver_name} cannot be used with event-only style {style!r}.")
    if style not in PERSISTENT_STYLES:
        column_for_style(style)


def _matching_rows(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> list[LanguageRow]:
    """Return ``rows`` filtered by optional ``style``/``role``/``tool_name``/``camera`` selectors."""
    return [
        row
        for row in rows
        if (style is None or row.get("style") == style)
        and (role is None or row.get("role") == role)
        and (tool_name is None or _row_has_tool_name(row, tool_name))
        and (camera is None or row.get("camera") == camera)
    ]


def _select_latest(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> LanguageRow | None:
    """Return the row tied for the latest ``timestamp`` (disambiguated by selectors)."""
    if not rows:
        return None
    rows = sorted(rows, key=_persistent_sort_key)
    latest_ts = _timestamp(rows[-1])
    return _select_one(
        [row for row in rows if _timestamp(row) == latest_ts],
        style=style,
        role=role,
        tool_name=tool_name,
        camera=camera,
        sort_key=_persistent_sort_key,
    )


def _select_one(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
    sort_key: Any,
) -> LanguageRow | None:
    """Return the single matching row, or raise if the selectors are ambiguous."""
    if not rows:
        return None
    if len(rows) > 1 and role is None and tool_name is None and camera is None:
        raise ValueError(
            f"Ambiguous resolver for style={style!r}; add role=..., tool_name=..., "
            f"or camera=... to disambiguate."
        )
    return sorted(rows, key=sort_key)[0]


def _persistent_sort_key(row: LanguageRow) -> tuple[float, str, str]:
    """Sort key for persistent rows: ``(timestamp, style, role)``."""
    return (_timestamp(row), row.get("style") or "", row.get("role") or "")


def _event_sort_key(row: LanguageRow) -> tuple[str, str]:
    """Sort key for event rows: ``(style, role)`` (timestamp is implicit in the frame)."""
    return (row.get("style") or "", row.get("role") or "")


def _timestamp(row: LanguageRow) -> float:
    """Extract a row's ``timestamp`` as a Python float (unwrapping numpy scalars)."""
    value = row["timestamp"]
    return float(value.item() if hasattr(value, "item") else value)


def _row_has_tool_name(row: LanguageRow, tool_name: str) -> bool:
    """Return ``True`` if any of the row's tool calls invokes ``tool_name``."""
    for tool_call in row.get("tool_calls") or []:
        if isinstance(tool_call, str):
            continue
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if isinstance(function, dict) and function.get("name") == tool_name:
            return True
    return False


def _normalize_rows(rows: Sequence[Any]) -> list[LanguageRow]:
    """Convert pyarrow scalars / mappings into a fresh list of plain dict rows."""
    normalized = []
    for row in rows:
        if row is None:
            continue
        if hasattr(row, "as_py"):
            row = row.as_py()
        if not isinstance(row, dict):
            raise TypeError(f"Language rows must be dictionaries, got {type(row).__name__}.")
        normalized.append(dict(row))
    return normalized
