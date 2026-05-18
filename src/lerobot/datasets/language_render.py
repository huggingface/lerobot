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

from lerobot.configs.recipe import DEFAULT_BINDINGS, PLACEHOLDER_RE, TrainingRecipe
from lerobot.utils.utils import unwrap_scalar

from .language import LANGUAGE_PERSISTENT, column_for_style

LanguageRow = dict[str, Any]
RenderedMessages = dict[str, list[Any]]

_RESOLVER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")


def active_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """Return the persistent row of ``style`` that is active at time ``t``.

    A persistent row is "active" at ``t`` when its own ``timestamp`` is the
    most recent one ``<= t`` for the given ``style``/``role``/``tool_name``/
    ``camera`` selector. Only valid for persistent styles.
    """
    _validate_persistent_resolver("active_at", style)
    matches = [
        row
        for row in _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera)
        if _timestamp(row) <= t
    ]
    if not matches:
        return None
    latest_ts = max(_timestamp(row) for row in matches)
    return _select_one(
        [row for row in matches if _timestamp(row) == latest_ts],
        style=style,
        role=role,
        tool_name=tool_name,
        camera=camera,
    )


EMITTED_AT_TOLERANCE_S = 0.1
"""Half-window for matching persistent rows to a frame timestamp in
``emitted_at``. Persistent timestamps come from parquet (float32) and ``t``
is also a float32 from parquet, so in the ideal hot path an exact match
would suffice — but any caller that derives ``t`` arithmetically (e.g.
``frame_idx / fps``) breaks bit-equality. A 0.1 s tolerance covers
common arithmetic drift without admitting frames that are visibly far
apart at typical control rates (30–100 Hz). This does mean two persistent
rows of the same selector emitted within 0.1 s of each other cannot be
told apart by ``emitted_at`` — acceptable because persistent annotations
(subtask / plan / memory transitions) change on a human-action timescale,
not at the camera frame rate."""


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
    is within ``EMITTED_AT_TOLERANCE_S`` of ``t`` (see that constant for why
    we use a tolerance instead of bit-equality). For event styles, the
    ``events`` list is assumed to come from the dataset row at frame ``t``
    (event rows carry no timestamp of their own), so all matching event rows
    are considered emitted at ``t``. ``camera`` filters by the row's
    ``camera`` field — required to disambiguate when multiple view-dependent
    rows share ``(t, role)`` across cameras.
    """
    if column_for_style(style) == LANGUAGE_PERSISTENT:
        matches = [
            row
            for row in _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera)
            if abs(_timestamp(row) - t) <= EMITTED_AT_TOLERANCE_S
        ]
    else:
        matches = _matching_rows(events, style=style, role=role, tool_name=tool_name, camera=camera)
    return _select_one(matches, style=style, role=role, tool_name=tool_name, camera=camera)


def nth_prev(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
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
    return _nth_relative("nth_prev", t, persistent, style, -offset, role, tool_name, camera)


def nth_next(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
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
    return _nth_relative("nth_next", t, persistent, style, offset, role, tool_name, camera)


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
        "task": _resolve_task(task, dataset_ctx, persistent=persistent, sample_idx=sample_idx),
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

    aug_rows = [r for r in persistent if r.get("style") == "task_aug" and r.get("role") == "user"]
    if aug_rows:
        # Deterministic, blake2b-based pick keyed on sample_idx so the
        # rotation is reproducible across runs (Python's built-in ``hash``
        # is process-randomized).
        digest = hashlib.blake2b(f"task_aug:{sample_idx}".encode(), digest_size=8).digest()
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

    if name == "emitted_at":
        return emitted_at(t, persistent=persistent, events=events, **kwargs)
    if name == "active_at":
        return active_at(t, persistent=persistent, **kwargs)
    if name == "nth_prev":
        return nth_prev(t, persistent=persistent, **kwargs)
    if name == "nth_next":
        return nth_next(t, persistent=persistent, **kwargs)
    raise ValueError(f"Unknown language resolver: {name!r}")


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

    if not target_indices:
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

    return PLACEHOLDER_RE.sub(replace, template)


def _validate_rendered(rendered: RenderedMessages) -> None:
    """Sanity-check the rendered output for stream/target alignment."""
    messages = rendered["messages"]
    streams = rendered["message_streams"]
    target_indices = rendered["target_message_indices"]

    if len(streams) != len(messages):
        raise ValueError("message_streams must be aligned with messages.")
    if not target_indices:
        raise ValueError("Rendered samples must contain at least one target message.")
    for idx in target_indices:
        if idx < 0 or idx >= len(messages):
            raise ValueError(f"Target message index {idx} is out of bounds.")
    # ``stream`` is enforced non-None at MessageTurn construction time
    # (see ``MessageTurn.__post_init__``), so a missing stream here would
    # mean the dataclass invariant was bypassed; no need to re-check.


def _nth_relative(
    name: str,
    t: float,
    persistent: Sequence[LanguageRow],
    style: str | None,
    offset: int,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> LanguageRow | None:
    """Shared body for ``nth_prev`` / ``nth_next`` with signed ``offset``."""
    _validate_persistent_resolver(name, style)
    if abs(offset) < 1:
        raise ValueError(f"{name} offset must be non-zero.")

    rows = sorted(
        _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera),
        key=_row_sort_key,
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


def _validate_persistent_resolver(name: str, style: str | None) -> None:
    """Reject calls with missing or event-only ``style`` for persistent resolvers."""
    if style is None:
        raise ValueError(f"{name} requires a persistent style.")
    if column_for_style(style) != LANGUAGE_PERSISTENT:
        raise ValueError(f"{name} cannot be used with event-only style {style!r}.")


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


def _select_one(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> LanguageRow | None:
    """Return the single matching row, or raise if the resolver is ambiguous.

    Multiple matches always raise — even when the caller already passed
    some selectors — because remaining ambiguity means the data has
    several rows that look identical to the resolver and the caller
    needs to pin down a specific one (e.g. add ``camera=...`` for VQA
    rows shared across cameras).
    """
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous resolver for style={style!r} role={role!r} "
            f"tool_name={tool_name!r} camera={camera!r}: {len(rows)} matching rows. "
            f"Add a selector that distinguishes them."
        )
    return rows[0]


def _row_sort_key(row: LanguageRow) -> tuple[float, str, str]:
    """Stable sort key for both persistent and event rows.

    Event rows lack ``timestamp`` (it is implicit in the frame), so default
    to ``0.0`` — within a single frame all event rows share the same sort
    bucket and are tiebroken by ``(style, role)``.
    """
    timestamp = row.get("timestamp")
    ts = float(unwrap_scalar(timestamp)) if timestamp is not None else 0.0
    return (ts, row.get("style") or "", row.get("role") or "")


def _timestamp(row: LanguageRow) -> float:
    """Extract a row's ``timestamp`` as a Python float (unwrapping numpy scalars)."""
    return float(unwrap_scalar(row["timestamp"]))


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
