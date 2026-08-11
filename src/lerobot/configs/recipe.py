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
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

MessageRole = Literal["user", "assistant", "system", "tool"]
MessageStream = Literal["high_level", "low_level"]
RecipeRoute = Literal["vqa"]

DEFAULT_BINDINGS = {
    "subtask": "active_at(t, style=subtask)",
    "memory": "active_at(t, style=memory)",
    "plan": "active_at(t, style=plan)",
    "speech": "emitted_at(t, role=assistant, tool_name=say)",
    "interjection": "emitted_at(t, style=interjection)",
    "vqa": "emitted_at(t, style=vqa, role=assistant)",
    "vqa_query": "emitted_at(t, style=vqa, role=user)",
}

PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
"""``${name}`` placeholder pattern used by both recipe binding-reference
discovery (here) and rendered-message substitution (in ``language_render``)."""

_VALID_ROLES = frozenset(get_args(MessageRole))
_VALID_STREAMS = frozenset(get_args(MessageStream))
_VALID_ROUTES = frozenset(get_args(RecipeRoute))


@dataclass
class MessageTurn:
    """A single chat-style turn in a recipe template.

    ``content`` may be a plain string, a list of HF-style multimodal blocks, or
    ``None`` when ``tool_calls_from`` supplies tool-call payloads instead.
    ``stream`` tags the turn for downstream filtering, ``target`` flags it as a
    training target, and ``if_present`` skips the turn when the named binding
    resolves to ``None``.
    """

    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    stream: MessageStream | None = None
    target: bool = False
    if_present: str | None = None
    tool_calls_from: str | None = None

    def __post_init__(self) -> None:
        """Validate role, stream, and content after dataclass construction."""
        if self.role not in _VALID_ROLES:
            raise ValueError(f"Unsupported message role: {self.role!r}")
        # ``stream`` is typed Optional only so the dataclass can keep its
        # field ordering, but recipes must always tag every turn with a
        # stream — the renderer's ``_validate_rendered`` would reject
        # ``None`` later on. Fail at construction so the bad recipe is
        # caught at YAML load time rather than at the first sample.
        if self.stream is None:
            raise ValueError(
                f"MessageTurn(role={self.role!r}) is missing a stream — "
                f"every turn must declare one of {sorted(_VALID_STREAMS)}."
            )
        if self.stream not in _VALID_STREAMS:
            raise ValueError(f"Unsupported message stream: {self.stream!r}")
        if self.content is None and self.tool_calls_from is None:
            raise ValueError("MessageTurn.content is required unless tool_calls_from is set.")
        if self.content is not None and not isinstance(self.content, str | list):
            raise TypeError("MessageTurn.content must be a string, a list of HF-style blocks, or None.")
        if isinstance(self.content, list):
            for block in self.content:
                if not isinstance(block, dict) or "type" not in block:
                    raise ValueError(
                        "Multimodal content blocks must be HF-style dictionaries with a type key."
                    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageTurn:
        """Construct a :class:`MessageTurn` from a plain dictionary."""
        return cls(**data)


@dataclass
class TrainingRecipe:
    """A recipe describing how to render training samples from language rows.

    A recipe is either a *message recipe* (``messages`` plus optional
    ``bindings``) or a *blend recipe* (``blend`` mapping names to weighted
    sub-recipes). ``weight`` and ``route`` are only meaningful inside a blend;
    ``route: vqa`` gives sparse VQA annotations priority over normal weighted
    selection.
    """

    messages: list[MessageTurn] | None = None
    bindings: dict[str, str] | None = None
    blend: dict[str, TrainingRecipe] | None = None
    weight: float | None = None
    route: RecipeRoute | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one of ``messages`` or ``blend`` is set."""
        if self.messages is not None and self.blend is not None:
            raise ValueError("TrainingRecipe must set only one of messages or blend.")
        if self.messages is None and self.blend is None:
            raise ValueError("TrainingRecipe must set one of messages or blend.")
        if self.route is not None and self.route not in _VALID_ROUTES:
            raise ValueError(f"Unsupported recipe route: {self.route!r}")
        if self.blend is not None and self.route is not None:
            raise ValueError("TrainingRecipe.route may only be set on a message recipe inside a blend.")

        if self.messages is not None:
            self._validate_message_recipe()
        if self.blend is not None:
            self._validate_blend_recipe()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingRecipe:
        """Construct a :class:`TrainingRecipe` from a nested dictionary."""
        data = dict(data)
        if data.get("messages") is not None:
            data["messages"] = [
                turn if isinstance(turn, MessageTurn) else MessageTurn.from_dict(turn)
                for turn in data["messages"]
            ]
        if data.get("blend") is not None:
            data["blend"] = {
                name: recipe if isinstance(recipe, TrainingRecipe) else cls.from_dict(recipe)
                for name, recipe in data["blend"].items()
            }
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingRecipe:
        """Load a :class:`TrainingRecipe` from a YAML file at ``path``."""
        import yaml  # type: ignore[import-untyped]

        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Recipe YAML must contain a mapping at the top level: {path}")
        return cls.from_dict(data)

    def _validate_message_recipe(self) -> None:
        """Validate bindings and require text or low-level action supervision."""
        if self.messages is None:
            raise ValueError("Cannot validate a message recipe without messages.")
        known_bindings = set(DEFAULT_BINDINGS) | set(self.bindings or {}) | {"task"}

        for turn in self.messages:
            missing = self._referenced_bindings(turn) - known_bindings
            if missing:
                raise ValueError(f"MessageTurn references unknown binding(s): {sorted(missing)}")

        has_target = any(turn.target for turn in self.messages)
        has_low_level = any(turn.stream == "low_level" for turn in self.messages)
        if not (has_target or has_low_level):
            raise ValueError(
                "Message recipes must contain at least one supervised turn — "
                "either ``target: true`` (text CE) or ``stream: low_level`` "
                "(flow/action loss)."
            )

    def _validate_blend_recipe(self) -> None:
        """Ensure each blend component is a non-empty, weighted message recipe."""
        if self.blend is None:
            raise ValueError("Cannot validate a blend recipe without blend components.")
        if not self.blend:
            raise ValueError("Blend recipes must contain at least one component.")

        for name, recipe in self.blend.items():
            if recipe.blend is not None:
                raise ValueError(f"Blend component {name!r} cannot itself define a blend.")
            if recipe.messages is None:
                raise ValueError(f"Blend component {name!r} must define messages.")
            if recipe.weight is None:
                raise ValueError(f"Blend component {name!r} must define weight.")
            if recipe.weight <= 0:
                raise ValueError(f"Blend component {name!r} must have a positive weight.")

    def referenced_binding_names(self) -> set[str]:
        """Names of every binding referenced by this recipe's message turns."""
        names: set[str] = set()
        for turn in self.messages or []:
            names |= self._referenced_bindings(turn)
        return names

    def prompt_turns(self, kind: str) -> list[MessageTurn]:
        """The turns preceding the target turn that supervises the ``kind`` binding.

        Searches this recipe's messages — or each blend component in declaration
        order — for the first assistant ``target: true`` turn whose content
        references the ``kind`` binding, and returns every turn before it. This is
        a lower-level recipe utility; the policy API maps its constrained public
        templates to target bindings internally.
        """
        components = [self] if self.messages is not None else list((self.blend or {}).values())
        for component in components:
            turns = component.messages or []
            for index, turn in enumerate(turns):
                if (
                    turn.target
                    and turn.role == "assistant"
                    and kind in _placeholders_in_content(turn.content)
                ):
                    return turns[:index]
        supervised = sorted(
            {
                name
                for component in components
                for turn in component.messages or []
                if turn.target and turn.role == "assistant"
                for name in _placeholders_in_content(turn.content)
            }
        )
        raise ValueError(
            f"Recipe has no assistant target turn supervising ${{{kind}}}. Supervised bindings: {supervised}."
        )

    def _referenced_bindings(self, turn: MessageTurn) -> set[str]:
        """Return the binding names that ``turn`` references via placeholders or attributes."""
        names: set[str] = set()
        if turn.if_present is not None:
            names.add(turn.if_present)
        if turn.tool_calls_from is not None:
            names.add(turn.tool_calls_from)
        names.update(_placeholders_in_content(turn.content))
        return names


def _placeholders_in_content(content: str | list[dict[str, Any]] | None) -> set[str]:
    """Return the set of ``${name}`` placeholders found anywhere in ``content``."""
    if content is None:
        return set()
    if isinstance(content, str):
        return set(PLACEHOLDER_RE.findall(content))

    names: set[str] = set()
    for block in content:
        for value in block.values():
            if isinstance(value, str):
                names.update(PLACEHOLDER_RE.findall(value))
    return names


def render_message_turns(
    turns: Sequence[MessageTurn],
    bindings: dict[str, Any],
) -> dict[str, list[Any]]:
    """Render recipe turns with substitution shared by training and inference.

    This lightweight primitive intentionally has no dataset dependency. Training
    renders complete recipes and validates supervision afterwards; the policy
    input processor uses it for the prefix before a text-generation target.
    """
    messages: list[dict[str, Any]] = []
    streams: list[str | None] = []
    target_indices: list[int] = []

    for turn in turns:
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

    return {
        "messages": messages,
        "message_streams": streams,
        "target_message_indices": target_indices,
    }


def _render_content(content: str | list[dict[str, Any]], bindings: dict[str, Any]) -> Any:
    """Substitute bindings into text or each string field of multimodal blocks."""
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


def _substitute(template: str, bindings: dict[str, Any]) -> str:
    """Replace ``${name}`` placeholders with string or language-row values."""

    def replace(match: re.Match[str]) -> str:
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


def load_recipe(path: str | Path) -> TrainingRecipe:
    """Load a :class:`TrainingRecipe` from a YAML file at ``path``."""
    return TrainingRecipe.from_yaml(path)


def resolve_recipe_override(
    recipe: TrainingRecipe | None,
    recipe_path: str | Path | None,
) -> TrainingRecipe | None:
    """Resolve an external recipe while keeping checkpoint-inline recipes portable.

    Fresh configurations fail when their requested override does not exist. A
    reloaded checkpoint may retain the original training-machine path alongside
    its serialized recipe; in that case the inline recipe remains authoritative.
    """
    if recipe_path is None:
        return recipe
    try:
        return load_recipe(recipe_path)
    except FileNotFoundError:
        if recipe is None:
            raise
        return recipe


def language_recipe_enabled(
    *,
    use_language_recipe: bool = False,
    recipe_path: str | Path | None = None,
) -> bool:
    """Whether training requested a built-in recipe or an external override."""
    return use_language_recipe or recipe_path is not None
