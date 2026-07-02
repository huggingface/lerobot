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

"""Tests for RECAP advantage conditioning recipes."""

from __future__ import annotations

from pathlib import Path

from lerobot.configs.recipe import load_recipe
from lerobot.datasets.language_render import render_sample

RECIPES_DIR = Path(__file__).resolve().parents[2] / "src" / "lerobot" / "configs" / "recipes"


def _persistent_rows(advantage: str | None = None):
    """Build minimal persistent rows with optional advantage."""
    rows = [
        {
            "role": "user",
            "content": "pick up the cup",
            "style": "task_aug",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": "reaching for the cup",
            "style": "subtask",
            "timestamp": 0.0,
            "camera": None,
            "tool_calls": None,
        },
    ]
    if advantage is not None:
        rows.append(
            {
                "role": "user",
                "content": advantage,
                "style": "advantage",
                "timestamp": 0.0,
                "camera": None,
                "tool_calls": None,
            }
        )
    return rows


def test_recap_advantage_recipe_loads():
    """The recap_advantage.yaml recipe loads without errors."""
    recipe = load_recipe(RECIPES_DIR / "recap_advantage.yaml")
    assert recipe.messages is not None
    assert len(recipe.messages) == 3
    assert recipe.bindings == {"advantage": "active_at(t, style=advantage)"}


def test_advantage_present_renders_indicator():
    """When advantage annotation exists, the prompt includes 'Advantage: positive'."""
    recipe = load_recipe(RECIPES_DIR / "recap_advantage.yaml")
    result = render_sample(
        recipe=recipe,
        persistent=_persistent_rows(advantage="positive"),
        events=[],
        t=0.5,
        sample_idx=0,
        task="pick up the cup",
    )
    assert result is not None
    messages = result["messages"]
    assert len(messages) == 3
    assert messages[1]["content"] == "Advantage: positive"


def test_advantage_negative_renders_indicator():
    """Negative advantage also appears in the prompt."""
    recipe = load_recipe(RECIPES_DIR / "recap_advantage.yaml")
    result = render_sample(
        recipe=recipe,
        persistent=_persistent_rows(advantage="negative"),
        events=[],
        t=0.5,
        sample_idx=0,
        task="pick up the cup",
    )
    assert result is not None
    messages = result["messages"]
    assert messages[1]["content"] == "Advantage: negative"


def test_advantage_absent_skips_turn():
    """When no advantage annotation exists (dropout), the advantage turn is skipped."""
    recipe = load_recipe(RECIPES_DIR / "recap_advantage.yaml")
    result = render_sample(
        recipe=recipe,
        persistent=_persistent_rows(advantage=None),
        events=[],
        t=0.5,
        sample_idx=0,
        task="pick up the cup",
    )
    assert result is not None
    messages = result["messages"]
    # Only task + subtask, no advantage turn
    assert len(messages) == 2
    assert messages[0]["content"] == "pick up the cup"
    assert messages[1]["content"] == "reaching for the cup"


def test_advantage_absent_still_has_target():
    """Even without advantage, the target message (subtask) is preserved."""
    recipe = load_recipe(RECIPES_DIR / "recap_advantage.yaml")
    result = render_sample(
        recipe=recipe,
        persistent=_persistent_rows(advantage=None),
        events=[],
        t=0.5,
        sample_idx=0,
        task="pick up the cup",
    )
    assert result is not None
    assert result["target_message_indices"] == [1]


def test_blend_recipe_loads():
    """The blend recipe has two components with correct weights."""
    recipe = load_recipe(RECIPES_DIR / "recap_advantage_blend.yaml")
    assert recipe.blend is not None
    assert "advantage_conditioned" in recipe.blend
    assert "unconditional" in recipe.blend
    assert recipe.blend["advantage_conditioned"].weight == 0.7
    assert recipe.blend["unconditional"].weight == 0.3
