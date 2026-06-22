#!/usr/bin/env python

"""Tests for RenderedMessagesToTaskStep and PI05 pipeline integration with advantage."""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.configs.recipe import MessageTurn, TrainingRecipe  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.render_messages_processor import RenderMessagesStep  # noqa: E402
from lerobot.processor.rendered_messages_to_task import RenderedMessagesToTaskStep  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402


def test_rendered_messages_to_task_noops_without_messages():
    """Without messages key, the step is a no-op."""
    transition = create_transition(complementary_data={"task": "pick up the cup"})
    step = RenderedMessagesToTaskStep()
    out = step(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]
    assert data["task"] == "pick up the cup"


def test_rendered_messages_to_task_extracts_user_content():
    """Extracts user-role message content and joins with newline."""
    transition = create_transition(
        complementary_data={
            "task": "original task",
            "messages": [
                {"role": "user", "content": "pick up the cup"},
                {"role": "user", "content": "Advantage: positive"},
                {"role": "assistant", "content": "reach for cup"},
            ],
            "message_streams": ["high_level", "high_level", "low_level"],
            "target_message_indices": [2],
        }
    )
    step = RenderedMessagesToTaskStep()
    out = step(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["task"] == "pick up the cup\nAdvantage: positive"
    assert "messages" not in data
    assert "message_streams" not in data
    assert "target_message_indices" not in data


def test_rendered_messages_to_task_handles_multimodal_blocks():
    """Extracts text from HF multimodal content blocks."""
    transition = create_transition(
        complementary_data={
            "task": "original",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": "describe this"},
                    ],
                },
                {"role": "assistant", "content": "a cup on a table"},
            ],
            "message_streams": ["high_level", "low_level"],
            "target_message_indices": [1],
        }
    )
    step = RenderedMessagesToTaskStep()
    out = step(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["task"] == "describe this"


def test_rendered_messages_to_task_preserves_list_task_format():
    """When original task is a list (batched), output is also a list."""
    transition = create_transition(
        complementary_data={
            "task": ["task1", "task2"],
            "messages": [
                {"role": "user", "content": "rendered task"},
                {"role": "assistant", "content": "do it", "target": True},
            ],
            "message_streams": ["high_level", "low_level"],
            "target_message_indices": [1],
        }
    )
    step = RenderedMessagesToTaskStep()
    out = step(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["task"] == ["rendered task", "rendered task"]


def test_full_render_then_flatten_pipeline():
    """RenderMessagesStep + RenderedMessagesToTaskStep produces correct task string."""
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(
                role="user",
                content="Advantage: ${advantage}",
                stream="high_level",
                if_present="advantage",
            ),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(
        complementary_data={
            "task": "pick up the cup",
            "timestamp": torch.tensor(0.5),
            "index": torch.tensor(0),
            "language_persistent": [
                {
                    "role": "assistant",
                    "content": "reach for the cup",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                },
                {
                    "role": "user",
                    "content": "positive",
                    "style": "advantage",
                    "timestamp": 0.1,
                    "camera": None,
                    "tool_calls": None,
                },
            ],
            "language_events": [],
        }
    )

    # Step 1: Render recipe
    rendered = RenderMessagesStep(recipe=recipe)(transition)
    # Step 2: Flatten to task string
    out = RenderedMessagesToTaskStep()(rendered)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert "pick up the cup" in data["task"]
    assert "Advantage: positive" in data["task"]


def test_full_render_advantage_absent_skips_turn():
    """When advantage row is absent, the advantage turn is skipped via if_present."""
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(
                role="user",
                content="Advantage: ${advantage}",
                stream="high_level",
                if_present="advantage",
            ),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(
        complementary_data={
            "task": "pick up the cup",
            "timestamp": torch.tensor(0.5),
            "index": torch.tensor(0),
            "language_persistent": [
                {
                    "role": "assistant",
                    "content": "reach for the cup",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                },
            ],
            "language_events": [],
        }
    )

    rendered = RenderMessagesStep(recipe=recipe)(transition)
    out = RenderedMessagesToTaskStep()(rendered)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["task"] == "pick up the cup"
    assert "Advantage" not in data["task"]
