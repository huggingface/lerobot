#!/usr/bin/env python

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.configs.recipe import MessageTurn, TrainingRecipe  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.render_messages_processor import RenderMessagesStep  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402


def test_render_messages_step_renders_task_fallback_without_language_columns():
    """No language columns + a task string → low-level task fallback render,
    matching what the policy sees at eval time on unannotated observations."""
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(complementary_data={"task": "do it"})

    out = RenderMessagesStep(recipe)(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages"] == [{"role": "user", "content": "do it"}]
    assert data["message_streams"] == ["low_level"]
    assert data["target_message_indices"] == []
    assert data["task"] == "do it"


def test_render_messages_step_noops_without_language_columns_or_task():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(complementary_data={})

    assert RenderMessagesStep(recipe)(transition) == transition


def test_render_messages_step_renders_and_drops_raw_language():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="low_level", target=True),
        ]
    )
    transition = create_transition(
        complementary_data={
            "task": "do it",
            "timestamp": torch.tensor(0.0),
            "index": torch.tensor(7),
            "language_persistent": [
                {
                    "role": "assistant",
                    "content": "reach carefully",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                }
            ],
            "language_events": [],
        }
    )

    out = RenderMessagesStep(recipe)(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert "language_persistent" not in data
    assert "language_events" not in data
    assert data["messages"][-1]["content"] == "reach carefully"
    assert data["message_streams"] == ["high_level", "low_level"]
    assert data["target_message_indices"] == [1]


def test_render_messages_step_falls_back_to_low_level_task_when_recipe_misses():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )
    transition = create_transition(
        complementary_data={
            "task": "pick the cube",
            "timestamp": torch.tensor(0.0),
            "index": torch.tensor(7),
            "language_persistent": [],
            "language_events": [{"style": "unmatched", "timestamp": 0.0}],
        }
    )

    out = RenderMessagesStep(recipe)(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages"] == [{"role": "user", "content": "pick the cube"}]
    assert data["message_streams"] == ["low_level"]
    assert data["target_message_indices"] == []


def test_render_messages_step_falls_back_per_sample_in_batched_language():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )
    transition = create_transition(
        action=torch.arange(4).reshape(2, 2),
        complementary_data={
            "task": ["pick the cube", "open the drawer"],
            "timestamp": torch.tensor([0.0, 1.0]),
            "index": torch.tensor([7, 8]),
            "language_persistent": [[], []],
            "language_events": [
                [{"style": "unmatched", "timestamp": 0.0}],
                [{"style": "unmatched", "timestamp": 1.0}],
            ],
        },
    )

    out = RenderMessagesStep(recipe)(transition)
    data = out[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages"] == [
        [{"role": "user", "content": "pick the cube"}],
        [{"role": "user", "content": "open the drawer"}],
    ]
    assert data["message_streams"] == [["low_level"], ["low_level"]]
    assert data["target_message_indices"] == [[], []]
