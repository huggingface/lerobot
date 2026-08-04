#!/usr/bin/env python

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.recipe import MessageTurn, TrainingRecipe  # noqa: E402
from lerobot.lerobot_types import TransitionKey  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.processor.render_messages_processor import (  # noqa: E402
    RenderMessagesStep,
    _fallback_low_level_render,
    _select_batch_indices,
)


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


def test_render_messages_step_rejects_mismatched_non_empty_language_batches():
    recipe = TrainingRecipe(
        messages=[
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            )
        ]
    )
    transition = create_transition(
        complementary_data={
            "timestamp": torch.tensor([0.0, 1.0, 2.0]),
            "language_persistent": [[], []],
            "language_events": [[{"style": "unmatched"}], [], []],
        }
    )

    with pytest.raises(ValueError, match="must have equal lengths"):
        RenderMessagesStep(recipe)(transition)


def test_select_batch_indices_slices_numpy_action():
    action = np.arange(6).reshape(3, 2)
    transition = create_transition(action=action)

    selected = _select_batch_indices(transition, [2, 0], batch_size=3)

    np.testing.assert_array_equal(selected[TransitionKey.ACTION], action[[2, 0]])


def test_select_batch_indices_slices_robot_action_dict():
    transition = create_transition(
        action={
            "joints": np.arange(6).reshape(3, 2),
            "gripper": torch.tensor([[0.0], [1.0], [2.0]]),
        }
    )

    selected = _select_batch_indices(transition, [2, 0], batch_size=3)

    np.testing.assert_array_equal(selected[TransitionKey.ACTION]["joints"], np.array([[4, 5], [0, 1]]))
    assert torch.equal(selected[TransitionKey.ACTION]["gripper"], torch.tensor([[2.0], [0.0]]))


def test_select_batch_indices_rejects_misaligned_list():
    transition = create_transition(complementary_data={"task": ["one", "two"]})

    with pytest.raises(ValueError, match="expected 3 values, got 2"):
        _select_batch_indices(transition, [2, 0], batch_size=3)


def test_fallback_low_level_render_rejects_partially_missing_task_batch():
    with pytest.raises(ValueError, match=r"missing task at indices \[1\]"):
        _fallback_low_level_render(["pick cube", None, "place cube"])
