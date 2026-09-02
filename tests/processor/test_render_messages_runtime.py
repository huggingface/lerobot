#!/usr/bin/env python

"""Runtime behavior of the unified recipe/message processor."""

from dataclasses import asdict

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe, render_message_turns
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import RenderMessagesStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT


def _recipe() -> TrainingRecipe:
    return TrainingRecipe(
        messages=[
            MessageTurn(role="system", content="Robot assistant", stream="high_level"),
            MessageTurn(role="user", content="Goal: ${task}", stream="high_level"),
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )


def _render(kind: str, text: str, *, recipe: TrainingRecipe | None = None):
    transition = create_transition(complementary_data={QUERY_KIND: kind, QUERY_TEXT: text})
    return RenderMessagesStep(recipe, render_training=False)(transition)


def test_vqa_preserves_caller_text_and_consumes_request_metadata():
    text = "Use ${task} literally and keep {braces}."

    output = _render("vqa", text)
    data = output[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages"] == [{"role": "user", "content": text}]
    assert QUERY_KIND not in data
    assert QUERY_TEXT not in data


def test_next_subtask_uses_the_training_recipe_prefix():
    recipe = _recipe()
    training = render_message_turns(
        recipe.messages or [],
        {"task": "tidy", "subtask": "pick up cup"},
    )["messages"]

    inference = _render("next_subtask", "tidy", recipe=recipe)[TransitionKey.COMPLEMENTARY_DATA]["messages"]

    assert inference == training[:-1]
    assert training[-1] == {"role": "assistant", "content": "pick up cup"}


def test_ordinary_action_inputs_and_existing_messages_pass_through():
    step = RenderMessagesStep(_recipe(), render_training=False)
    action_transition = create_transition(complementary_data={"task": "pick up cup"})
    messages_transition = create_transition(
        complementary_data={"messages": [{"role": "user", "content": "already rendered"}]}
    )

    assert step(action_transition) is action_transition
    assert step(messages_transition) is messages_transition


def test_runtime_rendering_is_stateless_and_does_not_mutate_inputs():
    recipe = _recipe()
    original_recipe = asdict(recipe)
    transition = create_transition(
        complementary_data={QUERY_KIND: "next_subtask", QUERY_TEXT: "tidy", "task": "old task"}
    )

    output = RenderMessagesStep(recipe, render_training=False)(transition)

    assert asdict(recipe) == original_recipe
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"] == "old task"
    assert output[TransitionKey.COMPLEMENTARY_DATA]["task"] == "old task"


def test_runtime_requests_validate_kind_text_and_recipe():
    with pytest.raises(ValueError, match="requires a checkpoint recipe"):
        _render("next_subtask", "tidy")
    with pytest.raises(ValueError, match="Unsupported query kind"):
        _render("caption", "describe this")
    with pytest.raises(TypeError, match="query_text"):
        RenderMessagesStep(render_training=False)(
            create_transition(complementary_data={QUERY_KIND: "vqa", QUERY_TEXT: 123})
        )


def test_runtime_query_cannot_mix_with_raw_training_language():
    transition = create_transition(
        complementary_data={QUERY_KIND: "vqa", QUERY_TEXT: "what is visible?", "language_events": []}
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        RenderMessagesStep(render_training=False)(transition)
