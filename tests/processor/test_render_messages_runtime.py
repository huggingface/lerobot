#!/usr/bin/env python

"""Tests for runtime requests handled by the unified semantic-message renderer."""

from dataclasses import asdict

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe, render_message_turns
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import RenderMessagesStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT


def _subtask_recipe() -> TrainingRecipe:
    return TrainingRecipe(
        bindings={"context": "active_at(t, style=memory)"},
        messages=[
            MessageTurn(role="system", content="Robot assistant", stream="high_level"),
            MessageTurn(role="user", content="Task: ${task}", stream="high_level"),
            MessageTurn(
                role="assistant",
                content="Context: ${context}",
                stream="high_level",
                if_present="context",
            ),
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ],
    )


def _process(step: RenderMessagesStep, text: str, kind: str = "vqa", **values):
    transition = create_transition(complementary_data={QUERY_TEXT: text, QUERY_KIND: kind, **values})
    return step(transition)


def test_vqa_preserves_text_and_never_consults_the_recipe(monkeypatch):
    recipe = _subtask_recipe()

    def fail_if_called(kind: str):
        raise AssertionError(f"raw unexpectedly selected recipe target {kind}")

    monkeypatch.setattr(recipe, "prompt_turns", fail_if_called)
    text = "Use ${task} literally and keep {braces}."

    output = _process(RenderMessagesStep(recipe), text, "vqa")
    data = output[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages"] == [{"role": "user", "content": text}]
    assert QUERY_TEXT not in data
    assert QUERY_KIND not in data


def test_missing_query_kind_is_a_noop_for_ordinary_action_preprocessing():
    transition = create_transition(complementary_data={"task": "pick up the cup"})

    assert RenderMessagesStep(_subtask_recipe(), render_training=False)(transition) is transition


def test_subtask_renders_roles_wording_substitution_and_conditional_turns():
    step = RenderMessagesStep(_subtask_recipe())

    without_context = _process(step, "Clear {the} table", "next_subtask")[TransitionKey.COMPLEMENTARY_DATA][
        "messages"
    ]
    with_context = _process(step, "Clear {the} table", "next_subtask", context="cup already moved")[
        TransitionKey.COMPLEMENTARY_DATA
    ]["messages"]

    assert without_context == [
        {"role": "system", "content": "Robot assistant"},
        {"role": "user", "content": "Task: Clear {the} table"},
    ]
    assert with_context == [
        {"role": "system", "content": "Robot assistant"},
        {"role": "user", "content": "Task: Clear {the} table"},
        {"role": "assistant", "content": "Context: cup already moved"},
    ]


def test_missing_required_prefix_binding_has_a_clear_error():
    recipe = TrainingRecipe(
        bindings={"context": "active_at(t, style=memory)"},
        messages=[
            MessageTurn(role="user", content="${task}: ${context}", stream="high_level"),
            MessageTurn(role="assistant", content="${subtask}", stream="high_level", target=True),
        ],
    )

    with pytest.raises(ValueError, match="Unknown template binding: 'context'"):
        _process(RenderMessagesStep(recipe), "Clear the table", "next_subtask")


@pytest.mark.parametrize(
    "target_turn",
    [
        MessageTurn(role="user", content="${subtask}", stream="high_level", target=True),
        MessageTurn(
            role="assistant",
            content="not a subtask binding",
            stream="high_level",
            target=True,
            if_present="subtask",
        ),
    ],
)
def test_invalid_subtask_recipe_contract_is_rejected(target_turn: MessageTurn):
    recipe = TrainingRecipe(messages=[target_turn])

    with pytest.raises(ValueError, match=r"no assistant target turn supervising \$\{subtask\}"):
        _process(RenderMessagesStep(recipe), "Clear the table", "next_subtask")


def test_training_and_inference_use_the_same_rendered_prefix():
    recipe = _subtask_recipe()
    bindings = {"task": "Clear the table", "context": None, "subtask": "pick up the cup"}
    training_messages = render_message_turns(recipe.messages or [], bindings)["messages"]

    inference_messages = _process(RenderMessagesStep(recipe), "Clear the table", "next_subtask")[
        TransitionKey.COMPLEMENTARY_DATA
    ]["messages"]

    assert inference_messages == training_messages[:-1]
    assert training_messages[-1] == {"role": "assistant", "content": "pick up the cup"}


def test_recipe_is_prepared_once_and_step_is_stateless(monkeypatch):
    recipe_dict = asdict(_subtask_recipe())
    step = RenderMessagesStep(recipe_dict)  # type: ignore[arg-type]
    prepared_recipe = step.recipe

    def fail_if_reparsed(cls, data):
        raise AssertionError("recipe was reparsed during generation")

    monkeypatch.setattr(TrainingRecipe, "from_dict", classmethod(fail_if_reparsed))

    first = _process(step, "Clear the table", "next_subtask")[TransitionKey.COMPLEMENTARY_DATA]["messages"]
    second = _process(step, "What do you see?")[TransitionKey.COMPLEMENTARY_DATA]["messages"]

    assert first[-1]["content"] == "Task: Clear the table"
    assert second == [{"role": "user", "content": "What do you see?"}]
    assert step.recipe is prepared_recipe


def test_step_does_not_mutate_recipe_transition_or_runtime_state():
    recipe = _subtask_recipe()
    original_recipe = asdict(recipe)
    transition = create_transition(
        complementary_data={
            QUERY_TEXT: "Clear the table",
            QUERY_KIND: "next_subtask",
            "subtask": "currently reaching",
        }
    )

    output = RenderMessagesStep(recipe)(transition)

    assert asdict(recipe) == original_recipe
    assert "messages" not in transition[TransitionKey.COMPLEMENTARY_DATA]
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == "currently reaching"
    assert output[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == "currently reaching"


def test_unsupported_query_kind_is_rejected():
    with pytest.raises(ValueError, match="Unsupported query kind: 'caption'"):
        _process(RenderMessagesStep(), "Describe this", "caption")
