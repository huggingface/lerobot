#!/usr/bin/env python

"""Tests for semantic text-generation prompt rendering."""

from dataclasses import asdict

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.datasets.language_render import render_message_turns
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.processor.text_generation_processor import RenderGenerationPromptStep


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


def _render(step: RenderGenerationPromptStep, text: str, template: str = "raw", **values):
    transition = create_transition(complementary_data={"text": text, "text_template": template, **values})
    return step(transition)[TransitionKey.COMPLEMENTARY_DATA]["messages"]


def test_raw_preserves_text_and_never_consults_the_recipe(monkeypatch):
    recipe = _subtask_recipe()

    def fail_if_called(kind: str):
        raise AssertionError(f"raw unexpectedly selected recipe target {kind}")

    monkeypatch.setattr(recipe, "prompt_turns", fail_if_called)
    text = "Use ${task} literally and keep {braces}."

    assert _render(RenderGenerationPromptStep(recipe), text) == [{"role": "user", "content": text}]


def test_subtask_renders_roles_wording_substitution_and_conditional_turns():
    step = RenderGenerationPromptStep(_subtask_recipe())

    without_context = _render(step, "Clear {the} table", "subtask")
    with_context = _render(step, "Clear {the} table", "subtask", context="cup already moved")

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
        _render(RenderGenerationPromptStep(recipe), "Clear the table", "subtask")


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
        _render(RenderGenerationPromptStep(recipe), "Clear the table", "subtask")


def test_training_and_inference_use_the_same_rendered_prefix():
    recipe = _subtask_recipe()
    bindings = {"task": "Clear the table", "context": None, "subtask": "pick up the cup"}
    training_messages = render_message_turns(recipe.messages or [], bindings)["messages"]

    inference_messages = _render(RenderGenerationPromptStep(recipe), "Clear the table", "subtask")

    assert inference_messages == training_messages[:-1]
    assert training_messages[-1] == {"role": "assistant", "content": "pick up the cup"}


def test_recipe_is_prepared_once_and_step_is_stateless(monkeypatch):
    recipe_dict = asdict(_subtask_recipe())
    step = RenderGenerationPromptStep(recipe_dict)  # type: ignore[arg-type]
    prepared_recipe = step.recipe

    def fail_if_reparsed(cls, data):
        raise AssertionError("recipe was reparsed during generation")

    monkeypatch.setattr(TrainingRecipe, "from_dict", classmethod(fail_if_reparsed))

    first = _render(step, "Clear the table", "subtask")
    second = _render(step, "What do you see?")

    assert first[-1]["content"] == "Task: Clear the table"
    assert second == [{"role": "user", "content": "What do you see?"}]
    assert step.recipe is prepared_recipe


def test_step_does_not_mutate_recipe_transition_or_runtime_state():
    recipe = _subtask_recipe()
    original_recipe = asdict(recipe)
    transition = create_transition(
        complementary_data={
            "text": "Clear the table",
            "text_template": "subtask",
            "subtask": "currently reaching",
        }
    )

    output = RenderGenerationPromptStep(recipe)(transition)

    assert asdict(recipe) == original_recipe
    assert "messages" not in transition[TransitionKey.COMPLEMENTARY_DATA]
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == "currently reaching"
    assert output[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == "currently reaching"
