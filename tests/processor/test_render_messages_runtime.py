#!/usr/bin/env python

"""Runtime behavior of the runtime message processor."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import RenderRuntimeMessagesStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT

if TYPE_CHECKING:
    from lerobot.datasets.recipe import TrainingRecipe


@pytest.fixture
def dataset_dependencies():
    pytest.importorskip("datasets", reason="recipes require lerobot[dataset]")
    pytest.importorskip("av", reason="recipes require lerobot[dataset]")


def _recipe() -> TrainingRecipe:
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe

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
    return RenderRuntimeMessagesStep(recipe)(transition)


def test_vqa_preserves_caller_text_and_consumes_request_metadata():
    text = "Use ${task} literally and keep {braces}."

    output = _render("vqa", text)
    data = output[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages_rendered"] == [{"role": "user", "content": text}]
    assert QUERY_KIND not in data
    assert QUERY_TEXT not in data


def test_next_subtask_uses_the_training_recipe_prefix(dataset_dependencies):
    from lerobot.datasets.recipe import render_message_turns

    recipe = _recipe()
    training = render_message_turns(
        recipe.messages or [],
        {"task": "tidy", "subtask": "pick up cup"},
    )["messages_rendered"]

    inference = _render("next_subtask", "tidy", recipe=recipe)[TransitionKey.COMPLEMENTARY_DATA][
        "messages_rendered"
    ]

    assert inference == training[:-1]
    assert training[-1] == {"role": "assistant", "content": "pick up cup"}


def test_next_subtask_tolerates_optional_runtime_bindings_that_are_absent(dataset_dependencies):
    from lerobot.datasets.recipe import MessageTurn, TrainingRecipe

    recipe = TrainingRecipe(
        messages=[
            MessageTurn(role="system", content="Memory: ${memory}", stream="high_level"),
            MessageTurn(
                role="assistant",
                content="Plan: ${plan}",
                stream="high_level",
                if_present="plan",
            ),
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

    data = _render("next_subtask", "tidy", recipe=recipe)[TransitionKey.COMPLEMENTARY_DATA]

    assert data["messages_rendered"] == [
        {"role": "system", "content": "Memory: "},
        {"role": "user", "content": "Goal: tidy"},
    ]


def test_ordinary_action_inputs_and_existing_messages_pass_through():
    step = RenderRuntimeMessagesStep()
    action_transition = create_transition(complementary_data={"task": "pick up cup"})
    messages_transition = create_transition(
        complementary_data={"messages_rendered": [{"role": "user", "content": "already rendered"}]}
    )

    assert step(action_transition) == action_transition
    assert step(messages_transition) == messages_transition


def test_runtime_rendering_is_stateless_and_does_not_mutate_inputs(dataset_dependencies):
    recipe = _recipe()
    original_recipe = asdict(recipe)
    transition = create_transition(
        complementary_data={QUERY_KIND: "next_subtask", QUERY_TEXT: "tidy", "task": "old task"}
    )

    output = RenderRuntimeMessagesStep(recipe)(transition)

    assert asdict(recipe) == original_recipe
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"] == "old task"
    assert output[TransitionKey.COMPLEMENTARY_DATA]["task"] == "old task"


def test_runtime_requests_validate_kind_text_and_recipe():
    with pytest.raises(ValueError, match="requires a checkpoint recipe"):
        _render("next_subtask", "tidy")
    with pytest.raises(ValueError, match="Unsupported query kind"):
        _render("caption", "describe this")
    with pytest.raises(TypeError, match="query_text"):
        RenderRuntimeMessagesStep()(
            create_transition(complementary_data={QUERY_KIND: "vqa", QUERY_TEXT: 123})
        )


def test_runtime_query_cannot_mix_with_raw_training_language():
    transition = create_transition(
        complementary_data={QUERY_KIND: "vqa", QUERY_TEXT: "what is visible?", "language_events": []}
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        RenderRuntimeMessagesStep()(transition)


def test_runtime_renderer_preserves_observations_and_actions():
    from lerobot.processor import ComplementaryDataProcessorStep

    observation = {"observation.state": object()}
    action = object()
    transition = create_transition(
        observation=observation,
        action=action,
        complementary_data={QUERY_KIND: "vqa", QUERY_TEXT: "what?"},
    )
    step = RenderRuntimeMessagesStep()
    assert isinstance(step, ComplementaryDataProcessorStep)
    output = step(transition)
    assert output[TransitionKey.OBSERVATION] is observation
    assert output[TransitionKey.ACTION] is action
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {QUERY_KIND: "vqa", QUERY_TEXT: "what?"}


def test_rendered_messages_survive_conversion_and_batching():
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.batch_processor import AddBatchDimensionComplementaryDataStep
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    messages = [{"role": "user", "content": "already rendered"}]
    batch = {"messages_rendered": messages, "message_streams": ["low_level"], "target_message_indices": []}
    round_trip = transition_to_batch(batch_to_transition(batch))
    assert {key: round_trip[key] for key in batch} == batch
    pipeline = PolicyProcessorPipeline(
        steps=[RenderRuntimeMessagesStep(), AddBatchDimensionComplementaryDataStep()]
    )
    result = pipeline(batch)
    assert result["messages_rendered"] == [messages]
    assert result["message_streams"] == [["low_level"]]
    assert result["target_message_indices"] == [[]]
    assert "messages" not in result


def test_same_pipeline_handles_training_then_text_and_action_inference(dataset_dependencies):
    import torch

    from lerobot.processor import PolicyProcessorPipeline, RenderTrainingMessagesStep

    pipeline = PolicyProcessorPipeline(
        steps=[RenderRuntimeMessagesStep(_recipe()), RenderTrainingMessagesStep(_recipe())]
    )
    training = pipeline({"task": "tidy", "action": torch.zeros(1, 1)})
    assert training["messages_rendered"] == [{"role": "user", "content": "tidy"}]
    assert training["message_streams"] == ["low_level"]
    inference = pipeline({"query_kind": "next_subtask", "query_text": "tidy"})
    assert inference["messages_rendered"] == [
        {"role": "system", "content": "Robot assistant"},
        {"role": "user", "content": "Goal: tidy"},
    ]
    assert "target_message_indices" not in inference
    assert "message_streams" not in inference
    action_input = pipeline({"task": "tidy"})
    assert "messages_rendered" not in action_input
    assert action_input["task"] == "tidy"
