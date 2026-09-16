#!/usr/bin/env python

import pytest

pytest.importorskip("datasets", reason="recipes require lerobot[dataset]")
pytest.importorskip("av", reason="recipes require lerobot[dataset]")

from lerobot.datasets.recipe import (  # noqa: E402
    MessageTurn,
    TrainingRecipe,
    render_message_turns,
)


def _minimal_message_turn(content: str = "${task}") -> MessageTurn:
    return MessageTurn(role="user", content=content, stream="high_level")


def _minimal_target_turn() -> MessageTurn:
    return MessageTurn(role="assistant", content="ok", stream="high_level", target=True)


# ── Message-recipe validation ────────────────────────────────────────


def test_message_recipe_validates_unknown_binding():
    with pytest.raises(ValueError, match="unknown binding"):
        TrainingRecipe(
            messages=[
                MessageTurn(role="user", content="${missing}", stream="high_level"),
                _minimal_target_turn(),
            ]
        )


def test_message_turn_requires_a_stream():
    """Every turn must declare a stream — None is rejected at construction.

    Previously this only failed at render time (``_validate_rendered``);
    catching it here means a malformed policy recipe errors at construction instead
    of at the first training sample.
    """
    with pytest.raises(ValueError, match="missing a stream"):
        MessageTurn(role="user", content="${task}")


def test_message_recipe_requires_at_least_one_target():
    with pytest.raises(ValueError, match="target"):
        TrainingRecipe(
            messages=[
                _minimal_message_turn(),
                MessageTurn(role="assistant", content="no target", stream="high_level"),
            ]
        )


def test_recipe_rejects_both_messages_and_blend():
    with pytest.raises(ValueError, match="only one"):
        TrainingRecipe(
            messages=[_minimal_message_turn(), _minimal_target_turn()],
            blend={"a": TrainingRecipe(weight=1.0, messages=[_minimal_target_turn()])},
        )


def test_recipe_rejects_neither_messages_nor_blend():
    with pytest.raises(ValueError, match="must set one"):
        TrainingRecipe()


# ── Blend validation ─────────────────────────────────────────────────


def test_blend_must_be_non_empty():
    with pytest.raises(ValueError, match="at least one component"):
        TrainingRecipe(blend={})


def test_blend_component_must_define_weight():
    with pytest.raises(ValueError, match="weight"):
        TrainingRecipe(blend={"a": TrainingRecipe(messages=[_minimal_target_turn()])})


def test_blend_component_weight_must_be_positive():
    with pytest.raises(ValueError, match="positive weight"):
        TrainingRecipe(blend={"a": TrainingRecipe(weight=0.0, messages=[_minimal_target_turn()])})


def test_recipe_route_must_be_supported():
    with pytest.raises(ValueError, match="Unsupported recipe route"):
        TrainingRecipe(weight=1.0, route="other", messages=[_minimal_target_turn()])


def test_route_cannot_be_set_on_blend_recipe():
    with pytest.raises(ValueError, match="only be set on a message recipe"):
        TrainingRecipe(
            route="vqa",
            blend={"a": TrainingRecipe(weight=1.0, messages=[_minimal_target_turn()])},
        )


def test_blend_component_must_define_messages():
    # A bare TrainingRecipe(weight=1.0) would itself raise; build it without
    # going through __post_init__ to exercise the blend-level validator.
    bad = TrainingRecipe.__new__(TrainingRecipe)
    bad.messages = None
    bad.bindings = None
    bad.blend = None
    bad.weight = 1.0
    with pytest.raises(ValueError, match="must define messages"):
        TrainingRecipe(blend={"a": bad})


def test_blend_components_cannot_themselves_define_a_blend():
    inner = TrainingRecipe(blend={"x": TrainingRecipe(weight=1.0, messages=[_minimal_target_turn()])})
    # Force-bypass the inner component's normal validation so the test
    # exercises the outer blend's "no nested blends" rule directly.
    nested = TrainingRecipe.__new__(TrainingRecipe)
    nested.messages = None
    nested.bindings = None
    nested.blend = inner.blend
    nested.weight = 1.0
    with pytest.raises(ValueError, match="cannot itself define a blend"):
        TrainingRecipe(blend={"outer": nested})


# ── from_dict ───────────────────────────────────────────────────────


def test_from_dict_with_nested_blend():
    recipe = TrainingRecipe.from_dict(
        {
            "blend": {
                "a": {
                    "weight": 1.0,
                    "messages": [
                        {"role": "user", "content": "${task}", "stream": "high_level"},
                        {"role": "assistant", "content": "a", "stream": "high_level", "target": True},
                    ],
                },
                "b": {
                    "weight": 2.0,
                    "messages": [
                        {"role": "user", "content": "${task}", "stream": "high_level"},
                        {"role": "assistant", "content": "b", "stream": "high_level", "target": True},
                    ],
                },
            }
        }
    )
    assert recipe.blend is not None
    assert set(recipe.blend) == {"a", "b"}
    assert recipe.blend["b"].weight == 2.0
    # Inner messages were promoted to MessageTurn instances.
    assert isinstance(recipe.blend["a"].messages[0], MessageTurn)


def test_prompt_turns_returns_prefix_before_matching_assistant_target():
    recipe = TrainingRecipe(
        messages=[
            _minimal_message_turn("Goal: ${task}"),
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )

    assert [turn.content for turn in recipe.prompt_turns("subtask")] == ["Goal: ${task}"]
    with pytest.raises(ValueError, match=r"no assistant target turn supervising \$\{memory\}"):
        recipe.prompt_turns("memory")


def test_prompt_turns_uses_first_matching_blend_component():
    def component(label: str) -> TrainingRecipe:
        return TrainingRecipe(
            weight=1.0,
            messages=[
                _minimal_message_turn(f"{label}: ${{task}}"),
                MessageTurn(
                    role="assistant",
                    content="${subtask}",
                    stream="high_level",
                    target=True,
                ),
            ],
        )

    recipe = TrainingRecipe(blend={"first": component("first"), "second": component("second")})

    assert recipe.prompt_turns("subtask")[0].content == "first: ${task}"


def test_render_message_turns_substitutes_without_dataset_dependencies():
    turns = [
        MessageTurn(role="user", content="Goal: ${task}", stream="high_level"),
        MessageTurn(
            role="assistant",
            content="${subtask}",
            stream="high_level",
            target=True,
        ),
    ]

    rendered = render_message_turns(turns, {"task": "tidy", "subtask": "pick up cup"})

    assert rendered == {
        "messages_rendered": [
            {"role": "user", "content": "Goal: tidy"},
            {"role": "assistant", "content": "pick up cup"},
        ],
        "message_streams": ["high_level", "high_level"],
        "target_message_indices": [1],
    }
