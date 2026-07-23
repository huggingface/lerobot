#!/usr/bin/env python

from pathlib import Path
from textwrap import dedent

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe, load_recipe


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
    catching it here means a malformed recipe YAML errors at load instead
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


# ── from_dict / from_yaml round-trips ────────────────────────────────


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


def test_from_yaml_round_trips_through_load_recipe(tmp_path: Path):
    yaml_text = dedent(
        """
        bindings:
          custom: "active_at(t, style=subtask)"
        messages:
          - {role: user, content: "${task}: ${custom}", stream: high_level}
          - {role: assistant, content: "ok", stream: high_level, target: true}
        """
    ).strip()
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml_text)

    via_classmethod = TrainingRecipe.from_yaml(path)
    via_helper = load_recipe(path)

    assert via_classmethod.bindings == {"custom": "active_at(t, style=subtask)"}
    assert via_classmethod.messages[1].target is True
    # ``load_recipe`` is just a wrapper, but assert the two paths agree
    # on the structural result so a future divergence is caught here.
    assert via_helper.bindings == via_classmethod.bindings
    assert len(via_helper.messages) == len(via_classmethod.messages)


def test_from_yaml_rejects_non_mapping(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping at the top level"):
        TrainingRecipe.from_yaml(path)
