#!/usr/bin/env python

from pathlib import Path

import pytest

from lerobot.configs.recipe import MessageTurn, TrainingRecipe


def test_message_recipe_validates_unknown_binding():
    with pytest.raises(ValueError, match="unknown binding"):
        TrainingRecipe(
            messages=[
                MessageTurn(role="user", content="${missing}", stream="high_level"),
                MessageTurn(role="assistant", content="ok", stream="high_level", target=True),
            ]
        )


def test_canonical_recipe_loads():
    recipe = TrainingRecipe.from_yaml(
        Path("src/lerobot/configs/recipes/subtask_mem_vqa_speech.yaml")
    )

    assert recipe.blend is not None
    assert set(recipe.blend) == {
        "memory_update",
        "user_interjection_response",
        "high_level_subtask",
        "low_level_execution",
        "ask_vqa_top",
        "ask_vqa_wrist",
    }
    assert sum(component.weight for component in recipe.blend.values()) == pytest.approx(1.0)
