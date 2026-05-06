#!/usr/bin/env python

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
