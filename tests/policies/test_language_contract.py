#!/usr/bin/env python

"""The recipe-driven language contract on `PreTrainedPolicy`.

The recipe carried by ``config.recipe`` is the single source of prompt wording:
`prompt_messages` / `build_prompt` replay its prompt turns, so inference prompts
cannot drift from the phrasing the checkpoint was trained on.
"""

from dataclasses import dataclass

import pytest

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.policies.pretrained import DEFAULT_LANGUAGE_RECIPE, PreTrainedPolicy


@dataclass
class ContractConfig(PreTrainedConfig):
    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError

    def get_scheduler_preset(self):
        raise NotImplementedError

    def validate_features(self) -> None:
        pass


class ContractPolicy(PreTrainedPolicy):
    config_class = ContractConfig
    name = "contract_test"

    def get_optim_params(self) -> dict:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def forward(self, batch):
        raise NotImplementedError

    def predict_action_chunk(self, batch, *, with_text: bool = False, **kwargs):
        raise NotImplementedError

    def select_action(self, batch, **kwargs):
        raise NotImplementedError


class TextHeadPolicy(ContractPolicy):
    def generate_text(self, batch, prompt: str) -> str:
        return f"echo: {prompt}"


def _memory_recipe() -> TrainingRecipe:
    return TrainingRecipe(
        bindings={
            "prior_memory": "nth_prev(style=memory, offset=1)",
            "completed_subtask": "nth_prev(style=subtask, offset=1)",
            "current_memory": "active_at(t, style=memory)",
        },
        messages=[
            MessageTurn(role="user", content="${task}", stream="high_level"),
            MessageTurn(
                role="assistant",
                content="Previous memory: ${prior_memory}",
                stream="high_level",
                if_present="prior_memory",
            ),
            MessageTurn(
                role="user",
                content="Completed subtask: ${completed_subtask}",
                stream="high_level",
                if_present="completed_subtask",
            ),
            MessageTurn(
                role="assistant",
                content="${current_memory}",
                stream="high_level",
                target=True,
                if_present="current_memory",
            ),
        ],
    )


def test_build_prompt_falls_back_to_the_default_recipe():
    policy = ContractPolicy(ContractConfig())
    assert policy.language_recipe() is DEFAULT_LANGUAGE_RECIPE
    prompt = policy.build_prompt("subtask", task="clear the table")
    assert prompt == "clear the table\nPredict the next action in language."


def test_checkpoint_recipe_overrides_the_wording():
    config = ContractConfig(
        recipe=TrainingRecipe(
            messages=[
                MessageTurn(role="user", content="Goal: ${task}\nNext step?", stream="high_level"),
                MessageTurn(
                    role="assistant",
                    content="${subtask}",
                    stream="high_level",
                    target=True,
                    if_present="subtask",
                ),
            ]
        )
    )
    policy = ContractPolicy(config)
    assert policy.build_prompt("subtask", task="fold the towel") == "Goal: fold the towel\nNext step?"


def test_language_recipe_normalizes_the_reloaded_dict():
    # A reloaded config.json carries the recipe as a plain dict.
    config = ContractConfig(
        recipe={
            "messages": [
                {"role": "user", "content": "${task}", "stream": "high_level"},
                {
                    "role": "assistant",
                    "content": "${subtask}",
                    "stream": "high_level",
                    "target": True,
                    "if_present": "subtask",
                },
            ]
        }
    )
    policy = ContractPolicy(config)
    recipe = policy.language_recipe()
    assert isinstance(recipe, TrainingRecipe)
    # Normalization is cached back onto the config so it runs once.
    assert config.recipe is recipe
    assert policy.build_prompt("subtask", task="stack the cups") == "stack the cups"


def test_prompt_messages_skips_if_present_turns_without_values():
    policy = ContractPolicy(ContractConfig(recipe=_memory_recipe()))
    messages = policy.prompt_messages("current_memory", task="tidy the desk")
    assert messages == [{"role": "user", "content": "tidy the desk"}]


def test_prompt_messages_keeps_if_present_turns_with_values_and_roles():
    policy = ContractPolicy(ContractConfig(recipe=_memory_recipe()))
    messages = policy.prompt_messages("current_memory", task="tidy the desk", prior_memory="drawer is sorted")
    assert messages == [
        {"role": "user", "content": "tidy the desk"},
        {"role": "assistant", "content": "Previous memory: drawer is sorted"},
    ]


def test_prompt_messages_raises_on_a_missing_placeholder_value():
    policy = ContractPolicy(ContractConfig())
    with pytest.raises(ValueError, match=r"references \['task'\]"):
        policy.build_prompt("subtask")


def test_build_prompt_unknown_kind_lists_supervised_kinds():
    policy = ContractPolicy(ContractConfig())
    with pytest.raises(ValueError, match="no target turn supervising 'memory'"):
        policy.build_prompt("memory", task="tidy")


def test_supports_text_generation_detects_an_overridden_generate_text():
    assert not ContractPolicy(ContractConfig()).supports_text_generation()
    assert TextHeadPolicy(ContractConfig()).supports_text_generation()


def test_generate_text_default_raises_with_guidance():
    policy = ContractPolicy(ContractConfig())
    with pytest.raises(NotImplementedError, match="has no text head"):
        policy.generate_text({}, "what do you see?")
