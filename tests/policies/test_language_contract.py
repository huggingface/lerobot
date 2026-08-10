#!/usr/bin/env python

"""Client- and policy-facing tests for the flat text-generation contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.policies.pretrained import PreTrainedPolicy


@PreTrainedConfig.register_subclass("language_contract_test")
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


class ChatTemplatePolicy(ContractPolicy):
    """A fake policy whose native formatter uses explicit chat role tokens."""

    def _format_text_generation_input(self, semantic_inputs: dict[str, Any]) -> dict[str, Any]:
        messages = semantic_inputs["messages"]
        native_prompt = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages
        )
        return {**semantic_inputs, "native_prompt": native_prompt}

    def _generate_preprocessed_text(self, model_inputs: dict[str, Any]) -> str:
        self.last_model_inputs = model_inputs
        return model_inputs["native_prompt"]


class InstructionTemplatePolicy(ContractPolicy):
    """A second fake policy with a different valid model-native prompt."""

    def _format_text_generation_input(self, semantic_inputs: dict[str, Any]) -> str:
        return "[INST] " + "\n".join(message["content"] for message in semantic_inputs["messages"])

    def _generate_preprocessed_text(self, model_inputs: str) -> str:
        return model_inputs


def _subtask_recipe() -> TrainingRecipe:
    return TrainingRecipe(
        messages=[
            MessageTurn(role="system", content="You control a robot.", stream="high_level"),
            MessageTurn(
                role="user",
                content="Goal: ${task}\nPredict the next action in language.",
                stream="high_level",
            ),
            MessageTurn(
                role="assistant",
                content="${subtask}",
                stream="high_level",
                target=True,
                if_present="subtask",
            ),
        ]
    )


def test_generate_text_defaults_to_raw_and_preserves_caller_payload():
    policy = ChatTemplatePolicy(ContractConfig(recipe=_subtask_recipe()))
    question = "Which {cup} is closest to the gripper?"

    answer = policy.generate_text({}, question)

    assert answer == f"<user>{question}</user>"
    assert policy.last_model_inputs["messages"] == [{"role": "user", "content": question}]


def test_generate_text_subtask_applies_checkpoint_recipe_internally():
    policy = ChatTemplatePolicy(ContractConfig(recipe=_subtask_recipe()))

    generated = policy.generate_text({}, "Clear the table", template="subtask")

    assert generated == (
        "<system>You control a robot.</system>"
        "<user>Goal: Clear the table\nPredict the next action in language.</user>"
    )


def test_two_policies_format_identical_raw_text_without_client_changes():
    text = "What is visible?"

    chat_prompt = ChatTemplatePolicy(ContractConfig()).generate_text({}, text)
    instruction_prompt = InstructionTemplatePolicy(ContractConfig()).generate_text({}, text)

    assert chat_prompt == "<user>What is visible?</user>"
    assert instruction_prompt == "[INST] What is visible?"


def test_subtask_requires_a_checkpoint_recipe():
    policy = ChatTemplatePolicy(ContractConfig())

    with pytest.raises(ValueError, match="Subtask generation requires a checkpoint recipe"):
        policy.generate_text({}, "Clear the table", template="subtask")


def test_unsupported_public_template_is_rejected():
    policy = ChatTemplatePolicy(ContractConfig())

    with pytest.raises(ValueError, match="Unsupported text template: 'caption'"):
        policy.generate_text({}, "Describe this", template="caption")  # type: ignore[arg-type]


def test_generate_text_does_not_mutate_batch_or_repeat_observation_processing():
    policy = ChatTemplatePolicy(ContractConfig())
    observation = torch.tensor([[1.0, 2.0]])
    batch = {"observation.state": observation, "subtask": "keep this state"}

    policy.generate_text(batch, "What is happening?")

    assert batch == {"observation.state": observation, "subtask": "keep this state"}
    assert policy.last_model_inputs["observation.state"] is observation
    assert policy.last_model_inputs["subtask"] == "keep this state"


def test_config_save_load_restores_a_normalized_recipe(tmp_path: Path):
    config = ContractConfig(recipe=_subtask_recipe())
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, ContractConfig)
    assert isinstance(loaded.recipe, TrainingRecipe)
    policy = ChatTemplatePolicy(loaded)
    assert "Goal: Fold the towel" in policy.generate_text({}, "Fold the towel", template="subtask")


def test_supports_text_generation_detects_the_protected_decoder_hook():
    assert not ContractPolicy(ContractConfig()).supports_text_generation()
    assert ChatTemplatePolicy(ContractConfig()).supports_text_generation()


def test_policy_without_text_head_raises_clear_error():
    policy = ContractPolicy(ContractConfig())

    with pytest.raises(NotImplementedError, match="has no text head"):
        policy.generate_text({}, "What do you see?")
