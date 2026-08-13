#!/usr/bin/env python

"""Runtime-, processor-, and policy-facing tests for text generation."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.recipe import MessageTurn, TrainingRecipe
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.policies.language import (
    join_semantic_message_text,
    last_semantic_message_text,
    normalize_semantic_messages,
    require_single_semantic_conversation,
    require_single_text_output,
    semantic_message_content_text,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import RenderMessagesStep
from lerobot.processor.batch_processor import AddBatchDimensionProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline, ProcessorStep
from lerobot.utils.constants import QUERY_KIND, QUERY_TEXT


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

    def predict_action_chunk(self, batch, **kwargs):
        raise NotImplementedError

    def select_action(self, batch, **kwargs):
        raise NotImplementedError


class TextHeadPolicy(ContractPolicy):
    def supports_text_generation(self) -> bool:
        return True

    def generate_text(self, batch) -> str:
        self.last_model_inputs = batch
        return batch["native_prompt"]


@dataclass
class ChatTemplateStep(ProcessorStep):
    """Fake policy-native formatter using explicit chat role tokens."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        data = dict(output[TransitionKey.COMPLEMENTARY_DATA])
        data["native_prompt"] = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in data["messages"]
        )
        output[TransitionKey.COMPLEMENTARY_DATA] = data
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
class InstructionTemplateStep(ProcessorStep):
    """Fake policy-native formatter using instruction delimiters."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        output = transition.copy()
        data = dict(output[TransitionKey.COMPLEMENTARY_DATA])
        data["native_prompt"] = "[INST] " + "\n".join(message["content"] for message in data["messages"])
        output[TransitionKey.COMPLEMENTARY_DATA] = data
        return output

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


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


def _generation_preprocessor(config: ContractConfig, formatter: ProcessorStep):
    return PolicyProcessorPipeline([RenderMessagesStep(config.recipe, render_training=False), formatter])


def test_runtime_query_metadata_is_formatted_by_the_input_pipeline():
    config = ContractConfig(recipe=_subtask_recipe())
    policy = TextHeadPolicy(config)
    preprocessor = _generation_preprocessor(config, ChatTemplateStep())
    request = {QUERY_KIND: "vqa", QUERY_TEXT: "Which {cup} is closest?"}

    batch = preprocessor(request)
    answer = policy.generate_text(batch)

    assert answer == "<user>Which {cup} is closest?</user>"
    assert QUERY_TEXT not in batch
    assert QUERY_KIND not in batch


def test_runtime_subtask_metadata_applies_checkpoint_recipe_before_decoding():
    config = ContractConfig(recipe=_subtask_recipe())
    policy = TextHeadPolicy(config)
    preprocessor = _generation_preprocessor(config, ChatTemplateStep())

    batch = preprocessor({QUERY_KIND: "next_subtask", QUERY_TEXT: "Clear the table"})
    generated = policy.generate_text(batch)

    assert generated == (
        "<system>You control a robot.</system>"
        "<user>Goal: Clear the table\nPredict the next action in language.</user>"
    )


def test_two_policy_processor_formatters_need_no_client_api_changes():
    config = ContractConfig()
    request = {QUERY_KIND: "vqa", QUERY_TEXT: "What is visible?"}

    chat_batch = _generation_preprocessor(config, ChatTemplateStep())(request)
    instruction_batch = _generation_preprocessor(config, InstructionTemplateStep())(request)
    policy = TextHeadPolicy(config)

    assert policy.generate_text(chat_batch) == "<user>What is visible?</user>"
    assert policy.generate_text(instruction_batch) == "[INST] What is visible?"


def test_policy_pipeline_serializes_explicit_renderer_before_batching():
    config = ContractConfig(recipe=_subtask_recipe())
    preprocessor = PolicyProcessorPipeline(
        [RenderMessagesStep(config.recipe, render_training=False), AddBatchDimensionProcessorStep()]
    )

    assert isinstance(preprocessor.steps[0], RenderMessagesStep)
    assert isinstance(preprocessor.steps[1], AddBatchDimensionProcessorStep)
    assert preprocessor.get_config()["steps"][0]["registry_name"] == "render_messages_processor"
    assert preprocessor({"task": "ordinary action input"})["task"] == ["ordinary action input"]
    assert preprocessor({QUERY_KIND: "vqa", QUERY_TEXT: "Question?"})["messages"] == [
        [{"role": "user", "content": "Question?"}]
    ]


def test_preprocessing_does_not_mutate_runtime_request_or_repeat_observation_transforms():
    config = ContractConfig()
    policy = TextHeadPolicy(config)
    preprocessor = _generation_preprocessor(config, ChatTemplateStep())
    observation = torch.tensor([[1.0, 2.0]])
    request = {
        "observation.state": observation,
        "task": "keep this state",
        QUERY_KIND: "vqa",
        QUERY_TEXT: "What is happening?",
    }

    batch = preprocessor(request)
    policy.generate_text(batch)

    assert request[QUERY_KIND] == "vqa"
    assert request[QUERY_TEXT] == "What is happening?"
    assert request["task"] == "keep this state"
    assert policy.last_model_inputs["observation.state"] is observation
    assert policy.last_model_inputs["task"] == "keep this state"


def test_config_save_load_restores_recipe_used_by_input_pipeline(tmp_path: Path):
    config = ContractConfig(recipe=_subtask_recipe())
    config._save_pretrained(tmp_path)

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, ContractConfig)
    assert isinstance(loaded.recipe, TrainingRecipe)
    batch = _generation_preprocessor(loaded, InstructionTemplateStep())(
        {QUERY_KIND: "next_subtask", QUERY_TEXT: "Fold the towel"}
    )
    assert "Goal: Fold the towel" in TextHeadPolicy(loaded).generate_text(batch)


def test_text_capable_policy_overrides_support_and_public_generation_method():
    assert not ContractPolicy(ContractConfig()).supports_text_generation()
    assert TextHeadPolicy(ContractConfig()).supports_text_generation()
    assert TextHeadPolicy.generate_text is not PreTrainedPolicy.generate_text


def test_policy_without_text_head_raises_clear_error():
    policy = ContractPolicy(ContractConfig())

    with pytest.raises(NotImplementedError, match="has no text head"):
        policy.generate_text({})


def test_semantic_message_utilities_normalize_extract_and_validate():
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "feature": "observation.images.front"},
                {"type": "text", "text": "What is visible?"},
            ],
        },
        {"role": "assistant", "content": "A blue cube."},
    ]

    assert normalize_semantic_messages(conversation, policy_name="Test") == [conversation]
    assert require_single_semantic_conversation([conversation], policy_name="Test") == conversation
    assert semantic_message_content_text(conversation[0]["content"]) == "What is visible?"
    assert join_semantic_message_text(conversation) == "What is visible?\nA blue cube."
    assert last_semantic_message_text(conversation, role="user") == "What is visible?"
    assert require_single_text_output([" answer "], policy_name="Test") == "answer"

    with pytest.raises(ValueError, match="expected one Test prompt"):
        require_single_semantic_conversation([conversation, conversation], policy_name="Test")
