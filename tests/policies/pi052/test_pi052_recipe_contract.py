"""Parity checks for PI052 on the shared recipe/runtime contract."""

from dataclasses import asdict

import torch

from lerobot.datasets.recipe import TrainingRecipe
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi052.configuration_pi052 import PI052Config
from lerobot.policies.pi052.text_processor_pi052 import PI052TextTokenizerStep
from lerobot.processor.render_messages_processor import RenderRuntimeMessagesStep, RenderTrainingMessagesStep
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_STATE, QUERY_KIND, QUERY_TEXT
from tests.policies.pi052.test_pi052_text_processor import _CharTokenizer


def test_subtask_recipe_is_embedded_and_keeps_70_30_mix():
    config = PI052Config(recipe_path="recipes/subtask.yaml", device="cpu", enable_fast_action_loss=False)
    assert config.recipe["blend"]["high_level_subtask"]["weight"] == 0.3
    assert config.recipe["blend"]["low_level_execution"]["weight"] == 0.7
    restored = PI052Config(recipe_path="/missing/recipe.yaml", recipe=config.recipe, device="cpu")
    assert restored.recipe == config.recipe


def test_runtime_prefill_matches_training_before_target():
    recipe = TrainingRecipe.from_yaml("src/lerobot/configs/recipes/subtask.yaml")
    tokenizer = PI052TextTokenizerStep(max_length=256)
    tokenizer._tokenizer = _CharTokenizer()
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 14)},
        TransitionKey.COMPLEMENTARY_DATA: {QUERY_KIND: "next_subtask", QUERY_TEXT: "clear table"},
    }
    rendered = RenderRuntimeMessagesStep(recipe)(transition)
    assert rendered[TransitionKey.COMPLEMENTARY_DATA]["messages_rendered"] == [
        {"role": "user", "content": "clear table"}
    ]
    processed = tokenizer(RenderTrainingMessagesStep(recipe)(rendered))
    ids = processed[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS][0]
    expected = "User: clear table\nAssistant:"
    expected_ids = tokenizer._tokenizer(
        expected,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding_side="right",
    )["input_ids"][0]
    assert torch.equal(ids, expected_ids)
    assert not processed[TransitionKey.COMPLEMENTARY_DATA]["predict_actions"].any()
    assert (processed[TransitionKey.COMPLEMENTARY_DATA]["text_labels"] == -100).all()


def test_action_only_runtime_uses_same_state_bearing_prompt_as_training():
    tokenizer = PI052TextTokenizerStep(max_length=256)
    tokenizer._tokenizer = _CharTokenizer()
    base = {TransitionKey.OBSERVATION: {OBS_STATE: torch.zeros(1, 14)}}
    runtime = tokenizer({**base, TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick plate"]}})
    training = tokenizer(
        {
            **base,
            TransitionKey.COMPLEMENTARY_DATA: {
                "messages_rendered": [[{"role": "user", "content": "pick plate"}]],
                "message_streams": [["low_level"]],
                "target_message_indices": [[]],
            },
        }
    )
    assert torch.equal(
        runtime[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS],
        training[TransitionKey.OBSERVATION][OBS_LANGUAGE_TOKENS],
    )


def test_goal_only_recipe_ignores_subtask_annotations():
    recipe = TrainingRecipe.from_dict(
        {"messages": [{"role": "user", "content": "${task}", "stream": "low_level"}]}
    )
    config = PI052Config(
        recipe_path=None,
        recipe=asdict(recipe),
        device="cpu",
        enable_fast_action_loss=False,
        text_loss_weight=0,
    )
    renderer = RenderTrainingMessagesStep(config.recipe)
    result = renderer(
        {
            TransitionKey.ACTION: torch.zeros(1, 14),
            TransitionKey.COMPLEMENTARY_DATA: {
                "task": "clear table",
                "timestamp": 0,
                "index": 0,
                "language_persistent": [
                    {"role": "assistant", "style": "subtask", "content": "secret subtask", "timestamp": 0}
                ],
            },
        }
    )
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["messages_rendered"] == [{"role": "user", "content": "clear table"}]
    assert complementary["target_message_indices"] == []
