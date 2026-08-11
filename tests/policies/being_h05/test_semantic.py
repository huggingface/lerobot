#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.being_h05.configuration_being_h05 import ROBOCASA_CAMERA_KEYS, BeingH05Config
from lerobot.policies.being_h05.processor_being_h05 import (
    ACTION_SLOTS,
    STATE_SLOTS,
    BeingH05BinaryActionStep,
    BeingH05MessagesStep,
    BeingH05SemanticPackStep,
    make_being_h05_pre_post_processors,
    pack_named,
)
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.processor.text_generation_processor import RenderGenerationPromptStep
from lerobot.utils.constants import ACTION


def _named_state(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        f"observation.state.{name}": torch.zeros(batch, end - start)
        for name, (start, end) in STATE_SLOTS.items()
    }


def test_semantic_slots_and_missing_modality_masks():
    named = {"eef_position": torch.tensor([[1.0, 2.0, 3.0]]), "control_mode": torch.ones(1, 1)}
    state, state_valid = pack_named({"eef_position": named["eef_position"]}, STATE_SLOTS)
    action, action_valid = pack_named({"control_mode": named["control_mode"]}, ACTION_SLOTS)
    assert state.shape == action.shape == (1, 200)
    assert state_valid[0, :3].all() and state_valid.sum() == 3
    assert action[0, 74] == 1 and action_valid.sum() == 1
    assert not state_valid[0, 3:].any()


def test_raw_task_reaches_audit_hook_unchanged_and_all_cameras():
    raw = "Pick the red mug — then stop?!"
    action = torch.zeros(1, 16, 12)
    observation = _named_state()
    for key in ROBOCASA_CAMERA_KEYS:
        observation[key] = torch.rand(1, 3, 256, 256)
    transition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.COMPLEMENTARY_DATA: {"task": [raw]},
    }
    step = BeingH05SemanticPackStep(
        image_keys=ROBOCASA_CAMERA_KEYS,
        prompt_template="TASK={task_description}; K={k}",
        chunk_size=16,
    )
    result = step(transition)
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary["being_h05_raw_task"] == [raw]
    assert complementary["being_h05_prompt"] == [f"TASK={raw}; K=16"]
    assert result[TransitionKey.OBSERVATION]["being_h05.pixel_values"].shape == (1, 3, 3, 224, 224)
    assert result[TransitionKey.OBSERVATION]["being_h05.image_valid"].all()


def test_missing_middle_camera_is_masked_without_changing_camera_roles():
    transition = {
        TransitionKey.OBSERVATION: {
            **_named_state(),
            ROBOCASA_CAMERA_KEYS[0]: torch.rand(1, 3, 256, 256),
            ROBOCASA_CAMERA_KEYS[2]: torch.rand(1, 3, 256, 256),
        },
        TransitionKey.ACTION: None,
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["close the drawer"]},
    }
    step = BeingH05SemanticPackStep(
        image_keys=ROBOCASA_CAMERA_KEYS,
        prompt_template="{task_description} {k}",
        chunk_size=16,
    )
    observation = step(transition)[TransitionKey.OBSERVATION]
    assert observation["being_h05.image_valid"].tolist() == [[True, False, True]]


def test_recipe_messages_are_serialized_and_route_joint_action_training():
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "messages": [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "feature": ROBOCASA_CAMERA_KEYS[0]},
                            {"type": "text", "text": "What should happen next?"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "say",
                                    "arguments": {"text": "I will close the drawer."},
                                }
                            }
                        ],
                    },
                ]
            ],
            "message_streams": [["low_level", "low_level"]],
            "target_message_indices": [[1]],
        }
    }

    complementary = BeingH05MessagesStep()(transition)[TransitionKey.COMPLEMENTARY_DATA]

    assert complementary["being_h05_messages"] == [
        [
            {"role": "user", "content": "What should happen next?"},
            {"role": "assistant", "content": "<say>I will close the drawer.</say>"},
        ]
    ]
    assert complementary["being_h05_target_message_indices"] == [[1]]
    assert complementary["being_h05_predict_actions"].tolist() == [True]


def test_recipe_text_targets_must_be_assistant_messages():
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {
            "messages": [[{"role": "user", "content": "question"}]],
            "message_streams": [["high_level"]],
            "target_message_indices": [[0]],
        }
    }

    with pytest.raises(ValueError, match="assistant messages"):
        BeingH05MessagesStep()(transition)


def test_config_and_factories_are_wired_without_importing_author_dependencies():
    config = make_policy_config(
        "being_h05",
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in ROBOCASA_CAMERA_KEYS
            },
            **{
                f"observation.state.{name}": PolicyFeature(type=FeatureType.STATE, shape=(end - start,))
                for name, (start, end) in STATE_SLOTS.items()
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
    )
    assert isinstance(config, BeingH05Config)
    assert config.text_loss_weight == 0.1
    assert config.action_loss_weight == 1.0
    assert get_policy_class("being_h05").name == "being_h05"
    preprocessor, postprocessor = make_pre_post_processors(config)
    assert isinstance(preprocessor.steps[0], RenderGenerationPromptStep)
    assert isinstance(preprocessor.steps[2], NormalizerProcessorStep)
    assert isinstance(preprocessor.steps[3], BeingH05SemanticPackStep)
    assert postprocessor.steps[0].get_config() == {}
    assert isinstance(postprocessor.steps[1], UnnormalizerProcessorStep)
    policy_class = get_policy_class("being_h05")
    assert policy_class.generate_text is not PreTrainedPolicy.generate_text
    assert policy_class.supports_text_generation is not PreTrainedPolicy.supports_text_generation
    assert not hasattr(policy_class, "generate_texts")


def test_recipe_pipeline_renders_language_columns_before_being_serialization():
    pytest.importorskip("datasets")
    from lerobot.processor.render_messages_processor import RenderMessagesStep

    config = BeingH05Config(
        use_language_recipe=True,
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in ROBOCASA_CAMERA_KEYS
            },
            **{
                f"observation.state.{name}": PolicyFeature(type=FeatureType.STATE, shape=(end - start,))
                for name, (start, end) in STATE_SLOTS.items()
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
    )

    preprocessor, _ = make_being_h05_pre_post_processors(config)

    assert isinstance(preprocessor.steps[0], RenderGenerationPromptStep)
    assert isinstance(preprocessor.steps[3], RenderMessagesStep)
    assert isinstance(preprocessor.steps[4], BeingH05SemanticPackStep)
    assert isinstance(preprocessor.steps[5], BeingH05MessagesStep)

    batch = {
        **_named_state(),
        **{key: torch.rand(1, 3, 224, 224) for key in ROBOCASA_CAMERA_KEYS},
        ACTION: torch.zeros(1, config.chunk_size, 12),
        "task": ["close the drawer"],
        "timestamp": torch.tensor([0.0]),
        "index": torch.tensor([0]),
        "language_persistent": [
            [
                {
                    "role": "assistant",
                    "content": "reach for the handle",
                    "style": "subtask",
                    "timestamp": 0.0,
                    "camera": None,
                    "tool_calls": None,
                }
            ]
        ],
        "language_events": [[]],
    }

    processed = preprocessor(batch)

    assert processed["being_h05_messages"] == [
        [
            {"role": "user", "content": "close the drawer"},
            {"role": "assistant", "content": "reach for the handle"},
        ]
    ]
    assert processed["being_h05_target_message_indices"] == [[1]]
    assert processed["being_h05_predict_actions"].tolist() == [True]
    assert processed[ACTION].shape == (1, config.chunk_size, 200)


def test_quantile_normalization_round_trips_continuous_and_binary_actions(tmp_path):
    state_features = {
        f"observation.state.{name}": PolicyFeature(type=FeatureType.STATE, shape=(end - start,))
        for name, (start, end) in STATE_SLOTS.items()
    }
    config = BeingH05Config(
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in ROBOCASA_CAMERA_KEYS
            },
            **state_features,
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    action_q01 = torch.zeros(12)
    action_q99 = torch.full((12,), 10.0)
    action_q99[[6, 11]] = 1.0
    dataset_stats = {
        **{
            key: {
                "q01": torch.zeros(feature.shape),
                "q99": torch.full(feature.shape, 10.0),
            }
            for key, feature in state_features.items()
        },
        ACTION: {"q01": action_q01, "q99": action_q99},
    }
    checkpoint_config = BeingH05Config(
        input_features=config.input_features,
        output_features=config.output_features,
    )
    preprocessor, postprocessor = make_being_h05_pre_post_processors(checkpoint_config)
    preprocessor.save_pretrained(tmp_path)
    postprocessor.save_pretrained(tmp_path)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(tmp_path),
        dataset_stats=dataset_stats,
    )
    trained_checkpoint = tmp_path / "trained"
    preprocessor.save_pretrained(trained_checkpoint)
    postprocessor.save_pretrained(trained_checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(trained_checkpoint),
    )
    assert isinstance(preprocessor.steps[2], BeingH05BinaryActionStep)
    assert isinstance(preprocessor.steps[3], NormalizerProcessorStep)
    assert isinstance(preprocessor.steps[4], BeingH05BinaryActionStep)
    assert isinstance(postprocessor.steps[1], BeingH05BinaryActionStep)
    assert isinstance(postprocessor.steps[2], UnnormalizerProcessorStep)
    assert isinstance(postprocessor.steps[3], BeingH05BinaryActionStep)

    raw_action = torch.tensor(
        [[[2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0.0, 7.5, 7.5, 7.5, 7.5, 1.0] for _ in range(config.chunk_size)]]
    )
    batch = {
        **{key: torch.full((1, *feature.shape), 2.5) for key, feature in state_features.items()},
        **{key: torch.rand(1, 3, 224, 224) for key in ROBOCASA_CAMERA_KEYS},
        ACTION: raw_action,
        "task": ["close the drawer"],
    }

    processed = preprocessor(batch)
    semantic_action = processed[ACTION]
    torch.testing.assert_close(semantic_action[..., 0:6], torch.full_like(semantic_action[..., 0:6], -0.5))
    torch.testing.assert_close(semantic_action[..., 70:74], torch.full_like(semantic_action[..., 70:74], 0.5))
    assert not semantic_action[..., 18].any()
    assert semantic_action[..., 74].all()
    torch.testing.assert_close(
        processed["being_h05.state"][..., 0:3],
        torch.full_like(processed["being_h05.state"][..., 0:3], -0.5),
    )

    round_tripped_action = postprocessor(semantic_action)
    torch.testing.assert_close(round_tripped_action, raw_action)


def test_processor_pipeline_save_reload(tmp_path):
    config = BeingH05Config(
        input_features={
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                for key in ROBOCASA_CAMERA_KEYS
            },
            **{
                f"observation.state.{name}": PolicyFeature(type=FeatureType.STATE, shape=(end - start,))
                for name, (start, end) in STATE_SLOTS.items()
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(12,))},
    )
    pre, post = make_being_h05_pre_post_processors(config)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    loaded_pre, loaded_post = make_pre_post_processors(config, pretrained_path=str(tmp_path))
    assert any(isinstance(step, BeingH05SemanticPackStep) for step in loaded_pre.steps)
    assert any(isinstance(step, NormalizerProcessorStep) for step in loaded_pre.steps)
    assert any(isinstance(step, UnnormalizerProcessorStep) for step in loaded_post.steps)
    assert loaded_post.name == "policy_postprocessor"
    semantic_action = torch.zeros(1, 200)
    semantic_action[:, 0:3] = torch.tensor([0.1, 0.2, 0.3])
    semantic_action[:, 18] = 1
    environment_action = loaded_post(semantic_action)
    assert environment_action.shape == (1, 12)
    torch.testing.assert_close(environment_action[:, 0:3], semantic_action[:, 0:3])
    assert environment_action[:, 6].item() == 1
    assert environment_action[:, 11].item() == 0
