# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

import lerobot.policies.factory as policy_factory


@pytest.mark.parametrize("rollout_stats", [{}, {"action": {"mean": torch.tensor([42.0])}}])
def test_pretrained_processors_ignore_rollout_dataset_stats(monkeypatch, rollout_stats):
    """Inference stats must never replace the checkpoint's training normalization."""
    cfg = SimpleNamespace(type="mock", use_language_recipe=True, recipe_path="recipe.yaml")
    saved_preprocessor = SimpleNamespace(steps=[])
    saved_postprocessor = SimpleNamespace(steps=[])
    load_pretrained = MagicMock(side_effect=[saved_preprocessor, saved_postprocessor])
    rebuild = MagicMock()
    monkeypatch.setattr(policy_factory.PolicyProcessorPipeline, "from_pretrained", load_pretrained)
    monkeypatch.setattr(policy_factory, "_make_processors_from_policy_config", rebuild)

    preprocessor, postprocessor = policy_factory.make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path="checkpoint",
        dataset_stats=rollout_stats,
    )

    assert (preprocessor, postprocessor) == (saved_preprocessor, saved_postprocessor)
    assert load_pretrained.call_args_list == [
        call(
            pretrained_model_name_or_path="checkpoint",
            config_filename="policy_preprocessor.json",
            overrides={},
            to_transition=policy_factory.batch_to_transition,
            to_output=policy_factory.transition_to_batch,
            revision=None,
        ),
        call(
            pretrained_model_name_or_path="checkpoint",
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=policy_factory.policy_action_to_transition,
            to_output=policy_factory.transition_to_policy_action,
            revision=None,
        ),
    ]
    rebuild.assert_not_called()


def test_training_can_explicitly_rebuild_pretrained_processors(monkeypatch):
    cfg = SimpleNamespace(type="mock")
    training_stats = {"action": {"mean": torch.tensor([1.0])}}
    rebuilt = (object(), object())
    rebuild = MagicMock(return_value=rebuilt)
    monkeypatch.setattr(policy_factory, "_make_processors_from_policy_config", rebuild)

    processors = policy_factory.make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path="checkpoint",
        dataset_stats=training_stats,
        rebuild_pretrained_processors=True,
    )

    assert processors == rebuilt
    rebuild.assert_called_once_with(config=cfg, dataset_stats=training_stats, dataset_meta=None)


def test_rebuilding_pretrained_processors_rejects_empty_stats():
    with pytest.raises(ValueError, match="non-empty training dataset statistics"):
        policy_factory.make_pre_post_processors(
            policy_cfg=SimpleNamespace(type="mock"),
            pretrained_path="checkpoint",
            dataset_stats={},
            rebuild_pretrained_processors=True,
        )


def test_make_policy_keeps_peft_adapter_and_base_revisions_separate(monkeypatch):
    cfg = SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path="user/adapter",
        pretrained_revision="adapter-sha",
        use_peft=True,
        input_features={},
        output_features={},
    )
    dataset_meta = SimpleNamespace(features={}, stats={})

    base_policy = torch.nn.Linear(1, 1)
    policy_from_pretrained = MagicMock(return_value=base_policy)
    policy_class = SimpleNamespace(from_pretrained=policy_from_pretrained)
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: policy_class)
    monkeypatch.setattr(policy_factory, "dataset_to_policy_features", lambda _: {})
    monkeypatch.setattr(policy_factory, "validate_visual_features_consistency", lambda *args: None)

    peft_config = SimpleNamespace(
        base_model_name_or_path="user/base-policy",
        revision="base-sha",
    )
    peft_config_from_pretrained = MagicMock(return_value=peft_config)
    adapted_policy = torch.nn.Linear(1, 1)
    peft_model_from_pretrained = MagicMock(return_value=adapted_policy)
    require_package = MagicMock()
    monkeypatch.setattr(policy_factory, "require_package", require_package)
    monkeypatch.setattr(
        policy_factory,
        "PeftConfig",
        SimpleNamespace(from_pretrained=peft_config_from_pretrained),
    )
    monkeypatch.setattr(
        policy_factory,
        "PeftModel",
        SimpleNamespace(from_pretrained=peft_model_from_pretrained),
    )

    policy = policy_factory.make_policy(cfg, ds_meta=dataset_meta)

    assert policy is adapted_policy
    require_package.assert_called_once_with("peft", extra="peft")
    peft_config_from_pretrained.assert_called_once_with(
        "user/adapter",
        revision="adapter-sha",
    )
    policy_from_pretrained.assert_called_once_with(
        config=cfg,
        dataset_stats=dataset_meta.stats,
        dataset_meta=dataset_meta,
        pretrained_name_or_path="user/base-policy",
        revision="base-sha",
    )
    peft_model_from_pretrained.assert_called_once_with(
        base_policy,
        "user/adapter",
        config=peft_config,
        revision="adapter-sha",
        is_trainable=True,
    )
