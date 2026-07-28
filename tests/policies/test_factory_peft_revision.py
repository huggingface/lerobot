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
from unittest.mock import MagicMock

import torch

import lerobot.policies.factory as policy_factory


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
