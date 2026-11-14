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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies import factory

pytestmark = pytest.mark.dataset


@pytest.mark.parametrize("parent", [None, "owner/base", Path("owner/base")])
@pytest.mark.parametrize("stage", ["features", "load", "deferred"])
def test_failed_resume_restores_parent_for_retry(
    monkeypatch: pytest.MonkeyPatch, parent: str | Path | None, stage: str
) -> None:
    cfg = SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path=parent,
        pretrained_revision=None,
        use_peft=False,
        input_features={},
        output_features={},
    )
    error = RuntimeError("injected failure")
    failing = True
    sources = []

    def features(_: object) -> dict:
        if failing and stage == "features":
            raise error
        return {}

    def build(**kwargs: object) -> torch.nn.Module:
        sources.append(cfg.pretrained_path)
        if failing:
            raise error
        policy = torch.nn.Linear(1, 1)
        policy.config = cfg
        return policy

    class Policy:
        from_pretrained = staticmethod(build)

        def __new__(cls, **kwargs: object) -> torch.nn.Module:
            return build(**kwargs)

    monkeypatch.setattr(factory, "get_policy_class", lambda _: Policy)
    monkeypatch.setattr(factory, "dataset_to_policy_features", features)
    monkeypatch.setattr(factory, "validate_visual_features_consistency", lambda *args: None)
    meta = SimpleNamespace(features={}, stats={})
    with pytest.raises(RuntimeError) as caught:
        factory.make_policy(
            cfg, ds_meta=meta, pretrained_path="run/checkpoint", defer_weight_load=stage == "deferred"
        )
    assert caught.value is error
    assert cfg.pretrained_path is parent
    failing = False
    result = factory.make_policy(cfg, ds_meta=meta)
    assert sources[-1] is parent
    assert result.config.pretrained_path is parent
