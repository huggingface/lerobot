#!/usr/bin/env python

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

import pytest

from lerobot.policies import factory
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi052 import fit_fast_tokenizer as fit_module


def test_pi0_fast_resolves_dataset_specific_tokenizer(monkeypatch, tmp_path):
    config = PI0FastConfig(
        auto_fit_fast_tokenizer=True,
        action_tokenizer_name="base-tokenizer",
        fast_tokenizer_cache_dir=str(tmp_path),
        fast_tokenizer_fit_samples=17,
        chunk_size=12,
        n_action_steps=12,
    )
    received = {}

    def fake_fit(**kwargs):
        received.update(kwargs)
        return "/cache/fitted-tokenizer"

    monkeypatch.setattr(fit_module, "fit_fast_tokenizer", fake_fit)

    assert fit_module.resolve_fast_tokenizer(config, "user/dataset") == "/cache/fitted-tokenizer"
    assert received == {
        "dataset_repo_id": "user/dataset",
        "cache_dir": tmp_path,
        "base_tokenizer_name": "base-tokenizer",
        "n_samples": 17,
        "chunk_size": 12,
        "dataset_root": None,
        "dataset_revision": None,
        "episodes": None,
        "exclude_episodes": None,
        "normalization_mode": config.normalization_mapping["ACTION"],
        "action_stats": None,
        "use_relative_actions": False,
        "relative_action_mask": None,
    }


def test_fast_fit_failure_is_not_silently_replaced(monkeypatch, tmp_path):
    config = PI0FastConfig(auto_fit_fast_tokenizer=True, fast_tokenizer_cache_dir=str(tmp_path))
    monkeypatch.setattr(
        fit_module,
        "fit_fast_tokenizer",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )

    with pytest.raises(RuntimeError, match="fit failed"):
        fit_module.resolve_fast_tokenizer(config, "user/dataset")


def test_only_global_rank_zero_fits_shared_tokenizer(monkeypatch):
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert not fit_module._is_global_leader()

    monkeypatch.setenv("RANK", "0")
    assert fit_module._is_global_leader()


def test_pretrained_pi0_fast_overrides_only_fitted_tokenizer(monkeypatch):
    config = PI0FastConfig(auto_fit_fast_tokenizer=True)
    calls = []

    monkeypatch.setattr(
        fit_module,
        "resolve_fast_tokenizer",
        lambda config, dataset_repo_id, *args: "/cache/fitted-tokenizer",
    )

    def fake_from_pretrained(cls, *args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(steps=[])

    monkeypatch.setattr(factory.PolicyProcessorPipeline, "from_pretrained", classmethod(fake_from_pretrained))

    factory.make_pre_post_processors(
        config,
        pretrained_path="checkpoint",
        dataset_repo_id="user/dataset",
    )

    assert calls[0]["overrides"] == {
        "action_tokenizer_processor": {"action_tokenizer_name": "/cache/fitted-tokenizer"}
    }
