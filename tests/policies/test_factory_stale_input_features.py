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

"""Regression tests for stale `input_features` on configs loaded from a base checkpoint.

https://github.com/huggingface/lerobot/issues/4517 — `make_policy` only refreshed
`input_features` when empty, so a config loaded from a checkpoint silently kept
features from the original training environment when fine-tuned elsewhere.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import lerobot.policies.factory as policy_factory
from lerobot.configs.types import FeatureType, PolicyFeature

DATASET_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.images.cam": {"dtype": "video", "shape": (256, 256, 3), "names": ["h", "w", "c"]},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}

EXPECTED_INPUT_FEATURES = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
}
EXPECTED_ACTION_FEATURE = PolicyFeature(type=FeatureType.ACTION, shape=(7,))


def _make_ds_meta():
    return SimpleNamespace(features={k: dict(v) for k, v in DATASET_FEATURES.items()}, stats={})


def _make_cfg(input_features):
    return SimpleNamespace(
        type="mock",
        device="cpu",
        pretrained_path=None,
        use_peft=False,
        input_features=input_features,
        output_features={},
    )


def _run_make_policy(cfg, monkeypatch):
    monkeypatch.setattr(
        policy_factory, "get_policy_class", lambda _: MagicMock(return_value=torch.nn.Linear(1, 1))
    )
    policy_factory.make_policy(cfg, ds_meta=_make_ds_meta())


def test_stale_input_features_are_refreshed(monkeypatch):
    # Simulates a config loaded from a checkpoint trained on a different robot:
    # stale 6-dim state instead of the dataset's 8-dim state, and a missing camera.
    stale = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    cfg = _make_cfg(dict(stale))

    _run_make_policy(cfg, monkeypatch)

    assert cfg.input_features == EXPECTED_INPUT_FEATURES
    assert cfg.output_features == {"action": EXPECTED_ACTION_FEATURE}


def test_matching_input_features_are_kept(monkeypatch):
    cfg = _make_cfg(
        {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        }
    )

    _run_make_policy(cfg, monkeypatch)

    assert cfg.input_features == EXPECTED_INPUT_FEATURES


def test_empty_input_features_are_filled(monkeypatch):
    cfg = _make_cfg({})

    _run_make_policy(cfg, monkeypatch)

    assert cfg.input_features == EXPECTED_INPUT_FEATURES


def test_stale_input_features_refresh_logs_warning(monkeypatch, caplog):
    stale = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))}
    cfg = _make_cfg(dict(stale))

    with caplog.at_level("WARNING", logger="lerobot.policies.factory"):
        _run_make_policy(cfg, monkeypatch)

    assert any("stale input_features" in r.message for r in caplog.records)


def test_env_features_differing_shape_dict_equality_is_shape_based(monkeypatch):
    # Equal dicts with same keys but different shapes must NOT be considered equal.
    a = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    b = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    assert a != b


@pytest.mark.parametrize(
    "stale_features",
    [
        {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3)),
        },
    ],
)
def test_parametrized_stale_features_are_refreshed(monkeypatch, stale_features):
    cfg = _make_cfg(dict(stale_features))

    _run_make_policy(cfg, monkeypatch)

    assert cfg.input_features == EXPECTED_INPUT_FEATURES
