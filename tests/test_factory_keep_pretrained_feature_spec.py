#!/usr/bin/env python

from pathlib import Path
from unittest.mock import patch

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class _DummyDatasetMeta:
    def __init__(self):
        self.features = {
            OBS_STATE: {
                "dtype": "float32",
                "shape": (8,),
                "names": [f"s{i}" for i in range(8)],
            },
            ACTION: {
                "dtype": "float32",
                "shape": (11,),
                "names": [f"a{i}" for i in range(11)],
            },
            f"{OBS_IMAGES}.camera0": {
                "dtype": "video",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
                "info": None,
            },
        }
        self.stats = {}


class _DummyPolicy(torch.nn.Module):
    name = "pi0"
    config_class = PI0Config

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self._probe = torch.nn.Linear(1, 1)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, config, **kwargs):
        return cls(config=config)


def _make_pretrained_cfg() -> PI0Config:
    cfg = PI0Config(device="cpu")
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(32,)),
        f"{OBS_IMAGES}.right_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.left_wrist_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        f"{OBS_IMAGES}.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
    }
    return cfg


def test_keep_pretrained_feature_spec_uses_pretrained_shapes_when_requested():
    cfg = PI0Config(pretrained_path=Path("dummy-pretrained"), keep_pretrained_feature_spec=True, device="cpu")
    cfg.input_features = {}
    cfg.output_features = {}

    ds_meta = _DummyDatasetMeta()
    pretrained_cfg = _make_pretrained_cfg()

    with (
        patch("lerobot.policies.factory.get_policy_class", return_value=_DummyPolicy),
        patch.object(PI0Config, "from_pretrained", return_value=pretrained_cfg),
    ):
        policy = make_policy(
            cfg=cfg,
            ds_meta=ds_meta,
            rename_map={"observation.images.camera0": "observation.images.right_wrist_0_rgb"},
        )

    assert isinstance(policy, _DummyPolicy)
    assert cfg.output_features[ACTION].shape == (32,)
    assert cfg.input_features[OBS_STATE].shape == (32,)


def test_keep_pretrained_feature_spec_false_preserves_dataset_override_behavior():
    cfg = PI0Config(pretrained_path=Path("dummy-pretrained"), keep_pretrained_feature_spec=False, device="cpu")
    cfg.input_features = {}
    cfg.output_features = {}

    ds_meta = _DummyDatasetMeta()

    with patch("lerobot.policies.factory.get_policy_class", return_value=_DummyPolicy):
        policy = make_policy(cfg=cfg, ds_meta=ds_meta)

    assert isinstance(policy, _DummyPolicy)
    assert cfg.output_features[ACTION].shape == (11,)
    assert cfg.input_features[OBS_STATE].shape == (8,)
