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
"""Checkpoint defaults must not override explicit feature selections (#4517)."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import draccus
import pytest
import torch
import yaml
from safetensors.torch import save_file

import lerobot.policies.factory as policy_factory
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy

STATE = "observation.state"
CAM1 = "observation.images.camera1"
CAM2 = "observation.images.camera2"
CAM3 = "observation.images.camera3"
DATASET_FEATURES = {
    STATE: {"dtype": "float32", "shape": (8,), "names": None},
    CAM1: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
    CAM2: {"dtype": "image", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}
CURRENT_INPUTS = {
    STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    CAM1: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    CAM2: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
}
CHECKPOINT_INPUTS = {
    STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for key in [CAM1, CAM2, CAM3]},
}


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    """A real, tiny local checkpoint: config parsing and weight loading are not mocked."""
    path = tmp_path / "checkpoint"
    cfg = DummyCheckpointConfig(device="cpu", input_features=deepcopy(CHECKPOINT_INPUTS))
    cfg.save_pretrained(path)
    policy = DummyCheckpointPolicy(cfg)
    with torch.no_grad():
        policy.net.weight.fill_(0.25)
    save_file(policy.state_dict(), path / "model.safetensors")
    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: DummyCheckpointPolicy)
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    return path


def load_config(checkpoint, **kwargs):
    cfg = PreTrainedConfig.from_pretrained(checkpoint, **kwargs)
    cfg.pretrained_path = checkpoint
    return cfg


def make_policy(cfg, source="dataset", features=None, **kwargs):
    features = deepcopy(DATASET_FEATURES if features is None else features)
    if source == "dataset":
        source_kwargs = {"ds_meta": SimpleNamespace(features=features, stats={})}
    else:
        # Exercise the real environment conversion, including HWC -> CHW.
        env_features = {
            key: PolicyFeature(
                type=FeatureType.ACTION
                if key == "action"
                else (FeatureType.VISUAL if ft["dtype"] in ["image", "video"] else FeatureType.STATE),
                shape=ft["shape"],
            )
            for key, ft in features.items()
        }
        source_kwargs = {
            "env_cfg": SimpleNamespace(features=env_features, features_map={k: k for k in features})
        }
    policy = policy_factory.make_policy(cfg, **source_kwargs, **kwargs)
    # Proves the pretrained branch loaded the on-disk weights, not random initialization.
    assert torch.all(policy.net.weight == 0.25)
    return policy


@pytest.mark.parametrize("source", ["dataset", "env"])
def test_checkpoint_defaults_refresh_from_current_features(checkpoint, source, caplog):
    cfg = load_config(checkpoint)
    make_policy(cfg, source)
    assert cfg.input_features == CURRENT_INPUTS
    assert cfg.output_features == {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    assert "Refreshing input_features inherited" in caplog.text
    assert "Previous:" in caplog.text and "resolved:" in caplog.text


@pytest.mark.parametrize("keys", [[STATE], [STATE, CAM1]])
@pytest.mark.parametrize("syntax", ["equals", "separate"])
def test_cli_selections_replace_checkpoint_mapping(checkpoint, keys, syntax):
    selected = {key: CURRENT_INPUTS[key] for key in keys}
    value = json.dumps(draccus.encode(selected))
    args = [f"--input_features={value}"] if syntax == "equals" else ["--input_features", value]
    cfg = load_config(checkpoint, cli_overrides=args)
    assert cfg.input_features == selected  # no checkpoint cameras merged back by draccus
    make_policy(cfg)
    assert cfg.input_features == selected


@pytest.mark.parametrize("mode", ["constructor", "keyword", "assignment", "in_place", "equal_assignment"])
def test_python_explicit_features_are_preserved(checkpoint, mode):
    selected = {STATE: CURRENT_INPUTS[STATE]}
    if mode == "constructor":
        cfg = DummyCheckpointConfig(
            device="cpu", pretrained_path=checkpoint, input_features=deepcopy(selected)
        )
    elif mode == "keyword":
        cfg = load_config(checkpoint, input_features=deepcopy(selected))
    else:
        cfg = load_config(checkpoint)
        if mode == "in_place":
            cfg.input_features.clear()
            cfg.input_features.update(deepcopy(selected))
        elif mode == "equal_assignment":
            selected = deepcopy(CHECKPOINT_INPUTS)
            cfg.input_features = deepcopy(selected)
        else:
            cfg.input_features = deepcopy(selected)
    make_policy(cfg)
    assert cfg.input_features == selected


def test_python_in_place_feature_shape_is_preserved(checkpoint):
    cfg = load_config(checkpoint)
    cfg.input_features[STATE].shape = (9,)
    selected = deepcopy(cfg.input_features)
    make_policy(cfg)
    assert cfg.input_features == selected


@pytest.mark.parametrize("empty", [None, {}])
@pytest.mark.parametrize("source", ["dataset", "env"])
def test_explicit_empty_features_still_request_inference(checkpoint, empty, source):
    cfg = load_config(checkpoint, cli_overrides=[f"--input_features={json.dumps(empty)}"])
    make_policy(cfg, source)
    assert cfg.input_features == CURRENT_INPUTS


@pytest.mark.parametrize("extension", ["yaml", "json"])
@pytest.mark.parametrize("path_source", ["config", "cli"])
@pytest.mark.parametrize("selection", ["state", "camera", "null", "empty", "cli_override", "inherited"])
def test_training_config_feature_precedence(
    checkpoint, tmp_path, monkeypatch, extension, path_source, selection
):
    selected = {STATE: CURRENT_INPUTS[STATE]}
    if selection in ["camera", "cli_override"]:
        selected[CAM1] = CURRENT_INPUTS[CAM1]
    if selection in ["null", "inherited"]:
        selected = None
    elif selection == "empty":
        selected = {}
    recipe = {
        "dataset": {"repo_id": "test/dataset"},
        "policy": {
            "path": str(checkpoint),
            "input_features": json.loads(json.dumps(draccus.encode(selected))),
        },
    }
    if selection == "inherited":
        del recipe["policy"]["input_features"]
    if path_source == "cli":
        del recipe["policy"]["path"]
        recipe["policy"]["type"] = "dummy_checkpoint"
    config_path = tmp_path / f"train.{extension}"
    config_path.write_text(yaml.safe_dump(recipe) if extension == "yaml" else json.dumps(recipe))
    args = ["lerobot-train", f"--config_path={config_path}"]
    if path_source == "cli":
        args.append(f"--policy.path={checkpoint}")
    if selection == "cli_override":
        selected = {STATE: CURRENT_INPUTS[STATE]}
        args.append(f"--policy.input_features={json.dumps(draccus.encode(selected))}")
    monkeypatch.setattr(sys, "argv", args)

    @parser.wrap()
    def parse_train(cfg: TrainPipelineConfig):
        cfg._resolve_pretrained_from_cli()
        return cfg

    cfg = parse_train().policy
    make_policy(cfg)
    assert cfg.input_features == (selected or CURRENT_INPUTS)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "rename", [None, {"unused": CAM1}, {"observation.images.top": "observation.images.wrong"}]
)
def test_missing_or_wrong_rename_map_is_rejected(checkpoint, explicit, rename):
    kwargs = (
        {"input_features": {STATE: CURRENT_INPUTS[STATE], CAM1: CURRENT_INPUTS[CAM1]}} if explicit else {}
    )
    cfg = load_config(checkpoint, **kwargs)
    original = deepcopy(cfg.input_features)
    features = {STATE: DATASET_FEATURES[STATE], "observation.images.top": DATASET_FEATURES[CAM1]}
    with pytest.raises(ValueError, match="rename_map"):
        make_policy(cfg, features=features, rename_map=rename)
    assert cfg.input_features == original


def test_correct_rename_map_validates_then_refreshes(checkpoint):
    cfg = load_config(checkpoint)
    features = {STATE: DATASET_FEATURES[STATE], "observation.images.top": DATASET_FEATURES[CAM1]}
    make_policy(cfg, features=features, rename_map={"observation.images.top": CAM1})
    assert cfg.input_features == {STATE: CURRENT_INPUTS[STATE], CAM1: CURRENT_INPUTS[CAM1]}
    assert "observation.images.top" in features


def test_matching_checkpoint_features_do_not_warn(checkpoint, caplog):
    DummyCheckpointConfig(device="cpu", input_features=deepcopy(CURRENT_INPUTS)).save_pretrained(checkpoint)
    cfg = load_config(checkpoint)
    make_policy(cfg)
    assert cfg.input_features == CURRENT_INPUTS
    assert "Refreshing input_features" not in caplog.text


def test_provenance_survives_copy_but_is_not_serialized(checkpoint, tmp_path):
    cfg = deepcopy(load_config(checkpoint))
    make_policy(cfg)
    saved = tmp_path / "resolved"
    cfg.save_pretrained(saved)
    payload = json.loads((saved / "config.json").read_text())
    assert not any(key.startswith("_") for key in payload)
    assert payload["input_features"] == draccus.encode(CURRENT_INPUTS)
    assert PreTrainedConfig.from_pretrained(saved).input_features == CURRENT_INPUTS


def test_policy_added_empty_camera_does_not_invalidate_selection(checkpoint, monkeypatch):
    class EmptyCameraPolicy(DummyCheckpointPolicy):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            config.input_features["observation.images.empty_camera_0"] = CURRENT_INPUTS[CAM1]

    monkeypatch.setattr(policy_factory, "get_policy_class", lambda _: EmptyCameraPolicy)
    selected = {STATE: CURRENT_INPUTS[STATE], CAM1: CURRENT_INPUTS[CAM1]}
    cfg = load_config(checkpoint, input_features=selected)
    make_policy(cfg)
    assert CAM2 not in cfg.input_features
    assert "observation.images.empty_camera_0" in cfg.input_features


@pytest.mark.parametrize("explicit", [False, True])
def test_training_cli_checkpoint_path(checkpoint, monkeypatch, explicit):
    args = ["lerobot-train", "--dataset.repo_id=test/dataset", f"--policy.path={checkpoint}"]
    selected = {STATE: CURRENT_INPUTS[STATE]}
    if explicit:
        args.extend(["--policy.input_features", json.dumps(draccus.encode(selected))])
    monkeypatch.setattr(sys, "argv", args)

    @parser.wrap()
    def parse_train(cfg: TrainPipelineConfig):
        cfg._resolve_pretrained_from_cli()
        return cfg

    cfg = parse_train().policy
    make_policy(cfg)
    assert cfg.input_features == (selected if explicit else CURRENT_INPUTS)
