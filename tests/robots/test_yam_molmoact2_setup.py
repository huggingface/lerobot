"""Validate the YAM checkpoint conversion without downloading model weights."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("transformers")
pytest.importorskip("scipy")


def converter():
    path = Path(__file__).parents[2] / "examples/yam/prepare_molmoact2.py"
    spec = importlib.util.spec_from_file_location("yam_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(module):
    return {
        "control_mode": "absolute joint pose",
        "camera_keys": module.IMAGE_KEYS,
        "normalize_gripper": False,
        **{
            key: {
                "names": list(module.YAM_FEATURE_NAMES),
                "mask": [True] * 6 + [False] + [True] * 6 + [False],
                "q01": [-1.0] * 14,
                "q99": [1.0] * 14,
            }
            for key in ("action_stats", "state_stats")
        },
    }


@pytest.mark.parametrize("change", ["ee", "camera_order", "joint_order", "gripper_mask"])
def test_rejects_incompatible_release_metadata(change):
    module = converter()
    data = copy.deepcopy(metadata(module))
    if change == "ee":
        data["control_mode"] = "absolute end-effector pose"
    elif change == "camera_order":
        data["camera_keys"].reverse()
    elif change == "joint_order":
        data["state_stats"]["names"].reverse()
    else:
        data["action_stats"]["mask"][6] = True
    with pytest.raises(ValueError):
        module.validate_metadata(data)


def test_conversion_preserves_training_mode_and_tagged_inference_processors(tmp_path, monkeypatch):
    module = converter()
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"metadata_by_tag": {module.NORM_TAG: metadata(module)}})
    )
    (tmp_path / "config.json").write_text(json.dumps({"action_mode": "both"}))
    monkeypatch.setattr(module, "hf_hub_download", lambda repo, filename, **kwargs: str(tmp_path / filename))
    captured = {}
    policy = Mock()
    policy.to.return_value.eval.return_value = policy
    policy.save_pretrained.side_effect = lambda path: path.mkdir()

    def create_policy(config):
        captured["policy"] = config
        return policy

    def create_processors(config):
        captured["processors"] = config
        return Mock(), Mock()

    monkeypatch.setattr(module, "MolmoAct2Policy", create_policy)
    monkeypatch.setattr(module, "make_molmoact2_pre_post_processors", create_processors)
    monkeypatch.setattr(module.torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--output", str(tmp_path / "out"), "--revision", "pinned-sha"]
    )
    module.main()
    config = captured["policy"]
    assert config.action_mode == "both"
    assert config.inference_action_mode == "continuous"
    assert config.train_mode_vlm == "fft"
    assert config.checkpoint_revision == "pinned-sha"
    assert captured["processors"].action_mode == "continuous"
    assert captured["processors"].norm_tag == "yam_dual_molmoact2"
    assert not captured["processors"].normalize_gripper
    assert config.dataset_feature_names["action"] == list(module.YAM_FEATURE_NAMES)
