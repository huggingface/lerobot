from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_DATASET = {
    "repo_id": "local/task1_picklift_real24_budget_extension_v1_accepted",
    "root": "/home/ubuntu24/Teleop/artifacts/derived/task1_picklift_real24_budget_extension_v1/accepted",
    "episodes": list(range(24)),
}


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_smoke_and_full_match_except_run_length_and_output() -> None:
    smoke = load("train_config_smoke.json")
    full = load("train_config_full.json")
    assert smoke["dataset"]["repo_id"] == EXPECTED_DATASET["repo_id"]
    assert smoke["dataset"]["root"] == EXPECTED_DATASET["root"]
    assert smoke["dataset"]["episodes"] == EXPECTED_DATASET["episodes"]
    assert full["dataset"] == smoke["dataset"]
    ignored = {"steps", "save_freq", "output_dir", "job_name", "log_freq"}
    assert {k: v for k, v in smoke.items() if k not in ignored} == {
        k: v for k, v in full.items() if k not in ignored
    }
    assert smoke["steps"] == smoke["save_freq"] == 500
    assert full["steps"] == 100000 and full["save_freq"] == 20000
    assert smoke["resume"] is False and full["resume"] is False


def test_frozen_act_recipe() -> None:
    cfg = load("train_config_full.json")
    policy = cfg["policy"]
    assert cfg["seed"] == 1000 and cfg["batch_size"] == 8 and cfg["num_workers"] == 4
    assert cfg["dataset"]["use_imagenet_stats"] is True
    assert cfg["dataset"]["image_transforms"]["enable"] is False
    assert policy["pretrained_path"] is None and policy["device"] == "cuda" and policy["use_amp"] is False
    assert policy["chunk_size"] == policy["n_action_steps"] == 67
    assert policy["vision_backbone"] == "resnet18"
    assert policy["pretrained_backbone_weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    assert (policy["dim_model"], policy["n_heads"], policy["dim_feedforward"]) == (512, 8, 3200)
    assert (policy["n_encoder_layers"], policy["n_decoder_layers"]) == (4, 1)
    assert policy["latent_dim"] == 32 and policy["kl_weight"] == 10.0
    assert policy["optimizer_lr"] == 1e-5 and policy["optimizer_weight_decay"] == 1e-4
    assert cfg["sample_weighting"] is None


def test_only_front_state_to_action() -> None:
    policy = load("train_config_full.json")["policy"]
    assert set(policy["input_features"]) == {"observation.state", "observation.images.front"}
    assert set(policy["output_features"]) == {"action"}
    assert policy["input_features"]["observation.state"]["shape"] == [6]
    assert policy["input_features"]["observation.images.front"]["shape"] == [3, 480, 640]
    assert policy["output_features"]["action"]["shape"] == [6]
