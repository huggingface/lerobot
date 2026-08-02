from __future__ import annotations

import json
from pathlib import Path

import audit_datasets
import verify_training_contract


ROOT = Path(__file__).parent


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def test_frozen_dataset_identities() -> None:
    assert audit_datasets.DATASETS["real48"]["tree_sha256"] == (
        "c4534befc536c10217638da91f5cbbaff59b0795ec91f0633e53e8a6d99507b9"
    )
    assert audit_datasets.DATASETS["real96"]["tree_sha256"] == (
        "58a5f8fa907c6b4433750c816f0eb80743ee861b06a1dd1356811fbc6800b1a1"
    )
    assert audit_datasets.DATASETS["real48"]["frames"] == 8955
    assert audit_datasets.DATASETS["real96"]["frames"] == 17439


def test_full_configs_have_only_allowed_identity_differences() -> None:
    real48 = load("real48_train_config_full.json")
    real96 = load("real96_train_config_full.json")
    assert verify_training_contract.training_semantics(real48) == (
        verify_training_contract.training_semantics(real96)
    )
    fields = {row["field"] for row in verify_training_contract.differences(real48, real96)}
    assert fields == {
        "dataset.repo_id",
        "dataset.root",
        "dataset.episodes",
        "output_dir",
        "job_name",
    }


def test_smokes_are_independent_from_scratch_and_matched() -> None:
    real48 = load("real48_train_config_smoke.json")
    real96 = load("real96_train_config_smoke.json")
    assert real48["steps"] == real96["steps"] == 500
    assert real48["save_freq"] == real96["save_freq"] == 500
    assert real48["policy"]["pretrained_path"] is None
    assert real96["policy"]["pretrained_path"] is None
    assert real48["resume"] is real96["resume"] is False


def test_formal_recipe_and_checkpoint_rule() -> None:
    manifest = load("experiment_manifest.json")
    recipe = manifest["recipe"]
    assert recipe["seed"] == 1000
    assert recipe["steps"] == 100000
    assert recipe["batch_size"] == 8
    assert recipe["save_frequency_steps"] == 20000
    assert recipe["selected_checkpoint_step"] == 100000
    assert recipe["chunk_size"] == recipe["n_action_steps"] == 67
    assert recipe["use_imagenet_stats"] is True
    assert recipe["sampling"] == "standard LeRobot pure-Real frame sampling"


def test_hardware_and_rollout_boundaries_are_false() -> None:
    boundaries = load("experiment_manifest.json")["boundaries"]
    assert all(value is False for value in boundaries.values())
