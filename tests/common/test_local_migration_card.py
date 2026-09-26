# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import json
import sys
from dataclasses import dataclass, field

import pytest
import torch
from huggingface_hub import ModelCard
from safetensors.torch import load_file, save_file

from lerobot.common.train_utils import generate_model_card
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.processor import make_default_pre_post_processors, migrate_policy_normalization as migration
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, DummyCheckpointPolicy


@PreTrainedConfig.register_subclass("dummy_migration_card")
@dataclass
class MigrationCardConfig(DummyCheckpointConfig):
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=dict)


@pytest.fixture
def legacy_checkpoint(tmp_path):
    path = tmp_path / "legacy"
    config = MigrationCardConfig(
        device="cpu",
        repo_id="fixture/policy",
        license="apache-2.0",
        input_features={"observation.state": PolicyFeature(FeatureType.STATE, (4,))},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        normalization_mapping={"STATE": NormalizationMode.MIN_MAX, "ACTION": NormalizationMode.MIN_MAX},
    )
    policy = DummyCheckpointPolicy(config)
    policy.save_pretrained(path)
    weights = load_file(str(path / "model.safetensors"))
    for name in ("normalize_inputs.buffer_observation_state", "unnormalize_outputs.buffer_action"):
        weights[name + ".min"] = torch.zeros(4)
        weights[name + ".max"] = torch.ones(4)
    save_file(weights, str(path / "model.safetensors"))
    (path / "train_config.json").write_text(json.dumps({"repo_id": "fixture/dataset"}))
    return path, config


def test_model_card_render_can_be_saved_without_hub_validation(monkeypatch, tmp_path, legacy_checkpoint):
    _, config = legacy_checkpoint

    def forbid_validation(self):
        pytest.fail("Local rendering attempted Hub validation")

    monkeypatch.setattr(ModelCard, "validate", forbid_validation)
    card = generate_model_card(config, validate_on_hub=False)
    target = tmp_path / "README.md"
    card.save(target)
    reloaded = ModelCard(target.read_text())
    assert reloaded.data.library_name == "lerobot"
    assert reloaded.data.license == "apache-2.0"


def test_default_model_card_generation_still_validates(monkeypatch, legacy_checkpoint):
    _, config = legacy_checkpoint
    calls = []
    monkeypatch.setattr(ModelCard, "validate", lambda self: calls.append(self))
    card = generate_model_card(config)
    assert calls == [card]


@pytest.mark.parametrize("publish, invalid", [(False, False), (True, False), (True, True)])
def test_migration_validates_final_metadata_only_for_publish(
    monkeypatch, tmp_path, legacy_checkpoint, publish, invalid
):
    path, _ = legacy_checkpoint
    output = tmp_path / "migrated"
    validated = []
    uploads = []

    def validate(card):
        validated.append(card.data.to_dict())
        if invalid:
            raise ValueError("invalid final metadata")

    class FakeApi:
        def upload_folder(self, **kwargs):
            uploads.append(kwargs)

    monkeypatch.setattr(ModelCard, "validate", validate)
    monkeypatch.setattr(migration, "HfApi", FakeApi)
    monkeypatch.setattr(migration, "get_policy_class", lambda _: DummyCheckpointPolicy)
    monkeypatch.setattr(
        migration,
        "make_pre_post_processors",
        lambda policy_cfg, dataset_stats: make_default_pre_post_processors(policy_cfg, dataset_stats),
    )
    argv = ["migration", "--pretrained-path", str(path), "--output-dir", str(output)]
    if publish:
        argv += ["--push-to-hub", "--hub-repo-id", "fixture/migrated"]
    monkeypatch.setattr(sys, "argv", argv)
    if invalid:
        with pytest.raises(ValueError, match="invalid final metadata"):
            migration.main()
        assert not uploads
    else:
        migration.main()
        assert (output / "README.md").is_file()
        assert (output / "model.safetensors").is_file()
        assert len(uploads) == int(publish)
    assert len(validated) == int(publish)
    if publish:
        assert validated[0]["datasets"] == "fixture/dataset"
        assert validated[0]["license"] == "apache-2.0"
        assert "dummy_migration_card" in validated[0]["tags"]
