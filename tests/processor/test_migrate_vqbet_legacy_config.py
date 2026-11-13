import json
import sys

import pytest

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.processor import migrate_policy_normalization as migration


class ConfigCapturedError(Exception):
    pass


@pytest.fixture
def migrate_until_config(tmp_path, monkeypatch):
    source = tmp_path / "original"
    source.mkdir()
    config_path = source / "config.json"
    captured = {}
    real_factory = migration.make_policy_config

    def capture_config(policy_type, **kwargs):
        captured["config"] = real_factory(policy_type, **kwargs)
        raise ConfigCapturedError

    monkeypatch.setattr(migration, "make_policy_config", capture_config)
    monkeypatch.setattr(migration, "load_safetensors", lambda _: {})
    monkeypatch.setattr(
        sys,
        "argv",
        ["migrate", "--pretrained-path", str(source), "--output-dir", str(tmp_path / "converted")],
    )

    def run(policy_type="vqbet", **extra):
        config = {
            "type": policy_type,
            "device": "cpu",
            "pretrained_backbone_weights": None,
            "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
            "input_features": {
                "observation.image": {"type": "VISUAL", "shape": [3, 96, 96]},
                "observation.state": {"type": "STATE", "shape": [2]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [2]}},
            **extra,
        }
        before = json.dumps(config).encode()
        config_path.write_bytes(before)
        try:
            migration.main()
        finally:
            assert config_path.read_bytes() == before

    return run, captured


@pytest.mark.parametrize("legacy_field", [False, True])
def test_vqbet_legacy_config_reaches_real_config_constructor(migrate_until_config, legacy_field):
    run, captured = migrate_until_config
    extra = {"mlp_hidden_dim": 1024} if legacy_field else {}
    with pytest.raises(ConfigCapturedError):
        run(**extra)
    config = captured["config"]
    assert config.type == "vqbet"
    assert config.input_features["observation.image"].type == FeatureType.VISUAL
    assert config.output_features["action"].shape == (2,)
    assert config.normalization_mapping[FeatureType.VISUAL] == NormalizationMode.IDENTITY


def test_vqbet_unknown_fields_still_raise(migrate_until_config):
    run, _ = migrate_until_config
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown_model_setting'"):
        run(unknown_model_setting=1)


def test_unused_vqbet_field_is_not_ignored_for_act(migrate_until_config):
    run, _ = migrate_until_config
    with pytest.raises(TypeError, match="unexpected keyword argument 'mlp_hidden_dim'"):
        run(policy_type="act", mlp_hidden_dim=1024)
