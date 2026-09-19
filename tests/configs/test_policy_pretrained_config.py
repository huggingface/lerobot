import json

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


def test_from_pretrained_restores_checkpoint_source(tmp_path):
    config = ACTConfig(device="cpu")
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["pretrained_path"] is None

    loaded = PreTrainedConfig.from_pretrained(tmp_path, revision="checkpoint-sha")

    assert loaded.pretrained_path == str(tmp_path)
    assert loaded.pretrained_revision == "checkpoint-sha"
