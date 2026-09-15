"""Regression test: PreTrainedConfig.from_pretrained must backfill
pretrained_path / pretrained_revision on the returned config.

config.json only carries train-time fields; the source path is not serialized.
Before the fix, from_pretrained() returned a config with pretrained_path=None,
and downstream make_policy() / make_pre_post_processors() silently built an
untrained processor pipeline (empty normalization stats) instead of loading
the pretrained stats — the policy appears to load fine but acts blind.
"""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("_dummy_backfill")
@dataclass
class _DummyBackfillConfig(PreTrainedConfig):
    lr: float = 1e-4

    # abstract members of PreTrainedConfig (not exercised by this test)
    def action_delta_indices(self) -> dict:
        return {}

    def observation_delta_indices(self) -> dict:
        return {}

    def reward_delta_indices(self) -> dict:
        return {}

    def get_optimizer_preset(self) -> dict:  # pragma: no cover
        return {}

    def get_scheduler_preset(self) -> dict:  # pragma: no cover
        return {}

    def validate_features(self) -> None:
        pass


def _write_checkpoint(tmpdir: str, extra: dict | None = None) -> Path:
    config = {
        "type": "_dummy_backfill",
        "n_obs_steps": 2,
        "input_features": {},
        "output_features": {},
        **(extra or {}),
    }
    path = Path(tmpdir) / "config.json"
    path.write_text(json.dumps(config))
    return Path(tmpdir)


def test_from_pretrained_local_dir_backfills_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoint(tmpdir)
        cfg = PreTrainedConfig.from_pretrained(tmpdir)
        assert isinstance(cfg, _DummyBackfillConfig)
        assert str(cfg.pretrained_path) == tmpdir
        assert cfg.pretrained_revision is None


def test_from_pretrained_cli_overrides_still_applied():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoint(tmpdir)
        cfg = PreTrainedConfig.from_pretrained(tmpdir, cli_overrides=["--lr=0.5"])
        assert cfg.lr == 0.5
        assert str(cfg.pretrained_path) == tmpdir
