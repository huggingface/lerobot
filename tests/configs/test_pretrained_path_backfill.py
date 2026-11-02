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

import pytest
from draccus.utils import DecodingError

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


def test_from_pretrained_backfills_path_type():
    # The field is `Path | None`; the backfill must not assign a bare str.
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_checkpoint(tmpdir)
        cfg = PreTrainedConfig.from_pretrained(tmpdir)
        assert isinstance(cfg.pretrained_path, Path)


def test_from_pretrained_hub_id_backfills_repo_and_revision(monkeypatch, tmp_path):
    # Hub-id resolution goes through hf_hub_download; mock it to hand back a
    # local config file and check the backfilled path/revision track the args.
    import lerobot.configs.policies as policies_mod

    def fake_download(*, repo_id, filename, revision=None, **kwargs):
        assert repo_id == "user/dummy_backfill"
        assert filename == "config.json"
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "type": "_dummy_backfill",
                    "n_obs_steps": 2,
                    "input_features": {},
                    "output_features": {},
                }
            )
        )
        return str(cfg_file)

    monkeypatch.setattr(policies_mod, "hf_hub_download", fake_download)
    cfg = PreTrainedConfig.from_pretrained("user/dummy_backfill", revision="v2")
    assert isinstance(cfg, _DummyBackfillConfig)
    assert str(cfg.pretrained_path) == "user/dummy_backfill"
    assert cfg.pretrained_revision == "v2"


# ── RewardModelConfig has the same pattern and needs the same backfill ──

from lerobot.configs.rewards import RewardModelConfig  # noqa: E402


@RewardModelConfig.register_subclass("_dummy_reward_backfill")
@dataclass
class _DummyRewardBackfillConfig(RewardModelConfig):
    lr: float = 1e-4

    def get_optimizer_preset(self):  # pragma: no cover
        return None


def test_reward_config_from_pretrained_backfills_path():
    # RewardModelConfig.from_pretrained left pretrained_path=None; direct
    # callers (PreTrainedRewardModel.from_pretrained, make_reward_model)
    # branch on it — e.g. RobometerRewardModel.__init__ takes the "fresh
    # model" branch instead of loading the checkpoint.
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "config.json").write_text(json.dumps({"type": "_dummy_reward_backfill", "lr": 0.1}))
        cfg = RewardModelConfig.from_pretrained(tmpdir)
        assert isinstance(cfg, _DummyRewardBackfillConfig)
        assert cfg.pretrained_path == tmpdir
        assert cfg.pretrained_revision is None


@pytest.mark.parametrize("reward", [False, True])
@pytest.mark.parametrize("failure", [None, "parse", "write"])
def test_pretrained_config_removes_temporary_json(tmp_path, monkeypatch, reward, failure):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(
        json.dumps({"type": "_dummy_reward_backfill" if reward else "_dummy_backfill", "lr": 0.1})
    )
    original = config_path.read_bytes()
    temporary_paths = []
    create_temporary_file = tempfile.NamedTemporaryFile

    def track_temporary_file(*args, **kwargs):
        kwargs.setdefault("dir", tmp_path)
        handle = create_temporary_file(*args, **kwargs)
        temporary_paths.append(Path(handle.name))
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", track_temporary_file)
    if failure == "write":

        def fail_write(*args, **kwargs):
            raise OSError("config write failed")

        monkeypatch.setattr(json, "dump", fail_write)

    config_class = RewardModelConfig if reward else PreTrainedConfig
    overrides = ["--lr=invalid"] if failure == "parse" else ["--lr=0.5"]
    if failure:
        with pytest.raises(DecodingError if failure == "parse" else OSError):
            config_class.from_pretrained(checkpoint, cli_overrides=overrides)
    else:
        config = config_class.from_pretrained(checkpoint, cli_overrides=overrides)
        assert config.lr == 0.5

    assert temporary_paths
    assert all(not path.exists() for path in temporary_paths)
    assert config_path.read_bytes() == original
