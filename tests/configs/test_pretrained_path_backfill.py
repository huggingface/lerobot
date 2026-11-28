"""Regression test: PreTrainedConfig.from_pretrained must backfill
pretrained_path / pretrained_revision on the returned config.

config.json only carries train-time fields; the source path is not serialized.
Before the fix, from_pretrained() returned a config with pretrained_path=None,
and downstream make_policy() / make_pre_post_processors() silently built an
untrained processor pipeline (empty normalization stats) instead of loading
the pretrained stats — the policy appears to load fine but acts blind.
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from draccus.utils import DecodingError

from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig


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


# ── The CLI entry points must hand --<field>.pretrained_revision to from_pretrained ──


def _eval_config() -> EvalPipelineConfig:
    return EvalPipelineConfig(env=None)


def _train_config() -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="user/dataset"))
    cfg._resolve_pretrained_from_cli()
    return cfg


@pytest.mark.parametrize(
    ("argv", "build", "config_cls", "field"),
    [
        (["lerobot-eval", "--policy.path=user/policy"], _eval_config, PreTrainedConfig, "policy"),
        (["lerobot-train", "--policy.path=user/policy"], _train_config, PreTrainedConfig, "policy"),
        (
            ["lerobot-train", "--reward_model.path=user/reward_model"],
            _train_config,
            RewardModelConfig,
            "reward_model",
        ),
    ],
)
def test_cli_pretrained_revision_reaches_from_pretrained(monkeypatch, argv, build, config_cls, field):
    # The CLI revision reaches from_pretrained and the returned config, ahead of the --config_path file.
    captured = {}

    def fake_from_pretrained(cls, pretrained_name_or_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            type="_dummy_backfill", pretrained_path=None, pretrained_revision=kwargs["revision"]
        )

    monkeypatch.setattr(sys, "argv", [*argv, f"--{field}.pretrained_revision=cli-sha"])
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: ["--pretrained_revision=yaml-sha"])
    monkeypatch.setattr(config_cls, "from_pretrained", classmethod(fake_from_pretrained))

    cfg = build()

    assert captured["revision"] == "cli-sha"
    assert getattr(cfg, field).pretrained_revision == "cli-sha"


@pytest.mark.parametrize("serialized_trust,caller_trust", [(True, False), (False, True)])
@pytest.mark.parametrize(
    ("config_cls", "subclass", "field"),
    [
        (PreTrainedConfig, _DummyBackfillConfig, "policy"),
        (RewardModelConfig, _DummyRewardBackfillConfig, "reward_model"),
    ],
)
def test_checkpoint_cannot_grant_code_consent(
    monkeypatch, tmp_path, serialized_trust, caller_trust, config_cls, subclass, field
):
    observed = []
    original = subclass.__post_init__

    def post_init(self):
        observed.append(self.trust_remote_code)
        original(self)

    monkeypatch.setattr(subclass, "__post_init__", post_init)
    (tmp_path / "config.json").write_text(
        json.dumps({"type": subclass.get_choice_name(subclass), "trust_remote_code": serialized_trust})
    )
    cfg = config_cls.from_pretrained(tmp_path, trust_remote_code=caller_trust)
    assert observed == [caller_trust]
    assert cfg.trust_remote_code is caller_trust
    cfg.save_pretrained(tmp_path)
    assert "trust_remote_code" not in json.loads((tmp_path / "config.json").read_text())
    TrainPipelineConfig(dataset=DatasetConfig(repo_id="user/data"), **{field: cfg}).save_pretrained(tmp_path)
    assert "trust_remote_code" not in json.loads((tmp_path / "train_config.json").read_text())[field]
    assert cfg.trust_remote_code is caller_trust


@pytest.mark.parametrize(
    ("build", "field", "config_type"),
    [
        (_eval_config, "policy", "_dummy_backfill"),
        (_train_config, "policy", "_dummy_backfill"),
        (_train_config, "reward_model", "_dummy_reward_backfill"),
    ],
)
def test_cli_consent_overrides_checkpoint(monkeypatch, tmp_path, build, field, config_type):
    (tmp_path / "config.json").write_text(json.dumps({"type": config_type, "trust_remote_code": False}))
    monkeypatch.setattr(
        sys, "argv", ["lerobot", f"--{field}.path={tmp_path}", f"--{field}.trust_remote_code=true"]
    )
    cfg = build()
    assert getattr(cfg, field).trust_remote_code is True
    assert not getattr(cfg, "trust_remote_code", False)


@pytest.mark.parametrize("source", ["hub", "directory", "checkpoint_file", "recipe"])
@pytest.mark.parametrize("caller_trust", [None, False, True])
@pytest.mark.parametrize(
    ("field", "subclass"),
    [("policy", _DummyBackfillConfig), ("reward_model", _DummyRewardBackfillConfig)],
)
def test_training_checkpoint_consent_is_reset_before_construction(
    monkeypatch, tmp_path, source, caller_trust, field, subclass
):
    import lerobot.configs.train as train_module

    observed = []
    original = subclass.__post_init__

    def post_init(self):
        observed.append(self.trust_remote_code)
        original(self)

    monkeypatch.setattr(subclass, "__post_init__", post_init)
    path = tmp_path / ("recipe.json" if source == "recipe" else "train_config.json")
    path.write_text(
        json.dumps(
            {
                "dataset": {"repo_id": "user/data"},
                field: {"type": subclass.get_choice_name(subclass), "trust_remote_code": True},
            }
        )
    )
    monkeypatch.setattr(train_module, "hf_hub_download", lambda **kwargs: str(path))
    checkpoint = "user/checkpoint" if source == "hub" else tmp_path if source == "directory" else path
    args = [] if caller_trust is None else [f"--{field}.trust_remote_code={str(caller_trust).lower()}"]

    config = TrainPipelineConfig.from_pretrained(checkpoint, cli_args=args)

    expected = source == "recipe" if caller_trust is None else caller_trust
    assert observed == [expected]
    assert getattr(config, field).trust_remote_code is expected


@pytest.mark.parametrize("caller_trust", [None, False, True])
@pytest.mark.parametrize("filename", ["train_config.json", "recipe.json"])
@pytest.mark.parametrize("field", ["policy", "reward_model"])
def test_cli_path_extraction_cannot_restore_checkpoint_consent(
    monkeypatch, tmp_path, caller_trust, filename, field
):
    config_cls = PreTrainedConfig if field == "policy" else RewardModelConfig
    subclass = _DummyBackfillConfig if field == "policy" else _DummyRewardBackfillConfig
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"type": subclass.get_choice_name(subclass)}))
    path = tmp_path / filename
    path.write_text(
        json.dumps(
            {"dataset": {"repo_id": "user/data"}, field: {"path": str(model), "trust_remote_code": True}}
        )
    )
    monkeypatch.setattr(parser, "_config_path_args", {})
    monkeypatch.setattr(parser, "_config_yaml_overrides", {})
    args = [] if caller_trust is None else [f"--{field}.trust_remote_code={str(caller_trust).lower()}"]
    monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--config_path={path}", *args])

    @parser.wrap()
    def load(config: TrainPipelineConfig):
        config._resolve_pretrained_from_cli()
        return config

    config = load()
    expected = filename == "recipe.json" if caller_trust is None else caller_trust
    assert isinstance(getattr(config, field), config_cls)
    assert getattr(config, field).trust_remote_code is expected


@pytest.mark.parametrize("consent", [False, True])
def test_training_export_omits_environment_classifier_consent(tmp_path, consent):
    from lerobot.envs.configs import HILSerlProcessorConfig, HILSerlRobotEnvConfig, RewardClassifierConfig

    classifier = RewardClassifierConfig(pretrained_path="user/classifier", trust_remote_code=consent)
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/data"),
        env=HILSerlRobotEnvConfig(processor=HILSerlProcessorConfig(reward_classifier=classifier)),
    )
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "train_config.json").read_text())
    assert "trust_remote_code" not in saved["env"]["processor"]["reward_classifier"]
    assert saved["env"]["processor"]["reward_classifier"]["pretrained_path"] == "user/classifier"
    assert classifier.trust_remote_code is consent


@pytest.mark.parametrize(
    "source", ["hub", "directory", "checkpoint_file", "recipe", "cli_checkpoint", "cli_recipe"]
)
@pytest.mark.parametrize("caller_trust", [None, False, True])
def test_training_environment_classifier_requires_fresh_consent(monkeypatch, tmp_path, source, caller_trust):
    import lerobot.configs.train as train_module

    is_recipe = source in ("recipe", "cli_recipe")
    path = tmp_path / ("recipe.json" if is_recipe else "train_config.json")
    payload = {
        "dataset": {"repo_id": "user/data"},
        # This also makes the CLI parser rewrite the checkpoint to a temporary file.
        "policy": {"type": "_dummy_backfill", "trust_remote_code": True},
        "env": {
            "type": "gym_manipulator",
            "processor": {
                "reward_classifier": {"pretrained_path": "user/classifier", "trust_remote_code": True}
            },
        },
    }
    path.write_text(json.dumps(payload))
    args = (
        []
        if caller_trust is None
        else [f"--env.processor.reward_classifier.trust_remote_code={str(caller_trust).lower()}"]
    )
    if source.startswith("cli_"):
        monkeypatch.setattr(parser, "_config_path_args", {})
        monkeypatch.setattr(parser, "_config_yaml_overrides", {})
        monkeypatch.setattr(sys, "argv", ["lerobot-train", f"--config_path={path}", *args])

        @parser.wrap()
        def load(config: TrainPipelineConfig):
            return config

        config = load()
    else:
        monkeypatch.setattr(train_module, "hf_hub_download", lambda **kwargs: str(path))
        checkpoint = "user/checkpoint" if source == "hub" else tmp_path if source == "directory" else path
        config = TrainPipelineConfig.from_pretrained(checkpoint, cli_args=args)

    classifier = config.env.processor.reward_classifier
    assert classifier.trust_remote_code is (is_recipe if caller_trust is None else caller_trust)
    assert classifier.pretrained_path == "user/classifier"
    assert json.loads(path.read_text()) == payload
