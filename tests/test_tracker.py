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

"""Tracker selection (`--tracker.type=wandb|trackio`), its config round-trip, and TrackioLogger.

The logger tests stub the `trackio` package, so they run whether or not it is installed.
"""

import json
import sys
import types

import draccus
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import lerobot.policies  # noqa: F401  # registers the choice registry `--policy.type` selects from
from lerobot.common.tracker_utils import TrackerLogger, cfg_to_group, make_tracker
from lerobot.configs.default import TrackioTrackerConfig, WandBTrackerConfig
from lerobot.configs.train import TrainPipelineConfig

BASE_ARGS = [
    "--dataset.repo_id=u/d",
    "--policy.type=act",
    "--policy.push_to_hub=false",
]


def parse(*extra: str) -> TrainPipelineConfig:
    return draccus.parse(TrainPipelineConfig, args=[*BASE_ARGS, *extra])


def round_trip(cfg: TrainPipelineConfig, tmp_path, drop_tracker_key: bool = False):
    """Save `cfg` the way a checkpoint does, then load it back."""
    config_file = tmp_path / "train_config.json"
    with open(config_file, "w") as f, draccus.config_type("json"):
        draccus.dump(cfg, f, indent=4)
    if drop_tracker_key:
        # Emulate a checkpoint written before the tracker field existed.
        payload = json.loads(config_file.read_text())
        del payload["tracker"]
        config_file.write_text(json.dumps(payload, indent=4))
    return TrainPipelineConfig.from_pretrained(config_file)


@pytest.fixture
def stub_wandb(monkeypatch):
    """Install a fake `wandb` module and return the calls it records."""
    calls = {"init": [], "log": []}

    fake = types.ModuleType("wandb")
    fake.run = types.SimpleNamespace(id="abc123", get_url=lambda: "http://wandb/run/abc123")
    fake.init = lambda **kwargs: calls["init"].append(kwargs)
    fake.log = lambda data=None, step=None: calls["log"].append((data, step))
    fake.define_metric = lambda *a, **kw: None

    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr("lerobot.common.tracker_utils.require_package", lambda *a, **kw: None)
    return calls


@pytest.fixture
def stub_trackio(monkeypatch):
    """Install a fake `trackio` module and return the calls it records."""
    calls = {"init": [], "log": [], "log_artifact": [], "video": []}

    class FakeRun:
        def __init__(self, name):
            self.name = name

    class FakeVideo:
        def __init__(self, value, caption=None, fps=None, format=None):
            self.value, self.fps, self.format = value, fps, format
            calls["video"].append(self)

    fake = types.ModuleType("trackio")
    fake.init = lambda **kwargs: (calls["init"].append(kwargs), FakeRun(kwargs["name"]))[1]
    fake.log = lambda metrics, step=None: calls["log"].append((metrics, step))
    fake.log_artifact = lambda path, name=None, type=None: calls["log_artifact"].append((path, name, type))
    fake.Video = FakeVideo

    monkeypatch.setitem(sys.modules, "trackio", fake)
    # The extra is not installed in CI; the stub stands in for it.
    monkeypatch.setattr("lerobot.common.tracker_utils.require_package", lambda *a, **kw: None)
    return calls


# ── Selection and CLI surface ─────────────────────────────────────────────────


def test_no_tracker_by_default():
    cfg = parse()
    cfg.validate()
    assert cfg.tracker is None
    assert make_tracker(cfg) is None


def test_tracker_type_selects_trackio():
    cfg = parse("--tracker.type=trackio", "--tracker.project=myproj", "--tracker.space_id=me/dash")
    assert isinstance(cfg.tracker, TrackioTrackerConfig)
    assert cfg.tracker.type == "trackio"
    assert (cfg.tracker.project, cfg.tracker.space_id) == ("myproj", "me/dash")


def test_tracker_type_selects_wandb():
    cfg = parse("--tracker.type=wandb", "--tracker.project=myproj", "--tracker.entity=me")
    assert isinstance(cfg.tracker, WandBTrackerConfig)
    assert cfg.tracker.type == "wandb"
    assert (cfg.tracker.project, cfg.tracker.entity) == ("myproj", "me")


def test_wandb_tracker_is_enabled_by_being_selected():
    """`enable` gates the legacy `--wandb.*` path only: a selected tracker must never claim to be off."""
    cfg = parse("--tracker.type=wandb", "--tracker.project=p")
    assert cfg.tracker.enable is True
    assert cfg.to_dict()["tracker"]["enable"] is True

    # And it must not masquerade as a switch, which would silently do nothing.
    with pytest.raises(SystemExit):
        parse("--tracker.type=wandb", "--tracker.enable=false")


def test_make_tracker_builds_a_wandb_logger(tmp_path, stub_wandb):
    """The wandb branch is the path every existing user takes, on both spellings of the flags."""
    from lerobot.common.wandb_utils import WandBLogger

    for args in (
        ("--tracker.type=wandb", "--tracker.project=p"),
        ("--wandb.enable=true", "--wandb.project=p"),
    ):
        cfg = parse(*args, f"--output_dir={tmp_path / '15-30-14_act'}")
        cfg.validate()
        logger = make_tracker(cfg)
        assert isinstance(logger, WandBLogger)
        assert isinstance(logger, TrackerLogger)
        # The logger reads the resolved tracker, not the legacy `wandb` block.
        assert logger.cfg is cfg.tracker
        # wandb owns run identity, so its id is written back for resume.
        assert cfg.tracker.run_id == "abc123"

    init_kwargs = stub_wandb["init"][0]
    assert init_kwargs["project"] == "p"
    assert init_kwargs["name"] == cfg.job_name


def test_make_tracker_rejects_an_unregistered_tracker_config():
    cfg = parse()
    cfg.validate()
    cfg.tracker = object()
    with pytest.raises(ValueError, match="Unsupported tracker config"):
        make_tracker(cfg)


# ── Config round-trip (this is what resume reads) ─────────────────────────────


@pytest.mark.parametrize(
    ("args", "expected_type"),
    [
        (("--tracker.type=trackio", "--tracker.project=rt"), TrackioTrackerConfig),
        (("--tracker.type=wandb", "--tracker.project=rt"), WandBTrackerConfig),
    ],
)
def test_tracker_survives_a_config_round_trip(tmp_path, args, expected_type):
    reloaded = round_trip(parse(*args), tmp_path)
    assert isinstance(reloaded.tracker, expected_type)
    assert reloaded.tracker.project == "rt"


def test_absent_tracker_survives_a_config_round_trip(tmp_path):
    cfg = parse()
    cfg.validate()
    assert cfg.to_dict()["tracker"] is None
    assert round_trip(cfg, tmp_path).tracker is None


# ── Legacy `--wandb.*` flags ──────────────────────────────────────────────────


def test_legacy_wandb_flags_build_a_wandb_tracker():
    cfg = parse("--wandb.enable=true", "--wandb.project=legacy", "--wandb.entity=me")
    assert cfg.tracker is None  # nothing happens until validate()
    cfg.validate()
    assert isinstance(cfg.tracker, WandBTrackerConfig)
    assert (cfg.tracker.project, cfg.tracker.entity) == ("legacy", "me")


def test_legacy_wandb_flags_are_ignored_when_disabled():
    cfg = parse("--wandb.project=legacy")
    cfg.validate()
    assert cfg.tracker is None


def test_an_explicit_tracker_wins_over_the_legacy_flags():
    cfg = parse("--wandb.enable=true", "--wandb.project=legacy", "--tracker.type=trackio")
    cfg.validate()
    assert isinstance(cfg.tracker, TrackioTrackerConfig)


def test_a_checkpoint_written_before_trackers_existed_still_loads(tmp_path):
    """Every existing checkpoint's train_config.json has no `tracker` key at all."""
    cfg = parse("--wandb.enable=true", "--wandb.project=old")
    reloaded = round_trip(cfg, tmp_path, drop_tracker_key=True)
    assert reloaded.tracker is None
    assert reloaded.wandb.enable is True
    reloaded.validate()
    assert isinstance(reloaded.tracker, WandBTrackerConfig)
    assert reloaded.tracker.project == "old"


# ── TrackioLogger ─────────────────────────────────────────────────────────────


def test_trackio_logger_names_a_fresh_run_after_its_output_dir(tmp_path, stub_trackio):
    cfg = parse("--tracker.type=trackio", f"--output_dir={tmp_path / '15-30-14_act'}")
    cfg.validate()
    logger = make_tracker(cfg)

    assert isinstance(logger, TrackerLogger)
    init_kwargs = stub_trackio["init"][0]
    # Not `job_name`: trackio resolves a resume by name to the most recently written run
    # carrying it, and job_name repeats across runs of the same policy and env.
    assert init_kwargs["name"] == "15-30-14_act"
    assert init_kwargs["resume"] == "never"
    assert init_kwargs["group"] == cfg_to_group(cfg)
    # Stored so that resuming this checkpoint reattaches to this run.
    assert cfg.tracker.run_id == "15-30-14_act"


def test_trackio_logger_reattaches_to_the_stored_run_on_resume(tmp_path, stub_trackio):
    cfg = parse(
        "--tracker.type=trackio",
        "--tracker.run_id=15-30-14_act",
        f"--output_dir={tmp_path / '16-00-00_resume'}",
    )
    cfg.validate()
    cfg.resume = True
    make_tracker(cfg)

    init_kwargs = stub_trackio["init"][0]
    assert init_kwargs["name"] == "15-30-14_act"
    assert init_kwargs["resume"] == "allow"


def test_trackio_logger_logs_scalars_and_skips_other_types(tmp_path, stub_trackio, caplog):
    cfg = parse("--tracker.type=trackio", f"--output_dir={tmp_path / 'run'}")
    cfg.validate()
    logger = make_tracker(cfg)

    logger.log_dict({"loss": 0.5, "note": "ok", "steps": 3, "tensor": object()}, step=7)
    metrics, step = stub_trackio["log"][0]
    assert metrics == {"train/loss": 0.5, "train/note": "ok", "train/steps": 3}
    assert step == 7
    assert "tensor" in caplog.text

    logger.log_dict({"reward": 1.0}, step=1, mode="eval")
    assert stub_trackio["log"][1][0] == {"eval/reward": 1.0}

    with pytest.raises(ValueError):
        logger.log_dict({"loss": 0.1}, step=1, mode="not_a_mode")
    with pytest.raises(ValueError, match="step or custom_step_key"):
        logger.log_dict({"loss": 0.1})


def test_trackio_logger_logs_a_custom_step_key_as_a_metric(tmp_path, stub_trackio):
    """Async RL logs against its own step counter; trackio has no `define_metric` to hide it."""
    cfg = parse("--tracker.type=trackio", f"--output_dir={tmp_path / 'run'}")
    cfg.validate()
    logger = make_tracker(cfg)

    logger.log_dict({"loss": 0.5, "Optimization step": 12}, custom_step_key="Optimization step")
    metrics, step = stub_trackio["log"][0]
    assert metrics == {"train/loss": 0.5, "train/Optimization step": 12}
    assert step is None  # trackio's own counter must not fight the custom one


def test_trackio_logger_logs_video(tmp_path, stub_trackio):
    cfg = parse("--tracker.type=trackio", f"--output_dir={tmp_path / 'run'}")
    cfg.validate()
    logger = make_tracker(cfg)

    logger.log_video("/tmp/rollout.mp4", step=3, mode="eval")
    metrics, step = stub_trackio["log"][0]
    assert set(metrics) == {"eval/video"}
    assert metrics["eval/video"] is stub_trackio["video"][0]
    assert step == 3


def test_trackio_logger_skips_artifacts_by_default(tmp_path, stub_trackio):
    """Checkpoints are multi-GB, so uploading them is opt-in."""
    cfg = parse("--tracker.type=trackio", f"--output_dir={tmp_path / 'run'}")
    cfg.validate()
    logger = make_tracker(cfg)
    assert cfg.tracker.disable_artifact is True

    logger.log_policy(tmp_path / "checkpoints" / "000100")
    assert stub_trackio["log_artifact"] == []


def test_trackio_logger_uploads_the_model_file_when_asked(tmp_path, stub_trackio):
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    cfg = parse(
        "--tracker.type=trackio",
        "--tracker.disable_artifact=false",
        f"--output_dir={tmp_path / 'run'}",
    )
    cfg.validate()
    logger = make_tracker(cfg)

    checkpoint_dir = tmp_path / "checkpoints" / "000100"
    model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights")

    logger.log_policy(checkpoint_dir)
    path, name, artifact_type = stub_trackio["log_artifact"][0]
    assert path == model_dir / "model.safetensors"
    assert artifact_type == "model"
    # Artifact names cannot carry ":" or "/", which cfg_to_group's tags do.
    assert ":" not in name and "/" not in name


def test_trackio_logger_warns_and_skips_a_missing_model_file(tmp_path, stub_trackio, caplog):
    cfg = parse(
        "--tracker.type=trackio",
        "--tracker.disable_artifact=false",
        f"--output_dir={tmp_path / 'run'}",
    )
    cfg.validate()
    logger = make_tracker(cfg)

    logger.log_policy(tmp_path / "checkpoints" / "000100")
    assert stub_trackio["log_artifact"] == []
    assert "Skipping model artifact upload" in caplog.text


def test_trackio_utils_does_not_import_trackio_at_module_level():
    """Importing the module must not require the optional extra."""
    sys.modules.pop("trackio", None)
    sys.modules.pop("lerobot.common.trackio_utils", None)
    import lerobot.common.trackio_utils  # noqa: F401

    assert "trackio" not in sys.modules
