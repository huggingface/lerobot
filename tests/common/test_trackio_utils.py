#!/usr/bin/env python

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

import json
import logging
import re
import sqlite3
import subprocess
import sys
import textwrap
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, sentinel

import draccus
import pytest

from lerobot.common.trackio_utils import TrackioLogger, cfg_to_group, get_safe_artifact_name
from lerobot.configs.default import TrackioConfig
from lerobot.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401  (registers `--policy.type=act`)
from lerobot.utils.constants import PRETRAINED_MODEL_DIR


@pytest.fixture
def trackio(monkeypatch):
    """Fake ``trackio`` module, installed before the logger imports it.

    The fast CI tiers do not install the ``training`` extra, so the installed-package check is patched out
    and the mock carries a real ``__spec__`` for ``importlib.util.find_spec``. ``init`` returns a local run
    (``url`` is None) unless a test says otherwise.
    """
    module = MagicMock()
    module.__spec__ = ModuleSpec("trackio", None)
    module.init.return_value.url = None
    monkeypatch.setitem(sys.modules, "trackio", module)
    monkeypatch.setattr("lerobot.common.trackio_utils.require_package", lambda *args, **kwargs: None)
    return module


def make_cfg(tmp_path: Path, **overrides) -> SimpleNamespace:
    cfg = SimpleNamespace(
        trackio=TrackioConfig(),
        output_dir=tmp_path,
        job_name="test-run",
        env=None,
        trainable_config=SimpleNamespace(type="test-policy"),
        is_reward_model_training=False,
        seed=42,
        dataset=None,
        resume=False,
        to_dict=lambda: {},
    )
    cfg.__dict__.update(overrides)
    return cfg


def make_checkpoint(tmp_path: Path, *filenames: str) -> Path:
    """Creates ``<tmp_path>/000100/<PRETRAINED_MODEL_DIR>/`` containing the given (empty) files."""
    checkpoint_dir = tmp_path / "000100"
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_model_dir.mkdir(parents=True)
    for filename in filenames:
        (pretrained_model_dir / filename).touch()
    return checkpoint_dir


# --- TrackioConfig ---


def test_trackio_config_defaults():
    cfg = TrackioConfig()

    assert cfg.enable is False
    assert cfg.disable_artifact is True
    assert cfg.project == "lerobot"
    assert cfg.space_id is None
    assert cfg.private is None
    assert cfg.run_name is None
    assert cfg.resume is None


@pytest.mark.parametrize("mode", ["allow", "must", "never", None])
def test_trackio_config_accepts_resume_modes(mode):
    assert TrackioConfig(resume=mode).resume == mode


def test_trackio_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="trackio.resume"):
        TrackioConfig(resume="auto")


# --- cfg_to_group / get_safe_artifact_name ---


def test_cfg_to_group_without_dataset_or_env(tmp_path):
    cfg = make_cfg(tmp_path)

    assert cfg_to_group(cfg) == "policy:test-policy-seed:42"


def test_cfg_to_group_with_dataset_and_env(tmp_path):
    cfg = make_cfg(
        tmp_path,
        dataset=SimpleNamespace(repo_id="user/repo"),
        env=SimpleNamespace(type="pusht", fps=10),
    )

    assert cfg_to_group(cfg) == "policy:test-policy-seed:42-dataset:user/repo-env:pusht"


def test_cfg_to_group_reward_model_prefix(tmp_path):
    cfg = make_cfg(tmp_path, is_reward_model_training=True, trainable_config=SimpleNamespace(type="rm"))

    assert cfg_to_group(cfg) == "reward_model:rm-seed:42"


def test_get_safe_artifact_name_replaces_separators():
    assert get_safe_artifact_name("policy:act-seed:42-dataset:user/repo-000100") == (
        "policy_act-seed_42-dataset_user_repo-000100"
    )


def test_get_safe_artifact_name_keeps_safe_names():
    assert get_safe_artifact_name("already-safe_name") == "already-safe_name"


# --- TrackioLogger.__init__ ---


def test_trackio_logger_forwards_init_kwargs(trackio, tmp_path):
    cfg = make_cfg(
        tmp_path,
        trackio=TrackioConfig(project="my-project", space_id="user/space", private=True),
        dataset=SimpleNamespace(repo_id="user/repo"),
        env=SimpleNamespace(type="pusht", fps=10),
        to_dict=lambda: sentinel.config,
    )

    TrackioLogger(cfg)

    trackio.init.assert_called_once()
    kwargs = dict(trackio.init.call_args.kwargs)
    name = kwargs.pop("name")
    assert re.fullmatch(r"test-run-[0-9a-f]{8}", name)
    assert cfg.trackio.run_name == name, "the generated run name is persisted on the config"
    assert kwargs == {
        "project": "my-project",
        "group": cfg_to_group(cfg),
        "space_id": "user/space",
        "private": True,
        "config": sentinel.config,
        "resume": "never",
    }


def test_trackio_logger_reuses_persisted_run_name(trackio, tmp_path):
    cfg = make_cfg(tmp_path, trackio=TrackioConfig(run_name="my-run"))

    TrackioLogger(cfg)

    assert trackio.init.call_args.kwargs["name"] == "my-run"
    assert cfg.trackio.run_name == "my-run"


def test_trackio_logger_resume_without_run_name_starts_new_run(trackio, tmp_path, caplog):
    cfg = make_cfg(tmp_path, resume=True)

    with caplog.at_level(logging.WARNING):
        TrackioLogger(cfg)

    assert trackio.init.call_args.kwargs["resume"] == "never"
    assert re.fullmatch(r"test-run-[0-9a-f]{8}", trackio.init.call_args.kwargs["name"])
    assert "no Trackio run name" in caplog.text


@pytest.mark.parametrize(
    ("cfg_resume", "run_name", "trackio_resume", "expected"),
    [
        (False, None, None, "never"),
        (True, "my-run", None, "allow"),
        (True, "my-run", "must", "must"),
        (False, None, "allow", "allow"),
        (False, "my-run", None, "never"),
    ],
)
def test_trackio_logger_resume(trackio, tmp_path, cfg_resume, run_name, trackio_resume, expected):
    cfg = make_cfg(
        tmp_path, trackio=TrackioConfig(run_name=run_name, resume=trackio_resume), resume=cfg_resume
    )

    TrackioLogger(cfg)

    assert trackio.init.call_args.kwargs["resume"] == expected


def test_trackio_logger_reports_space_and_stores_resolved_space_id(trackio, tmp_path, caplog):
    trackio.init.return_value.url = "alice/my-space"
    cfg = make_cfg(tmp_path, trackio=TrackioConfig(space_id="my-space"))

    with caplog.at_level(logging.INFO):
        TrackioLogger(cfg)

    assert cfg.trackio.space_id == "alice/my-space"
    assert "https://huggingface.co/spaces/alice/my-space" in caplog.text


def test_trackio_logger_reports_self_hosted_server(trackio, tmp_path, caplog, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    trackio.init.return_value.url = "https://tracker.example.com"
    cfg = make_cfg(tmp_path)

    with caplog.at_level(logging.INFO):
        TrackioLogger(cfg)

    assert cfg.trackio.space_id is None
    assert caplog.messages[-1] == "Track this run --> https://tracker.example.com"


def test_trackio_logger_reports_local_dashboard_command(trackio, tmp_path, caplog):
    cfg = make_cfg(tmp_path, trackio=TrackioConfig(project="my-project"))

    with caplog.at_level(logging.INFO):
        TrackioLogger(cfg)

    assert 'trackio show --project "my-project"' in caplog.text


# --- log_dict ---


def test_log_dict_with_step(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_dict({"loss": 0.5, "lr": 1e-4, "status": "ok"}, step=10)

    trackio.log.assert_called_once_with({"train/loss": 0.5, "train/lr": 1e-4, "train/status": "ok"}, step=10)


def test_log_dict_eval_mode(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_dict({"success_rate": 0.9}, step=3, mode="eval")

    trackio.log.assert_called_once_with({"eval/success_rate": 0.9}, step=3)


def test_log_dict_invalid_mode_raises(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    with pytest.raises(ValueError):
        logger.log_dict({"loss": 0.5}, step=1, mode="test")

    trackio.log.assert_not_called()


def test_log_dict_without_step_or_custom_step_key_raises(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    with pytest.raises(ValueError, match="step or custom_step_key"):
        logger.log_dict({"loss": 0.5})

    trackio.log.assert_not_called()


def test_log_dict_skips_non_scalar_values(trackio, tmp_path, caplog):
    logger = TrackioLogger(make_cfg(tmp_path))

    with caplog.at_level(logging.WARNING):
        logger.log_dict({"loss": 0.5, "weights": [0.1, 0.2], "obj": object()}, step=1)

    trackio.log.assert_called_once_with({"train/loss": 0.5}, step=1)
    assert '"weights"' in caplog.text
    assert '"obj"' in caplog.text


def test_log_dict_logs_nothing_when_everything_is_skipped(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_dict({"weights": [0.1, 0.2], "obj": object()}, step=1)

    trackio.log.assert_not_called()


def test_log_dict_with_custom_step_key(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_dict({"loss": 0.5, "interaction_step": 7}, custom_step_key="interaction_step")

    trackio.log.assert_called_once_with({"train/loss": 0.5, "train/interaction_step": 7})
    assert "step" not in trackio.log.call_args.kwargs


def test_log_dict_custom_step_key_missing_raises(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    with pytest.raises(ValueError, match="interaction_step"):
        logger.log_dict({"loss": 0.5}, custom_step_key="interaction_step")

    trackio.log.assert_not_called()


# --- log_video ---


def test_log_video(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_video("eval/videos/episode_0.mp4", step=5, mode="eval")

    trackio.Video.assert_called_once_with("eval/videos/episode_0.mp4", caption="eval step 5")
    trackio.log.assert_called_once_with({"eval/video": trackio.Video.return_value}, step=5)


def test_log_video_invalid_mode_raises(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    with pytest.raises(ValueError):
        logger.log_video("video.mp4", step=5, mode="test")

    trackio.Video.assert_not_called()
    trackio.log.assert_not_called()


# --- log_policy ---


def test_log_policy_standard_checkpoint(trackio, tmp_path):
    logger = TrackioLogger(
        make_cfg(
            tmp_path,
            dataset=SimpleNamespace(repo_id="user/repo"),
            trackio=TrackioConfig(disable_artifact=False),
        )
    )
    checkpoint_dir = make_checkpoint(tmp_path, "model.safetensors")

    logger.log_policy(checkpoint_dir)

    trackio.Artifact.assert_called_once()
    artifact_name = trackio.Artifact.call_args.args[0]
    assert ":" not in artifact_name
    assert "/" not in artifact_name
    assert artifact_name.endswith("-000100")
    assert trackio.Artifact.call_args.kwargs == {"type": "model"}

    artifact = trackio.Artifact.return_value
    artifact.add_file.assert_called_once_with(checkpoint_dir / PRETRAINED_MODEL_DIR / "model.safetensors")
    trackio.log_artifact.assert_called_once_with(artifact)


def test_log_policy_peft_checkpoint(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path, trackio=TrackioConfig(disable_artifact=False)))
    checkpoint_dir = make_checkpoint(
        tmp_path, "adapter_model.safetensors", "adapter_config.json", "config.json"
    )
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

    logger.log_policy(checkpoint_dir)

    artifact = trackio.Artifact.return_value
    assert artifact.add_file.call_args_list == [
        call(pretrained_model_dir / "adapter_model.safetensors"),
        call(pretrained_model_dir / "adapter_config.json"),
        call(pretrained_model_dir / "config.json"),
    ]
    trackio.log_artifact.assert_called_once_with(artifact)


def test_log_policy_empty_checkpoint_skips_upload(trackio, tmp_path, caplog):
    logger = TrackioLogger(make_cfg(tmp_path, trackio=TrackioConfig(disable_artifact=False)))
    checkpoint_dir = make_checkpoint(tmp_path)

    with caplog.at_level(logging.WARNING):
        logger.log_policy(checkpoint_dir)

    trackio.Artifact.return_value.add_file.assert_not_called()
    trackio.log_artifact.assert_not_called()
    assert "Skipping model artifact upload" in caplog.text


def test_log_policy_upload_failure_is_not_fatal(trackio, tmp_path, caplog):
    logger = TrackioLogger(make_cfg(tmp_path, trackio=TrackioConfig(disable_artifact=False)))
    checkpoint_dir = make_checkpoint(tmp_path, "model.safetensors")
    trackio.log_artifact.side_effect = RuntimeError("space is sleeping")

    with caplog.at_level(logging.WARNING):
        logger.log_policy(checkpoint_dir)

    assert "Skipping Trackio model artifact upload" in caplog.text
    assert "space is sleeping" in caplog.text


def test_log_policy_artifacts_disabled_by_default(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))
    checkpoint_dir = make_checkpoint(tmp_path, "model.safetensors")

    logger.log_policy(checkpoint_dir)

    trackio.Artifact.assert_not_called()
    trackio.log_artifact.assert_not_called()


# --- finish ---


def test_finish(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.finish()

    trackio.finish.assert_called_once_with()


# --- TrainPipelineConfig integration ---


def _parse_train_config(*extra_args: str) -> TrainPipelineConfig:
    return draccus.parse(
        TrainPipelineConfig, args=["--dataset.repo_id=user/repo", "--policy.type=act", *extra_args]
    )


def test_train_config_parses_trackio_flags_and_round_trips(tmp_path):
    cfg = _parse_train_config(
        "--trackio.enable=true",
        "--trackio.disable_artifact=true",
        "--trackio.project=my-project",
        "--trackio.space_id=user/space",
        "--trackio.private=true",
        "--trackio.run_name=my-run",
        "--trackio.resume=allow",
    )

    assert cfg.trackio == TrackioConfig(
        enable=True,
        disable_artifact=True,
        project="my-project",
        space_id="user/space",
        private=True,
        run_name="my-run",
        resume="allow",
    )

    # The checkpoint's train_config.json is what carries run_name across a resume.
    cfg.save_pretrained(tmp_path)
    reloaded = TrainPipelineConfig.from_pretrained(tmp_path)

    assert reloaded.trackio == cfg.trackio


def test_train_config_migrates_legacy_wandb_section(tmp_path, caplog):
    _parse_train_config().save_pretrained(tmp_path)
    config_file = tmp_path / TRAIN_CONFIG_NAME
    payload = json.loads(config_file.read_text())
    del payload["trackio"]
    payload["wandb"] = {
        "enable": True,
        "disable_artifact": True,
        "project": "old-project",
        "entity": "team",
        "notes": None,
        "run_id": "abc123",
        "resume": None,
        "mode": "online",
        "console": "wrap",
        "console_multipart": False,
        "console_chunk_max_seconds": 0,
        "add_tags": True,
    }
    config_file.write_text(json.dumps(payload))

    with caplog.at_level(logging.WARNING):
        cfg = TrainPipelineConfig.from_pretrained(tmp_path)

    assert cfg.trackio == TrackioConfig(enable=True, disable_artifact=True, project="old-project")
    assert "legacy `wandb` section" in caplog.text


def test_train_config_explicit_trackio_section_wins_over_legacy_wandb(tmp_path):
    _parse_train_config("--trackio.project=new-project").save_pretrained(tmp_path)
    config_file = tmp_path / TRAIN_CONFIG_NAME
    payload = json.loads(config_file.read_text())
    payload["wandb"] = {"enable": True, "project": "old-project"}
    config_file.write_text(json.dumps(payload))

    cfg = TrainPipelineConfig.from_pretrained(tmp_path)

    assert cfg.trackio == TrackioConfig(project="new-project")


# --- real package ---

_END_TO_END_SCRIPT = """
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lerobot.common.trackio_utils import TrackioLogger
from lerobot.configs.default import TrackioConfig
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

checkpoint_dir = Path(sys.argv[1])
pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
pretrained_dir.mkdir(parents=True)
(pretrained_dir / "model.safetensors").write_bytes(b"0" * 16)


def make_cfg(trackio_cfg, resume=False):
    return SimpleNamespace(
        trackio=trackio_cfg,
        output_dir=checkpoint_dir.parent,
        job_name="e2e_job",
        env=None,
        trainable_config=SimpleNamespace(type="act"),
        is_reward_model_training=False,
        seed=7,
        dataset=SimpleNamespace(repo_id="user/repo"),
        resume=resume,
        to_dict=lambda: {"seed": 7},
    )


first_cfg = TrackioConfig(enable=True, project="e2e", disable_artifact=False)
first = TrackioLogger(make_cfg(first_cfg))
first.log_dict({"loss": 1.0, "ignored": [1]}, step=0)
first.log_dict({"loss": 0.5}, step=1)
first.log_dict({"success": 0.25}, step=1, mode="eval")
first.log_dict({"Optimization step": 3, "Hz": 2.0}, custom_step_key="Optimization step")
first.log_policy(checkpoint_dir)
first.finish()

# A checkpoint resume reattaches to the run through the persisted run_name.
resumed = TrackioLogger(make_cfg(TrackioConfig(enable=True, project="e2e", run_name=first_cfg.run_name), resume=True))
resumed.log_dict({"loss": 0.25}, step=2)
resumed.finish()

print(json.dumps({"run_name": first_cfg.run_name, "same_run": resumed._run.id == first._run.id}))
"""


def test_trackio_logger_end_to_end_with_real_package(tmp_path):
    """Runs the logger against the installed trackio in a subprocess (its storage dir is read at import)."""
    pytest.importorskip("trackio")
    storage_dir = tmp_path / "trackio"
    checkpoint_dir = tmp_path / "checkpoints" / "000100"
    script = tmp_path / "e2e.py"
    script.write_text(textwrap.dedent(_END_TO_END_SCRIPT))

    result = subprocess.run(
        [sys.executable, str(script), str(checkpoint_dir)],
        env={"PATH": "", "HOME": str(tmp_path), "TRACKIO_DIR": str(storage_dir)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert re.fullmatch(r"e2e_job-[0-9a-f]{8}", summary["run_name"])
    assert summary["same_run"] is True

    with sqlite3.connect(storage_dir / "e2e.db") as db:
        rows = db.execute("SELECT run_name, step, metrics FROM metrics ORDER BY id").fetchall()
    logged = [(name, step, json.loads(metrics)) for name, step, metrics in rows]
    assert [row[0] for row in logged] == [summary["run_name"]] * len(logged)
    assert logged[0][1:] == (0, {"train/loss": 1.0})
    assert logged[2][1:] == (1, {"eval/success": 0.25})
    assert logged[3][2] == {"train/Optimization step": 3, "train/Hz": 2.0}
    assert logged[-1][1:] == (2, {"train/loss": 0.25})
    with sqlite3.connect(storage_dir / "e2e.db") as db:
        artifacts = db.execute("SELECT name, type FROM artifacts").fetchall()
    assert artifacts == [("policy_act-seed_7-dataset_user_repo-000100", "model")]
