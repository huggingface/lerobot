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
import shlex
import sqlite3
import subprocess
import sys
import textwrap
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
    """Fake ``trackio`` module, patched into the logger's module.

    The fast CI tiers do not install the ``training`` extra, so the module-level ``trackio`` name may be
    missing and the installed-package check is patched out. ``init`` returns a local run (``url`` is None)
    unless a test says otherwise.
    """
    module = MagicMock()
    module.init.return_value.url = None
    monkeypatch.setattr("lerobot.common.trackio_utils.trackio", module, raising=False)
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


@pytest.mark.parametrize("mode", ["allow", "must", "never", None])
def test_trackio_config_accepts_resume_modes(mode):
    assert TrackioConfig(resume=mode).resume == mode


def test_trackio_config_rejects_unknown_resume_mode():
    with pytest.raises(ValueError, match="trackio.resume"):
        TrackioConfig(resume="auto")


# --- cfg_to_group / get_safe_artifact_name ---


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "policy:test-policy-seed:42"),
        (
            {"dataset": SimpleNamespace(repo_id="user/repo"), "env": SimpleNamespace(type="pusht", fps=10)},
            "policy:test-policy-seed:42-dataset:user/repo-env:pusht",
        ),
        (
            {"is_reward_model_training": True, "trainable_config": SimpleNamespace(type="rm")},
            "reward_model:rm-seed:42",
        ),
    ],
)
def test_cfg_to_group(tmp_path, overrides, expected):
    assert cfg_to_group(make_cfg(tmp_path, **overrides)) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("policy:act-seed:42-dataset:user/repo-000100", "policy_act-seed_42-dataset_user_repo-000100"),
        ("already-safe_name", "already-safe_name"),
    ],
)
def test_get_safe_artifact_name(name, expected):
    assert get_safe_artifact_name(name) == expected


# --- TrackioLogger.__init__ ---


def test_trackio_logger_forwards_init_kwargs(trackio, tmp_path):
    cfg = make_cfg(
        tmp_path,
        trackio=TrackioConfig(
            project="my-project", space_id="user/space", private=True, capture_console=False
        ),
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
        "capture_console": False,
    }


@pytest.mark.parametrize(
    ("cfg_resume", "run_name", "trackio_resume", "expected", "warns"),
    [
        (True, None, None, "never", True),
        (True, None, "must", "must", False),
        (True, "my-run", None, "allow", False),
        (True, "my-run", "must", "must", False),
        (False, None, "allow", "allow", False),
        (False, "my-run", None, "never", False),
    ],
)
def test_trackio_logger_resume(
    trackio, tmp_path, caplog, cfg_resume, run_name, trackio_resume, expected, warns
):
    cfg = make_cfg(
        tmp_path, trackio=TrackioConfig(run_name=run_name, resume=trackio_resume), resume=cfg_resume
    )

    with caplog.at_level(logging.WARNING):
        TrackioLogger(cfg)

    kwargs = trackio.init.call_args.kwargs
    assert kwargs["resume"] == expected
    if run_name is None:
        assert re.fullmatch(r"test-run-[0-9a-f]{8}", kwargs["name"])
    else:
        assert kwargs["name"] == run_name
    assert cfg.trackio.run_name == kwargs["name"]
    assert ("no Trackio run name" in caplog.text) is warns


@pytest.mark.parametrize(
    ("space_id", "url", "expected_space_id", "expected_target"),
    [
        ("my-space", "alice/my-space", "alice/my-space", "https://huggingface.co/spaces/alice/my-space"),
        (None, "https://tracker.example.com", None, "https://tracker.example.com"),
        (None, None, None, 'trackio show --project "lerobot"'),
    ],
)
def test_trackio_logger_reports_run_location(
    trackio, tmp_path, caplog, monkeypatch, space_id, url, expected_space_id, expected_target
):
    monkeypatch.setenv("NO_COLOR", "1")
    trackio.init.return_value.url = url
    cfg = make_cfg(tmp_path, trackio=TrackioConfig(space_id=space_id))

    with caplog.at_level(logging.INFO):
        TrackioLogger(cfg)

    assert cfg.trackio.space_id == expected_space_id
    assert caplog.messages[-1] == f"Track this run --> {expected_target}"


# --- log_dict ---


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_log_dict_with_step(trackio, tmp_path, mode):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_dict({"loss": 0.5, "lr": 1e-4, "status": "ok"}, step=10, mode=mode)

    trackio.log.assert_called_once_with(
        {f"{mode}/loss": 0.5, f"{mode}/lr": 1e-4, f"{mode}/status": "ok"}, step=10
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step": 1, "mode": "test"}, "test"),
        ({}, "step or custom_step_key"),
        ({"custom_step_key": "interaction_step"}, "interaction_step"),
    ],
)
def test_log_dict_invalid_arguments_raise(trackio, tmp_path, kwargs, match):
    logger = TrackioLogger(make_cfg(tmp_path))

    with pytest.raises(ValueError, match=match):
        logger.log_dict({"loss": 0.5}, **kwargs)

    trackio.log.assert_not_called()


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"loss": 0.5, "weights": [0.1, 0.2], "obj": object()}, {"train/loss": 0.5}),
        ({"weights": [0.1, 0.2], "obj": object()}, None),
    ],
)
def test_log_dict_skips_non_scalar_values(trackio, tmp_path, caplog, metrics, expected):
    logger = TrackioLogger(make_cfg(tmp_path))

    with caplog.at_level(logging.WARNING):
        logger.log_dict(metrics, step=1)

    if expected is None:
        trackio.log.assert_not_called()
    else:
        trackio.log.assert_called_once_with(expected, step=1)
    assert '"weights"' in caplog.text
    assert '"obj"' in caplog.text


def test_log_dict_with_custom_step_key(trackio, tmp_path):
    logger = TrackioLogger(make_cfg(tmp_path))

    logger.log_dict({"loss": 0.5, "interaction_step": 7}, custom_step_key="interaction_step")

    trackio.log.assert_called_once_with({"train/loss": 0.5, "train/interaction_step": 7})


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

    trackio.Artifact.assert_called_once_with(
        "policy_test-policy-seed_42-dataset_user_repo-000100", type="model"
    )

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
print("console capture marker")
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
        configs = [json.loads(row[0]) for row in db.execute("SELECT config FROM configs").fetchall()]
    assert artifacts == [("policy_act-seed_7-dataset_user_repo-000100", "model")]
    assert shlex.split(configs[0]["_Command"])[1:] == [str(script), str(checkpoint_dir)]

    console_logs = sorted((storage_dir / "media" / "e2e" / summary["run_name"]).glob("console-*.log"))
    assert "console capture marker" in "".join(log.read_text() for log in console_logs)
