#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_batch_size,
    load_training_num_processes,
    load_training_state,
    load_training_step,
    push_checkpoint_to_hub,
    save_checkpoint,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_save_training_state_records_num_processes(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler, num_processes=4)
    assert load_training_num_processes(tmp_path) == 4


def test_load_training_num_processes_absent_returns_none(tmp_path, optimizer, scheduler):
    # Checkpoints written before the world size was recorded must still load (back-compat).
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert load_training_num_processes(tmp_path) is None


def test_save_training_state_records_batch_size(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler, batch_size=32)
    assert load_training_batch_size(tmp_path) == 32


def test_load_training_batch_size_absent_returns_none(tmp_path, optimizer, scheduler):
    # Checkpoints written before the batch size was recorded must still load (back-compat).
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert load_training_batch_size(tmp_path) is None


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


@patch("lerobot.common.train_utils.save_training_state")
def test_save_checkpoint(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


@patch("lerobot.common.train_utils.save_training_state")
def test_save_checkpoint_peft(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    policy.config = Mock()
    policy.config.save_pretrained = Mock()
    cfg = Mock()
    cfg.use_peft = True
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    policy.config.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


def test_save_load_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler


def test_load_training_state_skip_optimizer(tmp_path, optimizer, scheduler):
    # FSDP loads optimizer separately (after accelerator.prepare)
    # load_training_state(load_optimizer=False) must restore step + scheduler but leave the
    # optimizer untouched and never touch the on-disk optimizer state.
    save_training_state(tmp_path, 10, optimizer, scheduler)
    with patch("lerobot.common.train_utils.load_optimizer_state") as mock_load_optimizer_state:
        loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(
            tmp_path, optimizer, scheduler, load_optimizer=False
        )
    mock_load_optimizer_state.assert_not_called()
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler


def test_push_checkpoint_to_hub_creates_repo_and_uploads(tmp_path, monkeypatch):
    ckpt = tmp_path / "010000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    api = MagicMock()
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", lambda *a, **k: api)
    push_checkpoint_to_hub(ckpt, "user/run", private=True)
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is True
    assert api.create_repo.call_args.kwargs["repo_type"] == "model"
    api.upload_folder.assert_called_once()
    kwargs = api.upload_folder.call_args.kwargs
    assert kwargs["repo_id"] == "user/run"
    assert kwargs["repo_type"] == "model"
    assert kwargs["path_in_repo"] == "checkpoints/010000"
    assert kwargs["folder_path"] == str(ckpt)
    assert kwargs["commit_message"] == "checkpoint 010000"
    # A tag named after the checkpoint step is created so the checkpoint can be
    # recovered with --policy.pretrained_revision instead of a commit sha.
    api.create_tag.assert_called_once()
    tag_kwargs = api.create_tag.call_args.kwargs
    assert tag_kwargs["tag"] == "010000"
    assert tag_kwargs["revision"] == api.upload_folder.return_value.oid
    assert tag_kwargs["repo_type"] == "model"
    assert tag_kwargs["exist_ok"] is True


def test_push_checkpoint_to_hub_defaults_to_hub_default_visibility(tmp_path, monkeypatch):
    ckpt = tmp_path / "010000"
    (ckpt / "pretrained_model").mkdir(parents=True)
    api = MagicMock()
    monkeypatch.setattr("lerobot.common.train_utils.HfApi", lambda *a, **k: api)
    push_checkpoint_to_hub(ckpt, "user/run")
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is None


def test_resolve_resume_checkpoint_downloads_latest_and_links(tmp_path, monkeypatch):
    from lerobot.common import train_utils

    out = tmp_path / "run"

    def fake_snapshot_download(repo_id, repo_type, allow_patterns, local_dir):
        # Mimic the Hub layout the real download materializes locally.
        assert allow_patterns == "checkpoints/020000/*"
        (Path(local_dir) / "checkpoints" / "020000" / "pretrained_model").mkdir(parents=True)
        return local_dir

    monkeypatch.setattr("lerobot.common.train_utils.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        "lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: "checkpoints/020000"
    )

    checkpoint_dir = train_utils.resolve_resume_checkpoint("u/run", out)

    assert checkpoint_dir == out / CHECKPOINTS_DIR / "020000"
    last = out / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    assert last.is_symlink()
    # `last` points at the downloaded step dir.
    assert (last.parent / last.readlink()).resolve() == checkpoint_dir.resolve()


def test_resolve_resume_checkpoint_raises_without_checkpoints(tmp_path, monkeypatch):
    from lerobot.common import train_utils

    monkeypatch.setattr("lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: None)
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        train_utils.resolve_resume_checkpoint("u/run", tmp_path / "run")
