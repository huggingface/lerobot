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

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from lerobot.common import train_utils as train_utils_module
from lerobot.common.train_utils import (
    _is_windows_symlink_privilege_error,
    _link_latest_checkpoint,
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


def _make_winerror(code: int, message: str) -> OSError:
    exc = OSError(message)
    exc.winerror = code
    return exc


def test_is_windows_symlink_privilege_error_true_on_nt_with_1314(monkeypatch):
    monkeypatch.setattr(train_utils_module.os, "name", "nt")
    assert _is_windows_symlink_privilege_error(_make_winerror(1314, "privilege not held"))


def test_is_windows_symlink_privilege_error_false_for_other_winerror(monkeypatch):
    monkeypatch.setattr(train_utils_module.os, "name", "nt")
    assert not _is_windows_symlink_privilege_error(_make_winerror(183, "already exists"))


def test_is_windows_symlink_privilege_error_false_on_non_windows(monkeypatch):
    monkeypatch.setattr(train_utils_module.os, "name", "posix")
    assert not _is_windows_symlink_privilege_error(_make_winerror(1314, "privilege not held"))


def _assert_last_points_to(last: Path, checkpoint: Path) -> None:
    assert last.exists()
    assert last.is_dir()
    assert last.resolve() == checkpoint.resolve()
    assert os.path.samefile(last, checkpoint)


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
    _assert_last_points_to(last_checkpoint, checkpoint)
    if os.name != "nt" or last_checkpoint.is_symlink():
        assert last_checkpoint.is_symlink()


def test_link_latest_checkpoint_falls_back_to_junction_on_winerror_1314(tmp_path, monkeypatch):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    link = tmp_path / "last"
    junction_mock = MagicMock()

    def symlink_side_effect(self, target, target_is_directory=False):
        raise _make_winerror(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", symlink_side_effect)
    monkeypatch.setattr("lerobot.common.train_utils._create_directory_junction", junction_mock)
    monkeypatch.setattr(
        "lerobot.common.train_utils._is_windows_symlink_privilege_error",
        lambda exc: getattr(exc, "winerror", None) == 1314,
    )

    _link_latest_checkpoint(link, checkpoint)

    junction_mock.assert_called_once_with(link, checkpoint)


def test_link_latest_checkpoint_reraises_non_privilege_oserror(tmp_path, monkeypatch):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    link = tmp_path / "last"
    junction_mock = MagicMock()
    original_error = _make_winerror(183, "Cannot create a file when that file already exists")

    def symlink_side_effect(self, target, target_is_directory=False):
        raise original_error

    monkeypatch.setattr(Path, "symlink_to", symlink_side_effect)
    monkeypatch.setattr("lerobot.common.train_utils._create_directory_junction", junction_mock)
    monkeypatch.setattr("lerobot.common.train_utils._is_windows_symlink_privilege_error", lambda exc: False)

    with pytest.raises(OSError) as exc_info:
        _link_latest_checkpoint(link, checkpoint)
    assert exc_info.value is original_error
    junction_mock.assert_not_called()


def test_link_latest_checkpoint_non_privilege_error_does_not_call_junction(tmp_path, monkeypatch):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    link = tmp_path / "last"
    junction_mock = MagicMock()

    def symlink_side_effect(self, target, target_is_directory=False):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "symlink_to", symlink_side_effect)
    monkeypatch.setattr("lerobot.common.train_utils._create_directory_junction", junction_mock)
    monkeypatch.setattr("lerobot.common.train_utils._is_windows_symlink_privilege_error", lambda exc: False)

    with pytest.raises(OSError, match="permission denied"):
        _link_latest_checkpoint(link, checkpoint)
    junction_mock.assert_not_called()


def test_update_last_checkpoint_replaces_existing_pointer(tmp_path):
    ckpt1 = tmp_path / "0005"
    ckpt2 = tmp_path / "0010"
    ckpt1.mkdir()
    ckpt2.mkdir()
    update_last_checkpoint(ckpt1)
    update_last_checkpoint(ckpt2)
    _assert_last_points_to(tmp_path / LAST_CHECKPOINT_LINK, ckpt2)


def test_update_last_checkpoint_readable_via_last_pointer(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last = tmp_path / LAST_CHECKPOINT_LINK
    save_training_step(42, last)
    assert load_training_step(last) == 42


def test_update_last_checkpoint_does_not_copy_checkpoint_dir(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    marker = checkpoint / "marker.txt"
    marker.write_text("original")
    update_last_checkpoint(checkpoint)
    last = tmp_path / LAST_CHECKPOINT_LINK
    _assert_last_points_to(last, checkpoint)
    assert (last / "marker.txt").read_text() == "original"
    (last / "marker.txt").write_text("updated")
    assert marker.read_text() == "updated"
    assert {path.name for path in tmp_path.iterdir()} == {checkpoint.name, LAST_CHECKPOINT_LINK}


def test_update_last_checkpoint_path_with_spaces(tmp_path):
    checkpoints_dir = tmp_path / "my checkpoints"
    checkpoints_dir.mkdir()
    checkpoint = checkpoints_dir / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    _assert_last_points_to(checkpoints_dir / LAST_CHECKPOINT_LINK, checkpoint)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction integration")
def test_update_last_checkpoint_real_junction_without_symlink_privilege(tmp_path, monkeypatch):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()

    def symlink_side_effect(self, target, target_is_directory=False):
        raise _make_winerror(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(Path, "symlink_to", symlink_side_effect)

    update_last_checkpoint(checkpoint)
    last = tmp_path / LAST_CHECKPOINT_LINK
    _assert_last_points_to(last, checkpoint)
    assert not last.is_symlink()
    assert (last / "..").resolve() == tmp_path.resolve()
    save_training_step(42, last)
    assert load_training_step(last) == 42


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
    _assert_last_points_to(last, checkpoint_dir)
    if last.is_symlink():
        assert (last.parent / last.readlink()).resolve() == checkpoint_dir.resolve()


def test_resolve_resume_checkpoint_raises_without_checkpoints(tmp_path, monkeypatch):
    from lerobot.common import train_utils

    monkeypatch.setattr("lerobot.common.train_utils.find_latest_hub_checkpoint", lambda repo_id: None)
    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        train_utils.resolve_resume_checkpoint("u/run", tmp_path / "run")
