"""A failed publication preserves the prior target; EMA recovery preserves its declared state."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.common import train_utils
from lerobot.scripts.lerobot_train import _require_saved_ema_for_resume
from lerobot.utils.io_utils import write_json


def test_failed_replace_keeps_previous_checkpoint(tmp_path, monkeypatch):
    old, new = tmp_path / "000002", tmp_path / "000004"
    old.mkdir()
    new.mkdir()
    train_utils.update_last_checkpoint(old)

    def interrupt(source, destination):
        assert Path(source).is_symlink()
        assert Path(destination).resolve() == old
        raise OSError("interrupted publication")

    monkeypatch.setattr(train_utils.os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted publication"):
        train_utils.update_last_checkpoint(new)
    assert (tmp_path / "last").resolve() == old
    assert not list(tmp_path.glob(".last-*"))


def test_replacement_is_relative_and_survives_moving_tree(tmp_path):
    root = tmp_path / "run"
    new = root / "000004"
    new.mkdir(parents=True)
    train_utils.update_last_checkpoint(new)
    assert os.readlink(root / "last") == "000004"
    moved = tmp_path / "moved"
    root.rename(moved)
    assert (moved / "last").resolve() == moved / "000004"


@pytest.mark.parametrize("saved_enabled", [True, False, None])
@pytest.mark.parametrize("requested_enabled", [True, False])
def test_missing_ema_depends_on_saved_and_requested_training(cfg_path, saved_enabled, requested_enabled):
    saved = {} if saved_enabled is None else {"ema": {"enable": saved_enabled}}
    write_json(saved, cfg_path / "pretrained_model/train_config.json")
    cfg = SimpleNamespace(checkpoint_path=cfg_path, ema=SimpleNamespace(enable=requested_enabled))
    if saved_enabled and requested_enabled:
        with pytest.raises(FileNotFoundError, match="saved EMA training state"):
            _require_saved_ema_for_resume(cfg)
    else:
        _require_saved_ema_for_resume(cfg)


@pytest.fixture
def cfg_path(tmp_path):
    (tmp_path / "pretrained_model").mkdir()
    return tmp_path


def test_present_ema_is_left_to_native_deserializer(cfg_path):
    (cfg_path / "training_state").mkdir()
    (cfg_path / "training_state/ema_state.pt").write_bytes(b"native loader validates content")
    cfg = SimpleNamespace(checkpoint_path=cfg_path, ema=SimpleNamespace(enable=True))
    _require_saved_ema_for_resume(cfg)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_non_symlink_last_is_not_replaced(tmp_path, kind):
    checkpoint = tmp_path / "000004"
    checkpoint.mkdir()
    target = tmp_path / "last"
    if kind == "file":
        target.write_text("existing content")
    else:
        target.mkdir()
        (target / "keep").write_text("existing content")
    with pytest.raises(FileExistsError):
        train_utils.update_last_checkpoint(checkpoint)
    assert not target.is_symlink()
    preserved = target if kind == "file" else target / "keep"
    assert preserved.read_text() == "existing content"
