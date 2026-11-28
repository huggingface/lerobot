"""Tests for video backend availability checks and fallback behavior."""

import importlib
import importlib.util
import logging
from unittest import mock

import pytest

from lerobot.utils.import_utils import get_safe_default_video_backend, is_video_backend_available

OSERROR = OSError("Could not load this library: libtorchcodec_core.so")


def _mock_import(monkeypatch, *, fail_modules=(), stub_modules=()):
    """Patch importlib.import_module: names in fail_modules raise OSError,
    names in stub_modules import to a stub; everything else passes through."""
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name in fail_modules:
            raise OSERROR
        if name in stub_modules:
            return mock.Mock()
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


def test_explicit_backend_available_passes(monkeypatch):
    # Mock the successful import so the base CI tier (no dataset extra, no
    # real `av`) exercises this path instead of skipping or failing on it.
    _mock_import(monkeypatch, stub_modules=("av",))
    assert is_video_backend_available("pyav") is True


def test_pyav_unavailable_returns_false(monkeypatch):
    _mock_import(monkeypatch, fail_modules=("av",))
    assert is_video_backend_available("pyav") is False


def test_video_reader_alias_needs_pyav(monkeypatch):
    # "video_reader" is a deprecated alias of pyav: without `av` it must not
    # fall through to available, or old configs keep the late failure this
    # fail-fast work exists to remove.
    _mock_import(monkeypatch, fail_modules=("av",))
    assert is_video_backend_available("video_reader") is False


def test_torchcodec_broken_native_lib_reports_unavailable(monkeypatch):
    """Simulate the half-broken state: import raises OSError (native lib failed)."""
    _mock_import(monkeypatch, fail_modules=("torchcodec", "torchcodec._core"))
    assert is_video_backend_available("torchcodec") is False


def test_default_backend_falls_back_when_torchcodec_broken(monkeypatch, caplog):
    _mock_import(monkeypatch, fail_modules=("torchcodec", "torchcodec._core"))
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "torchcodec":
            return mock.Mock()  # pretend installed
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with caplog.at_level(logging.WARNING):
        backend = get_safe_default_video_backend()
    assert backend == "pyav"
    assert any("Falling back to 'pyav'" in r.message for r in caplog.records)


def test_unknown_backend_name_defaults_to_available():
    # Unknown names are not our business here; do not raise.
    assert is_video_backend_available("some_other") is True


def test_dataset_fails_fast_on_unloadable_explicit_backend(monkeypatch, tmp_path):
    # Constructor regression: an explicitly requested but unloadable backend
    # must raise in LeRobotDataset.__init__, before any metadata resolution,
    # Hub traffic, or local-root creation.
    from lerobot.datasets import LeRobotDataset

    _mock_import(monkeypatch, fail_modules=("av",))
    with pytest.raises(ValueError, match="cannot be loaded"):
        LeRobotDataset("unit/test", video_backend="pyav", root=tmp_path)


@pytest.mark.parametrize("backend", ["pyav", "video_reader"])
def test_streaming_dataset_fails_fast_on_unloadable_backend(monkeypatch, backend):
    # Same contract for StreamingLeRobotDataset: the check fires in __init__,
    # before any Hub access — not at the first decoded frame.
    from lerobot.datasets import StreamingLeRobotDataset

    _mock_import(monkeypatch, fail_modules=("av",))
    with pytest.raises(ValueError, match="cannot be loaded"):
        StreamingLeRobotDataset("unit/test", video_backend=backend)
