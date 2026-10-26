"""Tests for video backend availability checks and fallback behavior."""

import logging
from unittest import mock

import pytest

from lerobot.utils.import_utils import get_safe_default_video_backend, is_video_backend_available


def test_explicit_backend_available_passes():
    assert is_video_backend_available("pyav") is True


def test_torchcodec_broken_native_lib_reports_unavailable(monkeypatch):
    """Simulate the half-broken state: import raises OSError (native lib failed)."""
    import importlib
    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "torchcodec":
            raise OSError("Could not load this library: libtorchcodec_core.so")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    assert is_video_backend_available("torchcodec") is False


def test_default_backend_falls_back_when_torchcodec_broken(monkeypatch, caplog):
    import importlib
    import importlib.util
    real_import_module = importlib.import_module
    real_find_spec = importlib.util.find_spec

    def fake_import_module(name, *args, **kwargs):
        if name == "torchcodec":
            raise OSError("Could not load this library: libtorchcodec_core.so")
        return real_import_module(name, *args, **kwargs)

    def fake_find_spec(name, *args, **kwargs):
        if name == "torchcodec":
            return mock.Mock()  # pretend installed
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with caplog.at_level(logging.WARNING):
        backend = get_safe_default_video_backend()
    assert backend == "pyav"
    assert any("Falling back to 'pyav'" in r.message for r in caplog.records)


def test_unknown_backend_name_defaults_to_available():
    # Unknown names are not our business here; do not raise.
    assert is_video_backend_available("some_other") is True
