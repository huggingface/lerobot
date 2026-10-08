# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""The sidecar builder is a packaged, explicitly publishing command."""

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

from lerobot.utils import import_utils


def test_sidecar_command_is_registered_in_package() -> None:
    """Installed distributions must expose the builder, not a checkout-only script."""
    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    assert config["project"]["scripts"].get("lerobot-build-mp4-sidecar") == (
        "lerobot.scripts.lerobot_build_mp4_sidecar:main"
    )
    assert importlib.util.find_spec("lerobot.scripts.lerobot_build_mp4_sidecar") is not None


def test_sidecar_command_help(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Help is available without loading a dataset or starting a build."""
    module = importlib.import_module("lerobot.scripts.lerobot_build_mp4_sidecar")
    monkeypatch.setattr(sys, "argv", ["lerobot-build-mp4-sidecar", "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--repo-id" in help_text and "--push" in help_text


def test_sidecar_command_help_without_dataset_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The base installation can display help before requesting optional dataset dependencies."""
    monkeypatch.setattr(import_utils, "_datasets_available", False)
    spec = importlib.util.find_spec("lerobot.scripts.lerobot_build_mp4_sidecar")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not hasattr(module, "LeRobotDatasetMetadata")
    monkeypatch.setattr(sys, "argv", ["lerobot-build-mp4-sidecar", "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert "--repo-id" in capsys.readouterr().out
