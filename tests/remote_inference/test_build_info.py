# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Loaded-build diagnostics stay stable and never claim an unrelated revision."""

import subprocess
from dataclasses import FrozenInstanceError, asdict

import pytest

from lerobot import __version__
from lerobot.remote_inference import build_info
from lerobot.remote_inference.server import SessionWorker
from tests.inference.test_policy_runner import ConformingPolicy, runner_for, tiny_config


def test_source_revision_is_available_and_snapshot_survives_later_checkout_changes(tmp_path):
    def git(*arguments):
        return subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "one",
    )
    first_revision = git("rev-parse", "HEAD")
    loaded = build_info._capture_build(tmp_path)
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "two",
    )
    assert git("rev-parse", "HEAD") != first_revision
    assert loaded == build_info.SoftwareBuild(__version__, first_revision, False)
    (tmp_path / "changed.py").write_text("# source changed after process start\n")
    assert build_info._capture_build(tmp_path).dirty is True
    assert loaded.dirty is False
    with pytest.raises(FrozenInstanceError):
        loaded.revision = "new-checkout"


def test_wheel_or_archive_does_not_use_git_from_unrelated_working_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("looked up unrelated checkout"))
    assert build_info._capture_build(tmp_path) == build_info.SoftwareBuild(__version__)


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("git", 1)])
def test_revision_lookup_failure_is_honestly_unavailable(tmp_path, monkeypatch, error):
    (tmp_path / ".git").mkdir()

    def unavailable(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", unavailable)
    assert build_info._capture_build(tmp_path) == build_info.SoftwareBuild(__version__)


def test_serving_descriptor_uses_the_imported_build_without_rereading_checkout(monkeypatch):
    monkeypatch.setattr(
        build_info, "_capture_build", lambda *a: pytest.fail("reread the build after process startup")
    )
    worker = SessionWorker(
        runner_for(ConformingPolicy(tiny_config())),
        deployment="test",
        artifact_identity="artifact",
        semantics="radians-v1",
    )
    try:
        first = worker.descriptor["software"]
        assert first == asdict(build_info.SOFTWARE_BUILD)
        # A caller mutating a descriptor must not change the process snapshot.
        first["revision"] = "changed"
        assert worker.descriptor["software"] == asdict(build_info.SOFTWARE_BUILD)
    finally:
        worker.close()
