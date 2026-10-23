"""Retain native process stderr hidden by per-request Python stream redirection."""

import ast
from pathlib import Path
from types import SimpleNamespace

import lerobot.annotations.hands
from lerobot.annotations.hands.bridge import native_failure_detail


def test_hand_modules_parse_on_native_gpu_python310():
    """The controller's newer Python must not hide native worker syntax errors."""
    root = Path(lerobot.annotations.hands.__file__).parent
    for source in root.rglob("*.py"):
        ast.parse(source.read_text(), filename=str(source), feature_version=(3, 10))


def test_cuda_error_from_camera_service_log_is_in_failure(tmp_path):
    camera_log = tmp_path / "back.log"
    camera_log.write_text("TensorRT Error: native execution context could not load its CUDA library")
    hand_log = tmp_path / "hand.log"
    hand_log.write_text("Hand startup complete")
    workers = SimpleNamespace(_WORKERS={"back": SimpleNamespace(log=camera_log)})
    error = native_failure_detail(RuntimeError("DROID-W failed"), workers, SimpleNamespace(log_path=hand_log))
    assert "DROID-W failed" in error
    assert "TensorRT Error: native execution context" in error
    assert "Hand startup complete" in error


def test_native_diagnostics_are_bounded_and_tolerate_absent_logs(tmp_path):
    camera_log = tmp_path / "back.log"
    camera_log.write_text("old messages\n" + "x" * 7000 + "\nlast native failure")
    workers = SimpleNamespace(_WORKERS={"back": SimpleNamespace(log=camera_log)})
    error = native_failure_detail(
        RuntimeError("failed"), workers, SimpleNamespace(log_path=tmp_path / "absent")
    )
    assert "old messages" not in error
    assert error.endswith("last native failure")
    assert len(error) < 6100


def test_failed_camera_worker_keeps_native_log_after_registry_removal(tmp_path):
    camera_log = tmp_path / "back.log"
    camera_log.write_text("TensorRT context failure from removed native worker")
    workers = SimpleNamespace(_WORKERS={}, _STATE=SimpleNamespace(name=str(tmp_path)))
    error = native_failure_detail(
        RuntimeError("DROID failed"), workers, SimpleNamespace(log_path=tmp_path / "hand.log")
    )
    assert "camera service back native stderr" in error
    assert "TensorRT context failure from removed native worker" in error
