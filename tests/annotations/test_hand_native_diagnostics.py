"""Retain native process stderr hidden by per-request Python stream redirection."""

from types import SimpleNamespace

from lerobot.annotations.hands.bridge import native_failure_detail


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
