from examples.picklift_v3.practice import run_practice, validate_practice_config
from examples.picklift_v3.record import SyntheticBackend


class FakeUI:
    def __init__(self):
        self.calls = 0
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def show(self, *_args, **_kwargs):
        self.calls += 1
        return "stop" if self.calls == 3 else None

    def close(self):
        self.closed = True


def synthetic_config():
    return {
        "mode": "synthetic",
        "camera_device": "synthetic",
        "camera_acquisition_fps": 30,
        "robot_id": "synthetic",
        "follower_port": "synthetic",
        "leader_id": "synthetic",
        "leader_port": "synthetic",
        "control_hz": 50,
        "alignment_mode": "direct_absolute",
        "startup_hold_s": 0,
    }


def test_practice_stops_without_recording_data(tmp_path):
    cfg = synthetic_config()
    ui = FakeUI()
    result = run_practice(cfg, backend=SyntheticBackend(), ui=ui)
    assert result["frames"] == 3
    assert result["data_recorded"] is False
    assert ui.opened and ui.closed
    assert list(tmp_path.iterdir()) == []


def test_synthetic_practice_config_is_valid():
    validate_practice_config(synthetic_config())
