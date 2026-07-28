import json

import numpy as np
import pytest

from examples.picklift_v3.backend import RelativeRebaser, SyntheticBackend
from examples.picklift_v3.camera_profile import (
    ALIGNED_FRONT_CAMERA_PROFILE_ID,
    SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
)
from examples.picklift_v3.record import (
    FPS,
    features,
    record,
    spawn_region_for,
    validate_config,
)


def config(tmp_path):
    return {
        "mode": "synthetic",
        "dataset_root": str(tmp_path / "dataset"),
        "repo_id": "local/engineering_smoke",
        "operator_id": "engineering_smoke",
        "session_id": "smoke_session",
        "task_id": "picklift_smoke",
        "task_version": "0.0.1-engineering",
        "task": "engineering_smoke: contract check",
        "real_world_setup_version": "synthetic_v1",
        "camera_config_version": "synthetic_v1",
        "camera_profile_id": SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
        "camera_device": "synthetic",
        "camera_intrinsics_version": "n/a-v1",
        "camera_extrinsics_version": "n/a-v1",
        "robot_id": "synthetic_follower",
        "robot_calibration_id": "synthetic",
        "follower_serial_id": "synthetic",
        "leader_id": "synthetic_leader",
        "leader_calibration_id": "synthetic",
        "leader_serial_id": "synthetic",
        "spawn_id": "engineering_spawn_0000",
        "spawn_protocol_version": "picklift_spawn_v2",
        "spawn_region": "r2c2",
        "spawn_x_cm": 30,
        "spawn_y_cm": 18,
        "spawn_yaw_deg": 45,
        "result": "failure",
        "formal_data": False,
        "control_hz": 50,
        "camera_acquisition_fps": 30,
        "alignment_mode": "relative_rebase",
        "startup_hold_s": 0,
        "record_fps": FPS,
        "episode_seconds": 0.1,
        "success": False,
        "termination_reason": "engineering_smoke_complete",
        "use_videos": False,
    }


class EvidenceBackend(SyntheticBackend):
    def __init__(self):
        super().__init__()
        self.states = []
        self.sent = []

    def read_pre_action(self):
        result = super().read_pre_action()
        self.states.append(result[0].copy())
        return result

    def send_action(self, action):
        actual = (action - 0.25).astype(np.float32)
        self.sent.append(actual.copy())
        self.index += 1
        return actual


def test_v3_contract_pre_action_and_actual_sent(tmp_path):
    cfg = config(tmp_path)
    backend = EvidenceBackend()
    root = record(cfg, backend)
    from lerobot.datasets import LeRobotDataset

    ds = LeRobotDataset(cfg["repo_id"], root=root)
    assert ds.meta.fps == 20
    assert len(ds) == 2
    state = ds[0]["observation.state"].numpy()
    action = ds[0]["action"].numpy()
    assert any(np.array_equal(state, item) for item in backend.states)
    assert any(np.array_equal(action, item) for item in backend.sent)
    np.testing.assert_array_equal(action, np.arange(6, dtype=np.float32) + 0.25)
    assert tuple(ds[0]["observation.images.front"].shape) == (3, 480, 640)
    assert set(features(False)) == {
        "observation.state",
        "action",
        "observation.images.front",
    }


def test_required_provenance(tmp_path):
    cfg = config(tmp_path)
    root = record(cfg)
    provenance = json.loads((root / "provenance/episodes/episode_000000.json").read_text())
    session = json.loads((root / "provenance/session.json").read_text())
    for key in (
        "operator_id",
        "session_id",
        "task_id",
        "task_version",
        "real_world_setup_version",
        "backend",
        "control_mode",
        "collection_commit",
        "lerobot_version",
        "robot_calibration_id",
        "follower_serial_id",
        "camera_config_version",
        "camera_profile_id",
        "camera_profile",
        "record_fps",
        "start_time",
        "end_time",
        "termination_reason",
        "success",
        "dropped_frames",
        "sync_anomalies",
        "spawn_protocol_version",
        "spawn_id",
        "spawn_region",
        "spawn_x_cm",
        "spawn_y_cm",
        "spawn_yaw_deg",
        "result",
    ):
        assert key in provenance
    assert session["camera_profile"] == provenance["camera_profile"]
    assert session["spawn_protocol_version"] == "picklift_spawn_v2"
    assert provenance["formal_data"] is False


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"record_fps": 30}, "exactly 20"),
        ({"task_version": ""}, "missing explicit"),
        ({"operator_id": "person@example.com"}, "pseudonymous"),
        ({"control_hz": 10}, "control_hz"),
    ],
)
def test_fail_closed_config(tmp_path, mutation, error):
    cfg = config(tmp_path)
    cfg.update(mutation)
    with pytest.raises((ValueError, PermissionError), match=error):
        validate_config(cfg)


def test_real_mode_requires_powered_ack(tmp_path):
    cfg = config(tmp_path)
    cfg["mode"] = "real"
    cfg["camera_profile_id"] = ALIGNED_FRONT_CAMERA_PROFILE_ID
    cfg["camera_config_version"] = ALIGNED_FRONT_CAMERA_PROFILE_ID
    with pytest.raises(PermissionError, match="powered safety check"):
        validate_config(cfg)


def test_real_mode_rejects_camera_profile_version_mismatch(tmp_path):
    cfg = config(tmp_path)
    cfg["mode"] = "real"
    cfg["camera_profile_id"] = ALIGNED_FRONT_CAMERA_PROFILE_ID
    with pytest.raises(ValueError, match="camera_config_version"):
        validate_config(cfg)


def test_relative_rebase_is_zero_jump_then_tracks_delta():
    leader = np.asarray([-5, -105, 97, 74, 2, 0], dtype=np.float32)
    follower = np.asarray([3, -104, 70, 87, 1, 11], dtype=np.float32)
    rebaser = RelativeRebaser()
    np.testing.assert_array_equal(rebaser.initialize(leader, follower), follower)
    delta = np.asarray([1, -2, 3, -4, 5, -6], dtype=np.float32)
    np.testing.assert_array_equal(rebaser.apply(leader + delta), follower + delta)


def test_relative_rebase_fails_closed_before_initialization():
    with pytest.raises(RuntimeError, match="before initialization"):
        RelativeRebaser().apply(np.zeros(6, dtype=np.float32))


def test_direct_absolute_config_is_allowed(tmp_path):
    cfg = config(tmp_path)
    cfg["alignment_mode"] = "direct_absolute"
    validate_config(cfg)


@pytest.mark.parametrize(
    ("x_cm", "y_cm", "region"),
    [
        (20, 10, "r1c1"),
        (30, 18, "r2c2"),
        (40, 25, "r3c3"),
    ],
)
def test_spawn_region_mapping(x_cm, y_cm, region):
    assert spawn_region_for(x_cm, y_cm) == region


def test_legacy_spawn_v1_mapping_remains_available():
    assert spawn_region_for(20, 15, "picklift_spawn_v1") == "r1c1"
    assert spawn_region_for(30, 20, "picklift_spawn_v1") == "r2c2"
    assert spawn_region_for(40, 25, "picklift_spawn_v1") == "r3c3"


def test_spawn_v2_balanced_grid_covers_all_nine_regions():
    regions = {spawn_region_for(x_cm, y_cm) for y_cm in (12, 17, 22) for x_cm in (23, 30, 37)}
    assert regions == {f"r{row}c{column}" for row in range(1, 4) for column in range(1, 4)}
    assert spawn_region_for(30, 15) == "r2c2"
    assert spawn_region_for(30, 20) == "r3c2"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"spawn_x_cm": 19}, "20..40"),
        ({"spawn_y_cm": 9}, "10..25"),
        ({"spawn_protocol_version": "picklift_spawn_unknown"}, "unsupported"),
        ({"spawn_yaw_deg": 91}, "0..90"),
        ({"spawn_region": "r1c1"}, "does not match"),
        ({"result": "success", "success": False}, "success must be true"),
        ({"result": "pending"}, "requires operator_ui"),
    ],
)
def test_spawn_protocol_fails_closed(tmp_path, mutation, error):
    cfg = config(tmp_path)
    cfg.update(mutation)
    with pytest.raises(ValueError, match=error):
        validate_config(cfg)
