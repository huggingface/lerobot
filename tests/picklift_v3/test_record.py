import json
from pathlib import Path

import numpy as np
import pytest

from examples.picklift_v3.alignment_reference import (
    ALIGNMENT_REFERENCE_V2_ID,
    alignment_reference,
)
from examples.picklift_v3.backend import RelativeRebaser, SyntheticBackend
from examples.picklift_v3.camera_profile import (
    ALIGNED_FRONT_CAMERA_PROFILE_ID,
    SYNTHETIC_FRONT_CAMERA_PROFILE_ID,
)
from examples.picklift_v3.record import (
    FPS,
    features,
    record,
    spawn_contract,
    spawn_region_for,
    spawn_ui_summary,
    validate_config,
)
from examples.picklift_v3.task_frame import (
    TASK_GRID_FRAME_ID,
    TASK_GRID_FRAME_V1_ID,
    task_frame,
)


def config(tmp_path):
    return {
        "mode": "synthetic",
        "dataset_root": str(tmp_path / "dataset"),
        "repo_id": "local/engineering_smoke",
        "operator_id": "engineering_smoke",
        "session_id": "smoke_session",
        "task_id": "picklift_smoke",
        "task_version": "0.0.2-engineering-unmeasured-yaw",
        "task": "engineering_smoke: v5 unmeasured-yaw contract check",
        "task_spec_revision": "picklift_taskspec_v2_unmeasured_yaw",
        "task_frame_id": TASK_GRID_FRAME_ID,
        "alignment_reference_id": ALIGNMENT_REFERENCE_V2_ID,
        "real_world_setup_version": "synthetic_spawn_v4_5cm_grid_v1",
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
        "collection_protocol_version": "picklift_collection_v5",
        "spawn_protocol_version": "picklift_spawn_v5",
        "spawn_region": "r1c1",
        "spawn_x_cm": 22.5,
        "spawn_y_cm": -7.5,
        "spawn_yaw_deg": None,
        "yaw_annotation_mode": "unmeasured_random",
        "yaw_intended_range_deg": [0, 90],
        "yaw_sampling_method": "operator_unmeasured_arbitrary",
        "yaw_distribution_claim": "unknown",
        "yaw_randomization_confirmed": True,
        "success_annotation_source": "operator_visual_v1",
        "success_detection_mode": "manual_proxy_for_nexus_v1",
        "lift_height_m": None,
        "is_grasped": None,
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


V5_ONLY_FIELDS = (
    "collection_protocol_version",
    "task_spec_revision",
    "yaw_annotation_mode",
    "yaw_intended_range_deg",
    "yaw_sampling_method",
    "yaw_distribution_claim",
    "yaw_randomization_confirmed",
    "success_annotation_source",
    "success_detection_mode",
    "lift_height_m",
    "is_grasped",
)


def make_legacy_config(cfg: dict) -> dict:
    for key in V5_ONLY_FIELDS:
        cfg.pop(key, None)
    cfg["spawn_yaw_deg"] = 45
    return cfg


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


def test_training_contract_pre_action_and_actual_sent(tmp_path):
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
        "task_frame_id",
        "task_frame",
        "alignment_reference_id",
        "alignment_reference",
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
        "spawn_contract",
        "spawn_id",
        "spawn_region",
        "spawn_x_cm",
        "spawn_y_cm",
        "spawn_yaw_deg",
        "collection_protocol_version",
        "task_spec_revision",
        "yaw_annotation_mode",
        "yaw_intended_range_deg",
        "yaw_sampling_method",
        "yaw_distribution_claim",
        "yaw_randomization_confirmed",
        "success_annotation_source",
        "success_detection_mode",
        "lift_height_m",
        "is_grasped",
        "success_contract",
        "result",
    ):
        assert key in provenance
    assert session["camera_profile"] == provenance["camera_profile"]
    assert session["spawn_protocol_version"] == "picklift_spawn_v5"
    assert session["spawn_contract"] == {
        "protocol_version": "picklift_spawn_v5",
        "x_cm": {
            "min": 20.0,
            "max": 35.0,
            "description": "task-grid +X forward",
        },
        "y_cm": {
            "min": -10.0,
            "max": 10.0,
            "description": "task-grid +Y lateral",
        },
        "region_rows_increase_along": "x",
        "region_columns_increase_along": "y",
        "cell_size_cm": 5.0,
        "cell_edges_cm": {
            "x": [20.0, 25.0, 30.0, 35.0],
            "y": [-10.0, -5.0, 0.0, 5.0, 10.0],
        },
        "grid_shape": {"rows": 3, "columns": 4, "cells": 12},
    }
    assert session["task_frame"] == task_frame(TASK_GRID_FRAME_ID)
    assert session["alignment_reference"] == alignment_reference(ALIGNMENT_REFERENCE_V2_ID)
    assert session["alignment_reference"]["red_cube_center_m"] == {
        "x_forward": 0.25,
        "y_lateral": 0.0,
    }
    assert session["alignment_reference"]["physical_confirmation_status"] == ("pending_new_25cm_screenshot")
    assert session["collection_protocol_version"] == "picklift_collection_v5"
    assert session["task_spec_revision"] == "picklift_taskspec_v2_unmeasured_yaw"
    assert session["spawn_yaw_deg"] is None
    assert session["yaw_annotation_mode"] == "unmeasured_random"
    assert session["yaw_intended_range_deg"] == [0, 90]
    assert session["yaw_sampling_method"] == "operator_unmeasured_arbitrary"
    assert session["yaw_distribution_claim"] == "unknown"
    assert session["yaw_randomization_confirmed"] is True
    assert session["success_annotation_source"] == "operator_visual_v1"
    assert session["success_detection_mode"] == "manual_proxy_for_nexus_v1"
    assert session["lift_height_m"] is None
    assert session["is_grasped"] is None
    assert session["success_contract"] == {
        "criteria_version": "picklift_manual_success_v1",
        "automatic_detection_available": False,
        "minimum_visual_lift_height_m": 0.05,
        "recommended_target_lift_range_m": [0.06, 0.08],
        "requires_visible_bilateral_finger_grasp": True,
        "minimum_stable_hold_s": 0.5,
        "requires_success_pose_held_at_end": True,
        "annotation_timing": "operator_selects_result_after_manual_end",
        "failure_definition": "task_success_criteria_not_met",
        "discard_definition": "recording_configuration_or_safety_anomaly",
    }
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
        (20, -10, "r1c1"),
        (25, -5, "r2c2"),
        (30, 0, "r3c3"),
        (35, 10, "r3c4"),
    ],
)
def test_spawn_v5_exact_coarse_grid_boundaries(x_cm, y_cm, region):
    assert spawn_region_for(x_cm, y_cm) == region


def test_legacy_spawn_v1_mapping_remains_available():
    assert spawn_region_for(20, 15, "picklift_spawn_v1") == "r1c1"
    assert spawn_region_for(30, 20, "picklift_spawn_v1") == "r2c2"
    assert spawn_region_for(40, 25, "picklift_spawn_v1") == "r3c3"


def test_legacy_spawn_v2_mapping_remains_available():
    assert spawn_region_for(10, -10, "picklift_spawn_v2") == "r1c1"
    assert spawn_region_for(15, 0, "picklift_spawn_v2") == "r2c2"
    assert spawn_region_for(25, 10, "picklift_spawn_v2") == "r3c3"


def test_legacy_v2_config_infers_frozen_alignment_reference(tmp_path):
    cfg = make_legacy_config(config(tmp_path))
    cfg.update(
        {
            "task_frame_id": TASK_GRID_FRAME_V1_ID,
            "spawn_protocol_version": "picklift_spawn_v2",
            "spawn_region": "r2c1",
            "spawn_x_cm": 15,
        }
    )
    cfg.pop("alignment_reference_id")

    validate_config(cfg)

    assert cfg["alignment_reference_id"] == "picklift_red_cube_alignment_v1"


def test_spawn_v3_balanced_grid_covers_all_nine_regions():
    regions = {
        spawn_region_for(x_cm, y_cm, "picklift_spawn_v3") for x_cm in (22, 27, 32) for y_cm in (-7, 0, 7)
    }
    assert regions == {f"r{row}c{column}" for row in range(1, 4) for column in range(1, 4)}
    assert spawn_region_for(25, 0, "picklift_spawn_v3") == "r2c2"
    assert spawn_region_for(30, 0, "picklift_spawn_v3") == "r3c2"


def test_legacy_v3_config_remains_valid(tmp_path):
    cfg = make_legacy_config(config(tmp_path))
    cfg.update(
        {
            "spawn_protocol_version": "picklift_spawn_v3",
            "spawn_region": "r2c2",
            "spawn_x_cm": 25,
            "spawn_y_cm": 0,
        }
    )

    validate_config(cfg)

    assert "grid_shape" not in spawn_contract("picklift_spawn_v3")


def test_spawn_v4_grid_covers_all_twelve_cells():
    regions = {
        spawn_region_for(x_cm, y_cm, "picklift_spawn_v4")
        for x_cm in (22.5, 27.5, 32.5)
        for y_cm in (-7.5, -2.5, 2.5, 7.5)
    }
    assert regions == {f"r{row}c{column}" for row in range(1, 4) for column in range(1, 5)}
    assert spawn_contract("picklift_spawn_v4")["grid_shape"] == {
        "rows": 3,
        "columns": 4,
        "cells": 12,
    }


def test_legacy_v4_config_remains_valid(tmp_path):
    cfg = make_legacy_config(config(tmp_path))
    cfg["spawn_protocol_version"] = "picklift_spawn_v4"

    validate_config(cfg)


def test_spawn_v4_ui_summary_identifies_twelve_cell_grid(tmp_path):
    cfg = make_legacy_config(config(tmp_path))
    cfg.update(
        {
            "spawn_protocol_version": "picklift_spawn_v4",
            "spawn_id": "pl_v4_s012",
            "spawn_region": "r3c4",
            "spawn_x_cm": 32.5,
            "spawn_y_cm": 7.5,
            "spawn_yaw_deg": 90,
        }
    )

    summary = spawn_ui_summary(cfg)

    assert "picklift_spawn_v4 | pl_v4_s012 | r3c4" in summary
    assert "Xfwd=32.5cm Ylat=7.5cm yaw=90" in summary
    assert "12 cells | picklift_red_cube_alignment_v2" in summary


def test_spawn_v5_ui_requires_no_yaw_measurement(tmp_path):
    cfg = config(tmp_path)
    cfg.update(
        {
            "spawn_id": "pl_v5_s012",
            "spawn_region": "r3c4",
            "spawn_x_cm": 32.5,
            "spawn_y_cm": 7.5,
        }
    )

    summary = spawn_ui_summary(cfg)

    assert summary.splitlines() == [
        "picklift_spawn_v5 | 12 cells",
        "pl_v5_s012 r3c4 | X=32.5 Y=7.5",
        "Yaw arbitrary 0..90 | no measure",
    ]


def test_task_frame_v1_reference_is_immutable_and_v2_inherits_geometry():
    frame_v1 = task_frame(TASK_GRID_FRAME_V1_ID)
    frame_v2 = task_frame(TASK_GRID_FRAME_ID)
    assert frame_v1["known_reference"]["coordinates_m"] == {"x": 0.15, "y": 0.0}
    assert "known_reference" not in frame_v2
    assert frame_v2["geometry_parent_frame_id"] == TASK_GRID_FRAME_V1_ID
    for key in ("origin", "centerline", "axes", "units", "measurement_rule"):
        assert frame_v2[key] == frame_v1[key]


def test_new_red_cube_reference_is_pending_x_25cm_y_zero():
    reference = alignment_reference(ALIGNMENT_REFERENCE_V2_ID)
    assert reference["red_cube_center_m"] == {"x_forward": 0.25, "y_lateral": 0.0}
    assert reference["physical_confirmation_status"] == "pending_new_25cm_screenshot"
    assert spawn_region_for(25, 0) == "r2c3"
    assert spawn_contract("picklift_spawn_v3")["x_cm"] == {
        "min": 20.0,
        "max": 35.0,
        "description": "task-grid +X forward",
    }


def test_alignment_reference_point_is_rejected_for_formal_v5_episode(tmp_path):
    cfg = config(tmp_path)
    cfg.update(
        {
            "formal_data": True,
            "spawn_region": "r2c3",
            "spawn_x_cm": 25,
            "spawn_y_cm": 0,
        }
    )
    with pytest.raises(ValueError, match="alignment-only"):
        validate_config(cfg)


def test_v5_spawn_plan_has_twelve_valid_interior_recommendations():
    root = Path(__file__).parents[2]
    plan = json.loads((root / "examples/picklift_v3/spawn_plan.template.json").read_text())
    spawns = plan["spawns"]

    assert plan["protocol_version"] == "picklift_spawn_v5"
    assert plan["collection_protocol_version"] == "picklift_collection_v5"
    assert plan["task_spec_revision"] == "picklift_taskspec_v2_unmeasured_yaw"
    assert plan["grid_shape"] == {"rows": 3, "columns": 4, "cells": 12, "cell_size_cm": 5}
    assert plan["yaw_annotation"] == {
        "yaw_annotation_mode": "unmeasured_random",
        "spawn_yaw_deg": None,
        "yaw_intended_range_deg": [0, 90],
        "yaw_sampling_method": "operator_unmeasured_arbitrary",
        "yaw_distribution_claim": "unknown",
    }
    assert plan["success_annotation"] == {
        "success_annotation_source": "operator_visual_v1",
        "success_detection_mode": "manual_proxy_for_nexus_v1",
        "minimum_visual_lift_height_m": 0.05,
        "recommended_target_lift_range_m": [0.06, 0.08],
        "minimum_stable_hold_s": 0.5,
        "requires_visible_bilateral_finger_grasp": True,
        "requires_success_pose_held_at_end": True,
        "lift_height_m": None,
        "is_grasped": None,
    }
    assert len(spawns) == 12
    assert {item["spawn_region"] for item in spawns} == {
        f"r{row}c{column}" for row in range(1, 4) for column in range(1, 5)
    }
    for item in spawns:
        assert spawn_region_for(item["recommended_x_cm"], item["recommended_y_cm"]) == item["spawn_region"]
        assert (item["recommended_x_cm"], item["recommended_y_cm"]) != (25, 0)
        assert item["actual_x_cm"] is None
        assert item["actual_y_cm"] is None
        assert item["spawn_yaw_deg"] is None
        assert item["yaw_randomization_confirmed"] is None
        assert "recommended_yaw_deg" not in item
        assert "actual_yaw_deg" not in item


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"spawn_x_cm": 19}, "20..35"),
        ({"spawn_y_cm": 11}, "-10..10"),
        ({"spawn_protocol_version": "picklift_spawn_unknown"}, "unsupported"),
        ({"task_frame_id": "picklift_task_grid_unknown"}, "task_frame"),
        ({"alignment_reference_id": "picklift_alignment_unknown"}, "alignment_reference"),
        ({"task_frame_id": TASK_GRID_FRAME_V1_ID}, "does not match"),
        ({"spawn_yaw_deg": 45}, "unmeasured_random and spawn_yaw_deg=null"),
        ({"yaw_annotation_mode": "estimated"}, "unmeasured_random and spawn_yaw_deg=null"),
        ({"yaw_intended_range_deg": [0, 360]}, "yaw_intended_range_deg"),
        ({"yaw_sampling_method": "operator_estimated"}, "yaw_sampling_method"),
        ({"yaw_distribution_claim": "uniform"}, "yaw_distribution_claim"),
        ({"yaw_randomization_confirmed": "yes"}, "must be boolean"),
        ({"collection_protocol_version": "picklift_collection_v4"}, "collection_protocol_version"),
        ({"task_spec_revision": "picklift_taskspec_v1"}, "task_spec_revision"),
        ({"success_annotation_source": "automatic"}, "success_annotation_source"),
        ({"success_detection_mode": "automatic"}, "success_detection_mode"),
        ({"lift_height_m": 0.05}, "must remain null"),
        ({"is_grasped": True}, "must remain null"),
        ({"spawn_region": "r1c2"}, "does not match"),
        ({"result": "success", "success": False}, "success must be true"),
        ({"result": "pending"}, "requires operator_ui"),
    ],
)
def test_spawn_protocol_fails_closed(tmp_path, mutation, error):
    cfg = config(tmp_path)
    cfg.update(mutation)
    with pytest.raises(ValueError, match=error):
        validate_config(cfg)


def test_v5_requires_spawn_yaw_field_present_as_null(tmp_path):
    cfg = config(tmp_path)
    cfg.pop("spawn_yaw_deg")
    with pytest.raises(ValueError, match="missing explicit.*spawn_yaw_deg"):
        validate_config(cfg)
