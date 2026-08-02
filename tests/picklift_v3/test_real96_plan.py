import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from examples.picklift_v3.batch_record import validate_batch_config
from examples.picklift_v3.prepare_real96_session import make_config, validate_without_hardware
from examples.picklift_v3.real96_plan import (
    COLLECTION_PLAN_SHA256,
    SESSION_ITEMS_SHA256,
    batch_spawns,
    compact_session_bytes,
    real96_items,
    session_items,
    validate_session_source,
)


def device_config() -> dict:
    return {
        "camera_device": "/dev/v4l/by-id/front",
        "robot_id": "so101_follower_main",
        "follower_port": "/dev/serial/by-id/follower",
        "leader_id": "so101_leader_main",
        "leader_port": "/dev/serial/by-id/leader",
        "follower_calibration_path": "/calibration/follower.json",
        "follower_calibration_sha256": "f" * 64,
        "leader_calibration_path": "/calibration/leader.json",
        "leader_calibration_sha256": "e" * 64,
    }


def test_transferred_collection_plan_bytes_match_research_control():
    root = Path(__file__).parents[2]
    path = root / "examples/picklift_v3/contracts/task1-picklift-real96-collection-v1.json"
    assert len(path.read_bytes()) == 3496
    assert hashlib.sha256(path.read_bytes()).hexdigest() == COLLECTION_PLAN_SHA256
    assert json.loads(path.read_text())["plan_id"] == "task1_picklift_real96_collection_v1"


def test_real96_generator_reproduces_transferred_session1_exactly():
    validate_session_source(1)
    assert len(compact_session_bytes(1)) == 14171
    assert hashlib.sha256(compact_session_bytes(1)).hexdigest() == SESSION_ITEMS_SHA256[1]
    items = session_items(1)
    assert len(items) == 24
    assert [item["session_order"] for item in items] == list(range(1, 25))
    assert items[0]["plan_item_id"] == "real96_s01_r3c4_core_center_rep01_yaw00"
    assert items[-1]["plan_item_id"] == "real96_s01_r2c1_extension_q3_rep01_yaw45"


def test_full_real96_distribution_is_frozen_and_balanced():
    items = real96_items()
    assert len(items) == 96
    assert len({item["plan_item_id"] for item in items}) == 96
    assert len({item["nominal_pose_key"] for item in items}) == 72
    assert Counter(item["cell"] for item in items) == {
        f"r{row}c{column}": 8 for row in range(1, 4) for column in range(1, 5)
    }
    assert Counter(item["session_index"] for item in items) == {1: 24, 2: 24, 3: 24, 4: 24}
    assert Counter(item["yaw_degrees_modulo_90"] for item in items) == {0: 48, 45: 48}
    assert Counter(item["position_kind"] for item in items) == {"center": 48, "offset": 48}
    assert Counter(item["quadrant"] for item in items if item["quadrant"] is not None) == {
        "Q0": 12,
        "Q1": 12,
        "Q2": 12,
        "Q3": 12,
    }
    assert sum(item["real48_member"] for item in items) == 48


def test_session1_config_validates_without_devices_or_power(tmp_path):
    cfg = make_config(
        session_index=1,
        operator_id="operator_01",
        dataset_root=tmp_path / "new_raw_attempts",
        device_config=device_config(),
    )
    assert cfg["base_config"]["powered_real_run_ack"] == ""
    assert len(cfg["spawns"]) == 24
    validate_without_hardware(cfg)
    powered_copy = json.loads(json.dumps(cfg))
    powered_copy["base_config"]["powered_real_run_ack"] = "I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK"
    validate_batch_config(powered_copy)


def test_real96_plan_pose_or_identity_mutation_fails_closed(tmp_path):
    cfg = make_config(
        session_index=1,
        operator_id="operator_01",
        dataset_root=tmp_path / "new_raw_attempts",
        device_config=device_config(),
        powered_ack="I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK",
    )
    cfg["spawns"][0]["spawn_x_cm"] += 1
    with pytest.raises(ValueError, match="x_forward_m"):
        validate_batch_config(cfg)

    cfg = make_config(
        session_index=1,
        operator_id="operator_01",
        dataset_root=tmp_path / "other_raw_attempts",
        device_config=device_config(),
        powered_ack="I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK",
    )
    cfg["spawns"][0]["spawn_yaw_deg"] = 13
    with pytest.raises(ValueError, match="yaw"):
        validate_batch_config(cfg)


def test_batch_spawns_preserve_every_transferred_plan_field():
    source = session_items(1)[0]
    prepared = batch_spawns(1)[0]
    for key, value in source.items():
        assert prepared[key] == value
