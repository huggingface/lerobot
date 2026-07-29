import json

from examples.picklift_v3.backend import SyntheticBackend
from examples.picklift_v3.batch_record import (
    BATCH_WORKFLOW_VERSION,
    record_batch,
    validate_batch_config,
)
from tests.picklift_v3.test_record import config


class FakeBatchUI:
    def __init__(self, results):
        self.results = iter(results)
        self.opened = False
        self.closed = False
        self.next_start_count = 0
        self.finish_count = 0

    def open(self):
        self.opened = True

    def close(self):
        self.closed = True

    def wait_for_ready(self, _frame, _message):
        return True

    def show_status(self, _frame, **_kwargs):
        pass

    def wait_for_start(self, _frame_provider, message=""):
        assert "picklift_spawn_v5" in message

    def wait_for_next_start(self, frame_provider, message):
        assert frame_provider().shape == (480, 640, 3)
        assert "Follower LIVE" in message
        self.next_start_count += 1
        return True

    def wait_for_finish(self, frame_provider):
        assert frame_provider().shape == (480, 640, 3)
        self.finish_count += 1

    def show(self, _frame, **_kwargs):
        return "stop"

    def review_result(self, _frame):
        return next(self.results)

    def show_saving(self, _frame, *, result):
        assert result in {"success", "failure", "discard"}

    def show_attempt_complete(self, _frame, **_kwargs):
        pass


class CountingSyntheticBackend(SyntheticBackend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.connect_calls = 0
        self.close_calls = 0

    def connect(self):
        self.connect_calls += 1

    def close(self):
        self.close_calls += 1


def batch_config(tmp_path, *, successes_per_spawn=1, max_attempts=4):
    base = config(tmp_path)
    base.update(
        {
            "operator_ui": True,
            "result": "pending",
            "success": False,
            "episode_seconds": 0.1,
        }
    )
    return {
        "collection_workflow_version": BATCH_WORKFLOW_VERSION,
        "successes_per_spawn": successes_per_spawn,
        "max_attempts": max_attempts,
        "base_config": base,
        "spawns": [
            {
                "spawn_id": "pl_v5_s001",
                "spawn_region": "r1c1",
                "spawn_x_cm": 22.5,
                "spawn_y_cm": -7.5,
                "spawn_yaw_deg": None,
            },
            {
                "spawn_id": "pl_v5_s002",
                "spawn_region": "r1c2",
                "spawn_x_cm": 22.5,
                "spawn_y_cm": -2.5,
                "spawn_yaw_deg": None,
            },
        ],
    }


def test_continuous_batch_retries_failure_and_only_saves_successes(tmp_path):
    cfg = batch_config(tmp_path, successes_per_spawn=1, max_attempts=4)
    ui = FakeBatchUI(["failure", "success", "success"])
    backends = []

    def backend_factory(backend_cfg):
        backend = CountingSyntheticBackend(backend_cfg)
        backends.append(backend)
        return backend

    root = record_batch(cfg, backend_factory=backend_factory, ui=ui)

    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(cfg["base_config"]["repo_id"], root=root)
    assert len(dataset) == 2
    assert dataset.num_episodes == 2
    attempts = [
        json.loads(path.read_text()) for path in sorted((root / "provenance/attempts").glob("*.json"))
    ]
    episodes = [
        json.loads(path.read_text()) for path in sorted((root / "provenance/episodes").glob("*.json"))
    ]
    assert [item["spawn_id"] for item in attempts] == [
        "pl_v5_s001",
        "pl_v5_s001",
        "pl_v5_s002",
    ]
    assert [item["result"] for item in attempts] == ["failure", "success", "success"]
    assert [item["saved_to_training"] for item in attempts] == [False, True, True]
    assert [item["episode_index"] for item in attempts] == [None, 0, 1]
    assert [item["spawn_id"] for item in episodes] == ["pl_v5_s001", "pl_v5_s002"]
    manifest = json.loads((root / "provenance/session.json").read_text())
    assert manifest["complete"] is True
    assert manifest["attempt_count"] == 3
    assert manifest["saved_episode_count"] == 2
    assert manifest["training_view_rule"].startswith("only operator-confirmed SUCCESS")
    assert manifest["collection_workflow_version"] == BATCH_WORKFLOW_VERSION
    assert manifest["collection_commit"]
    assert manifest["lerobot_dataset_version"] == "v3.0"
    assert manifest["joint_order"][-1] == "gripper"
    assert manifest["gripper_alignment_mode"] == "direct_absolute_0_100"
    assert manifest["task_frame"]["frame_id"] == "picklift_task_grid_v2"
    assert manifest["camera_profile"]["profile_id"] == "synthetic_front_640x480_v1"
    assert manifest["batch_end_time"] is not None
    assert manifest["post_end_control_mode"] == "live_follow_no_recording"
    assert manifest["inter_episode_control"].startswith("live absolute Leader-to-Follower")
    assert manifest["ready_pose_policy"] == "operator_visual_similar_ready_area_no_numeric_threshold"
    assert manifest["action_mapping"] == "official_so101_direct_absolute"
    assert manifest["action_transform"] == "none"
    assert ui.next_start_count == 2
    assert ui.finish_count == 1
    assert ui.opened and ui.closed
    assert len(backends) == 1
    assert backends[0].connect_calls == 1
    assert backends[0].close_calls == 1


def test_multiple_successes_per_spawn_are_balanced_across_cells(tmp_path):
    cfg = batch_config(tmp_path, successes_per_spawn=2, max_attempts=5)
    ui = FakeBatchUI(["failure", "success", "success", "success", "success"])

    root = record_batch(cfg, backend_factory=SyntheticBackend, ui=ui)

    attempts = [
        json.loads(path.read_text()) for path in sorted((root / "provenance/attempts").glob("*.json"))
    ]
    assert [item["spawn_id"] for item in attempts] == [
        "pl_v5_s001_rep01",
        "pl_v5_s001_rep01",
        "pl_v5_s002_rep01",
        "pl_v5_s001_rep02",
        "pl_v5_s002_rep02",
    ]
    manifest = json.loads((root / "provenance/dataset.json").read_text())
    assert manifest["saved_episode_count"] == 4
    assert [item["saved_successes"] for item in manifest["planned_spawns"]] == [2, 2]
    assert [item["spawn_id"] for item in manifest["planned_spawns"]] == [
        "pl_v5_s001",
        "pl_v5_s002",
    ]


def test_real_batch_accepts_direct_absolute_future_protocol(tmp_path):
    cfg = batch_config(tmp_path)
    cfg["base_config"].update(
        {
            "mode": "real",
            "camera_config_version": "icspring_front_crop_1280x960_to_640x480_v1",
            "camera_profile_id": "icspring_front_crop_1280x960_to_640x480_v1",
            "follower_port": "/dev/serial/by-id/follower",
            "leader_port": "/dev/serial/by-id/leader",
            "powered_real_run_ack": "I_HAVE_COMPLETED_THE_POWERED_SAFETY_CHECK",
            "alignment_mode": "direct_absolute",
        }
    )

    validate_batch_config(cfg)


def test_batch_rejects_relative_rebase_and_legacy_protocol(tmp_path):
    cfg = batch_config(tmp_path)
    cfg["base_config"]["alignment_mode"] = "relative_rebase"
    try:
        validate_batch_config(cfg)
    except ValueError as exc:
        assert "direct_absolute" in str(exc)
    else:
        raise AssertionError("batch v3 accepted relative_rebase")

    cfg = batch_config(tmp_path)
    cfg["base_config"]["collection_protocol_version"] = "picklift_collection_v5"
    try:
        validate_batch_config(cfg)
    except ValueError as exc:
        assert "collection_protocol_version" in str(exc)
    else:
        raise AssertionError("batch v3 accepted legacy collection protocol")
