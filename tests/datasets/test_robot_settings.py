#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.dataset_tools import (
    convert_image_to_video_dataset,
    delete_episodes,
    merge_datasets,
    split_dataset,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.robot_settings import ROBOT_SETTINGS_COLUMN, settings_text
from lerobot.scripts.lerobot_record import RecordConfig, record
from tests.datasets.test_video_encoding import require_h264
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig

pytestmark = [pytest.mark.dataset, pytest.mark.hardware]

COLUMN = ROBOT_SETTINGS_COLUMN


def _record_cfg(root: Path, episodes: int = 1, repo_id: str = DUMMY_REPO_ID) -> RecordConfig:
    return RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=repo_id,
            single_task="Dummy task",
            root=root,
            num_episodes=episodes,
            episode_time_s=0.1,
            reset_time_s=0,
            push_to_hub=False,
            video=False,
            no_stamp=True,
        ),
        teleop=MockTeleopConfig(),
        play_sounds=False,
    )


@contextmanager
def _local_dataset():
    with (
        patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
        patch("lerobot.datasets.dataset_metadata.snapshot_download", return_value="unused"),
    ):
        yield


def _episode_files(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("*/*.parquet"))


def _column(root: Path) -> list:
    values = []
    for path in _episode_files(root):
        table = pq.read_table(path)
        if COLUMN not in table.column_names:
            values.extend([None] * table.num_rows)
        else:
            values.extend(table.column(COLUMN).to_pylist())
    return values


def _counting_settings(self):
    n = _counting_settings.n + 1
    _counting_settings.n = n
    return {"robot_type": "so101_follower", "robot_id": "arm", "episode": n, "motors": {}}


_counting_settings.n = 0


def test_default_robot_reports_no_settings():
    assert MockRobot(MockRobotConfig()).read_settings() is None
    assert settings_text(MockRobot(MockRobotConfig())) is None


def test_so_follower_reads_the_explicit_sts3215_list():
    pytest.importorskip("rustypot")
    from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
    from lerobot.robots.so_follower.so_follower import _STS3215_SETTING_REGISTERS, SOFollower

    assert "acceleration" in _STS3215_SETTING_REGISTERS
    assert "maximum_acceleration" in _STS3215_SETTING_REGISTERS
    assert "p_coefficient" in _STS3215_SETTING_REGISTERS
    assert "goal_position" not in _STS3215_SETTING_REGISTERS
    assert "torque_enable" not in _STS3215_SETTING_REGISTERS
    assert not any(name.startswith("present_") for name in _STS3215_SETTING_REGISTERS)

    class Bus:
        def __init__(self):
            self.calls = []
            self.motors = {
                "shoulder": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "other": Motor(2, "xl330", MotorNormMode.RANGE_M100_100),
            }

        def sync_read(self, register, names, normalize=True):
            self.calls.append((register, tuple(names), normalize))
            if register == "acceleration":
                raise RuntimeError("skip this register")
            return dict.fromkeys(names, 7)

    robot = SOFollower.__new__(SOFollower)
    robot.config = SimpleNamespace(type="so101_follower")
    robot.id = "arm-a"
    robot.robot_type = "so_follower"
    robot.calibration = {
        "shoulder": MotorCalibration(1, 0, 11, 0, 4095),
        "other": {"id": 2, "drive_mode": 0, "homing_offset": 3, "range_min": 0, "range_max": 100},
    }
    robot.bus = Bus()

    value = robot.read_settings()
    assert value["robot_type"] == "so101_follower"
    assert value["robot_id"] == "arm-a"
    assert value["motors"]["shoulder"]["model"] == "sts3215"
    assert value["motors"]["shoulder"]["calibration"]["homing_offset"] == 11
    assert value["motors"]["shoulder"]["registers"]["p_coefficient"] == 7
    assert "acceleration" not in value["motors"]["shoulder"]["registers"]
    assert "registers" not in value["motors"]["other"]
    assert {call[0] for call in robot.bus.calls} == set(_STS3215_SETTING_REGISTERS)
    assert all(call[1] == ("shoulder",) and call[2] is False for call in robot.bus.calls)
    text = settings_text(robot)
    assert isinstance(text, str)
    assert json.loads(text)["motors"]["shoulder"]["registers"]["p_coefficient"] == 7


def test_a_failed_read_still_writes_text(tmp_path):
    def boom(self):
        raise RuntimeError("bus down")

    root = tmp_path / "failed"
    with patch.object(MockRobot, "read_settings", boom):
        record(_record_cfg(root))
    stored = _column(root)
    assert len(stored) == 1
    assert json.loads(stored[0])["error"] == "bus down"


def test_record_and_resume_keep_the_earlier_episode_file(tmp_path):
    _counting_settings.n = 0
    root = tmp_path / "new"
    with patch.object(MockRobot, "read_settings", _counting_settings):
        record(_record_cfg(root))
    first = _episode_files(root)[0].read_bytes()
    cfg = _record_cfg(root)
    cfg.resume = True
    with _local_dataset(), patch.object(MockRobot, "read_settings", _counting_settings):
        record(cfg)
    assert _episode_files(root)[0].read_bytes() == first
    stored = _column(root)
    assert [json.loads(value)["episode"] for value in stored] == [1, 2]
    loaded = Dataset.from_parquet([str(path) for path in _episode_files(root)])
    assert loaded.features[COLUMN].dtype == "string"


def test_resume_of_an_old_dataset_writes_no_column(tmp_path):
    root = tmp_path / "old"
    record(_record_cfg(root))
    assert COLUMN not in pq.read_schema(_episode_files(root)[0]).names
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["calibration"] = {"motors": {"joint": {"id": 1}}}
    info_path.write_text(json.dumps(info))

    warnings = []
    cfg = _record_cfg(root)
    cfg.resume = True
    with (
        _local_dataset(),
        patch(
            "lerobot.scripts.lerobot_record.logging.warning",
            side_effect=lambda msg, *args, **kwargs: warnings.append(str(msg)),
        ),
    ):
        record(cfg)

    assert warnings == [
        "Not recording robot settings: this dataset has episodes and no "
        "robot_settings column. Start a new dataset to record them."
    ]
    info = json.loads(info_path.read_text())
    assert "calibration" not in info
    assert info["total_episodes"] == 2
    loaded = Dataset.from_parquet([str(path) for path in _episode_files(root)])
    assert COLUMN not in loaded.column_names
    assert loaded.num_rows == 2


def test_delete_split_and_conversion_keep_the_column(tmp_path):
    _counting_settings.n = 0
    root = tmp_path / "tools"
    with patch.object(MockRobot, "read_settings", _counting_settings):
        record(_record_cfg(root, episodes=2))
    original = [json.loads(value)["episode"] for value in _column(root)]
    with _local_dataset():
        dataset = LeRobotDataset(DUMMY_REPO_ID, root=root)
        deleted = delete_episodes(dataset, [1], output_dir=tmp_path / "deleted", repo_id="dummy/deleted")
        splits = split_dataset(dataset, {"train": [0], "val": [1]}, output_dir=tmp_path / "split")
    assert json.loads(_column(deleted.root)[0])["episode"] == original[0]
    assert json.loads(_column(splits["train"].root)[0])["episode"] == original[0]
    assert json.loads(_column(splits["val"].root)[0])["episode"] == original[1]


@require_h264
def test_image_to_video_keeps_the_column(tmp_path):
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (16, 16, 3), "names": None},
    }
    root = tmp_path / "images"
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID, fps=10, root=root, features=features, use_videos=False
    )
    for episode in (1, 2):
        dataset.add_frame(
            {
                "action": np.zeros(1, dtype=np.float32),
                "observation.state": np.zeros(1, dtype=np.float32),
                "observation.images.top": np.zeros((16, 16, 3), dtype=np.uint8),
                "task": "look",
            }
        )
        dataset.save_episode(robot_settings=json.dumps({"episode": episode}))
    dataset.finalize()
    with _local_dataset():
        source = LeRobotDataset(DUMMY_REPO_ID, root=root)
        converted = convert_image_to_video_dataset(
            source,
            output_dir=tmp_path / "video",
            repo_id="dummy/video",
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", g=2, crf=30, preset="ultrafast"),
            num_workers=1,
        )
    stored = _column(converted.root)
    assert [json.loads(value)["episode"] for value in stored] == [1, 2]
    assert pq.read_schema(_episode_files(converted.root)[0]).field(COLUMN).type == pa.string()


def test_mixed_merge_drops_the_column(tmp_path):
    _counting_settings.n = 0
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    with patch.object(MockRobot, "read_settings", _counting_settings):
        record(_record_cfg(new_root, repo_id="dummy/new"))
    record(_record_cfg(old_root, repo_id="dummy/old"))
    warnings = []
    with (
        _local_dataset(),
        patch(
            "lerobot.datasets.aggregate.logger.warning",
            side_effect=lambda msg, *args, **kwargs: warnings.append(str(msg)),
        ),
    ):
        new_ds = LeRobotDataset("dummy/new", root=new_root)
        old_ds = LeRobotDataset("dummy/old", root=old_root)
        merged = merge_datasets(
            [new_ds, old_ds],
            output_repo_id="dummy/merged",
            output_dir=tmp_path / "merged",
        )
    loaded = Dataset.from_parquet([str(path) for path in _episode_files(merged.root)])
    assert loaded.num_rows == 2
    assert COLUMN not in loaded.column_names
    assert warnings == ["Dropping robot_settings because not every dataset has it. dummy/old have none."]


def test_merge_keeps_the_column_when_every_source_has_it(tmp_path):
    _counting_settings.n = 0
    left = tmp_path / "left"
    right = tmp_path / "right"
    with patch.object(MockRobot, "read_settings", _counting_settings):
        record(_record_cfg(left, repo_id="dummy/left"))
        record(_record_cfg(right, repo_id="dummy/right"))
    with _local_dataset():
        merged = merge_datasets(
            [LeRobotDataset("dummy/left", root=left), LeRobotDataset("dummy/right", root=right)],
            output_repo_id="dummy/merged",
            output_dir=tmp_path / "merged",
        )
    stored = [json.loads(value)["episode"] for value in _column(merged.root)]
    assert stored == [1, 2]
    assert pq.read_schema(_episode_files(merged.root)[0]).field(COLUMN).type == pa.string()


def test_many_episodes_rotate_files_and_stay_strings(tmp_path):
    features = {
        "action": {"dtype": "float32", "shape": (1,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (1,), "names": None},
    }
    root = tmp_path / "many"
    dataset = LeRobotDataset.create(repo_id=DUMMY_REPO_ID, fps=10, root=root, features=features)
    dataset.meta.info.data_files_size_in_mb = 0
    for episode in range(11):
        dataset.add_frame(
            {
                "action": np.zeros(1, dtype=np.float32),
                "observation.state": np.zeros(1, dtype=np.float32),
                "task": "move",
            }
        )
        dataset.save_episode(robot_settings=json.dumps({"episode": episode}))
    dataset.finalize()
    files = _episode_files(root)
    assert len(files) >= 2
    values = []
    for path in files:
        schema = pq.read_schema(path)
        assert schema.field(COLUMN).type == pa.string()
        values.extend(pq.read_table(path).column(COLUMN).to_pylist())
    assert values == [json.dumps({"episode": episode}) for episode in range(11)]
    loaded = Dataset.from_parquet([str(path) for path in files])
    assert loaded.num_rows == 11
