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
from unittest.mock import patch

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets, split_dataset
from lerobot.datasets.io_utils import load_info, write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.servo_settings import (
    SERVO_SETTINGS_COLUMN,
    read_servo_settings,
    setting_register_names,
)
from lerobot.scripts.lerobot_record import RecordConfig, record
from tests.fixtures.constants import DUMMY_REPO_ID
from tests.mocks.mock_robot import MockRobotConfig
from tests.mocks.mock_teleop import MockTeleopConfig

pytestmark = [pytest.mark.dataset, pytest.mark.hardware]

_CAL_11 = {"motors": {"joint": {"id": 1, "homing_offset": 11}}}


def test_setting_registers_include_acceleration_and_skip_live_readings():
    pytest.importorskip("rustypot")
    names = setting_register_names("sts3215")
    assert "acceleration" in names
    assert "maximum_acceleration" in names
    assert "p_coefficient" in names
    assert "homing_offset" in names
    assert "offset" not in names
    assert "present_position" not in names
    assert "present_current" not in names
    assert "goal_position" not in names
    assert "torque_enable" not in names
    assert len(names) == len(set(names))


def test_read_servo_settings_stores_model_calibration_and_registers():
    pytest.importorskip("rustypot")

    class Motor:
        model = "sts3215"
        id = 1

    class Bus:
        def __init__(self):
            self.motors = {"shoulder": Motor()}
            self.read = []

        def sync_read(self, name, motors=None, normalize=True):
            self.read.append(name)
            assert normalize is False
            return {"shoulder": 4}

    robot = type("R", (), {})()
    robot.bus = Bus()
    robot.calibration = {
        "shoulder": {
            "id": 1,
            "drive_mode": 0,
            "homing_offset": 11,
            "range_min": 0,
            "range_max": 100,
        }
    }
    record = read_servo_settings(robot)
    motor = record["motors"]["shoulder"]
    assert motor["model"] == "sts3215"
    assert motor["calibration"]["homing_offset"] == 11
    assert motor["registers"]["acceleration"] == 4
    assert motor["registers"]["p_coefficient"] == 4
    assert "present_position" not in motor["registers"]
    assert "goal_position" not in robot.bus.read


def _record_cfg(root: Path, episodes: int = 1) -> RecordConfig:
    return RecordConfig(
        robot=MockRobotConfig(),
        dataset=DatasetRecordConfig(
            repo_id=DUMMY_REPO_ID,
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


def _settings(root: Path) -> list:
    values = []
    for path in _episode_files(root):
        frame = pq.read_table(path).to_pandas()
        if SERVO_SETTINGS_COLUMN not in frame.columns:
            values.extend([None] * len(frame))
        else:
            values.extend(frame[SERVO_SETTINGS_COLUMN].tolist())
    return values


def test_record_writes_servo_settings_on_each_episode(tmp_path):
    root = tmp_path / "new"
    record(_record_cfg(root, episodes=2))
    stored = _settings(root)
    assert len(stored) == 2
    for value in stored:
        parsed = json.loads(value)
        assert "motor_1" in parsed["motors"]
        assert parsed["motors"]["motor_1"]["model"] == "model_1"


def test_resume_of_an_old_dataset_writes_no_servo_settings(tmp_path):
    root = tmp_path / "old"
    record(_record_cfg(root))
    path = _episode_files(root)[0]
    frame = pd.read_parquet(path)
    frame = frame.drop(columns=[SERVO_SETTINGS_COLUMN])
    frame.to_parquet(path)
    info = load_info(root)
    info.calibration = _CAL_11
    write_info(info, root)

    warnings = []
    cfg = _record_cfg(root)
    cfg.resume = True
    with (
        _local_dataset(),
        patch(
            "lerobot.scripts.lerobot_record.follower_calibration",
            return_value={"motors": {"joint": {"id": 1, "homing_offset": 22}}},
        ),
        patch(
            "lerobot.scripts.lerobot_record.logging.warning",
            side_effect=lambda msg, *args, **kwargs: warnings.append(str(msg)),
        ),
    ):
        record(cfg)

    assert warnings == [
        "Not recording servo settings: this dataset has episodes and no "
        "servo_settings column. Start a new dataset to record them."
    ]
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["calibration"] == _CAL_11
    assert info["total_episodes"] == 2
    loaded = Dataset.from_parquet([str(path) for path in _episode_files(root)])
    assert SERVO_SETTINGS_COLUMN not in loaded.column_names
    assert loaded.num_rows == 2
    assert json.loads((root / "meta" / "info.json").read_text())["calibration"] == _CAL_11


def test_delete_and_split_keep_servo_settings(tmp_path):
    root = tmp_path / "tools"
    record(_record_cfg(root, episodes=2))
    original = json.loads(_settings(root)[0])
    with _local_dataset():
        dataset = LeRobotDataset(DUMMY_REPO_ID, root=root)
        deleted = delete_episodes(dataset, [1], output_dir=tmp_path / "deleted", repo_id="dummy/deleted")
        splits = split_dataset(dataset, {"train": [0], "val": [1]}, output_dir=tmp_path / "split")
    assert json.loads(_settings(deleted.root)[0]) == original
    assert json.loads(_settings(splits["train"].root)[0]) == original
    assert json.loads(_settings(splits["val"].root)[0]) == original


def test_merge_of_old_and_new_datasets_stays_loadable(tmp_path):
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    record(_record_cfg(new_root))
    record(_record_cfg(old_root))
    old_file = _episode_files(old_root)[0]
    frame = pd.read_parquet(old_file).drop(columns=[SERVO_SETTINGS_COLUMN])
    frame.to_parquet(old_file)
    warnings = []
    with (
        _local_dataset(),
        patch(
            "lerobot.datasets.aggregate.logger.warning",
            side_effect=lambda msg, *args, **kwargs: warnings.append(str(msg)),
        ),
    ):
        new_ds = LeRobotDataset(DUMMY_REPO_ID, root=new_root)
        old_ds = LeRobotDataset("dummy/old", root=old_root)
        merged = merge_datasets(
            [new_ds, old_ds],
            output_repo_id="dummy/merged",
            output_dir=tmp_path / "merged",
        )
    loaded = Dataset.from_parquet([str(path) for path in _episode_files(merged.root)])
    assert loaded.num_rows == 2
    assert SERVO_SETTINGS_COLUMN in loaded.column_names
    values = loaded[SERVO_SETTINGS_COLUMN]
    assert json.loads(values[0])["motors"]["motor_1"]["model"] == "model_1"
    assert values[1] is None
    assert any("dummy/old" in msg and "have none" in msg for msg in warnings)


def test_info_json_calibration_is_one_switch(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_record.RECORD_INFO_JSON_CALIBRATION", False)
    root = tmp_path / "flag"
    with patch(
        "lerobot.scripts.lerobot_record.follower_calibration",
        return_value=_CAL_11,
    ):
        record(_record_cfg(root))
    info = json.loads((root / "meta" / "info.json").read_text())
    assert "calibration" not in info
    assert json.loads(_settings(root)[0])["motors"]["motor_1"]["model"] == "model_1"
