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

"""Record and resume image streams without a robot.cameras attribute.

Sensor data and transport are synthetic. Dataset persistence and decoding use
the real recording workflow.
"""

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("deepdiff", reason="deepdiff is required (install lerobot[hardware])")

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.configs.types import PipelineFeatureType
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.processor import ObservationProcessorStep, make_default_processors
from lerobot.robots.microduck import MicroDuck, MicroDuckConfig
from lerobot.scripts.lerobot_record import RecordConfig, record
from tests.mocks.mock_teleop import MockTeleop, MockTeleopConfig

IMAGE_SHAPE = (32, 32, 3)
FRONT_IMAGE = np.full(IMAGE_SHAPE, (40, 80, 120), dtype=np.uint8)
ADDED_IMAGE = np.full(IMAGE_SHAPE, (160, 120, 80), dtype=np.uint8)


@dataclass
class ChangeImageStreamsStep(ObservationProcessorStep):
    operation: str

    def observation(self, observation):
        if self.operation == "drop":
            observation.pop("front")
        elif self.operation == "add":
            observation["processed"] = ADDED_IMAGE
        return observation

    def transform_features(self, features):
        observations = features[PipelineFeatureType.OBSERVATION].copy()
        if self.operation == "drop":
            observations.pop("front")
        elif self.operation == "add":
            observations["processed"] = IMAGE_SHAPE
        return {**features, PipelineFeatureType.OBSERVATION: observations}


@pytest.mark.parametrize("video", [False, True], ids=["image", "video"])
@pytest.mark.parametrize(
    "camera,operation,image_keys",
    [
        (False, "identity", []),
        (True, "identity", ["front"]),
        (True, "add", ["front", "processed"]),
        (False, "add", ["processed"]),
        (True, "drop", []),
    ],
    ids=["no-camera", "camera", "two-streams", "processor-only", "removed-camera"],
)
def test_record_and_resume_image_streams(tmp_path, monkeypatch, video, camera, operation, image_keys):
    robot_cfg = MicroDuckConfig(
        calibration_dir=tmp_path / "calibration",
        use_camera=camera,
        camera_height=IMAGE_SHAPE[0],
        camera_width=IMAGE_SHAPE[1],
    )
    robot = MicroDuck(robot_cfg)
    assert not hasattr(robot, "cameras")
    robot._latest_frame = FRONT_IMAGE
    monkeypatch.setattr(robot, "connect", lambda: setattr(robot, "_is_connected", True))
    monkeypatch.setattr(robot._cmd, "request", lambda *args, **kwargs: {"accepted": True})
    monkeypatch.setattr(robot._cmd, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr("lerobot.scripts.lerobot_record.make_robot_from_config", lambda cfg: robot)
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_record.init_keyboard_listener",
        lambda: (None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}),
    )
    action = dict.fromkeys(robot.action_features, 0.0)
    action["vx"] = 0.1
    monkeypatch.setattr(MockTeleop, "get_action", lambda self: action.copy())

    writer_settings = []
    start_image_writer = DatasetWriter.start_image_writer

    def track_image_writer(writer, num_processes, num_threads):
        writer_settings.append((num_processes, num_threads))
        start_image_writer(writer, num_processes, num_threads)

    monkeypatch.setattr(DatasetWriter, "start_image_writer", track_image_writer)
    observation_processor = make_default_processors()[2]
    observation_processor.steps = [ChangeImageStreamsStep(operation)]
    cfg = RecordConfig(
        robot=robot_cfg,
        teleop=MockTeleopConfig(calibration_dir=tmp_path / "teleop-calibration"),
        dataset=DatasetRecordConfig(
            repo_id="lerobot/microduck-image-streams-test",
            root=tmp_path / "dataset",
            no_stamp=True,
            single_task="Test image recording",
            fps=10,
            episode_time_s=0.1,
            reset_time_s=0,
            num_episodes=1,
            push_to_hub=False,
            video=video,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=18),
            num_image_writer_processes=0 if image_keys else 1,
            num_image_writer_threads_per_camera=2,
        ),
        play_sounds=False,
    )

    for resume in (False, True):
        cfg.resume = resume
        # Keep resume local, as in the existing recording tests.
        with (
            patch("lerobot.datasets.dataset_metadata.get_safe_version", return_value="v3.0"),
            patch("lerobot.datasets.dataset_metadata.snapshot_download", return_value=str(cfg.dataset.root)),
        ):
            dataset = record(cfg, robot_observation_processor=observation_processor)
            reloaded = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, video_backend="pyav")
        assert dataset.num_episodes == (2 if resume else 1)
        assert dataset.num_frames == (2 if resume else 1)
        assert reloaded.num_episodes == dataset.num_episodes
        assert reloaded.num_frames == dataset.num_frames
        expected_features = {f"observation.images.{key}" for key in image_keys}
        assert set(reloaded.meta.camera_keys) == expected_features
        sample = reloaded[reloaded.num_frames - 1]
        torch.testing.assert_close(sample["action"], torch.tensor(list(action.values())))
        for key in image_keys:
            feature = f"observation.images.{key}"
            assert reloaded.features[feature]["dtype"] == ("video" if video else "image")
            assert tuple(sample[feature].shape) == (3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
            image = FRONT_IMAGE if key == "front" else ADDED_IMAGE
            expected = torch.tensor(image[0, 0].copy(), dtype=torch.float32) / 255
            torch.testing.assert_close(sample[feature].mean(dim=(1, 2)), expected, atol=7 / 255, rtol=0)

    expected_settings = [(0, 2 * len(image_keys))] * 2 if image_keys else []
    assert writer_settings == expected_settings
