#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging

from lerobot.scripts import lerobot_find_cameras


def test_default_cli_lists_without_connecting(monkeypatch):
    fake_cameras = [{"type": "OpenCV", "id": "/dev/video0"}]
    calls = []

    monkeypatch.setattr(
        lerobot_find_cameras,
        "find_and_print_cameras",
        lambda camera_type_filter=None: fake_cameras,
    )
    monkeypatch.setattr(
        lerobot_find_cameras,
        "save_images_from_all_cameras",
        lambda **kwargs: calls.append(kwargs),
    )

    args = lerobot_find_cameras.build_parser().parse_args(["opencv"])

    assert lerobot_find_cameras.run(args) == fake_cameras
    assert calls == []


def test_save_images_cli_uses_capture_path(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(lerobot_find_cameras.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        lerobot_find_cameras,
        "save_images_from_all_cameras",
        lambda **kwargs: calls.append(kwargs),
    )

    args = lerobot_find_cameras.build_parser().parse_args(
        ["opencv", "--save-images", "--output-dir", str(tmp_path), "--record-time-s", "2"]
    )

    assert lerobot_find_cameras.run(args) is None
    assert calls == [{"output_dir": tmp_path, "record_time_s": 2.0, "camera_type": "opencv"}]


def test_output_dir_without_save_images_warns_and_does_not_capture(monkeypatch, tmp_path, caplog):
    calls = []
    fake_cameras = [{"type": "OpenCV", "id": "/dev/video0"}]

    monkeypatch.setattr(
        lerobot_find_cameras,
        "find_and_print_cameras",
        lambda camera_type_filter=None: fake_cameras,
    )
    monkeypatch.setattr(
        lerobot_find_cameras,
        "save_images_from_all_cameras",
        lambda **kwargs: calls.append(kwargs),
    )

    args = lerobot_find_cameras.build_parser().parse_args(["opencv", "--output-dir", str(tmp_path)])

    with caplog.at_level(logging.WARNING):
        assert lerobot_find_cameras.run(args) == fake_cameras

    assert "Ignoring --output-dir" in caplog.text
    assert calls == []


def test_record_time_without_save_images_warns_and_does_not_capture(monkeypatch, caplog):
    calls = []
    fake_cameras = [{"type": "OpenCV", "id": "/dev/video0"}]

    monkeypatch.setattr(
        lerobot_find_cameras,
        "find_and_print_cameras",
        lambda camera_type_filter=None: fake_cameras,
    )
    monkeypatch.setattr(
        lerobot_find_cameras,
        "save_images_from_all_cameras",
        lambda **kwargs: calls.append(kwargs),
    )

    args = lerobot_find_cameras.build_parser().parse_args(["opencv", "--record-time-s", "2"])

    with caplog.at_level(logging.WARNING):
        assert lerobot_find_cameras.run(args) == fake_cameras

    assert "Ignoring --record-time-s" in caplog.text
    assert calls == []


def test_linux_save_images_warns_about_opencv_capture(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(lerobot_find_cameras.platform, "system", lambda: "Linux")
    monkeypatch.setattr(lerobot_find_cameras, "save_images_from_all_cameras", lambda **kwargs: None)

    args = lerobot_find_cameras.build_parser().parse_args(
        ["opencv", "--save-images", "--output-dir", str(tmp_path)]
    )

    with caplog.at_level(logging.WARNING):
        assert lerobot_find_cameras.run(args) is None

    assert "Opening OpenCV cameras on Linux may alter active camera settings" in caplog.text
