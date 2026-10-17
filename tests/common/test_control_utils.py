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

import logging
from types import SimpleNamespace

from lerobot.common.control_utils import sanity_check_camera_fps


def robot_with(**camera_fps):
    return SimpleNamespace(cameras={name: SimpleNamespace(fps=fps) for name, fps in camera_fps.items()})


def test_a_camera_slower_than_the_recording_rate_is_reported(caplog):
    with caplog.at_level(logging.WARNING, logger="lerobot.common.control_utils"):
        problems = sanity_check_camera_fps(robot_with(front=5.0, wrist=30), fps=30)

    assert len(problems) == 1
    assert "'front' runs at 5 fps" in problems[0]
    assert "recording rate of 30 fps" in problems[0]
    assert "about 83% of its frames" in problems[0]
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == problems


def test_cameras_at_or_above_the_recording_rate_pass():
    assert sanity_check_camera_fps(robot_with(front=30, wrist=60), fps=30) == []


def test_small_shortfalls_pass_and_real_ones_are_reported():
    # 29.97 is how NTSC-style cameras report 30 fps: a repeat every ~33 s does not matter.
    assert sanity_check_camera_fps(robot_with(front=29.97), fps=30) == []
    assert sanity_check_camera_fps(robot_with(front=29.0), fps=30) == []  # 96.7% of the rate
    (problem,) = sanity_check_camera_fps(robot_with(front=25), fps=30)  # a PAL camera
    assert "about 17% of its frames" in problem


def test_cameras_that_report_no_rate_are_skipped():
    assert sanity_check_camera_fps(robot_with(front=None, wrist=0), fps=30) == []


def test_a_robot_without_cameras_passes():
    assert sanity_check_camera_fps(SimpleNamespace(), fps=30) == []
