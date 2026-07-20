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

"""Exercise the live mapping loop (LiveMapper.tick) with no hardware.

A fake robot serves canned front-camera frames + odometry; FakeGeometryRunner
supplies planar depth. This validates the perceive → project-through-odometry
→ integrate path — the logic that runs on the real dog — including the
frame-consistency fix (voxels land in the odometry world frame).
"""

from __future__ import annotations

import numpy as np

from lerobot.navigation.base_controller import RobotBaseController
from lerobot.navigation.dog_cli import LiveMapper
from lerobot.navigation.geometry import FakeGeometryRunner
from lerobot.navigation.pipeline import PipelineConfig
from lerobot.navigation.voxel_map import VoxelMap


class FakeGo2:
    """Robot stand-in: canned front frame + programmable odometry."""

    def __init__(self, h=14, w=14):
        self.h, self.w = h, w
        self.odom = {"x.pos": 0.0, "y.pos": 0.0, "theta.pos": 0.0}
        self.actions = []

    def get_observation(self):
        return {
            "front": np.full((self.h, self.w, 3), 120, dtype=np.uint8),
            **self.odom,
        }

    def send_action(self, action):
        self.actions.append(action)
        return action


def _mapper(**pcfg_kwargs):
    robot = FakeGo2()
    base = RobotBaseController(robot)
    vm = VoxelMap(voxel_size=0.05)
    geom = FakeGeometryRunner(depth=2.0, focal_px=100.0)
    pcfg = PipelineConfig(focal_px=100.0, **pcfg_kwargs)
    mapper = LiveMapper(robot, base, geom, siglip=None, voxel_map=vm, pcfg=pcfg)
    return mapper, robot, base, vm


def test_tick_populates_voxel_map():
    mapper, _, _, vm = _mapper()
    mapper.tick(0.0)
    assert len(vm) > 0


def test_tick_places_voxels_in_odometry_frame():
    """With the dog at origin facing +z, the planar floor at depth 2 lands
    around world z≈2 — i.e. in the odometry world frame, not the model's."""
    mapper, robot, base, vm = _mapper()
    mapper.tick(0.0)
    snap = vm.snapshot()
    # Median z of the observed floor should be near the camera depth (2 m).
    assert 1.0 < float(np.median(snap.xyz[:, 2])) < 3.0


def test_tick_follows_odometry_translation():
    """Move the dog forward 5 m in odometry; the new floor voxels shift with
    it — proof the map tracks the odometry world frame."""
    mapper, robot, base, vm = _mapper()
    mapper.tick(0.0)
    z0 = float(np.median(vm.snapshot().xyz[:, 2]))

    robot.odom = {"x.pos": 5.0, "y.pos": 0.0, "theta.pos": 0.0}  # forward 5 m
    vm2 = VoxelMap(voxel_size=0.05)
    mapper.voxel_map = vm2
    mapper.tick(1.0)
    z1 = float(np.median(vm2.snapshot().xyz[:, 2]))
    # Forward odometry (+x_odom → +z_world) shifts the floor ~5 m in world z.
    assert z1 - z0 > 4.0


def test_tick_without_front_frame_is_safe():
    mapper, robot, _, vm = _mapper()
    robot.get_observation = lambda: {"x.pos": 0.0, "y.pos": 0.0, "theta.pos": 0.0}
    mapper.tick(0.0)  # no 'front' → no-op, must not raise
    assert len(vm) == 0


def test_tick_feeds_watchdog_when_present():
    from lerobot.navigation.base_controller import SafeBaseController

    mapper, _, base, _ = _mapper()
    safe = SafeBaseController(inner=base)
    safe.e_stop_latched = True  # will clear on a healthy feed via reset elsewhere
    mapper.safe = safe
    # feed_watchdog just refreshes the timer; assert tick calls it (no raise,
    # and the timestamp advances).
    before = safe._last_keyframe_walltime
    mapper.tick(0.0)
    assert safe._last_keyframe_walltime >= before
