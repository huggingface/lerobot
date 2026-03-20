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

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.robots.bi_so_follower_simulated import BiSOFollowerSimulated, BiSOFollowerSimulatedConfig


def _write_stub_bridge(tmp_path: Path) -> Path:
    bridge_path = tmp_path / "task2_motors_bridge.py"
    bridge_path.write_text(
        """
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass
class _State:
    qpos_deg: np.ndarray
    images: dict[str, np.ndarray]


class _Backend:
    def __init__(self, render_size):
        self.robot_dofs = 6
        self.num_arms = 2
        ctrlrange_deg = np.array(
            [
                [-120, 120],
                [-90, 90],
                [-90, 90],
                [-90, 90],
                [-180, 180],
                [-20, 120],
                [-120, 120],
                [-90, 90],
                [-90, 90],
                [-90, 90],
                [-180, 180],
                [-20, 120],
            ],
            dtype=np.float32,
        )
        self.model = SimpleNamespace(actuator_ctrlrange=np.deg2rad(ctrlrange_deg))
        self._refcount = 0

        if render_size is None:
            images = {}
        else:
            height, width = render_size
            images = {
                "camera_front": np.full((height, width, 3), 7, dtype=np.uint8),
                "camera_top": np.full((height, width, 3), 9, dtype=np.uint8),
            }

        self._state = _State(
            qpos_deg=np.array(
                [10, 20, 30, 40, 50, 15, -10, -20, -30, -40, -50, 85],
                dtype=np.float32,
            ),
            images=images,
        )

    def start(self):
        self._refcount += 1

    def stop(self):
        self._refcount = max(0, self._refcount - 1)

    def get_state(self):
        return _State(self._state.qpos_deg.copy(), {k: v.copy() for k, v in self._state.images.items()})

    def set_arm_target_deg(self, arm_index, q_deg):
        q_deg = np.asarray(q_deg, dtype=np.float32)
        start = arm_index * self.robot_dofs
        end = start + self.robot_dofs
        self._state.qpos_deg[start:end] = q_deg


class _ArmBus:
    def __init__(self, backend, arm_index):
        self.backend = backend
        self.arm_index = arm_index

    def connect(self):
        self.backend.start()

    def disconnect(self):
        self.backend.stop()

    def write(self, values):
        self.backend.set_arm_target_deg(self.arm_index, values)


def make_task2_bimanual_buses(xml_path, robot_dofs, render_size, realtime, slowmo, launch_viewer):
    del xml_path, robot_dofs, realtime, slowmo, launch_viewer
    backend = _Backend(render_size)
    buses = {"arm0": _ArmBus(backend, 0), "arm1": _ArmBus(backend, 1)}
    return backend, buses
""",
        encoding="utf-8",
    )
    return bridge_path


def test_bi_so_follower_simulated_routes_actions_and_observations(tmp_path: Path):
    bridge_path = _write_stub_bridge(tmp_path)
    xml_path = tmp_path / "lerobot_pick_place_cube.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")

    robot = BiSOFollowerSimulated(
        BiSOFollowerSimulatedConfig(
            id="sim-test",
            bridge_path=bridge_path,
            xml_path=xml_path,
            render_size=(4, 5),
            camera_names=("front", "top"),
        )
    )

    robot.connect()
    assert robot.is_connected

    observation = robot.get_observation()
    assert observation["left_shoulder_pan.pos"] == pytest.approx(10.0)
    assert observation["right_shoulder_pan.pos"] == pytest.approx(-10.0)
    assert observation["left_gripper.pos"] == pytest.approx(25.0)
    assert observation["right_gripper.pos"] == pytest.approx(75.0)
    assert observation["front"].shape == (4, 5, 3)
    assert observation["top"].shape == (4, 5, 3)

    sent_action = robot.send_action(
        {
            "left_shoulder_pan.pos": 15.0,
            "right_gripper.pos": 60.0,
        }
    )
    assert sent_action == {
        "left_shoulder_pan.pos": 15.0,
        "right_gripper.pos": 60.0,
    }

    observation = robot.get_observation()
    assert observation["left_shoulder_pan.pos"] == pytest.approx(15.0)
    assert observation["right_gripper.pos"] == pytest.approx(60.0)
    assert observation["right_shoulder_pan.pos"] == pytest.approx(-10.0)

    robot.disconnect()
    assert not robot.is_connected


def test_bi_so_follower_simulated_applies_relative_action_limits(tmp_path: Path):
    bridge_path = _write_stub_bridge(tmp_path)
    xml_path = tmp_path / "lerobot_pick_place_cube.xml"
    xml_path.write_text("<mujoco/>", encoding="utf-8")

    robot = BiSOFollowerSimulated(
        BiSOFollowerSimulatedConfig(
            id="sim-clamped",
            bridge_path=bridge_path,
            xml_path=xml_path,
            max_relative_target=5.0,
        )
    )

    robot.connect()
    sent_action = robot.send_action({"left_shoulder_pan.pos": 50.0})
    assert sent_action["left_shoulder_pan.pos"] == pytest.approx(15.0)

    observation = robot.get_observation()
    assert observation["left_shoulder_pan.pos"] == pytest.approx(15.0)
    robot.disconnect()
