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

import importlib.util
import logging
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_bi_so_follower_simulated import BiSOFollowerSimulatedConfig

logger = logging.getLogger(__name__)

ARM_PREFIXES = ("left", "right")
MOTOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
GRIPPER_INDEX = len(MOTOR_NAMES) - 1
DEFAULT_SCENE_XML = "lerobot_pick_place_cube.xml"
DEFAULT_BRIDGE_PY = "task2_motors_bridge.py"
CAMERA_ALIASES = {
    "front": ("front", "camera_front"),
    "top": ("top", "camera_top"),
}


@dataclass(frozen=True)
class _SimCamera:
    height: int
    width: int
    channels: int = 3
    is_connected: bool = True


class BiSOFollowerSimulated(Robot):
    """Bimanual SO follower robot backed by an external MuJoCo Task2 bridge."""

    config_class = BiSOFollowerSimulatedConfig
    name = "bi_so_follower_simulated"

    def __init__(self, config: BiSOFollowerSimulatedConfig):
        super().__init__(config)
        self.config = config

        self._backend: Any | None = None
        self._bridge_module: ModuleType | None = None
        self._left_bus: Any | None = None
        self._right_bus: Any | None = None
        self._is_connected = False
        self._gripper_ctrlrange_deg: list[tuple[float, float]] = []

        if config.render_size is not None:
            height, width = config.render_size
            self.cameras = {name: _SimCamera(height=height, width=width) for name in config.camera_names}
        else:
            self.cameras = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{arm_prefix}_{motor_name}.pos": float
            for arm_prefix in ARM_PREFIXES
            for motor_name in MOTOR_NAMES
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        if self.config.render_size is None:
            return {}

        height, width = self.config.render_size
        return {camera_name: (height, width, 3) for camera_name in self.cameras}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate

        bridge_module = self._load_bridge_module()
        bridge_factory = getattr(bridge_module, self.config.bridge_factory_name, None)
        if bridge_factory is None:
            raise AttributeError(
                f"Bridge module '{bridge_module.__file__}' does not define "
                f"'{self.config.bridge_factory_name}'."
            )

        xml_path = self._resolve_xml_path()
        render_size = self.config.render_size if self.cameras else None

        try:
            backend, buses = bridge_factory(
                xml_path=str(xml_path),
                robot_dofs=self.config.robot_dofs,
                render_size=render_size,
                realtime=self.config.realtime,
                slowmo=self.config.slowmo,
                launch_viewer=self.config.launch_viewer,
            )
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "mujoco":
                raise RuntimeError(
                    "MuJoCo is required for `bi_so_follower_simulated`. Install the `mujoco` Python package."
                ) from exc
            raise

        self._backend = backend
        self._left_bus = buses.get("arm0")
        self._right_bus = buses.get("arm1")
        if self._left_bus is None or self._right_bus is None:
            raise RuntimeError(
                "The Task2 bridge must expose both `arm0` and `arm1` buses for "
                "`bi_so_follower_simulated`."
            )

        self._left_bus.connect()
        self._right_bus.connect()
        self._gripper_ctrlrange_deg = self._read_gripper_ctrlrange_deg()
        self._is_connected = True
        logger.info(f"{self} connected to MuJoCo scene '{xml_path}'.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[4]

    def _package_sim_root(self) -> Path:
        return Path(__file__).resolve().parent / "mujoco"

    def _candidate_sim_roots(self) -> list[Path]:
        repo_root = self._repo_root()
        candidates: list[Path] = []

        if self.config.sim_root is not None:
            sim_root = self.config.sim_root.expanduser()
            if not sim_root.is_absolute():
                sim_root = repo_root / sim_root
            candidates.append(sim_root.resolve())

        candidates.extend(
            [
                self._package_sim_root().resolve(),
                (repo_root / "sim").resolve(),
                (repo_root.parent / "AOSH" / "lerobot" / "sim").resolve(),
            ]
        )

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path not in seen:
                deduped.append(path)
                seen.add(path)
        return deduped

    def _resolve_bridge_path(self) -> Path:
        repo_root = self._repo_root()
        explicit = self.config.bridge_path
        if explicit is not None:
            bridge_path = explicit.expanduser()
            if not bridge_path.is_absolute():
                if self.config.sim_root is not None:
                    sim_root = self.config.sim_root.expanduser()
                    if not sim_root.is_absolute():
                        sim_root = repo_root / sim_root
                    bridge_path = sim_root / bridge_path
                else:
                    bridge_path = repo_root / bridge_path
            bridge_path = bridge_path.resolve()
            if bridge_path.exists():
                return bridge_path
            raise FileNotFoundError(f"Bridge file not found: {bridge_path}")

        for sim_root in self._candidate_sim_roots():
            candidate = sim_root / DEFAULT_BRIDGE_PY
            if candidate.exists():
                return candidate

        searched = ", ".join(str(path / DEFAULT_BRIDGE_PY) for path in self._candidate_sim_roots())
        raise FileNotFoundError(
            "Could not find `task2_motors_bridge.py`. "
            f"Searched: {searched}. Set `bridge_path` or `sim_root` in the robot config."
        )

    def _resolve_xml_path(self) -> Path:
        repo_root = self._repo_root()
        explicit = self.config.xml_path
        if explicit is not None:
            xml_path = explicit.expanduser()
            if not xml_path.is_absolute():
                if self.config.sim_root is not None:
                    sim_root = self.config.sim_root.expanduser()
                    if not sim_root.is_absolute():
                        sim_root = repo_root / sim_root
                    xml_path = sim_root / xml_path
                else:
                    xml_path = repo_root / xml_path
            xml_path = xml_path.resolve()
            if xml_path.exists():
                return xml_path
            raise FileNotFoundError(f"Scene XML not found: {xml_path}")

        candidate_paths = [sim_root / DEFAULT_SCENE_XML for sim_root in self._candidate_sim_roots()]

        bridge_path = self._resolve_bridge_path()
        bridge_xml = bridge_path.parent / DEFAULT_SCENE_XML
        if bridge_xml.resolve() not in [path.resolve() for path in candidate_paths]:
            candidate_paths.append(bridge_xml)

        for candidate in candidate_paths:
            candidate = candidate.resolve()
            if candidate.exists():
                return candidate

        searched = ", ".join(str(path.resolve()) for path in candidate_paths)
        raise FileNotFoundError(
            f"Could not find `{DEFAULT_SCENE_XML}`. Searched: {searched}. Set `xml_path` in the robot config."
        )

    def _load_bridge_module(self) -> ModuleType:
        if self._bridge_module is not None:
            return self._bridge_module

        bridge_path = self._resolve_bridge_path()
        sim_root = bridge_path.parent.resolve()
        for import_root in (sim_root.parent, sim_root):
            import_root_str = str(import_root)
            if import_root_str not in sys.path:
                sys.path.insert(0, import_root_str)

        module_name = f"_lerobot_task2_bridge_{abs(hash(bridge_path))}"
        spec = importlib.util.spec_from_file_location(module_name, bridge_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load bridge module from {bridge_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self._bridge_module = module
        return module

    def _read_gripper_ctrlrange_deg(self) -> list[tuple[float, float]]:
        if self._backend is None or not hasattr(self._backend, "model"):
            return [(0.0, 100.0), (0.0, 100.0)]

        ctrlrange = np.asarray(self._backend.model.actuator_ctrlrange, dtype=float)
        ctrlrange_deg = np.rad2deg(ctrlrange)
        ranges: list[tuple[float, float]] = []
        for arm_index in range(2):
            actuator_index = arm_index * self.config.robot_dofs + GRIPPER_INDEX
            if actuator_index >= len(ctrlrange_deg):
                ranges.append((0.0, 100.0))
            else:
                low, high = ctrlrange_deg[actuator_index]
                ranges.append((float(low), float(high)))
        return ranges

    def _get_backend_state(self) -> Any:
        if self._backend is None:
            raise RuntimeError("Simulation backend is not initialized.")
        return self._backend.get_state()

    def _get_arm_sim_qpos_deg(self, arm_index: int, state: Any | None = None) -> np.ndarray:
        state = self._get_backend_state() if state is None else state
        start = arm_index * self.config.robot_dofs
        end = start + self.config.robot_dofs
        return np.asarray(state.qpos_deg[start:end], dtype=np.float32)

    def _gripper_percent_to_deg(self, arm_index: int, value: float) -> float:
        low, high = self._gripper_ctrlrange_deg[arm_index]
        if np.isclose(high, low):
            return float(low)
        bounded = float(np.clip(value, 0.0, 100.0))
        return float(low + ((bounded / 100.0) * (high - low)))

    def _gripper_deg_to_percent(self, arm_index: int, value: float) -> float:
        low, high = self._gripper_ctrlrange_deg[arm_index]
        if np.isclose(high, low):
            return 0.0
        percent = ((float(value) - low) / (high - low)) * 100.0
        return float(np.clip(percent, 0.0, 100.0))

    def _sim_to_robot_arm_units(self, arm_index: int, sim_qpos_deg: np.ndarray) -> np.ndarray:
        robot_qpos = np.asarray(sim_qpos_deg, dtype=np.float32).copy()
        robot_qpos[GRIPPER_INDEX] = self._gripper_deg_to_percent(arm_index, robot_qpos[GRIPPER_INDEX])
        return robot_qpos

    def _robot_to_sim_arm_units(self, arm_index: int, robot_qpos: np.ndarray) -> np.ndarray:
        sim_qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        sim_qpos[GRIPPER_INDEX] = self._gripper_percent_to_deg(arm_index, sim_qpos[GRIPPER_INDEX])
        return sim_qpos

    def _current_motor_positions(self) -> dict[str, float]:
        state = self._get_backend_state()
        positions: dict[str, float] = {}
        for arm_index, arm_prefix in enumerate(ARM_PREFIXES):
            robot_qpos = self._sim_to_robot_arm_units(arm_index, self._get_arm_sim_qpos_deg(arm_index, state))
            for motor_name, value in zip(MOTOR_NAMES, robot_qpos, strict=True):
                positions[f"{arm_prefix}_{motor_name}.pos"] = float(value)
        return positions

    def _clip_requested_action(self, requested_action: RobotAction, current_positions: dict[str, float]) -> dict[str, float]:
        requested_goal = {key: float(value) for key, value in requested_action.items() if key in self.action_features}
        if not requested_goal or self.config.max_relative_target is None:
            return requested_goal

        if isinstance(self.config.max_relative_target, dict):
            max_relative_target = {
                key: self.config.max_relative_target[key] for key in requested_goal if key in self.config.max_relative_target
            }
            missing_keys = set(requested_goal) - set(max_relative_target)
            if missing_keys:
                raise ValueError(
                    "`max_relative_target` is missing limits for action keys: "
                    f"{sorted(missing_keys)}"
                )
        else:
            max_relative_target = self.config.max_relative_target

        goal_present_pos = {
            key: (requested_goal[key], current_positions[key]) for key in requested_goal
        }
        return ensure_safe_goal_position(goal_present_pos, max_relative_target)

    def _camera_image(self, images: dict[str, np.ndarray], camera_name: str) -> np.ndarray:
        for candidate_name in CAMERA_ALIASES.get(camera_name, (camera_name,)):
            if candidate_name in images:
                return images[candidate_name]

        if self.config.render_size is None:
            raise KeyError(camera_name)

        height, width = self.config.render_size
        return np.zeros((height, width, 3), dtype=np.uint8)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        state = self._get_backend_state()
        observation = self._current_motor_positions()
        for camera_name in self.cameras:
            observation[camera_name] = self._camera_image(state.images, camera_name)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        requested_action = {key: value for key, value in action.items() if key in self.action_features}
        if not requested_action:
            return {}

        current_positions = self._current_motor_positions()
        safe_requested_action = self._clip_requested_action(requested_action, current_positions)

        for arm_index, arm_prefix in enumerate(ARM_PREFIXES):
            arm_goal = np.asarray(
                [current_positions[f"{arm_prefix}_{motor_name}.pos"] for motor_name in MOTOR_NAMES],
                dtype=np.float32,
            )
            arm_has_update = False
            for motor_index, motor_name in enumerate(MOTOR_NAMES):
                key = f"{arm_prefix}_{motor_name}.pos"
                if key in safe_requested_action:
                    arm_goal[motor_index] = float(safe_requested_action[key])
                    arm_has_update = True

            if not arm_has_update:
                continue

            sim_goal = self._robot_to_sim_arm_units(arm_index, arm_goal)
            if arm_index == 0:
                self._left_bus.write(sim_goal)
            else:
                self._right_bus.write(sim_goal)

        return safe_requested_action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._is_connected = False

        for bus in (self._left_bus, self._right_bus):
            if bus is None:
                continue
            try:
                bus.disconnect()
            except Exception:  # nosec B110
                pass

        self._left_bus = None
        self._right_bus = None
        self._backend = None
        logger.info(f"{self} disconnected.")
