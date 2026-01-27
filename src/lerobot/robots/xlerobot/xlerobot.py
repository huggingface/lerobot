#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

# XLeRobot integration based on
#
#   https://github.com/Astera-org/brainbot
#   https://github.com/Vector-Wangel/XLeRobot
#   https://github.com/bingogome/lerobot

import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError

from ..robot import Robot
from ..so_follower.config_so_follower import SO101FollowerConfig
from ..so_follower.so_follower import SO101Follower
from .config_xlerobot import XLeRobotConfig
from .shared_bus_mode.shared_bus import SharedComponentAttachment, build_shared_bus_group
from .sub_robots.biwheel_base.biwheel_feetech import BiwheelFeetech
from .sub_robots.biwheel_base.biwheel_odrive import BiwheelODrive
from .sub_robots.biwheel_base.config_biwheel_base import (
    BiwheelFeetechConfig,
    BiwheelODriveConfig,
)
from .sub_robots.lekiwi_base.config import LeKiwiBaseConfig
from .sub_robots.lekiwi_base.lekiwi_base import LeKiwiBase
from .sub_robots.xlerobot_mount.config import XLeRobotMountConfig
from .sub_robots.xlerobot_mount.xlerobot_mount import XLeRobotMount

logger = logging.getLogger(__name__)


class XLeRobot(Robot):
    """Combined platform: bimanual SO-101 follower arms mounted on a configurable mobile base."""

    config_class = XLeRobotConfig
    name = "xlerobot"

    def __init__(self, config: XLeRobotConfig):
        super().__init__(config)
        self.config = config
        self._component_ports = dict(config.component_ports)

        self.left_arm = self._build_arm("left_arm")
        self.right_arm = self._build_arm("right_arm")
        self.base = self._build_base_robot()
        self.mount = self._build_mount_robot()
        self.camera_configs = dict(config.cameras)
        self.cameras = make_cameras_from_configs(self.camera_configs)
        self._last_camera_obs: dict[str, np.ndarray] = {
            name: self._make_blank_camera_obs(name) for name in self.cameras
        }
        self._shared_buses: dict[str, Any] = {}
        self._setup_shared_buses()

    def _build_arm(self, component_name: str) -> SO101Follower | None:
        spec = getattr(self.config, component_name, {}) or {}
        if not spec:
            return None
        cfg_dict = self._prepare_component_spec(component_name, spec)
        arm_config = SO101FollowerConfig(**cfg_dict)
        return SO101Follower(arm_config)

    def _build_base_robot(self) -> Robot | None:
        spec = getattr(self.config, "base", {}) or {}
        if not spec:
            return None
        base_type = spec.get("type")
        if base_type == "lekiwi_base":
            cfg_cls = LeKiwiBaseConfig
            robot_cls = LeKiwiBase
        elif base_type in ("biwheel_base", "biwheel_feetech", "biwheel_odrive"):
            driver = spec.get("driver")
            use_odrive = base_type == "biwheel_odrive" or driver == "odrive"
            if use_odrive:
                cfg_cls = BiwheelODriveConfig
                robot_cls = BiwheelODrive
            else:
                cfg_cls = BiwheelFeetechConfig
                robot_cls = BiwheelFeetech
        else:
            raise ValueError(
                "Base configuration must include a supported 'type' "
                "(lekiwi_base, biwheel_base, biwheel_feetech, or biwheel_odrive)."
            )

        spec_copy = dict(spec)
        spec_copy.pop("type", None)
        spec_copy.pop("driver", None)
        if use_odrive:
            spec_copy.setdefault("shared_bus", False)
        cfg_dict = self._prepare_component_spec("base", spec_copy)
        base_config = cfg_cls(**cfg_dict)
        return robot_cls(base_config)

    def _build_mount_robot(self) -> XLeRobotMount | None:
        spec = getattr(self.config, "mount", {}) or {}
        if not spec:
            return None
        cfg_dict = self._prepare_component_spec("mount", spec)
        mount_config = XLeRobotMountConfig(**cfg_dict)
        return XLeRobotMount(mount_config)

    def _make_blank_camera_obs(self, cam_key: str) -> np.ndarray:
        cam_config = self.camera_configs.get(cam_key)
        height = getattr(cam_config, "height", None) or 1
        width = getattr(cam_config, "width", None) or 1
        return np.zeros((height, width, 3), dtype=np.uint8)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        camera_features: dict[str, tuple[int, int, int]] = {}
        for cam_key, cam_config in self.camera_configs.items():
            height = getattr(cam_config, "height", None) or 1
            width = getattr(cam_config, "width", None) or 1
            camera_features[cam_key] = (height, width, 3)
        return camera_features

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        if self.left_arm:
            features.update(self._prefixed_features(self.left_arm.observation_features, "left_"))
        if self.right_arm:
            features.update(self._prefixed_features(self.right_arm.observation_features, "right_"))
        if self.base:
            features.update(self.base.observation_features)
        if self.mount:
            features.update(self.mount.observation_features)
        features.update(self._cameras_ft)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        if self.left_arm:
            features.update(self._prefixed_features(self.left_arm.action_features, "left_"))
        if self.right_arm:
            features.update(self._prefixed_features(self.right_arm.action_features, "right_"))
        if self.base:
            features.update(self.base.action_features)
        if self.mount:
            features.update(self.mount.action_features)
        return features

    @property
    def is_connected(self) -> bool:
        components_connected = all(
            comp.is_connected
            for comp in (self.left_arm, self.right_arm, self.base, self.mount)
            if comp is not None
        )
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return components_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.left_arm:
            self.left_arm.connect(calibrate=calibrate)
        if self.right_arm:
            self.right_arm.connect(calibrate=calibrate)
        if self.base:
            handshake = getattr(self.base.config, "handshake_on_connect", True)
            self.base.connect(calibrate=calibrate, handshake=handshake)
        if self.mount:
            self.mount.connect(calibrate=calibrate)
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return all(
            comp.is_calibrated
            for comp in (self.left_arm, self.right_arm, self.base, self.mount)
            if comp is not None
        )

    def calibrate(self) -> None:
        logger.info("Calibrating XLeRobot components")
        if self.left_arm:
            self.left_arm.calibrate()
        if self.right_arm:
            self.right_arm.calibrate()
        if self.base:
            self.base.calibrate()
        if self.mount:
            self.mount.calibrate()

    def configure(self) -> None:
        if self.left_arm:
            self.left_arm.configure()
        if self.right_arm:
            self.right_arm.configure()
        if self.base:
            self.base.configure()
        if self.mount:
            self.mount.configure()

    def setup_motors(self) -> None:
        if self.left_arm and hasattr(self.left_arm, "setup_motors"):
            self.left_arm.setup_motors()
        if self.right_arm and hasattr(self.right_arm, "setup_motors"):
            self.right_arm.setup_motors()
        if self.base and hasattr(self.base, "setup_motors"):
            self.base.setup_motors()
        if self.mount and hasattr(self.mount, "setup_motors"):
            self.mount.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("XLeRobot is not connected.")

        obs = {}
        if self.left_arm:
            obs.update(self._prefixed_obs(self.left_arm.get_observation(), "left_"))
        if self.right_arm:
            obs.update(self._prefixed_obs(self.right_arm.get_observation(), "right_"))
        if self.base:
            obs.update(self.base.get_observation())
        if self.mount:
            obs.update(self.mount.get_observation())
        for name, cam in self.cameras.items():
            try:
                frame = cam.async_read()
            except Exception as exc:
                logger.warning("Failed to read camera %s (%s); using cached frame", name, exc)
                frame = self._last_camera_obs.get(name)
                if frame is None:
                    frame = self._make_blank_camera_obs(name)
                    self._last_camera_obs[name] = frame
            else:
                self._last_camera_obs[name] = frame
            obs[name] = frame
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("XLeRobot is not connected.")

        sent: dict[str, Any] = {}

        if self.left_arm:
            left_action = self._extract_prefixed_action(action, "left_")
            sent.update(self._prefixed_obs(self.left_arm.send_action(left_action), "left_"))

        if self.right_arm:
            right_action = self._extract_prefixed_action(action, "right_")
            sent.update(self._prefixed_obs(self.right_arm.send_action(right_action), "right_"))

        if self.base:
            base_action = {}
            for key in ("x.vel", "y.vel", "theta.vel"):
                if key in action:
                    base_action[key] = action[key]
                elif f"base.{key}" in action:
                    base_action[key] = action[f"base.{key}"]
            sent.update(self.base.send_action(base_action))

        if self.mount:
            mount_keys = set(self.mount.action_features.keys())
            mount_action = {key: value for key, value in action.items() if key in mount_keys}
            if mount_action:
                sent.update(self.mount.send_action(mount_action))

        return sent

    def disconnect(self) -> None:
        for comp in (self.base, self.mount, self.left_arm, self.right_arm):
            if comp and comp.is_connected:
                self._safe_disconnect(comp)
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except DeviceNotConnectedError:
                logger.debug("Camera %s not connected during disconnect", cam, exc_info=False)
            except Exception:
                logger.warning("Failed to disconnect camera", exc_info=True)

    def _prefixed_features(self, features: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in features.items()}

    def _prefixed_obs(self, obs: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {f"{prefix}{key}": value for key, value in obs.items()}

    def _extract_prefixed_action(self, action: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {key.removeprefix(prefix): value for key, value in action.items() if key.startswith(prefix)}

    def _component_port(self, component_name: str) -> str:
        port = self._component_ports.get(component_name)
        if not port:
            raise ValueError(
                f"No shared bus provides a port for component '{component_name}'. "
                "Declare it in `shared_buses`."
            )
        return port

    def _prepare_component_spec(self, component_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(spec)
        shared_bus = cfg.pop("shared_bus", True)
        if component_name in self._component_ports:
            port = self._component_port(component_name)
            existing_port = cfg.get("port")
            if existing_port and existing_port != port:
                raise ValueError(
                    f"Component '{component_name}' specifies port '{existing_port}' but shared bus assigns '{port}'."
                )
            cfg["port"] = port
        elif shared_bus:
            raise ValueError(
                f"No shared bus provides a port for component '{component_name}'. "
                "Declare it in `shared_buses`, or set `shared_bus: false` to opt out."
            )
        if self.config.id and "id" not in cfg:
            cfg["id"] = f"{self.config.id}_{component_name}"
        if self.config.calibration_dir and cfg.get("calibration_dir") is None:
            cfg["calibration_dir"] = self.config.calibration_dir
        return cfg

    def _setup_shared_buses(self) -> None:
        if not getattr(self.config, "shared_buses_config", None):
            return

        component_map = {
            "left_arm": self.left_arm,
            "right_arm": self.right_arm,
            "base": self.base,
            "mount": self.mount,
        }

        for name, bus_cfg in self.config.shared_buses_config.items():
            attachments: list[SharedComponentAttachment] = []
            for device_cfg in bus_cfg.components:
                component = component_map.get(device_cfg.component)
                if component is None:
                    continue
                if getattr(component, "supports_shared_bus", True) is False:
                    raise ValueError(
                        f"Component '{device_cfg.component}' does not support shared buses but is "
                        "listed in `shared_buses`."
                    )
                if not hasattr(component, "bus"):
                    continue
                motor_names = tuple(component.bus.motors.keys())
                attachments.append(
                    SharedComponentAttachment(
                        name=device_cfg.component,
                        component=component,
                        motor_names=motor_names,
                        motor_id_offset=device_cfg.motor_id_offset,
                    )
                )
            if attachments:
                group, _ = build_shared_bus_group(
                    name,
                    port=bus_cfg.port,
                    attachments=attachments,
                    handshake_on_connect=bus_cfg.handshake_on_connect,
                )
                self._shared_buses[name] = group

    def _safe_disconnect(self, component: Robot) -> None:
        try:
            if hasattr(component, "stop_base"):
                component.stop_base()
            component.disconnect()
        except DeviceNotConnectedError:
            logger.debug("%s already disconnected", component, exc_info=False)
        except Exception as exc:
            detail = self._describe_component(component)
            exc_type = type(exc).__name__
            logger.warning(
                "Failed to disconnect %s (%s: %s). Details: %s",
                component,
                exc_type,
                exc,
                detail,
                exc_info=True,
            )
            print(f"[XLeRobot] Failed to disconnect {component}: {exc_type}: {exc} | {detail}")

    def _describe_component(self, component: Robot) -> str:
        parts: list[str] = [f"class={component.__class__.__name__}"]
        config = getattr(component, "config", None)
        if config:
            comp_id = getattr(config, "id", None)
            if comp_id:
                parts.append(f"id={comp_id}")
            port = getattr(config, "port", None)
            if port:
                parts.append(f"port={port}")
        bus = getattr(component, "bus", None)
        if bus:
            bus_port = getattr(bus, "port", None)
            if bus_port:
                parts.append(f"bus_port={bus_port}")
            motors = getattr(bus, "motors", None)
            if motors:
                parts.append(f"motors={list(motors.keys())}")
        return ", ".join(parts)
