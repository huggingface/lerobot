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

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .shared_bus_mode.component_assembly import SharedBusConfig, SharedBusDeviceConfig
from .sub_robots.biwheel_base.config_biwheel_base import BiWheelBaseConfig
from .sub_robots.lekiwi_base.config import LeKiwiBaseConfig


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLeRobotConfig(RobotConfig):
    left_arm: dict[str, Any] = field(default_factory=dict)
    right_arm: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    mount: dict[str, Any] = field(default_factory=dict)
    shared_buses: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.left_arm = self._normalize_component_dict(self.left_arm)
        self.right_arm = self._normalize_component_dict(self.right_arm)
        self.base = self._normalize_base_dict(self.base)
        self.mount = self._normalize_component_dict(self.mount)

        self.shared_buses_config, self.component_ports = self._coerce_shared_buses()
        self._validate_component_ports()

    def _normalize_component_dict(self, cfg: dict[str, Any] | Any) -> dict[str, Any]:
        if not cfg:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        if is_dataclass(cfg):
            return asdict(cfg)
        return dict(cfg)

    def _normalize_base_dict(self, cfg: dict[str, Any] | Any) -> dict[str, Any]:
        if not cfg:
            return {}

        if isinstance(cfg, dict):
            data = dict(cfg)
        elif isinstance(cfg, LeKiwiBaseConfig):
            data = asdict(cfg)
            data.setdefault("type", "lekiwi_base")
        elif isinstance(cfg, BiWheelBaseConfig):
            data = asdict(cfg)
            data.setdefault("type", "biwheel_base")
        elif is_dataclass(cfg):
            data = asdict(cfg)
        else:
            data = dict(cfg)

        if "type" not in data:
            raise ValueError("Base configuration must specify a 'type' field (e.g. 'lekiwi_base').")
        return data

    def _coerce_shared_buses(self) -> tuple[dict[str, SharedBusConfig], dict[str, str]]:
        if not self.shared_buses:
            raise ValueError("`shared_buses` must be provided for XLeRobot.")

        coerced: dict[str, SharedBusConfig] = {}
        component_ports: dict[str, str] = {}
        for name, value in self.shared_buses.items():
            if isinstance(value, SharedBusConfig):
                bus_cfg = value
            else:
                port = value.get("port")
                if not port:
                    raise ValueError(f"Shared bus '{name}' is missing required field 'port'.")
                components_data = value.get("components", [])
                components = [
                    device if isinstance(device, SharedBusDeviceConfig) else SharedBusDeviceConfig(**device)
                    for device in components_data
                ]
                bus_cfg = SharedBusConfig(
                    port=port,
                    components=components,
                    handshake_on_connect=value.get("handshake_on_connect", True),
                )

            if not bus_cfg.components:
                raise ValueError(f"Shared bus '{name}' must list at least one component.")

            for device in bus_cfg.components:
                previous = component_ports.get(device.component)
                if previous and previous != bus_cfg.port:
                    raise ValueError(
                        f"Component '{device.component}' is assigned to multiple shared buses "
                        f"({previous} vs {bus_cfg.port})."
                    )
                component_ports[device.component] = bus_cfg.port

            coerced[name] = bus_cfg

        return coerced, component_ports

    def _validate_component_ports(self) -> None:
        for component_name, spec in (
            ("left_arm", self.left_arm),
            ("right_arm", self.right_arm),
            ("base", self.base),
            ("mount", self.mount),
        ):
            if spec and component_name not in getattr(self, "component_ports", {}):
                raise ValueError(
                    f"Component '{component_name}' is configured but missing from `shared_buses`. "
                    "Declare a shared bus entry that references it."
                )
