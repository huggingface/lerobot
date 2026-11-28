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

import importlib

from .controller import RobotController

# The controllers of each robot type, by class name, with the module defining each. A module is imported
# only when its controller is selected, so a robot does not pull in every controller's dependencies.
CONTROLLERS: dict[str, dict[str, str]] = {
    "unitree_g1": {
        "GrootLocomotionController": "lerobot.controllers.unitree_g1.gr00t_locomotion",
        "HolosomaLocomotionController": "lerobot.controllers.unitree_g1.holosoma_locomotion",
        "SonicWholeBodyController": "lerobot.controllers.unitree_g1.sonic_whole_body",
    },
}


def make_robot_controller(robot_type: str, name: str) -> RobotController:
    """Instantiate the controller `name` of a robot type, e.g. `make_robot_controller("unitree_g1", "SonicWholeBodyController")`."""
    controllers = CONTROLLERS.get(robot_type)
    if controllers is None:
        raise ValueError(f"{robot_type} has no controllers. Robots with controllers: {sorted(CONTROLLERS)}")
    module_path = controllers.get(name)
    if module_path is None:
        raise ValueError(f"Unknown controller for {robot_type}: {name!r}. Available: {list(controllers)}")
    return getattr(importlib.import_module(module_path), name)()
