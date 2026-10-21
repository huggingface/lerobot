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

"""Each simulator's control conventions, in one place for the env server and in-process environments.

Importable without the simulators themselves, so a client of a served simulator can resolve them too.
A configuration's explicit values win; the ones left null come from here.

``robot_type`` names the simulated body, recorded as a sim robot's dataset ``robot_type``: the one a
lerobot robot class records on hardware when the body has one (``so_follower``), else the body's short
name. Left empty, a sim robot records ``sim``.
"""

from typing import Any

_LIBERO_STATE = (
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.axis_angle.x",
    "eef.axis_angle.y",
    "eef.axis_angle.z",
    "gripper.left.pos",
    "gripper.right.pos",
)
_LIBERO_ACTION = (
    "eef.delta.x",
    "eef.delta.y",
    "eef.delta.z",
    "eef.delta.axis_angle.x",
    "eef.delta.axis_angle.y",
    "eef.delta.axis_angle.z",
    "gripper.command",
)
_LIBERO = {
    "fps": 20,
    "control": "delta",
    "state_names": _LIBERO_STATE,
    "action_names": _LIBERO_ACTION,
    "gripper_indices": (6,),
    "robot_type": "panda",
}
_VLABENCH_NAMES = ("eef.x", "eef.y", "eef.z", "eef.euler.x", "eef.euler.y", "eef.euler.z", "gripper.opening")


def _joints(count: int) -> tuple[str, ...]:
    return tuple(f"command_{i}.pos" for i in range(count))


def _robotwin(kwargs: dict[str, Any]) -> dict[str, Any]:
    # Joint mode commands and observes the 14 joints; ee mode commands 16-D poses, left unnamed.
    names = _joints(14) if kwargs.get("action_mode", "joint") == "joint" else ()
    return {
        "fps": 25,
        "control": "position",
        "state_names": _joints(14),
        "action_names": names,
        "robot_type": "aloha_agilex",
    }


def _robomme(kwargs: dict[str, Any]) -> dict[str, Any]:
    joints = kwargs.get("action_space_type", "joint_angle") == "joint_angle"
    return {
        "fps": 10,
        "control": "position" if joints else "eef_pose",
        "state_names": _joints(8),
        "action_names": _joints(8) if joints else (),
    }


CONVENTIONS: dict[str, Any] = {
    "libero": _LIBERO,
    "libero_plus": _LIBERO,
    "robocerebra": _LIBERO,
    "metaworld": {
        "fps": 80,
        "control": "delta",
        "state_names": ("eef.x", "eef.y", "eef.z", "gripper.opening"),
        "action_names": ("eef.delta.x", "eef.delta.y", "eef.delta.z", "gripper.command"),
        "gripper_indices": (3,),
        "robot_type": "sawyer",
    },
    "robocasa": {
        "fps": 20,
        "control": "delta",
        "state_names": (
            *(f"base.position.{a}" for a in "xyz"),
            *(f"base.quaternion.{a}" for a in "xyzw"),
            *(f"eef.position.{a}" for a in "xyz"),
            *(f"eef.quaternion.{a}" for a in "xyzw"),
            "gripper.left.pos",
            "gripper.right.pos",
        ),
        "action_names": (
            *(f"base.motion.{a}" for a in ("x", "y", "z", "yaw")),
            "controller.mode",
            *(f"eef.delta.{a}" for a in "xyz"),
            *(f"eef.delta.rotation.{a}" for a in "xyz"),
            "gripper.command",
        ),
        "gripper_indices": (11,),
        "command_retention_indices": (4,),
        "robot_type": "panda_omron",
    },
    "vlabench": {
        "fps": 10,
        "control": "eef_pose",
        "state_names": _VLABENCH_NAMES,
        "action_names": _VLABENCH_NAMES,
        "gripper_indices": (6,),
    },
    "robotwin": _robotwin,
    "robomme": _robomme,
}

# What a simulator without declared conventions gets.
DEFAULTS: dict[str, Any] = {
    "fps": 30,
    "control": "position",
    "state_names": (),
    "action_names": (),
    "gripper_indices": (),
    "command_retention_indices": (),
    "robot_type": "",
}


def resolve(sim: dict[str, Any]) -> dict[str, Any]:
    """Fill a simulator configuration's null settings from its conventions, else the defaults.

    ``semantics`` defaults to ``<type>-<control>-v1``. EnvHub repositories declare theirs in ``env.py``.
    """
    if sim.get("type") == "envhub":
        from lerobot.sims.native.envhub import resolve_envhub_sim

        sim = resolve_envhub_sim(sim)
    declared = CONVENTIONS.get(sim.get("type", ""), {})
    if callable(declared):
        declared = declared(sim.get("kwargs") or {})
    resolved = dict(sim)
    for key, default in DEFAULTS.items():
        if resolved.get(key) is None:
            resolved[key] = declared.get(key, default)
    if resolved.get("semantics") is None:
        resolved["semantics"] = f"{resolved['type']}-{resolved['control'].replace('_', '-')}-v1"
    return resolved


def _plain(value: Any) -> Any:
    """Compare YAML, wire and dataclass values alike: tuples as lists, recursively."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def served_differences(served: dict[str, Any], expected: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Compare a server's resolved simulator with the one a client expects.

    Returns the differences that change the environment (type, task, control conventions), then the
    kwargs that differ. Only the settings ``expected`` sets are compared; EnvHub settings left null are
    not resolved here, which would run the repository's code in the client.
    """
    if expected.get("type") != "envhub":
        expected = resolve(expected)
    errors, warnings = [], []
    for key, value in expected.items():
        if key == "kwargs" or value is None or _plain(value) == _plain(served.get(key)):
            continue
        errors.append(f"{key}: server {served.get(key)!r}, expected {value!r}")
    served_kwargs = served.get("kwargs") or {}
    for key, value in (expected.get("kwargs") or {}).items():
        # The env server loads an EnvHub repository from where its image put it.
        if key not in ("hub_path", "trust_remote_code") and _plain(value) != _plain(served_kwargs.get(key)):
            warnings.append(f"kwargs.{key}: server {served_kwargs.get(key)!r}, expected {value!r}")
    return errors, warnings
