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

"""Per-episode servo settings recorded beside each episode.

``meta/info.json`` does not carry a dataset-wide calibration. The episode column
is the only record.
"""

from __future__ import annotations

import contextlib
import json
from functools import cache
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _rustypot_available

if TYPE_CHECKING or _rustypot_available:
    import rustypot
else:
    rustypot = None

SERVO_SETTINGS_COLUMN = "servo_settings"

_CALIBRATION_KEYS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")

# Commands and the torque switch change during a recording. A stored copy would
# differ on every episode and would not describe how the servo was set up.
_COMMAND_REGISTERS = frozenset({"goal_position", "goal_time", "goal_speed", "goal_velocity", "torque_enable"})

# When two names share an address, keep the one LeRobot already uses.
_PREFERRED_NAMES = frozenset(
    {
        "baud_rate",
        "min_position_limit",
        "max_position_limit",
        "homing_offset",
        "operating_mode",
    }
)


def dataset_records_servo_settings(dataset: Any) -> bool:
    """Whether this dataset's episode table should gain a ``servo_settings`` value.

    A dataset with no episodes yet is new, so it records them. A dataset whose
    episode files already lack the column keeps that format.
    """
    episodes = dataset.meta.episodes
    if episodes is None or dataset.meta.total_episodes == 0:
        return True
    return SERVO_SETTINGS_COLUMN in episodes.column_names


def episodes_have_servo_settings(meta: Any) -> bool:
    """True when this metadata's episode table has the column and at least one episode."""
    episodes = meta.episodes
    if episodes is None or meta.total_episodes == 0:
        return False
    return SERVO_SETTINGS_COLUMN in episodes.column_names


@cache
def setting_register_names(model: str) -> tuple[str, ...]:
    """Setting registers for ``model``, one name per address, in address order.

    Read-only live values (position, speed, load, voltage, temperature, current)
    are left out. So are goal commands and the torque switch. Acceleration is kept.
    An unknown model, or a missing rustypot install, yields no registers.
    """
    if not _rustypot_available or rustypot is None:
        return ()
    found = rustypot.find_model(model)
    if found is None:
        return ()
    chosen: dict[int, str] = {}
    for register in found[1].registers():
        name = register.name
        if not str(register.access).endswith("ReadWrite"):
            continue
        if name in _COMMAND_REGISTERS:
            continue
        current = chosen.get(register.addr)
        if current is None or (name in _PREFERRED_NAMES and current not in _PREFERRED_NAMES):
            chosen[register.addr] = name
    return tuple(chosen[addr] for addr in sorted(chosen))


def _calibration_fields(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    try:
        return {
            key: int(value[key] if isinstance(value, dict) else getattr(value, key))
            for key in _CALIBRATION_KEYS
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _register_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_servo_settings(robot: Any) -> dict[str, Any]:
    """Servo model, calibration, and setting registers for the connected follower.

    Call this at an episode boundary. It is not for the control loop: one read
    of each setting register, then the values are stored with the episode.
    A register the bus cannot read is omitted. The other registers are kept.
    """
    bus = getattr(robot, "bus", None)
    motors = getattr(bus, "motors", None) or {}
    calibration = getattr(robot, "calibration", None)
    if not isinstance(calibration, dict):
        calibration = {}

    by_model: dict[str, list[str]] = {}
    record: dict[str, Any] = {"motors": {}}
    for name, motor in motors.items():
        model = getattr(motor, "model", None) or ""
        entry: dict[str, Any] = {"model": model}
        fields = _calibration_fields(calibration.get(name))
        if fields is not None:
            entry["calibration"] = fields
        entry["registers"] = {}
        record["motors"][name] = entry
        by_model.setdefault(model, []).append(name)

    if bus is None:
        return record

    for model, names in by_model.items():
        for register in setting_register_names(model):
            values = None
            with contextlib.suppress(Exception):
                values = bus.sync_read(register, names, normalize=False)
            if not isinstance(values, dict):
                continue
            for motor_name, value in values.items():
                stored = _register_value(value)
                if stored is None or motor_name not in record["motors"]:
                    continue
                record["motors"][motor_name]["registers"][register] = stored
    for entry in record["motors"].values():
        if not entry["registers"]:
            entry.pop("registers")
    return record


def servo_settings_json(robot: Any) -> str:
    """``read_servo_settings`` encoded for the episode parquet column."""
    return json.dumps(read_servo_settings(robot), sort_keys=True, separators=(",", ":"))
