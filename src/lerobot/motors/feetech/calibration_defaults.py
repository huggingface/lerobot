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

"""
Default configuration and constants for Feetech STS/SO servo auto-calibration.

Shared by auto_calibration.py, lerobot_auto_calibrate_feetech.py, etc.
"""

from .. import Motor, MotorNormMode

# ---------------------------------------------------------------------------
# Default position ranges (range_min, range_max)
# ---------------------------------------------------------------------------
# Default mapping from SO/STS joint names to (range_min, range_max) in raw encoder values (4096 resolution).
# Joints not listed here will use the full range (0, max_res).
SO_STS_DEFAULT_RANGES: dict[str, tuple[int, int]] = {
    "shoulder_pan": (0, 4095),
    "shoulder_lift": (0, 4095),
    "elbow_flex": (0, 4095),
    "wrist_flex": (0, 4095),
    "wrist_roll": (0, 4095),
    "gripper": (0, 4095),
}


# SO 6-DOF arm joint names -> motor IDs (used for display, etc.)
SO_MOTOR_NUMBERS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

# SO 6-DOF arm joint name list (same order as SO_MOTOR_NUMBERS)
MOTOR_NAMES: list[str] = list(SO_MOTOR_NUMBERS.keys())

# SO calibration motor table (name -> Motor, normalization mode for calibration/scripts)
SO_FOLLOWER_MOTORS: dict[str, Motor] = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def motor_label(name: str) -> str:
    """Motor label for display: name(id), e.g. shoulder_pan(1)."""
    n = SO_MOTOR_NUMBERS.get(name, "")
    return f"{name}({n})" if n != "" else name


# ---------------------------------------------------------------------------
# Resolution and midpoint (4096 steps per revolution)
# ---------------------------------------------------------------------------
FULL_TURN = 4096
MID_POS = 2047
STS_HALF_TURN_RAW = 2047  # Same as MID_POS; used for centering and normalization

# Homing_Offset register uses 12-bit sign-magnitude encoding (sign_bit_index=11), range [-2047, 2047]
HOMING_OFFSET_MAX_MAG = 2047

# ---------------------------------------------------------------------------
# Calibration / measurement parameters
# ---------------------------------------------------------------------------
DEFAULT_VELOCITY_LIMIT = 1000  # Limit-seeking velocity during calibration (constant-speed mode Goal_Velocity)
DEFAULT_MAX_TORQUE = 1000  # Maximum torque (Max_Torque_Limit)
DEFAULT_TORQUE_LIMIT = 800  # Torque limit (Torque_Limit)
DEFAULT_ACCELERATION = 50  # Acceleration (matches project configure_motors)
DEFAULT_POS_SPEED = 1000  # Default speed for servo mode WritePosEx
DEFAULT_P_COEFFICIENT = 16  # PID P coefficient (matches so_follower)
DEFAULT_I_COEFFICIENT = 0  # PID I coefficient
DEFAULT_D_COEFFICIENT = 32  # PID D coefficient
DEFAULT_TIMEOUT = 20.0  # Timeout for single-direction limit seeking during calibration (seconds)
POSITION_TOLERANCE = 20  # Tolerance for reaching target position (steps)
# Stall detection AND conditions: near-zero velocity + stable position + Moving=0 (preferred over Status BIT5)
STALL_VELOCITY_THRESHOLD = (
    3  # Velocity near-zero threshold (|Present_Velocity| below this is considered stopped)
)
STALL_POSITION_DELTA_THRESHOLD = (
    3  # Position change between samples below this step count is considered stationary
)
OVERLOAD_SETTLE_TIME = 0.2  # Wait time after stall before disabling torque (seconds)
SAFE_IO_RETRIES = 5  # Number of retries for safe read/write operations
SAFE_IO_INTERVAL = 0.2  # Interval between safe read/write retries (seconds)

# ---------------------------------------------------------------------------
# Unfold parameters
# ---------------------------------------------------------------------------
DEFAULT_UNFOLD_ANGLE = 45.0  # Unfold angle (degrees)
DEFAULT_UNFOLD_TIMEOUT = 6.0  # Timeout for a single unfold movement (seconds)
UNFOLD_OVERLOAD_SETTLE = 0.3  # Wait time after stall during unfold before disabling torque (seconds)
UNFOLD_TOLERANCE_DEG = 5.0  # Unfold position tolerance: error within this many degrees is considered success

# ---------------------------------------------------------------------------
# Calibration / unfold order (SO 6-DOF arm)
# ---------------------------------------------------------------------------
CALIBRATE_FIRST: list[str] = ["shoulder_pan"]
CALIBRATE_REST: list[str] = ["wrist_roll", "gripper", "wrist_flex", "elbow_flex", "shoulder_lift"]
UNFOLD_ORDER: list[str] = ["wrist_flex", "elbow_flex", "shoulder_lift"]
