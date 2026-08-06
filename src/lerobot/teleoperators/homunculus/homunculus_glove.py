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

import logging
import threading
from collections import deque
from pprint import pformat
from typing import TYPE_CHECKING

from lerobot.motors import MotorCalibration
from lerobot.motors.motors_bus import MotorNormMode
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _serial_available, require_package

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..teleoperator import Teleoperator
from .config_homunculus import HomunculusGloveConfig
from .joints_translation import homunculus_glove_to_hope_jr_hand

logger = logging.getLogger(__name__)

LEFT_HAND_INVERSIONS = [
    "thumb_cmc",
    "index_dip",
    "middle_mcp_abduction",
    "middle_dip",
    "pinky_mcp_abduction",
    "pinky_dip",
]

RIGHT_HAND_INVERSIONS = [
    "thumb_mcp",
    "thumb_cmc",
    "thumb_pip",
    "thumb_dip",
    "index_mcp_abduction",
    # "index_dip",
    "middle_mcp_abduction",
    # "middle_dip",
    "ring_mcp_abduction",
    "ring_mcp_flexion",
    # "ring_dip",
    "pinky_mcp_abduction",
]


class HomunculusGlove(Teleoperator):
    """Homunculus Glove designed by NepYope & Hugging Face: a wearable exoskeleton glove read over a serial link.

    The glove streams raw encoder values for each finger joint continuously over a background thread;
    readings are smoothed with an exponential moving average, normalized, then remapped from glove joint
    names to HopeJR hand joint names via [`~teleoperators.homunculus.homunculus_glove_to_hope_jr_hand`]. It
    only produces actions and accepts no feedback.

    See [`~teleoperators.Teleoperator`] for the contract every method here implements.
    """

    config_class = HomunculusGloveConfig
    name = "homunculus_glove"

    def __init__(self, config: HomunculusGloveConfig):
        """Open the serial connection and set up the background reader thread.

        Args:
            config (`HomunculusGloveConfig`):
                The teleoperator's configuration. Its `port` determines what is connected and `side`
                selects which joints are inverted for the left vs. right hand.
        """
        require_package("pyserial", extra="pyserial-dep", import_name="serial")
        super().__init__(config)
        self.config = config
        self.serial = serial.Serial(config.port, config.baud_rate, timeout=1)
        self.serial_lock = threading.Lock()

        self.joints = {
            "thumb_cmc": MotorNormMode.RANGE_0_100,
            "thumb_mcp": MotorNormMode.RANGE_0_100,
            "thumb_pip": MotorNormMode.RANGE_0_100,
            "thumb_dip": MotorNormMode.RANGE_0_100,
            "index_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "index_mcp_flexion": MotorNormMode.RANGE_0_100,
            "index_dip": MotorNormMode.RANGE_0_100,
            "middle_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "middle_mcp_flexion": MotorNormMode.RANGE_0_100,
            "middle_dip": MotorNormMode.RANGE_0_100,
            "ring_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "ring_mcp_flexion": MotorNormMode.RANGE_0_100,
            "ring_dip": MotorNormMode.RANGE_0_100,
            "pinky_mcp_abduction": MotorNormMode.RANGE_M100_100,
            "pinky_mcp_flexion": MotorNormMode.RANGE_0_100,
            "pinky_dip": MotorNormMode.RANGE_0_100,
        }
        self.inverted_joints = RIGHT_HAND_INVERSIONS if config.side == "right" else LEFT_HAND_INVERSIONS

        n = 10
        # EMA parameters ---------------------------------------------------
        self.n: int = n
        self.alpha: float = 2 / (n + 1)
        # one deque *per joint* so we can inspect raw history if needed
        self._buffers: dict[str, deque[int]] = {joint: deque(maxlen=n) for joint in self.joints}
        # running EMA value per joint – lazily initialised on first read
        self._ema: dict[str, float | None] = dict.fromkeys(self._buffers)

        self._state: dict[str, float] | None = None
        self.new_state_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True, name=f"{self} _read_loop")
        self.state_lock = threading.Lock()

    @property
    def action_features(self) -> dict:
        """The glove's raw per-joint positions, before remapping to HopeJR hand joint names.

        Returns:
            `dict`: `"<joint>.pos"` keys mapped to `float`, one per entry in `self.joints`.
        """
        return {f"{joint}.pos": float for joint in self.joints}

    @property
    def feedback_features(self) -> dict:
        """This glove accepts no feedback.

        Returns:
            `dict`: Always empty.
        """
        return {}

    @property
    def is_connected(self) -> bool:
        """Same as [`~teleoperators.Teleoperator.is_connected`].

        The serial port is open and the background reader thread is alive.
        """
        with self.serial_lock:
            return self.serial.is_open and self.thread.is_alive()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open the serial port, start the background reader thread, and wait for the first reading.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Whether to run calibration when no calibration file exists yet. Calibration is
                interactive and prompts on stdin.

        Raises:
            TimeoutError: If no state is received from the glove within 2 seconds of starting.
        """
        if not self.serial.is_open:
            self.serial.open()
        self.thread.start()

        # wait for the thread to ramp up & 1st state to be ready
        if not self.new_state_event.wait(timeout=2):
            raise TimeoutError(f"{self}: Timed out waiting for state after 2s.")

        if not self.is_calibrated and calibrate:
            self.calibrate()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Whether a calibration file has been saved for this glove.

        Returns:
            `bool`: `True` if the calibration file exists on disk.
        """
        return self.calibration_fpath.is_file()

    def calibrate(self) -> None:
        """Interactively record each finger's range of motion and save it as the glove's calibration.

        Prompts the operator to move each finger through its full range, one finger at a time, then
        persists the observed min/max encoder values to the calibration file.
        """
        range_mins, range_maxes = {}, {}
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            print(
                f"\nMove {finger} through its entire range of motion."
                "\nRecording positions. Press ENTER to stop..."
            )
            finger_joints = [joint for joint in self.joints if joint.startswith(finger)]
            finger_mins, finger_maxes = self._record_ranges_of_motion(finger_joints)
            range_mins.update(finger_mins)
            range_maxes.update(finger_maxes)

        self.calibration = {}
        for id_, joint in enumerate(self.joints):
            self.calibration[joint] = MotorCalibration(
                id=id_,
                drive_mode=1 if joint in self.inverted_joints else 0,
                homing_offset=0,
                range_min=range_mins[joint],
                range_max=range_maxes[joint],
            )

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    # TODO(Steven): This function is copy/paste from the `HomunculusArm` class. Consider moving it to an utility to reduce duplicated code.
    def _record_ranges_of_motion(
        self, joints: list[str] | None = None, display_values: bool = True
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Interactively record the min/max encoder values of each joint.

        Move the joints while the method streams live positions. Press :kbd:`Enter` to finish.

        Args:
            joints (list[str] | None, optional):  Joints to record. Defaults to every joint (`None`).
            display_values (bool, optional): When `True` (default) a live table is printed to the console.

        Raises:
            TypeError: `joints` is not `None` or a list.
            ValueError: any joint's recorded min and max are the same.

        Returns:
            tuple[dict[str, int], dict[str, int]]: Two dictionaries *mins* and *maxes* with the extreme values
            observed for each joint.
        """
        if joints is None:
            joints = list(self.joints)
        elif not isinstance(joints, list):
            raise TypeError(joints)

        display_len = max(len(key) for key in joints)

        start_positions = self._read(joints, normalize=False)
        mins = start_positions.copy()
        maxes = start_positions.copy()

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self._read(joints, normalize=False)
            mins = {joint: int(min(positions[joint], min_)) for joint, min_ in mins.items()}
            maxes = {joint: int(max(positions[joint], max_)) for joint, max_ in maxes.items()}

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<{display_len}} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for joint in joints:
                    print(
                        f"{joint:<{display_len}} | {mins[joint]:>6} | {positions[joint]:>6} | {maxes[joint]:>6}"
                    )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                # Move cursor up to overwrite the previous output
                move_cursor_up(len(joints) + 3)

        same_min_max = [joint for joint in joints if mins[joint] == maxes[joint]]
        if same_min_max:
            raise ValueError(f"Some joints have the same min and max values:\n{pformat(same_min_max)}")

        return mins, maxes

    def configure(self) -> None:
        """No-op: the glove requires no runtime configuration beyond calibration."""
        pass

    # TODO(Steven): This function is copy/paste from the `HomunculusArm` class. Consider moving it to an utility to reduce duplicated code.
    def _normalize(self, values: dict[str, int]) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for joint, val in values.items():
            min_ = self.calibration[joint].range_min
            max_ = self.calibration[joint].range_max
            drive_mode = self.calibration[joint].drive_mode
            bounded_val = min(max_, max(min_, val))

            if self.joints[joint] is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[joint] = -norm if drive_mode else norm
            elif self.joints[joint] is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[joint] = 100 - norm if drive_mode else norm

        return normalized_values

    def _apply_ema(self, raw: dict[str, int]) -> dict[str, int]:
        """Update buffers & running EMA values; return smoothed dict as integers."""
        smoothed: dict[str, int] = {}
        for joint, value in raw.items():
            # maintain raw history
            self._buffers[joint].append(value)

            # initialise on first run
            if self._ema[joint] is None:
                self._ema[joint] = float(value)
            else:
                self._ema[joint] = self.alpha * value + (1 - self.alpha) * self._ema[joint]

            # Convert back to int for compatibility with normalization
            smoothed[joint] = int(round(self._ema[joint]))
        return smoothed

    def _read(
        self, joints: list[str] | None = None, normalize: bool = True, timeout: float = 1
    ) -> dict[str, int | float]:
        """Return the most recent values from the reader thread.

        Optionally applies calibration.
        """
        if not self.new_state_event.wait(timeout=timeout):
            raise TimeoutError(f"{self}: Timed out waiting for state after {timeout}s.")

        with self.state_lock:
            state = self._state

        self.new_state_event.clear()

        if state is None:
            raise RuntimeError(f"{self} Internal error: Event set but no state available.")

        if joints is not None:
            state = {k: v for k, v in state.items() if k in joints}

        # Apply EMA smoothing to raw values first
        state = self._apply_ema(state)

        # Then normalize if requested
        if normalize:
            state = self._normalize(state)

        return state

    def _read_loop(self):
        """Continuously read from the serial buffer in its own thread.

        Sends values to the main thread through a queue.
        """
        while not self.stop_event.is_set():
            try:
                positions = None
                with self.serial_lock:
                    if self.serial.in_waiting > 0:
                        lines = []
                        while self.serial.in_waiting > 0:
                            line = self.serial.read_until().decode("utf-8").strip()
                            if line:
                                lines.append(line.split(" "))

                        if lines:
                            positions = lines[-1]

                if positions is None or len(positions) != len(self.joints):
                    continue

                joint_positions = {joint: int(pos) for joint, pos in zip(self.joints, positions, strict=True)}

                with self.state_lock:
                    self._state = joint_positions
                self.new_state_event.set()

            except Exception as e:
                logger.debug(f"Error reading frame in background thread for {self}: {e}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        """Read the most recent EMA-smoothed, normalized joint positions, remapped to HopeJR hand joints.

        Returns:
            `dict[str, float]`: `"<joint>.pos"` keys, named after the HopeJR hand's joints, mapped to
            their normalized position. See
            [`~teleoperators.homunculus.homunculus_glove_to_hope_jr_hand`] for the remapping.

        Raises:
            TimeoutError: If no new reading arrives from the background thread within 1 second.
        """
        joint_positions = self._read()
        return homunculus_glove_to_hope_jr_hand(
            {f"{joint}.pos": pos for joint, pos in joint_positions.items()}
        )

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Not supported: the glove has no actuators to receive feedback.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        """Stop the background reader thread and close the serial port."""
        self.stop_event.set()
        self.thread.join(timeout=1)
        self.serial.close()
        logger.info(f"{self} disconnected.")
