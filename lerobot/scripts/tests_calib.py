import select
import sys
import time
from unittest.mock import call, patch

import numpy as np
import pytest


class MotorCalibrator:
    def __init__(self, motor_names):
        self.motor_names = motor_names

    def reset_offsets(self):
        """Pretend to reset offsets for all motors."""
        pass

    def set_min_max(self, motor_name, min_max):
        """Pretend to store min/max for that motor."""
        pass

    def set_calibration(self, motor_name, offset):
        """Pretend to set the offset for that motor."""
        pass

    def read(self, register, motor_names):
        """
        In real code, read "Present_Position" from the hardware
        for each motor in 'motor_names'. Returns list or dict.
        """
        pass

    def find_min_max(self):
        """
        1) Reads Present_Position for *all* motors simultaneously until user presses Enter.
        2) After user is done, it computes each motor's min and max.
        3) If any motor crosses a total range >= 4096 ticks (full 360 deg),
        raises an error for that motor.
        4) Otherwise, calls set_min_max(...) on each motor individually.
        """
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")

        # This will be a list of length N, each an array of motor positions [pos1, pos2, ...]
        recorded_positions = []

        while True:
            positions = self.read("Present_Position", motor_names=self.motor_names)
            recorded_positions.append(positions)
            time.sleep(0.01)

            # Check if user pressed Enter
            ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
            if ready_to_read:
                line = sys.stdin.readline()
                if line.strip() == "":
                    break  # user pressed Enter

        # Convert recorded_positions (list of arrays) to a 2D numpy array: shape (num_timesteps, num_motors)
        all_positions = np.array(recorded_positions, dtype=np.float32)

        # For each motor, find min, max
        for i, mname in enumerate(self.motor_names):
            motor_column = all_positions[:, i]
            raw_range = motor_column.max() - motor_column.min()

            # Check if motor made a full 360-degree rotation or more set min max at 0 and 4095
            if raw_range >= 4000:
                physical_min = 0
                physical_max = 4095
            else:
                physical_min = int(motor_column.min())
                physical_max = int(motor_column.max())

            print(f"Setting '{mname}' min={physical_min}, max={physical_max}")
            self.set_min_max(mname, (physical_min, physical_max))

    def find_offset(self):
        self.reset_offsets()

        input("Move robot to the middle of its range of motion and press ENTER....")

        for _, name in enumerate(self.motor_names):
            middle = self.read("Present_Position", motor_names=[name])[0]
            zero_offset = (
                middle - 2047
            )  # The zero_offset is set so that the original middle reading is centered at 2047.
            self.set_calibration(name, zero_offset)


@pytest.mark.parametrize(
    "motor_names, mocked_data, expected_calls",
    [
        (
            ["m1"],
            [1000, 1200, 1800, 1500, 900],
            [call("m1", (900, 1800))],
        ),
        (
            ["m2"],
            [1000, 5000, 100, 50],
            [call("m2", (0, 4095))],
        ),
        (
            ["mA", "mB"],
            [
                [100, 200],
                [4200, 300],
                [2100, 2900],
            ],
            [
                call("mA", (0, 4095)),
                call("mB", (200, 2900)),
            ],
        ),
    ],
)
def test_find_min_max(motor_names, mocked_data, expected_calls):
    """
    Test that find_min_max:
      - loops reading positions,
      - stops reading on ENTER,
      - sets min/max based on range < or >= 4000
    """
    calibrator = MotorCalibrator(motor_names)

    # We'll patch the 'read' method so each call returns the next row of data.
    # If we run out, just return the last row again.
    if not isinstance(mocked_data[0], list):
        # If user gave a single motor's positions as a flat list, wrap them
        # so each read call returns [pos].
        # e.g. [1000,1200,1800] => each read => [pos_of_m1]
        mocked_data = [[val] for val in mocked_data]

    data_iter = iter(mocked_data)

    def read_side_effect(*args, **kwargs):
        try:
            return next(data_iter)
        except StopIteration:
            return mocked_data[-1]

    # Patch 'select.select' to let the loop run as many times as we have data,
    # then pretend user pressed ENTER to break out of the while loop.
    select_call_count = 0

    def select_side_effect(rlist, wlist, xlist, timeout):
        nonlocal select_call_count
        # We want the loop to keep going len(mocked_data) times.
        # On the next call, we simulate user pressing Enter.
        if select_call_count < len(mocked_data) - 1:
            select_call_count += 1
            return ([], [], [])
        else:
            return ([sys.stdin], [], [])

    with (
        patch.object(calibrator, "read", side_effect=read_side_effect),
        patch("select.select", side_effect=select_side_effect),
        patch("sys.stdin.readline", return_value="\n"),
        patch("time.sleep", return_value=None),
        patch.object(calibrator, "set_min_max") as mock_set_min_max,
    ):
        calibrator.find_min_max()

    # Verify the set_min_max calls match expectation
    mock_set_min_max.assert_has_calls(expected_calls, any_order=False)


def test_find_offset():
    """
    Test that find_offset:
      1) Calls reset_offsets() once.
      2) Waits for user input (patch input).
      3) Then for each motor in calibrator.motor_names,
         reads once and sets offset = pos - 2047
      4) Calls set_calibration(mname, offset).
    """
    motor_names = ["m1", "m2", "m3"]
    calibrator = MotorCalibrator(motor_names)

    read_values = {
        "m1": [2200],
        "m2": [300],
        "m3": [4095],
    }

    def read_side_effect(register, motor_names):
        # We'll return a one-element list for each motor
        return [read_values[motor_names[0]].pop(0)]

    with (
        patch.object(calibrator, "reset_offsets") as mock_reset,
        patch("builtins.input", return_value=""),
        patch.object(calibrator, "read", side_effect=read_side_effect) as mock_read,
        patch.object(calibrator, "set_calibration") as mock_set_calib,
    ):
        calibrator.find_offset()

    mock_reset.assert_called_once()

    assert mock_read.call_count == len(motor_names)

    # offset = position - 2047
    expected_calls = [
        call("m1", 2200 - 2047),  # => 153
        call("m2", 300 - 2047),  # => -1747
        call("m3", 4095 - 2047),  # => 2048
    ]
    mock_set_calib.assert_has_calls(expected_calls, any_order=False)
