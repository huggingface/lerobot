import time
from unittest.mock import patch

import pytest


class MotorCalibrator:
    def reset_offset(self, motor_name):
        # In real code, this would reset any stored offset.
        pass

    def set_calibration(self, motor_name, min_max, zero_offset):
        """
        Dummy calibration setter -- in real code, you'd store these in nonvolatile memory
        or otherwise save them. We'll leave it as a no-op.
        """
        pass

    def read(self, register, motor_names):
        """
        Mocked read method. In practice, this would query the real motor's present position.
        """
        pass

    def calibrate_motor(self, motor_name: str):
        self.reset_offset(motor_name)
        time.sleep(0.1)

        input(f"Move {motor_name} to the middle of its range of motion and press ENTER....")
        # The first read is the middle position.
        middle = self.read("Present_Position", motor_names=[motor_name])
        recorded_positions = []

        print(f"Move {motor_name} through its entire range of motion (hitting its limits on both sides).")

        try:
            while True:
                pos = self.read("Present_Position", motor_names=[motor_name])
                recorded_positions.append(pos)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

        # Unwrap the recorded positions if the range is too big.
        if max(recorded_positions) - min(recorded_positions) > 2048:
            adjusted = [p if p >= 2048 else p + 4096 for p in recorded_positions]
        else:
            adjusted = recorded_positions

        physical_min = min(adjusted)
        physical_max = max(adjusted)

        # Wrap the calibration limits back if needed.
        cal_min = physical_min if physical_min < 4096 else physical_min - 4096
        cal_max = physical_max if physical_max < 4096 else physical_max - 4096

        # The zero_offset is set so that the original middle reading is centered at 2047.
        zero_offset = middle - 2047

        # Adjust calibration limits by the zero_offset.
        cal_min_offset = (cal_min - zero_offset) % 4096
        cal_max_offset = (cal_max - zero_offset) % 4096

        print("Done recording. Computing min, max...")

        self.set_calibration(motor_name, (cal_min_offset, cal_max_offset), zero_offset)

        return cal_max, cal_min, middle, zero_offset, cal_max_offset, cal_min_offset


@pytest.mark.parametrize(
    "mocked_positions, expected_output, expected_zero_offset",
    [
        # Case 1: Crossing 4096->0
        # First value (middle) = 0.0, then recorded positions [3696, 3896, 200, 400, 200, 3896, 3696, 200, 400].
        # Adjusted: [3696, 3896, 4296, 4496] -> physical min = 3696, physical max = 4496.
        # Wrapped: cal_min = 3696, cal_max = 4496-4096 = 400.
        # zero_offset = 0.0 - 2047 = -2047.
        # Offset-adjusted limits: cal_max_offset = 400 - (-2047)=2447, cal_min_offset = 3696 - (-2047)= 5743 % 4096 = 1647.
        (
            [0.0, 3696, 3896, 200, 400, 200, 3896, 3696, 200, 400],
            (400, 3696, 0.0, -2047, 2447, 1647),
            -2047,
        ),
        # Case 2: Crossing 0->4096
        # First value (middle) = 0.0, then recorded positions [400, 200, 3896, 3696, 3800, 100, 200].
        # Adjusted: [4496, 4296, 3896, 3696] -> physical min = 3696, physical max = 4496.
        # Wrapped: cal_min = 3696, cal_max = 4496-4096 = 400.
        # zero_offset = 0.0 - 2047 = -2047.
        # Offset-adjusted limits: cal_max_offset = 400 - (-2047)=2447, cal_min_offset = 3696 - (-2047)= 5743 % 4096 = 1647.
        (
            [0.0, 400, 200, 3896, 3696, 3800, 100, 200],
            (400, 3696, 0.0, -2047, 2447, 1647),
            -2047,
        ),
        # Case 3: Not Crossing 0..4096
        # First value (middle) = 1150, then recorded positions [300, 500, 1000, 2000].
        # No adjustment needed: physical min = 300, physical max = 2000.
        # Wrapped: cal_min = 300, cal_max = 2000.
        # zero_offset = 1150 - 2047 = -897.
        # Offset-adjusted limits: cal_max_offset = 2000 - (-897)=2897, cal_min_offset = 300 - (-897)=1197.
        (
            [1150, 300, 500, 1000, 2000],
            (2000, 300, 1150, -897, 2897, 1197),
            -897,
        ),
        # Case 4: Not Crossing 0..4096
        # First value (middle) = 1150, then recorded positions [1150, 2000, 1000, 500, 300, 500, 1000, 1900].
        # physical min = 300, physical max = 2000.
        # Wrapped: cal_min = 300, cal_max = 2000.
        # zero_offset = 1150 - 2047 = -897.
        # Offset-adjusted limits: cal_max_offset = 2000 - (-897)=2897, cal_min_offset = 300 - (-897)=1197.
        (
            [1150, 2000, 1000, 500, 300, 500, 1000, 1900],
            (2000, 300, 1150, -897, 2897, 1197),
            -897,
        ),
    ],
)
def test_calibrate_motor(mocked_positions, expected_output, expected_zero_offset):
    """
    Tests the calibrate_motor function against several scenarios.
    The function now returns:
        (cal_max, cal_min, middle, zero_offset, cal_max_offset, cal_min_offset)
    """
    calibrator = MotorCalibrator()

    # Build a side_effect list for read: the first value is taken as the "middle" reading,
    # and then all subsequent values are used until a KeyboardInterrupt is raised.
    side_effect_list = mocked_positions + [KeyboardInterrupt()]

    with (
        patch("builtins.input", return_value=""),
        patch.object(calibrator, "read", side_effect=side_effect_list),
        patch("time.sleep", return_value=None),
    ):
        got_max, got_min, got_mid, got_zero_offset, got_max_offset, got_min_offset = (
            calibrator.calibrate_motor("test_motor")
        )

    # Compare the complete output tuple.
    expected = expected_output
    got = (
        got_max,
        got_min,
        round(got_mid),
        got_zero_offset,
        got_max_offset,
        got_min_offset,
    )
    assert got == expected, f"For input {mocked_positions}, expected {expected} but got {got}"

    # Additionally, check zero_offset separately.
    assert round(got_zero_offset) == expected_zero_offset, (
        f"For input {mocked_positions}, expected zero_offset={expected_zero_offset} but got {got_zero_offset}"
    )
