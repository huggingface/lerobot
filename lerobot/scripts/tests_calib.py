import time
from unittest.mock import patch

import pytest


class MotorCalibrator:
    def reset_offset(self, motor_name):
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

        print(f"Move {motor_name} through its entire range of motion (hitting its limits on both sides).")

        recorded_positions = []
        try:
            while True:
                pos = self.read("Present_Position", motor_names=[motor_name])
                recorded_positions.append(pos)
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

        print("Done recording. Computing min, max, and middle...")

        # We make a new list of positions that accounts for any wrap-around jumps
        unwrapped = [recorded_positions[0]]
        for i in range(1, len(recorded_positions)):
            prev = recorded_positions[i - 1]
            curr = recorded_positions[i]
            diff = curr - prev

            # If we see a huge positive jump >2048, treat it as having wrapped backwards 4096
            if diff > 2048:
                unwrapped.append(unwrapped[-1] + (diff - 4096))
            # If we see a huge negative jump < -2048, treat it as having wrapped forwards 4096
            elif diff < -2048:
                unwrapped.append(unwrapped[-1] + (diff + 4096))
            else:
                unwrapped.append(unwrapped[-1] + diff)

        # Find unwrapped min & max and their indices
        unwrapped_min = min(unwrapped)
        unwrapped_max = max(unwrapped)
        idx_min = unwrapped.index(unwrapped_min)
        idx_max = unwrapped.index(unwrapped_max)

        # Corresponding (wrapped) recorded min & max
        recorded_min = recorded_positions[idx_min]
        recorded_max = recorded_positions[idx_max]

        # Compute midpoint in unwrapped space, then wrap it to [0..4095]
        unwrapped_mid = (unwrapped_min + unwrapped_max) / 2.0
        recorded_mid = unwrapped_mid % 4096

        zero_offset = recorded_mid - 2047
        self.set_calibration(motor_name, (recorded_min, recorded_max), zero_offset)

        return recorded_max, recorded_min, recorded_mid, zero_offset


@pytest.mark.parametrize(
    "mocked_positions, expected_output, expected_offset",
    [
        # Case 1: Crossing 4096->0
        ([3696, 3896, 200, 400], (400, 3696, 0.0), -2047),
        # Case 2: Crossing 0->4096
        (
            [400, 200, 3896, 3696],
            (
                400,
                3696,
                0.0,
            ),
            -2047,
        ),
        # Case 3: Not Crossing 0..4096
        ([300, 500, 1000, 2000], (2000, 300, 1150), -897),
        # Case 4: Not Crossing 0..4096
        ([2000, 1000, 500, 300], (2000, 300, 1150), -897),
    ],
)
def test_calibrate_motor(mocked_positions, expected_output, expected_offset):
    """
    Tests the calibrate_motor function against several scenarios.
    """
    calibrator = MotorCalibrator()

    # Build a side_effect list for read: it returns each mocked position then raises KeyboardInterrupt.
    side_effect_list = mocked_positions + [KeyboardInterrupt()]

    with (
        patch("builtins.input", return_value=""),
        patch.object(calibrator, "read", side_effect=side_effect_list),
        patch("time.sleep", return_value=None),
    ):
        got_max, got_min, got_mid, got_offset = calibrator.calibrate_motor("test_motor")

    # Compare (max, min, mid, direction)
    assert (got_max, got_min, round(got_mid)) == expected_output, (
        f"For input {mocked_positions}, expected {(expected_output)} but got "
        f"({got_max}, {got_min}, {got_mid})"
    )
    # Check zero_offset as well
    assert round(got_offset) == expected_offset, (
        f"For input {mocked_positions}, expected offset={expected_offset} but got {got_offset}"
    )
