import time
from unittest.mock import patch

import pytest


class MotorCalibrator:
    def reset_offset(self, motor_name):
        pass

    def set_calibration(self, motor_name, min_max, zero_offset, direction):
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

        input(f"Move {motor_name} to middle of its range of motion, and press ENTER....")
        middle_pos = self.read("Present_Position", motor_names=[motor_name])

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

            # If we see a huge positive jump > 2048, treat it as having wrapped backwards 4096
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

        # Determine travel direction
        direction = idx_min < idx_max

        # If direction is True, swap recorded min/max so we return them in "max, min" order
        if direction:
            recorded_min, recorded_max = recorded_max, recorded_min

        # zero_offset = how far from 2047 the midpoint is
        zero_offset = middle_pos - 2047

        # Store or persist calibration
        self.set_calibration(motor_name, (recorded_min, recorded_max), zero_offset, direction)

        # Return the 5-tuple so the test can check each piece individually
        # (If you *only* need the original 4 items, just return those and test zero_offset separately.)
        return recorded_max, recorded_min, middle_pos, direction, zero_offset


@pytest.mark.parametrize(
    "mocked_positions, expected_output, expected_offset",
    [
        # The *first* element is the middle_pos,
        # the rest are the "range of motion" positions.
        # For example, middle=0.0, then range=[3696, 3896, 200, 400].
        #
        # The test expects calibrate_motor to return: (max, min, mid, direction, offset).
        #
        # Case 1: Crossing 4096->0, direction = True
        ([0.0, 3696, 3896, 200, 400], (3696, 400, 0.0, True), -2047),
        # Case 2: Crossing 0->4096, direction = False
        ([0.0, 400, 200, 3896, 3696], (400, 3696, 0.0, False), -2047),
        # Case 3: Not Crossing 0..4096, direction = True
        ([1150, 300, 500, 1000, 2000], (300, 2000, 1150, True), -897),
        # Case 4: Not Crossing 0..4096, direction = False
        ([1150, 2000, 1000, 500, 300], (2000, 300, 1150, False), -897),
    ],
)
def test_calibrate_motor(mocked_positions, expected_output, expected_offset):
    """
    Tests the calibrate_motor function against four scenarios.
    The first element in mocked_positions is the "middle" position.
    """
    calibrator = MotorCalibrator()

    # side_effect => [middle_pos, range_position1, range_position2, ..., KeyboardInterrupt()]
    side_effect_list = mocked_positions + [KeyboardInterrupt()]

    with (
        # Patch builtins.input so it won't block the test
        patch("builtins.input", return_value=""),
        # Patch calibrator.read so it will return each item in side_effect_list
        patch.object(calibrator, "read", side_effect=side_effect_list),
        # Patch time.sleep so it doesn't block the test
        patch("time.sleep", return_value=None),
    ):
        got_max, got_min, got_mid, got_dir, got_offset = calibrator.calibrate_motor("test_motor")

    # Compare the first 4 items: (max, min, mid, direction).
    # We can round the mid if you need integer comparisons:
    assert (got_max, got_min, round(got_mid), got_dir) == expected_output, (
        f"For input {mocked_positions}, expected {expected_output} but got "
        f"({got_max}, {got_min}, {got_mid}, {got_dir})"
    )

    # Optionally also check the zero_offset
    assert round(got_offset) == expected_offset, (
        f"For input {mocked_positions}, expected offset={expected_offset} but got {got_offset}"
    )
