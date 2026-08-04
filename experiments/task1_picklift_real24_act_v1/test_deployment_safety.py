from __future__ import annotations

import numpy as np
import pytest

from deployment_safety import CALIBRATION_BOUNDS_DEG, clamp_action_fail_closed


def test_in_range_action_is_unchanged() -> None:
    action = np.zeros(6, dtype=np.float32)
    clipped, mask = clamp_action_fail_closed(action)
    np.testing.assert_array_equal(clipped, action)
    assert not mask.any()


def test_out_of_range_action_is_clipped_per_joint() -> None:
    action = np.asarray([-200, 200, -200, 200, -200, 200], dtype=np.float32)
    clipped, mask = clamp_action_fail_closed(action)
    np.testing.assert_allclose(
        clipped,
        np.asarray(
            [
                CALIBRATION_BOUNDS_DEG[0, 0],
                CALIBRATION_BOUNDS_DEG[1, 1],
                CALIBRATION_BOUNDS_DEG[2, 0],
                CALIBRATION_BOUNDS_DEG[3, 1],
                CALIBRATION_BOUNDS_DEG[4, 0],
                CALIBRATION_BOUNDS_DEG[5, 1],
            ]
        ),
        rtol=0,
        atol=1e-5,
    )
    assert mask.all()


@pytest.mark.parametrize("action", [np.zeros(5), np.zeros((1, 6)), np.zeros(7)])
def test_wrong_shape_fails_closed(action: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="shape"):
        clamp_action_fail_closed(action)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_fails_closed(bad_value: float) -> None:
    action = np.zeros(6)
    action[2] = bad_value
    with pytest.raises(RuntimeError, match="NaN or infinity"):
        clamp_action_fail_closed(action)
