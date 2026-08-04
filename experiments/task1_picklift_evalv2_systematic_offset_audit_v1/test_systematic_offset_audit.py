from __future__ import annotations

import numpy as np

from systematic_offset_audit import (
    URDF_PATH,
    UrdfForwardKinematics,
    first_closure,
    fit_task_to_pixel_homography,
    image_to_task,
    project_points,
    red_component,
)


def test_red_component_uses_nominal_nearest_component() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[260:281, 200:221, 0] = 255
    rgb[350:401, 330:381, 0] = 255
    measured = red_component(rgb, expected_px=[210, 270])
    assert np.allclose(measured["centroid_px"], [210, 270], atol=0.1)
    assert measured["component_selection"]["rule"].startswith("nearest")


def test_homography_round_trip() -> None:
    centroids = {}
    for row in range(1, 4):
        for column in range(1, 5):
            cell = f"r{row}c{column}"
            x = 0.225 + 0.05 * (row - 1)
            y = -0.075 + 0.05 * (column - 1)
            centroids[cell] = [100 + 600 * y, 400 - 500 * x]
    homography, fit = fit_task_to_pixel_homography(centroids)
    point = np.asarray([[0.29, 0.04]])
    pixel = project_points(homography, point)[0]
    recovered = image_to_task(homography, pixel.tolist())
    assert np.allclose(recovered, point[0], atol=1e-10)
    assert fit["maximum_anchor_residual_px"] < 1e-5


def test_so101_fk_ready_pose_is_stable() -> None:
    fk = UrdfForwardKinematics.load(URDF_PATH)
    ready = [
        7.4285712242126465,
        -98.32967376708984,
        45.010990142822266,
        92.21977996826172,
        1.8461538553237915,
    ]
    assert np.allclose(
        fk.position(ready),
        [0.14820313, -0.01452723, 0.13314776],
        atol=1e-4,
    )


def test_first_closure_midpoint_and_low_approach() -> None:
    class FakeFk:
        def position(self, state):
            tick = state[0]
            return np.asarray([tick, 0.0, abs(tick - 4.0)])

    gripper = [6, 8, 9, 10, 9, 7, 5, 3, 2]
    rows = [
        {"observation_state": [tick, 0, 0, 0, 0, value]}
        for tick, value in enumerate(gripper)
    ]
    result = first_closure(rows, FakeFk())
    assert result["open_peak_tick"] == 3
    assert result["first_closure_tick"] == 6
    assert result["low_approach_tick"] == 4
