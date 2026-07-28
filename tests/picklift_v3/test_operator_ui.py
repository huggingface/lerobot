import numpy as np
import pytest

from examples.picklift_v3.operator_ui import render_dashboard


def test_dashboard_renders_expected_size():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dashboard = render_dashboard(
        frame,
        status="RECORDING",
        elapsed_s=1.5,
        frames=30,
        target_frames=200,
        message="engineering smoke",
    )
    assert dashboard.shape == (760, 1280, 3)
    assert dashboard.dtype == np.uint8
    assert np.count_nonzero(dashboard) > 0


def test_dashboard_rejects_noncanonical_front():
    with pytest.raises(ValueError, match="expected RGB"):
        render_dashboard(np.zeros((720, 1280, 3), dtype=np.uint8), status="WAITING")
