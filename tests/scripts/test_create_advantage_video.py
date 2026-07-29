import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2")

from lerobot.scripts.lerobot_create_advantage_video import (
    _contiguous_segments,
    _draw_dashboard,
    _draw_status_badge,
)


def _episode_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predicted_value": [-0.8, -0.6, -0.4, -0.2],
            "predicted_value_raw": [-0.81, -0.55, -0.45, -0.18],
            "mc_return": [-0.9, -0.6, -0.3, 0.0],
            "advantage": [-0.1, 0.1, -0.05, 0.2],
            "advantage_label": ["negative", "positive", "negative", "positive"],
            "intervention": [False, False, True, False],
        }
    )


def test_contiguous_advantage_segments():
    labels = np.array(["negative", "negative", "positive", "positive", "negative"])

    assert _contiguous_segments(labels) == [
        (0, 2, "negative"),
        (2, 4, "positive"),
        (4, 5, "negative"),
    ]


def test_advantage_dashboard_shape():
    dashboard = _draw_dashboard(
        width=640,
        height=140,
        episode=_episode_frame(),
        current_index=2,
        fps=30,
        task="stack the yellow cube on the red cube",
    )

    assert dashboard.shape == (140, 640, 3)
    assert dashboard.dtype == np.uint8
    assert dashboard.var() > 0


def test_status_badge_modifies_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _draw_status_badge(frame, "positive", intervention=False, episode_index=12)

    assert frame.sum() > 0
