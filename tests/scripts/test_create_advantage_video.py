import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
import pytest

from lerobot.scripts.lerobot_create_advantage_video import (
    _contiguous_segments,
    _create_decode_proxy,
    _draw_dashboard,
    _draw_label_timeline,
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


def test_label_timeline_modifies_only_lower_frame_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    labels = np.array(["negative", "negative", "positive", "positive"])

    _draw_label_timeline(frame, labels, current_index=2)

    assert frame.sum() > 0
    assert frame[:350].sum() == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_ffmpeg_decode_proxy_handles_av1(tmp_path):
    source = tmp_path / "source.mkv"
    proxy = tmp_path / "proxy.mp4"
    encode = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=1",
            "-c:v",
            "libaom-av1",
            "-cpu-used",
            "8",
            "-crf",
            "40",
            str(source),
        ],
        capture_output=True,
    )
    if encode.returncode != 0:
        pytest.skip("ffmpeg does not provide the libaom-av1 encoder")

    _create_decode_proxy(
        source,
        proxy,
        start_timestamp=0,
        end_timestamp=1,
        fps=10,
        expected_frames=10,
    )

    capture = cv2.VideoCapture(str(proxy))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frame_count == 10
