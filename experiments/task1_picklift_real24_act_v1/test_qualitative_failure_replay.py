"""Pure contract tests for the qualitative failure replay."""

from __future__ import annotations

import pytest

from run_qualitative_failure_replay import validate_video_probe


def valid_probe() -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_name": "h264",
                "width": 640,
                "height": 480,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "20/1",
                "avg_frame_rate": "20/1",
                "nb_read_frames": "600",
            }
        ],
        "format": {"duration": "30.000000"},
    }


def test_video_probe_accepts_exact_contract() -> None:
    result = validate_video_probe(valid_probe())

    assert result["frames"] == 600
    assert result["duration_seconds"] == 30.0


def test_video_probe_rejects_partial_video() -> None:
    probe = valid_probe()
    probe["streams"][0]["nb_read_frames"] = "599"

    with pytest.raises(RuntimeError, match="600 frames"):
        validate_video_probe(probe)
