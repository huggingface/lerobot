#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the GStreamer encoding backend."""

import numpy as np
import pytest

from lerobot.configs.video import GST_VIDEO_CODECS, RGBEncoderConfig
from lerobot.datasets.gstreamer_utils import (
    GST_BITRATE_ONLY_CODECS,
    default_bitrate_for_crf,
    detect_available_encoders_gst,
    gst_codec_options,
    is_gstreamer_available,
)

FRAME_W, FRAME_H = 320, 256

_gst_missing = pytest.mark.skipif(
    not is_gstreamer_available(), reason="GStreamer Python bindings are not installed"
)


def _available_gst_encoders() -> list[str]:
    if not is_gstreamer_available():
        return []
    return detect_available_encoders_gst(GST_VIDEO_CODECS)


_no_gst_encoder = pytest.mark.skipif(
    len(_available_gst_encoders()) == 0,
    reason="no GStreamer hardware encoder available on this machine",
)


class TestCodecOptions:
    """Option translation is pure, so it is testable without an encoder."""

    def test_gop_maps_to_iframe_interval(self):
        opts = gst_codec_options("nvv4l2h265enc", crf=None, preset=None, g=2)
        assert opts["iframeinterval"] == 2
        assert opts["idrinterval"] == 2

    def test_crf_maps_to_fixed_qp_range(self):
        opts = gst_codec_options("nvv4l2h265enc", crf=28, preset=None, g=None)
        assert opts["control-rate"] == 0
        assert opts["qp-range"] == "28,28:28,28:28,28"

    def test_crf_is_clamped_to_the_valid_quantizer_range(self):
        assert gst_codec_options("nvv4l2h265enc", crf=999, preset=None, g=None)["qp-range"].startswith("51,")
        assert gst_codec_options("nvv4l2h265enc", crf=-5, preset=None, g=None)["qp-range"].startswith("0,")

    def test_bitrate_only_codecs_skip_qp_range(self):
        for codec in GST_BITRATE_ONLY_CODECS:
            opts = gst_codec_options(codec, crf=30, preset=None, g=2)
            assert "qp-range" not in opts

    def test_named_and_numeric_presets(self):
        assert gst_codec_options("nvv4l2h265enc", None, "ultrafast", None)["preset-level"] == 1
        assert gst_codec_options("nvv4l2h265enc", None, 3, None)["preset-level"] == 3

    def test_extra_options_do_not_override_derived_ones(self):
        opts = gst_codec_options(
            "nvv4l2h265enc", crf=30, preset=None, g=2, extra_options={"iframeinterval": 60, "vbv-size": 4}
        )
        assert opts["iframeinterval"] == 2
        assert opts["vbv-size"] == 4

    def test_default_bitrate_falls_with_rising_crf(self):
        high = default_bitrate_for_crf(1920, 1080, 30, crf=10)
        low = default_bitrate_for_crf(1920, 1080, 30, crf=45)
        assert high > low
        assert low >= 100_000

    def test_default_bitrate_scales_with_pixel_rate(self):
        small = default_bitrate_for_crf(640, 480, 30, crf=30)
        big = default_bitrate_for_crf(1920, 1080, 30, crf=30)
        assert big > small


class TestConfigIntegration:
    def test_gstreamer_codecs_are_valid_vcodec_values(self):
        cfg = RGBEncoderConfig(vcodec="nvv4l2h265enc", video_backend="gstreamer")
        assert cfg.vcodec == "nvv4l2h265enc"

    def test_ffmpeg_codec_is_rejected_on_the_gstreamer_backend(self):
        with pytest.raises(ValueError):
            RGBEncoderConfig(vcodec="libsvtav1", video_backend="gstreamer")

    def test_get_codec_options_returns_gstreamer_properties(self):
        cfg = RGBEncoderConfig(vcodec="nvv4l2h265enc", video_backend="gstreamer", crf=30, g=2)
        opts = cfg.get_codec_options()
        assert "iframeinterval" in opts
        assert "crf" not in opts

    def test_as_strings_stringifies_every_value(self):
        cfg = RGBEncoderConfig(vcodec="nvv4l2h265enc", video_backend="gstreamer", crf=30, g=2)
        assert all(isinstance(v, str) for v in cfg.get_codec_options(as_strings=True).values())

    def test_pyav_backend_is_unaffected(self):
        cfg = RGBEncoderConfig(vcodec="libsvtav1")
        opts = cfg.get_codec_options()
        assert opts["crf"] == 30
        assert "iframeinterval" not in opts


@_gst_missing
class TestEncoderDetection:
    def test_detection_returns_a_subset_of_what_was_asked_for(self):
        found = detect_available_encoders_gst(GST_VIDEO_CODECS)
        assert set(found).issubset(set(GST_VIDEO_CODECS))

    def test_unknown_element_is_not_reported_available(self):
        assert detect_available_encoders_gst(["definitely_not_an_element"]) == []

    def test_a_bare_string_is_accepted(self):
        assert isinstance(detect_available_encoders_gst("definitely_not_an_element"), list)


@_gst_missing
@_no_gst_encoder
class TestEncoding:
    """Round trips against a real encoder, skipped where none exists."""

    @pytest.fixture
    def codec(self):
        return _available_gst_encoders()[0]

    @pytest.fixture
    def frames(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        return [np.roll(base, i, axis=1) for i in range(20)]

    def test_writes_a_decodable_video_with_every_frame(self, tmp_path, codec, frames):
        av = pytest.importorskip("av")
        from lerobot.datasets.gstreamer_utils import GStreamerVideoWriter

        out = tmp_path / "out.mp4"
        h, w, _ = frames[0].shape
        with GStreamerVideoWriter(out, fps=10, width=w, height=h, vcodec=codec, crf=30) as writer:
            for frame in frames:
                writer.write(frame)

        assert out.stat().st_size > 0
        with av.open(str(out)) as container:
            stream = container.streams.video[0]
            assert (stream.codec_context.width, stream.codec_context.height) == (w, h)
            assert sum(1 for _ in container.decode(video=0)) == len(frames)

    def test_streaming_encoder_end_to_end(self, tmp_path, codec, frames):
        av = pytest.importorskip("av")
        from lerobot.datasets.video_utils import StreamingVideoEncoder

        encoder = StreamingVideoEncoder(
            fps=10,
            rgb_encoder=RGBEncoderConfig(vcodec=codec, video_backend="gstreamer", crf=30),
        )
        key = "observation.images.cam"
        encoder.start_episode([key], tmp_path)
        for frame in frames:
            encoder.feed_frame(key, frame)
        results = encoder.finish_episode()
        encoder.close()

        path, _ = results[key]
        with av.open(str(path)) as container:
            assert sum(1 for _ in container.decode(video=0)) == len(frames)

    def test_wrong_frame_size_is_rejected(self, tmp_path, codec, frames):
        from lerobot.datasets.gstreamer_utils import GStreamerVideoWriter

        with (
            GStreamerVideoWriter(
                tmp_path / "o.mp4", fps=10, width=FRAME_W, height=FRAME_H, vcodec=codec
            ) as writer,
            pytest.raises(ValueError, match="pipeline expects"),
        ):
            writer.write(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_close_is_idempotent(self, tmp_path, codec, frames):
        from lerobot.datasets.gstreamer_utils import GStreamerVideoWriter

        writer = GStreamerVideoWriter(tmp_path / "o.mp4", fps=10, width=FRAME_W, height=FRAME_H, vcodec=codec)
        writer.write(frames[0])
        writer.close()
        writer.close()

    def test_write_after_close_raises(self, tmp_path, codec, frames):
        from lerobot.datasets.gstreamer_utils import GStreamerVideoWriter

        writer = GStreamerVideoWriter(tmp_path / "o.mp4", fps=10, width=FRAME_W, height=FRAME_H, vcodec=codec)
        writer.write(frames[0])
        writer.close()
        with pytest.raises(RuntimeError, match="after close"):
            writer.write(frames[0])

    def test_unsupported_codec_name_raises(self, tmp_path):
        from lerobot.datasets.gstreamer_utils import GStreamerVideoWriter

        with pytest.raises(ValueError, match="Unsupported GStreamer video codec"):
            GStreamerVideoWriter(
                tmp_path / "o.mp4", fps=10, width=FRAME_W, height=FRAME_H, vcodec="libsvtav1"
            )

    def test_auto_selects_an_available_encoder(self):
        cfg = RGBEncoderConfig(vcodec="auto", video_backend="gstreamer")
        assert cfg.vcodec in _available_gst_encoders()
