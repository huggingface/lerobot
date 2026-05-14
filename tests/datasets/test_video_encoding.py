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

"""Unit tests for ``lerobot.datasets.video_utils`` encoding functions and ``lerobot.configs.video.VideoEncoderConfig`` config class."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av  # noqa: E402

from lerobot.configs import VALID_VIDEO_CODECS, VideoEncoderConfig
from lerobot.datasets.image_writer import write_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import get_codec
from lerobot.datasets.utils import INFO_PATH
from lerobot.datasets.video_utils import (
    concatenate_video_files,
    encode_video_frames,
    get_video_info,
)
from tests.fixtures.constants import DUMMY_VIDEO_INFO


# Per-codec skip markers — validation tests only fire when the codec is available
def _require_encoder(vcodec: str) -> pytest.MarkDecorator:
    """Skip the test if ``vcodec`` is not available in the local FFmpeg build."""
    return pytest.mark.skipif(get_codec(vcodec) is None, reason=f"{vcodec!r} not in local FFmpeg build")


require_libsvtav1 = _require_encoder("libsvtav1")
require_h264 = _require_encoder("h264")
require_videotoolbox = _require_encoder("h264_videotoolbox")
require_nvenc = _require_encoder("h264_nvenc")
require_vaapi = _require_encoder("h264_vaapi")
require_qsv = _require_encoder("h264_qsv")


# ─── VideoEncoderConfig / codec options ──────────────────────────────


class TestCodecOptions:
    @require_libsvtav1
    def test_libsvtav1_defaults(self):
        cfg = VideoEncoderConfig()
        opts = cfg.get_codec_options()
        assert opts["g"] == 2
        assert opts["crf"] == 30
        assert opts["preset"] == 12

    @require_libsvtav1
    def test_libsvtav1_custom_preset(self):
        cfg = VideoEncoderConfig(preset=8)
        assert cfg.get_codec_options()["preset"] == 8

    @require_h264
    def test_h264_options(self):
        cfg = VideoEncoderConfig(vcodec="h264", g=10, crf=23, preset=None)
        opts = cfg.get_codec_options()
        assert opts["g"] == 10
        assert opts["crf"] == 23
        assert "preset" not in opts

    @require_videotoolbox
    def test_videotoolbox_options(self):
        cfg = VideoEncoderConfig(vcodec="h264_videotoolbox", g=2, crf=30, preset=None)
        opts = cfg.get_codec_options()
        assert opts["g"] == 2
        assert opts["q:v"] == 40
        assert "crf" not in opts

    @_require_encoder("h264_nvenc")
    def test_nvenc_options(self):
        cfg = VideoEncoderConfig(vcodec="h264_nvenc", g=2, crf=25, preset=None)
        opts = cfg.get_codec_options()
        assert opts["rc"] == 0
        assert opts["qp"] == 25
        assert "crf" not in opts
        assert opts["g"] == 2

    @_require_encoder("h264_vaapi")
    def test_vaapi_options(self):
        cfg = VideoEncoderConfig(vcodec="h264_vaapi", crf=28, preset=None)
        assert cfg.get_codec_options()["qp"] == 28

    @_require_encoder("h264_qsv")
    def test_qsv_options(self):
        cfg = VideoEncoderConfig(vcodec="h264_qsv", crf=25, preset=None)
        assert cfg.get_codec_options()["global_quality"] == 25

    @require_h264
    def test_no_g_no_crf(self):
        cfg = VideoEncoderConfig(vcodec="h264", g=None, crf=None, preset=None)
        opts = cfg.get_codec_options()
        assert "g" not in opts
        assert "crf" not in opts

    @require_libsvtav1
    def test_encoder_threads_libsvtav1(self):
        cfg = VideoEncoderConfig(fast_decode=0)
        opts = cfg.get_codec_options(encoder_threads=4)
        assert "lp=4" in opts.get("svtav1-params", "")

    @require_h264
    def test_encoder_threads_h264(self):
        cfg = VideoEncoderConfig(vcodec="h264", preset=None)
        assert cfg.get_codec_options(encoder_threads=2)["threads"] == 2

    @require_libsvtav1
    def test_fast_decode_libsvtav1(self):
        cfg = VideoEncoderConfig(fast_decode=1)
        opts = cfg.get_codec_options()
        assert "fast-decode=1" in opts.get("svtav1-params", "")

    @require_libsvtav1
    def test_libsvtav1_fast_decode_clamped_to_svt_range(self):
        """Out-of-range fast_decode is clamped to [0, 2] in svtav1-params (SVT-AV1 FastDecode)."""
        cfg = VideoEncoderConfig(fast_decode=100)
        assert "fast-decode=2" in cfg.get_codec_options().get("svtav1-params", "")
        cfg_neg = VideoEncoderConfig(fast_decode=-5)
        assert "fast-decode=0" in cfg_neg.get_codec_options().get("svtav1-params", "")

    @require_h264
    def test_fast_decode_h264(self):
        cfg = VideoEncoderConfig(vcodec="h264", fast_decode=1, preset=None)
        assert cfg.get_codec_options()["tune"] == "fastdecode"

    @require_libsvtav1
    def test_pix_fmt_unsupported_raises(self):
        """Passing an unsupported pix_fmt is a hard error."""
        with pytest.raises(ValueError, match="pix_fmt"):
            VideoEncoderConfig(pix_fmt="yuv444p")  # libsvtav1 only supports yuv420p variants

    @require_libsvtav1
    @require_h264
    def test_preset_default_behaviour(self):
        """Empty constructor picks preset=12 (libsvtav1 path); other codecs stay None."""
        assert VideoEncoderConfig().preset == 12
        assert VideoEncoderConfig(vcodec="libsvtav1").preset == 12
        assert VideoEncoderConfig(vcodec="h264").preset is None
        assert VideoEncoderConfig(vcodec="h264", preset=None).preset is None

    @require_h264
    def test_preset_string_on_h264(self):
        """h264 accepts string presets and forwards them to FFmpeg."""
        cfg = VideoEncoderConfig(vcodec="h264", preset="slow")
        assert cfg.get_codec_options()["preset"] == "slow"

    @require_videotoolbox
    def test_preset_on_videotoolbox_not_set(self):
        """videotoolbox has no preset option at all."""
        cfg = VideoEncoderConfig(vcodec="h264_videotoolbox", preset="slow")
        assert "preset" not in cfg.get_codec_options()

    @require_libsvtav1
    def test_libsvtav1_preset_out_of_range_raises(self):
        """libsvtav1 preset must sit in [-2, 13] as exposed by PyAV."""
        with pytest.raises(ValueError, match="out of range"):
            VideoEncoderConfig(vcodec="libsvtav1", preset=100)
        with pytest.raises(ValueError, match="out of range"):
            VideoEncoderConfig(vcodec="libsvtav1", preset=-3)

    @require_libsvtav1
    def test_libsvtav1_crf_out_of_range_raises(self):
        """libsvtav1 crf must sit in [0, 63]."""
        with pytest.raises(ValueError, match="crf.*out of range"):
            VideoEncoderConfig(vcodec="libsvtav1", crf=64)

    @require_libsvtav1
    def test_libsvtav1_crf_rejects_python_float(self):
        """libsvtav1 exposes ``crf`` as an INT AVOption; Python float must not pass validation."""
        with pytest.raises(ValueError, match="float values are not allowed"):
            VideoEncoderConfig(vcodec="libsvtav1", crf=2.5)

    @require_libsvtav1
    def test_libsvtav1_extra_crf_rejects_fractional_string(self):
        """INT options reject fractional values even when supplied only via ``extra_options``."""
        with pytest.raises(ValueError, match="float values are not allowed"):
            VideoEncoderConfig(
                vcodec="libsvtav1",
                crf=None,
                extra_options={"crf": "2.5"},
            )

    @require_libsvtav1
    def test_libsvtav1_extra_crf_rejects_float(self):
        with pytest.raises(ValueError, match="float values are not allowed"):
            VideoEncoderConfig(
                vcodec="libsvtav1",
                crf=None,
                extra_options={"crf": 2.5},
            )

    @require_h264
    def test_h264_crf_accepts_float_and_int(self):
        """x264 exposes crf as a FLOAT option, so both int and float are accepted."""
        assert VideoEncoderConfig(vcodec="h264", crf=23).get_codec_options()["crf"] == 23
        assert VideoEncoderConfig(vcodec="h264", crf=23.5).get_codec_options()["crf"] == 23.5

    @require_libsvtav1
    def test_validate_is_rerunnable(self):
        """After mutating a field, validate() re-checks and surfaces new issues."""
        cfg = VideoEncoderConfig(vcodec="libsvtav1")
        cfg.preset = 100  # now out of range
        with pytest.raises(ValueError, match="out of range"):
            cfg.validate()


class TestExtraOptions:
    @require_libsvtav1
    def test_default_is_empty_dict(self):
        cfg = VideoEncoderConfig()
        assert cfg.extra_options == {}

    @require_libsvtav1
    def test_unknown_key_passes_through(self):
        """Keys not published as AVOptions are forwarded to FFmpeg."""
        cfg = VideoEncoderConfig(extra_options={"totally_made_up_option": "value"})
        assert cfg.extra_options == {"totally_made_up_option": "value"}

    @require_libsvtav1
    def test_numeric_value_in_range_ok(self):
        """libsvtav1 exposes ``qp`` as INT in [0, 63]."""
        cfg = VideoEncoderConfig(extra_options={"qp": 30})
        assert cfg.extra_options == {"qp": 30}

    @require_libsvtav1
    def test_numeric_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"qp=.*out of range"):
            VideoEncoderConfig(extra_options={"qp": 999})

    @require_libsvtav1
    def test_numeric_string_accepted_in_range(self):
        """Numeric strings are accepted for numeric options (mirrors FFmpeg)."""
        cfg = VideoEncoderConfig(extra_options={"qp": "18"})
        assert cfg.extra_options == {"qp": "18"}

    @require_libsvtav1
    def test_numeric_string_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"qp=.*out of range"):
            VideoEncoderConfig(extra_options={"qp": "999"})

    @require_libsvtav1
    def test_non_numeric_string_on_numeric_option_raises(self):
        with pytest.raises(ValueError, match=r"qp=.*not numeric"):
            VideoEncoderConfig(extra_options={"qp": "medium"})

    @require_libsvtav1
    def test_bool_on_numeric_option_raises(self):
        """``bool`` is explicitly rejected for numeric options."""
        with pytest.raises(ValueError, match=r"qp=.*not numeric"):
            VideoEncoderConfig(extra_options={"qp": True})

    @require_h264
    def test_string_option_passes_through_unchecked(self):
        """String-typed AVOptions are NOT enum-checked (too many accept freeform)."""
        cfg = VideoEncoderConfig(vcodec="h264", preset=None, extra_options={"tune": "some-future-tune"})
        assert cfg.extra_options == {"tune": "some-future-tune"}

    @require_libsvtav1
    def test_merged_into_codec_options_and_stringified(self):
        """Typed merge by default; ``as_strings=True`` matches FFmpeg option dict."""
        cfg = VideoEncoderConfig(extra_options={"qp": 20})
        opts = cfg.get_codec_options()
        assert opts["qp"] == 20
        assert isinstance(opts["qp"], int)
        assert cfg.get_codec_options(as_strings=True)["qp"] == "20"

    @require_libsvtav1
    def test_structured_fields_win_on_collision(self):
        """A colliding extra_options key is discarded; the structured field wins."""
        cfg = VideoEncoderConfig(crf=30, extra_options={"crf": 18})
        assert cfg.get_codec_options()["crf"] == 30


class TestEncoderDetection:
    @require_h264
    def test_explicit_codec_kept_when_available(self):
        cfg = VideoEncoderConfig(vcodec="h264")
        assert cfg.vcodec == "h264"

    @require_videotoolbox
    def test_auto_picks_videotoolbox_when_available(self):
        """``h264_videotoolbox`` sits at the top of ``HW_VIDEO_CODECS`` so it wins when present."""
        cfg = VideoEncoderConfig(vcodec="auto")
        assert cfg.vcodec == "h264_videotoolbox"

    def test_invalid_codec_raises(self):
        with pytest.raises(ValueError, match="Invalid vcodec"):
            VideoEncoderConfig(vcodec="not_a_real_codec")

    def test_hw_encoder_names_listed_as_valid(self):
        assert "auto" in VALID_VIDEO_CODECS
        assert "h264_videotoolbox" in VALID_VIDEO_CODECS
        assert "h264_nvenc" in VALID_VIDEO_CODECS


TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "encoded_videos"

# Default video feature set used by persistence tests.
VIDEO_FEATURES = {
    "observation.images.cam": {
        "dtype": "video",
        "shape": (64, 96, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}
VIDEO_KEY = "observation.images.cam"


def _write_frames(imgs_dir: Path, num_frames: int = 4, height: int = 64, width: int = 96) -> None:
    imgs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        write_image(arr, imgs_dir / f"frame-{i:06d}.png")


def _encode_video(
    path: Path, num_frames: int = 4, fps: int = 30, cfg: VideoEncoderConfig | None = None
) -> Path:
    imgs_dir = path.parent / f"imgs_{path.stem}"
    _write_frames(imgs_dir, num_frames=num_frames)
    encode_video_frames(imgs_dir, path, fps=fps, camera_encoder=cfg, overwrite=True)
    return path


def _read_feature_info(dataset: LeRobotDataset) -> dict:
    info = json.loads((dataset.root / INFO_PATH).read_text())
    return info["features"][VIDEO_KEY]["info"]


def _add_frames(dataset: LeRobotDataset, num_frames: int) -> None:
    shape = dataset.meta.features[VIDEO_KEY]["shape"]
    for _ in range(num_frames):
        dataset.add_frame(
            {
                VIDEO_KEY: np.random.randint(0, 256, shape, dtype=np.uint8),
                "action": np.zeros(2, dtype=np.float32),
                "task": "test",
            }
        )


class TestGetVideoInfo:
    def test_returns_all_stream_fields(self):
        info = get_video_info(TEST_ARTIFACTS_DIR / "clip_4frames.mp4")

        assert info["video.height"] == 64
        assert info["video.width"] == 96
        assert info["video.pix_fmt"] == "yuv420p"
        assert info["video.fps"] == 30
        assert info["video.channels"] == 3
        assert info["video.is_depth_map"] is False
        assert info["has_audio"] is False
        assert "video.g" not in info
        assert "video.crf" not in info
        assert "video.preset" not in info

    @require_libsvtav1
    def test_merges_encoder_config_as_video_prefixed_entries(self):
        cfg = VideoEncoderConfig(vcodec="libsvtav1", g=2, crf=30, preset=12)

        info = get_video_info(TEST_ARTIFACTS_DIR / "clip_4frames.mp4", camera_encoder=cfg)

        assert info["video.g"] == 2
        assert info["video.crf"] == 30
        assert info["video.preset"] == 12
        assert info["video.fast_decode"] == 0
        assert info["video.video_backend"] == "pyav"
        assert info["video.extra_options"] == {}

    @require_libsvtav1
    def test_stream_derived_keys_take_precedence_over_config(self):
        cfg = VideoEncoderConfig(vcodec="libsvtav1", pix_fmt="yuv420p")

        info = get_video_info(TEST_ARTIFACTS_DIR / "clip_4frames.mp4", camera_encoder=cfg)

        assert info["video.codec"]  # populated from stream, not from config's vcodec
        assert info["video.pix_fmt"] == "yuv420p"


class TestEncodeVideoFrames:
    @require_libsvtav1
    def test_produces_readable_mp4(self, tmp_path):
        video_path = _encode_video(tmp_path / "out.mp4")

        assert video_path.exists()
        info = get_video_info(video_path)
        assert info["video.height"] == 64
        assert info["video.width"] == 96

    @require_libsvtav1
    def test_frame_count_and_duration_match_input(self, tmp_path):
        num_frames = 10
        fps = 30
        video_path = _encode_video(tmp_path / "out.mp4", num_frames=num_frames, fps=fps)

        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            actual_frames = sum(1 for _ in container.decode(stream))
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else float(container.duration / av.time_base)
            )

        assert actual_frames == num_frames
        assert abs(duration - num_frames / fps) < 0.1

    def test_overwrite_false_skips_existing_file(self, tmp_path):
        imgs_dir = tmp_path / "imgs"
        _write_frames(imgs_dir)
        video_path = tmp_path / "out.mp4"
        sentinel = b"pre-existing content"
        video_path.write_bytes(sentinel)

        encode_video_frames(imgs_dir, video_path, fps=30, overwrite=False)

        assert video_path.read_bytes() == sentinel

    @require_libsvtav1
    def test_overwrite_true_replaces_existing_file(self, tmp_path):
        imgs_dir = tmp_path / "imgs"
        _write_frames(imgs_dir)
        video_path = tmp_path / "out.mp4"
        video_path.write_bytes(b"stale content")

        encode_video_frames(imgs_dir, video_path, fps=30, overwrite=True)

        info = get_video_info(video_path)
        assert info["video.height"] == 64

    @require_libsvtav1
    def test_custom_encoder_config_fields_stored_in_info(self, tmp_path):
        """All stream-derived and encoder config fields are present after encoding."""
        cfg = VideoEncoderConfig(vcodec="libsvtav1", g=4, crf=25, preset=10)
        video_path = _encode_video(tmp_path / "out.mp4", num_frames=4, fps=30, cfg=cfg)

        info = get_video_info(video_path, camera_encoder=cfg)

        # Stream-derived
        assert info["video.height"] == 64
        assert info["video.width"] == 96
        assert info["video.channels"] == 3
        assert info["video.codec"] == "av1"
        assert info["video.pix_fmt"] == "yuv420p"
        assert info["video.fps"] == 30
        assert info["video.is_depth_map"] is False
        assert info["has_audio"] is False
        # Encoder config
        assert info["video.g"] == 4
        assert info["video.crf"] == 25
        assert info["video.preset"] == 10
        assert info["video.fast_decode"] == 0
        assert info["video.video_backend"] == "pyav"
        assert info["video.extra_options"] == {}


class TestConcatenateVideoFiles:
    def test_two_clips_frame_count(self, tmp_path):
        """Output frame count equals the sum of the two input frame counts."""
        out = tmp_path / "out.mp4"
        concatenate_video_files(
            [TEST_ARTIFACTS_DIR / "clip_6frames.mp4", TEST_ARTIFACTS_DIR / "clip_4frames.mp4"], out
        )

        with av.open(str(out)) as container:
            total = sum(1 for _ in container.decode(video=0))
        assert total == 10

    def test_three_clips_frame_count(self, tmp_path):
        out = tmp_path / "out.mp4"
        clip = TEST_ARTIFACTS_DIR / "clip_5frames.mp4"
        concatenate_video_files([clip, clip, clip], out)

        with av.open(str(out)) as container:
            total = sum(1 for _ in container.decode(video=0))
        assert total == 15

    @require_libsvtav1
    def test_geometry_preserved(self, tmp_path):
        """Output resolution, fps, codec and pixel format must match the inputs."""
        out = tmp_path / "out.mp4"
        concatenate_video_files(
            [TEST_ARTIFACTS_DIR / "clip_4frames.mp4", TEST_ARTIFACTS_DIR / "clip_4frames.mp4"], out
        )

        info = get_video_info(out)
        assert info["video.height"] == 64
        assert info["video.width"] == 96
        assert info["video.fps"] == 30
        assert info["video.codec"] == "av1"
        assert info["video.pix_fmt"] == "yuv420p"

    def test_compatibility_check_raises_on_different_codec(self, tmp_path):
        with pytest.raises(ValueError):
            concatenate_video_files(
                [TEST_ARTIFACTS_DIR / "clip_4frames.mp4", TEST_ARTIFACTS_DIR / "clip_h264.mp4"],
                tmp_path / "out.mp4",
                compatibility_check=True,
            )

    def test_compatibility_check_raises_on_different_resolution(self, tmp_path):
        with pytest.raises(ValueError):
            concatenate_video_files(
                [TEST_ARTIFACTS_DIR / "clip_4frames.mp4", TEST_ARTIFACTS_DIR / "clip_32x48.mp4"],
                tmp_path / "out.mp4",
                compatibility_check=True,
            )


class TestEncoderConfigPersistence:
    """Encoder config must be stored as ``video.<field>`` entries in
    ``info["features"][key]["info"]`` when the first episode is saved.
    """

    @require_libsvtav1
    def test_first_episode_save_persists_encoder_config(self, tmp_path, empty_lerobot_dataset_factory):
        cfg = VideoEncoderConfig(vcodec="libsvtav1", g=2, crf=30, preset=12)
        dataset = empty_lerobot_dataset_factory(
            root=tmp_path / "ds", features=VIDEO_FEATURES, use_videos=True, camera_encoder=cfg
        )

        _add_frames(dataset, num_frames=4)
        dataset.save_episode()
        dataset.finalize()

        info = _read_feature_info(dataset)

        assert info["video.height"] == 64
        assert info["video.width"] == 96
        assert info["video.fps"] == 30
        assert info["video.g"] == 2
        assert info["video.crf"] == 30
        assert info["video.preset"] == 12
        assert info["video.fast_decode"] == 0
        assert info["video.video_backend"] == "pyav"
        assert info["video.extra_options"] == {}

    @require_libsvtav1
    def test_second_episode_does_not_overwrite_encoder_fields(self, tmp_path, empty_lerobot_dataset_factory):
        cfg = VideoEncoderConfig(vcodec="libsvtav1", g=2, crf=30, preset=12)
        dataset = empty_lerobot_dataset_factory(
            root=tmp_path / "ds", features=VIDEO_FEATURES, use_videos=True, camera_encoder=cfg
        )

        _add_frames(dataset, num_frames=4)
        dataset.save_episode()
        first_info = dict(_read_feature_info(dataset))

        _add_frames(dataset, num_frames=4)
        dataset.save_episode()
        dataset.finalize()

        assert _read_feature_info(dataset) == first_info


class TestFromVideoInfo:
    """``VideoEncoderConfig.from_video_info`` reconstructs an encoder config
    from the ``video.*`` keys persisted in a dataset's ``info.json``.
    """

    @require_libsvtav1
    def test_reconstructs_from_dummy_video_info(self):
        cfg = VideoEncoderConfig.from_video_info(DUMMY_VIDEO_INFO)

        # Canonical stream codec ``"av1"`` is aliased to the encoder name.
        assert cfg.vcodec == "libsvtav1"
        assert cfg.pix_fmt == DUMMY_VIDEO_INFO["video.pix_fmt"]
        assert cfg.g == DUMMY_VIDEO_INFO["video.g"]
        assert cfg.crf == DUMMY_VIDEO_INFO["video.crf"]
        assert cfg.preset == DUMMY_VIDEO_INFO["video.preset"]
        assert cfg.fast_decode == DUMMY_VIDEO_INFO["video.fast_decode"]
        assert cfg.video_backend == DUMMY_VIDEO_INFO["video.video_backend"]
        # ``{}`` placeholder (typical after a merge with disagreeing sources)
        # must not leak into the reconstructed config.
        assert cfg.extra_options == VideoEncoderConfig().extra_options
