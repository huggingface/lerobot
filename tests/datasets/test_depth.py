"""Tests for the depth-integration feature.

Covers quantization/dequantization round-trips (depth_utils), image writer
depth support (image_writer), hardware→dataset feature routing
(feature_utils), video info helpers (video_utils / configs.video), and
feature-to-file-format routing through the dataset writer.

Depth metadata detection on ``LeRobotDatasetMetadata.depth_keys`` (canonical
and legacy marker variants) lives in ``test_dataset_metadata.py``.
"""

from pathlib import Path

import av
import numpy as np
import PIL.Image
import pytest
import torch

from lerobot.configs import DepthEncoderConfig
from lerobot.configs.video import DEPTH_QMAX, VALID_VIDEO_CODECS
from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth
from lerobot.datasets.image_writer import (
    image_array_to_pil_image,
    save_kwargs_for_path,
    write_image,
)
from lerobot.datasets.video_utils import get_video_pixel_channels
from tests.fixtures.constants import (
    DEFAULT_FPS,
    DUMMY_CAMERA_FEATURES,
    DUMMY_DEPTH_CAMERA_FEATURES,
    DUMMY_MOTOR_FEATURES,
    DUMMY_REPO_ID,
)

H, W = 48, 64
DEPTH_MIN = 0.01
DEPTH_MAX = 10.0


# ── 1. Quantize / Dequantize round-trips ────────────────────────────


class TestQuantizeDequantize:
    """Core numerical tests for depth_utils.quantize_depth / dequantize_depth."""

    def _make_depth_metres(self) -> np.ndarray:
        """Linearly-spaced float32 depth in metres covering the default range."""
        return np.linspace(DEPTH_MIN, DEPTH_MAX, H * W, dtype=np.float32).reshape(H, W)

    def test_roundtrip_linear_metres(self):
        depth = self._make_depth_metres()
        quantized = quantize_depth(depth, use_log=False, video_backend=None)
        recovered = dequantize_depth(quantized, use_log=False, output_unit="m")

        assert recovered.shape == (H, W, 1), f"Expected (H,W,1), got {recovered.shape}"
        assert recovered.dtype == np.float32
        tol = (DEPTH_MAX - DEPTH_MIN) / DEPTH_QMAX
        np.testing.assert_allclose(recovered[..., 0], depth, atol=tol + 1e-6)

    def test_roundtrip_log_metres(self):
        depth = self._make_depth_metres()
        quantized = quantize_depth(depth, use_log=True, video_backend=None)
        recovered = dequantize_depth(quantized, use_log=True, output_unit="m")

        assert recovered.shape == (H, W, 1)
        near = depth < 1.0
        far = depth > 8.0
        err_near = np.abs(recovered[..., 0][near] - depth[near])
        err_far = np.abs(recovered[..., 0][far] - depth[far])
        assert err_near.mean() < err_far.mean(), "Log quant should be more precise at close range"

    def test_roundtrip_mm_uint16_input(self):
        depth_mm = np.linspace(10, 10000, H * W, dtype=np.float64).reshape(H, W).astype(np.uint16)
        quantized = quantize_depth(depth_mm, use_log=False, video_backend=None, input_unit="mm")
        recovered = dequantize_depth(quantized, use_log=False, output_unit="mm")

        assert recovered.dtype == np.uint16
        tol_mm = (DEPTH_MAX - DEPTH_MIN) * 1000.0 / DEPTH_QMAX
        np.testing.assert_allclose(
            recovered[..., 0].astype(np.float64), depth_mm.astype(np.float64), atol=tol_mm + 1.0
        )

    def test_quantize_clamps_out_of_range(self):
        depth = np.array([[0.001, 99.0]], dtype=np.float32)
        quantized = quantize_depth(depth, use_log=False, video_backend=None)
        assert quantized[0, 0] == 0
        assert quantized[0, 1] == DEPTH_QMAX

    def test_quantize_accepts_torch_tensor(self):
        t = torch.rand(H, W, dtype=torch.float32) * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
        result = quantize_depth(t, video_backend=None)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint16

    def test_quantize_squeezes_channel_dim(self):
        depth = self._make_depth_metres()
        for shape in [(H, W, 1), (1, H, W)]:
            reshaped = depth.reshape(shape)
            quantized = quantize_depth(reshaped, video_backend=None)
            assert quantized.ndim == 2, f"Input shape {shape} should be squeezed to 2D"

    def test_quantize_returns_pyav_frame(self):
        depth = self._make_depth_metres()
        result = quantize_depth(depth, video_backend="pyav")
        assert isinstance(result, av.VideoFrame)

    def test_dequantize_output_tensor(self):
        quantized = np.full((H, W), DEPTH_QMAX // 2, dtype=np.uint16)
        result = dequantize_depth(quantized, output_unit="m", output_tensor=True)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (H, W, 1)

    def test_invalid_log_params_raises(self):
        depth = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="depth_min \\+ shift must be positive"):
            quantize_depth(depth, depth_min=1.0, shift=-2.0, use_log=True, video_backend=None)


# ── 2. Image writer depth support ───────────────────────────────────


class TestImageWriterDepth:
    """image_array_to_pil_image and write_image for single-channel depth maps."""

    def test_pil_uint16_grayscale(self):
        arr = np.arange(H * W, dtype=np.uint16).reshape(H, W)
        img = image_array_to_pil_image(arr)
        assert isinstance(img, PIL.Image.Image)
        assert img.mode == "I;16"
        assert img.size == (W, H)

    def test_pil_float32_grayscale(self):
        arr = np.random.rand(H, W).astype(np.float32)
        img = image_array_to_pil_image(arr)
        assert img.mode == "F"

    def test_pil_squeeze_hwc1_and_1hw(self):
        arr_uint16 = np.zeros((H, W), dtype=np.uint16)
        for input_arr in [arr_uint16.reshape(H, W, 1), arr_uint16.reshape(1, H, W)]:
            img = image_array_to_pil_image(input_arr)
            assert img.size == (W, H)

    def test_save_kwargs_png_vs_tiff(self):
        png_kw = save_kwargs_for_path(Path("frame.png"), compress_level=5)
        assert png_kw == {"compress_level": 5}

        tiff_kw = save_kwargs_for_path(Path("frame.tiff"), compress_level=5)
        assert tiff_kw == {"compression": "raw"}

        assert save_kwargs_for_path(Path("frame.jpg"), compress_level=5) == {}

    def test_write_image_tiff_roundtrip(self, tmp_path):
        arr = np.arange(H * W, dtype=np.uint16).reshape(H, W)
        fpath = tmp_path / "depth.tiff"
        write_image(arr, fpath)

        assert fpath.exists()
        with PIL.Image.open(fpath) as loaded:
            recovered = np.array(loaded)
        np.testing.assert_array_equal(recovered, arr)


# ── 3. Feature routing ──────────────────────────────────────────────


class TestHwToDatasetFeaturesDepth:
    """hw_to_dataset_features marks single-channel cameras as depth."""

    def test_single_channel_cam_marked_depth(self):
        from lerobot.utils.feature_utils import hw_to_dataset_features

        features = hw_to_dataset_features({"cam": (480, 640, 1)}, prefix="observation")
        ft = features["observation.images.cam"]
        assert ft["info"]["is_depth_map"] is True

    def test_three_channel_cam_not_depth(self):
        from lerobot.utils.feature_utils import hw_to_dataset_features

        features = hw_to_dataset_features({"cam": (480, 640, 3)}, prefix="observation")
        ft = features["observation.images.cam"]
        assert ft["info"]["is_depth_map"] is False

    def test_invalid_channel_count_raises(self):
        from lerobot.utils.feature_utils import hw_to_dataset_features

        with pytest.raises(ValueError, match="Expected a 3-tuple"):
            hw_to_dataset_features({"cam": (480, 640, 2)}, prefix="observation")


# ── 4. Video info depth flag ────────────────────────────────────────


class TestVideoInfoDepthFlag:
    """Misc depth-related constants and helpers in video_utils / configs."""

    def test_get_video_pixel_channels_gray(self):
        assert get_video_pixel_channels("gray12le") == 1
        assert get_video_pixel_channels("gray8") == 1

    def test_ffv1_in_valid_codecs(self):
        assert "ffv1" in VALID_VIDEO_CODECS


# ── 5. Feature-to-file-format routing ───────────────────────────────


def _build_mixed_features(dtype: str) -> dict:
    """Build a feature dict with one RGB camera and one depth camera.

    Uses shapes from ``DUMMY_CAMERA_FEATURES`` and ``DUMMY_DEPTH_CAMERA_FEATURES``
    defined in ``tests.fixtures.constants``.
    """
    rgb_cam = next(iter(DUMMY_CAMERA_FEATURES.values()))
    depth_cam = next(iter(DUMMY_DEPTH_CAMERA_FEATURES.values()))
    return {
        "observation.images.rgb": {"dtype": dtype, **rgb_cam},
        "observation.images.depth": {"dtype": dtype, **depth_cam},
        **{k: {"dtype": v["dtype"], **v} for k, v in DUMMY_MOTOR_FEATURES.items()},
    }


def _make_mixed_frame(features: dict) -> dict:
    """Build a valid frame dict matching the given feature schema."""
    frame: dict = {"task": "test task"}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ("image", "video"):
            channels = shape[-1]
            if channels == 1:
                frame[key] = np.random.randint(0, 4095, shape, dtype=np.uint16)
            else:
                frame[key] = np.random.randint(0, 255, shape, dtype=np.uint8)
        else:
            frame[key] = np.random.randn(*shape).astype(ft["dtype"])
    return frame


class TestFeatureFileRouting:
    """Verify that depth vs RGB features are routed to the correct file format."""

    NUM_FRAMES = 5

    def test_no_video_depth_tiff_rgb_png(self, tmp_path):
        """Without video encoding: depth -> .tiff, RGB -> .png."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = _build_mixed_features(dtype="image")

        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=False,
        )

        for _ in range(self.NUM_FRAMES):
            dataset.add_frame(_make_mixed_frame(features))

        buf = dataset.writer.episode_buffer
        depth_paths = [Path(p) for p in buf["observation.images.depth"]]
        rgb_paths = [Path(p) for p in buf["observation.images.rgb"]]

        assert all(p.suffix == ".tiff" for p in depth_paths), "Depth frames should be .tiff"
        assert all(p.suffix == ".png" for p in rgb_paths), "RGB frames should be .png"
        assert all(p.exists() for p in depth_paths), "Depth TIFF files should exist on disk"
        assert all(p.exists() for p in rgb_paths), "RGB PNG files should exist on disk"

        dataset.save_episode()
        dataset.finalize()

    def test_video_depth_uses_depth_encoder(self, tmp_path):
        """With streaming video encoding: depth keys use DepthEncoderConfig, RGB keys do not."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = _build_mixed_features(dtype="video")

        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=True,
            streaming_encoding=True,
        )

        assert dataset.writer._streaming_encoder is not None
        encoder = dataset.writer._streaming_encoder

        for _ in range(self.NUM_FRAMES):
            dataset.add_frame(_make_mixed_frame(features))

        rgb_thread = encoder._threads["observation.images.rgb"]
        depth_thread = encoder._threads["observation.images.depth"]

        assert not isinstance(rgb_thread.video_encoder, DepthEncoderConfig)
        assert isinstance(depth_thread.video_encoder, DepthEncoderConfig)
        assert depth_thread.is_depth is True
        assert rgb_thread.is_depth is False

        dataset.save_episode()
        dataset.finalize()
