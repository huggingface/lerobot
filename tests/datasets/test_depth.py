"""Tests for the depth-integration feature.

Covers:
- ``depth_utils`` quantize/dequantize round-trips and backend agreement.
- Image-writer support for single-channel depth.
- Hardware-feature → depth flag routing.
- Feature-to-file-format routing through the dataset writer.

Depth metadata detection on ``LeRobotDatasetMetadata.depth_keys`` lives in
``test_dataset_metadata.py``. Depth video encoding/decoding lives in
``test_video_encoding.py``.
"""

from pathlib import Path

import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import av
import numpy as np
import PIL.Image
import torch

from lerobot.configs import DepthEncoderConfig
from lerobot.configs.video import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEPTH_METER_UNIT,
    DEPTH_MILLIMETER_UNIT,
    DEPTH_QMAX,
)
from lerobot.datasets.depth_utils import dequantize_depth, quantize_depth
from lerobot.datasets.image_writer import image_array_to_pil_image, write_image
from lerobot.utils.constants import DEFAULT_FEATURES
from tests.fixtures.constants import (
    DEFAULT_FPS,
    DUMMY_CAMERA_FEATURES,
    DUMMY_CAMERA_FEATURES_WITH_DEPTH,
    DUMMY_CHW,
    DUMMY_DEPTH_CAMERA_FEATURES,
    DUMMY_REPO_ID,
)
from tests.fixtures.dataset_factories import add_frames

_, H, W = DUMMY_CHW


def _depth_metres_ramp() -> np.ndarray:
    """Linearly-spaced float32 depth in metres covering the default range."""
    return np.linspace(DEFAULT_DEPTH_MIN, DEFAULT_DEPTH_MAX, H * W, dtype=np.float32).reshape(H, W)


# ── 1. Quantize / dequantize round-trips ──────────────────────────────


class TestQuantizeDequantize:
    """Numerical contract of ``quantize_depth`` / ``dequantize_depth``."""

    @pytest.mark.parametrize("use_log", [False, True])
    @pytest.mark.parametrize("output_unit", [DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT])
    @pytest.mark.parametrize("output_channel_last", [False, True])
    def test_roundtrip(self, use_log, output_unit, output_channel_last):
        """quantize → dequantize recovers depth; layout and unit are honored."""
        depth = _depth_metres_ramp()
        quantized = quantize_depth(depth, use_log=use_log, video_backend=None)
        recovered = dequantize_depth(
            quantized,
            use_log=use_log,
            output_unit=output_unit,
            output_tensor=False,
            output_channel_last=output_channel_last,
        )

        expected_shape = (H, W, 1) if output_channel_last else (1, H, W)
        assert recovered.shape == expected_shape

        recovered_m = recovered.astype(np.float32)
        if output_unit == DEPTH_MILLIMETER_UNIT:
            recovered_m = recovered_m / 1000.0
        recovered_2d = recovered_m[..., 0] if output_channel_last else recovered_m[0]

        if use_log:
            # Log mode: tighter near-range error than far-range (the whole point).
            near = depth < 1.0
            far = depth > 8.0
            err_near = np.abs(recovered_2d[near] - depth[near])
            err_far = np.abs(recovered_2d[far] - depth[far])
            assert err_near.mean() < err_far.mean()
        else:
            # Linear mode: bounded by quant step + 1 mm of unit-conversion rounding.
            tol = (DEFAULT_DEPTH_MAX - DEFAULT_DEPTH_MIN) / DEPTH_QMAX + 1e-3
            np.testing.assert_allclose(recovered_2d, depth, atol=tol)

    @pytest.mark.parametrize("use_log", [False, True])
    @pytest.mark.parametrize("output_unit", [DEPTH_METER_UNIT, DEPTH_MILLIMETER_UNIT])
    def test_numpy_torch_agree(self, use_log, output_unit):
        """Batched torch path produces the same values as the numpy path."""
        batch_size = 3
        per_frame = np.linspace(0, DEPTH_QMAX, H * W, dtype=np.uint16).reshape(H, W)
        batch_np = np.broadcast_to(per_frame[None, None, ...], (batch_size, 1, H, W)).copy()
        batch_t = torch.from_numpy(batch_np.astype(np.int32))  # torch.uint16 support is patchy.

        ref = dequantize_depth(batch_np, use_log=use_log, output_unit=output_unit, output_tensor=False)
        out = dequantize_depth(batch_t, use_log=use_log, output_unit=output_unit, output_tensor=True)

        assert isinstance(out, torch.Tensor)
        assert out.shape == (batch_size, 1, H, W)
        # ``m``: float32 noise (~10 µm in log mode, after ``exp``) — still 200× below the ~2 mm quant step.
        # ``mm`` + tensor stays in float32 (no uint16 round-trip), so allow 1 mm slop.
        atol = 1e-5 if output_unit == DEPTH_METER_UNIT else 1.0
        np.testing.assert_allclose(out.cpu().numpy().astype(np.float64), ref.astype(np.float64), atol=atol)

    @pytest.mark.parametrize(
        "input_shape,output_shape",
        [
            ((H, W), (1, H, W)),
            ((1, H, W), (1, H, W)),
            ((H, W, 1), (1, H, W)),
            ((3, 1, H, W), (3, 1, H, W)),
            ((3, H, W, 1), (3, 1, H, W)),
        ],
    )
    def test_input_layouts_accepted(self, input_shape, output_shape):
        """All documented input layouts decode to the channel-first default."""
        quantized = np.full(input_shape, DEPTH_QMAX // 2, dtype=np.uint16)
        out = dequantize_depth(quantized, output_unit=DEPTH_METER_UNIT, output_tensor=False)
        assert out.shape == output_shape

    def test_pyav_frame_roundtrip(self):
        """quantize → av.VideoFrame → dequantize works."""
        depth = _depth_metres_ramp()
        frame = quantize_depth(depth, use_log=False, video_backend="pyav")
        assert isinstance(frame, av.VideoFrame)

        recovered = dequantize_depth(frame, use_log=False, output_unit=DEPTH_METER_UNIT, output_tensor=False)
        assert recovered.shape == (1, H, W)
        tol = (DEFAULT_DEPTH_MAX - DEFAULT_DEPTH_MIN) / DEPTH_QMAX + 1e-3
        np.testing.assert_allclose(recovered[0], depth, atol=tol)

    def test_invalid_log_params_raises(self):
        with pytest.raises(ValueError, match=r"depth_min \+ shift must be positive"):
            quantize_depth(_depth_metres_ramp(), depth_min=1.0, shift=-2.0, use_log=True, video_backend=None)


# ── 2. Image writer depth support ─────────────────────────────────────


class TestImageWriterDepth:
    """``image_array_to_pil_image`` and ``write_image`` for depth maps."""

    @pytest.mark.parametrize("dtype,expected_mode", [(np.uint16, "I;16"), (np.float32, "F")])
    @pytest.mark.parametrize("shape", [(H, W), (H, W, 1), (1, H, W)])
    def test_pil_depth_modes_and_squeeze(self, dtype, expected_mode, shape):
        """Single-channel depth converts to PIL with the right mode and (W, H) size."""
        arr = np.zeros(shape, dtype=dtype)
        img = image_array_to_pil_image(arr)
        assert img.mode == expected_mode
        assert img.size == (W, H)

    def test_write_image_tiff_roundtrip(self, tmp_path):
        """uint16 depth round-trips through .tiff."""
        arr = np.arange(H * W, dtype=np.uint16).reshape(H, W)
        fpath = tmp_path / "depth.tiff"
        write_image(arr, fpath)
        with PIL.Image.open(fpath) as loaded:
            recovered = np.array(loaded)
        np.testing.assert_array_equal(recovered, arr)


# ── 3. Hardware-feature → depth flag ──────────────────────────────────


class TestHwToDatasetFeaturesDepth:
    """``hw_to_dataset_features`` flags single-channel cameras as depth."""

    @pytest.mark.parametrize("channels,is_depth", [(1, True), (3, False)])
    def test_depth_marker_by_channels(self, channels, is_depth):
        from lerobot.utils.feature_utils import hw_to_dataset_features

        features = hw_to_dataset_features({"cam": (480, 640, channels)}, prefix="observation")
        assert features["observation.images.cam"]["info"]["is_depth_map"] is is_depth

    def test_invalid_channel_count_raises(self):
        from lerobot.utils.feature_utils import hw_to_dataset_features

        with pytest.raises(ValueError, match="Expected a 3-tuple"):
            hw_to_dataset_features({"cam": (480, 640, 2)}, prefix="observation")


# ── 4. Feature-to-file-format routing ────────────────────────────────


# Keys derived from DUMMY_CAMERA_FEATURES_WITH_DEPTH; pick one RGB and the depth camera.
RGB_KEY = next(iter(DUMMY_CAMERA_FEATURES))
DEPTH_KEY = next(iter(DUMMY_DEPTH_CAMERA_FEATURES))


class TestFeatureFileRouting:
    """Depth vs RGB features route to the correct file format."""

    NUM_FRAMES = 5

    def test_image_mode_depth_tiff_rgb_png(self, tmp_path, features_factory):
        """Without video encoding: depth → .tiff, RGB → .png."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = features_factory(camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH, use_videos=False)
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=False,
        )

        add_frames(dataset, num_frames=self.NUM_FRAMES)

        buf = dataset.writer.episode_buffer
        assert all(Path(p).suffix == ".tiff" for p in buf[DEPTH_KEY])
        assert all(Path(p).suffix == ".png" for p in buf[RGB_KEY])

        dataset.save_episode()
        dataset.finalize()

    def test_video_mode_depth_uses_depth_encoder(self, tmp_path, features_factory):
        """With streaming video encoding: depth → DepthEncoderConfig, RGB does not."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = features_factory(camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH, use_videos=True)
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=tmp_path / "ds",
            use_videos=True,
            streaming_encoding=True,
        )

        add_frames(dataset, num_frames=self.NUM_FRAMES)

        encoder = dataset.writer._streaming_encoder
        assert encoder is not None
        assert isinstance(encoder._threads[DEPTH_KEY].video_encoder, DepthEncoderConfig)
        assert not isinstance(encoder._threads[RGB_KEY].video_encoder, DepthEncoderConfig)

        dataset.save_episode()
        dataset.finalize()


class TestDepthUnitMetadata:
    """The depth unit is inferred once from dtype, stored in ``info``, and drives stats + reads."""

    NUM_FRAMES = 4

    def _record(self, root, features_factory, depth_dtype, value, use_videos):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = features_factory(camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH, use_videos=use_videos)
        dataset = LeRobotDataset.create(
            repo_id=DUMMY_REPO_ID,
            fps=DEFAULT_FPS,
            features=features,
            root=root,
            use_videos=use_videos,
            streaming_encoding=use_videos,
        )
        for _ in range(self.NUM_FRAMES):
            frame: dict = {"task": "test"}
            for key, ft in dataset.meta.features.items():
                if key in DEFAULT_FEATURES:
                    continue
                if key in dataset.meta.depth_keys:
                    frame[key] = np.full(ft["shape"], value, dtype=depth_dtype)
                elif key in dataset.meta.camera_keys:
                    frame[key] = np.random.randint(0, 256, ft["shape"], dtype=np.uint8)
                else:
                    frame[key] = np.zeros(ft["shape"], dtype=np.float32)
            dataset.add_frame(frame)
        return dataset

    @pytest.mark.parametrize("use_videos", [False, True])
    @pytest.mark.parametrize(
        ("depth_dtype", "value", "expected_unit"),
        [(np.float32, 2.0, DEPTH_METER_UNIT), (np.uint16, 2000, DEPTH_MILLIMETER_UNIT)],
    )
    def test_recorded_unit_inferred_persisted_and_kept_in_stats(
        self, tmp_path, features_factory, use_videos, depth_dtype, value, expected_unit
    ):
        """Unit is inferred from the first frame's dtype, drives stats (raw, never canonicalized), and survives a reload."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = self._record(tmp_path / "ds", features_factory, depth_dtype, value, use_videos)
        assert dataset.meta.features[DEPTH_KEY]["info"]["depth_unit"] == expected_unit
        dataset.save_episode()
        mean = float(np.asarray(dataset.meta.stats[DEPTH_KEY]["mean"]).reshape(-1)[0])
        np.testing.assert_allclose(mean, value, rtol=0.05)
        dataset.finalize()

        reloaded = LeRobotDataset(repo_id=DUMMY_REPO_ID, root=tmp_path / "ds")
        assert reloaded.meta.features[DEPTH_KEY]["info"]["depth_unit"] == expected_unit

    @pytest.mark.parametrize("use_videos", [False, True])
    @pytest.mark.parametrize(
        ("output_unit", "expected"),
        [(DEPTH_MILLIMETER_UNIT, 2000.0), (DEPTH_METER_UNIT, 2.0)],
    )
    def test_read_honors_output_unit_for_frames_and_stats(
        self, tmp_path, features_factory, use_videos, output_unit, expected
    ):
        """Reloading with a ``depth_output_unit`` converts metre frames (image mode) and rescales stats while preserving count."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = self._record(tmp_path / "ds", features_factory, np.float32, 2.0, use_videos=use_videos)
        dataset.save_episode()
        count = float(np.asarray(dataset.meta.stats[DEPTH_KEY]["count"]).reshape(-1)[0])
        dataset.finalize()

        read_dataset = LeRobotDataset(
            repo_id=DUMMY_REPO_ID, root=tmp_path / "ds", depth_output_unit=output_unit
        )
        stats = read_dataset.meta.stats[DEPTH_KEY]
        np.testing.assert_allclose(float(np.asarray(stats["mean"]).reshape(-1)[0]), expected, rtol=0.05)
        np.testing.assert_allclose(float(np.asarray(stats["count"]).reshape(-1)[0]), count)

        if not use_videos:
            depth = read_dataset[0][DEPTH_KEY]
            assert torch.allclose(depth, torch.full_like(depth, expected))

            from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

            stream_dataset = StreamingLeRobotDataset(
                repo_id=DUMMY_REPO_ID, root=tmp_path / "ds", depth_output_unit=output_unit
            )
            stream_depth = next(iter(stream_dataset))[DEPTH_KEY]
            assert torch.allclose(stream_depth, torch.full_like(stream_depth, expected))
