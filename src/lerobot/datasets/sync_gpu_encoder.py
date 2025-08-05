"""
Synchronous GPU-accelerated video encoding utilities for LeRobot.

This module provides synchronous GPU-accelerated video encoding capabilities
that can be used independently of async encoding. It's designed to replace
the traditional CPU-based video encoding with GPU acceleration while maintaining
the same synchronous interface.
"""

import logging
import time
from pathlib import Path

from .gpu_video_encoder import GPUVideoEncoder, create_gpu_encoder_config
from .video_utils import encode_video_frames

logger = logging.getLogger(__name__)


class SyncGPUVideoEncoder:
    """
    Synchronous GPU-accelerated video encoder.

    This class provides GPU-accelerated video encoding with a synchronous interface,
    making it a drop-in replacement for the traditional CPU-based encoding.
    """

    def __init__(
        self,
        gpu_encoding: bool = False,
        gpu_encoder_config: dict | None = None,
        enable_logging: bool = True,
    ):
        """
        Initialize the synchronous GPU video encoder.

        Args:
            gpu_encoding: Whether to use GPU acceleration
            gpu_encoder_config: Configuration for GPU encoding
            enable_logging: Whether to enable detailed logging
        """
        self.gpu_encoding = gpu_encoding
        self.enable_logging = enable_logging
        self.gpu_encoder = None

        if self.gpu_encoding:
            config = gpu_encoder_config or {}
            gpu_config = create_gpu_encoder_config(
                encoder_type=config.get("encoder_type", "auto"),
                codec=config.get("codec", "h264"),
                preset=config.get("preset", "fast"),
                quality=config.get("quality", 23),
                bitrate=config.get("bitrate"),
                gpu_id=config.get("gpu_id", 0),
                enable_logging=enable_logging,
            )
            self.gpu_encoder = GPUVideoEncoder(gpu_config)

            if self.enable_logging:
                encoder_info = self.gpu_encoder.get_encoder_info()
                logger.info(
                    f"Sync GPU encoder initialized: {encoder_info['selected_encoder']} {encoder_info['selected_codec']}"
                )
        else:
            if self.enable_logging:
                logger.info("Sync GPU encoder initialized: CPU encoding only")

    def encode_episode_videos(
        self, episode_index: int, video_keys: list[str], fps: int, root_path: Path
    ) -> None:
        """
        Encode videos for a single episode using GPU or CPU acceleration.

        This method provides the same interface as the traditional CPU encoding
        but uses GPU acceleration when enabled.

        Args:
            episode_index: Index of the episode to encode
            video_keys: List of video keys to encode
            fps: Frames per second
            root_path: Root path of the dataset
        """
        if self.enable_logging:
            logger.info(
                f"Encoding videos for episode {episode_index} using {'GPU' if self.gpu_encoding else 'CPU'} acceleration"
            )

        start_time = time.time()

        for video_key in video_keys:
            # Construct paths
            video_path = (
                root_path
                / "videos"
                / f"chunk-{episode_index // 1000:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            img_dir = root_path / "images" / video_key / f"episode_{episode_index:06d}"

            # Ensure video directory exists
            video_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if video already exists
            if video_path.exists():
                if self.enable_logging:
                    logger.debug(f"Video {video_path} already exists, skipping")
                continue

            # Encode the video using GPU or CPU
            if self.gpu_encoding and self.gpu_encoder:
                # Use GPU encoding
                success = self.gpu_encoder.encode_video(input_dir=img_dir, output_path=video_path, fps=fps)

                if not success:
                    if self.enable_logging:
                        logger.warning(f"GPU encoding failed for {video_path}, falling back to CPU encoding")
                    # Fallback to CPU encoding
                    encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=fps, overwrite=True)
            else:
                # Use CPU encoding
                encode_video_frames(imgs_dir=img_dir, video_path=video_path, fps=fps, overwrite=True)

            if self.enable_logging:
                logger.debug(f"Encoded video: {video_path}")

        encoding_time = time.time() - start_time

        if self.enable_logging:
            logger.info(
                f"Episode {episode_index} encoding completed in {encoding_time:.2f}s using {'GPU' if self.gpu_encoding else 'CPU'} acceleration"
            )

    def batch_encode_videos(
        self, start_episode: int, end_episode: int, video_keys: list[str], fps: int, root_path: Path
    ) -> None:
        """
        Batch encode videos for multiple episodes using GPU or CPU acceleration.

        Args:
            start_episode: Starting episode index (inclusive)
            end_episode: Ending episode index (exclusive)
            video_keys: List of video keys to encode
            fps: Frames per second
            root_path: Root path of the dataset
        """
        if self.enable_logging:
            logger.info(
                f"Starting batch video encoding for episodes {start_episode} to {end_episode - 1} using {'GPU' if self.gpu_encoding else 'CPU'} acceleration"
            )

        start_time = time.time()

        # Encode all episodes
        for ep_idx in range(start_episode, end_episode):
            if self.enable_logging:
                logger.info(f"Encoding videos for episode {ep_idx}")

            self.encode_episode_videos(ep_idx, video_keys, fps, root_path)

        total_time = time.time() - start_time

        if self.enable_logging:
            logger.info(
                f"Batch video encoding completed in {total_time:.2f}s using {'GPU' if self.gpu_encoding else 'CPU'} acceleration"
            )

    def get_encoder_info(self) -> dict:
        """Get information about the current encoder configuration."""
        if self.gpu_encoding and self.gpu_encoder:
            return self.gpu_encoder.get_encoder_info()
        else:
            return {
                "selected_encoder": "cpu",
                "selected_codec": "h264",
                "config": {"encoder_type": "software", "codec": "h264", "preset": "medium", "quality": 23},
            }


def create_sync_gpu_encoder(
    gpu_encoding: bool = False, gpu_encoder_config: dict | None = None, enable_logging: bool = True
) -> SyncGPUVideoEncoder:
    """
    Create a synchronous GPU video encoder.

    Args:
        gpu_encoding: Whether to use GPU acceleration
        gpu_encoder_config: Configuration for GPU encoding
        enable_logging: Whether to enable detailed logging

    Returns:
        Configured SyncGPUVideoEncoder instance
    """
    return SyncGPUVideoEncoder(
        gpu_encoding=gpu_encoding, gpu_encoder_config=gpu_encoder_config, enable_logging=enable_logging
    )


def test_sync_gpu_encoding():
    """Test synchronous GPU encoding capabilities."""
    print("=" * 80)
    print("SYNCHRONOUS GPU ENCODING TEST")
    print("=" * 80)

    dataset_path = Path("datasets/async_strawberry_picking")

    if not dataset_path.exists():
        print("❌ Test dataset not found. Please run a recording session first.")
        return False

    # Test different configurations
    configs = [
        {"name": "CPU Encoding (Baseline)", "gpu_encoding": False, "config": None},
        {
            "name": "GPU Encoding (NVIDIA NVENC H.264)",
            "gpu_encoding": True,
            "config": {"encoder_type": "nvenc", "codec": "h264", "preset": "fast", "quality": 23},
        },
        {
            "name": "GPU Encoding (NVIDIA NVENC HEVC)",
            "gpu_encoding": True,
            "config": {"encoder_type": "nvenc", "codec": "hevc", "preset": "fast", "quality": 23},
        },
    ]

    for test_config in configs:
        print(f"\n--- Testing {test_config['name']} ---")

        # Create sync GPU encoder
        encoder = create_sync_gpu_encoder(
            gpu_encoding=test_config["gpu_encoding"],
            gpu_encoder_config=test_config["config"],
            enable_logging=True,
        )

        # Test encoding episode 0
        start_time = time.time()
        encoder.encode_episode_videos(
            episode_index=0,
            video_keys=["observation.images.front", "observation.images.wrist"],
            fps=30,
            root_path=dataset_path,
        )
        encoding_time = time.time() - start_time

        print(f"  Encoding time: {encoding_time:.2f}s")

        # Get encoder info
        info = encoder.get_encoder_info()
        print(f"  Encoder: {info['selected_encoder']} {info['selected_codec']}")

        # Check for video files
        video_files = list(dataset_path.glob("videos/**/*.mp4"))
        print(f"  Video files: {len(video_files)}")

        if len(video_files) > 0:
            print(f"  ✅ {test_config['name']} test PASSED!")
        else:
            print(f"  ❌ {test_config['name']} test FAILED!")
            return False

    print("\n" + "=" * 80)
    print("ALL SYNCHRONOUS GPU ENCODING TESTS COMPLETED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_sync_gpu_encoding()
