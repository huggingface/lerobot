import gc
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

os.environ["SVT_LOG"] = "0"

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    d = tmp_path / "test_crash_recovery"
    if d.exists():
        shutil.rmtree(d)
    return d


class TestCrashRecovery:
    """Test suite for checkpoint crash recovery functionality."""

    REPO_ID = "test/crash_recovery"
    FPS = 30
    FRAMES_PER_EPISODE = 10
    IMAGE_SIZE = (30, 30, 3)

    @staticmethod
    def create_solid_color_frame(episode_index: int, size: tuple = (30, 30, 3)) -> np.ndarray:
        """Create a solid color frame where RGB = (episode_index, episode_index, episode_index).

        This allows us to verify video content by checking pixel values match episode index.
        """
        # Clamp to valid range [0, 255]
        color_value = min(episode_index, 255)
        return np.full(size, color_value, dtype=np.uint8)

    @staticmethod
    def simulate_crash(dataset: LeRobotDataset):
        """Simulate a hard crash by preventing graceful shutdown."""
        dataset._close_writer = lambda: None
        dataset.meta._close_writer = lambda: None
        if dataset.writer:
            dataset.writer.close = lambda: None
        if dataset.meta.writer:
            dataset.meta.writer.close = lambda: None
        del dataset
        gc.collect()

    def create_dataset(self, root: Path) -> LeRobotDataset:
        """Create a fresh dataset for testing."""
        return LeRobotDataset.create(
            repo_id=self.REPO_ID,
            fps=self.FPS,
            root=root,
            features={
                "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
                "action": {"dtype": "float32", "shape": (10,), "names": ["action"]},
                "observation.image": {
                    "dtype": "video",
                    "shape": self.IMAGE_SIZE,
                    "names": ["height", "width", "channels"],
                },
            },
            use_videos=True,
        )

    def add_episode(self, dataset: LeRobotDataset, episode_index: int):
        """Add a single episode with solid color frames and unique task name."""
        task_name = f"task_ep{episode_index}"  # Unique task per episode
        for _frame_idx in range(self.FRAMES_PER_EPISODE):
            frame = {
                "state": np.ones(10, dtype=np.float32) * episode_index,
                "action": np.ones(10, dtype=np.float32) * episode_index,
                "task": task_name,
                "observation.image": self.create_solid_color_frame(episode_index),
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    def verify_video_content(self, dataset: LeRobotDataset, expected_episodes: list[int]):
        """Verify video frames have correct solid color values matching episode index."""
        for ep_idx in expected_episodes:
            start_frame = ep_idx * self.FRAMES_PER_EPISODE
            frame = dataset[start_frame]

            assert "observation.image" in frame, f"Missing image for episode {ep_idx}"

            # Frame shape is (C, H, W) after transforms
            image = frame["observation.image"]
            assert image.shape == (3, 30, 30), f"Wrong shape: {image.shape}"

            # Check that all pixels have the expected color value
            expected_color = min(ep_idx, 255)
            actual_color = image[0, 0, 0].item()  # Get first channel, first pixel

            # Allow small tolerance for video compression artifacts
            assert abs(actual_color - expected_color) < 10, (
                f"Episode {ep_idx}: expected color ~{expected_color}, got {actual_color}"
            )

    def _setup_crash_scenario(self, tmp_dataset_dir) -> LeRobotDataset:
        """Create dataset, add 7 episodes, checkpoint at 5, crash, and return recovered dataset."""
        dataset = self.create_dataset(tmp_dataset_dir)

        # Add 7 episodes, checkpoint after episode 5
        checkpoint_at = 5
        total_episodes = 7

        for ep in range(total_episodes):
            self.add_episode(dataset, ep)
            if (ep + 1) == checkpoint_at:
                dataset.checkpoint()

        # Simulate crash
        self.simulate_crash(dataset)

        # Reload and return
        return LeRobotDataset(repo_id=self.REPO_ID, root=tmp_dataset_dir)

    def test_crash_recovery_after_checkpoint(self, tmp_dataset_dir):
        """
        Test scenario: Checkpoint at episode 5, continue to episode 7, then 'crash'.

        Expectation: Only episodes 0-4 (checkpointed) should be recoverable.
        """
        dataset_reloaded = self._setup_crash_scenario(tmp_dataset_dir)

        assert dataset_reloaded.num_episodes == 5
        assert len(dataset_reloaded) == 50  # 5 episodes * 10 frames

        # Verify frame indices are consistent
        for i in range(len(dataset_reloaded)):
            frame = dataset_reloaded[i]
            assert frame["episode_index"] == i // self.FRAMES_PER_EPISODE
            assert frame["frame_index"] == i % self.FRAMES_PER_EPISODE

        # Verify video content
        self.verify_video_content(dataset_reloaded, list(range(5)))

    def test_continue_after_crash_recovery(self, tmp_dataset_dir):
        """
        Test scenario: Recover from crash, then continue adding episodes and finalize.

        1. Reuse crash scenario to get recovered dataset (5 episodes)
        2. Continue adding episodes 5-9
        3. Finalize with consolidation
        4. Verify all data is correct
        """
        # Phase 1 & 2: Create crash scenario and get recovered dataset
        dataset = self._setup_crash_scenario(tmp_dataset_dir)

        # Phase 3: Continue adding episodes (5-9)
        # add_frame() auto-initializes episode_buffer if None

        for ep in range(5, 10):  # Add episodes 5-9
            self.add_episode(dataset, ep)

        # Checkpoint after new episodes
        dataset.checkpoint()

        # Finalize with consolidation
        dataset.finalize(consolidate=True)

        # Phase 4: Reload final dataset and verify
        del dataset
        gc.collect()

        dataset_final = LeRobotDataset(repo_id=self.REPO_ID, root=tmp_dataset_dir)

        assert dataset_final.num_episodes == 10, f"Expected 10 episodes, got {dataset_final.num_episodes}"
        assert len(dataset_final) == 100, f"Expected 100 frames, got {len(dataset_final)}"

        # Verify all frame indices
        for i in range(len(dataset_final)):
            frame = dataset_final[i]
            expected_ep = i // self.FRAMES_PER_EPISODE
            expected_frame = i % self.FRAMES_PER_EPISODE

            assert frame["episode_index"] == expected_ep, (
                f"Frame {i}: expected episode {expected_ep}, got {frame['episode_index']}"
            )
            assert frame["frame_index"] == expected_frame, (
                f"Frame {i}: expected frame_index {expected_frame}, got {frame['frame_index']}"
            )

        # Verify video content for all episodes
        self.verify_video_content(dataset_final, list(range(10)))

        # Verify consolidation: should have single data file
        data_files = list((tmp_dataset_dir / "data").rglob("*.parquet"))
        assert len(data_files) == 1, f"Expected 1 consolidated file, got {len(data_files)}: {data_files}"

        video_files = list((tmp_dataset_dir / "videos").rglob("*.mp4"))
        assert len(video_files) == 1, f"Expected 1 video file, got {len(video_files)}: {video_files}"

    def test_video_content_integrity(self, tmp_dataset_dir):
        """
        Test that video content is correctly preserved through checkpoint and recovery.

        Uses distinct solid colors per episode to verify no data mixing occurs.
        """
        dataset = self.create_dataset(tmp_dataset_dir)

        # Add 5 episodes with distinct colors
        for ep in range(5):
            self.add_episode(dataset, ep)

        dataset.checkpoint()
        dataset.finalize()

        # Reload and verify each episode has correct color
        dataset_reloaded = LeRobotDataset(repo_id=self.REPO_ID, root=tmp_dataset_dir)

        for ep in range(5):
            # Check first and last frame of each episode
            for frame_offset in [0, self.FRAMES_PER_EPISODE - 1]:
                frame_idx = ep * self.FRAMES_PER_EPISODE + frame_offset
                frame = dataset_reloaded[frame_idx]

                image = frame["observation.image"]
                expected_color = ep

                # Check multiple pixels to ensure consistency
                for c in range(3):  # RGB channels
                    for y in [0, 15, 29]:  # Sample rows
                        for x in [0, 15, 29]:  # Sample columns
                            actual = image[c, y, x].item()
                            assert abs(actual - expected_color) < 10, (
                                f"Ep {ep}, frame {frame_offset}, pixel ({c},{y},{x}): "
                                f"expected ~{expected_color}, got {actual}"
                            )
