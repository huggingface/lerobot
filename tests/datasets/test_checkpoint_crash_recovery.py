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
        self.verify_video_content(dataset_reloaded, list(range(dataset_reloaded.num_episodes)))

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

    def test_crash_with_partial_episode_in_buffer(self, tmp_dataset_dir):
        """
        Test crash recovery when there's a partially completed episode in the buffer.

        Scenario:
        1. Add 4 complete episodes (0-3)
        2. Checkpoint after episode 3
        3. Add episode 4 (complete, saved)
        4. Start episode 5 but only add 5 of 10 frames (NOT saved)
        5. Crash

        Expectation:
        - Episodes 0-3 should be fully recoverable (checkpointed)
        - Episode 4 should be LOST (saved after checkpoint but not checkpointed)
        - Episode 5 partial frames should be LOST (never saved)
        - Total recoverable: 4 episodes, 40 frames
        """
        dataset = self.create_dataset(tmp_dataset_dir)

        # Add 4 complete episodes
        for ep in range(4):
            self.add_episode(dataset, ep)

        # Checkpoint after episode 3
        dataset.checkpoint()

        # Add episode 4 (complete but after checkpoint)
        self.add_episode(dataset, 4)

        # Start episode 5 but only add 5 frames (half complete)
        partial_frames = 5
        for _frame_idx in range(partial_frames):
            frame = {
                "state": np.ones(10, dtype=np.float32) * 5,
                "action": np.ones(10, dtype=np.float32) * 5,
                "task": "task_ep5_partial",
                "observation.image": self.create_solid_color_frame(5),
            }
            dataset.add_frame(frame)
        # NOTE: save_episode() is NOT called for episode 5

        # Simulate crash
        self.simulate_crash(dataset)

        # Reload dataset
        dataset_recovered = LeRobotDataset(repo_id=self.REPO_ID, root=tmp_dataset_dir)

        # Verify only checkpointed episodes are recovered
        assert dataset_recovered.num_episodes == 4, (
            f"Expected 4 episodes (checkpointed), got {dataset_recovered.num_episodes}"
        )
        assert len(dataset_recovered) == 40, (
            f"Expected 40 frames (4 episodes * 10 frames), got {len(dataset_recovered)}"
        )

        # Verify the recovered episodes are 0-3 (not 4 or partial 5)
        episode_indices = set()
        for i in range(len(dataset_recovered)):
            ep_idx = dataset_recovered[i]["episode_index"].item()
            episode_indices.add(ep_idx)

        assert episode_indices == {0, 1, 2, 3}, f"Expected episodes {{0, 1, 2, 3}}, got {episode_indices}"

        # Verify video content for recovered episodes
        self.verify_video_content(dataset_recovered, [0, 1, 2, 3])

    def test_multiple_partial_episodes_across_checkpoints(self, tmp_dataset_dir):
        """
        Test complex scenario with multiple checkpoints and partial episodes.

        IMPORTANT: checkpoint() does NOT clear the episode buffer!
        Frames added to the buffer before checkpoint (but not saved) will remain
        in the buffer and be included when save_episode() is eventually called.

        Scenario:
        1. Episodes 0-2: complete, checkpoint
        2. Episodes 3-4: complete (no checkpoint yet)
        3. Add 7 frames to buffer (NOT saved), then checkpoint
           - The 7 frames REMAIN in the buffer (checkpoint doesn't clear it)
        4. Add 10 more frames and save_episode() -> Episode 5 has 17 frames total
        5. Episode 6: complete (10 frames)
        6. Checkpoint
        7. Start episode 7: add 2 frames only (never saved)
        8. Crash

        Expectation after recovery:
        - Episodes 0-4: 50 frames (10 each)
        - Episode 5: 17 frames (7 partial + 10 complete)
        - Episode 6: 10 frames
        - Episode 7 partial: LOST (never saved)
        - Total: 7 episodes, 77 frames
        """
        dataset = self.create_dataset(tmp_dataset_dir)

        # Phase 1: Episodes 0-2, checkpoint
        for ep in range(3):
            self.add_episode(dataset, ep)
        dataset.checkpoint()

        # Phase 2: Episodes 3-4 (no checkpoint yet)
        for ep in range(3, 5):
            self.add_episode(dataset, ep)

        # Phase 3: Add partial frames to buffer (simulating interrupted collection)
        # NOTE: checkpoint() does NOT clear the buffer, so these will persist
        for _frame_idx in range(7):
            frame = {
                "state": np.ones(10, dtype=np.float32) * 5,
                "action": np.ones(10, dtype=np.float32) * 5,
                "task": "task_ep5",
                "observation.image": self.create_solid_color_frame(5),
            }
            dataset.add_frame(frame)

        # Checkpoint - saves 0-4, but buffer with 7 frames persists
        dataset.checkpoint()

        # Add 3 more frames to episode 5 and save
        # Episode 5 will have 7 + 3 = 10 frames total
        for _frame_idx in range(3):
            frame = {
                "state": np.ones(10, dtype=np.float32) * 5,
                "action": np.ones(10, dtype=np.float32) * 5,
                "task": "task_ep5",
                "observation.image": self.create_solid_color_frame(5),
            }
            dataset.add_frame(frame)
        dataset.save_episode()

        # Episode 6
        self.add_episode(dataset, 6)
        dataset.checkpoint()

        # Start episode 7 partial (will be lost)
        for _frame_idx in range(2):
            frame = {
                "state": np.ones(10, dtype=np.float32) * 7,
                "action": np.ones(10, dtype=np.float32) * 7,
                "task": "task_ep7_partial",
                "observation.image": self.create_solid_color_frame(7),
            }
            dataset.add_frame(frame)

        # Crash
        self.simulate_crash(dataset)

        # Recover
        dataset_recovered = LeRobotDataset(repo_id=self.REPO_ID, root=tmp_dataset_dir)

        # Should have episodes 0-6 (7 episodes)
        assert dataset_recovered.num_episodes == 7, (
            f"Expected 7 episodes, got {dataset_recovered.num_episodes}"
        )
        # 5*10 + 7 + 3 + 10 = 70 frames
        assert len(dataset_recovered) == 70, f"Expected 70 frames, got {len(dataset_recovered)}"

        # Verify episode indices
        episode_indices = set()
        for i in range(len(dataset_recovered)):
            ep_idx = dataset_recovered[i]["episode_index"].item()
            episode_indices.add(ep_idx)

        assert episode_indices == set(range(7)), f"Expected episodes 0-6, got {episode_indices}"

    def test_clear_episode_buffer_before_checkpoint(self, tmp_dataset_dir):
        """
        Test that users can manually clear the episode buffer if they want to
        discard partial frames before a checkpoint.

        This demonstrates the workaround if you DON'T want partial frames persisted.
        """
        dataset = self.create_dataset(tmp_dataset_dir)

        # Add 2 complete episodes
        for ep in range(2):
            self.add_episode(dataset, ep)
        dataset.checkpoint()

        # Add partial frames that we want to discard
        for _frame_idx in range(5):
            frame = {
                "state": np.ones(10, dtype=np.float32) * 99,
                "action": np.ones(10, dtype=np.float32) * 99,
                "task": "unwanted_task",
                "observation.image": self.create_solid_color_frame(99),
            }
            dataset.add_frame(frame)

        # Manually clear the buffer to discard partial frames
        dataset.episode_buffer = dataset.create_episode_buffer()

        # Add episode 2 properly
        self.add_episode(dataset, 2)
        dataset.finalize()

        # Verify
        dataset_loaded = LeRobotDataset(repo_id=self.REPO_ID, root=tmp_dataset_dir)
        assert dataset_loaded.num_episodes == 3
        assert len(dataset_loaded) == 30  # 3 episodes * 10 frames each

        # Verify no frames with value 99 (the discarded partial frames)
        for i in range(len(dataset_loaded)):
            state_val = dataset_loaded[i]["state"][0].item()
            assert state_val != 99, f"Found discarded frame at index {i}"
