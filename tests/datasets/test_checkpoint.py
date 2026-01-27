import shutil

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    d = tmp_path / "test_dataset"
    if d.exists():
        shutil.rmtree(d)
    return d


def test_checkpoint_persistence(tmp_dataset_dir):
    """
    Regression test for data loss bug when using intermediate saves.
    Ensures that calling checkpoint() forces a file rollover so that
    previous data is preserved in separate parquet files.
    """
    repo_id = "test/checkpoint_persistence"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=tmp_dataset_dir,
        features={
            "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (10,), "names": ["action"]},
        },
        use_videos=False,
    )

    # Episode 0: Save 10 frames
    for i in range(10):
        dataset.add_frame(
            {
                "state": np.ones(10, dtype=np.float32) * i,
                "action": np.ones(10, dtype=np.float32) * i,
                "task": "test_task",
            }
        )
    dataset.save_episode()

    # Checkpoint should safely flush and create state where next save requires new file
    dataset.checkpoint()

    # Episode 1: Save 10 frames
    for i in range(10):
        dataset.add_frame(
            {
                "state": np.ones(10, dtype=np.float32) * (i + 10),
                "action": np.ones(10, dtype=np.float32) * (i + 10),
                "task": "test_task",
            }
        )
    dataset.save_episode()

    dataset.finalize()

    # Verify data on disk
    files = sorted((tmp_dataset_dir / "meta/episodes").rglob("*.parquet"))
    assert len(files) >= 2, f"Expected at least 2 parquet files (rolled over), found {len(files)}"

    all_indices = []
    for f in files:
        df = pd.read_parquet(f)
        all_indices.extend(df["episode_index"].tolist())

    assert len(all_indices) == 2, f"Expected 2 episodes, found {len(all_indices)}"
    assert sorted(all_indices) == [0, 1]


def test_checkpoint_consolidation(tmp_dataset_dir):
    """
    Test that finalize(consolidate=True) merges the fragmented files created by checkpoint().
    """
    repo_id = "test/checkpoint_consolidation"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=tmp_dataset_dir,
        features={
            "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (10,), "names": ["action"]},
        },
        use_videos=False,
    )

    # Save 10 episodes, checkpointing every 2 episodes
    # This should create roughly 5 files if we didn't consolidate
    for i in range(10):
        for _ in range(10):
            dataset.add_frame(
                {
                    "state": np.ones(10, dtype=np.float32) * i,
                    "action": np.ones(10, dtype=np.float32) * i,
                    "task": "test",
                }
            )
        dataset.save_episode()
        if (i + 1) % 2 == 0:
            dataset.checkpoint()

    # Before finalize, we expect multiple files
    files_before = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(files_before) == 5, f"Expected fragmentation before consolidate, got {len(files_before)} files"

    # Finalize with consolidation
    # Since total size is small, should merge into 1 file
    dataset.finalize(consolidate=True)

    # Verify we have fewer files
    files_after = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(files_after) == 1, (
        f"Expected 1 consolidated file, got {len(files_after)}: {[f.name for f in files_after]}"
    )

    # Verify data integrity
    df = pd.read_parquet(files_after[0])
    assert len(df) == 100, f"Expected 100 frames total, found {len(df)}"
    assert len(df["episode_index"].unique()) == 10

    # Verify episodes metadata is also consolidated/updated
    episodes_files = list((tmp_dataset_dir / "meta/episodes").rglob("*.parquet"))
    assert len(episodes_files) == 1, f"Expected consolidated episodes metadata, found {len(episodes_files)}"


def test_consolidation_splits_by_size(tmp_dataset_dir):
    """
    Test that consolidate_dataset respects data_files_size_in_mb and creates multiple files
    when the total data exceeds that limit.
    """
    repo_id = "test/consolidation_size_split"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=tmp_dataset_dir,
        features={
            # Use larger arrays to get bigger file sizes
            "state": {"dtype": "float32", "shape": (100,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (100,), "names": ["action"]},
        },
        use_videos=False,
    )

    # Save 20 episodes with 50 frames each = 1000 frames total
    # Each frame has 200 floats = 800 bytes, so ~800KB total (before compression)
    for i in range(20):
        for _ in range(50):
            dataset.add_frame(
                {
                    "state": np.random.randn(100).astype(np.float32),
                    "action": np.random.randn(100).astype(np.float32),
                    "task": "test",
                }
            )
        dataset.save_episode()
        # Checkpoint every 4 episodes to create fragmentation
        if (i + 1) % 4 == 0:
            dataset.checkpoint()

    # Before consolidation - should have multiple fragmented files
    files_before = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(files_before) == 5, f"Expected fragmentation, got {len(files_before)} files"

    # Set a very small size limit to force splitting
    dataset.meta.info["data_files_size_in_mb"] = 0.5  # MB

    # Finalize with consolidation
    dataset.finalize(consolidate=True)

    # Should have created multiple files due to size limit
    files_after = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(files_after) > 1, f"Expected multiple files due to size limit, got {len(files_after)}"

    # Verify total data integrity across all files
    all_frames = []
    for f in files_after:
        df = pd.read_parquet(f)
        all_frames.append(df)

    combined_df = pd.concat(all_frames, ignore_index=True)
    assert len(combined_df) == 1000, f"Expected 1000 frames total, found {len(combined_df)}"
    assert len(combined_df["episode_index"].unique()) == 20, "Should have all 20 episodes"

    # Verify episodes metadata correctly references the new files
    dataset_reloaded = LeRobotDataset(repo_id=repo_id, root=tmp_dataset_dir)
    assert dataset_reloaded.num_episodes == 20
    assert len(dataset_reloaded) == 1000

    # Verify each episode can be accessed
    for ep_idx in range(20):
        ep_data = [dataset_reloaded[i] for i in range(ep_idx * 50, (ep_idx + 1) * 50)]
        assert len(ep_data) == 50


def test_checkpoint_consolidation_with_videos(tmp_dataset_dir):
    """
    Test checkpoint and consolidation with video datasets.
    This matches the user's reported use case with image and wrist_image video features.
    """
    from PIL import Image

    repo_id = "test/checkpoint_videos"
    width = 64
    height = 64
    channels = 3
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        root=tmp_dataset_dir,
        features={
            "image": {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float64", "shape": (8,), "names": ["state"]},
            "action": {"dtype": "float64", "shape": (7,), "names": ["action"]},
        },
        use_videos=True,
    )

    # Create 8 episodes, checkpoint every 4
    for ep in range(8):
        for _frame_idx in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (height, width, channels), dtype=np.uint8))
            wrist_img = Image.fromarray(np.random.randint(0, 255, (height, width, channels), dtype=np.uint8))
            frame_data = {
                "image": img,
                "wrist_image": wrist_img,
                "state": np.random.randn(8),
                "action": np.random.randn(7),
                "task": "test_task",
            }
            dataset.add_frame(frame_data)
        dataset.save_episode()

        if (ep + 1) % 4 == 0:
            dataset.checkpoint()

    # Before consolidation - should have multiple data files
    data_files_before = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(data_files_before) >= 2, f"Expected fragmentation, got {len(data_files_before)} files"

    # Finalize with consolidation
    dataset.finalize(consolidate=True)

    # Verify data files consolidated
    data_files_after = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(data_files_after) == 1, f"Expected 1 consolidated data file, got {len(data_files_after)}"

    # Verify video files consolidated
    video_files_image = list((tmp_dataset_dir / "videos/image").rglob("*.mp4"))
    video_files_wrist = list((tmp_dataset_dir / "videos/wrist_image").rglob("*.mp4"))
    assert len(video_files_image) == 1, f"Expected 1 consolidated image video, got {len(video_files_image)}"
    assert len(video_files_wrist) == 1, f"Expected 1 consolidated wrist video, got {len(video_files_wrist)}"

    # Verify dataset can be loaded and has correct counts
    loaded = LeRobotDataset(repo_id=repo_id, root=tmp_dataset_dir)
    assert loaded.num_episodes == 8, f"Expected 8 episodes, got {loaded.num_episodes}"
    assert loaded.num_frames == 40, f"Expected 40 frames, got {loaded.num_frames}"

    # Verify all data is accessible
    for item in loaded:
        assert item["state"].shape == (8,)
        assert item["action"].shape == (7,)
        assert item["image"].shape == (channels, height, width), (
            f"LeRobot saves data in (C, W, H), ({channels, height, width}), got {item['image'].shape}"
        )
        assert item["wrist_image"].shape == (channels, height, width), (
            f"LeRobot saves data in (C, W, H), ({channels, height, width}), got {item['wrist_image'].shape}"
        )
        assert isinstance(item["task"], str)


def test_checkpoint_consolidation_with_embedded_images(tmp_dataset_dir):
    """
    Test checkpoint and consolidation with embedded image datasets (not video).
    Ensures pyarrow correctly handles embedded image data during consolidation.
    """
    from PIL import Image

    repo_id = "test/checkpoint_embedded_images"
    width = 64
    height = 64
    channels = 3
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=20,
        root=tmp_dataset_dir,
        features={
            "observation.images.top": {
                "dtype": "image",
                "shape": (height, width, channels),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float64", "shape": (8,), "names": ["state"]},
            "action": {"dtype": "float64", "shape": (7,), "names": ["action"]},
        },
        use_videos=False,
    )

    # Create 6 episodes, checkpoint every 2
    for ep in range(6):
        for _frame_idx in range(5):
            img = Image.fromarray(np.random.randint(0, 255, (height, width, channels), dtype=np.uint8))
            frame_data = {
                "observation.images.top": img,
                "state": np.random.randn(8),
                "action": np.random.randn(7),
                "task": "test_task",
            }
            dataset.add_frame(frame_data)
        dataset.save_episode()

        if (ep + 1) % 2 == 0:
            dataset.checkpoint()

    # Before consolidation - should have multiple data files
    data_files_before = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(data_files_before) >= 3, f"Expected fragmentation, got {len(data_files_before)} files"

    # Finalize with consolidation
    dataset.finalize(consolidate=True)

    # Verify data files consolidated
    data_files_after = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(data_files_after) == 1, f"Expected 1 consolidated data file, got {len(data_files_after)}"

    # Verify dataset can be loaded
    loaded = LeRobotDataset(repo_id=repo_id, root=tmp_dataset_dir)
    assert loaded.num_episodes == 6, f"Expected 6 episodes, got {loaded.num_episodes}"
    assert loaded.num_frames == 30, f"Expected 30 frames, got {loaded.num_frames}"

    # Verify images are accessible and have correct shape
    item = loaded[0]
    assert item["observation.images.top"].shape == (channels, height, width), (
        f"LeRobot saves data in (C, W, H), ({channels, height, width}), got {item['observation.images.top'].shape}"
    )
    assert item["state"].shape == (8,)


def test_multiple_checkpoints_no_data_loss(tmp_dataset_dir):
    """
    Regression test: Verify that calling checkpoint() multiple times does NOT cause data loss.
    This was the original bug where finalize() being called multiple times would overwrite files.
    """
    repo_id = "test/multiple_checkpoints"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=tmp_dataset_dir,
        features={
            "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (10,), "names": ["action"]},
        },
        use_videos=False,
    )

    total_episodes = 10
    frames_per_episode = 10

    # Save episodes with frequent checkpoints (every episode)
    for ep in range(total_episodes):
        for frame in range(frames_per_episode):
            dataset.add_frame(
                {
                    "state": np.ones(10, dtype=np.float32) * (ep * frames_per_episode + frame),
                    "action": np.ones(10, dtype=np.float32) * (ep * frames_per_episode + frame),
                    "task": "test",
                }
            )
        dataset.save_episode()
        dataset.checkpoint()  # Checkpoint after EVERY episode

    dataset.finalize()

    # Verify ALL episodes are present
    loaded = LeRobotDataset(repo_id=repo_id, root=tmp_dataset_dir)
    assert loaded.num_episodes == total_episodes, (
        f"Data loss detected! Expected {total_episodes} episodes, got {loaded.num_episodes}"
    )
    assert loaded.num_frames == total_episodes * frames_per_episode, (
        f"Data loss detected! Expected {total_episodes * frames_per_episode} frames, got {loaded.num_frames}"
    )

    # Verify episode indices are contiguous
    episode_indices = set()
    for i in range(len(loaded)):
        episode_indices.add(loaded[i]["episode_index"].item())
    assert episode_indices == set(range(total_episodes)), (
        f"Missing episodes! Expected {set(range(total_episodes))}, got {episode_indices}"
    )


def test_continue_and_consolidate_without_checkpoint(tmp_dataset_dir):
    """
    Test scenario:
    1. Create dataset, add episodes, checkpoint.
    2. Continue dataset in new session.
    3. Add more episodes.
    4. Finalize with consolidation=True WITHOUT calling checkpoint() in the second session.

    This ensures that the consolidation logic picks up the new episodes even if they haven't been
    explicitly checkpointed (which would normally move them from temporary buffer to permanent files).
    """
    repo_id = "test/continue_consolidate_no_checkpoint"

    # Phase 1: Initial creation and checkpoint
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        root=tmp_dataset_dir,
        features={
            "state": {"dtype": "float32", "shape": (10,), "names": ["state"]},
            "action": {"dtype": "float32", "shape": (10,), "names": ["action"]},
        },
        use_videos=False,
    )

    # Add 5 episodes and checkpoint
    for i in range(5):
        dataset.add_frame(
            {
                "state": np.ones(10, dtype=np.float32) * i,
                "action": np.ones(10, dtype=np.float32) * i,
                "task": "test",
            }
        )
        dataset.save_episode()

    dataset.checkpoint()

    # Simulate session end
    del dataset

    # Phase 2: Continue and finalize without checkpoint
    dataset = LeRobotDataset(repo_id=repo_id, root=tmp_dataset_dir)

    # Add 5 more episodes
    for i in range(5, 10):
        dataset.add_frame(
            {
                "state": np.ones(10, dtype=np.float32) * i,
                "action": np.ones(10, dtype=np.float32) * i,
                "task": "test",
            }
        )
        dataset.save_episode()

    # Crucial: verify we have mixed state before consolidation
    # - 0-4 are in checkpointed parquet files
    # - 5-9 are in temporary buffer (since we didn't call checkpoint)

    # Call finalize with consolidation
    dataset.finalize(consolidate=True)

    # Verification
    # 1. Check we have a single consolidated file
    files_after = list((tmp_dataset_dir / "data").rglob("*.parquet"))
    assert len(files_after) == 1, f"Expected 1 consolidated file, got {len(files_after)}"

    # 2. Verify all data is present directly from the parquet file
    df = pd.read_parquet(files_after[0])
    assert len(df) == 10, f"Expected 10 frames total (1 per episode), found {len(df)}"
    # We only added 1 frame per episode in this test for brevity

    # 3. Verify through LeRobotDataset
    dataset_reloaded = LeRobotDataset(repo_id=repo_id, root=tmp_dataset_dir)
    assert dataset_reloaded.num_episodes == 10

    # Check data content
    for i in range(10):
        item = dataset_reloaded[i]  # 1 frame per episode, so index i corresponds to episode i
        expected_val = float(i)
        assert item["state"][0] == expected_val
