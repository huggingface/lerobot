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
    assert len(files_before) >= 5, f"Expected fragmentation before consolidate, got {len(files_before)} files"

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
    assert len(files_before) >= 5, f"Expected fragmentation, got {len(files_before)} files"

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
