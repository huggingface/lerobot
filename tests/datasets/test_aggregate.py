import torch

from lerobot.common.datasets.aggregate import aggregate_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID


def assert_episode_and_frame_counts(aggr_ds, expected_episodes, expected_frames):
    """Test that total number of episodes and frames are correctly aggregated."""
    assert aggr_ds.num_episodes == expected_episodes, (
        f"Expected {expected_episodes} episodes, got {aggr_ds.num_episodes}"
    )
    assert aggr_ds.num_frames == expected_frames, (
        f"Expected {expected_frames} frames, got {aggr_ds.num_frames}"
    )


def assert_dataset_content_integrity(aggr_ds, ds_0, ds_1):
    """Test that the content of both datasets is preserved correctly in the aggregated dataset."""
    # Test first part of dataset corresponds to ds_0, check first item (index 0) matches ds_0[0]
    aggr_first_item = aggr_ds[0]
    ds_0_first_item = ds_0[0]

    # Compare all keys except episode_index and index which should be updated
    for key in ds_0_first_item:
        if key not in ["episode_index", "index"]:
            # Handle both tensor and non-tensor data
            if torch.is_tensor(aggr_first_item[key]) and torch.is_tensor(ds_0_first_item[key]):
                assert torch.allclose(aggr_first_item[key], ds_0_first_item[key], atol=1e-6), (
                    f"First item key '{key}' doesn't match between aggregated and ds_0"
                )
            else:
                assert aggr_first_item[key] == ds_0_first_item[key], (
                    f"First item key '{key}' doesn't match between aggregated and ds_0"
                )

    # Check last item of ds_0 part (index len(ds_0)-1) matches ds_0[-1]
    aggr_ds_0_last_item = aggr_ds[len(ds_0) - 1]
    ds_0_last_item = ds_0[-1]

    for key in ds_0_last_item:
        if key not in ["episode_index", "index"]:
            # Handle both tensor and non-tensor data
            if torch.is_tensor(aggr_ds_0_last_item[key]) and torch.is_tensor(ds_0_last_item[key]):
                assert torch.allclose(aggr_ds_0_last_item[key], ds_0_last_item[key], atol=1e-6), (
                    f"Last ds_0 item key '{key}' doesn't match between aggregated and ds_0"
                )
            else:
                assert aggr_ds_0_last_item[key] == ds_0_last_item[key], (
                    f"Last ds_0 item key '{key}' doesn't match between aggregated and ds_0"
                )

    # Test second part of dataset corresponds to ds_1
    # Check first item of ds_1 part (index len(ds_0)) matches ds_1[0]
    aggr_ds_1_first_item = aggr_ds[len(ds_0)]
    ds_1_first_item = ds_1[0]

    for key in ds_1_first_item:
        if key not in ["episode_index", "index"]:
            # Handle both tensor and non-tensor data
            if torch.is_tensor(aggr_ds_1_first_item[key]) and torch.is_tensor(ds_1_first_item[key]):
                assert torch.allclose(aggr_ds_1_first_item[key], ds_1_first_item[key], atol=1e-6), (
                    f"First ds_1 item key '{key}' doesn't match between aggregated and ds_1"
                )
            else:
                assert aggr_ds_1_first_item[key] == ds_1_first_item[key], (
                    f"First ds_1 item key '{key}' doesn't match between aggregated and ds_1"
                )

    # Check last item matches ds_1[-1]
    aggr_last_item = aggr_ds[-1]
    ds_1_last_item = ds_1[-1]

    for key in ds_1_last_item:
        if key not in ["episode_index", "index"]:
            # Handle both tensor and non-tensor data
            if torch.is_tensor(aggr_last_item[key]) and torch.is_tensor(ds_1_last_item[key]):
                assert torch.allclose(aggr_last_item[key], ds_1_last_item[key], atol=1e-6), (
                    f"Last item key '{key}' doesn't match between aggregated and ds_1"
                )
            else:
                assert aggr_last_item[key] == ds_1_last_item[key], (
                    f"Last item key '{key}' doesn't match between aggregated and ds_1"
                )


def assert_metadata_consistency(aggr_ds, ds_0, ds_1):
    """Test that metadata is correctly aggregated."""
    # Test basic info
    assert aggr_ds.fps == ds_0.fps == ds_1.fps, "FPS should be the same across all datasets"
    assert aggr_ds.meta.info["robot_type"] == ds_0.meta.info["robot_type"] == ds_1.meta.info["robot_type"], (
        "Robot type should be the same"
    )

    # Test features are the same
    assert aggr_ds.features == ds_0.features == ds_1.features, "Features should be the same"

    # Test tasks aggregation
    expected_tasks = set(ds_0.meta.tasks.index) | set(ds_1.meta.tasks.index)
    actual_tasks = set(aggr_ds.meta.tasks.index)
    assert actual_tasks == expected_tasks, f"Expected tasks {expected_tasks}, got {actual_tasks}"


def assert_episode_indices_updated_correctly(aggr_ds, ds_0, ds_1):
    """Test that episode indices are correctly updated after aggregation."""
    # ds_0 episodes should have episode_index 0 to ds_0.num_episodes-1
    for i in range(len(ds_0)):
        assert aggr_ds[i]["episode_index"] < ds_0.num_episodes, (
            f"Episode index {aggr_ds[i]['episode_index']} at position {i} should be < {ds_0.num_episodes}"
        )

    def ds1_episodes_condition(ep_idx):
        return (ep_idx >= ds_0.num_episodes) and (ep_idx < ds_0.num_episodes + ds_1.num_episodes)

    # ds_1 episodes should have episode_index ds_0.num_episodes to total_episodes-1
    for i in range(len(ds_0), len(ds_0) + len(ds_1)):
        expected_min_episode_idx = ds_0.num_episodes
        assert ds1_episodes_condition(aggr_ds[i]["episode_index"]), (
            f"Episode index {aggr_ds[i]['episode_index']} at position {i} should be >= {expected_min_episode_idx}"
        )


def assert_video_frames_integrity(aggr_ds, ds_0, ds_1):
    """Test that video frames are correctly preserved and frame indices are updated."""

    def visual_frames_equal(frame1, frame2):
        return torch.allclose(frame1, frame2)

    video_keys = list(
        filter(
            lambda key: aggr_ds.meta.info["features"][key]["dtype"] == "video",
            aggr_ds.meta.info["features"].keys(),
        )
    )

    # Test the section corresponding to the first dataset (ds_0)
    for i in range(len(ds_0)):
        assert aggr_ds[i]["index"] == i, (
            f"Frame index at position {i} should be {i}, but got {aggr_ds[i]['index']}"
        )
        for key in video_keys:
            assert visual_frames_equal(aggr_ds[i][key], ds_0[i][key]), (
                f"Visual frames at position {i} should be equal between aggregated and ds_0"
            )

    # Test the section corresponding to the second dataset (ds_1)
    for i in range(len(ds_0), len(ds_0) + len(ds_1)):
        # The frame index in the aggregated dataset should also match its position.
        assert aggr_ds[i]["index"] == i, (
            f"Frame index at position {i} should be {i}, but got {aggr_ds[i]['index']}"
        )
        for key in video_keys:
            assert visual_frames_equal(aggr_ds[i][key], ds_1[i - len(ds_0)][key]), (
                f"Visual frames at position {i} should be equal between aggregated and ds_1"
            )


def assert_dataset_iteration_works(aggr_ds):
    """Test that we can iterate through the entire dataset without errors."""
    for _ in aggr_ds:
        pass


def test_aggregate_datasets(tmp_path, lerobot_dataset_factory):
    """Test basic aggregation functionality with standard parameters."""
    ds_0_num_frames = 400
    ds_1_num_frames = 400
    ds_0_num_episodes = 10
    ds_1_num_episodes = 10

    # Create two datasets with different number of frames and episodes
    ds_0 = lerobot_dataset_factory(
        root=tmp_path / "test_0",
        repo_id=f"{DUMMY_REPO_ID}_0",
        total_episodes=ds_0_num_episodes,
        total_frames=ds_0_num_frames,
    )
    ds_1 = lerobot_dataset_factory(
        root=tmp_path / "test_1",
        repo_id=f"{DUMMY_REPO_ID}_1",
        total_episodes=ds_1_num_episodes,
        total_frames=ds_1_num_frames,
    )

    aggregate_datasets(
        repo_ids=[ds_0.repo_id, ds_1.repo_id],
        roots=[ds_0.root, ds_1.root],
        aggr_repo_id=f"{DUMMY_REPO_ID}_aggr",
        aggr_root=tmp_path / "test_aggr",
    )

    aggr_ds = LeRobotDataset(f"{DUMMY_REPO_ID}_aggr", root=tmp_path / "test_aggr")

    # Run all assertion functions
    expected_total_episodes = ds_0.num_episodes + ds_1.num_episodes
    expected_total_frames = ds_0.num_frames + ds_1.num_frames

    assert_episode_and_frame_counts(aggr_ds, expected_total_episodes, expected_total_frames)
    assert_dataset_content_integrity(aggr_ds, ds_0, ds_1)
    assert_metadata_consistency(aggr_ds, ds_0, ds_1)
    assert_episode_indices_updated_correctly(aggr_ds, ds_0, ds_1)
    assert_video_frames_integrity(aggr_ds, ds_0, ds_1)
    assert_dataset_iteration_works(aggr_ds)
