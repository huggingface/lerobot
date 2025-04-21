from lerobot.common.datasets.aggregate import aggregate_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tests.fixtures.constants import DUMMY_REPO_ID


def test_aggregate_datasets(tmp_path, lerobot_dataset_factory):
    ds_0 = lerobot_dataset_factory(
        root=tmp_path / "test_0",
        repo_id=f"{DUMMY_REPO_ID}_0",
        total_episodes=10,
        total_frames=400,
    )
    ds_1 = lerobot_dataset_factory(
        root=tmp_path / "test_1",
        repo_id=f"{DUMMY_REPO_ID}_1",
        total_episodes=10,
        total_frames=400,
    )

    aggregate_datasets(
        repo_ids=[ds_0.repo_id, ds_1.repo_id],
        roots=[ds_0.root, ds_1.root],
        aggr_repo_id=f"{DUMMY_REPO_ID}_aggr",
        aggr_root=tmp_path / "test_aggr",
    )

    aggr_ds = LeRobotDataset(f"{DUMMY_REPO_ID}_aggr", root=tmp_path / "test_aggr")
    for _ in aggr_ds:
        pass
