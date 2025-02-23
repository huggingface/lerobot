from lerobot.common.datasets.aggregate import aggregate_datasets
from tests.fixtures.constants import DUMMY_REPO_ID


def test_aggregate_datasets(tmp_path, lerobot_dataset_factory):
    dataset_0 = lerobot_dataset_factory(
        root=tmp_path / "test_0",
        repo_id=DUMMY_REPO_ID + "_0",
        total_episodes=10,
        total_frames=400,
    )
    dataset_1 = lerobot_dataset_factory(
        root=tmp_path / "test_1",
        repo_id=DUMMY_REPO_ID + "_1",
        total_episodes=10,
        total_frames=400,
    )

    dataset_2 = aggregate_datasets([dataset_0, dataset_1])
