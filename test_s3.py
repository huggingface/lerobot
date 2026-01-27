# test_lerobot_dataset_metadata.py

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

REPO_ID = "airoa-org/airoa-moma"
REVISION = "main"
S3_ENDPOINT_URL = "https://obs.ru-moscow-1.hc.sbercloud.ru"
S3_PATH = "s3://d-gigachat-vision/robodata/airoa-moma"
EPISODES = [0, 10, 11, 23]


def test_lerobot_dataset_metadata_initialization(
    repo_id: str,
    root: str = None,
    revision: str = "main",
    s3_endpoint_url: str = None,
):
    meta_data = LeRobotDatasetMetadata(
        repo_id=repo_id, root=root, revision=revision, s3_endpoint_url=s3_endpoint_url
    )

    # Проверяем, что поля установлены корректно
    assert meta_data.repo_id == repo_id
    assert meta_data.revision == revision

    print("Loaded metadata successfully!")


def test_lerobot_dataset_item(
    repo_id: str,
    root: str = None,
    revision: str = "main",
    s3_endpoint_url: str = None,
    episodes: list[int] = None,
):
    dataset = LeRobotDataset(repo_id=REPO_ID, revision="main", episodes=[0, 10, 11, 23])
    episode_index = 0
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
    camera_key = dataset.meta.camera_keys[0]
    frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

    # The objects returned by the dataset are all torch.Tensors
    print(type(frames[0]))
    print(frames[0].shape)

    print(dataset[0])


if __name__ == "__main__":
    # offline path
    test_lerobot_dataset_metadata_initialization(repo_id=REPO_ID)
    test_lerobot_dataset_item(repo_id=REPO_ID, episodes=EPISODES)

    # online path
    test_lerobot_dataset_metadata_initialization(
        repo_id=REPO_ID, root=S3_PATH, s3_endpoint_url=S3_ENDPOINT_URL
    )
    test_lerobot_dataset_item(
        repo_id=REPO_ID, root=S3_PATH, s3_endpoint_url=S3_ENDPOINT_URL, episodes=EPISODES
    )
