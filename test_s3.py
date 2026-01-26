# test_lerobot_dataset_metadata.py

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

S3_PATH = "s3://your-bucket-name/test-folder/"
S3_ENDPOINT_URL = "https://s3.your-provider.com"


def test_lerobot_dataset_metadata_initialization():
    meta_data = LeRobotDatasetMetadata(
        repo_id="robodata/airoa-moma", root=S3_PATH, revision="main", s3_endpoint_url=S3_ENDPOINT_URL
    )

    # Проверяем, что поля установлены корректно
    assert meta_data.repo_id == "robodata/airoa-moma"
    # root может быть Path-like, поэтому сравниваем как строку
    assert str(meta_data.root) == S3_PATH
    assert meta_data.revision == "main"

    print("Loaded metadata successfully!")


def test_lerobot_dataset():
    dataset = LeRobotDataset(
        repo_id="robodata/airoa-moma", root=S3_PATH, revision="main", s3_endpoint_url=S3_ENDPOINT_URL
    )

    print("Loaded dataset successfully!")

    print("Dataset length", len(dataset))


def test_lerobot_dataset_item():
    dataset = LeRobotDataset(
        repo_id="robodata/airoa-moma",
        root=S3_PATH,
        revision="main",
        episodes=[0, 10, 11, 23],
        s3_endpoint_url=S3_ENDPOINT_URL,
    )
    item = dataset[0]
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
    # test_lerobot_dataset_metadata_initialization()
    # test_lerobot_dataset()
    test_lerobot_dataset_item()
