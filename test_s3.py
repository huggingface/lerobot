# test_lerobot_dataset_metadata.py
import sys
from upath import UPath as Path
from dotenv import load_dotenv

# Добавляем локальный src в sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

def test_lerobot_dataset_metadata_initialization():
    s3_path = "s3://d-gigachat-vision/robodata/airoa-moma"

    meta_data = LeRobotDatasetMetadata(
        repo_id="robodata/airoa-moma",
        root=s3_path,
        revision="main",
    )

    # Проверяем, что поля установлены корректно
    assert meta_data.repo_id == "robodata/airoa-moma"
    # root может быть Path-like, поэтому сравниваем как строку
    assert str(getattr(meta_data, "root")) == s3_path
    assert meta_data.revision == "main"

    print("Loaded metadata successfully!")

def test_lerobot_dataset():
    s3_path = "s3://d-gigachat-vision/robodata/airoa-moma"

    dataset = LeRobotDataset(
        repo_id="robodata/airoa-moma",
        root=s3_path,
        revision="main",
    )

    print("Loaded dataset successfully!")

    print("Dataset length", len(dataset))

def test_lerobot_dataset_item():
    s3_path = "s3://d-gigachat-vision/robodata/airoa-moma"

    dataset = LeRobotDataset(
        repo_id="robodata/airoa-moma",
        root=s3_path,
        revision="main",
        episodes=[0, 10, 11, 23]
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