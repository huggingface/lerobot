# test_lerobot_dataset_metadata.py
import sys
from upath import UPath as Path
from dotenv import load_dotenv

# Добавляем локальный src в sys.path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

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

if __name__ == "__main__":
    test_lerobot_dataset_metadata_initialization()