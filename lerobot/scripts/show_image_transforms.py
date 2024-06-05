from pathlib import Path

from torchvision.transforms import ToPILImage

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import make_transforms
from lerobot.common.utils.utils import init_hydra_config

DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"
to_pil = ToPILImage()


def main(repo_id):
    """
    Apply a series of image transformations to a frame from a dataset and save the transformed images.

    Args:
        repo_id (str): The ID of the repository.
    """

    transforms = ["colorjitter", "sharpness", "blur"]

    dataset = LeRobotDataset(repo_id, transform=None)
    output_dir = Path("outputs/image_transforms") / Path(repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first frame of 1st episode
    first_idx = dataset.episode_data_index["from"][0].item()
    frame = dataset[first_idx][dataset.camera_keys[0]]
    to_pil(frame).save(output_dir / "original_frame.png", quality=100)

    transforms = ["brightness", "contrast", "saturation", "hue", "sharpness"]

    # Apply each single transformation
    for transform_name in transforms:
        overrides = [
                "image_transform.enable=True",
                "image_transform.max_num_transforms=1",
        ]
        for t in transforms:
            if t == transform_name:
                overrides.append(f"image_transform.{t}.weight=1")
                overrides.append(f"image_transform.{t}_p=1")
            else:
                overrides.append(f"image_transform.{t}.weight=0")
        cfg = init_hydra_config(
            DEFAULT_CONFIG_PATH,
            overrides=overrides,
        )
        transform = make_transforms(cfg.image_transform)
        img = transform(frame)
        to_pil(img).save(output_dir / f"{transform_name}.png", quality=100)


if __name__ == "__main__":
    repo_id = "cadene/reachy2_teleop_remi"
    main(repo_id)
