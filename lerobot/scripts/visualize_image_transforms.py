from pathlib import Path

import hydra
from torchvision.transforms import ToPILImage

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import make_transforms

to_pil = ToPILImage()


def main(cfg, output_dir=Path("outputs/image_transforms")):

    dataset = LeRobotDataset(cfg.dataset_repo_id, transform=None)

    output_dir = Path(output_dir) / Path(cfg.dataset_repo_id.split("/")[-1])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first frame of 1st episode
    first_idx = dataset.episode_data_index["from"][0].item()
    frame = dataset[first_idx][dataset.camera_keys[0]]
    to_pil(frame).save(output_dir / "original_frame.png", quality=100)

    transforms = ["brightness", "contrast", "saturation", "hue", "sharpness"]

    # Apply each single transformation
    for transform_name in transforms:
        for t in transforms:
            if t == transform_name:
                cfg.image_transform[t].weight = 1
            else:
                cfg.image_transform[t].weight = 0

        transform = make_transforms(cfg.image_transform)
        img = transform(frame)
        to_pil(img).save(output_dir / f"{transform_name}.png", quality=100)


@hydra.main(version_base="1.2", config_name="default", config_path="../configs")
def visualize_transforms_cli(cfg: dict):
    main(
        cfg,
    )


if __name__ == "__main__":
    visualize_transforms_cli()
