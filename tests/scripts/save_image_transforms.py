from pathlib import Path

import torch
from torchvision.transforms import v2
from safetensors.torch import save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import RangeRandomSharpness
from lerobot.common.utils.utils import seeded_context

DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"
ARTIFACT_DIR = "tests/data/save_image_transforms"
SEED = 1336
to_pil = v2.ToPILImage()


def main(repo_id):
    dataset = LeRobotDataset(repo_id, image_transforms=None)
    output_dir = Path(ARTIFACT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get first frame of given episode
    from_idx = dataset.episode_data_index["from"][0].item()
    original_frame = dataset[from_idx][dataset.camera_keys[0]]
    to_pil(original_frame).save(output_dir / "original_frame.png", quality=100)

    transforms = {
        "brightness": v2.ColorJitter(brightness=(0.0, 2.0)),
        "contrast": v2.ColorJitter(contrast=(0.0, 2.0)),
        "saturation": v2.ColorJitter(saturation=(0.0, 2.0)),
        "hue": v2.ColorJitter(hue=(-0.5, 0.5)),
        "sharpness": RangeRandomSharpness(0.0, 2.0),
    }

    # frames = {"original_frame": original_frame}
    for name, transform in transforms.items():
        with seeded_context(SEED):
            # transform = v2.Compose([transform, v2.ToDtype(torch.float32, scale=True)])
            transformed_frame = transform(original_frame)
            # frames[name] = transform(original_frame)
            to_pil(transformed_frame).save(output_dir / f"{SEED}_{name}.png", quality=100)

    # save_file(frames, output_dir / f"transformed_frames_{SEED}.safetensors")

if __name__ == "__main__":
    repo_id = "lerobot/aloha_mobile_shrimp"
    main(repo_id)
