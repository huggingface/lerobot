
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.transforms import make_transforms

from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CONFIG_PATH = "configs/default.yaml"

def show_image_transforms(cfg, repo_id, episode_index=0, output_dir="outputs/show_image_transforms"):
    """
    Apply a series of image transformations to a frame from a dataset and save the transformed images.

    Args:
        cfg (ConfigNode): The configuration object containing the image transformation settings and the dataset to sample.
        repo_id (str): The ID of the repository.
        episode_index (int, optional): The index of the episode to use. Defaults to 0.
        output_dir (str, optional): The directory to save the transformed images. Defaults to "outputs/show_image_transforms".
    """
  
    dataset = LeRobotDataset(repo_id)

    print(f"Getting frame from camera: {dataset.camera_keys[0]}")

    # Get first frame of given episode
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    frame = dataset[from_idx][dataset.camera_keys[0]]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_filename = f"{output_dir}/episode_{episode_index}"

    # Apply each transformation and save the result
    for transform in cfg.list:
        cfg = init_hydra_config(
            DEFAULT_CONFIG_PATH,
            overrides=[
                f"image_transform.list=[{transform}]",
                "image_transform.enable=True",
                "image_transform.n_subset=1",
                f"image_transform.{transform}_p=1",
            ])
        
        cfg = cfg.image_transform

        t = make_transforms(cfg)
        
        # Apply transformation to frame
        transformed_frame = t(frame)
        transformed_frame = transformed_frame.permute(1, 2, 0).numpy()

        # Save transformed frame
        plt.imshow(transformed_frame)
        plt.savefig(f'{base_filename}_max_transform_{transform}.png')
        plt.close()

    frame = frame.permute(1, 2, 0).numpy()
    # Save original frame
    plt.imshow(frame)
    plt.savefig(f'{base_filename}_original.png')
    plt.close()

    print(f"Saved transformed images.")
