from pathlib import Path

from torchvision.transforms import ToPILImage, v2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

to_pil = ToPILImage()

# Create a directory to store the training checkpoint.
output_dir = Path("outputs/image_transforms")
output_dir.mkdir(parents=True, exist_ok=True)

repo_id = "lerobot/aloha_static_tape"

# Create a LeRobotDataset with no transformations
dataset = LeRobotDataset(repo_id, transform=None)

# Get the index of the first frame in the first episode
first_idx = dataset.episode_data_index["from"][0].item()

# Get the frame from the first camera
frame = dataset[first_idx][dataset.camera_keys[0]]

# Save the original frame
to_pil(frame).save(output_dir / "original_frame.png", quality=100)
print(f"Original frame saved to {output_dir / 'original_frame.png'}.")


# Define the transformations
transforms = v2.Compose(
    [
        v2.ColorJitter(brightness=(0.5, 1.5)),
        v2.ColorJitter(contrast=(0.5, 1.5)),
        v2.RandomAdjustSharpness(sharpness_factor=2, p=1),
    ]
)

# Create a LeRobotDataset with the defined transformations
transformed_dataset = LeRobotDataset(repo_id, transform=transforms)

# Get a frame from the transformed dataset
transformed_frame = transformed_dataset[first_idx][transformed_dataset.camera_keys[0]]

# Save the transformed frame
to_pil(transformed_frame).save(output_dir / "transformed_frame.png", quality=100)
print(f"Transformed frame saved to {output_dir / 'transformed_frame.png'}.")
