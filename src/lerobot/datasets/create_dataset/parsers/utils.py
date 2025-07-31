"""Utility functions for data parsing."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def extract_episode_number(filepath: Path) -> int:
    """Extract episode number from filename."""
    import re
    match = re.search(r'trajectory_(\d+)', filepath.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract episode number from {filepath}")


def load_image(img_path: Path) -> np.ndarray:
    """Load and convert image to numpy array."""
    with Image.open(img_path) as img:
        img_array = np.array(img)

        # Ensure grayscale images have a channel dimension
        if len(img_array.shape) == 2:
            # Add channel dimension for grayscale images
            img_array = img_array[..., np.newaxis]

        # Convert grayscale (1 channel) to RGB (3 channels) by repeating
        if img_array.shape[-1] == 1:
            img_array = np.repeat(img_array, 3, axis=-1)

        return img_array


def find_sample_image(input_dir: Path, pattern: str, extension: str) -> Optional[Path]:
    """Find a sample image to determine dimensions."""
    pattern = pattern.replace("{episode}", "*").replace("{frame}", "*")
    pattern += extension
    images = list(input_dir.glob(pattern))
    return images[0] if images else None


def get_image_dimensions(img_path: Path) -> tuple[int, int, int]:
    """Get dimensions (height, width, channels) from an image file."""
    with Image.open(img_path) as img:
        width, height = img.size
        # Handle grayscale images (mode 'L') as having 3 channels since we convert to RGB
        if img.mode == 'L':
            channels = 3  # Convert grayscale to RGB
        else:
            channels = len(img.getbands())
    return height, width, channels
