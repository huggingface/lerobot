from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


def write_shape_on_image_inplace(image):
    height, width = image.shape[:2]
    text = f"Width: {width} Height: {height}"

    # Define the font, scale, color, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # Blue in BGR
    thickness = 2

    position = (10, height - 10)  # 10 pixels from the bottom-left corner
    cv2.putText(image, text, position, font, font_scale, color, thickness)


def save_color_image(image, path, write_shape=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_shape:
        write_shape_on_image_inplace(image)
    cv2.imwrite(str(path), image)


def save_depth_image(depth, path, write_shape=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)

    if write_shape:
        write_shape_on_image_inplace(depth_image)
    cv2.imwrite(str(path), depth_image)


# Defines a camera type
class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: str | None = None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...
