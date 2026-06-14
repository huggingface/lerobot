from typing import Dict, Tuple

import torch
from torchvision.transforms import v2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGENET_STATS_CACHE: Dict[Tuple[torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
LAM_IMAGE_HW: Tuple[int, int] = (256, 256)  # (H, W)
LAM_PATCH_SIZE: int = 16
LAM_GRID_HW: Tuple[int, int] = (LAM_IMAGE_HW[0] // LAM_PATCH_SIZE, LAM_IMAGE_HW[1] // LAM_PATCH_SIZE)

def _build_train_aug(
    output_size: Tuple[int, int],
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> v2.Transform:
    return v2.Compose(
        [
            v2.RandomResizedCrop(output_size, scale=scale, ratio=ratio, antialias=True),
            v2.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
        ]
    )


def _uint8_to_unit_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32).mul_(1.0 / 255.0)
    return tensor.to(dtype=torch.float32)




def _to_batched_chw_uint8(videos_uint8: torch.Tensor) -> torch.Tensor:
    if videos_uint8.ndim != 5:
        raise ValueError(f"Expected 5D tensor, got {tuple(videos_uint8.shape)}")
    if videos_uint8.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 videos, got {videos_uint8.dtype}")

    if videos_uint8.shape[-1] == 3:
        frames = videos_uint8.permute(0, 1, 4, 2, 3)
    elif videos_uint8.shape[2] == 3:
        frames = videos_uint8
    else:
        raise ValueError(f"Expected channel dim=3 in axis 2 or -1, got {tuple(videos_uint8.shape)}")

    return frames.contiguous()


def gpu_two_view_video_aug(
    videos_uint8: torch.Tensor,
    *,
    output_size: Tuple[int, int] = LAM_IMAGE_HW,
    scale: Tuple[float, float] = (0.8, 1.0),
    ratio: Tuple[float, float] = (1.0, 1.3),
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.25,
    hue: float = 0.04,
    training: bool = True,
    dual_view_aug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-side two-view video augmentation.

    Args:
        videos_uint8: [B,T,H,W,C] or [B,T,C,H,W] uint8 tensor.
    Returns:
        video1, video2: [B,T,3,OH,OW] float32, ImageNet normalized.
    """
    frames = _to_batched_chw_uint8(videos_uint8)

    if training:
        train_aug = _build_train_aug(
            output_size=output_size,
            scale=scale,
            ratio=ratio,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
        # Batch-level augmentation: apply one transform call on the whole batch tensor.
        video1 = train_aug(frames)
        if dual_view_aug:
            video2 = train_aug(frames)
        else:
            video2 = None
    else:
        # Deterministic resize/crop now happens in data_config, so eval only normalizes.
        video1 = frames
        video2 = None

    video1 = _uint8_to_unit_float(video1)
    imagenet_normalize_(video1)

    if video2 is not None:
        video2 = _uint8_to_unit_float(video2)
        imagenet_normalize_(video2)
    else:
        video2 = video1.clone()
    return video1, video2


def _get_imagenet_stats(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (device, dtype)
    if key not in _IMAGENET_STATS_CACHE:
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        _IMAGENET_STATS_CACHE[key] = (mean, std)
    return _IMAGENET_STATS_CACHE[key]


def imagenet_normalize_(tensor: torch.Tensor) -> torch.Tensor:
    """
    In-place ImageNet normalization for tensor [...,3,H,W].
    """
    mean, std = _get_imagenet_stats(tensor.device, tensor.dtype)
    tensor.sub_(mean).div_(std)
    return tensor
