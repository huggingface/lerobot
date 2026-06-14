from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms import functional as TF

MAX_SOURCE_ASPECT = 4.0 / 3.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def align_feature_dim_with_mask(seq: torch.Tensor, target_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if seq.ndim != 2:
        raise ValueError(f"Expected [T, D] tensor, got shape={tuple(seq.shape)}")
    if target_dim < 0:
        raise ValueError(f"target_dim must be >= 0, got {target_dim}")

    curr_dim = int(seq.shape[1])
    if target_dim == 0:
        aligned = seq[:, :0]
        mask = torch.zeros((0,), dtype=torch.bool, device=seq.device)
        return aligned, mask

    if curr_dim == target_dim:
        return seq, torch.ones((target_dim,), dtype=torch.bool, device=seq.device)

    if curr_dim > target_dim:
        return seq[:, :target_dim], torch.ones((target_dim,), dtype=torch.bool, device=seq.device)

    pad = torch.zeros(
        (int(seq.shape[0]), target_dim - curr_dim),
        dtype=seq.dtype,
        device=seq.device,
    )
    aligned = torch.cat([seq, pad], dim=1)
    mask = torch.zeros((target_dim,), dtype=torch.bool, device=seq.device)
    mask[:curr_dim] = True
    return aligned, mask


def sample_or_pad_sequence_with_mask(seq: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if seq.shape[0] == target_len:
        return seq, torch.ones((target_len,), dtype=torch.bool, device=seq.device)
    if seq.shape[0] > target_len:
        raise ValueError(
            "Action sequence exceeds configured action_horizon: "
            f"actual_len={int(seq.shape[0])}, target_len={int(target_len)}."
        )
    pad_len = target_len - seq.shape[0]
    pad = torch.zeros((pad_len, seq.shape[1]), dtype=seq.dtype, device=seq.device)
    out = torch.cat([seq, pad], dim=0)
    mask = torch.zeros((target_len,), dtype=torch.bool, device=seq.device)
    mask[: seq.shape[0]] = True
    return out, mask


def build_placeholder_masks(
    input_ids: torch.Tensor,
    *,
    act_queries: int,
    flow_queries: int,
    placeholder_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expected = int(act_queries + flow_queries)
    placeholder = input_ids == int(placeholder_id)
    order = placeholder.cumsum(dim=1)
    act_mask = placeholder & (order <= int(act_queries))
    flow_mask = placeholder & (order > int(act_queries)) & (order <= expected)
    return act_mask, flow_mask


def apply_shared_color_jitter_uint8(frames_uint8: torch.Tensor, *, severity: float = 1.0) -> torch.Tensor:
    if frames_uint8.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N, 3, H, W], got {tuple(frames_uint8.shape)}.")
    if frames_uint8.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 frames, got {frames_uint8.dtype}.")
    if int(frames_uint8.shape[1]) != 3:
        raise ValueError(f"Expected channel-first frame tensor with C=3, got {tuple(frames_uint8.shape)}.")

    sev = float(max(0.0, min(1.0, severity)))
    augmented = v2.ColorJitter(
        brightness=0.1 * sev,
        contrast=0.2 * sev,
        saturation=0.2 * sev,
        hue=0.03 * sev,
    )(frames_uint8.contiguous())
    if augmented.dtype != torch.uint8:
        augmented = augmented.round().clamp_(0, 255).to(dtype=torch.uint8)
    return augmented.contiguous()


def _crop_size_for_area_scale(height: int, width: int, area_scale: float) -> tuple[int, int]:
    if not (0.0 < float(area_scale) <= 1.0):
        raise ValueError(f"area_scale must be in (0, 1], got {area_scale}.")
    crop_h = max(1, int(round(height * float(area_scale) ** 0.5)))
    crop_w = max(1, int(round(width * float(area_scale) ** 0.5)))
    return min(crop_h, height), min(crop_w, width)


def apply_shared_random_resized_crop_uint8(frames_uint8: torch.Tensor, *, area_scale: float = 0.9) -> torch.Tensor:
    if frames_uint8.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N, 3, H, W], got {tuple(frames_uint8.shape)}.")
    if frames_uint8.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 frames, got {frames_uint8.dtype}.")
    if int(frames_uint8.shape[1]) != 3:
        raise ValueError(f"Expected channel-first frame tensor with C=3, got {tuple(frames_uint8.shape)}.")

    _, _, height, width = frames_uint8.shape
    crop_h, crop_w = _crop_size_for_area_scale(height, width, area_scale)
    max_top = max(0, height - crop_h)
    max_left = max(0, width - crop_w)
    top = 0 if max_top == 0 else int(torch.randint(0, max_top + 1, ()).item())
    left = 0 if max_left == 0 else int(torch.randint(0, max_left + 1, ()).item())
    cropped = TF.resized_crop(
        frames_uint8.contiguous(),
        top=top,
        left=left,
        height=crop_h,
        width=crop_w,
        size=[height, width],
        antialias=True,
    )
    if cropped.dtype != torch.uint8:
        cropped = cropped.round().clamp_(0, 255).to(dtype=torch.uint8)
    return cropped.contiguous()


def apply_center_crop_90_uint8(frames_uint8: torch.Tensor) -> torch.Tensor:
    if frames_uint8.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N, 3, H, W], got {tuple(frames_uint8.shape)}.")
    if frames_uint8.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 frames, got {frames_uint8.dtype}.")
    if int(frames_uint8.shape[1]) != 3:
        raise ValueError(f"Expected channel-first frame tensor with C=3, got {tuple(frames_uint8.shape)}.")

    _, _, height, width = frames_uint8.shape
    crop_h, crop_w = _crop_size_for_area_scale(height, width, area_scale=0.9)
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    cropped = TF.resized_crop(
        frames_uint8.contiguous(),
        top=top,
        left=left,
        height=crop_h,
        width=crop_w,
        size=[height, width],
        antialias=True,
    )
    if cropped.dtype != torch.uint8:
        cropped = cropped.round().clamp_(0, 255).to(dtype=torch.uint8)
    return cropped.contiguous()


def imagenet_normalize_video_(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 5 or int(tensor.shape[2]) != 3:
        raise ValueError(f"Expected video tensor with shape [B, T, 3, H, W], got {tuple(tensor.shape)}.")
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def imagenet_normalize_image_batch_(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or int(tensor.shape[1]) != 3:
        raise ValueError(f"Expected image tensor with shape [B, 3, H, W], got {tuple(tensor.shape)}.")
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor


def prepare_frame_spatial_uint8(
    frame: np.ndarray,
    target_hw: Tuple[int, int],
    *,
    apply_center_crop_90: bool = False,
) -> torch.Tensor:
    frame_chw_uint8 = _hwc_uint8_to_chw_tensor(_frame_to_hwc_uint8(frame))
    frame_chw_uint8 = _cap_source_aspect_video_uint8(frame_chw_uint8.unsqueeze(0))[0]
    if apply_center_crop_90:
        frame_chw_uint8 = apply_center_crop_90_uint8(frame_chw_uint8.unsqueeze(0))[0]
    return _apply_default_spatial_preprocess_chw_uint8(frame_chw_uint8, target_hw=target_hw)


def _frame_to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    frame_arr = np.asarray(frame)
    if frame_arr.ndim == 2:
        frame_arr = frame_arr[:, :, None]
    if frame_arr.ndim != 3:
        raise ValueError(f"Expected frame with shape [H, W, C] or [H, W], got {tuple(frame_arr.shape)}.")

    if np.issubdtype(frame_arr.dtype, np.floating):
        max_val = float(np.max(frame_arr)) if frame_arr.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            frame_arr = frame_arr * 255.0
    frame_u8 = np.clip(frame_arr, 0, 255).astype(np.uint8, copy=False)

    channels = int(frame_u8.shape[2])
    if channels == 1:
        frame_u8 = np.repeat(frame_u8, 3, axis=2)
    elif channels != 3:
        raise ValueError(f"Expected 1 or 3 channels, got C={channels}.")
    return np.ascontiguousarray(frame_u8)


def _hwc_uint8_to_chw_tensor(frame_uint8_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame_uint8_hwc).permute(2, 0, 1).contiguous()


def _cap_source_aspect_video_uint8(videos_uint8: torch.Tensor, max_aspect: float = MAX_SOURCE_ASPECT) -> torch.Tensor:
    if videos_uint8.ndim == 4:
        flat_frames = videos_uint8
        restore_shape = tuple(videos_uint8.shape)
        restore_batch = False
    elif videos_uint8.ndim == 5:
        restore_shape = tuple(videos_uint8.shape)
        flat_frames = videos_uint8.reshape(-1, *videos_uint8.shape[-3:])
        restore_batch = True
    else:
        raise ValueError(f"Expected tensor with shape [T, 3, H, W] or [B, T, 3, H, W], got {tuple(videos_uint8.shape)}.")

    if int(flat_frames.shape[1]) != 3:
        raise ValueError(f"Expected channel-first uint8 tensor with C=3, got {tuple(videos_uint8.shape)}.")

    height = int(flat_frames.shape[-2])
    width = int(flat_frames.shape[-1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid frame spatial shape HxW={height}x{width}.")
    if (width / height) <= float(max_aspect):
        return videos_uint8.contiguous()

    crop_width = max(1, int(np.floor(height * float(max_aspect))))
    left = (width - crop_width) // 2
    cropped = flat_frames[..., left : left + crop_width].contiguous()
    if not restore_batch:
        return cropped
    return cropped.reshape(restore_shape[0], restore_shape[1], 3, height, crop_width).contiguous()


def _apply_default_spatial_preprocess_chw_uint8(
    frames_uint8: torch.Tensor,
    target_hw: Tuple[int, int],
) -> torch.Tensor:
    single_frame = False
    if frames_uint8.ndim == 3:
        frames_uint8 = frames_uint8.unsqueeze(0)
        single_frame = True
    if frames_uint8.ndim != 4 or int(frames_uint8.shape[1]) != 3:
        raise ValueError(f"Expected frames tensor with shape [T, 3, H, W] or [3, H, W], got {tuple(frames_uint8.shape)}.")

    resized = v2.Resize(target_hw, antialias=True)(frames_uint8.contiguous())
    if resized.dtype != torch.uint8:
        resized = resized.round().clamp_(0, 255).to(dtype=torch.uint8)
    if single_frame:
        return resized[0].contiguous()
    return resized.contiguous()
