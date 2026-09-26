import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms.v2 import functional as tvF
import logging

logger = logging.getLogger(__name__)

IMAGE_KEYS = (
    "camera_top",
    "camera_wrist_left",
    "camera_wrist_right",
)


def _visual_hw(visual: Tensor) -> tuple[int, int]:
    if visual.ndim == 3:
        return int(visual.shape[1]), int(visual.shape[2])
    if visual.ndim == 4:
        return int(visual.shape[2]), int(visual.shape[3])
    raise ValueError(f"Expected image (C,H,W) or video (T,C,H,W), got {tuple(visual.shape)}")


def sample_visual_augmentation_params(
    reference: Tensor,
) -> dict:
    """Sample one replayable augmentation config for all views/frames in a sample."""
    _visual_hw(reference)
    device = reference.device
    return {
        "brightness": 0.7 + torch.rand((), device=device) * 0.6,
        "contrast": 0.6 + torch.rand((), device=device) * 0.8,
        "saturation": 0.5 + torch.rand((), device=device),
    }


def apply_visual_augmentation(
    visual: Tensor,
    params: dict,
) -> Tensor:
    """Apply the same sampled crop/rotate/color params to an image or video clip.

    Accepts images with shape (C,H,W) and video clips with shape (T,C,H,W).
    The output keeps the input shape and dtype, so it can be reused before image
    or video processors.
    """
    if visual.ndim == 3:
        visual_bchw = visual.unsqueeze(0)
        squeeze = True
    elif visual.ndim == 4:
        visual_bchw = visual
        squeeze = False
    else:
        raise ValueError(f"Expected image (C,H,W) or video (T,C,H,W), got {tuple(visual.shape)}")

    orig_dtype = visual.dtype
    height, width = int(visual_bchw.shape[-2]), int(visual_bchw.shape[-1])
    image = visual_bchw.to(torch.float32)
    if image.max() > 1.0:
        image = image / 255.0

    brightness = params["brightness"].to(device=image.device, dtype=image.dtype)
    contrast = params["contrast"].to(device=image.device, dtype=image.dtype)
    saturation = params["saturation"].to(device=image.device, dtype=image.dtype)
    image = image * brightness
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast + mean
    gray = image.mean(dim=1, keepdim=True)
    image = gray + (image - gray) * saturation
    image = image.clamp(0.0, 1.0)

    if orig_dtype == torch.uint8:
        image = (image * 255.0).round().to(torch.uint8)
    else:
        image = image.to(orig_dtype)
    return image.squeeze(0) if squeeze else image


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Line-for-line replica of Qwen2-VL's smart_resize (integer grid rounding).

    Copied so the GPU fast path derives the exact same target grid as the HF
    processor without paying its Python wrapper overhead.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def prepare_images_on_device(image_processor, images: dict[str, Tensor], device):
    """Batched on-device equivalent of Qwen2VLImageProcessor._preprocess.

    Calls the exact same torch ops as the HF torchvision backend (tvF.resize
    bicubic+antialias, fused rescale/normalize, view/permute patchify) but skips
    the wrapper overhead (kwargs validation, per-image process_image, shape
    grouping/reordering, BatchFeature) that dominates wall time on a busy host.
    Verified bit-exact against the CPU path by bench/check_gpu_preprocess.py.

    Args:
        image_processor: the HF processor instance (params read off it live)
        images: {key: CHW float tensor in [0, 255]}
        device: target device string, e.g. "cuda"

    Returns:
        (pixel_values_per_key: dict[str, Tensor], image_grid_thw: Tensor (n, 3))
        — all on `device`.
    """
    patch_size = image_processor.patch_size
    merge_size = image_processor.merge_size
    temporal_patch_size = image_processor.temporal_patch_size
    size = image_processor.size
    size_min = size["shortest_edge"] if isinstance(size, dict) else size.shortest_edge
    size_max = size["longest_edge"] if isinstance(size, dict) else size.longest_edge
    # Prefer the processor's explicit pixel bounds when present; fall back to the
    # size dict's shortest/longest edge otherwise.
    min_pixels = getattr(image_processor, "min_pixels", None)
    if min_pixels is None:
        min_pixels = size_min
    max_pixels = getattr(image_processor, "max_pixels", None)
    if max_pixels is None:
        max_pixels = size_max
    rescale_factor = image_processor.rescale_factor
    image_mean = image_processor.image_mean
    image_std = image_processor.image_std

    # Fused rescale+normalize constants, replicating
    # TorchvisionBackend.rescale_and_normalize: normalize((x), mean/rf, std/rf).
    mean_t = torch.tensor(image_mean, device=device, dtype=torch.float32) * (1.0 / rescale_factor)
    std_t = torch.tensor(image_std, device=device, dtype=torch.float32) * (1.0 / rescale_factor)
    mean_t = mean_t.view(-1, 1, 1)
    std_t = std_t.view(-1, 1, 1)

    # Group cameras by input shape (one stacked call per distinct shape).
    groups: dict[tuple[int, int], list[str]] = {}
    for key, img in images.items():
        groups.setdefault((int(img.shape[-2]), int(img.shape[-1])), []).append(key)

    out: dict[str, Tensor] = {}
    grids: dict[str, Tensor] = {}
    for (height, width), keys in groups.items():
        x = torch.stack([images[k] for k in keys]).to(device=device, non_blocking=True)
        x = x.to(torch.float32)
        resized_height, resized_width = _smart_resize(
            height, width, factor=patch_size * merge_size, min_pixels=min_pixels, max_pixels=max_pixels
        )
        if (resized_height, resized_width) != (height, width):
            x = tvF.resize(
                x,
                [resized_height, resized_width],
                interpolation=tvF.InterpolationMode.BICUBIC,
                antialias=True,
            )
        x = (x - mean_t) / std_t

        patches = x.unsqueeze(1)  # (b, 1, c, h, w)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size, grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        for i, key in enumerate(keys):
            out[key] = flatten_patches[i]
            grids[key] = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long)
    return out, grids


def prepare_images(
    image_processor,
    observation: dict[str, Tensor],
    image_keys=None,
    train=False,
    use_depth_align=False,
    return_image_grid_thw=False,
    augment_params=None,
    return_augment_params=False,
    preprocess_device=None,
):
    """Normalize, resize, and pad images and stack them into a tensor.

    Args:
        observation (dict[str, Tensor])
        preprocess_device: when set (and not train / use_depth_align), all present
            cameras are uploaded to this device and run through the HF image
            processor in ONE batched call (the TorchvisionBackend keeps torch
            tensors in torch, so resize/rescale/normalize/patchify execute on
            device). Outputs stay on the device, letting the downstream vision
            tower consume them without a second H2D copy.

    Returns:
        images (torch.Tensor): (*b, n, c, h, w) images in range [-1.0, 1.0]
        img_masks (torch.Tensor): (*b, n) masks for images, True if image is present, False if missing
    """
    dtype = observation["state"].dtype  # fp32
    images, img_masks = [], []
    image_grid_thw_list = []
    image_dict = {}
    image_grid_thw_dict = {}
    image_keys = image_keys if image_keys is not None else IMAGE_KEYS
    if train:
        for key in image_keys:
            if key in observation["image"]:
                if augment_params is None:
                    augment_params = sample_visual_augmentation_params(observation["image"][key])
                break

    if use_depth_align:
        pil_images = []
        pil_image_dict = {}

    # The fast path is a GPU optimization. ``preprocess_device="cpu"`` is the
    # explicit opt-out used by deployment configs: keep the original per-camera CPU
    # processor rather than running this tensor-only implementation on CPU.
    gpu_fast_path = (
        image_processor is not None
        and preprocess_device is not None
        and str(preprocess_device).startswith("cuda")
        and not train
        and not use_depth_align
    )

    if gpu_fast_path:
        # One fused on-device pass for all present cameras (same torch ops as the
        # HF torchvision backend, minus its Python wrapper overhead). Outputs stay
        # on-device for the vision tower — no second H2D copy.
        image_dict, grid_dict = prepare_images_on_device(
            image_processor, observation["image"], preprocess_device
        )
        if return_image_grid_thw:
            image_grid_thw_dict = {k: v.unsqueeze(0) for k, v in grid_dict.items()}
    else:
        for key in observation["image"]:
            img = observation["image"][key]
            assert img.ndim == 3, f"Expected 3D image, got {img.shape}"
            if train:
                if augment_params is None:
                    augment_params = sample_visual_augmentation_params(img)

                img = apply_visual_augmentation(
                    img,
                    augment_params,
                )

            if use_depth_align:
                pil_image_dict[key] = img.cpu().numpy()

            if image_processor is None:
                img = img.to(dtype) / 127.5 - 1.0  # to [-1, 1]
            else:
                processed = image_processor(img)
                img = processed[
                    "pixel_values"
                ]  # (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size) in qwen2.5vl, 256, 3*2*14*14
                if return_image_grid_thw and "image_grid_thw" in processed:
                    image_grid_thw_dict[key] = processed["image_grid_thw"]
            image_dict[key] = img
    if not image_dict:
        raise ValueError(
            f"None of the configured camera keys are present in the observation; missing: {list(image_keys)}"
        )
    for key in image_keys:
        if key in image_dict:
            img = image_dict[key]
            images.append(img)
            img_masks.append(True)
            if return_image_grid_thw:
                image_grid_thw_list.append(image_grid_thw_dict[key])
            if use_depth_align:
                pil_img = pil_image_dict[key]
                pil_images.append(pil_img)
        else:
            # zero padding
            img = image_dict[list(image_dict.keys())[0]]
            if use_depth_align:
                pil_img = pil_image_dict[list(pil_image_dict.keys())[0]]
            if isinstance(img, torch.Tensor):
                img = torch.full_like(img, fill_value=-1.0)  # paligemma [-1,1], now Qwen3vl
            else:
                img = np.zeros_like(img)  # Qwen2.5vl
                if use_depth_align:
                    pil_img = np.zeros_like(pil_img)
            images.append(img)
            if return_image_grid_thw:
                image_grid_thw_list.append(image_grid_thw_dict.get(list(image_dict.keys())[0], None))
            if use_depth_align:
                pil_images.append(pil_img)
            img_masks.append(False)

    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)  # (n, c, h, w)
    elif isinstance(images[0], np.ndarray):
        images = torch.from_numpy(np.stack(images, axis=0))  # (n, c, h, w)
    img_masks = torch.tensor(img_masks, dtype=torch.bool)  # (*n)

    if use_depth_align:
        # pil_images = np.stack(pil_images, axis=0)
        # pil_images = [pil_images[i].transpose(1,2,0) for i in range(pil_images.shape[0])]
        # pil_images = np.concatenate(pil_images, axis=1)
        pil_images = torch.from_numpy(np.stack(pil_images, axis=0))  # (n, c, h, w)
    else:
        pil_images = []

    if return_image_grid_thw:
        image_grid_thw = []
        for grid_thw in image_grid_thw_list:
            if grid_thw is None:
                raise ValueError(
                    "return_image_grid_thw=True requires image_processor to return image_grid_thw."
                )
            if not isinstance(grid_thw, torch.Tensor):
                grid_thw = torch.as_tensor(grid_thw)
            image_grid_thw.append(grid_thw.reshape(-1, 3)[0])
        image_grid_thw = torch.stack(image_grid_thw, dim=0).to(dtype=torch.long)
    else:
        image_grid_thw = None

    if return_augment_params:
        return images, img_masks, pil_images, image_grid_thw, augment_params
    return images, img_masks, pil_images, image_grid_thw
