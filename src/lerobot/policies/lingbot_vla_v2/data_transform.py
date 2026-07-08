from typing import Dict
from pathlib import Path

import numpy as np
import torch
import math
import einops
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import v2
import logging

logger = logging.getLogger(__name__)

IMAGE_KEYS = (
    "camera_top",
    "camera_wrist_left",
    "camera_wrist_right",
)
IGNORE_INDEX = -100


def _as_debug_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def dict_apply(func, d):
    """
    Apply a function to all values in a dictionary recursively.
    If the value is a dictionary, it will apply the function to its values.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            dict_apply(func, value)
        else:
            d[key] = func(value)
    return d


class Normalizer:
    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, np.ndarray]],
        norm_type: Dict[str, str] | None = None,
    ):
        self.norm_stats = dict_apply(lambda x: np.array(x), norm_stats)
        self.norm_type = norm_type or {}

    def _get_stat(self, key: str, stat_name: str, value):
        stat = self.norm_stats[key][stat_name]

        # Action stats generated with default chunk_size=50 are [50, dim].
        # When training with a smaller chunk_size, align stats to the actual
        # action horizon instead of relying on broadcasting.
        if getattr(stat, "ndim", None) == 2 and getattr(value, "ndim", None) == 2:
            if stat.shape[-1] != value.shape[-1]:
                raise ValueError(
                    f"Normalization dim mismatch for {key}.{stat_name}: "
                    f"stat shape {stat.shape}, value shape {tuple(value.shape)}"
                )
            if value.shape[0] > stat.shape[0]:
                raise ValueError(
                    f"Normalization horizon mismatch for {key}.{stat_name}: "
                    f"stat horizon {stat.shape[0]}, value horizon {value.shape[0]}. "
                    "Please recompute norm stats with a larger chunk_size."
                )
            if stat.shape[0] != value.shape[0]:
                stat = stat[: value.shape[0]]

        # When the value is a torch tensor (e.g. a batched, on-device training
        # item), coerce the numpy stat onto the same device/dtype so the arithmetic
        # stays in torch instead of triggering a cuda->numpy conversion.
        if isinstance(value, torch.Tensor):
            stat = torch.as_tensor(stat, dtype=value.dtype, device=value.device)

        return stat

    def normalize(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        normalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                # print(f'===Normalizing with {norm_type} for {key}===')
                if norm_type == "meanstd":
                    mean = self._get_stat(key, "mean", value)
                    std = self._get_stat(key, "std", value)
                    normalized_value = (value - mean) / (std + 1e-6)
                elif norm_type == "bounds_98":
                    low = self._get_stat(key, "q02", value)
                    high = self._get_stat(key, "q98", value)
                    normalized_value = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
                    normalized_value = torch.clamp(normalized_value, min=-1.5, max=1.5)
                elif norm_type == "bounds_99":
                    low = self._get_stat(key, "q01", value)
                    high = self._get_stat(key, "q99", value)
                    normalized_value = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
                    normalized_value = torch.clamp(normalized_value, min=-1.5, max=1.5)
                elif norm_type == "bounds_98_woclip":
                    low = self._get_stat(key, "q02", value)
                    high = self._get_stat(key, "q98", value)
                    normalized_value = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
                elif norm_type == "bounds_99_woclip":
                    low = self._get_stat(key, "q01", value)
                    high = self._get_stat(key, "q99", value)
                    normalized_value = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
                elif norm_type == "std":
                    std = self._get_stat(key, "std", value)
                    normalized_value = value / (std + 1e-6)
                elif norm_type == "minmax":
                    min_val = self._get_stat(key, "min", value)
                    max_val = self._get_stat(key, "max", value)
                    normalized_value = (value - min_val) / (max_val - min_val + 1e-6) * 2 - 1
                    normalized_value = torch.clamp(normalized_value, min=-1, max=1)
                elif norm_type == "minmax_woclip":
                    min_val = self._get_stat(key, "min", value)
                    max_val = self._get_stat(key, "max", value)
                    normalized_value = (value - min_val) / (max_val - min_val + 1e-6) * 2 - 1
                elif norm_type == "sincos":
                    if not key.startswith("observation.state."):
                        raise ValueError(f"sincos normalization is only allowed for state keys, got {key}")
                    if isinstance(value, torch.Tensor):
                        normalized_value = torch.cat([torch.cos(value), torch.sin(value)], dim=-1)
                    else:
                        normalized_value = np.concatenate([np.cos(value), np.sin(value)], axis=-1)
                elif norm_type == "identity":
                    normalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', 'sincos', and 'identity'."
                    )
                normalized_data[key] = normalized_value
            else:
                # If the key is not in norm_stats, we assume no normalization is needed
                normalized_data[key] = value
        return normalized_data

    def unnormalize(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Unnormalize the given data using stored normalization statistics.

        Args:
            data (Dict[str, np.ndarray]): Dictionary of normalized arrays to unnormalize.

        Returns:
            Dict[str, np.ndarray]: Dictionary of unnormalized arrays.
        """
        unnormalized_data = {}
        for key, value in data.items():
            if key in self.norm_stats:
                norm_type = self.norm_type.get(key, "identity")
                stats = self.norm_stats[key]
                if norm_type == "meanstd":
                    mean = self._get_stat(key, "mean", value)
                    std = self._get_stat(key, "std", value)
                    unnormalized_value = value * (std + 1e-6) + mean
                elif norm_type == "bounds_98" or norm_type == "bounds_98_woclip":
                    low = self._get_stat(key, "q02", value)
                    high = self._get_stat(key, "q98", value)
                    unnormalized_value = ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
                elif norm_type == "bounds_99" or norm_type == "bounds_99_woclip":
                    low = self._get_stat(key, "q01", value)
                    high = self._get_stat(key, "q99", value)
                    unnormalized_value = ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
                elif norm_type == "std":
                    std = self._get_stat(key, "std", value)
                    unnormalized_value = value * (std + 1e-6)
                elif norm_type == "minmax" or norm_type == "minmax_woclip":
                    min_val = self._get_stat(key, "min", value)
                    max_val = self._get_stat(key, "max", value)
                    # Reverse: (x + 1)/2 * (max-min+eps) + min
                    unnormalized_value = (value + 1) / 2.0 * (max_val - min_val + 1e-6) + min_val
                elif norm_type == "sincos":
                    if not key.startswith("observation.state."):
                        raise ValueError(f"sincos normalization is only allowed for state keys, got {key}")
                    if value.shape[-1] % 2 != 0:
                        raise ValueError(
                            f"Cannot unnormalize sincos state with odd last dim for {key}: {value.shape}"
                        )
                    half_dim = value.shape[-1] // 2
                    cos_value = value[..., :half_dim]
                    sin_value = value[..., half_dim:]
                    if isinstance(value, torch.Tensor):
                        unnormalized_value = torch.atan2(sin_value, cos_value)
                    else:
                        unnormalized_value = np.arctan2(sin_value, cos_value)
                elif norm_type == "identity":
                    unnormalized_value = value
                else:
                    raise ValueError(
                        f"Unknown normalization type: {norm_type}. Supported types are 'meanstd', 'minmax', 'sincos', and 'identity'."
                    )
                unnormalized_data[key] = unnormalized_value
            else:
                # If no normalization was applied, return as-is
                unnormalized_data[key] = value
        return unnormalized_data


def resize_with_pad_item(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 3:
        raise ValueError(f"(c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[1:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    ).squeeze(0)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


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


def prepare_images(
    image_processor,
    observation: dict[str, Tensor],
    image_keys=None,
    train=False,
    use_depth_align=False,
    return_image_grid_thw=False,
    augment_params=None,
    return_augment_params=False,
):
    """Normalize, resize, and pad images and stack them into a tensor.

    Args:
        observation (dict[str, Tensor])

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
        # print(pil_images.shape)#(224, 672, 3)
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


def prepare_state(observation: dict[str, Tensor], max_state_dim):
    """Pad the state to the maximum state dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        state (torch.Tensor): (*b, max_state_dim) padded state tensor
    """
    state = observation["state"]
    state = F.pad(state, (0, max_state_dim - state.shape[-1]))
    return state


def prepare_action(observation: dict[str, Tensor], max_action_dim):
    """Pad the action to the maximum action dimension.

    Args:
        observation (dict[str, Tensor])

    Returns:
        action (torch.Tensor): (*b, n, max_action_dim) padded action tensor
        action_dim (int): the actual dimension of the action before padding
    """
    action = observation["action"]
    action = F.pad(action, (0, max_action_dim - action.shape[-1]))
    return action


def prepare_language(config, language_tokenizer, observation: dict[str, Tensor]):
    """If `prompt` is provided, modify it to PaliGemma format and tokenize it.
    If `lang_tokens` and `lang_masks` are provided, use them directly.

    PaliGemma expects prefix prompts to be formatted as:
    <images> .... <images> <bos> prompt <sep>, where <sep> uses `\\n`.
    So here we format the prompt to start with `<bos>` and end with `\\n`.
    Later, we will concatenate the images and language tokens into a single sequence.

    Args:
        observation (dict[str, Tensor])

    Returns:
        lang_tokens (torch.Tensor): (*b, l) language tokens
        lang_masks (torch.Tensor): (*b, l) masks for language tokens, True if token is present, False if missing
    """
    lang_tokens = observation.get("lang_tokens", None)
    lang_masks = observation.get("lang_masks", None)
    prompt = observation.get("prompt", None)

    # either provide `prompt` or (`lang_tokens`, `lang_masks`)
    if prompt is None and (lang_tokens is None or lang_masks is None):
        raise ValueError(
            "Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation."
        )

    device = observation["state"].device
    if prompt is not None and (lang_tokens is None or lang_masks is None):
        if getattr(config, "use_qwen3_chat_template", False):
            prompt = [
                language_tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for p in prompt
            ]
        else:
            prompt = [p if p.startswith("<bos>") else f"<bos>{p}" for p in prompt]
            prompt = [p if p.endswith("\n") else f"{p}\n" for p in prompt]
        # print(prompt)
        tokenized_prompt = language_tokenizer.__call__(
            prompt,
            padding="max_length",
            padding_side="right",
            max_length=config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
    else:
        lang_tokens = observation["lang_tokens"].to(device=device)
        lang_masks = observation["lang_masks"].to(device=device, dtype=torch.bool)

    return lang_tokens.squeeze(0), lang_masks.squeeze(0)


def expert_visual_transform(config, observation: dict[str, Tensor], image_keys=None):
    # TODO： pad for redundant views
    image_keys = image_keys if image_keys is not None else IMAGE_KEYS
    images = []
    image_dict = {}
    resize = v2.Resize(config.resize_imgs_with_padding, antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    for key in observation["image"]:
        img = observation["image"][key]
        assert img.ndim == 3, f"Expected 3D image, got {img.shape}"
        img = resize(img)
        img = to_float(img)
        img = normalize(img)
        image_dict[key] = img
    for key in image_keys:
        if key in image_dict:
            img = image_dict[key]
            images.append(img)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)  # (n, c, h, w)
    elif isinstance(images[0], np.ndarray):
        images = torch.from_numpy(np.stack(images, axis=0))  # (n, c, h, w)

    return images
