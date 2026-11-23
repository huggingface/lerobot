# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MolmoAct2 pre- and post-processing.

The preprocessor normalizes the state (and the action targets in training) and packs the cameras, the discretized
state and the task into the MolmoAct2 prompt with transformers' Molmo2 processor. The postprocessor clips and
unnormalizes the predicted actions. This module also holds the FAST action tokenizer and its constrained decoding.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    JointAffineProcessorStep,
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    UnnormalizerProcessorStep,
    load_pretrained_policy_processors,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.hub import split_hub_reference
from lerobot.utils.import_utils import _scipy_available, _transformers_available, require_package

from .configuration_molmoact2 import MolmoAct2Config

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

if TYPE_CHECKING or _transformers_available:
    from transformers import (
        Molmo2ImageProcessor,
        Molmo2Processor,
        Molmo2VideoProcessor,
        PreTrainedTokenizerFast,
        Qwen2Tokenizer,
    )
    from transformers.image_utils import PILImageResampling
    from transformers.models.molmo2.image_processing_molmo2 import (
        Molmo2ImageProcessorKwargs,
        arange_for_pooling,
        batch_pixels_to_patches,
        build_resized_image,
    )
    from transformers.utils import cached_file
else:
    Molmo2ImageProcessor = Molmo2Processor = object
    Molmo2ImageProcessorKwargs = TypedDict
    Molmo2VideoProcessor = PILImageResampling = PreTrainedTokenizerFast = Qwen2Tokenizer = None
    arange_for_pooling = batch_pixels_to_patches = build_resized_image = cached_file = None

if TYPE_CHECKING or _scipy_available:
    from scipy.fft import dct, idct
else:
    dct = idct = None

logger = logging.getLogger(__name__)

# Token ids of the released MolmoAct2 vocabulary (upstream config names).
EOS_TOKEN_ID = 151645  # "<|im_end|>", also the BOS token
_ACTION_START_TOKEN_ID = 151932
_ACTION_END_TOKEN_ID = 151933
_ACTION_TOKEN_START_ID = 151934  # "<action_k>" is FAST bin k
_NUM_ACTION_TOKENS = 2048

_DEFAULT_NUM_STATE_TOKENS = 256
# Collapse nearby prompt/action lengths onto the same torch.compile graph while
# adding at most seven masked tokens per batch. transformers' ``Molmo2Processor``
# prepends BOS to the text before tokenizing, so the final width is a multiple of 8
# (the vendored upstream processor inserted BOS after padding: ``8k + 1``).
_SEQUENCE_PAD_MULTIPLE = 8

_ACTION_OUTPUT_TOKEN = "<action_output>"  # nosec B105
_ACTION_START_TOKEN = "<action_start>"  # nosec B105
_ACTION_END_TOKEN = "<action_end>"  # nosec B105
_STATE_START_TOKEN = "<state_start>"  # nosec B105
_STATE_END_TOKEN = "<state_end>"  # nosec B105
_SETUP_START_TOKEN = "<setup_start>"  # nosec B105
_SETUP_END_TOKEN = "<setup_end>"  # nosec B105
_CONTROL_START_TOKEN = "<control_start>"  # nosec B105
_CONTROL_END_TOKEN = "<control_end>"  # nosec B105

_QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,…"
_QUESTION_TRAILING_CLOSERS = "\"'”’)]}"
_QUESTION_SURROUNDING_DELIMITERS = "\"'`“”‘’[](){}"
_QUESTION_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


# Molmo2 processor


# The mypy hook has no transformers, so it sees `object` as the base and rejects `total`.
class MolmoAct2ImageProcessorKwargs(Molmo2ImageProcessorKwargs, total=False):  # type: ignore[call-arg]
    crop_mode: str


class MolmoAct2ImageProcessor(Molmo2ImageProcessor):
    """Molmo2 image processor with the `crop_mode="resize"` layout of the MolmoAct2 checkpoints.

    Each image is resized to `size` (378x378, bilinear, no antialiasing), cut into 14x14 patches and pooled 2x2 into
    a 14x14 token grid `[rows, cols, 0, 0]`, without high-resolution crops. The overlapping crop modes are Molmo2's.
    """

    valid_kwargs = MolmoAct2ImageProcessorKwargs
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    crop_mode = "resize"

    def _image_batch_to_patches_and_grids(
        self,
        images: torch.Tensor,
        max_crops: int,
        overlap_margins: list[int],
        crop_size: int,
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: list[float],
        image_std: list[float],
        image_patch_size: int,
        image_pooling_w: int,
        image_pooling_h: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.crop_mode in ("overlap-and-resize-c2", "overlap-and-resize"):
            return super()._image_batch_to_patches_and_grids(
                images,
                max_crops,
                overlap_margins,
                crop_size,
                resample,
                do_rescale,
                rescale_factor,
                do_normalize,
                image_mean,
                image_std,
                image_patch_size,
                image_pooling_w,
                image_pooling_h,
            )
        if self.crop_mode != "resize":
            raise ValueError(f"Unsupported MolmoAct2 crop_mode {self.crop_mode!r}.")
        resized, index_grid = build_resized_image(
            self,
            images,
            crop_size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            image_patch_size,
        )
        pooling_indices, num_rows, num_cols = arange_for_pooling(index_grid, image_pooling_h, image_pooling_w)
        num_images = images.shape[0]
        # One crop per image: [N, 1, num_patches, pixels_per_patch].
        patches = batch_pixels_to_patches(resized.flatten(0, 1), image_patch_size).unsqueeze(1)
        image_grid = torch.tensor([[num_rows, num_cols, 0, 0]], dtype=torch.int64, device=images.device)
        return (
            image_grid.expand(num_images, -1),
            patches,
            pooling_indices.unsqueeze(0).expand(num_images, -1, -1),
        )

    def get_image_grid_and_crops(
        self, height: int, width: int, images_kwargs: dict[str, Any] | None = None
    ) -> tuple[list[int], int]:
        """Return the pooled grid `[low_res_h, low_res_w, high_res_h, high_res_w]` and the crop count of an image.

        Args:
            height (`int`):
                Image height in pixels.
            width (`int`):
                Image width in pixels.
            images_kwargs (`dict[str, Any]`, *optional*):
                Overrides of the processor's settings.

        Returns:
            `tuple[list[int], int]`: The pooled grid and the number of crops.
        """
        image_grid, num_crops = super().get_image_grid_and_crops(height, width, images_kwargs)
        if self.crop_mode != "resize":
            return image_grid, num_crops
        return [image_grid[0], image_grid[1], 0, 0], 1


class MolmoAct2Processor(Molmo2Processor):
    """Molmo2 processor that writes the image tokens of `crop_mode="resize"` images as the MolmoAct2 checkpoints do.

    An image with grid `[rows, cols, 0, 0]` becomes `<im_start>`, `rows` rows of `cols` `<im_patch>` tokens (each
    followed by `<im_col>` when single-crop column tokens are on) and `<im_end>`.
    """

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs: Any) -> str:
        """Return the tokens that replace the `image_idx`-th `<|image|>` placeholder.

        Args:
            image_inputs (`dict`):
                Output of the image processor, with `image_grids`.
            image_idx (`int`):
                Index of the image in the batch.
            **kwargs:
                Passed to [`~transformers.Molmo2Processor.replace_image_token`] for images with crops.

        Returns:
            `str`: The image tokens.
        """
        resized_h, resized_w, height, width = (int(value) for value in image_inputs["image_grids"][image_idx])
        if height and width:
            return super().replace_image_token(image_inputs, image_idx, **kwargs)
        use_col_tokens = (
            self.image_use_col_tokens
            if self.use_single_crop_col_tokens is None
            else self.use_single_crop_col_tokens
        )
        tokens_per_row = ["<im_patch>"] * resized_w + (["<im_col>"] if use_col_tokens else [])
        return "".join(["<im_start>", *(tokens_per_row * resized_h), "<im_end>"])


# Image part of the upstream `processor_config.json`, identical in every released MolmoAct2 checkpoint. The pack
# step saves it in `policy_preprocessor.json`, so LeRobot checkpoints carry their processor; only the tokenizer
# files come from the upstream repo. The video settings are left out: MolmoAct2 takes no video.
_PROCESSOR_CONFIG: dict[str, Any] = {
    "image_processor": {
        "crop_mode": "resize",
        "do_convert_rgb": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "max_crops": 8,
        "overlap_margins": [4, 4],
        "patch_size": 14,
        "pooling_size": [2, 2],
        "resample": 2,
        "size": {"height": 378, "width": 378},
    },
    "image_use_col_tokens": True,
    "use_single_crop_col_tokens": False,
    "use_single_crop_start_token": True,
}


def _load_molmoact2_processor(processor_config: dict[str, Any], tokenizer_name: str) -> MolmoAct2Processor:
    """Build the MolmoAct2 processor from its image settings and the upstream tokenizer files, running no code.

    The upstream repos are remote-code repos, so `AutoProcessor` would need `trust_remote_code`; only the tokenizer
    files are downloaded, never the weights or the code.
    """
    require_package("transformers", extra="molmoact2")
    repo_id, revision = split_hub_reference(tokenizer_name)
    tokenizer = Qwen2Tokenizer.from_pretrained(repo_id, revision=revision)
    # Left padding keeps the last prompt token of every row in the last column, where
    # discrete generation continues. Released checkpoints do not consistently persist
    # this tokenizer setting and Qwen2 otherwise defaults to right padding.
    tokenizer.padding_side = "left"
    options = {key: value for key, value in processor_config.items() if key != "image_processor"}
    return MolmoAct2Processor(
        image_processor=MolmoAct2ImageProcessor(**processor_config["image_processor"]),
        video_processor=Molmo2VideoProcessor(),
        tokenizer=tokenizer,
        chat_template=tokenizer.chat_template,
        **options,
    )


# FAST action tokenizer


class UniversalActionProcessor:
    """FAST action tokenizer: DCT over time, scaled and rounded coefficients, then BPE.

    Port of `UniversalActionProcessor` from `allenai/MolmoAct2-FAST-Tokenizer`, without `fit`.

    Args:
        bpe_tokenizer (`PreTrainedTokenizerFast`):
            BPE tokenizer over the characters `chr(coefficient - min_token)`.
        scale (`float`, *optional*, defaults to 10.0):
            Factor applied to the DCT coefficients before rounding.
        min_token (`int`, *optional*, defaults to 0):
            Smallest rounded coefficient.
    """

    def __init__(
        self, bpe_tokenizer: PreTrainedTokenizerFast, scale: float = 10.0, min_token: int = 0
    ) -> None:
        require_package("scipy", extra="molmoact2")
        self.bpe_tokenizer = bpe_tokenizer
        self.scale = scale
        self.min_token = min_token

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str = "allenai/MolmoAct2-FAST-Tokenizer", revision: str | None = None
    ) -> UniversalActionProcessor:
        """Load the tokenizer from its `tokenizer.json` and `processor_config.json`, running no code from the repo.

        Args:
            repo_id_or_path (`str`, *optional*, defaults to `"allenai/MolmoAct2-FAST-Tokenizer"`):
                Hub repo id or local directory, optionally pinned as `repo@revision`.
            revision (`str`, *optional*):
                Hub branch, tag or commit, when `repo_id_or_path` does not pin one.

        Returns:
            `UniversalActionProcessor`: The tokenizer.
        """
        require_package("transformers", extra="molmoact2")
        repo_id, revision = split_hub_reference(repo_id_or_path, revision)
        config_path = cached_file(
            repo_id,
            "processor_config.json",
            revision=revision,
            _raise_exceptions_for_missing_entries=False,
        )
        config = json.loads(Path(config_path).read_text()) if config_path is not None else {}
        return cls(
            PreTrainedTokenizerFast.from_pretrained(repo_id, revision=revision),
            scale=config.get("scale", 10.0),
            min_token=config.get("min_token", 0),
        )

    def __call__(self, actions: np.ndarray) -> list[list[int]]:
        """Encode action chunks `(T, D)` or `(B, T, D)` into one list of token ids per chunk."""
        actions = np.asarray(actions)
        if actions.ndim == 2:
            actions = actions[None]
        coefficients = np.around(dct(actions, axis=1, norm="ortho") * self.scale)
        tokens = []
        for chunk_coefficients in coefficients:
            characters = np.maximum(chunk_coefficients.flatten() - self.min_token, 0).astype(int)
            tokens.append(self.bpe_tokenizer("".join(map(chr, characters)))["input_ids"])
        return tokens

    def decode(self, tokens: Sequence[Sequence[int]], time_horizon: int, action_dim: int) -> np.ndarray:
        """Decode token ids into action chunks `(len(tokens), time_horizon, action_dim)`.

        A sequence that does not decode to `time_horizon * action_dim` coefficients becomes zeros, with a warning.
        """
        actions = []
        for sequence in tokens:
            try:
                text = self.bpe_tokenizer.decode(list(sequence))
                coefficients = np.array([ord(character) for character in text]) + self.min_token
                coefficients = coefficients.reshape(time_horizon, action_dim)
            except Exception as error:  # noqa: BLE001
                # Generated sequences can be malformed, including ids outside the FAST vocabulary.
                logger.warning("Could not decode FAST tokens %s: %s", list(sequence), error)
                coefficients = np.zeros((time_horizon, action_dim))
            actions.append(idct(coefficients / self.scale, axis=0, norm="ortho"))
        return np.stack(actions)


def _build_discrete_action_string_from_action(action: np.ndarray, fast: UniversalActionProcessor) -> str:
    """Encode a normalized action chunk `(T, D)` as `<action_start><action_k>...<action_end>` text."""
    (token_ids,) = fast(action)
    return _ACTION_START_TOKEN + "".join(f"<action_{token_id}>" for token_id in token_ids) + _ACTION_END_TOKEN


def _extract_discrete_token_bins(generated_ids: list[int]) -> list[int]:
    """FAST bins of the action tokens after `<action_start>` (or from the start), up to `<action_end>`."""
    start = generated_ids.index(_ACTION_START_TOKEN_ID) + 1 if _ACTION_START_TOKEN_ID in generated_ids else 0
    span = generated_ids[start:]
    if _ACTION_END_TOKEN_ID in span:
        span = span[: span.index(_ACTION_END_TOKEN_ID)]
    return [
        token_id - _ACTION_TOKEN_START_ID
        for token_id in span
        if _ACTION_TOKEN_START_ID <= token_id < _ACTION_TOKEN_START_ID + _NUM_ACTION_TOKENS
    ]


def select_constrained_discrete_action_token(
    logits: Tensor,
    generated_ids: list[list[int]],
    *,
    action_tokenizer: UniversalActionProcessor,
    target_coefficients: int,
) -> Tensor:
    """Select each row's best token that preserves a fixed-size FAST action span.

    Port of upstream's `_select_constrained_discrete_action_token`, batched (upstream takes one row): force
    `<action_start>`, take the most likely action token whose BPE expansion still fits, force `<action_end>` once
    the span decodes to exactly `target_coefficients` values, then EOS.

    Args:
        logits (`Tensor`):
            Next-token logits `[B, vocab]`.
        generated_ids (`list[list[int]]`):
            Tokens generated so far, per row.
        action_tokenizer (`UniversalActionProcessor`):
            The FAST tokenizer.
        target_coefficients (`int`):
            `chunk_size * action_dim`, the length of one decoded chunk.

    Returns:
        `Tensor`: The next token of each row, `[B]`.
    """
    next_tokens = []
    for row, ids in enumerate(generated_ids):
        if _ACTION_START_TOKEN_ID not in ids:
            next_tokens.append(_ACTION_START_TOKEN_ID)
            continue
        span_ids = ids[ids.index(_ACTION_START_TOKEN_ID) + 1 :]
        if _ACTION_END_TOKEN_ID in span_ids:
            next_tokens.append(EOS_TOKEN_ID)
            continue
        token_bins = _extract_discrete_token_bins(span_ids)
        decoded_length = len(action_tokenizer.bpe_tokenizer.decode(token_bins))
        if decoded_length > target_coefficients:
            raise RuntimeError(
                "Constrained FAST generation exceeded its target coefficient count: "
                f"{decoded_length} > {target_coefficients}."
            )
        if decoded_length == target_coefficients:
            next_tokens.append(_ACTION_END_TOKEN_ID)
            continue
        action_logits = logits[row, _ACTION_TOKEN_START_ID : _ACTION_TOKEN_START_ID + _NUM_ACTION_TOKENS]
        for candidate_bin in action_logits.argsort(descending=True).tolist():
            candidate_length = len(action_tokenizer.bpe_tokenizer.decode([*token_bins, candidate_bin]))
            if decoded_length < candidate_length <= target_coefficients:
                next_tokens.append(_ACTION_TOKEN_START_ID + candidate_bin)
                break
        else:
            raise RuntimeError(
                "No action BPE token can advance constrained FAST generation without "
                f"overshooting {target_coefficients} coefficients from {decoded_length}."
            )
    return torch.tensor(next_tokens, device=logits.device)


def decode_discrete_action_chunk(
    generated_ids: list[list[int]],
    action_tokenizer: UniversalActionProcessor,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Decode each row's generated FAST span into a normalized action chunk `(B, action_horizon, action_dim)`."""
    token_bins = [_extract_discrete_token_bins(ids) for ids in generated_ids]
    if not all(token_bins):
        raise RuntimeError("Model generated no decodable action tokens between <action_start>/<action_end>.")
    return action_tokenizer.decode(token_bins, action_horizon, action_dim)


# Normalization


def _to_float64(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def _like(reference: Any, values: np.ndarray) -> Any:
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if isinstance(reference, np.ndarray):
        return values.astype(reference.dtype)
    return values.tolist()


def _normalization_stats(
    stats: dict[str, dict[str, Any]],
    names: dict[str, list[str] | None],
    normalize_gripper: bool | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Adjust the stats so LeRobot's `QUANTILES` normalizer and unnormalizer reproduce MolmoAct2's normalization.

    Two upstream rules are written into the state and action stats:

    - Gripper dims (by name, or where the stats' `mask` is False) pass through when `normalize_gripper` is False,
      or when it is None and their min and max lie in [-1, 1]: their bounds become -1 and 1.
    - Constant dims normalize to 0: the normalizer bounds become `v - 1` and `v + 1`, while the unnormalizer keeps
      `v`.

    Returns:
        The normalizer stats and the unnormalizer stats.
    """
    normalizer_stats = deepcopy(stats)
    unnormalizer_stats = deepcopy(stats)
    for key in (OBS_STATE, ACTION):
        if key not in stats:
            continue
        feature_stats = stats[key]
        min_values = _to_float64(feature_stats["min"])
        max_values = _to_float64(feature_stats["max"])
        dim = min_values.shape[-1]
        in_unit_range = (min_values >= -1.0) & (max_values <= 1.0)

        mask = feature_stats.get("mask")
        if normalize_gripper is not True and mask is not None:
            passthrough = ~np.asarray(mask, dtype=bool)
        else:
            is_gripper = np.zeros(dim, dtype=bool)
            feature_names = names.get(key)
            if normalize_gripper is not True:
                if feature_names is None or len(feature_names) != dim:
                    logger.warning(
                        "MolmoAct2 can't find the gripper dims of %r (expected %d names, got %r): all dims are "
                        "normalized.",
                        key,
                        dim,
                        feature_names,
                    )
                else:
                    is_gripper = np.array(["gripper" in name.lower() for name in feature_names], dtype=bool)
            passthrough = is_gripper if normalize_gripper is False else is_gripper & in_unit_range
        if (passthrough & ~in_unit_range).any():
            logger.warning(
                "MolmoAct2 passes the gripper dims of %r through, but they leave [-1, 1] and will be clipped. Set "
                "normalize_gripper=True to normalize them.",
                key,
            )
        constant = min_values == max_values

        for stat_name, bound in (("q01", -1.0), ("q99", 1.0), ("min", -1.0), ("max", 1.0)):
            if stat_name not in feature_stats:
                continue
            unnormalizer_values = np.where(passthrough, bound, _to_float64(feature_stats[stat_name]))
            normalizer_values = np.where(constant, min_values + bound, unnormalizer_values)
            normalizer_stats[key][stat_name] = _like(feature_stats[stat_name], normalizer_values)
            unnormalizer_stats[key][stat_name] = _like(feature_stats[stat_name], unnormalizer_values)
        normalizer_stats[key].pop("mask", None)
        unnormalizer_stats[key].pop("mask", None)
    return normalizer_stats, unnormalizer_stats


def _load_norm_tag(config: MolmoAct2Config) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load the stats and metadata of `config.norm_tag` from the `norm_stats.json` of `config.norm_stats_path`.

    Args:
        config (`MolmoAct2Config`):
            The policy config.

    Returns:
        `tuple[dict[str, dict[str, Any]], dict[str, Any]]`: The state and action stats, and the tag's metadata
        (`setup_type`, `control_mode`, `camera_keys`, `action_horizon`, ...).
    """
    source = config.norm_stats_path or "allenai/MolmoAct2"
    path = Path(source).expanduser()
    if not path.is_file():
        path = Path(cached_file(source, "norm_stats.json"))
    metadata_by_tag = json.loads(path.read_text())["metadata_by_tag"]
    if config.norm_tag not in metadata_by_tag:
        raise ValueError(
            f"Unknown norm_tag {config.norm_tag!r} in {path}. Available tags: {sorted(metadata_by_tag)}."
        )
    metadata = metadata_by_tag[config.norm_tag]
    stats = {
        key: {name: value for name, value in metadata[stats_key].items() if name != "names"}
        for key, stats_key in ((OBS_STATE, "state_stats"), (ACTION, "action_stats"))
    }
    return stats, metadata


# Prompt


def _normalize_question_text(text: str) -> str:
    """Normalize a task like MolmoAct2 training: no quotes, "Task:" prefix or final punctuation, lowercase."""
    normalized = re.sub(r"\s+", " ", text).strip()
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in _QUESTION_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", normalized) if sentence.strip()]
    if len(sentences) > 1:
        normalized = "; ".join(sentences)
    return normalized.lower()


def _wrap_setup_text(setup_type: str) -> str:
    if not setup_type or (
        setup_type.startswith(_SETUP_START_TOKEN) and setup_type.endswith(_SETUP_END_TOKEN)
    ):
        return setup_type
    return f"{_SETUP_START_TOKEN}{setup_type}{_SETUP_END_TOKEN}"


def _wrap_control_text(control_mode: str) -> str:
    if not control_mode or (
        control_mode.startswith(_CONTROL_START_TOKEN) and control_mode.endswith(_CONTROL_END_TOKEN)
    ):
        return control_mode
    return f"{_CONTROL_START_TOKEN}{control_mode}{_CONTROL_END_TOKEN}"


def _discretize_normalized_state(state: np.ndarray) -> np.ndarray:
    """Bin each normalized state dim of [-1, 1] into one of 256 state tokens."""
    state = np.clip(np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    return np.rint((state + 1.0) / 2.0 * (_DEFAULT_NUM_STATE_TOKENS - 1)).astype(np.int64)


def _build_discrete_state_string(state: np.ndarray) -> str:
    token_ids = _discretize_normalized_state(state)
    return _STATE_START_TOKEN + "".join(f"<state_{int(k)}>" for k in token_ids) + _STATE_END_TOKEN


def _build_robot_text(
    task: str, discrete_state_string: str, setup_type: str, control_mode: str, num_images: int
) -> str:
    """Return the MolmoAct2 prompt of one sample."""
    prompt = (
        f"The task is to {task}. The setup is {_wrap_setup_text(setup_type)}. "
        f"The current state of the robot is {discrete_state_string}. The expected control mode is "
        f"{_wrap_control_text(control_mode)}. Given these, what action should the robot take to complete the task?"
    )
    image_prefix = (
        "<|image|>" if num_images == 1 else "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    )
    return (
        f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{_ACTION_OUTPUT_TOKEN}"
    )


# Processor steps


@ProcessorStepRegistry.register(name="molmoact2_pack_inputs")
@dataclass
class MolmoAct2PackInputsProcessorStep(ProcessorStep):
    """Build the MolmoAct2 model inputs from a normalized batch.

    Each sample becomes a prompt with its cameras ("Image k<|image|>"), task, setup, discretized state and control
    mode, tokenized with left padding. With an action (training), the action is clipped to [-1, 1] and zero padded
    to `max_action_dim`, the cameras can be shuffled, and with `action_mode` "discrete" or "both" the prompt is
    followed by the action's FAST tokens, which become the `labels`.

    Args:
        tokenizer_name (`str`, *optional*, defaults to `"allenai/MolmoAct2"`):
            Hub repo id or local directory of the upstream MolmoAct2 tokenizer files, optionally pinned as
            `repo@revision`.
        processor_config (`dict[str, Any]`, *optional*):
            Image processor settings and image token options of upstream's `processor_config.json`. Defaults to
            those of the released checkpoints.
        action_mode (`str`, *optional*, defaults to `"both"`):
            Training objective: "continuous", "discrete" or "both".
        action_tokenizer_name (`str`, *optional*, defaults to `"allenai/MolmoAct2-FAST-Tokenizer"`):
            FAST tokenizer of the action tokens, optionally pinned as `repo@revision`.
        image_keys (`list[str]`, *optional*):
            Cameras in prompt order. Empty uses every `observation.images.*` key, sorted.
        random_camera_order (`str`, *optional*, defaults to `"none"`):
            Shuffle the cameras in training: "none", once per "episode", or for "all" samples.
        setup_type (`str`, *optional*, defaults to `""`):
            Embodiment description of the prompt.
        control_mode (`str`, *optional*, defaults to `""`):
            Action space description of the prompt.
        action_dim (`int`, *optional*, defaults to 32):
            Number of real action dims at inference; in training, the action's own size is used.
        max_action_dim (`int`, *optional*, defaults to 32):
            Action size of the model.
    """

    tokenizer_name: str = "allenai/MolmoAct2"
    processor_config: dict[str, Any] = field(default_factory=lambda: deepcopy(_PROCESSOR_CONFIG))
    action_mode: str = "both"
    action_tokenizer_name: str = "allenai/MolmoAct2-FAST-Tokenizer"
    image_keys: list[str] = field(default_factory=list)
    random_camera_order: str = "none"
    setup_type: str = ""
    control_mode: str = ""
    action_dim: int = 32
    max_action_dim: int = 32

    def __post_init__(self) -> None:
        self._processor = _load_molmoact2_processor(self.processor_config, self.tokenizer_name)
        self._fast: UniversalActionProcessor | None = None
        if self.action_mode in ("discrete", "both"):
            self._fast = UniversalActionProcessor.from_pretrained(self.action_tokenizer_name)
        tokenizer = self._processor.tokenizer
        self._eos_id = tokenizer.eos_token_id
        self._action_start_id = tokenizer.convert_tokens_to_ids(_ACTION_START_TOKEN)
        self._action_end_id = tokenizer.convert_tokens_to_ids(_ACTION_END_TOKEN)

    def _tasks(
        self, observation: dict[str, Any], complementary: dict[str, Any], batch_size: int
    ) -> list[str]:
        sources = (
            complementary.get("task"),
            observation.get("task"),
            observation.get("observation.language"),
            complementary.get("language_instruction"),
        )
        task = next((source for source in sources if source is not None), "")
        tasks = [task] if isinstance(task, str) else [str(item) for item in task]
        if len(tasks) == 1:
            tasks *= batch_size
        if len(tasks) != batch_size:
            raise ValueError(f"MolmoAct2 expected {batch_size} tasks, got {len(tasks)}.")
        return [_normalize_question_text(task) for task in tasks]

    def _camera_orders(
        self, num_cameras: int, complementary: dict[str, Any], batch_size: int
    ) -> list[list[int]]:
        if self.random_camera_order == "none" or num_cameras < 2:
            return [list(range(num_cameras))] * batch_size
        if self.random_camera_order == "episode":
            episodes = torch.as_tensor(complementary["episode_index"]).reshape(-1).tolist()
            generators = [torch.Generator().manual_seed(int(episode)) for episode in episodes]
            return [torch.randperm(num_cameras, generator=generator).tolist() for generator in generators]
        return [torch.randperm(num_cameras).tolist() for _ in range(batch_size)]

    def _labels(self, input_ids: Tensor) -> Tensor:
        """Label the `<action_start>...<action_end>` span of each row and the `<|im_end|>` after it."""
        labels = torch.full_like(input_ids, -100)
        for row, ids in enumerate(input_ids.tolist()):
            start = ids.index(self._action_start_id)
            end = ids.index(self._action_end_id, start) + 1
            if end < len(ids) and ids[end] == self._eos_id:
                end += 1
            labels[row, start:end] = input_ids[row, start:end]
        return labels

    def _get_encoder_attention_mask(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """Mask of the prompt tokens the action expert attends to.

        With `action_mode="both"`, the action expert never sees `<|im_end|>` (also the BOS token) or the
        `<action_start> ... <action_end>` span, as in upstream training. The released checkpoints are "both"
        checkpoints, but the mask follows the action mode of the run: continuous-only training and its
        inference keep the ordinary prompt mask.
        """
        mask = attention_mask.bool()
        if self.action_mode != "both":
            return mask
        is_end = input_ids == self._action_end_id
        started = (input_ids == self._action_start_id).cumsum(-1)
        ended = is_end.cumsum(-1)
        in_span = (started > ended) | (is_end & (started >= ended))
        return mask & (input_ids != self._eos_id) & ~in_span

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        if OBS_STATE not in observation:
            raise ValueError(f"MolmoAct2 needs {OBS_STATE!r} for its prompt.")
        state = torch.as_tensor(observation[OBS_STATE], dtype=torch.float32)
        batch_size = state.shape[0]
        tasks = self._tasks(observation, complementary, batch_size)
        complementary["task"] = tasks
        action = transition.get(TransitionKey.ACTION)

        image_keys = self.image_keys or sorted(key for key in observation if key.startswith(f"{OBS_IMAGES}."))
        missing = [key for key in image_keys if key not in observation]
        if missing or not image_keys:
            raise ValueError(f"MolmoAct2 needs the camera images {missing or f'{OBS_IMAGES}.*'}.")
        cameras = []
        for key in image_keys:
            images = torch.as_tensor(observation[key])
            if images.is_floating_point():
                # Float images are in [0, 1]; the cast to uint8 truncates, as in upstream training.
                images = (images * 255.0).clamp(0, 255).to(torch.uint8)
            # Channels-first tensors stay on their device: the Molmo2 image processor resizes and patchifies them
            # there (on the GPU after GPU augmentation or streaming), without a round trip through the CPU.
            cameras.append(images)
        orders = (
            self._camera_orders(len(cameras), complementary, batch_size)
            if action is not None
            else [list(range(len(cameras)))] * batch_size
        )
        images_per_sample = [
            cameras[camera][sample] for sample in range(batch_size) for camera in orders[sample]
        ]

        texts = [
            _build_robot_text(
                task,
                _build_discrete_state_string(sample_state),
                self.setup_type,
                self.control_mode,
                len(image_keys),
            )
            for task, sample_state in zip(tasks, state.cpu().numpy(), strict=True)
        ]
        action_dim = self.action_dim
        if action is not None:
            # Clamp the normalized targets to the range used by the old trainer (robot_processing.py).
            action = torch.as_tensor(action, dtype=torch.float32).clamp(-1.0, 1.0)
            if action.ndim != 3 or action.shape[-1] > self.max_action_dim:
                raise ValueError(
                    f"MolmoAct2 expects actions [B, T, D] with D <= {self.max_action_dim}, got {tuple(action.shape)}."
                )
            action_dim = action.shape[-1]
            if self._fast is not None:
                eos = self._processor.tokenizer.eos_token
                action_texts = [
                    _build_discrete_action_string_from_action(chunk, self._fast)
                    for chunk in action.cpu().numpy()
                ]
                texts = [
                    f"{text}{action_text}{eos}" for text, action_text in zip(texts, action_texts, strict=True)
                ]
            transition[TransitionKey.ACTION] = F.pad(action, (0, self.max_action_dim - action_dim))

        inputs = self._processor(
            text=texts,
            images=images_per_sample,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=_SEQUENCE_PAD_MULTIPLE,
        )
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"].long()
        complementary.update(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # Native MolmoAct2 training assigns positions only across valid tokens, so keep the real
            # prompt/image tokens at 0..N-1 instead of shifting them by each row's left-padding length.
            position_ids=(attention_mask.cumsum(-1) - 1).clamp(min=0),
            mm_token_type_ids=inputs["mm_token_type_ids"].long(),
            encoder_attention_mask=self._get_encoder_attention_mask(input_ids, attention_mask),
            pixel_values=inputs["pixel_values"].float(),
            image_token_pooling=inputs["image_token_pooling"].long(),
            image_grids=inputs["image_grids"].long(),
            action_dim_is_pad=(torch.arange(self.max_action_dim) >= action_dim).repeat(batch_size, 1),
        )
        if action is not None and self._fast is not None:
            complementary["labels"] = self._labels(input_ids)
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "processor_config": deepcopy(self.processor_config),
            "action_mode": self.action_mode,
            "action_tokenizer_name": self.action_tokenizer_name,
            "image_keys": list(self.image_keys),
            "random_camera_order": self.random_camera_order,
            "setup_type": self.setup_type,
            "control_mode": self.control_mode,
            "action_dim": self.action_dim,
            "max_action_dim": self.max_action_dim,
        }


@ProcessorStepRegistry.register(name="molmoact2_clip_actions")
@dataclass
class MolmoAct2ClipActionsProcessorStep(PolicyActionProcessorStep):
    """Clip normalized actions to [-1, 1] before unnormalizing them, as upstream does."""

    def action(self, action: PolicyAction) -> PolicyAction:
        return action.clamp(-1.0, 1.0)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# Pipelines


def _feature_names(
    config: MolmoAct2Config, dataset_meta: LeRobotDatasetMetadata | None
) -> dict[str, list[str] | None]:
    """Per-dimension names of the state and action, which locate the gripper dims."""
    # lerobot-train passes no `dataset_meta` to the processors; `make_policy` attaches it to the config.
    meta = dataset_meta if dataset_meta is not None else getattr(config, "_runtime_dataset_meta", None)
    names = meta.names if meta is not None else {}
    return {key: names.get(key) for key in (OBS_STATE, ACTION)}


def make_molmoact2_pre_post_processors(
    config: MolmoAct2Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    dataset_meta: LeRobotDatasetMetadata | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the MolmoAct2 preprocessor and postprocessor.

    The preprocessor renames the observations, adds a batch dim, maps the state (and action targets) to the
    checkpoint's joint units and frame, normalizes them, packs the model inputs and moves them to `config.device`.
    The postprocessor clips and unnormalizes the actions, maps them back to the robot's joint units and frame and
    moves them to the CPU. Dataset stats are mapped to the checkpoint's joint units too.

    With `config.norm_tag`, the stats come from that embodiment of the checkpoint's `norm_stats.json`, and so do the
    setup type, control mode and cameras left unset in the config.

    Args:
        config (`MolmoAct2Config`):
            The policy config.
        dataset_stats (`dict[str, dict[str, Any]]`, *optional*):
            Dataset stats per feature key.
        dataset_meta (`LeRobotDatasetMetadata`, *optional*):
            Dataset metadata, for the names of the state and action dims. Defaults to the one `make_policy`
            attaches to the config.

    Returns:
        `tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]`: The preprocessor and the postprocessor.
    """
    # Zero-shot MolmoAct2-SO100_101 on SO-100/101 arms calibrated with LeRobot >= 0.5.0 needs a sign flip on
    # shoulder_lift and 90 deg offsets on shoulder_lift and elbow_flex; MolmoAct2-BimanualYAM needs radians and
    # grippers in [0, 1].
    joint_transform = JointAffineProcessorStep(scales=config.joint_scales or [], offsets=config.joint_offsets)
    stats = dataset_stats or {}
    metadata: dict[str, Any] = {}
    if config.norm_tag:
        stats, metadata = _load_norm_tag(config)
    else:
        # The dataset's stats are in the robot's units; the normalizer sees the checkpoint's.
        stats = joint_transform.transform_stats(stats)
    normalizer_stats, unnormalizer_stats = _normalization_stats(
        stats, _feature_names(config, dataset_meta), config.normalize_gripper
    )
    image_keys = list(config.image_keys)
    camera_keys = metadata.get("camera_keys") or []
    if not image_keys and camera_keys and all(key in config.input_features for key in camera_keys):
        image_keys = list(camera_keys)
    action_feature = config.output_features.get(ACTION)

    steps = make_default_policy_processor_steps(config, normalizer_stats)
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        joint_transform,
        steps.normalize,
        MolmoAct2PackInputsProcessorStep(
            tokenizer_name=config.tokenizer_name,
            action_mode=config.action_mode,
            action_tokenizer_name=config.action_tokenizer_name,
            image_keys=image_keys,
            random_camera_order=config.random_camera_order,
            setup_type=config.setup_type or metadata.get("setup_type", ""),
            control_mode=config.control_mode or metadata.get("control_mode", ""),
            action_dim=action_feature.shape[0] if action_feature is not None else config.max_action_dim,
            max_action_dim=config.max_action_dim,
        ),
        steps.to_device,
    ]
    output_steps: list[ProcessorStep] = [
        MolmoAct2ClipActionsProcessorStep(),
        # Constant dims normalize with centred stats but unnormalize with the original ones.
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=unnormalizer_stats
        ),
        JointAffineProcessorStep(
            scales=config.joint_scales or [], offsets=config.joint_offsets, inverse=True
        ),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


def make_molmoact2_pre_post_processors_from_pretrained(
    config: MolmoAct2Config,
    pretrained_path: str | Path,
    *,
    revision: str | None = None,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
    dataset_meta: LeRobotDatasetMetadata | None = None,
    preprocessor_overrides: dict[str, dict[str, Any]] | None = None,
    postprocessor_overrides: dict[str, dict[str, Any]] | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load saved MolmoAct2 pipelines, adjusting the new stats of a fine-tuning run.

    Fine-tuning passes the new dataset's stats as `stats` overrides of `normalizer_processor` and
    `unnormalizer_processor`; they are mapped to the checkpoint's joint units and adjusted as when the processors are
    first built (or replaced by the stats of `config.norm_tag`). Without new stats (resume, inference), the saved
    stats are kept. The image processor settings are the saved ones. The joint transform, the tokenizer, the training
    objective, the real action size and the camera shuffling follow the current config, and so do the setup type,
    control mode and cameras when the config (or the norm tag, for the first two) sets them, e.g. when fine-tuning the
    base checkpoint, whose prompt has none.

    Args:
        config (`MolmoAct2Config`):
            The policy config.
        pretrained_path (`str | Path`):
            Hub repo id or local directory of the saved pipelines.
        revision (`str`, *optional*):
            Hub branch, tag or commit of `pretrained_path`.
        dataset_stats (`dict[str, dict[str, Any]]`, *optional*):
            Unused: new stats arrive through the overrides.
        dataset_meta (`LeRobotDatasetMetadata`, *optional*):
            Dataset metadata, for the names of the state and action dims. Defaults to the one `make_policy`
            attaches to the config.
        preprocessor_overrides (`dict[str, dict[str, Any]]`, *optional*):
            Constructor overrides of the preprocessor steps, by registry name.
        postprocessor_overrides (`dict[str, dict[str, Any]]`, *optional*):
            Constructor overrides of the postprocessor steps, by registry name.
        preprocessor_config_filename (`str`, *optional*, defaults to `"policy_preprocessor.json"`):
            File name of the saved preprocessor.
        postprocessor_config_filename (`str`, *optional*, defaults to `"policy_postprocessor.json"`):
            File name of the saved postprocessor.

    Returns:
        `tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]`: The preprocessor and the postprocessor.
    """
    del dataset_stats
    preprocessor_overrides = {name: dict(values) for name, values in (preprocessor_overrides or {}).items()}
    postprocessor_overrides = {name: dict(values) for name, values in (postprocessor_overrides or {}).items()}
    normalizer_overrides = preprocessor_overrides.setdefault("normalizer_processor", {})
    unnormalizer_overrides = postprocessor_overrides.setdefault("unnormalizer_processor", {})
    joint_transform = JointAffineProcessorStep(scales=config.joint_scales or [], offsets=config.joint_offsets)
    for overrides in (preprocessor_overrides, postprocessor_overrides):
        overrides["joint_affine_processor"] = {
            "scales": joint_transform.scales,
            "offsets": joint_transform.offsets,
        }
    stats = normalizer_overrides.get("stats") or unnormalizer_overrides.get("stats")
    metadata: dict[str, Any] = {}
    if stats and config.norm_tag:
        stats, metadata = _load_norm_tag(config)
    elif stats:
        # The dataset's stats are in the robot's units; the normalizer sees the checkpoint's.
        stats = joint_transform.transform_stats(stats)
    if stats:
        normalizer_overrides["stats"], unnormalizer_overrides["stats"] = _normalization_stats(
            stats, _feature_names(config, dataset_meta), config.normalize_gripper
        )
    pack_overrides = preprocessor_overrides.setdefault("molmoact2_pack_inputs", {})
    pack_overrides["tokenizer_name"] = config.tokenizer_name
    pack_overrides["action_mode"] = config.action_mode
    pack_overrides["random_camera_order"] = config.random_camera_order
    action_feature = config.output_features.get(ACTION)
    if action_feature is not None:
        pack_overrides["action_dim"] = action_feature.shape[0]
    prompt = {
        "setup_type": config.setup_type or metadata.get("setup_type"),
        "control_mode": config.control_mode or metadata.get("control_mode"),
        "image_keys": config.image_keys,
    }
    pack_overrides.update({name: value for name, value in prompt.items() if value})
    return load_pretrained_policy_processors(
        pretrained_path,
        revision=revision,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
        preprocessor_config_filename=preprocessor_config_filename,
        postprocessor_config_filename=postprocessor_config_filename,
        trust_remote_code=config.trust_remote_code,
    )
