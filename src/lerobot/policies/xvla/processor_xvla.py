# ------------------------------------------------------------------------------
# Copyright 2025 The HuggingFace Inc. team and 2toINF (https://github.com/2toINF)
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
# ------------------------------------------------------------------------------

from typing import Any

import torch
from transformers import ProcessorMixin

from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


class XVLAProcessor(ProcessorMixin):
    """
    XVLAProcessor: Unified multimodal processor for XVLA models.

    Handles:
      - Multi-view image inputs (e.g., from multiple cameras).
      - Batch processing for multiple samples.
      - Joint tokenization and image tensor preparation.

    This processor combines an image processor and a tokenizer under a single interface
    so that users can call it directly like:

        >>> processor = XVLAProcessor.from_pretrained("path/to/xvla")
        >>> inputs = processor(images=batch_images, language_instruction=batch_texts)

    It is fully compatible with the Hugging Face AutoProcessor API.

    Attributes
    ----------
    num_views : int, default=3
        Expected number of image views per sample. Missing views will be padded with zeros.
    language_max_length : int, default=50
        Maximum token length for text encoding.
    attributes : list
        Required by ProcessorMixin to know which submodules are stored and reloaded.
    image_processor_class : str
        The name of the associated image processor class.
    tokenizer_class : tuple(str)
        The names of compatible tokenizer classes.
    """

    num_views: int = 3
    language_max_length: int = 50

    # Hugging Face ProcessorMixin-required metadata
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None):
        """
        Initialize XVLAProcessor.

        Parameters
        ----------
        image_processor : PreTrainedImageProcessor, optional
            The image processor used to normalize/resize images.
        tokenizer : PreTrainedTokenizer, optional
            The tokenizer used for text tokenization.
        """
        # ProcessorMixin automatically saves these under self.image_processor / self.tokenizer
        super().__init__(image_processor, tokenizer)

    # ================== LANGUAGE ENCODING ==================
    def encode_language(self, language_instruction: str | list[str]) -> dict[str, torch.Tensor]:
        """
        Tokenize one or more language instructions.

        Parameters
        ----------
        language_instruction : str or List[str]
            A single instruction or a batch of instructions.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "input_ids": tensor of shape [B, L]
            }
        """
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        inputs = self.tokenizer(
            language_instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.language_max_length,
            truncation=True,
        )
        return {"input_ids": inputs["input_ids"]}

    # ================== IMAGE ENCODING ==================
    def encode_image(self, images: list | list[list], **kwargs) -> dict[str, torch.Tensor]:
        """
        Preprocess one or more sets of multi-view images.

        Parameters
        ----------
        images : List or List[List]
            Single sample: [img1, img2, ...]
            Batch: [[img1a, img1b], [img2a, img2b, img2c], ...]
            Each image may be a PIL.Image, NumPy array, or torch.Tensor.

        kwargs : dict
            Extra arguments passed to the underlying image processor
            (e.g., `do_resize=False`, `size=(224,224)`).

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "image_input": tensor [B, num_views, C, H, W],
              "image_mask": tensor [B, num_views]
            }
        """
        # Normalize to batch form
        if not isinstance(images[0], (list, tuple)):
            images = [images]  # convert single sample to batch of size 1

        batch_imgs, batch_masks = [], []

        for sample_imgs in images:
            processed = self.image_processor(sample_imgs, return_tensors="pt", **kwargs)["pixel_values"]
            V_exist = processed.size(0)

            # Pad to self.num_views
            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed, processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            # Mask: True for valid slots, False for padding
            image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

        return {"image_input": image_input, "image_mask": image_mask}

    # ================== COMBINED CALL ==================
    def __call__(
        self,
        images: list | list[list] | None = None,
        language_instruction: str | list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Combine image and text encoding into a unified multimodal input.

        Parameters
        ----------
        images : List or List[List], optional
            Single-sample or batched multi-view images.
        language_instruction : str or List[str], optional
            Corresponding text instructions.
        kwargs : dict
            Extra args passed to image processor.

        Returns
        -------
        Dict[str, torch.Tensor]
            {
              "input_ids": [B, L], optional,
              "image_input": [B, num_views, C, H, W], optional,
              "image_mask": [B, num_views], optional
            }
        """
        outputs: dict[str, Any] = {}

        # Encode language if provided
        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))

        # Encode image if provided
        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        # Sanity check for batch alignment
        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"Batch mismatch: text batch {outputs['input_ids'].size(0)} "
                f"!= image batch {outputs['image_input'].size(0)}"
            )
        return outputs


def make_xvla_pre_post_processors(
    config: XVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Build the LeRobot processor pipelines for XVLA.
    """

    features = {**config.input_features, **config.output_features}
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding=config.pad_language_to,
            padding_side=config.tokenizer_padding_side,
        ),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features=features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
