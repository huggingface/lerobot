"""Processor using nanoVLM's native image splitting and chat-token layout."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms.functional import to_pil_image

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_nanovlm_value_function import NanoVLMVFConfig

NANOVLM_IMAGES = "observation.nanovlm.images"
NANOVLM_INPUT_IDS = "observation.nanovlm.input_ids"
NANOVLM_ATTENTION_MASK = "observation.nanovlm.attention_mask"


@ProcessorStepRegistry.register(name="nanovlm_native_processor")
@dataclass
class NanoVLMNativeProcessorStep(ComplementaryDataProcessorStep):
    pretrained_path: str
    code_path: str
    image_keys: tuple[str, ...]
    max_length: int
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _image_processor: Any = field(default=None, init=False, repr=False)
    _get_image_string: Any = field(default=None, init=False, repr=False)
    _mp_image_token_length: int = field(default=64, init=False, repr=False)

    def __post_init__(self):
        code_path = Path(self.code_path)
        if not code_path.is_absolute():
            code_path = Path(__file__).resolve().parents[4] / code_path
        if str(code_path) not in sys.path:
            sys.path.insert(0, str(code_path))

        from data.processors import get_image_processor, get_image_string, get_tokenizer

        config_path = _resolve_checkpoint_file(self.pretrained_path, "config.json")
        config = json.loads(Path(config_path).read_text())
        if self.max_length > config["lm_max_length"]:
            raise ValueError(
                f"tokenizer_max_length={self.max_length} exceeds nanoVLM's "
                f"lm_max_length={config['lm_max_length']}"
            )
        self._tokenizer = get_tokenizer(
            config["lm_tokenizer"],
            config["vlm_extra_tokens"],
            config["lm_chat_template"],
        )
        self._image_processor = get_image_processor(
            config["max_img_size"],
            config["vit_img_size"],
            config["resize_to_max_side_len"],
        )
        self._get_image_string = get_image_string
        self._mp_image_token_length = config["mp_image_token_length"]

    def complementary_data(self, complementary_data):
        raw_tasks = complementary_data.get("task")
        if raw_tasks is None:
            raise ValueError("Task is required for nanoVLM value processing")
        observation = self.transition[TransitionKey.OBSERVATION]
        present_image_keys = [key for key in self.image_keys if key in observation]
        if not present_image_keys:
            raise ValueError("No configured nanoVLM image key is present in the observation")
        batch_size = observation[present_image_keys[0]].shape[0]
        tasks = [raw_tasks] * batch_size if isinstance(raw_tasks, str) else list(raw_tasks)
        if len(tasks) != batch_size:
            raise ValueError(f"Received {len(tasks)} tasks for an image batch of size {batch_size}")

        processed_batch = []
        input_rows = []
        attention_rows = []
        for batch_index in range(batch_size):
            processed_images = []
            split_counts = []
            for key in self.image_keys:
                if key not in observation:
                    continue
                image = observation[key][batch_index]
                if image.ndim != 3:
                    raise ValueError(f"nanoVLM expects CHW images, got {tuple(image.shape)} for {key}")
                if image.shape[0] not in (1, 3, 4) and image.shape[-1] in (1, 3, 4):
                    image = image.permute(2, 0, 1)
                if image.dtype != torch.uint8:
                    image = image.float()
                    if image.min() < -1e-6 or image.max() > 1.0 + 1e-6:
                        raise ValueError(
                            f"nanoVLM expects uint8 [0,255] or float [0,1] images; "
                            f"{key} has range [{image.min().item()}, {image.max().item()}]"
                        )
                    image = image.clamp(0, 1)
                pil_image = to_pil_image(image.cpu()).convert("RGB")
                processed, split_count = self._image_processor(pil_image)
                processed_images.append(processed)
                split_counts.append(split_count)

            image_string = self._get_image_string(
                self._tokenizer,
                split_counts,
                self._mp_image_token_length,
            )
            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": image_string + f"Task: {tasks[batch_index]}."}],
                tokenize=False,
                add_generation_prompt=True,
            )
            tokenized = self._tokenizer(
                prompt,
                truncation=False,
                add_special_tokens=False,
            )
            if len(tokenized["input_ids"]) > self.max_length:
                raise ValueError(
                    f"nanoVLM prompt has {len(tokenized['input_ids'])} tokens, exceeding "
                    f"tokenizer_max_length={self.max_length}. The native nanoVLM collator "
                    "discards over-length examples instead of truncating image placeholders."
                )
            input_rows.append(tokenized["input_ids"])
            attention_rows.append(tokenized.get("attention_mask", [1] * len(tokenized["input_ids"])))
            processed_batch.append(processed_images)

        max_length = max(map(len, input_rows))
        for input_ids, attention_mask in zip(input_rows, attention_rows, strict=True):
            padding = max_length - len(input_ids)
            input_ids[:0] = [self._tokenizer.pad_token_id] * padding
            attention_mask[:0] = [0] * padding

        observation = dict(observation)
        observation[NANOVLM_IMAGES] = processed_batch
        observation[NANOVLM_INPUT_IDS] = torch.tensor(input_rows, dtype=torch.long)
        observation[NANOVLM_ATTENTION_MASK] = torch.tensor(attention_rows, dtype=torch.bool)
        self.transition[TransitionKey.OBSERVATION] = observation
        return complementary_data

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ):
        return features

    def get_config(self):
        return {
            "pretrained_path": self.pretrained_path,
            "code_path": self.code_path,
            "image_keys": self.image_keys,
            "max_length": self.max_length,
        }


def _resolve_checkpoint_file(repo_id_or_path: str, filename: str) -> str:
    local_path = Path(repo_id_or_path) / filename
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id_or_path, filename=filename)


def make_nanovlm_vf_pre_post_processors(
    config: NanoVLMVFConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    image_keys = tuple(
        key for key, feature in config.input_features.items() if feature.type == FeatureType.VISUAL
    )
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
            NanoVLMNativeProcessorStep(
                pretrained_path=config.nanovlm_pretrained_path,
                code_path=config.nanovlm_code_path,
                image_keys=image_keys,
                max_length=config.tokenizer_max_length,
            ),
            DeviceProcessorStep(device=config.device or "cpu"),
        ],
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline(
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
    )
    return preprocessor, postprocessor
