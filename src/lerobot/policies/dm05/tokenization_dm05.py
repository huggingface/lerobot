#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

"""DM05 tokenization utilities for Gemma3 chat-template inputs."""

from typing import Any

import numpy as np
import torch
from PIL import Image

# Legacy OpenDM dataset camera keys used by DM05 prompt rendering.
OPENDM_CAMERA_LABELS = {
    "images_1": "Head image: ",
    "images_2": "Left wrist image: ",
    "images_3": "Right wrist image: ",
    "front": "Head image: ",
    "cam_high": "Head image: ",
    "wrist": "Left wrist image: ",
    "left_wrist": "Left wrist image: ",
    "cam_left_wrist": "Left wrist image: ",
    "right_wrist": "Right wrist image: ",
    "cam_right_wrist": "Right wrist image: ",
}
DEFAULT_CAMERA_LABELS = tuple(OPENDM_CAMERA_LABELS.values())


def get_camera_labels(meta_data: dict | None, num_images: int) -> list[str]:
    """Resolve human-readable camera labels for the DM05 chat prompt."""
    image_keys = None
    if isinstance(meta_data, dict):
        image_keys = meta_data.get("image_keys")
        if image_keys is None:
            dataset_meta = meta_data.get("dataset_meta", {})
            if isinstance(dataset_meta, dict):
                image_keys = dataset_meta.get("image_keys")
        if not isinstance(image_keys, (list, tuple)):
            image_keys = None
    labels: list[str] = []
    for i in range(num_images):
        if image_keys and i < len(image_keys):
            key = str(image_keys[i])
            short_key = key.split(".")[-1]
            if (label := OPENDM_CAMERA_LABELS.get(key, OPENDM_CAMERA_LABELS.get(short_key))) is None:
                text = " ".join(key.split(".")[-1].replace("_", " ").split()).capitalize()
                label = f"{text} image: "
            labels.append(label)
        elif i < len(DEFAULT_CAMERA_LABELS):
            labels.append(DEFAULT_CAMERA_LABELS[i])
        else:
            labels.append(f"Camera {i + 1} image: ")
    return labels


def action_to_bin_tokens(
    action: np.ndarray,
    n_bins: int = 256,
) -> list[int]:
    """Quantize normalized values into DM05 discrete bin ids."""
    bins = np.floor(((np.clip(action, -1.0, 1.0) + 1.0) / 2.0) * (n_bins - 1)).astype(int)
    return np.clip(bins, 0, n_bins - 1).tolist()


def format_embodiment_spec(meta_data: dict) -> str:
    """Render optional robot and control-mode metadata for the prompt."""
    meta_data = meta_data if isinstance(meta_data, dict) else {}
    robot_type, control_mode = meta_data.get("robot_type"), meta_data.get("control_mode")
    dataset_meta = meta_data.get("dataset_meta", {})
    if robot_type is None:
        robot_type = dataset_meta.get("robot_type")
    if hasattr(robot_type, "value"):
        robot_type = robot_type.value
    if control_mode is None:
        control_mode = dataset_meta.get("control_mode")
    return "".join(
        f"{label}: {value}\n"
        for label, value in (("Robot", robot_type), ("Control mode", control_mode))
        if value is not None
    )


def format_speed_value(speed: Any) -> str | None:
    """Normalize optional speed metadata into prompt text."""
    if speed is None:
        return None

    if isinstance(speed, torch.Tensor):
        if speed.numel() == 0:
            return None
        speed = speed.detach().cpu().numpy()

    if isinstance(speed, np.ndarray):
        if (flat := speed.reshape(-1)).size == 0:
            return None
        return " ".join(f"{float(value):.1f}" for value in flat)

    if isinstance(speed, (list, tuple)):
        return " ".join(filter(None, (format_speed_value(item) for item in speed))) or None

    if isinstance(speed, (int, float, np.integer, np.floating)):
        return f"{float(speed):.1f}"

    if not (speed_text := str(speed).strip()):
        return None
    try:
        return f"{float(speed_text):.1f}"
    except ValueError:
        return speed_text


class DM05Tokenization:
    """Build Gemma3 chat-template inputs for DM05 robot batches."""

    def __init__(
        self,
        processor,
        n_bins: int = 256,
        max_length: int | None = None,
        add_state: bool = True,
    ):
        self.processor = processor
        self.n_bins = n_bins
        self.max_length = max_length
        self.add_state = bool(add_state)

    def _build_user_content(
        self,
        *,
        prompt: str,
        images: list[Image.Image],
        state: np.ndarray,
        meta_data: dict,
        speed_text: str | None,
    ) -> list:
        """Build the multimodal user message consumed by Gemma3Processor."""
        text = format_embodiment_spec(meta_data)
        text += f"Overall speed: {speed_text or '0.5'}\n"
        prompt = prompt.strip()
        text += f"Task: {prompt}{'' if prompt.endswith('.') else '.'}\n"
        if not images:
            raise ValueError("Expected at least one robot image")

        labels = get_camera_labels(meta_data, len(images))
        user_content = [{"type": "text", "text": text + labels[0]}, {"type": "image", "image": images[0]}]
        for image, label in zip(images[1:], labels[1:], strict=True):
            user_content.extend(({"type": "text", "text": label}, {"type": "image", "image": image}))

        if self.add_state:
            state_for_text = np.asarray(state, dtype=np.float32)
            if state_for_text.ndim == 0:
                state_for_text = state_for_text[None]
            if state_for_text.ndim != 1:
                raise ValueError(f"state for text must be a 1D vector, got shape={state_for_text.shape}")
            if (valid_dim_mask := meta_data.get("valid_dim_mask") if meta_data else None) is not None:
                mask = np.asarray(valid_dim_mask, dtype=bool).reshape(-1)
                usable = min(state_for_text.shape[0], mask.shape[0])
                state_for_text = state_for_text[:usable][mask[:usable]]
            state_text = " ".join(str(b) for b in action_to_bin_tokens(state_for_text, n_bins=self.n_bins))
            user_content.append({"type": "text", "text": "States: " + state_text})
        return user_content

    def tokenize_robot_batch(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """Render flow-matching inputs as one processor batch on CPU.

        DM05's LeRobot training objective has no autoregressive token loss.  Use the
        same generation prompt as inference instead of appending discrete action text
        only to mask it again before the action-expert suffix.
        """

        messages = []
        for sample in samples:
            meta_data = sample["meta_data"]
            user_content = self._build_user_content(
                prompt=sample["prompt"],
                images=sample["images"],
                state=sample["state"],
                meta_data=meta_data,
                speed_text=format_speed_value(meta_data.get("speed")),
            )
            messages.append([{"role": "user", "content": user_content}])

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={"padding": True, "padding_side": "right"},
        )
        if self.max_length is not None and inputs["input_ids"].shape[1] > self.max_length:
            lengths = inputs["attention_mask"].sum(dim=1)
            raise ValueError(
                f"Robot batch contains {int(lengths.max())} tokens; "
                f"tokenizer_max_length={self.max_length}. Shorten the task prompt or increase the limit."
            )
        return inputs
