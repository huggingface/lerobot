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

from .constants import IGNORE_AND_MASK_INDEX, IGNORE_INDEX

# Legacy OpenDM dataset camera keys used by DM05 prompt rendering.
OPENDM_CAMERA_LABELS = {
    "images_1": "Head image: ",
    "images_2": "Left wrist image: ",
    "images_3": "Right wrist image: ",
}
DEFAULT_CAMERA_LABELS = tuple(OPENDM_CAMERA_LABELS.values())


def get_camera_labels(meta_data: dict | None, num_images: int) -> list[str]:
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
            if (label := OPENDM_CAMERA_LABELS.get(key)) is None:
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
    bins = np.floor(((np.clip(action, -1.0, 1.0) + 1.0) / 2.0) * (n_bins - 1)).astype(int)
    return np.clip(bins, 0, n_bins - 1).tolist()


def format_embodiment_spec(meta_data: dict) -> str:
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
    def __init__(
        self,
        processor,
        n_bins: int = 256,
        max_length: int | None = None,
        add_state: bool = True,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.n_bins = n_bins
        self.max_length = max_length
        self.add_state = bool(add_state)

    def _encode_as_tensor(self, text: str, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(text, add_special_tokens=False), dtype=like.dtype, device=like.device
        )

    def _build_user_content(
        self,
        *,
        prompt: str,
        images: list[Image.Image],
        state: np.ndarray,
        meta_data: dict,
        speed_text: str | None,
    ) -> list:
        text = format_embodiment_spec(meta_data)
        if speed_text is not None:
            text += f"Overall speed: {speed_text}\n"
        text += f"Task: {prompt}\n"
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
            user_content.append({"type": "text", "text": "State: " + state_text + "\n"})
        return user_content

    def _render_messages(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = False,
    ):
        kwargs = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
        if add_generation_prompt:
            kwargs["add_generation_prompt"] = True
        inputs = self.processor.apply_chat_template(messages, **kwargs)
        input_ids, attention_mask = inputs["input_ids"].squeeze(0), inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        if (token_type_ids := inputs.get("token_type_ids")) is not None:
            token_type_ids = token_type_ids.squeeze(0)
        return input_ids, attention_mask, pixel_values, token_type_ids

    def tokenize_robot(
        self,
        prompt: str,
        images: list[Image.Image],
        state: np.ndarray,
        action: np.ndarray,
        meta_data: dict,
    ) -> dict[str, torch.Tensor]:
        user_content = self._build_user_content(
            prompt=prompt,
            images=images,
            state=state,
            meta_data=meta_data,
            speed_text=format_speed_value(meta_data.get("speed") if meta_data else None),
        )
        action_text = str(meta_data.get("action_text") or "").strip()
        if not action_text:
            action_for_text = np.asarray(action)
            if action_for_text.ndim != 1:
                raise ValueError(f"Unsupported action shape for tokenization: {action_for_text.shape}")
            if (
                valid_action_mask := meta_data.get("valid_action_dim_mask") if meta_data else None
            ) is not None:
                mask = np.asarray(valid_action_mask, dtype=bool).reshape(-1)
                usable = min(action_for_text.shape[0], mask.shape[0])
                action_for_text = action_for_text[:usable][mask[:usable]]
            action_text = " ".join(str(b) for b in action_to_bin_tokens(action_for_text, n_bins=self.n_bins))
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        ]

        input_ids, attention_mask, pixel_values, token_type_ids = self._render_messages(messages)

        if self.max_length is not None and input_ids.shape[0] > self.max_length:
            prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            overflow = int(input_ids.shape[0]) - int(self.max_length)
            keep_tokens = max(0, len(prompt_token_ids) - overflow - 16)
            if keep_tokens < len(prompt_token_ids):
                shortened_prompt = self.tokenizer.decode(
                    prompt_token_ids[:keep_tokens], skip_special_tokens=False
                ).strip()
                for content_item in messages[0]["content"]:
                    if content_item.get("type") != "text" or prompt not in (
                        text := content_item.get("text", "")
                    ):
                        continue
                    content_item["text"] = text.replace(prompt, shortened_prompt, 1)
                    break
                input_ids, attention_mask, pixel_values, token_type_ids = self._render_messages(messages)

        if self.max_length is not None and input_ids.shape[0] > self.max_length:
            raise ValueError(
                "Robot sample length exceeds max_length: "
                f"seq_len={int(input_ids.shape[0])}, max_length={int(self.max_length)}. "
                "Prompt was already shortened; reduce history/context or increase "
                "model_max_length."
            )
        labels = self._build_labels(input_ids, messages)

        result = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        return result

    def tokenize_robot_infer(
        self,
        prompt: str,
        images: list[Image.Image],
        state: np.ndarray,
        meta_data: dict,
        speed: Any = None,
    ) -> dict[str, torch.Tensor]:
        user_content = self._build_user_content(
            prompt=prompt,
            images=images,
            state=state,
            meta_data=meta_data,
            speed_text=format_speed_value(speed if speed is not None else meta_data.get("speed")),
        )
        messages = [{"role": "user", "content": user_content}]
        input_ids, attention_mask, pixel_values, token_type_ids = self._render_messages(
            messages, add_generation_prompt=True
        )

        if self.max_length is not None and input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[: self.max_length]

            vision_token_id = getattr(self.processor, "image_token_id", None)
            if not isinstance(vision_token_id, int) or vision_token_id < 0:
                vision_token_id = getattr(getattr(self.processor, "config", None), "image_token_index", None)
            if pixel_values is not None and isinstance(vision_token_id, int) and vision_token_id >= 0:
                mm_tokens = getattr(getattr(self.processor, "config", None), "mm_tokens_per_image", 256)
                kept_vision = int((input_ids == vision_token_id).sum().item())
                kept_images = kept_vision // mm_tokens if mm_tokens > 0 else 0
                pixel_values = pixel_values[:kept_images]
                effective_vision = kept_images * mm_tokens
                if effective_vision < kept_vision:
                    vision_positions = torch.nonzero(input_ids == vision_token_id, as_tuple=False).flatten()
                    if vision_positions.numel() > effective_vision:
                        keep_len = int(vision_positions[effective_vision].item())
                    else:
                        keep_len = int(input_ids.shape[0])
                    input_ids = input_ids[:keep_len]
                    attention_mask = attention_mask[:keep_len]
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids[:keep_len]

        result = {"input_ids": input_ids, "attention_mask": attention_mask}
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        return result

    def _build_labels(
        self,
        input_ids: torch.Tensor,
        messages: list[dict],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        assistant_texts = [
            item["text"]
            for msg in messages
            if msg["role"] == "assistant"
            for item in msg["content"]
            if item.get("type") == "text"
        ]

        assistant_prefix = self._encode_as_tensor("<start_of_turn>model\n", input_ids)
        eos_token_id = self.tokenizer.encode("<end_of_turn>", add_special_tokens=False)[0]
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        if assistant_prefix.numel() == 0:
            return labels

        search_start = 0
        eos_tensor = torch.tensor([eos_token_id], dtype=input_ids.dtype, device=input_ids.device)
        for assistant_text in assistant_texts:
            assistant_ids = self._encode_as_tensor(assistant_text, input_ids)
            if assistant_ids.numel() == 0:
                continue

            prefix_pos = self._find_subsequence(input_ids, assistant_prefix, start=search_start)
            if prefix_pos < 0:
                break
            turn_start = prefix_pos + int(assistant_prefix.shape[0])
            turn_end = self._find_subsequence(input_ids, eos_tensor, start=turn_start)
            if turn_end < 0:
                break

            pos = self._find_subsequence(input_ids, assistant_ids, start=turn_start)
            if pos >= 0 and pos + int(assistant_ids.shape[0]) <= turn_end:
                end = pos + int(assistant_ids.shape[0])
                labels[pos:end] = input_ids[pos:end]

                if end < input_ids.shape[0] and input_ids[end].item() == eos_token_id:
                    labels[end] = input_ids[end]
                    end += 1
                if newline_ids and end < input_ids.shape[0] and input_ids[end].item() == newline_ids[0]:
                    labels[end] = input_ids[end]
                    end += 1

                for tag_text in ("<eef_actions>", "<joint_actions>"):
                    tag_ids = self._encode_as_tensor(tag_text, input_ids)
                    if tag_ids.numel() == 0:
                        continue
                    tag_len = int(tag_ids.shape[0])
                    tag_pos = self._find_subsequence(input_ids, tag_ids, start=pos)
                    while tag_pos >= 0 and tag_pos + tag_len <= end:
                        labels[tag_pos : tag_pos + tag_len] = IGNORE_AND_MASK_INDEX
                        tag_pos = self._find_subsequence(input_ids, tag_ids, start=tag_pos + tag_len)

                search_start = end
            else:
                search_start = turn_end

        return labels

    @staticmethod
    def _find_subsequence(
        sequence: torch.Tensor,
        subsequence: torch.Tensor,
        start: int = 0,
    ) -> int:
        seq_len, sub_len = sequence.shape[0], subsequence.shape[0]
        if sub_len > seq_len:
            return -1
        for i in range(start, seq_len - sub_len + 1):
            if torch.equal(sequence[i : i + sub_len], subsequence):
                return i
        return -1
