from __future__ import annotations

from typing import Any
from collections.abc import Sequence

import torch

from lerobot.policies.lawam.latent_world.batch_utils import (
    align_feature_dim_with_mask,
    apply_shared_color_jitter_uint8,
    apply_shared_random_resized_crop_uint8,
    build_placeholder_masks,
    imagenet_normalize_video_,
    sample_or_pad_sequence_with_mask,
)
from lerobot.policies.lawam.latent_world.processor_utils import (
    LatentWorldProcessorSpec,
    load_latent_world_processor,
)
from lerobot.policies.lawam.latent_world.types import (
    LatentWorldPolicyTrainBatch,
    LatentWorldPolicyTrainRawSample,
)
from lerobot.policies.lawam.latent_world.vlm_adapter import build_qwenvl_messages
from lerobot.policies.lawam.latent_world.vlm_adapter import (
    DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
    DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
)


def valid_action_horizon_steps(*, window_size: int, horizon_sec: float, action_hz: float) -> int:
    return min(int(window_size), int(float(horizon_sec) * float(action_hz)))


class LatentWorldTrainCollator:
    def __init__(
        self,
        *,
        policy_cfg: Any,
        processor_spec: LatentWorldProcessorSpec,
        act_queries: int,
        flow_queries: int,
        enable_primary_video_aug: bool,
        enable_primary_random_resized_crop: bool,
        cot_prompt_before_wrist: str | None = None,
        cot_prompt_after_wrist: str | None = None,
    ) -> None:
        self.policy_cfg = policy_cfg
        self.processor_spec = processor_spec
        self.placeholder_token = str(processor_spec.placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)
        self.enable_primary_video_aug = bool(enable_primary_video_aug)
        self.enable_primary_random_resized_crop = bool(enable_primary_random_resized_crop)
        self.cot_prompt_before_wrist = (
            str(cot_prompt_before_wrist)
            if cot_prompt_before_wrist is not None
            else DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT
        )
        self.cot_prompt_after_wrist = (
            str(cot_prompt_after_wrist)
            if cot_prompt_after_wrist is not None
            else DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT
        )
        self.training = True
        self._processor: Any | None = None
        self._placeholder_token_id: int | None = None

    def train(self) -> LatentWorldTrainCollator:
        self.training = True
        return self

    def eval(self) -> LatentWorldTrainCollator:
        self.training = False
        return self

    def _ensure_processor(self) -> tuple[Any, int]:
        if self._processor is None:
            processor, _, placeholder_token_id = load_latent_world_processor(self.processor_spec)
            self._processor = processor
            self._placeholder_token_id = placeholder_token_id
        assert self._processor is not None
        assert self._placeholder_token_id is not None
        return self._processor, self._placeholder_token_id

    @staticmethod
    def _to_wrist_view_list(wrist_images: torch.Tensor) -> list[torch.Tensor]:
        # Match inference semantics: missing wrist views are represented as an empty view list.
        if int(wrist_images.shape[0]) == 0:
            return []
        return [img.contiguous() for img in wrist_images]

    @staticmethod
    def _to_primary_view_list(primary_videos: torch.Tensor) -> list[torch.Tensor]:
        if primary_videos.ndim != 5 or int(primary_videos.shape[1]) < 1:
            raise ValueError(
                "Expected primary_videos with shape [V, T, C, H, W] and T>=1, "
                f"got {tuple(primary_videos.shape)}."
            )
        return [video[0].contiguous() for video in primary_videos]

    @staticmethod
    def _augment_primary_and_wrist_views(
        *,
        primary_videos: torch.Tensor,
        wrist_images: torch.Tensor,
        enable_random_resized_crop: bool,
        enable_color_jitter: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary_videos.ndim != 5:
            raise ValueError(
                f"Expected primary_videos with shape [V, T, C, H, W], got {tuple(primary_videos.shape)}."
            )
        primary_view_count = int(primary_videos.shape[0])
        primary_frame_count = int(primary_videos.shape[1])
        flat_primary = primary_videos.reshape(
            primary_view_count * primary_frame_count, *primary_videos.shape[-3:]
        )

        stacked_views = [flat_primary]
        if int(wrist_images.shape[0]) > 0:
            stacked_views.append(wrist_images)
        augmented_views = torch.cat(stacked_views, dim=0)
        if enable_random_resized_crop:
            augmented_views = apply_shared_random_resized_crop_uint8(augmented_views)
        if enable_color_jitter:
            augmented_views = apply_shared_color_jitter_uint8(augmented_views)

        primary_flat_count = int(flat_primary.shape[0])
        processed_primary_videos = (
            augmented_views[:primary_flat_count].reshape_as(primary_videos).contiguous()
        )
        processed_wrist_images = augmented_views[primary_flat_count:].contiguous()
        return processed_primary_videos, processed_wrist_images

    def __call__(self, features: Sequence[LatentWorldPolicyTrainRawSample]) -> LatentWorldPolicyTrainBatch:
        if len(features) == 0:
            raise ValueError("LatentWorldTrainCollator received an empty feature list.")

        processor, placeholder_token_id = self._ensure_processor()

        image_views_batch: list[list[Any]] = []
        wrist_image_views_batch: list[list[Any]] = []
        primary_video_uint8_seqs: list[torch.Tensor] = []
        instructions: list[str] = []
        actions_list: list[torch.Tensor] = []
        states_list: list[torch.Tensor] = []
        embodiment_ids: list[int] = []
        action_hz_list: list[float] = []

        for sample in features:
            primary_videos = sample["primary_videos"].contiguous()
            wrist_images = sample["wrist_images"].contiguous()
            state_tensor = sample["state"].to(dtype=torch.float32).contiguous()
            action_tensor = sample["action"].to(dtype=torch.float32).contiguous()

            instruction = str(sample["lang"])
            embodiment_id = int(sample["embodiment_id"])
            action_hz = float(sample["action_hz"])

            processed_primary_videos = primary_videos
            processed_wrist_images = wrist_images
            if self.training and (self.enable_primary_random_resized_crop or self.enable_primary_video_aug):
                processed_primary_videos, processed_wrist_images = self._augment_primary_and_wrist_views(
                    primary_videos=primary_videos,
                    wrist_images=wrist_images,
                    enable_random_resized_crop=self.enable_primary_random_resized_crop,
                    enable_color_jitter=self.enable_primary_video_aug,
                )

            image_views_batch.append(self._to_primary_view_list(processed_primary_videos))
            wrist_image_views_batch.append(self._to_wrist_view_list(processed_wrist_images))
            primary_video_uint8_seqs.append(processed_primary_videos[0].contiguous())
            instructions.append(instruction)
            actions_list.append(action_tensor)
            states_list.append(state_tensor)
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)

        qwen_messages = build_qwenvl_messages(
            images=image_views_batch,
            wrist_images=wrist_image_views_batch,
            instructions=instructions,
            placeholder_token=self.placeholder_token,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            cot_prompt_before_wrist=self.cot_prompt_before_wrist,
            cot_prompt_after_wrist=self.cot_prompt_after_wrist,
        )
        qwen_inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = build_placeholder_masks(
            input_ids,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            placeholder_id=placeholder_token_id,
        )

        action_dim = int(self.policy_cfg.flow_cfg.action_dim)
        window_size = int(self.policy_cfg.action_horizon)
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        action_tensors = []
        action_masks = []
        state_tensors = []
        state_masks = []
        for idx, state_tensor in enumerate(states_list):
            action_tensor = actions_list[idx]
            action_tensor, action_dim_mask = align_feature_dim_with_mask(action_tensor, action_dim)
            action_tensor, action_time_mask = sample_or_pad_sequence_with_mask(action_tensor, window_size)
            valid_steps = valid_action_horizon_steps(
                window_size=window_size,
                horizon_sec=float(self.policy_cfg.flow_cfg.horizon_sec),
                action_hz=action_hz_list[idx],
            )
            action_horizon_mask = torch.arange(window_size) < valid_steps
            action_time_mask = action_time_mask & action_horizon_mask
            action_tensors.append(action_tensor)
            action_masks.append(action_time_mask.unsqueeze(1) & action_dim_mask.unsqueeze(0))

            aligned_state, state_dim_mask = align_feature_dim_with_mask(state_tensor, state_dim)
            state_tensors.append(aligned_state[-1].contiguous())
            state_masks.append(state_dim_mask.contiguous())

        primary_video_uint8_batch = torch.stack(primary_video_uint8_seqs, dim=0)
        primary_video_batch = primary_video_uint8_batch.to(dtype=torch.float32).div_(255.0)
        imagenet_normalize_video_(primary_video_batch)

        batch: LatentWorldPolicyTrainBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "primary_video": primary_video_batch,
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "actions": torch.stack(action_tensors, dim=0),
            "actions_mask": torch.stack(action_masks, dim=0),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        return batch
