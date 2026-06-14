from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .batch_utils import (
    align_feature_dim_with_mask,
    build_placeholder_masks,
    imagenet_normalize_image_batch_,
    prepare_frame_spatial_uint8,
)
from .types import LatentWorldPolicyInferBatch, LatentWorldPolicyInferExample


class LatentWorldPolicyInferBatchBuilder:
    DEFAULT_INFER_IMAGE_HW = (256, 256)
    _ALLOWED_INFER_KEYS = {
        "lang",
        "primary_image",
        "action_hz",
        "embodiment_id",
        "state",
        "state_mask",
        "wrist_image",
    }
    _REQUIRED_INFER_KEYS = {"lang", "primary_image", "action_hz", "embodiment_id"}

    def __init__(
        self,
        *,
        policy_cfg,
        policy_backend,
        policy_vlm_adapter,
        enable_primary_random_resized_crop: bool = False,
    ) -> None:
        self.policy_cfg = policy_cfg
        self.policy_backend = policy_backend
        self.policy_vlm_adapter = policy_vlm_adapter
        self.infer_image_hw = self.DEFAULT_INFER_IMAGE_HW
        self.enable_primary_random_resized_crop = bool(enable_primary_random_resized_crop)

    def _target_device(self) -> torch.device:
        return next(self.policy_backend.parameters()).device

    @staticmethod
    def _move_batch_to_device(batch: LatentWorldPolicyInferBatch, device: torch.device) -> LatentWorldPolicyInferBatch:
        moved: dict[str, torch.Tensor | None] = {}
        for key, value in batch.items():
            moved[key] = value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value
        return batch.__class__(moved)

    def build_infer_batch(self, examples: Sequence[LatentWorldPolicyInferExample]) -> LatentWorldPolicyInferBatch:
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        image_views_batch: list[list[Image.Image]] = []
        wrist_image_views_batch: list[list[Image.Image]] = []
        primary_image_tensors: list[torch.Tensor] = []
        instructions: list[str] = []
        state_tensors: list[torch.Tensor] = []
        state_masks: list[torch.Tensor] = []
        embodiment_ids: list[int] = []
        action_hz_list: list[float] = []

        for ex_idx, ex in enumerate(examples):
            extra_keys = sorted(set(ex.keys()) - self._ALLOWED_INFER_KEYS)
            if extra_keys:
                raise KeyError(
                    "inference examples[{idx}] contains unsupported keys {keys}. "
                    "Allowed keys are: {allowed}.".format(
                        idx=ex_idx,
                        keys=extra_keys,
                        allowed=sorted(self._ALLOWED_INFER_KEYS),
                    )
                )
            missing_keys = sorted(self._REQUIRED_INFER_KEYS - set(ex.keys()))
            if missing_keys:
                raise KeyError(
                    f"inference examples[{ex_idx}] missing required keys {missing_keys}. "
                    f"Required keys are: {sorted(self._REQUIRED_INFER_KEYS)}."
                )

            instruction = str(ex["lang"])
            primary_frames = self._extract_primary_frames(ex["primary_image"], ex_idx=ex_idx)
            wrist_frames = self._extract_wrist_frames(ex)

            action_hz = float(ex["action_hz"])
            if action_hz <= 0.0:
                raise ValueError(f"inference examples[{ex_idx}]['action_hz'] must be > 0, got {action_hz}.")
            embodiment_id = int(ex["embodiment_id"])

            if "state" in ex:
                state_seq = self._to_2d_sequence_tensor(ex["state"], field_name="state", ex_idx=ex_idx)
                aligned_state, state_dim_mask = align_feature_dim_with_mask(state_seq, state_dim)
                if "state_mask" in ex:
                    explicit_state_mask = self._to_2d_sequence_mask_tensor(
                        ex["state_mask"],
                        field_name="state_mask",
                        ex_idx=ex_idx,
                    )
                    explicit_state_mask = self._align_mask_tensor(
                        explicit_state_mask,
                        target_dim=state_dim,
                    )
                    if explicit_state_mask.shape[0] not in (1, aligned_state.shape[0]):
                        raise ValueError(
                            "examples[{idx}]['state_mask'] must have shape [D] or [T, D] matching state T; "
                            "got state_mask={mask_shape}, state={state_shape}.".format(
                                idx=ex_idx,
                                mask_shape=tuple(explicit_state_mask.shape),
                                state_shape=tuple(aligned_state.shape),
                            )
                        )
                    state_dim_mask = state_dim_mask & explicit_state_mask[-1]
                state_tensors.append(aligned_state[-1])
                state_masks.append(state_dim_mask)
            else:
                if "state_mask" in ex:
                    raise KeyError(
                        f"inference examples[{ex_idx}] provides `state_mask` without `state`; "
                        "please provide both or neither."
                    )
                state_tensors.append(torch.zeros((state_dim,), dtype=torch.float32))
                state_masks.append(torch.zeros((state_dim,), dtype=torch.bool))

            processed_primary_frames_uint8 = [
                prepare_frame_spatial_uint8(
                    primary_frame,
                    target_hw=self.infer_image_hw,
                    apply_center_crop_90=self.enable_primary_random_resized_crop,
                )
                for primary_frame in primary_frames
            ]
            processed_wrist_pils = [
                self._chw_uint8_to_pil(
                    prepare_frame_spatial_uint8(
                        wrist_frame,
                        target_hw=self.infer_image_hw,
                        apply_center_crop_90=self.enable_primary_random_resized_crop,
                    )
                )
                for wrist_frame in wrist_frames
            ]

            image_views_batch.append([
                self._chw_uint8_to_pil(processed_primary_frame_uint8)
                for processed_primary_frame_uint8 in processed_primary_frames_uint8
            ])
            wrist_image_views_batch.append(processed_wrist_pils)
            primary_image_tensors.append(processed_primary_frames_uint8[0].to(dtype=torch.float32).div_(255.0))
            instructions.append(instruction)
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)

        qwen_inputs = self.policy_vlm_adapter.build_qwenvl_inputs(
            images=image_views_batch,
            wrist_images=wrist_image_views_batch,
            instructions=instructions,
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = build_placeholder_masks(
            input_ids,
            act_queries=int(self.policy_backend.num_action_queries),
            flow_queries=int(self.policy_backend.flow_action_query.shape[0]),
            placeholder_id=int(self.policy_backend.placeholder_token_id),
        )

        primary_image_batch = torch.stack(primary_image_tensors, dim=0)
        imagenet_normalize_image_batch_(primary_image_batch)

        batch: LatentWorldPolicyInferBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "primary_image": primary_image_batch,
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        return self._move_batch_to_device(batch, self._target_device())

    @staticmethod
    def _chw_uint8_to_pil(frame: torch.Tensor) -> Image.Image:
        if frame.ndim != 3 or int(frame.shape[0]) != 3:
            raise ValueError(f"Expected frame tensor with shape [3, H, W], got {tuple(frame.shape)}.")
        frame_hwc = frame.permute(1, 2, 0).contiguous().cpu().numpy()
        return Image.fromarray(frame_hwc, mode="RGB")

    @staticmethod
    def _extract_primary_frames(primary_image, *, ex_idx: int) -> list[np.ndarray]:
        if not isinstance(primary_image, (list, tuple)):
            raise ValueError(
                f"examples[{ex_idx}]['primary_image'] must be a list/tuple, got type={type(primary_image)}."
            )
        if len(primary_image) < 1:
            raise ValueError(
                f"examples[{ex_idx}]['primary_image'] must contain at least 1 primary view, got {len(primary_image)}."
            )
        primary_frames: list[np.ndarray] = []
        for primary_idx, primary_frame in enumerate(primary_image):
            if not isinstance(primary_frame, np.ndarray):
                raise ValueError(
                    f"examples[{ex_idx}]['primary_image'][{primary_idx}] must be np.ndarray, "
                    f"got type={type(primary_frame)}."
                )
            primary_frames.append(primary_frame)
        return primary_frames

    @staticmethod
    def _extract_wrist_frames(ex) -> list[np.ndarray]:
        # Match training semantics: missing wrist views are represented as an empty view list.
        if "wrist_image" not in ex:
            return []

        wrist_image = ex["wrist_image"]
        if wrist_image is None:
            return []
        if not isinstance(wrist_image, (list, tuple)):
            raise ValueError(f"`wrist_image` must be a list/tuple, got type={type(wrist_image)}.")
        if len(wrist_image) < 1:
            return []

        wrist_frames: list[np.ndarray] = []
        for wrist_idx, wrist_frame in enumerate(wrist_image):
            if not isinstance(wrist_frame, np.ndarray):
                raise ValueError(
                    f"`wrist_image`[{wrist_idx}] must be np.ndarray, got type={type(wrist_frame)}."
                )
            wrist_frames.append(wrist_frame)
        return wrist_frames

    @staticmethod
    def _to_cpu_float_tensor(value, *, field_name: str, ex_idx: int) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.detach()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            tensor = torch.as_tensor(value)
        else:
            raise TypeError(
                f"examples[{ex_idx}]['{field_name}'] must be torch.Tensor/np.ndarray/list/tuple, "
                f"got type={type(value)}."
            )
        return tensor.to(dtype=torch.float32)

    @classmethod
    def _to_2d_sequence_tensor(cls, value, *, field_name: str, ex_idx: int) -> torch.Tensor:
        tensor = cls._to_cpu_float_tensor(value, field_name=field_name, ex_idx=ex_idx)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"examples[{ex_idx}]['{field_name}'] must have shape [D] or [T, D], got {tuple(tensor.shape)}."
            )
        return tensor

    @classmethod
    def _to_2d_sequence_mask_tensor(cls, value, *, field_name: str, ex_idx: int) -> torch.Tensor:
        tensor = cls._to_cpu_float_tensor(value, field_name=field_name, ex_idx=ex_idx)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(
                f"examples[{ex_idx}]['{field_name}'] must have shape [D] or [T, D], got {tuple(tensor.shape)}."
            )
        return tensor.to(dtype=torch.bool)

    @staticmethod
    def _align_mask_tensor(mask: torch.Tensor, *, target_dim: int) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask tensor with shape [T, D], got {tuple(mask.shape)}.")
        curr_dim = int(mask.shape[1])
        if curr_dim == target_dim:
            return mask
        if curr_dim > target_dim:
            return mask[:, :target_dim]
        pad = torch.zeros((int(mask.shape[0]), target_dim - curr_dim), dtype=torch.bool, device=mask.device)
        return torch.cat([mask, pad], dim=1)
