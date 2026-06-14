from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

import torch

from lerobot.policies.lawam.latent_world.types import (
    LatentWorldPolicyInferBatch,
    LatentWorldPolicyInferExample,
    LatentWorldPolicyTrainBatch,
)

from .output_mapper import map_policy_infer_output, map_policy_train_output

if TYPE_CHECKING:
    from lerobot.policies.lawam.latent_world.batch_builder import LatentWorldPolicyInferBatchBuilder
    from lerobot.policies.lawam.vlas.lawam import LatentWorldPolicyBackend


class LatentWorldPolicyRunner:
    def __init__(
        self,
        *,
        policy_backend: "LatentWorldPolicyBackend",
        infer_batch_builder: "LatentWorldPolicyInferBatchBuilder",
    ) -> None:
        self.policy_backend = policy_backend
        self.infer_batch_builder = infer_batch_builder

    def train_step(self, batch: LatentWorldPolicyTrainBatch) -> Dict[str, torch.Tensor]:
        policy_output = self.policy_backend.forward(batch=batch)
        return map_policy_train_output(policy_output)

    @torch.inference_mode()
    def infer_step(
        self,
        examples: Sequence[LatentWorldPolicyInferExample],
        *,
        return_intermediates: bool = False,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
    ) -> Dict[str, Any]:
        if len(examples) == 0:
            raise ValueError("`infer_step` requires at least one example.")
        batch = self.infer_batch_builder.build_infer_batch(examples)
        batch_size = int(batch["action_hz"].shape[0])
        if batch_size != len(examples):
            raise ValueError(
                "Inference batch size mismatch after batch build: "
                f"examples={len(examples)}, batch_size={batch_size}."
            )

        horizon_sec = float(self.policy_backend.flow.config.horizon_sec)
        hz_values = batch["action_hz"].detach().cpu().tolist()
        expected_lens = [int(math.floor(horizon_sec * float(hz))) for hz in hz_values]
        if any(expected_len < 1 for expected_len in expected_lens):
            bad_idx = next(idx for idx, expected_len in enumerate(expected_lens) if expected_len < 1)
            raise ValueError(
                "Invalid effective action length for inference: "
                f"sample={bad_idx}, floor(horizon_sec * action_hz)={expected_lens[bad_idx]}, "
                f"horizon_sec={horizon_sec}, action_hz={float(hz_values[bad_idx])}."
            )
        if len(set(expected_lens)) != 1:
            raise ValueError(
                "Real-time batched inference currently requires all examples to share the same "
                "effective action length. Got expected_lens={expected_lens} from action_hz={hz_values} "
                "with horizon_sec={horizon_sec}.".format(
                    expected_lens=expected_lens,
                    hz_values=hz_values,
                    horizon_sec=horizon_sec,
                )
            )
        expected_len = int(expected_lens[0])

        actions = self.policy_backend.predict_action(
            batch=batch,
            return_intermediates=bool(return_intermediates),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        intermediates = None
        if isinstance(actions, tuple):
            actions, intermediates = actions
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions)
        actual_len = int(actions.shape[1])
        if actual_len != expected_len:
            raise ValueError(
                "Inference action length mismatch: "
                f"actual_len={actual_len}, expected_len={expected_len}, "
                f"horizon_sec={horizon_sec}, action_hz={hz_values}."
            )
        return map_policy_infer_output(actions, intermediates=intermediates)

    @torch.inference_mode()
    def infer_step_with_aligned_targets_from_train_batch(
        self,
        train_batch: LatentWorldPolicyTrainBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        primary_video = train_batch["primary_video"]
        if primary_video.ndim != 5 or int(primary_video.shape[1]) < 1:
            raise ValueError(
                "Expected `primary_video` tensor with shape [B, T, C, H, W] and T>=1 "
                f"for aligned inference, got {tuple(primary_video.shape)}."
            )

        infer_batch: LatentWorldPolicyInferBatch = {
            "pixel_values": train_batch["pixel_values"],
            "input_ids": train_batch["input_ids"],
            "attention_mask": train_batch["attention_mask"],
            "act_placeholder_mask": train_batch["act_placeholder_mask"],
            "flow_placeholder_mask": train_batch["flow_placeholder_mask"],
            "primary_image": primary_video[:, 0, :, :, :],
            "state": train_batch["state"],
            "state_mask": train_batch["state_mask"],
            "embodiment_id": train_batch["embodiment_id"],
            "action_hz": train_batch["action_hz"],
            "image_grid_thw": train_batch["image_grid_thw"],
        }

        pred_actions = self.policy_backend.predict_action(
            batch=infer_batch,
            return_padded=True,
        )
        if isinstance(pred_actions, tuple):
            pred_actions = pred_actions[0]
        if not torch.is_tensor(pred_actions):
            pred_actions = torch.as_tensor(pred_actions)

        target_actions = train_batch["actions"]
        action_mask = train_batch["actions_mask"]
        if not torch.is_tensor(target_actions) or not torch.is_tensor(action_mask):
            raise TypeError("LatentWorld eval expects tensor `actions` and `actions_mask` from train_batch.")

        pred_actions = pred_actions.detach()
        target_actions = target_actions.to(
            device=pred_actions.device,
            dtype=pred_actions.dtype,
            non_blocking=True,
        )
        action_mask = action_mask.to(device=pred_actions.device, dtype=torch.bool, non_blocking=True)
        return pred_actions, target_actions, action_mask
