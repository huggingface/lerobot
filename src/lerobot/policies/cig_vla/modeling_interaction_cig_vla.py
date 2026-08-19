from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .action_semantics import LiberoSafetyActionSemantics
from .configuration_cig_vla import CIGVLAConfig
from .flow_controller import FlowMatchingController
from .flow_matching import compute_flow_loss, make_flow_training_sample, velocity_to_action_estimate
from .interaction_head import InteractionGeometryHead
from .qwen3vl_backbone import Qwen3VLGroundingBackbone
from .trajectory_geometry import TrajectoryGeometryTargetBuilder


def _masked_mean(values, mask):
    mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return values[mask].mean() if mask.any() else values.sum() * 0


class CIGVLAPolicy(PreTrainedPolicy):
    config_class = CIGVLAConfig
    name = "cig_vla"

    def __init__(self, config, backbone: nn.Module | None = None, dataset_stats=None, dataset_meta=None):
        super().__init__(config)
        del dataset_meta
        state_dim = (
            config.robot_state_feature.shape[0] if config.robot_state_feature else config.max_state_dim
        )
        action_dim = config.action_feature.shape[0] if config.action_feature else config.max_action_dim
        self.backbone = backbone or Qwen3VLGroundingBackbone(
            config.qwen_model_name,
            config.torch_dtype,
            config.freeze_vision_tower,
            config.gradient_checkpointing,
            config.lora_rank if config.enable_qwen_lora else 0,
            config.lora_alpha,
            config.lora_dropout,
            config.lora_bias,
        )
        self.grounding_head = InteractionGeometryHead(
            self.backbone.hidden_size,
            state_dim,
            config.grounding_hidden_dim,
            config.grounding_num_heads,
            config.grounding_num_layers,
        )
        self.controller = FlowMatchingController(
            state_dim,
            action_dim,
            config.controller_hidden_dim,
            config.controller_num_layers,
            config.controller_num_heads,
            config.bottleneck_mode,
        )
        self.dataset_stats = dataset_stats
        self.target_builder = TrajectoryGeometryTargetBuilder(require_physical_scale=True)
        self.action_semantics = LiberoSafetyActionSemantics()
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._action_queue.clear()

    def _images(self, batch):
        keys = sorted(key for key in batch if key.startswith("observation.images."))
        if not keys:
            keys = [key for key in ("observation.image", "observation.wrist_image") if key in batch]
        if len(keys) != 2:
            raise ValueError(f"CIG-VLA LIBERO-Safety expects exactly two cameras, got {keys}")
        return [[batch[key][index] for key in keys] for index in range(batch[keys[0]].shape[0])]

    def _tasks(self, batch):
        tasks = batch.get("task")
        if tasks is None:
            raise ValueError(
                "LIBERO-Safety task_index must be mapped to task instruction before policy forward"
            )
        return [tasks] if isinstance(tasks, str) else list(tasks)

    def _predict(self, batch):
        hidden, attention_mask = self.backbone.encode_multimodal(self._images(batch), self._tasks(batch))
        return self.grounding_head(hidden, attention_mask, batch[OBS_STATE])

    def compute_geometry_loss(self, prediction, target):
        valid = target.valid_mask
        direction_valid = valid & (target.translation_magnitude > self.target_builder.motion_threshold)
        gripper_valid = valid
        gripper_transition_mask = valid & (
            target.gripper_transition.abs() > self.target_builder.motion_threshold
        )
        translation = _masked_mean(
            functional.smooth_l1_loss(prediction.translation_goal, target.translation_goal, reduction="none"),
            valid,
        )
        direction = _masked_mean(
            1
            - functional.cosine_similarity(prediction.approach_direction, target.approach_direction, dim=-1),
            direction_valid.squeeze(-1),
        )
        magnitude = _masked_mean(
            functional.smooth_l1_loss(
                prediction.translation_magnitude, target.translation_magnitude, reduction="none"
            ),
            valid,
        )
        gripper = _masked_mean(
            functional.smooth_l1_loss(
                prediction.gripper_transition, target.gripper_transition, reduction="none"
            ),
            gripper_valid,
        )
        rotation = translation * 0
        total = (
            self.config.translation_goal_loss_weight * translation
            + self.config.approach_direction_loss_weight * direction
            + self.config.translation_magnitude_loss_weight * magnitude
            + self.config.rotation_goal_loss_weight * rotation
            + self.config.gripper_transition_loss_weight * gripper
        )
        return total, {
            "geometry_loss": total.detach(),
            "translation_goal_loss": translation.detach(),
            "approach_direction_loss": direction.detach(),
            "direction_loss": direction.detach(),
            "translation_magnitude_loss": magnitude.detach(),
            "rotation_goal_loss": rotation.detach(),
            "gripper_transition_loss": gripper.detach(),
            "trajectory_geometry_valid_count": valid.sum().detach(),
            "valid_interaction_count": valid.sum().detach(),
            "direction_valid_count": direction_valid.sum().detach(),
            "rotation_valid_count": torch.zeros((), device=valid.device, dtype=torch.long),
            "gripper_transition_count": gripper_transition_mask.sum().detach(),
        }

    def forward(self, batch):
        prediction = self._predict(batch)
        target = self.target_builder.build(
            batch[ACTION], batch[OBS_STATE], self.dataset_stats, batch.get("action_is_pad")
        )
        geometry_loss, metrics = self.compute_geometry_loss(prediction, target)
        bottleneck = prediction.detached() if self.config.detach_bottleneck_for_main_action else prediction
        flow = make_flow_training_sample(batch[ACTION], batch.get("action_is_pad"))
        velocity = self.controller(bottleneck, batch[OBS_STATE], flow.noisy_actions, flow.timestep)
        action_loss = compute_flow_loss(velocity, flow.target_velocity, flow.action_is_pad)
        causal_loss = action_loss * 0
        if self.config.enable_causal_intervention:
            causal_bottleneck = (
                prediction.detached() if self.config.detach_bottleneck_for_causal_branch else prediction
            )
            offset = torch.zeros_like(causal_bottleneck.translation_goal)
            offset[:, 0] = self.config.translation_goal_shift_m
            intervened = causal_bottleneck.with_translation_offset(offset)
            original_velocity = self.controller(
                causal_bottleneck, batch[OBS_STATE], flow.noisy_actions, flow.timestep
            )
            changed_velocity = self.controller(
                intervened, batch[OBS_STATE], flow.noisy_actions, flow.timestep
            )
            original_estimate = velocity_to_action_estimate(
                flow.noisy_actions, original_velocity, flow.timestep
            )
            changed_estimate = velocity_to_action_estimate(
                flow.noisy_actions, changed_velocity, flow.timestep
            )
            original_physical = self.action_semantics.denormalize_actions(
                original_estimate, self.dataset_stats
            )
            changed_physical = self.action_semantics.denormalize_actions(changed_estimate, self.dataset_stats)
            if original_physical is not None and changed_physical is not None:
                delta = self.action_semantics.aggregate_translation(
                    changed_physical - original_physical, self.config.causal_action_prefix_steps
                )
                signed_response = (delta * offset).sum(dim=-1)
                response = (changed_estimate - original_estimate).flatten(1).norm(dim=-1)
                causal_loss = (
                    functional.relu(-signed_response).mean()
                    + functional.relu(self.config.causal_response_margin - response).mean()
                )
        total = (
            self.config.action_loss_weight * action_loss
            + geometry_loss
            + self.config.causal_loss_weight * causal_loss
        )
        metrics.update(
            {"loss": total.detach(), "action_loss": action_loss.detach(), "causal_loss": causal_loss.detach()}
        )
        return total, metrics

    @torch.no_grad()
    def predict_geometric_bottleneck(self, batch):
        return self._predict(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch, bottleneck_override=None):
        bottleneck = bottleneck_override or self.predict_geometric_bottleneck(batch)
        state = batch[OBS_STATE]
        action_dim = (
            self.config.action_feature.shape[0] if self.config.action_feature else self.config.max_action_dim
        )
        noise = torch.randn(
            state.shape[0], self.config.chunk_size, action_dim, device=state.device, dtype=state.dtype
        )
        return euler_integrate(
            lambda actions, timestep: self.controller(bottleneck, state, actions, timestep),
            noise,
            self.config.num_inference_steps,
        )

    @torch.no_grad()
    def select_action(self, batch):
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk[:, index] for index in range(chunk.shape[1]))
        return self._action_queue.popleft()
