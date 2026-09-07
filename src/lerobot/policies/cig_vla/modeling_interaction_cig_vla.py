from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .action_semantics import make_action_semantics
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
        # `config.action_semantics` picks the adapter that interprets the raw `action` tensor
        # (delta vs. absolute pose, dims, denormalization) -- see action_semantics.py. This used
        # to be hardcoded to LIBERO's delta-OSC_POSE semantics regardless of config, which is
        # wrong for e.g. VLABench's absolute-EE-pose actions (see
        # VLABenchAbsoluteEEFActionSemantics's docstring for why that silently corrupts the
        # geometry/causal-intervention training targets rather than erroring).
        self.action_semantics = make_action_semantics(config.action_semantics)
        self.target_builder = TrajectoryGeometryTargetBuilder(
            action_semantics=self.action_semantics, require_physical_scale=True
        )
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        self._action_queue.clear()

    def _images(self, batch):
        # Camera count/naming isn't fixed to LIBERO-Safety's two (image, wrist_image) --
        # Qwen3VLGroundingBackbone.build_inputs() just interleaves however many camera keys are
        # present as separate image tokens, so this works for e.g. VLABench's three cameras
        # (image/second_image/wrist_image) too. Sorted so key order is stable across batches.
        #
        # Exclude "*_is_pad": resolve_delta_timestamps() applies cfg.observation_delta_indices
        # to visual keys too, and dataset_reader._get_query_indices() emits a same-prefixed
        # f"{key}_is_pad" BoolTensor companion for every key that gets delta timestamps (see
        # datasets/factory.py / datasets/dataset_reader.py). Those also start with
        # "observation.images." but shape to (batch, len(delta_idx)) -- with LIBERO-Safety's
        # hardcoded `!= 2` camera check removed, batch[key][index] on one of these hands a
        # (1,)-shaped tensor to the Qwen processor as if it were a camera frame, which fails
        # deep inside HF's image preprocessing with "Unsupported number of image dimensions: 1"
        # rather than erroring here with a clear message.
        keys = sorted(
            key
            for key in batch
            if key.startswith("observation.images.") and not key.endswith("_is_pad")
        )
        if not keys:
            keys = [key for key in ("observation.image", "observation.wrist_image") if key in batch]
        if not keys:
            raise ValueError("CIG-VLA expects at least one observation.images.* camera in the batch, got none")
        return [[batch[key][index] for key in keys] for index in range(batch[keys[0]].shape[0])]

    def _tasks(self, batch):
        tasks = batch.get("task")
        if tasks is None:
            raise ValueError(
                "LIBERO-Safety task_index must be mapped to task instruction before policy forward"
            )
        return [tasks] if isinstance(tasks, str) else list(tasks)

    def _state(self, batch):
        """(batch, state_dim) proprio state for this single-observation-step model.

        LIBERO-Safety's own adapter hands over an already-squeezed (batch, state_dim)
        tensor, but the standard LeRobotDataset path (e.g. VLABench's
        lerobot/vlabench_unified) keeps the observation-delta-timestep axis even when it
        has length 1 (config.n_obs_steps == 1, config.observation_delta_indices == [0]) --
        see resolve_delta_timestamps() / dataset_reader.py, which apply that delta to every
        "observation.*" key, not just LIBERO-Safety's pre-flattened ones. Neither
        InteractionGeometryHead nor FlowMatchingController model state history (both do a
        single Linear over the last dim), and both assert/break on the extra axis -- e.g.
        broadcasting the (batch, 1, 1, hidden) projected state against (batch, 5, hidden)
        queries silently produces a 4-D tensor that only fails deep inside
        nn.MultiheadAttention ("received 4-D query tensor") -- so normalize once here
        rather than at every call site.
        """
        state = batch[OBS_STATE]
        return state[:, -1] if state.ndim == 3 else state

    def _predict(self, batch):
        hidden, attention_mask = self.backbone.encode_multimodal(self._images(batch), self._tasks(batch))
        return self.grounding_head(hidden, attention_mask, self._state(batch))

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
        state = self._state(batch)
        prediction = self._predict(batch)
        target = self.target_builder.build(
            batch[ACTION], state, self.dataset_stats, batch.get("action_is_pad")
        )
        geometry_loss, metrics = self.compute_geometry_loss(prediction, target)
        bottleneck = prediction.detached() if self.config.detach_bottleneck_for_main_action else prediction
        flow = make_flow_training_sample(batch[ACTION], batch.get("action_is_pad"))
        velocity = self.controller(bottleneck, state, flow.noisy_actions, flow.timestep)
        action_loss = compute_flow_loss(velocity, flow.target_velocity, flow.action_is_pad)
        causal_loss = action_loss * 0
        if self.config.enable_causal_intervention:
            causal_bottleneck = (
                prediction.detached() if self.config.detach_bottleneck_for_causal_branch else prediction
            )
            offset = torch.zeros_like(causal_bottleneck.translation_goal)
            offset[:, 0] = self.config.translation_goal_shift_m
            intervened = causal_bottleneck.with_translation_offset(offset)
            original_velocity = self.controller(causal_bottleneck, state, flow.noisy_actions, flow.timestep)
            changed_velocity = self.controller(intervened, state, flow.noisy_actions, flow.timestep)
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
        state = self._state(batch)
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
