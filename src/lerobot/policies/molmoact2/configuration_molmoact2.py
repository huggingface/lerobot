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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    LRSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig


class MolmoAct2AdamW(torch.optim.AdamW):
    """AdamW with component clipping and low-memory BF16 update compensation.

    LeRobot's shared trainer clips the full policy as one vector before calling
    ``Optimizer.step``.  Official MolmoAct2 instead clips each optimizer group
    (LLM, ViT, connector, and action expert) independently.  Keeping the clip
    here lets the policy match that behavior without changing the shared
    trainer or any other policy.

    Parameter and optimizer-state dtypes follow the Pi0.5-style storage policy:
    the large VLM tensors remain BF16, while the action expert and explicitly
    sensitive tensors remain FP32. A lazy BF16 Kahan-style residual preserves
    VLM updates smaller than one BF16 parameter ULP. This costs one BF16 tensor
    per trainable BF16 parameter, rather than the FP32 parameter and optimizer
    copies required by the official AMP/FSDP recipe. Native AdamW remains the
    fast path for FP32 parameters (including the complete action expert and
    LoRA adapters).
    """

    def __init__(self, params, *, group_grad_clip_norm: float, **kwargs) -> None:
        if group_grad_clip_norm <= 0:
            raise ValueError(f"MolmoAct2 group_grad_clip_norm must be positive, got {group_grad_clip_norm}.")
        super().__init__(params, **kwargs)
        self.group_grad_clip_norm = float(group_grad_clip_norm)

    def _clip_grad_groups(self) -> tuple[torch.Tensor, ...]:
        norms: list[torch.Tensor] = []
        for group in self.param_groups:
            params_with_grad = [param for param in group["params"] if param.grad is not None]
            if not params_with_grad:
                continue
            norms.append(
                torch.nn.utils.clip_grad_norm_(
                    params_with_grad,
                    max_norm=self.group_grad_clip_norm,
                    error_if_nonfinite=False,
                )
            )
        return tuple(norms)

    def _step_native_non_bfloat16(self) -> None:
        """Run PyTorch's native AdamW only for non-BF16 parameters."""
        original_group_params: list[list[torch.Tensor]] = []
        try:
            for group in self.param_groups:
                original_params = group["params"]
                original_group_params.append(original_params)
                group["params"] = [param for param in original_params if param.dtype != torch.bfloat16]
            super().step()
        finally:
            for group, original_params in zip(self.param_groups, original_group_params, strict=True):
                group["params"] = original_params

    def _step_compensated_bfloat16(self) -> None:
        """Apply AdamW to BF16 storage while retaining sub-ULP updates.

        Adam moments intentionally remain in BF16 to keep the Pi0.5-style
        memory envelope. ``effective_parameter`` is one per-parameter FP32
        temporary, never a persistent model-sized master copy. The residual is
        BF16 and is initialized lazily only for parameters that receive a
        gradient, so frozen VLM weights and LoRA-only runs pay no extra cost.
        """
        for group in self.param_groups:
            bfloat16_group = dict(group)
            bfloat16_group["params"] = [param for param in group["params"] if param.dtype == torch.bfloat16]

            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            self._init_group(
                bfloat16_group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            if torch.is_tensor(lr):
                lr = lr.item()
            lr = float(lr)

            for index, parameter in enumerate(params_with_grad):
                grad = grads[index]
                if group["maximize"]:
                    grad = -grad

                state = self.state[parameter]
                compensation = state.get("compensation")
                if compensation is None:
                    compensation = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["compensation"] = compensation

                state_step = state_steps[index]
                state_step.add_(1)
                step_value = state_step.item()

                exp_avg = exp_avgs[index]
                exp_avg_sq = exp_avg_sqs[index]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_value
                bias_correction2_sqrt = (1 - beta2**step_value) ** 0.5
                step_size = lr / bias_correction1
                if group["amsgrad"]:
                    max_exp_avg_sq = max_exp_avg_sqs[index]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                effective_parameter = parameter.float().add_(compensation)
                if group["weight_decay"] != 0:
                    effective_parameter.mul_(1 - lr * group["weight_decay"])
                effective_parameter.addcdiv_(exp_avg, denom, value=-step_size)
                parameter.copy_(effective_parameter)
                compensation.copy_(effective_parameter.sub_(parameter))

    @staticmethod
    def _any_nonfinite_across_ranks(norms: tuple[torch.Tensor, ...]) -> bool:
        """Match official MolmoAct2's all-rank non-finite step guard."""
        local_nonfinite = any(not bool(torch.isfinite(norm).all().item()) for norm in norms)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local_nonfinite

        backend = str(torch.distributed.get_backend()).lower()
        if "nccl" in backend:
            flag_device = torch.device("cuda", torch.cuda.current_device())
        else:
            flag_device = torch.device("cpu")
        nonfinite_flag = torch.tensor(int(local_nonfinite), device=flag_device, dtype=torch.int32)
        torch.distributed.all_reduce(nonfinite_flag, op=torch.distributed.ReduceOp.MAX)
        return bool(nonfinite_flag.item())

    @torch.no_grad()
    def step(self, closure=None):
        # LeRobot never supplies a closure, but preserve standard Optimizer
        # semantics for callers that do.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grad_norms = self._clip_grad_groups()
        if self._any_nonfinite_across_ranks(grad_norms):
            # Official MolmoAct2 skips the update on every rank and clears the
            # invalid gradients.  In particular, Adam moments and step counts
            # must not advance when one component has a non-finite norm.
            self.zero_grad(set_to_none=True)
            return loss
        self._step_native_non_bfloat16()
        self._step_compensated_bfloat16()
        return loss


@OptimizerConfig.register_subclass("molmoact2_adamw")
@dataclass
class MolmoAct2AdamWConfig(OptimizerConfig):
    """Policy-local AdamW preset with independent per-component clipping."""

    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-6
    weight_decay: float = 0.0
    # Zero disables the shared trainer's global clipping. The policy's existing
    # optimizer_grad_clip_norm is passed through as the group-wise threshold.
    grad_clip_norm: float = 0.0
    group_grad_clip_norm: float = 1.0

    def build(self, params) -> torch.optim.Optimizer:
        return MolmoAct2AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            group_grad_clip_norm=self.group_grad_clip_norm,
        )


@LRSchedulerConfig.register_subclass("molmoact2_cosine_with_warmup")
@dataclass
class MolmoAct2CosineWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Official MolmoAct2 warmup followed by cosine decay.

    The shared LeRobot Pi0 scheduler evaluates its cosine against the absolute
    global step. That creates a learning-rate discontinuity at the end of
    warmup (especially visible in short profiles). Native MolmoAct2 instead
    starts cosine time at zero after warmup and applies the same multiplier to
    every component-specific base LR.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> LambdaLR:
        if self.num_warmup_steps < 0:
            raise ValueError(f"num_warmup_steps must be >= 0, got {self.num_warmup_steps}.")
        if self.num_decay_steps < 1:
            raise ValueError(f"num_decay_steps must be >= 1, got {self.num_decay_steps}.")
        if self.peak_lr <= 0:
            raise ValueError(f"peak_lr must be > 0, got {self.peak_lr}.")
        if not 0 <= self.decay_lr < self.peak_lr:
            raise ValueError(
                f"decay_lr must be in [0, peak_lr), got decay_lr={self.decay_lr}, peak_lr={self.peak_lr}."
            )

        # Official Trainer uses its configured max_duration as the cosine
        # endpoint. Keep that clock when LeRobot intentionally runs a shorter
        # diagnostic gate (for example, 3K updates on the final 30K schedule).
        # Clamping here would silently compress the gate to the decay floor.
        decay_steps = int(self.num_decay_steps)
        warmup_steps = min(int(self.num_warmup_steps), decay_steps)
        alpha = float(self.decay_lr / self.peak_lr)

        def lr_lambda(current_step: int) -> float:
            # LambdaLR installs lambda(0) before the first optimizer update and
            # LeRobot advances it after every update. Official MolmoAct2 first
            # increments global_step, then evaluates the LR before that
            # update, so its kth optimizer update uses f(k), not f(k - 1).
            step = max(int(current_step) + 1, 0)
            if warmup_steps > 0 and step < warmup_steps:
                return float(step / warmup_steps)
            if step >= decay_steps:
                return alpha
            cosine_span = decay_steps - warmup_steps
            if cosine_span <= 0:
                return alpha
            cosine_step = step - warmup_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_span))
            return alpha + (1.0 - alpha) * cosine_decay

        return LambdaLR(optimizer, lr_lambda, -1)


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """MolmoAct2 policy backed by the converted HF checkpoint implementation."""

    checkpoint_path: str = "allenai/MolmoAct2"
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    # Official MolmoAct2 robot fine-tuning optimizes only the continuous
    # flow-matching objective. Released checkpoints retain discrete-action
    # weights and ``both`` remains available as an explicit ablation.
    action_mode: str = "continuous"
    inference_action_mode: str | None = "continuous"
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    discrete_generation_max_steps: int | None = None
    norm_tag: str | None = None
    # Optional standalone norm_stats.json.  This lets a generic base model use
    # the exact embodiment statistics published with a downstream checkpoint
    # without changing which model weights are loaded.
    norm_stats_path: str | None = None

    setup_type: str = ""
    control_mode: str = ""
    image_keys: list[str] = field(default_factory=list)
    normalize_language: bool = True
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    normalize_gripper: bool = False
    num_state_tokens: int = 256
    # Leave unset for the default MolmoAct2 sequence budget inferred from the fixed
    # image/prompt/state/action token layout. Override only for unusual long prompts.
    max_sequence_length: int | None = None

    # Fixed by released MolmoAct2 checkpoints. We validate this at model load.
    expected_max_action_dim: int = 32

    # Flow-matching training knobs copied from the original MolmoAct2 training path.
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_inference_steps: int | None = None
    mask_action_dim_padding: bool = True
    enable_inference_cuda_graph: bool = True
    # MolmoAct2-local eval option. When enabled, stochastic continuous action
    # generation uses a rollout-local generator derived from eval_seed.
    per_episode_seed: bool = False
    eval_seed: int | None = None
    rtc_config: RTCConfig | None = None

    # Joint frame transform for cross-calibration compatibility.
    # Some MolmoAct2 checkpoints were trained on data using a different joint
    # convention than the current LeRobot calibration. Set both to apply a
    # sign/offset correction at runtime (state before model, action after).
    # See: https://huggingface.co/docs/lerobot/backwardcomp
    # Default is None (no transform). Both must be set together.
    joint_signs: list[float] | None = None
    joint_offsets: list[float] | None = None

    # Controls only the VLM side. The action expert is always fully fine-tuned.
    train_mode_vlm: str = "lora"
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    enable_knowledge_insulation: bool = False
    freeze_embedding: bool = True
    gradient_checkpointing: bool = False
    # Pi0.5-style public switch backed by MolmoAct2's official block-wise
    # compilation strategy. The policy intentionally keeps the compiler
    # backend/scope internal so there is only one supported execution plan.
    compile_model: bool = False

    # Pi0.5-style precision switch controlling parameter storage and autocast.
    # ``bfloat16`` stores large text and vision matrices in bf16 while the full
    # action expert, selected norm/head/LoRA parameters, and RoPE state stay
    # fp32; operator compute follows bf16 autocast plus explicit sensitive fp32
    # math.
    # ``float32`` keeps both the full model and compute in fp32.
    dtype: str = "bfloat16"
    # Official fine-tuning from the released ``allenai/MolmoAct2`` HF base
    # explicitly applies unmasked residual dropout 0.1 and disables the
    # response-only variant.  The converted HF decoder therefore matches the
    # official HF-checkpoint path with this ordinary residual-dropout value.
    llm_residual_dropout: float = 0.1
    softmax_auxiliary_loss: bool = True
    softmax_auxiliary_loss_scale: float = 1e-4
    discrete_loss_token_weighting: str = "root_subsegments_root_tokens"

    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_connector_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 200
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    dataset_feature_names: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.joint_signs is None) != (self.joint_offsets is None):
            raise ValueError("joint_signs and joint_offsets must both be set or both be None.")
        if self.joint_signs is not None and len(self.joint_signs) != len(self.joint_offsets):
            raise ValueError("joint_signs and joint_offsets must have the same length.")
        if self.action_mode not in {"continuous", "discrete", "both"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}. "
                "Expected one of {'continuous', 'discrete', 'both'}."
            )
        if self.inference_action_mode not in {None, "continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {None, 'continuous', 'discrete'}."
            )
        if self.inference_action_mode == "continuous" and self.action_mode == "discrete":
            raise ValueError("MolmoAct2 action_mode='discrete' cannot run continuous inference.")
        if self.inference_action_mode == "discrete" and self.action_mode == "continuous":
            raise ValueError("MolmoAct2 action_mode='continuous' cannot run discrete inference.")
        if self.norm_stats_path is not None and not str(self.norm_tag or "").strip():
            raise ValueError("MolmoAct2 norm_stats_path requires norm_tag to select an embodiment.")
        if self.train_mode_vlm not in {"fft", "lora", "freeze"}:
            raise ValueError(
                f"Unsupported train_mode_vlm={self.train_mode_vlm!r}. "
                "Expected one of {'fft', 'lora', 'freeze'}."
            )
        if self.train_mode_vlm == "freeze" and self.action_mode != "continuous":
            raise ValueError("MolmoAct2 train_mode_vlm='freeze' requires action_mode='continuous'.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.expected_max_action_dim != 32:
            raise ValueError("MolmoAct2 released checkpoints use expected_max_action_dim=32.")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError(f"Unsupported dtype={self.dtype!r}. Expected 'float32' or 'bfloat16'.")
        if not 0 <= self.llm_residual_dropout <= 1:
            raise ValueError(f"llm_residual_dropout must be in [0, 1], got {self.llm_residual_dropout}.")
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}.")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}.")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}.")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                f"Unsupported lora_bias={self.lora_bias!r}. Expected one of 'none', 'all', or 'lora_only'."
            )
        if self.discrete_loss_token_weighting not in {
            "none",
            "token",
            "root_tokens",
            "root_subsegments",
            "root_subsegments_root_tokens",
        }:
            raise ValueError(
                f"Unsupported discrete_loss_token_weighting={self.discrete_loss_token_weighting!r}."
            )
        if self.discrete_generation_max_steps is not None and self.discrete_generation_max_steps < 1:
            raise ValueError(
                f"discrete_generation_max_steps must be >= 1 or None, got {self.discrete_generation_max_steps}."
            )
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError(f"max_sequence_length must be >= 1 or None, got {self.max_sequence_length}.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return MolmoAct2AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=0.0,
            group_grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return MolmoAct2CosineWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        self.dataset_feature_names = {}
        for key in (ACTION, OBS_STATE):
            feature = features.get(key) if isinstance(features, dict) else None
            if isinstance(feature, dict) and feature.get("names") is not None:
                self.dataset_feature_names[key] = feature["names"]

    def validate_features(self) -> None:
        """Validate and set up MolmoAct2 input and output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "MolmoAct2 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(0,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.expected_max_action_dim,),
            )
            self.output_features[ACTION] = action_feature
