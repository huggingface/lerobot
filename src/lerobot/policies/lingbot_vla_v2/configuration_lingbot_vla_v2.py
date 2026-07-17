# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("lingbot_vla_v2")
@dataclass
class LingbotVLAV2Config(PreTrainedConfig):
    """
    Configuration class for the LingBot-VLA 2.0 policy.

    LingBot-VLA 2.0 is a Qwen3-VL-4B based vision-language-action model that predicts
    action chunks via flow matching. Relative to v1 (``lingbot_vla``) it adds:

    * a **Qwen3-VL** backbone with native-resolution image tokens (``image_grid_thw``),
    * a **sparse Mixture-of-Experts (MoE)** action expert for cross-embodiment scaling,
    * a **unified 55-dim canonical** state/action representation (arms, end-effectors,
      grippers, dexterous hands, waist, head, mobile base, reserved slots), and
    * an optional **predictive-dynamics distillation** branch (depth + DINO-Video).

    The canonical layout mirrors the upstream v2 repo (Robbyant/lingbot-vla-v2). The
    feature -> canonical-slot mapping itself is data driven and handled by the processor
    via a per-embodiment robot config (see ``processor_lingbot_vla_v2``).
    """

    # ==================== Input / Output Structure ====================
    n_obs_steps: int = 1
    chunk_size: int = 50  # action_horizon in LingBot-VLA
    n_action_steps: int = 50

    # Unified cross-embodiment canonical slots (real state/action padded to these dims).
    # v2 canonical vector is 55-D: 14 arm + 14 end-effector + 2 gripper + 12 hand
    # + 4 waist + 2 head + 3 mobility + 4 reserved. Kept configurable so a released
    # checkpoint trained with a different padding width can be loaded exactly.
    max_action_dim: int = 55
    max_state_dim: int = 55

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ==================== Pretrained backbone ====================
    pretrained_name_or_path: str = "robbyant/lingbot-vla-v2-6b"
    tokenizer_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_family: str = "qwen3_vl"
    tokenizer_max_length: int = 72

    # Image resize target (width, height). Qwen3-VL consumes native-resolution tokens,
    # so this is the pre-patchify resize applied by the image processor.
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Number of flow-matching denoising steps at inference.
    num_steps: int = 10

    # ==================== Feature transform (robot-config slot mapping) ====================
    # Per-embodiment robot config (YAML) mapping raw dataset state/action/image keys
    # onto the unified canonical slots, and the matching normalization-stats JSON.
    # Both are resolved by the processor; leave None to fall back to a pass-through
    # single-arm mapping built from the dataset's own features.
    robot_config_path: str | None = None
    norm_stats_path: str | None = None
    # Path (or hub id) to the Qwen3-VL processor (image processor + tokenizer). Falls
    # back to ``tokenizer_path`` when None.
    processor_path: str | None = None
    # Compute dtype for the whole model. The Qwen3-VL backbone defaults to bfloat16
    # while our added heads default to float32; we cast everything to this single dtype
    # after build so the streams stay consistent (mixed dtypes break the custom AdaRMSNorm
    # linears under autocast). lerobot-train also reads this to drive Accelerate autocast.
    dtype: str = "bfloat16"
    # Canonical joint vocabulary (name -> dim) and per-joint normalization mode. These
    # define the unified cross-embodiment layout the checkpoint was trained with and
    # MUST match it. Defaults mirror the v2 55-D canonical vector.
    canonical_joints: dict[str, int] = field(
        default_factory=lambda: {
            "arm.position": 14,
            "end.position": 14,
            "effector.position": 2,
            "hand.position": 12,
            "waist.position": 4,
            "head.position": 2,
            "base.velocity": 3,
            "reserved.slots": 4,
        }
    )
    canonical_norm_type: dict[str, str] = field(
        default_factory=lambda: {
            "arm.position": "meanstd",
            "end.position": "meanstd",
            "effector.position": "meanstd",
            "hand.position": "meanstd",
            "waist.position": "meanstd",
            "head.position": "meanstd",
            "base.velocity": "meanstd",
            "reserved.slots": "meanstd",
        }
    )
    # Canonical camera-view slots the checkpoint expects. The robot config maps raw
    # dataset cameras onto these; missing views are zero-filled at inference.
    canonical_cameras: list[str] = field(
        default_factory=lambda: ["camera_top", "camera_wrist_left", "camera_wrist_right"]
    )

    # Qwen3-VL specific token/vision handling.
    use_qwen3_chat_template: bool = True
    return_image_grid_thw: bool = True
    qwen3vl_use_vision_boundaries: bool = True
    precompute_grid_thw: bool = False
    use_qwen3_fixed_grid_cache: bool = True

    # ==================== Action expert (Qwen2 decoder, MoE-capable) ====================
    expert_hidden_size: int = 768
    expert_intermediate_size: int = 2752
    action_num_attention_heads: int = 32
    action_num_key_value_heads: int = 8
    action_head_dim: int = 128
    action_fp32: bool = False

    # ==================== Sparse MoE action expert ====================
    use_moe: bool = False
    token_moe_layers: list | None = None
    token_num_experts: int = 32
    token_top_k: int = 1
    token_moe_intermediate_size: int = 256
    token_shared_intermediate_size: int = 256
    bias_update_speed: float = 0.001
    sequence_wise_loss_coeff: float = 0.001
    sequence_wise_mode: str = "per_sequence"
    router_z_loss_coeff: float = 0.0
    router_activation: str = "softmax"
    routed_scaling_factor: float = 1.0
    use_shared_expert_gate: bool = True
    # None keeps a dense (non-fused) eager path; "fused" enables grouped-GEMM experts.
    moe_implementation: str | None = None

    # ==================== Optional predictive-dynamics distillation branch ====================
    # Only used by the native-depth (6B) checkpoint. Empty ``align_params`` disables it,
    # which keeps the action path identical to the depth-free variant.
    use_depth: bool = False
    num_task_tokens: int = 8
    align_params: dict = field(default_factory=dict)
    enable_expert_vision: bool = False
    expert_vision_type: str | None = None

    # ==================== Modeling internals (FlowMatching / dual-stream expert) ====================
    # Attention used inside the vendored dual-stream model.
    # "eager" is the safe default and required where flash-attn is unavailable (e.g. Jetson).
    attention_implementation: str = "eager"
    vit_attn_implementation: str = "eager"
    use_cache: bool = True
    post_training: bool = True
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True
    vlm_causal: bool = False
    # 0 keeps the Qwen3-VL vocab as-is (no resize).
    vocab_size: int = 0
    use_lm_head: bool = False
    loss_type: str = "fm"  # flow-matching MSE loss ("L1_fm" for L1)

    # Adaptive layernorm settings for the action expert (LingBot training defaults).
    adanorm_time: bool = True
    split_gate_liner: bool = False
    nosplit_gate_liner: bool = False
    separate_time_proj: bool = False
    final_norm_adanorm: bool = False
    norm_qkv: bool = False

    # ==================== Optimizer / Scheduler Presets ====================
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 1e-5  # constant lr schedule (decay_lr == peak_lr)

    def __post_init__(self):
        super().__post_init__()

        # The vendored QwenvlWithExpertV2 reads the expert-storage layout from
        # ``_moe_implementation``; expose our public ``moe_implementation`` under that
        # private name so "fused" selects the stacked-parameter experts that the
        # released MoE checkpoints (e.g. the 6B) were saved with.
        self._moe_implementation = self.moe_implementation

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.attention_implementation not in ["eager", "sdpa", "fa2", "flex", "flex_cached"]:
            raise ValueError(
                f"attention_implementation must be one of 'eager', 'sdpa', 'fa2', 'flex', "
                f"'flex_cached', got {self.attention_implementation}"
            )

        if self.split_gate_liner and self.nosplit_gate_liner:
            raise ValueError("split_gate_liner and nosplit_gate_liner cannot both be True.")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "LingBot-VLA 2.0 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
        else:
            state_shape = self.input_features[OBS_STATE].shape
            state_dim = state_shape[0] if state_shape else 0
            if state_dim > self.max_state_dim:
                raise ValueError(
                    f"State dimension {state_dim} exceeds max_state_dim {self.max_state_dim}. "
                    f"Either reduce state dimension or increase max_state_dim in config."
                )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
        else:
            action_shape = self.output_features[ACTION].shape
            action_dim = action_shape[0] if action_shape else 0
            if action_dim > self.max_action_dim:
                raise ValueError(
                    f"Action dimension {action_dim} exceeds max_action_dim {self.max_action_dim}. "
                    f"Either reduce action dimension or increase max_action_dim in config."
                )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
