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
from typing import Any

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    AdamWConfig,
    CosineDecayWithWarmupSchedulerConfig,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features


@dataclass
class SlotMapping:
    """Raw dims gathered into one canonical slot.

    ``origin_keys`` is a list of ``{raw_key: {"start": int, "end": int}}`` spans, concatenated in
    order. ``raw_key`` must be ``observation.state`` for state slots and ``action`` for action slots.
    """

    origin_keys: list[dict[str, dict[str, int]]] = field(default_factory=list)


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
      grippers, dexterous hands, waist, head, mobile base, reserved slots).

    The canonical layout mirrors the upstream v2 repo (Robbyant/lingbot-vla-v2). The raw
    state/action dims are mapped onto it by ``state_slots`` / ``action_slots``.
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

    # ==================== Slot mapping ====================
    # Canonical slot (``arm.position`` or ``observation.state.arm.position``) -> raw spans.
    # None maps the raw feature 1:1 onto the canonical layout (base checkpoint, 55-D features).
    state_slots: dict[str, SlotMapping] | None = None
    action_slots: dict[str, SlotMapping] | None = None
    # Relative actions: subtract the current state from the action, except for the
    # dims whose name contains one of ``relative_exclude_joints``.
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated from the dataset at training time.
    action_feature_names: list[str] | None = None

    # ==================== Pretrained backbone ====================
    tokenizer_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    tokenizer_max_length: int = 72

    # Image resize target (width, height). Qwen3-VL consumes native-resolution tokens,
    # so this is the pre-patchify resize applied by the image processor.
    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    # Qwen3-VL dynamic-resolution bounds (cap the vision-token budget).
    image_max_pixels: int = 262144
    image_min_pixels: int = 131072
    # Number of flow-matching denoising steps at inference.
    num_steps: int = 10
    # Compute dtype for the whole model. The Qwen3-VL backbone defaults to bfloat16
    # while our added heads default to float32; we cast everything to this single dtype
    # after build so the streams stay consistent (mixed dtypes break the custom AdaRMSNorm
    # linears under autocast). lerobot-train also reads this to drive Accelerate autocast.
    dtype: torch.dtype | None = torch.bfloat16
    # Canonical joint vocabulary (name -> dim): the unified cross-embodiment layout the
    # checkpoint was trained with, which MUST match it. Defaults mirror the v2 55-D vector.
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

    # Qwen3-VL specific token/vision handling.
    qwen3vl_use_vision_boundaries: bool = True
    precompute_grid_thw: bool = False

    # ==================== Action expert (Qwen2 decoder, MoE-capable) ====================
    expert_hidden_size: int = 768
    expert_intermediate_size: int = 2752
    action_num_attention_heads: int = 32
    action_num_key_value_heads: int = 8
    action_head_dim: int = 128
    action_fp32: bool = False

    # ==================== Sparse MoE action expert ====================
    # Defaults track the released v2 checkpoint recipe (upstream
    # configs/vla/{robotwin,real_robot}.yaml): MoE on every expert layer,
    # 32 experts, top-4 routing.
    use_moe: bool = True
    # Released 6B uses MoE on every Qwen2 expert layer.
    token_moe_layers: list = field(default_factory=lambda: list(range(36)))
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    # ----- MoE load balancing -----
    # Auxiliary-LOSS balancing (DeepSeek-V3 sequence-wise) — the PRIMARY balancer
    # in the released recipe. Added as a differentiable penalty to the loss.
    sequence_wise_loss_coeff: float = 1e-3
    sequence_wise_mode: str = "per_sequence"
    # Router z-loss on raw router logits (released recipe: 1e-4).
    router_z_loss_coeff: float = 1e-4
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    # The released upstream checkpoint stores experts in the stacked/fused layout,
    # the only supported one. Routed experts run the pure-torch padded sparse path
    # (argsort -> scatter into [E, T, H] -> 2 bmm -> gather -> weighted combine)
    # with the capacity pinned to T, so shapes stay static (CUDA-graph capturable).
    moe_implementation: str = "fused"

    # ==================== Modeling internals (FlowMatching / dual-stream expert) ====================
    # Attention used inside the vendored dual-stream model. "sdpa" (fused flash /
    # memory-efficient kernels, O(L) memory) is the default; "eager" materializes
    # the [B, H, L, L] score matrix and is only kept for debugging.
    attention_implementation: str = "sdpa"
    # Same implementation choices, applied to the vision tower (ViT) attention.
    vit_attn_implementation: str = "sdpa"
    # Recompute each dual-stream layer in backward instead of storing activations
    # (training only; ~60% slower step for ~half the activation memory — enables
    # 2-4x larger batches on a single 80GB card).
    gradient_checkpointing: bool = False
    # Inference speed-ups (CUDA only, fixed input shapes, no extra deps). The CUDA graphs
    # replay the identical kernels (bit-exact in bf16); together with compile they take
    # a RoboTwin chunk from ~750 ms to ~120 ms (bf16, RTX PRO 6000).
    # torch.compile the per-step velocity prediction (first call compiles for ~1 min).
    compile_predict_velocity: bool = False
    compile_predict_velocity_mode: str = "default"
    # Capture the denoise loop as one CUDA graph (re-captured if shapes change).
    use_cudagraph_denoise: bool = False
    # Capture the 36-layer prefix KV fill as one CUDA graph.
    use_cudagraph_prefix: bool = False
    # Also capture the vision tower and embedding glue in the prefix graph.
    use_cudagraph_prefix_full: bool = False
    # Real-Time Chunking (RTC) guidance, set by `lerobot-rollout --inference.type=rtc`.
    rtc_config: RTCConfig | None = None
    # Compute/log the MoE monitoring metrics (per-layer MaxVio/entropy/dead-expert,
    # plus the per-metric .item() syncs) once every N training steps. 1 = every
    # step (original behavior).
    moe_metrics_interval: int = 50
    use_cache: bool = True
    # Match the official RoboTwin SFT recipe for newly-created configs. Existing
    # checkpoints retain their serialized values when loaded from --policy.path.
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    vlm_causal: bool = True
    # 0 keeps the Qwen3-VL vocab as-is (no resize).
    vocab_size: int = 0
    use_lm_head: bool = False
    loss_type: str = "L1_fm"

    # Adaptive layernorm settings for the action expert (LingBot training defaults).
    adanorm_time: bool = True
    final_norm_adanorm: bool = False

    # ==================== Optimizer / Scheduler Presets ====================
    # Mirror upstream ``use_moe_expert_lr`` (configs/vla/robotwin/robotwin.yaml):
    # routed experts train at base_lr * (token_num_experts / token_top_k) ** 0.5
    # (= sqrt(32/4) ≈ 2.83) while everything else keeps the base LR. The released
    # upstream recipe always trains with this scaling (and Muon applies the same
    # groups internally — see upstream ``optim/optimizer.py::build_muon_optimizer``).
    use_moe_expert_lr: bool = True
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

        if self.moe_implementation != "fused":
            raise ValueError(f"moe_implementation must be 'fused', got {self.moe_implementation!r}.")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.attention_implementation not in ["eager", "sdpa"]:
            raise ValueError(
                f"attention_implementation must be one of 'eager', 'sdpa', got {self.attention_implementation}"
            )

        for key, slots in ((OBS_STATE, self.state_slots), (ACTION, self.action_slots)):
            for name, mapping in (slots or {}).items():
                joint = name.removeprefix(f"{key}.")
                if joint not in self.canonical_joints:
                    raise ValueError(
                        f"Unknown {key} slot {name!r}; expected one of {list(self.canonical_joints)}."
                    )
                width = 0
                for origin in mapping.origin_keys:
                    for raw_key, span in origin.items():
                        if raw_key != key:
                            raise ValueError(f"{key} slot {name!r} must read from {key!r}, got {raw_key!r}.")
                        width += span["end"] - span["start"]
                if width > self.canonical_joints[joint]:
                    raise ValueError(
                        f"{key} slot {name!r} spans {width} dims, wider than its canonical "
                        f"dimension {self.canonical_joints[joint]}."
                    )

    def slot_spans(self, key: str) -> dict[str, list[list[int]]]:
        """Canonical joint -> ``[[start, end], ...]`` spans on the raw ``key`` feature."""
        slots = self.state_slots if key == OBS_STATE else self.action_slots
        if slots is None:
            spans, offset = {}, 0
            for joint, dim in self.canonical_joints.items():
                spans[joint] = [[offset, offset + dim]]
                offset += dim
            return spans
        return {
            name.removeprefix(f"{key}."): [
                [span["start"], span["end"]] for origin in mapping.origin_keys for span in origin.values()
            ]
            for name, mapping in slots.items()
        }

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        """Adopt the dataset's raw state shape (a fine-tuned base checkpoint would keep its 55-D one)."""
        state = dataset_to_policy_features(features).get(OBS_STATE)
        if state is not None:
            self.input_features = {**(self.input_features or {}), OBS_STATE: state}

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "LingBot-VLA 2.0 policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

        self.input_features.setdefault(
            OBS_STATE, PolicyFeature(type=FeatureType.STATE, shape=(self.max_state_dim,))
        )
        self.output_features.setdefault(
            ACTION, PolicyFeature(type=FeatureType.ACTION, shape=(self.max_action_dim,))
        )
        for key, feature, flag in (
            (OBS_STATE, self.input_features[OBS_STATE], "state_slots"),
            (ACTION, self.output_features[ACTION], "action_slots"),
        ):
            end = max((span[1] for spans in self.slot_spans(key).values() for span in spans), default=0)
            if end > feature.shape[-1]:
                raise ValueError(
                    f"{key} has {feature.shape[-1]} dims but its slot mapping reads up to dim {end}; "
                    f"set --policy.{flag} to map the robot's dims onto the canonical slots."
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
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
