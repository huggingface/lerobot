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
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    CosineDecayWithWarmupSchedulerConfig,
    LingbotAdamWConfig,
    LingbotMuonConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def resolve_robot_config_and_stats(config: "LingbotVLAV2Config") -> None:
    """Fill ``config.robot_config`` / ``config.norm_stats`` from their path fields.

    Explicit paths win over the embedded contents, so fine-tuning a checkpoint on a new
    embodiment actually picks up the new assets (previously the embedded stats from the
    source checkpoint silently shadowed the CLI-provided paths). Embedded contents are
    the fallback when the paths are missing — e.g. the checkpoint was moved to another
    machine — which keeps saved checkpoints self-contained.
    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    if config.robot_config_path and os.path.exists(config.robot_config_path):
        import yaml

        with open(config.robot_config_path) as f:
            robot_config = yaml.safe_load(f)
        if config.robot_config is not None and config.robot_config != robot_config:
            logger.warning(
                "config.robot_config_path contents differ from the checkpoint's embedded "
                "robot_config; using the path version."
            )
        config.robot_config = robot_config

    stats_path = config.norm_stats_path
    if stats_path is None and config.robot_config:
        stats_path = config.robot_config.get("norm_stats")
    if stats_path and os.path.exists(stats_path):
        import json

        with open(stats_path) as f:
            norm_stats = json.load(f)
        if config.norm_stats is not None and config.norm_stats != norm_stats:
            logger.warning(
                "config.norm_stats_path contents differ from the checkpoint's embedded "
                "norm_stats; using the path version."
            )
        config.norm_stats = norm_stats


def build_feature_transform_configs(cfg) -> tuple:
    """Build the ``data_config`` / ``model_config`` namespaces for ``FeatureTransform``.

    Single source of truth shared by the processor (apply side) and the policy's
    inference de-normalizer (unapply side). Both sides must agree on the canonical
    layout and the Qwen3-VL token math, otherwise actions get de-normalized under a
    different layout than they were produced with. Accepts either the policy config
    (exposes ``canonical_cameras``) or the processor step (exposes ``cameras``).
    """
    from types import SimpleNamespace

    cameras = getattr(cfg, "canonical_cameras", None) or cfg.cameras
    data_config = SimpleNamespace(
        joints=[f"{{'{k}': {v}}}" for k, v in cfg.canonical_joints.items()],
        norm_type=[f"{{'{k}': '{v}'}}" for k, v in cfg.canonical_norm_type.items()],
        cameras=list(cameras),
        img_size=cfg.resize_imgs_with_padding[0],
        chat_template="default",
        text_keys="task",
    )
    model_config = SimpleNamespace(
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
        chunk_size=cfg.chunk_size,
        tokenizer_max_length=cfg.tokenizer_max_length,
        use_qwen3_chat_template=cfg.use_qwen3_chat_template,
        return_image_grid_thw=cfg.return_image_grid_thw,
        qwen3vl_use_vision_boundaries=cfg.qwen3vl_use_vision_boundaries,
        resize_imgs_with_padding=tuple(cfg.resize_imgs_with_padding),
    )
    return data_config, model_config


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
    tokenizer_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    vlm_family: str = "qwen3_vl"
    tokenizer_max_length: int = 72

    # Image resize target (width, height). Qwen3-VL consumes native-resolution tokens,
    # so this is the pre-patchify resize applied by the image processor.
    resize_imgs_with_padding: tuple[int, int] = (224, 224)
    # Qwen3-VL dynamic-resolution bounds forwarded to AutoProcessor. These cap the
    # vision-token budget and are serialized by the LingBot feature-transform step.
    image_max_pixels: int = 262144
    image_min_pixels: int = 131072

    # Optional device (e.g. "cuda") for the image preprocessing fast path: camera
    # frames are uploaded once and the HF image processor runs batched on-device
    # (resize/rescale/normalize/patchify are all torch ops in the torchvision
    # backend), so the vision tower consumes GPU tensors without a second copy.
    # None keeps the default per-camera CPU path. Inference-only; training
    # (augmentation) and depth-align paths always stay on CPU.
    preprocess_device: str | None = None

    # Number of flow-matching denoising steps at inference.
    num_steps: int = 10

    # ==================== Feature transform (robot-config slot mapping) ====================
    # Per-embodiment robot config (YAML) mapping raw dataset state/action/image keys
    # onto the unified canonical slots, and the matching normalization-stats JSON.
    # Both are resolved by the processor; a robot config is REQUIRED — the processor
    # raises when neither ``robot_config_path`` nor embedded ``robot_config`` is set.
    robot_config_path: str | None = None
    norm_stats_path: str | None = None
    # Parsed contents of the two files above. They are filled in when the processor /
    # policy is built and are serialized into config.json so a saved checkpoint is
    # self-contained and stays valid on machines where the original paths do not exist.
    robot_config: dict | None = None
    norm_stats: dict | None = None
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
    # When ``use_cudagraph_prefix`` is on, also fold the vision tower (ViT) and the
    # embed glue (language embedding, mrope position ids, attention masks) into the
    # captured prefix CUDA graph, instead of running them eagerly and capturing only
    # the 36-layer KV fill. The grid-derived metadata (pos_embeds / cu_seqlens /
    # split_sizes / position_ids) is hoisted into a fixed-grid cache seeded at capture
    # time, so replay copies only the raw inputs (pixels + tokens + masks). Requires a
    # fixed input layout (image grid + token length) per capture; a layout change drops
    # the graph and re-captures. Inference-only (driven from ``sample_actions``).
    use_cudagraph_prefix_full: bool = False

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
    # The released v2 checkpoints balance experts with the auxiliary-LOSS terms
    # below (sequence-wise + router-z); the loss-free bias hook is left disabled
    # (bias_update_speed=0) there. Both mechanisms are wired here so either can
    # be used, matching upstream train_lingbotvla.py.
    #
    # Auxiliary-loss-FREE bias correction (upstream-only). ``bias_update_speed``
    # is parsed for checkpoint-config compatibility, but the optimizer pre-hook
    # that would consume it is not ported — the field has no effect here.
    # Released recipe leaves it at 0.
    bias_update_speed: float = 0.0
    # Center the correction bias each update (subtract per-layer mean) to pin
    # sum(bias)=0 and prevent cumulative drift. Routing-invariant hygiene.
    bias_centering: bool = False
    # Apply the bias update once every N optimizer steps, accumulating
    # tokens_per_expert in between (>1 stabilizes sign(load-mean) for small
    # global batches). Matches upstream ``bias_update_interval``.
    bias_update_interval: int = 1
    # Auxiliary-LOSS balancing (DeepSeek-V3 sequence-wise) — the PRIMARY balancer
    # in the released recipe. Added as a differentiable penalty to the loss.
    sequence_wise_loss_coeff: float = 1e-3
    sequence_wise_mode: str = "per_sequence"
    # Router z-loss on raw router logits (released recipe: 1e-4).
    router_z_loss_coeff: float = 1e-4
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    # The released upstream checkpoint stores experts in the stacked/fused layout.
    # Set to None only for fresh experiments that intentionally use a ModuleList MoE.
    moe_implementation: str | None = "fused"
    # Token-count ceiling for the dense two-GEMM pure-torch MoE path (fused layout
    # only): when B*T <= this, every expert is computed with two plain matmuls and
    # the routing weights are folded into the down GEMM — no argsort/gather/scatter,
    # static shapes, no torch.compile graph breaks. At the flow-matching denoise
    # token count (51) this is much faster than routed dispatch; the 8x FLOP waste
    # is free at that scale. 0 disables (falls back to triton/grouped-eager).
    moe_dense_max_tokens: int = 512
    # MoE execution backend. "sparse_static" is the shipped default: real
    # per-token expert activation via the pure-torch padded bmm path
    # (argsort -> scatter into [E, T, H] -> 2 bmm -> gather -> weighted combine),
    # with the padded capacity pinned to T so shapes stay static (CUDA-graph
    # capturable, no host sync). It beats dense in training (-2.7% step time at
    # B=8) at bit-identical loss and costs +1.5~2.7% model-only inference time.
    # Alternatives: "auto" (dense for small T, grouped-eager otherwise),
    # "dense", "sparse" (dynamic capacity, one .item() sync), "sparse_gmm" /
    # "sparse_static_gmm" (grouped_mm 3D GEMM, torch >= 2.11 on sm89+), "eager".
    moe_backend: str = "sparse_static"

    # ==================== Optional predictive-dynamics distillation branch ====================
    # Only used by the native-depth (6B) checkpoint. Empty ``align_params`` disables it,
    # which keeps the action path identical to the depth-free variant.
    use_depth: bool = False
    num_task_tokens: int = 8
    align_params: dict = field(default_factory=dict)
    enable_expert_vision: bool = False
    expert_vision_type: str | None = None
    # Future-frame spacing for the distillation branch, in dataset frames. Upstream
    # samples the "future" camera frame at ``chunk_size - 1`` frames ahead — the
    # horizon the action chunk predicts — and derives the DINO-video teacher's
    # effective fps as ``fps / max(1, chunk_size - 1)``. None keeps that default;
    # override only to experiment with other spacings.
    future_frame_offset: int | None = None
    # fps of the training dataset, used to synthesize ``future_video_effective_fps``
    # for the DINO-video teacher (upstream injects it per item from the dataset).
    # None leaves the teacher on the effective_fps baked into its config.yaml.
    dataset_fps: int | None = None

    # ==================== Modeling internals (FlowMatching / dual-stream expert) ====================
    # Attention used inside the vendored dual-stream model. "sdpa" (fused flash /
    # memory-efficient kernels, O(L) memory) is the default; "eager" materializes
    # the [B, H, L, L] score matrix and is only kept for debugging; "fa2" needs the
    # flash-attn package; "flex"/"flex_cached" use torch flex-attention BlockMasks.
    attention_implementation: str = "sdpa"
    # Same implementation choices, applied to the vision tower (ViT) attention.
    vit_attn_implementation: str = "sdpa"
    # Upcast attention Q/K/V (and the KV cache) to fp32 — the original upstream
    # parity path. False (default) runs attention in the model dtype (bf16 tensor
    # cores, half the KV-cache memory); outputs match to bf16 reassociation error.
    attention_fp32: bool = False
    # Force a specific SDPA kernel backend (an `SDPBackend` enum name, e.g.
    # "CUDNN_ATTENTION") instead of torch's auto-selection. Only applies when
    # attention_implementation="sdpa". Motivation: with bool masks torch 2.8
    # auto-selects mem_efficient, but the cuDNN backend is ~14% faster end-to-end
    # in training (measured on A100, B=4, bf16) and cuts activation memory by a
    # third. None (default) keeps torch auto-selection.
    sdpa_backend: str | None = None
    # Split the joint dual-stream attention into two calls: (1) prefix
    # self-attention — prefix rows only ever see prefix keys (att-mask cumsum is
    # constant over the prefix), so this equals their rows in the joint call —
    # and (2) suffix rows over the full K/V. When the prefix (VLM) stream has no
    # trainable params (expert-only LoRA), the prefix halves are detached, so
    # autograd never tracks the frozen VLM activations (the joint call tracks
    # them spuriously: gradients flow into prefix rows via the concatenated
    # Q/K/V tensors and die at the frozen params). Mathematically identical
    # outputs; the only numeric difference is kernel reassociation.
    attn_split_prefix_suffix: bool = False
    # Kernel for the prefix half when attn_split_prefix_suffix is on: "flash"
    # (flash_attn_varlen_func over the unpadded valid prefix tokens; needs
    # flash-attn and fp16/bf16) or "sdpa" (the regular attention path).
    attn_split_prefix_backend: str = "flash"
    # Recompute each dual-stream layer in backward instead of storing activations
    # (training only; ~60% slower step for ~half the activation memory — enables
    # 2-4x larger batches on a single 80GB card).
    gradient_checkpointing: bool = False
    # torch.compile the per-step velocity prediction (inductor fusion, CUDA graphs
    # disabled). The denoise loop is launch-overhead bound (51-token suffix through
    # 36 dual-stream layers), so this gives a large latency win on GPU. First call
    # compiles (minutes); shapes must stay fixed across calls.
    compile_predict_velocity: bool = False
    # Inductor mode for compile_predict_velocity: "default" (fast compile) or
    # "max-autotune-no-cudagraphs" (slow first compile, GEMM autotuning; CUDA
    # graphs stay disabled either way).
    compile_predict_velocity_mode: str = "default"
    # Also compile the prefix path (embed_prefix: vision tower + language/state
    # embedding + mrope position ids, then the 36-layer prefix KV fill) with the
    # same inductor mode. Requires compile_predict_velocity=True to take effect
    # (the flag only matters when the denoise loop is compiled). The prefix runs
    # once per action chunk; compiling it removes the per-layer launch gaps that
    # dominate its eager wall time.
    compile_prefix: bool = False
    # Capture the whole denoise loop (the num_steps predict_velocity calls plus
    # the Euler updates) as one CUDA graph and replay it per action chunk: the
    # loop's per-step guard evaluations and Python glue disappear into a single
    # graph replay, which is the dominant host-side cost once the loop is
    # compiled. Numerically lossless — a replay re-executes the identical kernel
    # sequence on copied-in inputs (validated bitwise against the plain loop).
    # CUDA only; works with or without compile_predict_velocity. The first call
    # pays two extra warmup iterations plus capture; if observation shapes
    # change the stale graph is dropped and re-captured (warning once per new
    # shape), and a warm-up/capture failure disables the graph for this
    # instance and falls back to the plain loop with a warning.
    use_cudagraph_denoise: bool = False
    # Capture the prefix pass (the 36-layer KV fill after embed_prefix) as one
    # CUDA graph and replay it per action chunk, removing the prefix forward's
    # per-layer launch gaps. The vision tower and the embed/glue stay eager
    # (their host syncs forbid capture); the graph covers only
    # qwenvl_with_expert.forward(inputs_embeds=[prefix_embs, None],
    # fill_kv_cache=True). Numerically lossless — a replay re-executes the
    # identical kernel sequence (validated bitwise). When on, it supersedes
    # compile_prefix for the prefix path. Capturable attention stacks:
    # eager / sdpa / flex / flex_cached; attn_split_prefix_suffix=True with
    # the flash backend cannot capture (its varlen packing host-syncs) and
    # falls back with a warning. The KV outputs live in the graph's private
    # pool and are aliased directly into the denoise CUDA graph when both
    # graphs are on (skipping the per-chunk KV copy); every prefix-graph
    # state transition (re-capture / drop / disable) invalidates the alias
    # via a generation counter in the denoise graph's shape signature.
    # Capture-failure/warm-up discipline matches use_cudagraph_denoise, plus
    # a re-capture circuit breaker (shape flicker stops re-capturing and
    # falls back to the eager prefix).
    use_cudagraph_prefix: bool = False
    # Compute/log the MoE monitoring metrics (per-layer MaxVio/entropy/dead-expert,
    # plus the per-metric .item() syncs) once every N training steps. 1 = every
    # step (original behavior).
    moe_metrics_interval: int = 50
    use_cache: bool = True
    post_training: bool = True
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
    split_gate_liner: bool = False
    nosplit_gate_liner: bool = False
    separate_time_proj: bool = False
    final_norm_adanorm: bool = False
    norm_qkv: bool = False

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
    # fused AdamW (single-kernel step on GPU). Same math as the default foreach
    # path; measured a few % faster per training step on A100.
    optimizer_fused: bool = False
    # "adamw" (default, lingbot_adamw) or "muon" (lingbot_muon: upstream
    # DistributedMuon on 2D/3D weights + AdamW on 1D/embed/lm_head). Muon is
    # valid under single-process / DDP / FSDP2 and rejected under FSDP1.
    optimizer_type: str = "adamw"
    # Upstream ``muon_momentum`` / ``muon_nesterov`` / ``muon_ns_steps``; the
    # adjust_lr_fn is pinned to the official recipe's "match_rms_adamw".
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 1e-5  # constant lr schedule (decay_lr == peak_lr)

    def __post_init__(self):
        super().__post_init__()

        # The vendored QwenvlWithExpertV2 reads the expert-storage layout from
        # ``_moe_implementation``; expose our public ``moe_implementation`` under that
        # private name so "fused" selects the stacked-parameter experts that the
        # released MoE checkpoints (e.g. the 6B) were saved with.
        if self.moe_implementation is not None and self.moe_implementation not in ("eager", "fused"):
            raise ValueError(f"Invalid moe_implementation: {self.moe_implementation}")
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

        # The optional predictive-dynamics distillation branch (native depth /
        # DINO-video) is driven by the upstream-compatible ``align_params`` dict.
        # Validate its schema here, at config construction, so a malformed
        # ``--policy.align_params='{...}'`` fails before any dataset / model /
        # teacher initialization. Teacher weight *paths* are not checked here —
        # they are only needed at training time (see teachers/depth_teachers.py).
        if self.align_params:
            self._validate_align_params()

        # The expert-vision branch remains unported in this integration (no
        # forward path, no weight loading) and is NOT the DINO-video teacher —
        # keep rejecting it separately from align_params.
        if self.enable_expert_vision:
            raise NotImplementedError(
                "enable_expert_vision is not available in this LeRobot integration: the "
                "expert-vision branch has no forward path and no weight loading here. It is "
                "NOT the DINO-video distillation teacher (that lives under "
                "align_params.video in the upstream codebase — see "
                "docs/source/lingbot_vla_v2_depth_dino_README.md). Keep "
                "enable_expert_vision=false for action-only training."
            )

    _ALIGN_REQUIRED_TOP_KEYS = ("mode", "num_task_tokens", "depth_loss_weight", "llm", "depth")
    _ALIGN_REQUIRED_LLM_KEYS = ("dim_out", "image_token_size", "image_input_size")
    _ALIGN_REQUIRED_DEPTH_KEYS = (
        "model_type",
        "token_size",
        "input_size",
        "num_backbone_tokens",
        "dim_out",
        "num_layers",
        "num_heads",
        "dim_head",
        "ff_mult",
    )
    _ALIGN_REQUIRED_VIDEO_KEYS = (
        "num_backbone_tokens",
        "dim_out",
        "num_layers",
        "num_heads",
        "dim_head",
        "ff_mult",
    )

    def _validate_align_params(self) -> None:
        """Schema validation for the upstream ``align_params`` dict.

        Mirrors the hard requirements the model code enforces (mode/model_type
        exclusivity, required keys, query-divisibility) but raises them all at
        config-construction time with actionable messages, upstream values and
        the full key list in ``docs/source/lingbot_vla_v2_depth_dino_README.md``.
        """
        params = self.align_params

        def _require(mapping, keys, where):
            missing = [key for key in keys if key not in mapping]
            if missing:
                raise ValueError(f"align_params.{where} is missing required keys: {missing}")

        _require(params, self._ALIGN_REQUIRED_TOP_KEYS, "")
        if params["mode"] != "query":
            raise ValueError(f"align_params.mode must be 'query', got {params['mode']!r}.")
        if params["depth"]["model_type"] != "MoRGBD":
            raise ValueError(
                f"align_params.depth.model_type must be 'MoRGBD', got {params['depth']['model_type']!r}."
            )
        _require(params["llm"], self._ALIGN_REQUIRED_LLM_KEYS, "llm")
        _require(params["depth"], self._ALIGN_REQUIRED_DEPTH_KEYS, "depth")

        num_backbone_tokens = params["depth"]["num_backbone_tokens"]
        num_task_tokens = int(params["num_task_tokens"])
        if num_backbone_tokens % num_task_tokens:
            raise ValueError(
                f"align_params.depth.num_backbone_tokens ({num_backbone_tokens}) must be "
                f"divisible by align_params.num_task_tokens ({num_task_tokens})."
            )

        if params.get("use_future_video", False):
            _require(params.get("video", {}), self._ALIGN_REQUIRED_VIDEO_KEYS, "video")
            if not params["depth"].get("use_future_depth", False) and params["video"].get(
                "share_future_depth_query", False
            ):
                raise ValueError(
                    "align_params.video.share_future_depth_query=True requires "
                    "align_params.depth.use_future_depth=True."
                )
            if params["video"].get("use_shared_future_task_proj", False) and not params["video"].get(
                "share_future_depth_query", False
            ):
                raise ValueError(
                    "align_params.video.use_shared_future_task_proj=True requires "
                    "align_params.video.share_future_depth_query=True."
                )

    def num_task_tokens_from(self, params: dict) -> int:
        """num_task_tokens read from align_params (the field the model consumes)."""
        return int(params["num_task_tokens"])

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

    def get_optimizer_preset(self) -> LingbotAdamWConfig | LingbotMuonConfig:
        expert_lr_scale = 1.0
        if self.use_moe and self.use_moe_expert_lr and self.token_top_k > 0:
            expert_lr_scale = (self.token_num_experts / self.token_top_k) ** 0.5
        if self.optimizer_type == "muon":
            return LingbotMuonConfig(
                lr=self.optimizer_lr,
                weight_decay=self.optimizer_weight_decay,
                momentum=self.muon_momentum,
                nesterov=self.muon_nesterov,
                ns_steps=self.muon_ns_steps,
                adamw_betas=self.optimizer_betas,
                adamw_eps=self.optimizer_eps,
                grad_clip_norm=self.optimizer_grad_clip_norm,
                expert_lr_scale=expert_lr_scale,
            )
        if self.optimizer_type != "adamw":
            raise ValueError(
                f"optimizer_type must be 'adamw' or 'muon', got {self.optimizer_type!r}."
            )
        return LingbotAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            expert_lr_scale=expert_lr_scale,
            fused=getattr(self, "optimizer_fused", False),
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def use_depth_align(self) -> bool:
        """The native-depth / DINO-video branch is keyed on a non-empty align_params."""
        return bool(self.align_params)

    @property
    def use_future_image(self) -> bool:
        """Future frames are required by future-depth or DINO-video distillation."""
        if not self.align_params:
            return False
        return bool(
            self.align_params.get("depth", {}).get("use_future_depth", False)
            or self.align_params.get("use_future_video", False)
        )

    @property
    def observation_delta_indices(self) -> list:
        # Future-frame sampling for the distillation branch, mirroring upstream
        # ``base_dataset.get_video_delta_timestamps``: [current, future] per camera
        # with the future frame ``chunk_size - 1`` frames ahead (divided by the
        # dataset fps by ``resolve_delta_timestamps``). State is sliced back to the
        # current frame inside the processor step; inference never samples futures.
        if self.use_depth_align and self.use_future_image:
            offset = self.future_frame_offset if self.future_frame_offset is not None else self.chunk_size - 1
            return [0, max(1, offset)]
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def as_official_recipe(self) -> "LingbotVLAV2Config":
        """Return a copy carrying the official upstream RoboTwin SFT recipe values.

        Kept for callers that start from an existing config object (e.g. a loaded
        checkpoint with legacy port defaults) and want the official training
        values applied in one call. Newly-created configs already default to the
        official recipe.
        """
        import copy

        recipe = copy.copy(self)
        recipe.loss_type = "L1_fm"
        recipe.freeze_vision_encoder = False
        recipe.vlm_causal = True
        recipe.optimizer_lr = 1e-4
        recipe.scheduler_decay_lr = 5e-5
        recipe.scheduler_warmup_steps = 0
        return recipe

    def _save_pretrained(self, save_directory: Path) -> None:
        # Saved checkpoints must be self-contained: clear trainer-local paths once
        # their contents are embedded. Explicit paths still win at load time so a
        # fine-tune can intentionally supply new embodiment assets.
        if self.robot_config is not None:
            self.robot_config_path = None
        if self.norm_stats is not None:
            self.norm_stats_path = None
            if isinstance(self.robot_config, dict):
                self.robot_config = {k: v for k, v in self.robot_config.items() if k != "norm_stats"}
        super()._save_pretrained(save_directory)
