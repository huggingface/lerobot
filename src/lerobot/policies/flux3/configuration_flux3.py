# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Configuration of the FLUX 3 Action (``flux3``) policy.

Defaults follow the SO-101 PEFT example: history conditioning, command deltas, 32 predicted / 32
executed actions at 30 Hz, and two cameras on a 256x512 canvas. Camera names, absolute action channels
and per-channel loss weights remain embodiment-specific. Fresh heads are sized from ``output_features``;
saved checkpoints carry their own configuration. Enable adapters through the trainer's PEFT settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import FrozenWarmupConstantSchedulerConfig
from lerobot.policies.flux3.f3 import packing
from lerobot.utils.constants import ACTION, OBS_STATE

# Frozen video VAE from FLUX 3 Action and the official Qwen3-VL text encoder.
DEFAULT_VIDEO_VAE_ID = "black-forest-labs/flux-3-action-base:video_vae.safetensors"
DEFAULT_TEXT_ENCODER_ID = "black-forest-labs/flux-3-action-base:text_encoder"

CAMERA_LAYOUTS = packing.CAMERA_LAYOUTS
SAMPLERS = ("cosmos_unipc", "euler")
ATTN_MODES = ("torch", "flash", "cudnn")


@PreTrainedConfig.register_subclass("flux3")
@dataclass
class Flux3Config(PreTrainedConfig):
    """FLUX 3 Action video-action policy: joint video + action denoising on the FLUX 3 trunk.

    Attributes:
        conditioning: ``frame`` for a current observation and causal target video; ``history`` for
            independently encoded snapshots and future video.
        packer: Optional registered token-layout override. ``None`` selects the built-in packer named
            by ``conditioning``. Overrides are responsible for handling the configured inputs.
        condition_on_past_actions: Concatenate processed past commands with state history. Requires
            history conditioning and doubles the conditioning head width.
        action_representation: ``absolute`` commands or consecutive command ``delta`` values. Command
            deltas require the paired history processors; they differ from state-relative actions.
        delta_absolute_dims: Channels left absolute in command-delta mode (e.g. ``[-1]`` for a gripper).
        n_obs_steps: Observation timesteps used at inference. The PEFT default uses the current images
            and measured state only, encoded as one snapshot with no past-action conditioning.
        chunk_size: Actions predicted per call (the model's action chunk; 32 for the reference recipe).
        n_action_steps: Actions executed per ``select_action`` cycle before the next prediction.
        fps: Control / video frame rate the checkpoint was trained at.
        camera_layout: How the cameras are composited onto the canvas: ``"droid"`` (three 360x640 cameras
            [wrist, left, right]: wrist full-res on top, exteriors half-res below, reflect-padded) or
            ``"single"`` (one resized camera), ``"side_by_side"`` (scene then wrist), or ``"grid"`` (any
            number of cameras, each resized into a cell of a near-square grid in ``camera_keys`` order;
            cameras may differ in resolution). The first three are the layouts our checkpoints were trained
            with; ``grid`` runs for any setup but no checkpoint was trained on it.
        camera_keys: Ordered image feature keys feeding the layout. ``None`` = sorted image features
            (for the ``droid`` layout the key containing ``wrist`` is moved to the front).
        canvas_hw: Pixel canvas the video VAE encodes.
        action_modality: Name of the action modality streams inside the DiT (``x_<name>`` /
            ``x_<name>_cond``). Any name works; it becomes part of the checkpoint keys.
        action_scale: Representation scale applied to state and action tokens (self-inverting at decode). A
            property of the pretrained trunk, not of the dataset: keep the checkpoint's value.
        gripper_flip_dims: Action / state dims the model sees as ``1 - x`` (the reference recipe stores the
            gripper as an open fraction). Applied symmetrically on the way in and out. ``[]`` = none.
        use_relative_actions: Train / deploy on state-relative actions (deltas to the current state);
            deployment is unavailable until RTC is implemented. The gripper stays absolute (``relative_exclude_joints``).
        trunk_weights: Action-pretrained trunk to start a finetune from (``.safetensors`` path or Hub
            ``repo_id:filename``). ``None`` = random init (wiring tests / weights loaded afterwards by
            ``from_pretrained``).
        video_vae_id: Frozen video VAE weights (path or Hub id). Never saved with the policy.
        text_encoder_id: Frozen Qwen3-VL text encoder (Hub id or local path).
        dit_config: Overrides for the DiT hyper-parameters (``JointSingleSeqParams`` fields); tiny values
            give a CPU-testable model.
        head_init_seed: Seed of the xavier draw for embodiment heads the trunk does not carry (their final
            layers start at zero, as in the reference). Fixed, so every rank builds identical heads.
        compile_model: ``torch.compile`` the frozen video VAE, the text encoder and the DiT forward used by
            inference (compiled lazily on the first prediction; one warmup per text-length bucket, none
            with ``text_fixed_length``). Training keeps the eager DiT forward: compilation under
            caption-length batching and gradient checkpointing is not measured yet.
    """

    # Conditioning and action representation are independent of the robot name.
    # Independent snapshot/future encoding, including a single current snapshot.
    conditioning: str = "history"
    packer: str | None = None
    condition_on_past_actions: bool = False
    action_representation: str = "delta"  # absolute commands or consecutive command deltas
    delta_absolute_dims: list[int] = field(default_factory=list)
    text_fixed_length: int | None = 320
    video_position_fps: float | None = 24.0
    history_snapshots: int = 1
    # Optional bootstrap statistics for exporting processors; never read by the model.
    normalization_stats: dict[str, dict[str, list[float]]] | None = None
    normalization_clip: float = 6.0
    separate_timesteps: bool = True
    video_logit_mean: float = 1.08
    video_logit_std: float = 1.0
    conditioning_noise_max: float = 0.2
    loss_reduction: str = "modalities"
    action_channel_weights: list[float] | None = None
    gradient_checkpointing: bool = True
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32
    fps: float = 30.0

    # --- observation / action contract ---
    camera_layout: str = "side_by_side"
    camera_keys: list[str] | None = None
    canvas_hw: tuple[int, int] = (256, 512)
    action_modality: str = "action"
    # The pretrained action stream uses twice the normalized values against unit-variance noise.
    # Packing applies this scale and inference divides it out; retain the checkpoint's value.
    action_scale: float = 2.0
    gripper_flip_dims: list[int] = field(default_factory=list)
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # --- model components ---
    trunk_weights: str | None = None
    video_vae_id: str | None = DEFAULT_VIDEO_VAE_ID
    text_encoder_id: str = DEFAULT_TEXT_ENCODER_ID
    dit_config: dict[str, Any] | None = None
    attn_mode: str = "torch"
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"
    compile_model: bool = False

    # --- sampling ---
    sampler: str = "cosmos_unipc"
    num_inference_steps: int = 4
    guidance_scale: float = 4.0
    guidance_scale_action: float | None = 1.0
    sampler_shift: float = 5.0
    inference_seed: int = 0
    head_init_seed: int = 0

    # --- PEFT training defaults ---
    train_timestep_width: float = 0.75  # logit-logistic scale
    train_timestep_shift: float = 42.0  # rational time shift of the training distribution
    action_loss_weight: float = 0.5
    video_loss_weight: float = 1.0
    caption_dropout: float = 0.10  # empty caption through the same chat template
    augment: bool = False
    optimizer_lr: float = 1e-4  # LoRA adapters; fresh heads use 5x
    optimizer_lr_heads_multiplier: float = 5.0
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_freeze_backbone_steps: int = 0
    scheduler_warmup_steps: int = 0
    scheduler_warmup_steps_heads: int = 0
    scheduler_decay_steps: int = 0
    scheduler_cooldown_steps: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.text_encoder_id or not self.text_encoder_id.strip():
            raise ValueError("text_encoder_id must be a non-empty Hub ID or local path")
        if len(self.canvas_hw) != 2:
            raise ValueError(f"canvas_hw must be (height, width), got {self.canvas_hw!r}")
        self.canvas_hw = (int(self.canvas_hw[0]), int(self.canvas_hw[1]))
        if len(self.optimizer_betas) != 2:
            raise ValueError(f"optimizer_betas must be (beta1, beta2), got {self.optimizer_betas!r}")
        self.optimizer_betas = (float(self.optimizer_betas[0]), float(self.optimizer_betas[1]))
        if self.camera_layout not in CAMERA_LAYOUTS:
            raise ValueError(f"camera_layout must be one of {CAMERA_LAYOUTS}, got {self.camera_layout!r}")
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {self.sampler!r}")
        if self.attn_mode not in ATTN_MODES:
            raise ValueError(f"attn_mode must be one of {ATTN_MODES}, got {self.attn_mode!r}")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.fps, self.fps if self.video_position_fps is None else self.video_position_fps)
        ):
            raise ValueError("frame rates must be positive and finite")
        if self.text_fixed_length is not None and self.text_fixed_length < 1:
            raise ValueError("text_fixed_length must be positive")
        if self.conditioning not in ("frame", "history"):
            raise ValueError("conditioning must be frame or history")
        packing.build_packer(self)  # Fail unknown layouts before building the model.
        if self.action_representation not in ("absolute", "delta"):
            raise ValueError("action_representation must be absolute or delta")
        if self.conditioning != "history" and (
            self.condition_on_past_actions or self.action_representation == "delta"
        ):
            raise ValueError(
                "Past-action conditioning and command deltas currently require history conditioning"
            )
        if self.action_representation != "delta" and self.delta_absolute_dims:
            raise ValueError("delta_absolute_dims requires action_representation=delta")
        if self.loss_reduction not in ("joint_tokens", "modalities"):
            raise ValueError("loss_reduction must be joint_tokens or modalities")
        if not 0 <= self.conditioning_noise_max <= 1:
            raise ValueError("conditioning_noise_max must be in [0, 1]")
        if self.train_timestep_width <= 0 or self.train_timestep_shift <= 0:
            raise ValueError("training timestep width and shift must be positive")
        if self.video_logit_std <= 0 or self.normalization_clip <= 0:
            raise ValueError("video_logit_std and normalization_clip must be positive")
        if self.conditioning == "history":
            if not 1 <= self.history_snapshots <= self.n_obs_steps:
                raise ValueError("History conditioning requires 1 <= history_snapshots <= n_obs_steps")
            if self.chunk_size < 17:
                raise ValueError("Independent future-video encoding needs at least 17 frames")
            if self.use_relative_actions or self.gripper_flip_dims:
                raise ValueError(
                    "History processors own the action representation; do not combine with state-relative actions or gripper flips"
                )
            if any(mode != NormalizationMode.IDENTITY for mode in self.normalization_mapping.values()):
                raise ValueError(
                    "History processors own range normalization; processor mapping must be IDENTITY"
                )
        if self.conditioning == "frame" and self.n_obs_steps != 1:
            raise ValueError("flux3 conditions on a single observation frame: n_obs_steps must be 1")
        if self.chunk_size < 1 or not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"need 1 <= n_action_steps ({self.n_action_steps}) <= chunk_size ({self.chunk_size}) and "
                "chunk_size >= 1"
            )
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        if not 0.0 <= self.caption_dropout <= 1.0:
            raise ValueError("caption_dropout must be in [0, 1]")
        if self.action_scale <= 0:
            raise ValueError("action_scale must be > 0")
        if any(c in self.action_modality for c in ". "):
            raise ValueError(
                "action_modality must be a plain identifier (it becomes part of checkpoint keys)"
            )
        if self.input_features and self.output_features:
            self.validate_features()

    # ---- derived ----
    @property
    def action_dim(self) -> int:
        if self.action_feature is None:
            raise ValueError(f"flux3 needs the action feature {ACTION!r}")
        return int(self.action_feature.shape[0])

    @property
    def window_frames(self) -> int:
        """Frame: one observed frame + targets. History: observations + independently encoded future."""
        return self.chunk_size + (self.n_obs_steps if self.conditioning == "history" else 1)

    @property
    def content_hw(self) -> tuple[int, int]:
        """Canvas area carrying image content, excluding the DROID layout's reflect padding."""
        if self.camera_layout == "droid":
            h, w = self.image_features[self.camera_order[0]].shape[-2:]
            return h + h // 2, w
        return self.canvas_hw

    @property
    def latent_hw(self) -> tuple[int, int]:
        return packing.latent_hw(self.content_hw)

    @property
    def camera_order(self) -> list[str]:
        if self.camera_keys:
            return list(self.camera_keys)
        keys = sorted(self.image_features)
        if self.camera_layout == "droid":
            keys.sort(key=lambda k: 0 if "wrist" in k else 1)  # stable: wrist first, exteriors keep order
        return keys

    @property
    def drop_n_first_frames(self) -> int:
        return self.n_obs_steps if self.conditioning == "history" else 0

    @property
    def drop_n_last_frames(self) -> int:
        # Both layouts require images through offset +chunk_size.
        return self.chunk_size

    # ---- lerobot contract ----
    def validate_features(self) -> None:
        if self.input_features is None:
            raise ValueError("`input_features` must be resolved before `validate_features()` is called.")
        if not self.image_features:
            raise ValueError(
                "flux3 needs at least one camera feature, named observation.images.<camera> (the singular "
                "observation.image is accepted as well)"
            )
        n_expected = {"droid": 3, "single": 1, "side_by_side": 2}.get(self.camera_layout)  # grid: any
        cams = self.camera_order
        if n_expected is not None and len(cams) != n_expected:
            raise ValueError(
                f"camera_layout {self.camera_layout!r} needs {n_expected} camera(s), got {len(cams)}: {cams}"
            )
        missing = [k for k in cams if k not in self.input_features]
        if missing:
            raise ValueError(f"camera_keys not among the input features: {missing}")
        if self.camera_layout == "droid":
            for k in cams:
                hw = tuple(self.input_features[k].shape[-2:])
                if hw != (360, 640):  # Fixed geometry of the pretrained DROID layout.
                    raise ValueError(f"droid layout needs (360, 640) cameras, {k} is {hw}")
        if self.robot_state_feature is None:
            raise ValueError(f"flux3 needs the proprioceptive state feature {OBS_STATE!r}")
        if self.action_feature is None:
            raise ValueError(f"flux3 needs the action feature {ACTION!r}")
        state_dim = int(self.robot_state_feature.shape[0])
        if state_dim != self.action_dim:
            raise ValueError(
                f"the state token shares the action channels: state dim {state_dim} != action dim "
                f"{self.action_dim}. Map your state to the action space in the dataset / processor."
            )
        if self.action_channel_weights is not None and (
            len(self.action_channel_weights) != self.action_dim
            or any(not math.isfinite(w) or w <= 0 for w in self.action_channel_weights)
        ):
            raise ValueError("action_channel_weights must contain one positive finite weight per action")
        if self.normalization_stats is not None:
            for stream in ("action", "state"):
                stats = self.normalization_stats.get(stream, {})
                for q in ("q01", "q99"):
                    values = stats.get(q, [])
                    if len(values) != self.action_dim or any(not math.isfinite(v) for v in values):
                        raise ValueError(
                            f"normalization_stats.{stream}.{q} needs {self.action_dim} finite values"
                        )
                if any(hi < lo for lo, hi in zip(stats["q01"], stats["q99"], strict=True)):
                    raise ValueError(f"normalization_stats.{stream} has inverted quantiles")
        if any(not -self.action_dim <= d < self.action_dim for d in self.delta_absolute_dims):
            raise ValueError("delta_absolute_dims entries must be within the action dimension")
        if len({d % self.action_dim for d in self.delta_absolute_dims}) != len(self.delta_absolute_dims):
            raise ValueError("delta_absolute_dims must identify distinct channels")
        for d in self.gripper_flip_dims:
            if not -self.action_dim <= d < self.action_dim:
                raise ValueError(f"gripper_flip_dims entry {d} outside the action dim {self.action_dim}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> FrozenWarmupConstantSchedulerConfig:
        return FrozenWarmupConstantSchedulerConfig(
            freeze_steps=self.scheduler_freeze_backbone_steps,
            num_warmup_steps=self.scheduler_warmup_steps,
            warmup_steps_heads=self.scheduler_warmup_steps_heads,
            decay_steps=self.scheduler_decay_steps,
            cooldown_steps=self.scheduler_cooldown_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """Training window for image observations: the conditioning frame plus ``chunk_size`` future frames."""
        return self.image_observation_delta_indices

    @property
    def image_observation_delta_indices(self) -> list[int]:
        return (
            list(range(1 - self.n_obs_steps, self.chunk_size + 1))
            if self.conditioning == "history"
            else list(range(self.window_frames))
        )

    @property
    def state_observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1)) if self.conditioning == "history" else [0]

    @property
    def action_delta_indices(self) -> list[int]:
        # Extra preceding command establishes the first history delta.
        return (
            list(range(-self.n_obs_steps, self.chunk_size))
            if self.conditioning == "history"
            else list(range(self.chunk_size))
        )

    @property
    def reward_delta_indices(self) -> None:
        return None
