# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration of the Praxis-1 policy adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import PraxisAdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig, LRSchedulerConfig
from lerobot.policies.praxis.praxis_core.robot_io import contiguous_chunk_geometry
from lerobot.utils.constants import ACTION, OBS_STATE

#: The praxis action row: the DROID Cartesian pose and the gripper, in the Franka flange frame.
PRAXIS_ACTION_NAMES: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
PRAXIS_ACTION_DIM = len(PRAXIS_ACTION_NAMES)
# The praxis pose8 of the model's proprio: ``[x, y, z, qw, qx, qy, qz, gripper]`` (the pose processor
# step converts the dataset's 7 values to it).
PRAXIS_POSE_DIM = 8
# The command the policy emits: the praxis wire pose ``[x, y, z, rotation matrix (9, row-major)]`` and
# the gripper (the command processor step converts it back to the 7 values).
PRAXIS_COMMAND_DIM = 13
ATTENTION_BACKENDS: tuple[str, ...] = ("auto", "flash", "triton", "eager")
#: The arms of each rig (``embodiment`` of the checkpoint config). A bimanual rig stacks the two
#: arms (left, then right): 14 state and action values, 16 pose values, 26 command values.
EMBODIMENT_ARMS: dict[str, int] = {"single_arm": 1, "bimanual": 2}
#: The default camera keys of the bimanual rig, in canvas order (the single-arm rig has
#: `exterior_image_key` and `wrist_image_key`).
BIMANUAL_CAMERA_KEYS: tuple[str, ...] = (
    "observation.images.top_left",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


@PreTrainedConfig.register_subclass("praxis")
@dataclass
class PraxisConfig(PreTrainedConfig):
    """Configuration of the Praxis-1 single-arm policy.

    Praxis-1 is a video-action model: a chunk-causal video diffusion backbone (a
    mixture-of-experts transformer) denoises the next 1.6 s of video of the two
    rig cameras while an action expert denoises the matching 24 robot commands
    at 15 Hz. The Wan2.2 video autoencoder and the Qwen3.5-2B text encoder are
    frozen and come from their public weights.

    The architecture, the chunking, the camera canvas and the normalization
    statistics live in ``model_config``: the ``config.json`` of the praxis
    checkpoint, kept verbatim. ``examples/praxis/convert_checkpoint.py`` fills it.

    Observations: ``observation.images.<exterior>`` and ``observation.images.<wrist>``
    (RGB, any size; the processor makes the tile of the checkpoint: 224 x 416 with an
    antialiased resize for the released checkpoint `praxis-1-droid`, 256 x 416 for the earlier DROID checkpoint),
    ``observation.state`` as ``[x, y, z, roll, pitch, yaw, gripper]`` in metres and
    radians (extrinsic x-y-z Euler angles, the DROID convention, Franka flange frame)
    and ``task`` (the instruction). Actions: the same seven values, absolute targets.

    Args:
        model_config: The praxis checkpoint ``config.json`` (architecture, chunking,
            video canvas, text encoder, action space, normalization statistics).
        vae_id: The Wan2.2 VAE weights: a local ``.pth`` path or ``repo_id:filename``
            on the Hub. Frozen, never saved with the policy.
        text_encoder_id: The Qwen3.5-2B text encoder: a Hub repo id or a local
            directory in the Hugging Face layout. Frozen, never saved with the policy.
        exterior_image_key: The observation key of the fixed camera.
        wrist_image_key: The observation key of the wrist camera.
        action_feature_names: The names of the action dimensions, filled by ``make_policy`` from
            the dataset (``None`` when it has none). Praxis reads the DROID pose and gripper,
            ``PRAXIS_ACTION_NAMES``; the names only need to be as many as the action values.
        chunk_size: Commands per model call (24; the first call of a context window
            gives 21).
        n_action_steps: Commands executed per call. Below ``chunk_size`` every call
            is a cold start (no context), as in praxis.
        max_context_chunks: Executed chunks kept as context (0 to 3).
        num_steps: Euler steps of the flow sampler per call.
        t_shift: Shift of the timestep grid (1.0 = uniform).
        cfg_scale: Text guidance scale of the actions (1.0 = off).
        cfg_video_scale: Text guidance scale of the video (1.0 = off).
        seed: Base sampler seed. Chunk ``k`` of an episode with scene seed ``s``
            samples with ``seed + seed_stride * s + k``. The scene seed comes from
            ``observation.scene_seed`` when the environment provides it, else 0.
        seed_stride: See ``seed``.
        attention_backend: ``flash`` (flex attention with the FlashAttention-4 CuTe
            kernel, needs ``flash-attn-4``; not on compute capability 12.x, where it has
            no block-sparse forward), ``triton`` (the Triton flex-attention kernel),
            ``eager`` (uncompiled flex attention, for CPU) or ``auto`` (``flash`` where it
            runs and ``flash-attn-4`` is installed, else ``triton``).
        return_prediction: Also return the model's predicted camera frames as a
            ``PolicyOutput`` prediction (for ``--display_data``). Costs one VAE
            decode per call.
        text_pad_to_multiple_of: Padding of the tokenized prompt. 1 encodes the prompt
            without padding; 32 is the praxis setting (the default).
    """

    model_config: dict[str, Any] = field(default_factory=dict)

    vae_id: str = "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth"
    text_encoder_id: str = "Qwen/Qwen3.5-2B"

    exterior_image_key: str = "observation.images.exterior"
    wrist_image_key: str = "observation.images.wrist"
    action_feature_names: list[str] | None = None
    # The camera keys in the canvas order of the checkpoint. Empty: the two keys above for the
    # single-arm rig, `BIMANUAL_CAMERA_KEYS` (top, left wrist, right wrist) for the bimanual rig.
    camera_keys: list[str] = field(default_factory=list)

    n_obs_steps: int = 1
    chunk_size: int = 24
    n_action_steps: int = 24
    max_context_chunks: int = 1

    # The recommended sampler settings of `praxis-1-droid` (the praxis repository's `configs/inference_droid.yaml`).
    num_steps: int = 2
    t_shift: float = 12.5
    cfg_scale: float = 1.0
    cfg_video_scale: float = 4.0

    seed: int = 0
    seed_stride: int = 1000

    attention_backend: str = "auto"
    return_prediction: bool = False
    # Prompt padding of the text encoder: praxis pads the tokens to a multiple of 32 (the validated
    # setting). 1 encodes the prompt without padding; both differ only in the last bits.
    text_pad_to_multiple_of: int = 32

    dtype: torch.dtype | None = torch.bfloat16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Fine-tuning presets: the recipe of ``praxis/train/README.md`` (full fine-tune).
    optimizer_lr: float = 3e-6
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0

    # The fine-tuning objective: the training objective of the checkpoint (the `objective` block of
    # praxis `configs/finetune_droid.yaml`). `forward` draws the timesteps, the noise, the text and
    # proprio dropout per window; the MoE routing biases move after each optimizer step (`update`).
    video_weight: float = 1.0
    action_weight: float = 1.0
    video_timestep_shift: float = 5.0
    text_dropout: float = 0.1
    proprio_dropout: float = 0.1
    moe_balance_speed: float = 1e-3
    moe_seq_balance_coef: float = 1e-4
    action_guard: float = 5.0
    train_moe_dispatch_chunk_size: int = 65536
    # Recompute every transformer block in the backward pass (the praxis recipe; it fits a full
    # fine-tune on 4 x 80 GB with FSDP).
    activation_checkpointing: bool = True
    # The learning-rate schedule of the recipe: a linear warmup, then a cosine decay from
    # optimizer_lr to scheduler_decay_lr over scheduler_decay_steps.
    scheduler_warmup_steps: int = 50
    scheduler_decay_steps: int = 1200
    scheduler_decay_lr: float = 3e-7

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {ATTENTION_BACKENDS}, got {self.attention_backend!r}"
            )
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.text_pad_to_multiple_of < 1:
            raise ValueError(f"text_pad_to_multiple_of must be >= 1, got {self.text_pad_to_multiple_of}")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError(
                f"n_action_steps must be in [1, chunk_size={self.chunk_size}], got {self.n_action_steps}"
            )
        if self.max_context_chunks < 0:
            raise ValueError(f"max_context_chunks must be >= 0, got {self.max_context_chunks}")
        if self.t_shift <= 0.0:
            raise ValueError(f"t_shift must be > 0, got {self.t_shift}")
        if self.seed < 0 or self.seed_stride < 1:
            # numpy's SeedSequence rejects negative entries; a stride of 0 gives every scene the same seed
            raise ValueError(
                f"seed must be >= 0 and seed_stride >= 1, got {self.seed} and {self.seed_stride}"
            )
        if self.n_obs_steps != 1:
            raise ValueError("Praxis keeps its own frame history; n_obs_steps must be 1")
        if self.exterior_image_key == self.wrist_image_key:
            raise ValueError("exterior_image_key and wrist_image_key must differ")
        for name in ("text_dropout", "proprio_dropout"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        for name in ("video_weight", "action_weight", "moe_balance_speed", "moe_seq_balance_coef"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.action_guard <= 0.0 or self.video_timestep_shift <= 0.0:
            raise ValueError("action_guard and video_timestep_shift must be > 0")
        if self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps < 1 or self.scheduler_decay_lr < 0.0:
            raise ValueError(
                "scheduler_warmup_steps >= 0, scheduler_decay_steps >= 1, scheduler_decay_lr >= 0"
            )
        if self.model_config:
            chunking = self.model_config.get("chunking", {})
            action_chunk = int(chunking.get("action_chunk", self.chunk_size))
            if action_chunk != self.chunk_size:
                raise ValueError(
                    f"chunk_size={self.chunk_size} differs from the checkpoint's action_chunk={action_chunk}"
                )
            n_chunks = int(chunking.get("n_chunks", 4))
            if self.max_context_chunks > n_chunks - 1:
                raise ValueError(f"max_context_chunks must be <= {n_chunks - 1} for this checkpoint")

    @property
    def embodiment(self) -> str:
        """The rig of the checkpoint: ``single_arm`` (DROID) or ``bimanual`` (YAM)."""
        return str((self.model_config or {}).get("embodiment", "single_arm"))

    @property
    def n_arms(self) -> int:
        return EMBODIMENT_ARMS.get(self.embodiment, 1)

    @property
    def state_dim(self) -> int:
        """Values of `observation.state` and `action`: the DROID pose and gripper of each arm."""
        return PRAXIS_ACTION_DIM * self.n_arms

    @property
    def pose_dim(self) -> int:
        """Values of the praxis pose rows: pose8 of each arm."""
        return PRAXIS_POSE_DIM * self.n_arms

    @property
    def command_dim(self) -> int:
        """Values of one command row of `select_action`: the wire pose and gripper of each arm."""
        return PRAXIS_COMMAND_DIM * self.n_arms

    @property
    def image_keys(self) -> list[str]:
        """The camera keys in canvas order."""
        if self.camera_keys:
            return list(self.camera_keys)
        if self.embodiment == "bimanual":
            return list(BIMANUAL_CAMERA_KEYS)
        return [self.exterior_image_key, self.wrist_image_key]

    def validate_features(self) -> None:
        """Require the camera inputs of the rig and its pose-and-gripper state and action."""
        if not self.model_config:
            raise ValueError(
                "PraxisConfig.model_config is empty: load a converted praxis checkpoint "
                "(examples/praxis/convert_checkpoint.py) or pass the praxis config.json."
            )
        if self.embodiment == "base":
            raise ValueError(
                "This is the base checkpoint (no action heads): make a policy of your rig with "
                "examples/praxis/init_heads.py first (docs/source/praxis.mdx, 'Fine-tune the base checkpoint')."
            )
        if self.embodiment not in EMBODIMENT_ARMS:
            raise ValueError(f"Praxis supports the single_arm and bimanual rigs, got {self.embodiment!r}")
        views = ((self.model_config.get("video") or {}).get("views")) or []
        if views and len(views) != len(self.image_keys):
            raise ValueError(
                f"the checkpoint has {len(views)} views {list(views)} but camera_keys names "
                f"{len(self.image_keys)}: {self.image_keys}"
            )
        images = self.image_features
        missing = [k for k in self.image_keys if k not in images]
        if missing:
            raise ValueError(
                f"Praxis needs the camera inputs {missing}; configured image features: {sorted(images)}. "
                "Set camera_keys (or exterior_image_key / wrist_image_key) or use --rename_map."
            )
        if self.action_feature is None:
            raise ValueError("Praxis requires an action output feature.")
        if tuple(self.action_feature.shape) != (self.state_dim,):
            raise ValueError(
                f"Praxis actions are {PRAXIS_ACTION_NAMES} per arm ({self.state_dim} values for the "
                f"{self.embodiment} rig); got shape {tuple(self.action_feature.shape)}"
            )
        state = self.robot_state_feature
        if state is None:
            raise ValueError(f"Praxis requires {OBS_STATE} with {PRAXIS_ACTION_NAMES} per arm")
        if tuple(state.shape) != (self.state_dim,):
            raise ValueError(f"{OBS_STATE} must have shape ({self.state_dim},), got {tuple(state.shape)}")
        names = self.action_feature_names
        if names is not None and len(names) != self.state_dim:
            raise ValueError(
                f"action_feature_names has {len(names)} names for the {self.state_dim} action values "
                f"({PRAXIS_ACTION_NAMES} per arm): {names}"
            )

    def get_optimizer_preset(self) -> PraxisAdamWConfig:
        return PraxisAdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            seed=self.seed,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    def objective_settings(self):  # noqa: ANN201 (the objective module is heavy; imported on use)
        """The `ObjectiveSettings` of the fine-tuning objective, from the fields above."""
        from lerobot.policies.praxis.praxis_core.train.objective import ObjectiveSettings

        return ObjectiveSettings(
            video_weight=self.video_weight,
            action_weight=self.action_weight,
            video_timestep_shift=self.video_timestep_shift,
            text_dropout=self.text_dropout,
            proprio_dropout=self.proprio_dropout,
            moe_balance_speed=self.moe_balance_speed,
            moe_seq_balance_coef=self.moe_seq_balance_coef,
            action_guard=self.action_guard,
            moe_dispatch_chunk_size=self.train_moe_dispatch_chunk_size,
        )

    @property
    def window_frames(self) -> int:
        """Frames of one training window: 93 for the released checkpoint (chunks of 21, 24, 24, 24)."""
        chunking = (self.model_config or {}).get("chunking", {})
        n_chunks = int(chunking.get("n_chunks", 4))
        action_chunk = int(chunking.get("action_chunk", self.chunk_size))
        return sum(contiguous_chunk_geometry(action_chunk, n_chunks)[-1])

    @property
    def drop_n_last_frames(self) -> int:
        """Frames at the end of an episode that start no training window: the window itself.

        The praxis recipe draws the window start uniformly in time over the episode minus the
        window and snaps it down to the frame grid, so the last window of an episode of ``n``
        frames starts at ``n - window_frames - 1`` and ends one frame before the last one; the
        episode-aware sampler draws the same starts with this many frames dropped. Episodes too
        short for the window and the recipe's margin are left out by
        `examples/praxis/select_droid_episodes.py`.
        """
        return self.window_frames

    @property
    def observation_delta_indices(self) -> list[int]:
        """The training window: `window_frames` consecutive frames from the sampled one."""
        return list(range(self.window_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.window_frames))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def action_key(self) -> str:
        return ACTION
