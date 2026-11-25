# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from .sana_robot_core.multiview import SPATIAL_LAYOUT_2X2, SPATIAL_LAYOUT_T_SHAPE
from .sana_robot_core.robot80 import ACTION_MODE_EEF, ACTION_MODE_JOINT, ROBOT_DIM
from .sana_robot_core.video import RESIZE_CROP, RESIZE_STRETCH, VAE_SPATIAL_STRIDE


@PreTrainedConfig.register_subclass("sana_robot")
@dataclass
class SanaRobotConfig(PreTrainedConfig):
    """Configuration of SANA-Robot-AR.

    The defaults are those of the LIBERO checkpoint. A checkpoint's config stores what its training dataset
    fixed: the cameras, the robot slots, the prompt fields and the control frequency.

    Args:
        camera_keys (`list[str]`, *optional*):
            The observation keys of the cameras, in the order the checkpoint was trained with.
        view_sizes (`list[list[int]]`, *optional*):
            The pixel (height, width) the policy reads every camera at, multiples of 32.
        view_slots (`list[int]`, *optional*):
            The slot of every camera in the spatial layout: 0 for the head or external camera, 1 for the left
            wrist camera and 2 for the right wrist camera.
        view_descriptions (`list[str]`, *optional*):
            The description of every camera in its prompt.
        resize (`str`, *optional*, defaults to `"crop"`):
            How a frame is brought to its view size: `"stretch"`, or `"crop"`, which scales the frame until it
            covers the size and cuts it at the centre.
        embodiment (`str`, *optional*, defaults to `"single-arm Franka Panda robot with a parallel gripper"`):
            The robot's description in the prompts.
        action_mode (`str`, *optional*, defaults to `"robot_base_eef"`):
            `"robot_base_eef"` for end-effector targets in the robot base frame, `"joint_only"` for joint
            targets.
        state_slots (`list[int]`, *optional*):
            The slots of the 80-slot robot row that `observation.state` fills, in order.
        action_slots (`list[int]`, *optional*):
            The slots of the 80-slot robot row that `action` fills, in order.
        fps (`float`, *optional*, defaults to `20.0`):
            The control frequency the checkpoint was trained at.
        chunk_size (`int`, *optional*, defaults to `32`):
            Control steps per action chunk.
        n_action_steps (`int`, *optional*, defaults to `16`):
            Control steps between two generated chunks: a multiple of `video_frame_stride`, at most `chunk_size`.
        video_frame_stride (`int`, *optional*, defaults to `4`):
            Control steps between two video frames the policy reads.
        num_inference_steps (`int`, *optional*, defaults to `2`):
            Denoising steps per chunk.
        video_flow_shift (`float`, *optional*, defaults to `5.0`):
            Flow shift of the video latents.
        action_flow_shift (`float`, *optional*, defaults to `1.0`):
            Flow shift of the actions.
        video_cfg_scale (`float`, *optional*, defaults to `1.0`):
            Classifier-free guidance scale of the video velocity, against the prompt without the instruction;
            `1.0` turns it off. The actions are not guided.
        hidden_size (`int`, *optional*, defaults to `2560`):
            Token width.
        depth (`int`, *optional*, defaults to `32`):
            Number of layers.
        cross_attention_heads (`int`, *optional*, defaults to `20`):
            Heads of the text cross-attention.
        in_channels (`int`, *optional*, defaults to `128`):
            Channels of the video latents.
        caption_channels (`int`, *optional*, defaults to `2304`):
            Width of the text encoder's embeddings.
        linear_head_dim (`int`, *optional*, defaults to `128`):
            Head width of the Gated DeltaNet layers.
        softmax_head_dim (`int`, *optional*, defaults to `256`):
            Head width of the softmax attention layers.
        softmax_ratio (`float`, *optional*, defaults to `0.25`):
            Share of the layers that use softmax attention.
        attn_res_block_size (`int`, *optional*, defaults to `8`):
            Layers per AttnRes block.
        mlp_ratio (`float`, *optional*, defaults to `4.0`):
            Hidden width of the MLP relative to `hidden_size`.
        vae_temporal_stride (`int`, *optional*, defaults to `8`):
            Video frames per latent frame of the video VAE.
        spatial_layout (`str`, *optional*, defaults to `"2x2"`):
            Where the views sit on the spatial rotary positions: `"2x2"` or `"t_shape"`.
        sliding_window_chunks (`int`, *optional*, defaults to `1`):
            How many past chunks' clean latents the softmax layers read besides the first frame.
        vae_path (`str`, *optional*, defaults to `"Efficient-Large-Model/LTX-2.3-Diffusers"`):
            Hub repository or folder of the LTX-2 video VAE in the diffusers layout, with a `vae` subfolder.
        vae_revision (`str`, *optional*, defaults to `"362acdf779d42e785fb26910c32254e9458e78c2"`):
            The Hub commit the VAE is downloaded from; change it together with `vae_path`. A folder ignores it.
        text_encoder_path (`str`, *optional*, defaults to `"google/gemma-2-2b-it"`):
            Hub repository or folder of Gemma 2 2B.
        text_encoder_revision (`str`, *optional*, defaults to `"299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"`):
            The Hub commit the text encoder is downloaded from; change it together with `text_encoder_path`. A
            folder ignores it.
        dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            The precision of the policy's weights. Train with `torch.float32` weights and
            `--accelerator.mixed_precision=bf16`. The video VAE and the text encoder run in BF16.
        max_episode_steps (`int`, *optional*, defaults to `512`):
            Training reads this many control steps before and after the sampled frame, a multiple of `chunk_size`.
            An episode of at most this length is trained as a whole; a longer one is trained on the part inside
            this reach, which then starts after the first frame of the episode.
        keep_checkpoint_statistics (`bool`, *optional*, defaults to `False`):
            Fine-tune a checkpoint with the normalization it holds instead of the one computed from the new
            dataset, e.g. to train on more data of the robot the checkpoint was trained on.
        action_loss_weight (`float`, *optional*, defaults to `1.0`):
            The weight of the action flow loss next to the video flow loss.
        timestep_weighting (`bool`, *optional*, defaults to `True`):
            Weigh the loss of a chunk by its noise level, more in the middle of the range.
        instruction_dropout (`float`, *optional*, defaults to `0.1`):
            The probability that a training window is given the prompt without its instruction.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            Recompute every layer in the backward pass instead of keeping its activations.
        optimizer_lr (`float`, *optional*, defaults to `5e-5`):
            Peak learning rate of AdamW.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`):
            AdamW moment decays.
        optimizer_eps (`float`, *optional*, defaults to `1e-10`):
            AdamW denominator term.
        optimizer_weight_decay (`float`, *optional*, defaults to `0.01`):
            AdamW weight decay.
        optimizer_decay_biases_and_norms (`bool`, *optional*, defaults to `True`):
            Decay the biases and the LayerNorm weights too. The RMSNorm gains are always decayed.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to `1.0`):
            Maximum norm of the gradient.
        scheduler_warmup_steps (`int`, *optional*, defaults to `1000`):
            Steps of the linear warmup.
        scheduler_decay_steps (`int`, *optional*, defaults to `20000`):
            The step at which the cosine reaches `scheduler_decay_lr`; scaled down for a shorter run.
        scheduler_decay_lr (`float`, *optional*, defaults to `5e-7`):
            The learning rate at the end of the cosine.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            The state and the actions use `MEAN_STD` with the checkpoint's center and scale of every slot.
    """

    camera_keys: list[str] = field(
        default_factory=lambda: ["observation.images.image", "observation.images.image2"]
    )
    view_sizes: list[list[int]] = field(default_factory=lambda: [[256, 256], [256, 256]])
    view_slots: list[int] = field(default_factory=lambda: [0, 2])
    view_descriptions: list[str] = field(
        default_factory=lambda: ["front external camera", "wrist-mounted camera on the single controlled arm"]
    )
    resize: str = RESIZE_CROP
    embodiment: str = "single-arm Franka Panda robot with a parallel gripper"
    action_mode: str = ACTION_MODE_EEF
    state_slots: list[int] = field(default_factory=lambda: list(range(29, 46)))
    action_slots: list[int] = field(default_factory=lambda: list(range(36, 46)))
    fps: float = 20.0

    chunk_size: int = 32
    n_action_steps: int = 16
    video_frame_stride: int = 4

    num_inference_steps: int = 2
    video_flow_shift: float = 5.0
    action_flow_shift: float = 1.0
    video_cfg_scale: float = 1.0

    hidden_size: int = 2560
    depth: int = 32
    cross_attention_heads: int = 20
    in_channels: int = 128
    caption_channels: int = 2304
    linear_head_dim: int = 128
    softmax_head_dim: int = 256
    softmax_ratio: float = 0.25
    attn_res_block_size: int = 8
    mlp_ratio: float = 4.0
    vae_temporal_stride: int = 8
    spatial_layout: str = SPATIAL_LAYOUT_2X2
    sliding_window_chunks: int = 1

    vae_path: str | None = "Efficient-Large-Model/LTX-2.3-Diffusers"
    vae_revision: str | None = "362acdf779d42e785fb26910c32254e9458e78c2"
    text_encoder_path: str = "google/gemma-2-2b-it"
    text_encoder_revision: str | None = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"

    dtype: torch.dtype | None = torch.bfloat16

    max_episode_steps: int = 512
    keep_checkpoint_statistics: bool = False
    action_loss_weight: float = 1.0
    timestep_weighting: bool = True
    instruction_dropout: float = 0.1
    gradient_checkpointing: bool = True

    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-10
    optimizer_weight_decay: float = 0.01
    optimizer_decay_biases_and_norms: bool = True
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 20000
    scheduler_decay_lr: float = 5e-7
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        views = len(self.camera_keys)
        if not views or len(set(self.camera_keys)) != views:
            raise ValueError("camera_keys must be nonempty and unique, in checkpoint order")
        if not (len(self.view_sizes) == len(self.view_slots) == len(self.view_descriptions) == views):
            raise ValueError("view_sizes, view_slots and view_descriptions need one entry per camera")
        if any(
            len(size) != 2 or min(size) <= 0 or any(side % VAE_SPATIAL_STRIDE for side in size)
            for size in self.view_sizes
        ):
            raise ValueError(
                f"view_sizes must be (height, width) pairs of positive multiples of {VAE_SPATIAL_STRIDE}"
            )
        if self.resize not in (RESIZE_CROP, RESIZE_STRETCH):
            raise ValueError(f"resize must be {RESIZE_CROP!r} or {RESIZE_STRETCH!r}")
        if self.action_mode not in (ACTION_MODE_EEF, ACTION_MODE_JOINT):
            raise ValueError(f"action_mode must be {ACTION_MODE_EEF!r} or {ACTION_MODE_JOINT!r}")
        if self.spatial_layout not in (SPATIAL_LAYOUT_2X2, SPATIAL_LAYOUT_T_SHAPE):
            raise ValueError(f"spatial_layout must be {SPATIAL_LAYOUT_2X2!r} or {SPATIAL_LAYOUT_T_SHAPE!r}")
        for name, slots in (("state_slots", self.state_slots), ("action_slots", self.action_slots)):
            if not slots or len(set(slots)) != len(slots) or min(slots) < 0 or max(slots) >= ROBOT_DIM:
                raise ValueError(f"{name} must be nonempty, unique slots of the {ROBOT_DIM}-slot robot row")
        frames_per_latent = self.vae_temporal_stride * self.video_frame_stride
        if self.chunk_size <= 0 or self.chunk_size % frames_per_latent:
            raise ValueError(f"chunk_size must be a positive multiple of {frames_per_latent} control steps")
        if not 0 < self.n_action_steps <= self.chunk_size or self.n_action_steps % self.video_frame_stride:
            raise ValueError("n_action_steps must be a multiple of video_frame_stride, at most chunk_size")
        if self.num_inference_steps <= 0 or min(self.video_flow_shift, self.action_flow_shift) <= 0:
            raise ValueError("num_inference_steps and the flow shifts must be positive")
        if self.max_episode_steps <= 0 or self.max_episode_steps % self.chunk_size:
            raise ValueError(
                f"max_episode_steps must be a positive multiple of chunk_size ({self.chunk_size})"
            )
        if not 0.0 <= self.instruction_dropout <= 1.0:
            raise ValueError("instruction_dropout must be in [0, 1]")
        if not self.input_features:
            self.input_features = {
                **{
                    key: PolicyFeature(FeatureType.VISUAL, (3, *size))
                    for key, size in zip(self.camera_keys, self.view_sizes, strict=True)
                },
                OBS_STATE: PolicyFeature(FeatureType.STATE, (len(self.state_slots),)),
            }
        if not self.output_features:
            self.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (len(self.action_slots),))}
        self.validate_features()

    @property
    def image_observation_delta_indices(self) -> list[int]:
        """The frames of a training sample: every `video_frame_stride`-th control step within
        `max_episode_steps` of the sampled frame. `SanaRobotPolicy.forward` keeps those of the episode's window."""
        return list(range(-self.max_episode_steps, self.max_episode_steps + 1, self.video_frame_stride))

    @property
    def state_observation_delta_indices(self) -> list[int]:
        """The states of a training sample: the control steps that can start a chunk, one every `chunk_size`."""
        return list(range(-self.max_episode_steps, self.max_episode_steps, self.chunk_size))

    @property
    def observation_delta_indices(self) -> list[int]:
        return self.state_observation_delta_indices

    @property
    def action_delta_indices(self) -> list[int]:
        """The actions of a training sample: every control step within `max_episode_steps` of the sampled frame."""
        return list(range(-self.max_episode_steps, self.max_episode_steps))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if set(self.image_features) != set(self.camera_keys):
            raise ValueError("input_features must contain exactly camera_keys")
        if any(len(feature.shape) != 3 or feature.shape[0] != 3 for feature in self.image_features.values()):
            raise ValueError("SANA-Robot expects RGB cameras [3, H, W]")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (len(self.state_slots),):
            raise ValueError("observation.state does not match state_slots")
        if self.action_feature is None or self.action_feature.shape != (len(self.action_slots),):
            raise ValueError("action does not match action_slots")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
