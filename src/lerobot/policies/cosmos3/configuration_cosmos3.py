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

"""Configuration of the Cosmos3 policy.

Cosmos3 is a world action model: a Mixture-of-Transformers that jointly denoises future video latents and
robot actions from robot observations and a language instruction. Upstream implementation:
https://github.com/NVIDIA/cosmos-framework.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import draccus
import torch
from draccus.utils import DecodingError

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import WarmupStableDecaySchedulerConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from huggingface_hub.dataclasses import strict
    from transformers.configuration_utils import PretrainedConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
else:
    PretrainedConfig = object
    Qwen3VLConfig = object
    Qwen3VLTextConfig = object
    Qwen3VLVisionConfig = object
    Siglip2VisionConfig = object

    def strict(cls):
        return cls


# Per-channel mean and std of the Wan2.2 latents (`vae/config.json` of the released checkpoints).
WAN2PT2_LATENTS_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891,
    0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723,
    -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]  # fmt: skip
WAN2PT2_LATENTS_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]  # fmt: skip

TimeDistribution = Literal["uniform", "logitnormal", "waver"]
# Values of `Cosmos3Config.mode` the policy supports.
SUPPORTED_MODES = ("wam",)
# Resolution tiers and the largest shorter side of each (native `get_vision_data_resolution`).
RESOLUTION_TIER_LIMITS = (("256", 256), ("480", 640), ("720", 960))


@strict
class Qwen3VLMoTVisionConfig(Qwen3VLVisionConfig):
    """Qwen3-VL vision config of the Cosmos3 MoT backbone, defaults from ``nvidia/Cosmos3-Nano``."""

    out_hidden_size: int = 4096


@strict
class Qwen3VLMoTTextConfig(Qwen3VLTextConfig):
    """
    Qwen3-VL text config of the Cosmos3 MoT backbone (Cosmos3 Nano and Super). The fields set here are the
    ``text_config`` of ``nvidia/Cosmos3-Nano`` where it differs from Qwen3-VL.
    """

    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rope_parameters: dict | None = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 5_000_000,
            "mrope_section": [24, 20, 20],
            "mrope_interleaved": True,
        }
    )
    # Joint attention over the understanding and generation pathways. Only "two_way" is supported.
    joint_attn_implementation: str = "two_way"

    def validate_architecture(self):
        super().validate_architecture()
        if self.joint_attn_implementation != "two_way":
            raise NotImplementedError(
                f"joint_attn_implementation={self.joint_attn_implementation!r} is not supported, only 'two_way'."
            )


@strict
class Qwen3VLMoTConfig(Qwen3VLConfig):
    """Config of the Qwen3-VL MoT backbone (Cosmos3 Nano and Super), defaults from ``nvidia/Cosmos3-Nano``."""

    sub_configs = {"vision_config": Qwen3VLMoTVisionConfig, "text_config": Qwen3VLMoTTextConfig}


@strict
class Nemotron3DenseVLMoTTextConfig(PretrainedConfig):
    """Text config of the Nemotron 3 Dense VL MoT backbone, defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl_text"
    base_config_key = "text_config"
    default_theta = 100_000_000.0
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "relu2"
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    rope_parameters: dict | None = field(
        default_factory=lambda: {
            "rope_type": "default",
            "rope_theta": 100_000_000,
            "mrope_section": [24, 20, 20],
        }
    )
    # Padding index of the token embedding.
    pad_token_id: int | None = 0
    # Joint attention over the understanding and generation pathways. Only "two_way" is supported.
    joint_attn_implementation: str = "two_way"

    def validate_architecture(self):
        super().validate_architecture()
        if self.joint_attn_implementation != "two_way":
            raise NotImplementedError(
                f"joint_attn_implementation={self.joint_attn_implementation!r} is not supported, only 'two_way'."
            )


@strict
class Nemotron3DenseVLVisionConfig(Siglip2VisionConfig):
    """SigLIP2 vision config of the Nemotron 3 Dense VL backbone, defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl_vision"

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    # Side of the square patch blocks the projector merges into one token.
    spatial_merge_size: int = 2


@strict
class Nemotron3DenseVLProjectorConfig(PretrainedConfig):
    """Patch merger config of the Nemotron 3 Dense VL backbone, defaults from ``nvidia/Cosmos3-Edge``."""

    model_type = "nemotron_3_dense_vl_projector"
    base_config_key = "projector_config"

    input_hidden_size: int = 1152
    merger_intermediate_size: int = 11520
    out_hidden_size: int = 2048
    spatial_merge_size: int = 2
    use_postshuffle_norm: bool = False


@strict
class Nemotron3DenseVLMoTConfig(PretrainedConfig):
    """
    Config of the Nemotron 3 Dense VL MoT backbone (Cosmos3 Edge), defaults from ``nvidia/Cosmos3-Edge``.
    Sub-configs given as dicts or left out are built in `__post_init__`.
    """

    model_type = "nemotron_3_dense_vl"
    sub_configs = {
        "vision_config": Nemotron3DenseVLVisionConfig,
        "projector_config": Nemotron3DenseVLProjectorConfig,
        "text_config": Nemotron3DenseVLMoTTextConfig,
    }

    text_config: dict | PretrainedConfig | None = None
    vision_config: dict | PretrainedConfig | None = None
    projector_config: dict | PretrainedConfig | None = None
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        for name, config_cls in self.sub_configs.items():
            sub_config = getattr(self, name)
            if isinstance(sub_config, dict):
                setattr(self, name, config_cls(**sub_config))
            elif sub_config is None:
                setattr(self, name, config_cls())

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        super().validate_architecture()
        if (
            isinstance(self.vision_config, Nemotron3DenseVLVisionConfig)
            and isinstance(self.projector_config, Nemotron3DenseVLProjectorConfig)
            and self.vision_config.spatial_merge_size != self.projector_config.spatial_merge_size
        ):
            raise ValueError(
                f"vision_config.spatial_merge_size ({self.vision_config.spatial_merge_size}) must match "
                f"projector_config.spatial_merge_size ({self.projector_config.spatial_merge_size})."
            )


def _encode_mot_backbone_config(value: PretrainedConfig, declared_type: type | None = None) -> dict[str, Any]:
    """Serialize a MoT backbone config into the JSON of `config.json`, see `Cosmos3Config.mot_backbone`."""
    return value.to_dict()


def _decode_mot_backbone_config(
    config_class: type[PretrainedConfig], raw_value: Any, path: Sequence[str] = ()
) -> PretrainedConfig:
    """
    Rebuild a MoT backbone config from `config.json` or the command line.

    Args:
        config_class: The config class the field declares, one member of the union of `mot_backbone`.
        raw_value: The serialized config, the dict written by `_encode_mot_backbone_config`, or some of its fields
            from the command line.
        path: The field breadcrumb draccus prefixes onto error messages.
    """
    if isinstance(raw_value, config_class):
        return raw_value
    if not isinstance(raw_value, dict):
        raise DecodingError(
            path, f"expected a dict for {config_class.__name__}, got {type(raw_value).__name__}"
        )
    # Rejects the backbone config of the other policy type.
    if raw_value.get("model_type", config_class.model_type) != config_class.model_type:
        raise DecodingError(
            path, f"model_type {raw_value.get('model_type')!r} is not {config_class.model_type!r}"
        )
    return config_class.from_dict(raw_value)


if TYPE_CHECKING or _transformers_available:
    # draccus encoding and decoding of the two backbone configs, for config.json and the command line.
    for _config_class in (Qwen3VLMoTConfig, Nemotron3DenseVLMoTConfig):
        draccus.encode.register(_config_class, _encode_mot_backbone_config)
        draccus.decode.register(_config_class, partial(_decode_mot_backbone_config, _config_class))


# spellchecker:off (`temperal_downsample` is the argument name of `AutoencoderKLWan.__init__`)
@dataclass
class Wan2pt2VAEConfig:
    """
    Config of the Wan2.2 VAE (`diffusers.AutoencoderKLWan`) of the released Cosmos3 checkpoints: 48 latent
    channels, 16x spatial and 4x temporal compression. The fields are the arguments of
    `AutoencoderKLWan.__init__`, the defaults the `vae/config.json` of the released checkpoints.

    Args:
        base_dim: Channels of the first encoder stage.
        decoder_base_dim: Channels of the first decoder stage.
        z_dim: Latent channels.
        dim_mult: Channel multiplier of every stage.
        num_res_blocks: Residual blocks per stage.
        attn_scales: Scales with self attention.
        temperal_downsample: Whether every downsampling stage also halves the frames.
        dropout: Dropout of the residual blocks.
        latents_mean: Mean of every latent channel, `z_dim` values.
        latents_std: Standard deviation of every latent channel, `z_dim` values.
        is_residual: Residual down and up sampling blocks of the Wan2.2 VAE.
        in_channels: Input channels after the pixel patchify, 3 colors times `patch_size**2`.
        out_channels: Output channels before the pixel unpatchify.
        patch_size: Pixel patch size of the input patchify.
        scale_factor_temporal: Temporal compression from frames to latents.
        scale_factor_spatial: Spatial compression from pixels to latents.
    """

    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    dim_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    attn_scales: list[float] = field(default_factory=list)
    temperal_downsample: list[bool] = field(default_factory=lambda: [False, True, True])
    dropout: float = 0.0
    latents_mean: list[float] = field(default_factory=lambda: list(WAN2PT2_LATENTS_MEAN))
    latents_std: list[float] = field(default_factory=lambda: list(WAN2PT2_LATENTS_STD))
    is_residual: bool = True
    in_channels: int = 12
    out_channels: int = 12
    patch_size: int | None = 2
    scale_factor_temporal: int | None = 4
    scale_factor_spatial: int | None = 16

    def __post_init__(self):
        if len(self.latents_mean) != self.z_dim or len(self.latents_std) != self.z_dim:
            raise ValueError(
                f"latents_mean and latents_std need z_dim={self.z_dim} values, got {len(self.latents_mean)} "
                f"and {len(self.latents_std)}."
            )


# spellchecker:on


@dataclass
class Cosmos3Config(PreTrainedConfig):
    """
    Configuration shared by the two Cosmos3 policy types, `cosmos3_nano` (`Cosmos3NanoConfig`) and
    `cosmos3_edge` (`Cosmos3EdgeConfig`), which set the MoT backbone and its tokenizer. The recipe fields
    default to the native `action_policy_libero_nano` post-training recipe. The config.json of the released
    base checkpoints holds the native DROID recipe.

    The architecture fields are fixed by the weights. The training switches and the flow matching fields only
    affect training, the sampling fields only inference. The data fields have to match the training data.
    """

    # LeRobot policy settings
    # Observation steps, the current one and the past states as history actions. Only 1 is supported.
    n_obs_steps: int = 1
    chunk_size: int = 16  # Number of action steps predicted per denoising call
    n_action_steps: int = 16  # Number of action steps to execute
    # Actions are zero padded to this width, the width of the action projections.
    max_action_dim: int = 64
    # Storage dtype of the parameters, except the modules kept in float32.
    dtype: torch.dtype | None = torch.float32
    # Normalization with the dataset statistics. Action quantiles as in the native LIBERO recipe, the DROID
    # and RoboCasa recipes use IDENTITY.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Architecture (`Cosmos3VFMNetwork`), fixed by the weights.
    # MoT backbone, `Qwen3VLMoTConfig` (Cosmos3 Nano) or `Nemotron3DenseVLMoTConfig` (Cosmos3 Edge), and the
    # Hub repo or local directory of its tokenizer. Each size sets them.
    mot_backbone: Qwen3VLMoTConfig | Nemotron3DenseVLMoTConfig = field(kw_only=True)
    tokenizer_name: str = field(kw_only=True)
    # The `diffusers.AutoencoderKLWan` VAE.
    vae: Wan2pt2VAEConfig = field(default_factory=Wan2pt2VAEConfig)
    latent_patch_size: int = 2  # Side of the square latent patch that forms one generation token
    timestep_scale: float = 0.001  # Network timestep in [0, num_train_timesteps] to timestep embedding input
    timestep_embedding_dim: int = 256  # Width of the sinusoidal timestep features
    num_embodiment_domains: int = 32  # Embodiments with their own action projections
    # The M-RoPE time axis advances base_fps / temporal compression positions per second of video.
    base_fps: float = 24.0
    # M-RoPE gap between the last text position and the generation stream.
    temporal_modality_margin: int = 15000
    # Interleave video and action tokens per latent frame with temporally causal attention (not implemented).
    video_temporal_causal: bool = False

    # Training switches (`Cosmos3VFMNetwork`): which parameters train.
    # Train only the generation pathway of the backbone and the layers around it, as the native recipes do.
    train_gen_stream_only: bool = True
    freeze_vision_encoder: bool = True
    # Load the action modules (`action2llm`, `llm2action`, `action_modality_embed`) from the checkpoint. False
    # keeps their initialization, as the native post-training recipes do, and turns back to true once loaded.
    load_action_modules: bool = True
    # transformers gradient checkpointing of the decoder layers and the vision blocks of the backbone.
    gradient_checkpointing: bool = False
    # torch.compile the decoder layers of the backbone (one graph shared by the layers, dynamic shapes) and the VAE
    # encoder. `compile_mode` is the `torch.compile` mode, the CUDA graph modes ("reduce-overhead",
    # "max-autotune") are not supported.
    compile_model: bool = False
    compile_mode: str = "default"
    # Attention backend of the backbone, the transformers `attn_implementation` ("eager", "sdpa",
    # "flash_attention_2", ...). None takes the transformers default.
    attn_implementation: str | None = None

    # Flow matching training (`OmniMoTModel.forward`). Defaults of the native DROID, LIBERO and RoboCasa recipes.
    # Scale from the noise level sigma in [0, 1] to the network timestep, for training and sampling.
    num_train_timesteps: int = 1000
    # Distribution of the video noise level before the shift.
    video_time_distribution: TimeDistribution = "waver"
    # Shift of the training noise level, `shift * t / (1 + (shift - 1) * t)`. A dict maps the resolution tier of
    # the canvas ("256", "480", "720") to its shift.
    training_shift: float | dict[str, float] = field(
        default_factory=lambda: {"256": 3.0, "480": 5.0, "720": 10.0}
    )
    # Draw the action noise level separately, otherwise the actions share the noise level of their video.
    independent_action_schedule: bool = False
    action_time_distribution: TimeDistribution = "logitnormal"  # With independent_action_schedule
    action_training_shift: float | None = None  # With independent_action_schedule, None takes training_shift
    video_loss_weight: float = 10.0  # The native `loss_scale`
    action_loss_weight: float = 10.0

    # Sampling (`OmniMoTModel.generate`), only used at inference.
    num_inference_steps: int = 30
    # Classifier-free guidance scale on the text, 1 disables the unconditional branch.
    guidance_scale: float = 1.0
    # Apply the guidance only at timesteps strictly inside (low, high), in [0, num_train_timesteps]. Everywhere
    # when None.
    guidance_interval: tuple[float, float] | None = None
    sampling_shift: float = 5.0  # Shift of the sampling noise levels
    sampler: Literal["unipc", "edm"] = "unipc"  # Solver of the flow ODE, EDM is not implemented yet
    # Keep the predicted video latents of every chunk for lerobot-eval to decode and save.
    save_predicted_video: bool = False

    # Data layout, has to match the training data. `fps` and `view_size` left at None are taken from the
    # training dataset when the policy is built.
    # Robot mode of the sequence plan (`OmniMoTModel.build_sequence_plan`): which frames and actions are
    # clean conditions.
    mode: str = "wam"
    # Put the current state, in the action layout, in front of the actions as a clean state action.
    use_state: bool = False
    # Embodiment row of the action projections: 5 (LIBERO), 8 (DROID), 30 (RoboCasa) in the native recipes.
    domain_id: int = 5
    # Frame rate of the prompt, of the M-RoPE time axis of the video and of the idle frame thresholds. None takes the
    # frame rate of the training dataset.
    fps: float | None = None
    # Height and width every camera view is resized to before the canvas. None takes the size of the first camera.
    view_size: tuple[int, int] | None = None
    # Layout of the canvas, appended to the multi view template of the prompt.
    view_description: str = (
        "The left half shows the third-person view; the right half shows the wrist-mounted camera."
    )
    # Count the idle frames over the state action and the actions, as the native DROID recipe does.
    idle_state: bool = False
    # Names of the action channels, filled by make_policy from the dataset metadata. Names
    # `[<prefix>_]<type>_<index>` give the channel types of the idle frames of the prompt: `pos`, a rotation
    # (`rot6d`, `rot9d`, `axisangle`, `euler_xyz`, `quat_xyzw`, `quat_wxyz`), `gripper` or `joint`, the prefix tells
    # arms apart (`left_rot6d_0`).
    action_feature_names: list[str] | None = None
    # Probability of training a sample on the empty prompt, the unconditional branch of classifier-free guidance.
    cfg_dropout_rate: float = 0.1

    # Optimizer settings
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.05
    optimizer_grad_clip_norm: float = 1.0
    # Learning rate multiplier of the action projections and the action modality embedding, the
    # `lr_multipliers` of the native action policy recipes.
    optimizer_action_lr_multiplier: float = 5.0

    # Scheduler settings: the LambdaLinear schedule of the native action policy recipes, linear warmup over
    # `scheduler_warmup_steps`, then linear decay that reaches zero at step `scheduler_decay_steps`.
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 16_000

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError(f"Unsupported dtype={self.dtype!r}. Expected torch.bfloat16 or torch.float32.")
        if self.video_temporal_causal:
            raise NotImplementedError(
                "video_temporal_causal=True needs the three way attention, which is not implemented."
            )
        if (
            self.independent_action_schedule
            and self.action_training_shift is None
            and isinstance(self.training_shift, dict)
        ):
            raise ValueError(
                "independent_action_schedule with a per-resolution shift needs action_training_shift."
            )
        if self.sampler == "edm":
            raise NotImplementedError("The EDM sampler is not implemented yet, use 'unipc'.")
        if self.mode not in SUPPORTED_MODES:
            raise NotImplementedError(f"mode={self.mode!r} is not supported, only {SUPPORTED_MODES}.")
        if self.n_obs_steps != 1:
            raise NotImplementedError(
                f"n_obs_steps={self.n_obs_steps} is not supported, the history rows are not implemented yet."
            )
        if not 0 <= self.domain_id < self.num_embodiment_domains:
            raise ValueError(f"domain_id={self.domain_id} is outside [0, {self.num_embodiment_domains}).")
        temporal_compression = self.vae.scale_factor_temporal or 1
        if self.chunk_size % temporal_compression != 0:
            raise ValueError(
                f"chunk_size={self.chunk_size} must be a multiple of the VAE temporal compression "
                f"{temporal_compression}: the clip of chunk_size + 1 frames encodes to whole latent frames."
            )
        if not 0.0 <= self.cfg_dropout_rate <= 1.0:
            raise ValueError(f"cfg_dropout_rate={self.cfg_dropout_rate} is outside [0, 1].")

    @property
    def original_action_dim(self) -> int | None:
        """Width of the action the policy returns (`output_features`), None before make_policy sets it."""
        if not self.output_features or ACTION not in self.output_features:
            return None
        return self.output_features[ACTION].shape[0]

    def validate_features(self) -> None:
        """
        Fill `fps` and `view_size` left at None from the training dataset and the first camera, and check the
        state shape for `use_state`.
        """
        if self.input_features is None or self.output_features is None:
            raise ValueError("Cosmos3 requires `input_features` and `output_features` to be resolved first.")
        meta = getattr(self, "_runtime_dataset_meta", None)  # the training dataset, set by make_policy
        if self.fps is None and meta is not None:
            self.fps = float(meta.fps)
        if self.fps is None:
            raise ValueError("The frame rate is unknown: train on a dataset or pass --policy.fps.")
        if self.view_size is None:
            image_features = list(self.image_features)
            channels, height, width = self.input_features[image_features[0]].shape
            if channels != 3:
                raise ValueError(f"The first camera {image_features[0]} is not [3, H, W].")
            self.view_size = (height, width)
        state = self.robot_state_feature
        action_shape = self.output_features[ACTION].shape
        if self.use_state and (state is None or state.shape != action_shape):
            raise ValueError(
                "use_state puts observation.state in front of the actions as a row of the action layout, it needs "
                f"the shape of the action {action_shape}, got {None if state is None else state.shape}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> WarmupStableDecaySchedulerConfig:
        return WarmupStableDecaySchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps - self.scheduler_warmup_steps,
            num_stable_steps=0,
            decay_type="linear",
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int]:
        """The video clip of a training sample: the condition frame and the chunk_size frames the actions lead to."""
        return list(range(self.chunk_size + 1))

    @property
    def drop_n_last_frames(self) -> int:
        """Training windows end inside their episode (no padded clips)."""
        return self.chunk_size

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None


@PreTrainedConfig.register_subclass("cosmos3_nano")
@dataclass
class Cosmos3NanoConfig(Cosmos3Config):
    """Configuration of Cosmos3 Nano, `--policy.type=cosmos3_nano`: the Qwen3-VL MoT backbone of `nvidia/Cosmos3-Nano`."""

    mot_backbone: Qwen3VLMoTConfig = field(default_factory=Qwen3VLMoTConfig)
    tokenizer_name: str = "nvidia/Cosmos3-Nano"


@PreTrainedConfig.register_subclass("cosmos3_edge")
@dataclass
class Cosmos3EdgeConfig(Cosmos3Config):
    """
    Configuration of Cosmos3 Edge, `--policy.type=cosmos3_edge`: the Nemotron 3 Dense VL MoT backbone of
    `nvidia/Cosmos3-Edge`.
    """

    mot_backbone: Nemotron3DenseVLMoTConfig = field(default_factory=Nemotron3DenseVLMoTConfig)
    tokenizer_name: str = "nvidia/Cosmos3-Edge"
