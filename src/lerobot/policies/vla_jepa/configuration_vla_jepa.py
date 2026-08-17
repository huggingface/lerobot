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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_STATE


@PreTrainedConfig.register_subclass("vla_jepa")
@dataclass
class VLAJEPAConfig(PreTrainedConfig):
    """Configuration class for the VLA-JEPA policy.

    VLA-JEPA combines a Qwen3-VL vision-language backbone, a flow-matching (DiT) action head, and an
    optional V-JEPA2 world model trained to predict future video-frame embeddings from the backbone's
    action tokens. The world model is an auxiliary training loss only; it is not used at inference.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy. Unused by this policy: the
            actual observation window is `num_video_frames`, via `observation_delta_indices`.
        input_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from input feature name to its `PolicyFeature` (type and shape). Populated
            automatically from the dataset when not explicitly provided.
        output_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from output feature name to its `PolicyFeature` (type and shape). Populated
            automatically from the dataset when not explicitly provided.
        device (`str | None`, *optional*):
            Device the policy runs on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`. Falls back to the
            best available device if unset or unavailable.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub after training.
        repo_id (`str | None`, *optional*):
            Hugging Face Hub repository id to push the policy to, when `push_to_hub` is enabled.
        private (`bool | None`, *optional*):
            Whether to create/push the Hub repository as private.
        tags (`list[str] | None`, *optional*):
            Tags to attach to the policy's Hub model card.
        license (`str | None`, *optional*):
            License identifier to add to the policy's Hub model card.
        pretrained_path (`Path | None`, *optional*):
            Path or Hub repo id of pretrained weights to initialize the policy from. If `None`, the
            policy is initialized from scratch.
        pretrained_revision (`str | None`, *optional*):
            Hub revision (branch, tag, or commit hash) pinning the pretrained model version.
        chunk_size (`int`, *optional*, defaults to 7):
            The size of the action prediction chunk, in units of environment steps.
        n_action_steps (`int`, *optional*, defaults to 7):
            The number of action steps to run in the environment for one invocation of the policy. Must
            be no greater than `chunk_size`.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps a feature type name (e.g. `"STATE"`, `"VISUAL"`) to the `NormalizationMode` to apply to
            it. Defaults to identity normalization for visual features, mean/std for state, and min/max
            for action features.
        qwen_model_name (`str`, *optional*, defaults to `"Qwen/Qwen3-VL-2B-Instruct"`):
            Name or path of the pretrained Qwen3-VL vision-language backbone.
        jepa_encoder_name (`str`, *optional*, defaults to `"facebook/vjepa2-vitl-fpc64-256"`):
            Name or path of the pretrained V-JEPA2 encoder used as the world model's (frozen) video
            target encoder.
        freeze_qwen (`bool`, *optional*, defaults to `False`):
            Whether to freeze the Qwen3-VL backbone's parameters during training. Enabling this also
            disables `enable_world_model`, since no gradient would otherwise flow into it.
        enable_world_model (`bool`, *optional*, defaults to `True`):
            Whether to build and train the V-JEPA world-model auxiliary loss and its encoder/predictor
            modules. Forced to `False` when `freeze_qwen` is `True`.
        reinit_modules (`list[str] | None`, *optional*):
            Key prefixes allowed to have shape mismatches when loading pretrained weights, for
            cross-embodiment transfer to a robot with a different action or state dimensionality (e.g.
            `["model.action_model.action_encoder", "model.action_model.state_encoder"]`). Mismatched
            tensors under these prefixes are randomly re-initialized instead of raising; any other
            mismatch still raises.
        tokenizer_padding_side (`str`, *optional*, defaults to `"left"`):
            Padding side used by the Qwen3-VL tokenizer.
        prompt_template (`str`, *optional*, defaults to `"Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."`):
            Template used to build the language prompt fed to Qwen3-VL, formatted with the task
            instruction and the action/embodied-action token placeholders.
        special_action_token (`str`, *optional*, defaults to `"<|action_{}|>"`):
            Format string for the per-timestep action token(s) added to the tokenizer's vocabulary.
        embodied_action_token (`str`, *optional*, defaults to `"<|embodied_action|>"`):
            Special token added to the tokenizer's vocabulary, whose hidden states condition the action
            head.
        action_dim (`int`, *optional*, defaults to 7):
            Dimensionality of the action vector. Overwritten from the dataset's action feature shape in
            `validate_features`.
        state_dim (`int`, *optional*, defaults to 8):
            Dimensionality of the robot state vector. Overwritten from the dataset's state feature shape
            in `validate_features`, when a state feature is present.
        num_action_tokens_per_timestep (`int`, *optional*, defaults to 8):
            Number of action tokens allocated per prompted timestep in the Qwen3-VL prompt.
        num_embodied_action_tokens_per_instruction (`int`, *optional*, defaults to 32):
            Number of embodied-action tokens allocated per instruction in the Qwen3-VL prompt; also sets
            the action head's future-token embedding count.
        num_inference_timesteps (`int`, *optional*, defaults to 4):
            Number of integration steps used by the flow-matching action head at inference time.
        action_hidden_size (`int`, *optional*, defaults to 1024):
            Hidden dimension of the action head's DiT output projection.
        action_model_type (`str`, *optional*, defaults to `"DiT-B"`):
            Named DiT preset (`"DiT-B"`, `"DiT-L"`, or `"DiT-test"`) providing default attention head
            count and head dimension for the action head, unless overridden by `action_num_heads` and
            `action_attention_head_dim`.
        action_num_layers (`int`, *optional*, defaults to 16):
            Number of transformer blocks in the action head's DiT.
        action_num_heads (`int | None`, *optional*):
            Number of attention heads for the action head's DiT. `None` uses the `action_model_type`
            preset's value.
        action_attention_head_dim (`int | None`, *optional*):
            Per-head attention dimension for the action head's DiT. `None` uses the `action_model_type`
            preset's value.
        action_dropout (`float`, *optional*, defaults to 0.2):
            Dropout used in the action head's DiT transformer blocks.
        action_num_timestep_buckets (`int`, *optional*, defaults to 1000):
            Number of discrete buckets the continuous flow-matching timestep is quantized into before
            being embedded.
        action_noise_beta_alpha (`float`, *optional*, defaults to 1.5):
            Alpha parameter of the Beta distribution used to sample the flow-matching timestep during
            training.
        action_noise_beta_beta (`float`, *optional*, defaults to 1.0):
            Beta parameter of the Beta distribution used to sample the flow-matching timestep during
            training.
        action_noise_s (`float`, *optional*, defaults to 0.999):
            Scale used to rescale the Beta-sampled value into a flow-matching timestep, as
            `(action_noise_s - sample) / action_noise_s`.
        num_target_vision_tokens (`int`, *optional*, defaults to 32):
            Reserved configuration field; not currently referenced by the model implementation.
        action_max_seq_len (`int`, *optional*, defaults to 1024):
            Reserved configuration field; not currently referenced by the model implementation.
        num_video_frames (`int`, *optional*, defaults to 8):
            Total number of video frames loaded per sample for the world model.
        predictor_depth (`int`, *optional*, defaults to 12):
            Number of transformer blocks in the world model's video predictor.
        predictor_num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the world model's video predictor.
        predictor_mlp_ratio (`float`, *optional*, defaults to 4.0):
            Feed-forward expansion ratio in the world model's video predictor.
        predictor_dropout (`float`, *optional*, defaults to 0.0):
            Reserved configuration field; not currently referenced by the model implementation.
        world_model_loss_weight (`float`, *optional*, defaults to 0.1):
            Weight applied to the world-model's video-prediction loss before adding it to the action
            loss.
        jepa_tubelet_size (`int`, *optional*, defaults to 2):
            Number of camera views the world model consumes (video tensors are padded or trimmed to this
            many views), and the fallback tubelet size used to size action-token prompt placeholders
            when the world model is disabled. Should match the JEPA encoder's actual tubelet size (e.g.
            2 for `vjepa2-vitl-fpc64-256`).
        repeated_diffusion_steps (`int`, *optional*, defaults to 8):
            Number of independent noise draws per batch item used to repeat the flow-matching loss
            computation (CogACT-style).
        resize_images_to (`tuple[int, int] | None`, *optional*):
            Target `(height, width)` to resize input images to before inference. `None` (the default)
            keeps the original resolution. Not applied during training.
        binarize_gripper_action (`bool`, *optional*, defaults to `True`):
            Whether to binarize the gripper action dimension after unnormalization, in the
            post-processing pipeline built by `make_vla_jepa_pre_post_processors`.
        pre_snap_gripper_action (`bool`, *optional*, defaults to `True`):
            Whether to snap the gripper action dimension to `{0, 1}` before unnormalization, in the
            post-processing pipeline built by `make_vla_jepa_pre_post_processors`.
        clip_normalized_actions (`bool`, *optional*, defaults to `True`):
            Whether to clip normalized actions to `[-1, 1]` before unnormalization, in the
            post-processing pipeline built by `make_vla_jepa_pre_post_processors`.
        gripper_dim (`int`, *optional*, defaults to 6):
            Index of the gripper dimension within the action vector, used by the gripper pre/post
            -processing steps.
        gripper_threshold (`float`, *optional*, defaults to 0.5):
            Threshold used by the gripper pre/post-processing steps to binarize the gripper action.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Torch dtype (`"float32"`, `"float16"`, or `"bfloat16"`) used to load the Qwen3-VL backbone
            and (when enabled) the V-JEPA world-model encoder.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Learning rate for the AdamW optimizer preset.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`):
            Adam beta coefficients for the AdamW optimizer preset.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Epsilon for the AdamW optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 1e-10):
            Weight decay for the AdamW optimizer preset.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 10.0):
            Gradient-clipping norm for the AdamW optimizer preset.
        scheduler_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of warmup steps for the cosine-decay-with-warmup learning rate scheduler preset.
        scheduler_decay_steps (`int`, *optional*, defaults to 30000):
            Number of decay steps for the cosine-decay-with-warmup learning rate scheduler preset.
        scheduler_decay_lr (`float`, *optional*, defaults to 2.5e-06):
            Final learning rate at the end of decay, for the cosine-decay-with-warmup scheduler preset.
    """

    n_obs_steps: int = 1
    chunk_size: int = 7
    n_action_steps: int = 7

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    qwen_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    jepa_encoder_name: str = "facebook/vjepa2-vitl-fpc64-256"
    freeze_qwen: bool = False
    enable_world_model: bool = True
    reinit_modules: list[str] | None = None

    tokenizer_padding_side: str = "left"
    prompt_template: str = "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."
    special_action_token: str = "<|action_{}|>"
    embodied_action_token: str = "<|embodied_action|>"

    action_dim: int = 7
    state_dim: int = 8

    num_action_tokens_per_timestep: int = 8
    num_embodied_action_tokens_per_instruction: int = 32
    num_inference_timesteps: int = 4

    action_hidden_size: int = 1024
    action_model_type: str = "DiT-B"
    action_num_layers: int = 16
    action_num_heads: int | None = None
    action_attention_head_dim: int | None = None
    action_dropout: float = 0.2
    action_num_timestep_buckets: int = 1000
    action_noise_beta_alpha: float = 1.5
    action_noise_beta_beta: float = 1.0
    action_noise_s: float = 0.999
    num_target_vision_tokens: int = 32
    action_max_seq_len: int = 1024

    num_video_frames: int = 8
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    world_model_loss_weight: float = 0.1
    jepa_tubelet_size: int = 2
    repeated_diffusion_steps: int = 8

    resize_images_to: tuple[int, int] | None = None
    binarize_gripper_action: bool = True
    pre_snap_gripper_action: bool = True
    clip_normalized_actions: bool = True
    gripper_dim: int = 6
    gripper_threshold: float = 0.5
    torch_dtype: str = "bfloat16"

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the JEPA world-model and action-head configuration."""
        super().__post_init__()
        if self.freeze_qwen and self.enable_world_model:
            # freezing qwen backbone makes world model training irrelevant since no grad flows
            self.enable_world_model = False
        if self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be <= `chunk_size`.")
        if self.num_video_frames < 2 * self.jepa_tubelet_size:
            raise ValueError(
                f"`video_horizon` ({self.num_video_frames}) must be >= 2 * `jepa_tubelet_size` "
                f"({self.jepa_tubelet_size}) to have at least one context and one GT temporal position."
            )

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        if not self.image_features:
            raise ValueError("VLAJEPA requires at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("VLAJEPA requires an action output feature.")
        self.action_dim = self.action_feature.shape[0]
        if self.robot_state_feature is not None:
            self.state_dim = self.robot_state_feature.shape[0]

    def set_dataset_feature_metadata(self, dataset_features: dict[str, Any]) -> None:
        """Add `observation.state` to `input_features` if missing, so it gets normalized."""
        if OBS_STATE in self.input_features or OBS_STATE not in dataset_features:
            return
        shape = tuple(dataset_features[OBS_STATE]["shape"])
        self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=shape)

    def get_optimizer_preset(self) -> AdamWConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        # load video_horizon frames starting from current timestep: [t, t+1, ..., t+video_horizon-1]
        # matches original repo's observation_indices=list(range(video_horizon))
        return list(range(self.num_video_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
