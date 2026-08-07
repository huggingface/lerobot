# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    """Configuration class for the SmolVLA flow-matching vision-language-action policy.

    SmolVLA pairs a SmolVLM2 vision-language backbone with a smaller flow-matching action expert that
    cross-attends (or self-attends, depending on `attention_mode`) into the VLM's hidden states to
    generate action chunks.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy (the current step and
            additional steps going back).
        input_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from input feature name to its `PolicyFeature` (type and shape). Inferred from the
            dataset when left empty.
        output_features (`dict[str, PolicyFeature] | None`, *optional*):
            Mapping from output feature name to its `PolicyFeature` (type and shape). Inferred from the
            dataset when left empty.
        device (`str | None`, *optional*):
            Device to run the model on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`. Auto-detected when
            `None`.
        use_amp (`bool`, *optional*, defaults to `False`):
            Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether the policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub.
        repo_id (`str | None`, *optional*):
            Repository ID to push the trained policy to on the Hub.
        private (`bool | None`, *optional*):
            Whether to create the Hub repository as private.
        tags (`list[str] | None`, *optional*):
            Tags to attach to the policy's Hub repository.
        license (`str | None`, *optional*):
            License identifier to attach to the policy's Hub repository.
        pretrained_path (`Path | None`, *optional*):
            Repo ID on the Hub or local directory to load pretrained weights from. The policy is
            initialized from scratch when `None`.
        pretrained_revision (`str | None`, *optional*):
            Hub revision (commit hash, branch, or tag) to pin the pretrained model version.
        chunk_size (`int`, *optional*, defaults to 50):
            Number of action steps predicted per model invocation.
        n_action_steps (`int`, *optional*, defaults to 50):
            Number of predicted action steps actually executed in the environment before predicting a new
            chunk. Must not exceed `chunk_size`.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Mapping from feature type (`"VISUAL"`, `"STATE"`, `"ACTION"`) to the `NormalizationMode` used
            for it.
        max_state_dim (`int`, *optional*, defaults to 32):
            Dimension the observation state vector is zero-padded to when shorter.
        max_action_dim (`int`, *optional*, defaults to 32):
            Dimension the action vector is zero-padded to when shorter.
        resize_imgs_with_padding (`tuple[int, int]`, *optional*, defaults to `(512, 512)`):
            Target `(width, height)` images are resized (with aspect-ratio-preserving padding) to before
            being fed to the vision encoder.
        empty_cameras (`int`, *optional*, defaults to 0):
            Number of empty (zero-padded) camera views to add, e.g. for the aloha_sim variants that expect
            extra wrist cameras.
        adapt_to_pi_aloha (`bool`, *optional*, defaults to `False`):
            Whether to convert joint and gripper values from the standard Aloha space to the space used by
            the pi internal runtime the base model was trained with.
        use_delta_joint_actions_aloha (`bool`, *optional*, defaults to `False`):
            Whether to convert joint dimensions (gripper excluded) to values relative to the current state
            before passing them to the model. Not yet ported in LeRobot; raises if enabled.
        tokenizer_max_length (`int`, *optional*, defaults to 48):
            Maximum token length for the language tokenizer.
        num_steps (`int`, *optional*, defaults to 10):
            Number of flow-matching denoising steps performed at inference time.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use a key/value cache in the VLM and action expert for faster inference.
        freeze_vision_encoder (`bool`, *optional*, defaults to `True`):
            Whether to freeze the vision encoder's weights during training.
        train_expert_only (`bool`, *optional*, defaults to `True`):
            Whether to freeze the VLM and train only the action expert.
        train_state_proj (`bool`, *optional*, defaults to `True`):
            Whether to train the state projection layer.
        optimizer_lr (`float`, *optional*, defaults to 0.0001):
            Peak learning rate for the AdamW optimizer preset.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`):
            AdamW `(beta1, beta2)` coefficients.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            AdamW epsilon term for numerical stability.
        optimizer_weight_decay (`float`, *optional*, defaults to 1e-10):
            AdamW weight decay coefficient.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 10):
            Maximum gradient norm for clipping.
        scheduler_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of warmup steps for the cosine-decay-with-warmup learning rate scheduler.
        scheduler_decay_steps (`int`, *optional*, defaults to 30000):
            Number of decay steps for the learning rate scheduler.
        scheduler_decay_lr (`float`, *optional*, defaults to 2.5e-06):
            Learning rate the scheduler decays to at the end of `scheduler_decay_steps`.
        vlm_model_name (`str`, *optional*, defaults to `"HuggingFaceTB/SmolVLM2-500M-Video-Instruct"`):
            Hub identifier of the SmolVLM2 backbone to use.
        load_vlm_weights (`bool`, *optional*, defaults to `False`):
            Whether to load the VLM's pretrained weights. Set `False` when training the expert from
            scratch, `True` when initializing from pretrained SmolVLA weights.
        add_image_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether to surround image features with special image tokens.
        attention_mode (`str`, *optional*, defaults to `"cross_attn"`):
            How the action expert attends into the VLM's hidden states.
        prefix_length (`int`, *optional*, defaults to -1):
            Fixed length the VLM prefix (image and language tokens) is padded to. `-1` disables padding.
        pad_language_to (`str`, *optional*, defaults to `"longest"`):
            Padding strategy for the language tokenizer, e.g. `"longest"` or `"max_length"`.
        num_expert_layers (`int`, *optional*, defaults to -1):
            Number of transformer layers in the action expert. A value `<= 0` uses the same number of
            layers as the VLM; otherwise the expert has fewer layers.
        num_vlm_layers (`int`, *optional*, defaults to 16):
            Number of layers used from the VLM backbone (the first `num_vlm_layers` layers).
        self_attn_every_n_layers (`int`, *optional*, defaults to 2):
            Interleave a self-attention layer every `self_attn_every_n_layers` expert layers.
        expert_width_multiplier (`float`, *optional*, defaults to 0.75):
            The action expert's hidden size, expressed as a multiplier of the VLM's hidden size.
        min_period (`float`, *optional*, defaults to 0.004):
            Minimum period of the sinusoidal positional encoding used to embed the flow-matching timestep.
        max_period (`float`, *optional*, defaults to 4.0):
            Maximum period of the sinusoidal positional encoding used to embed the flow-matching timestep.
        rtc_config (`RTCConfig | None`, *optional*):
            Real-Time Chunking configuration. `None` disables RTC inference.
        compile_model (`bool`, *optional*, defaults to `False`):
            Whether to compile the model with `torch.compile`.
        compile_mode (`str`, *optional*, defaults to `"max-autotune"`):
            The `torch.compile` mode to use when `compile_model` is enabled.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the SmolVLM backbone configuration."""
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        """See [`~configs.PreTrainedConfig.get_optimizer_preset`]."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        """See [`~configs.PreTrainedConfig.get_scheduler_preset`]."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return [0]

    @property
    def action_delta_indices(self) -> list:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
