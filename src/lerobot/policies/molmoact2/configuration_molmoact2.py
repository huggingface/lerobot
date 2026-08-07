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

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    AdamWConfig,
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """Configuration for the MolmoAct2 policy, backed by the converted HF checkpoint implementation.

    MolmoAct2 supports three training modes via `action_mode`: `"continuous"` (flow-matching only),
    `"discrete"` (autoregressive token prediction only), or `"both"` (joint loss). At inference,
    `inference_action_mode` selects which head generates actions.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1): Number of environment steps of observation to
            pass to the policy (the current step plus this many additional steps looking back).
        input_features (`dict[str, PolicyFeature]`, *optional*): Mapping from input feature name to its
            `PolicyFeature` (type and shape). Left empty to be inferred from the dataset.
        output_features (`dict[str, PolicyFeature]`, *optional*): Mapping from output feature name
            (e.g. `"action"`) to its `PolicyFeature`. Left empty to be inferred from the dataset.
        device (`str | None`, *optional*): Device the policy runs on, e.g. `"cuda"`, `"cuda:0"`, `"cpu"`, or `"mps"`. If unset or unavailable, auto-selected on construction.
        use_amp (`bool`, *optional*, defaults to `False`): Whether to use Automatic Mixed Precision for training and evaluation.
        use_peft (`bool`, *optional*, defaults to `False`): Whether this policy is trained with PEFT (parameter-efficient fine-tuning) adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`): Whether to push the trained policy to the Hugging Face Hub after training.
        repo_id (`str | None`, *optional*): Hugging Face Hub repository id to push the policy to, when `push_to_hub` is enabled.
        private (`bool | None`, *optional*): Whether to create/push the Hub repository as private.
        tags (`list[str] | None`, *optional*): Tags to attach to the policy's Hub model card.
        license (`str | None`, *optional*): License identifier to add to the policy's Hub model card.
        pretrained_path (`pathlib.Path | None`, *optional*): Path or Hub repo id of pretrained weights to initialize the policy from. If `None`, the policy is initialized from scratch.
        pretrained_revision (`str | None`, *optional*): Hub revision (branch, tag, or commit hash) pinning the pretrained model version.
        checkpoint_path (`str`, *optional*, defaults to `"allenai/MolmoAct2"`): Hub id or local path of
            the pretrained MolmoAct2 HF checkpoint to load.
        checkpoint_revision (`str | None`, *optional*): Hub revision (commit hash, branch, or tag) for
            `checkpoint_path`.
        checkpoint_force_download (`bool`, *optional*, defaults to `False`): Whether to force
            re-downloading the checkpoint files, overriding the existing cache.
        chunk_size (`int`, *optional*, defaults to 30): The size of the action prediction chunk decoded
            per call to `predict_action_chunk`.
        n_action_steps (`int`, *optional*, defaults to 30): The number of actions from a predicted
            chunk that are actually queued for execution. Must not exceed `chunk_size`.
        action_mode (`str`, *optional*, defaults to `"both"`): Which action head(s) to train:
            `"continuous"`, `"discrete"`, or `"both"`.
        inference_action_mode (`str | None`, *optional*): Which action head to use at inference time,
            `"continuous"` or `"discrete"`. `None` defers to `action_mode`; must be compatible with it.
        discrete_action_tokenizer (`str`, *optional*, defaults to `"allenai/MolmoAct2-FAST-Tokenizer"`): Hub
            id of the FAST tokenizer used for discrete action generation.
        discrete_generation_max_steps (`int | None`, *optional*): Maximum number of autoregressive
            decoding steps for discrete action generation. `None` uses the checkpoint-derived default.
        norm_tag (`str | None`, *optional*): Tag identifying which normalization statistics to load
            from the checkpoint when `dataset_stats` isn't supplied to the processor factory.
        setup_type (`str`, *optional*, defaults to `""`): Setup-token identifier injected into the prompt; the empty
            default falls back to checkpoint metadata.
        control_mode (`str`, *optional*, defaults to `""`): Control-token identifier injected into the prompt; the empty
            default falls back to checkpoint metadata.
        image_keys (`list[str]`, *optional*): Explicit observation image keys to feed the model, in
            order. Falls back to checkpoint metadata, then to the visual features in `input_features`,
            when empty.
        normalize_language (`bool`, *optional*, defaults to `True`): Whether to normalize the language
            instruction text before tokenization.
        add_setup_tokens (`bool`, *optional*, defaults to `True`): Whether to inject setup tokens into
            the prompt.
        add_control_tokens (`bool`, *optional*, defaults to `True`): Whether to inject control tokens
            into the prompt.
        normalize_gripper (`bool`, *optional*, defaults to `False`): Whether to apply a dedicated
            gripper mask when normalizing/unnormalizing state and action.
        num_state_tokens (`int`, *optional*, defaults to 256): Number of tokens used to represent the
            proprioceptive state.
        max_sequence_length (`int | None`, *optional*): Maximum input sequence length. `None` uses the
            default MolmoAct2 sequence budget inferred from the fixed image/prompt/state/action token
            layout; override only for unusually long prompts.
        expected_max_action_dim (`int`, *optional*, defaults to 32): Action dimension the released
            MolmoAct2 checkpoints are fixed to; validated against the loaded checkpoint at model load.
        num_flow_timesteps (`int`, *optional*, defaults to 8): Number of flow-matching timesteps
            sampled during training.
        flow_matching_cutoff (`float`, *optional*, defaults to 1.0): Upper cutoff for the sampled
            flow-matching timestep fraction.
        flow_matching_time_offset (`float`, *optional*, defaults to 0.001): Offset applied to the
            sampled flow-matching timestep.
        flow_matching_time_scale (`float`, *optional*, defaults to 0.999): Scale applied to the sampled
            flow-matching timestep.
        flow_matching_beta_alpha (`float`, *optional*, defaults to 1.0): Alpha shape parameter of the
            Beta distribution used to sample flow-matching timesteps.
        flow_matching_beta_beta (`float`, *optional*, defaults to 1.5): Beta shape parameter of the Beta
            distribution used to sample flow-matching timesteps.
        num_inference_steps (`int | None`, *optional*): Number of flow-matching denoising steps at
            inference time. `None` keeps the checkpoint default.
        mask_action_dim_padding (`bool`, *optional*, defaults to `True`): Whether to mask out the
            zero-padded action dimensions during flow-matching denoising.
        enable_inference_cuda_graph (`bool`, *optional*, defaults to `True`): Whether to allow the
            backbone's CUDA graph manager to accelerate inference.
        per_episode_seed (`bool`, *optional*, defaults to `False`): MolmoAct2-local eval option; when
            enabled, stochastic continuous action generation uses a rollout-local generator derived
            from `eval_seed`.
        eval_seed (`int | None`, *optional*): Seed used to derive the rollout-local generator when
            `per_episode_seed` is set.
        rtc_config (`RTCConfig | None`, *optional*): Real-Time Chunking configuration. `None` disables
            RTC.
        joint_signs (`list[float] | None`, *optional*): Per-dimension sign correction applied to the
            observation state before the model and to the predicted action after it, for
            cross-calibration compatibility. Must be set together with `joint_offsets`.
        joint_offsets (`list[float] | None`, *optional*): Per-dimension offset correction applied
            alongside `joint_signs`. Must be set together with `joint_signs` and have the same length.
        enable_lora_vlm (`bool`, *optional*, defaults to `False`): Whether to apply LoRA adapters to the
            VLM instead of full fine-tuning.
        lora_rank (`int`, *optional*, defaults to 64): LoRA rank.
        lora_alpha (`int`, *optional*, defaults to 16): LoRA alpha.
        lora_dropout (`float`, *optional*, defaults to 0.05): LoRA dropout probability.
        lora_bias (`str`, *optional*, defaults to `"none"`): Which biases to train with LoRA:
            `"none"`, `"all"`, or `"lora_only"`.
        enable_lora_action_expert (`bool`, *optional*, defaults to `False`): Whether to also apply LoRA
            to the action expert. Requires `enable_lora_vlm`.
        enable_knowledge_insulation (`bool`, *optional*, defaults to `False`): Whether to stop the
            action expert's gradients from flowing back into the VLM.
        freeze_embedding (`bool`, *optional*, defaults to `True`): Whether to freeze the input
            embeddings during training.
        train_action_expert_only (`bool`, *optional*, defaults to `False`): Whether to train only the
            action expert parameters. Requires `action_mode="continuous"` and is incompatible with
            `enable_lora_vlm`.
        gradient_checkpointing (`bool`, *optional*, defaults to `False`): Whether to enable gradient
            checkpointing on the backbone.
        model_dtype (`str`, *optional*, defaults to `"bfloat16"`): Torch dtype to load the checkpoint
            in: `"float32"`, `"bfloat16"`, or `"float16"`.
        softmax_auxiliary_loss (`bool`, *optional*, defaults to `True`): Whether to add the softmax
            z-loss auxiliary term to the discrete-token loss.
        softmax_auxiliary_loss_scale (`float`, *optional*, defaults to 0.0001): Scale of the softmax
            auxiliary z-loss term.
        discrete_loss_token_weighting (`str`, *optional*, defaults to `"root_subsegments_root_tokens"`): How
            to weight tokens in the discrete cross-entropy loss.
        optimizer_lr (`float`, *optional*, defaults to 1e-05): Base AdamW learning rate.
        optimizer_vit_lr (`float`, *optional*, defaults to 5e-06): AdamW learning rate for the vision
            tower.
        optimizer_connector_lr (`float`, *optional*, defaults to 5e-06): AdamW learning rate for the
            vision-language connector.
        optimizer_action_expert_lr (`float`, *optional*, defaults to 5e-05): AdamW learning rate for the
            action expert.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.95)`): AdamW betas.
        optimizer_eps (`float`, *optional*, defaults to 1e-06): AdamW epsilon.
        optimizer_weight_decay (`float`, *optional*, defaults to 0.0): AdamW weight decay.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 1.0): Gradient clipping norm.
        scheduler_warmup_steps (`int`, *optional*, defaults to 200): Number of warmup steps for the
            cosine-decay-with-warmup scheduler.
        scheduler_decay_steps (`int`, *optional*, defaults to 100000): Number of decay steps for the
            scheduler.
        scheduler_decay_lr (`float`, *optional*, defaults to 1e-06): Final learning rate at the end of
            the decay schedule.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*): Per-feature-type
            normalization mode; defaults to `IDENTITY` for vision and `QUANTILES` for state/action.
        dataset_feature_names (`dict[str, Any]`, *optional*): Per-key feature names populated by
            `set_dataset_feature_metadata`; not meant to be set directly.
    """

    checkpoint_path: str = "allenai/MolmoAct2"
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    action_mode: str = "both"
    inference_action_mode: str | None = None
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    discrete_generation_max_steps: int | None = None
    norm_tag: str | None = None

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

    # Default is full finetuning with gradients from the action expert flowing into the VLM.
    enable_lora_vlm: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    enable_lora_action_expert: bool = False
    enable_knowledge_insulation: bool = False
    freeze_embedding: bool = True
    train_action_expert_only: bool = False
    gradient_checkpointing: bool = False

    model_dtype: str = "bfloat16"
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
    scheduler_decay_steps: int = 100_000
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
        """Validate the action-mode, LoRA, and joint-frame-transform field combinations.

        Raises:
            ValueError: If any of the cross-field constraints on `action_mode`,
                `inference_action_mode`, `joint_signs`/`joint_offsets`, `lora_*`, or the chunking/
                sequence-length fields are violated.
        """
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
        if self.train_action_expert_only and self.action_mode != "continuous":
            raise ValueError("MolmoAct2 train_action_expert_only requires action_mode='continuous'.")
        if self.train_action_expert_only and self.enable_lora_vlm:
            raise ValueError("MolmoAct2 train_action_expert_only is incompatible with enable_lora_vlm.")
        if self.enable_lora_action_expert and not self.enable_lora_vlm:
            raise ValueError("MolmoAct2 enable_lora_action_expert requires enable_lora_vlm.")
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
        if self.model_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                f"Unsupported model_dtype={self.model_dtype!r}. Expected 'float32', 'bfloat16', or 'float16'."
            )
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
        """Return indices for delta observations (None for MolmoAct2)."""
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """Return indices for delta actions."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """Return indices for delta rewards (None for MolmoAct2)."""
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        """Return the AdamW optimizer configuration built from the `optimizer_*` fields."""
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        """Return the cosine-decay-with-warmup scheduler configuration built from the `scheduler_*` fields."""
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        """Record the dataset's action/state feature names into `dataset_feature_names`.

        Args:
            features (dict[str, Any]): Dataset feature metadata, keyed by feature name (as found in
                `LeRobotDatasetMetadata.features`).
        """
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
