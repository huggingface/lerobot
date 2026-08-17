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

import logging
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


@PreTrainedConfig.register_subclass("evo1")
@dataclass
class Evo1Config(PreTrainedConfig):
    """Configuration for the EVO1 vision-language-action policy.

    EVO1 pairs an InternVL3 vision-language backbone with a flow-matching action head. Training proceeds
    in two stages (`training_stage`): stage 1 freezes the VLM and trains only the action head, stage 2
    fine-tunes the whole model.

    Args:
        n_obs_steps (`int`, *optional*, defaults to 1):
            Number of environment steps of observation to pass to the policy.
        input_features (`dict[str, PolicyFeature]`, *optional*):
            Input feature specification, keyed by feature name. Left empty to infer from the dataset.
        output_features (`dict[str, PolicyFeature]`, *optional*):
            Output feature specification, keyed by feature name. Left empty to infer from the dataset.
        device (`str`, *optional*):
            Torch device to run the policy on, e.g. `"cuda"` or `"cpu"`. Auto-selected when unset or
            unavailable.
        use_amp (`bool`, *optional*, defaults to `True`):
            Whether to use Automatic Mixed Precision. EVO1 also manages its own bfloat16 autocast around
            its forward passes independently of this flag; see `dtype`-related fields below.
        use_peft (`bool`, *optional*, defaults to `False`):
            Whether this policy is trained with PEFT adapters.
        push_to_hub (`bool`, *optional*, defaults to `True`):
            Whether to push the trained policy to the Hugging Face Hub.
        repo_id (`str`, *optional*):
            Hub repository id to push the policy to.
        private (`bool`, *optional*):
            Whether the pushed Hub repository is private.
        tags (`list[str]`, *optional*):
            Tags to attach to the policy on the Hub.
        license (`str`, *optional*):
            License identifier for the policy on the Hub.
        pretrained_path (`Path`, *optional*):
            Repo id or local directory of pretrained weights saved with `save_pretrained`. Left unset to
            initialize from scratch.
        pretrained_revision (`str`, *optional*):
            Hub revision to pin when loading `pretrained_path`.
        training_stage (`str`, *optional*, defaults to `"stage1"`):
            Either `"stage1"` (VLM frozen, only the action head trains) or `"stage2"` (the whole model
            trains). Drives the default `finetune_*` flags unless they are set explicitly and
            `apply_training_stage_defaults` is `False`.
        chunk_size (`int`, *optional*, defaults to 50):
            Number of actions predicted by the flow-matching head per inference call.
        n_action_steps (`int`, *optional*, defaults to 50):
            Number of actions from a predicted chunk that are actually executed before re-querying the
            policy. Must not exceed `chunk_size`.
        max_state_dim (`int`, *optional*, defaults to 24):
            Padded dimensionality of the state vector fed to the action head.
        max_action_dim (`int`, *optional*, defaults to 24):
            Padded dimensionality of the action vector fed to the action head.
        max_views (`int`, *optional*, defaults to 3):
            Maximum number of camera streams the policy accepts.
        image_resolution (`tuple[int, int]`, *optional*, defaults to `(448, 448)`):
            Target resolution images are resized to before the InternVL3 embedder. Must be square.
        empty_cameras (`int`, *optional*, defaults to 0):
            Number of placeholder, always-masked-out camera views added to `input_features` so the batch
            has a fixed number of views regardless of how many real cameras the dataset provides.
        postprocess_action_dim (`int`, *optional*):
            Overrides the action dimensionality the postprocessor crops predictions down to. Falls back to
            the dataset's action feature width, or `max_action_dim` if that is unavailable.
        binarize_gripper (`bool`, *optional*, defaults to `False`):
            Whether the postprocessor snaps the gripper action channel to one of two fixed values instead
            of passing through the continuous prediction.
        gripper_index (`int`, *optional*, defaults to 6):
            Index of the gripper channel within the action vector, used when `binarize_gripper` is `True`.
        gripper_threshold (`float`, *optional*, defaults to 0.5):
            Decision threshold applied to the gripper channel when `binarize_gripper` is `True`.
        gripper_below_threshold_value (`float`, *optional*, defaults to 1.0):
            Value written to the gripper channel when it is at or below `gripper_threshold`.
        gripper_above_threshold_value (`float`, *optional*, defaults to -1.0):
            Value written to the gripper channel when it is above `gripper_threshold`.
        normalization_mapping (`dict[str, NormalizationMode]`, *optional*):
            Maps each `FeatureType` to the `NormalizationMode` used to normalize/unnormalize it.
        vlm_model_name (`str`, *optional*, defaults to `"OpenGVLab/InternVL3-1B-hf"`):
            Hugging Face Hub id of the InternVL3 vision-language backbone.
        vlm_num_layers (`int`, *optional*, defaults to 14):
            Number of transformer layers kept from the InternVL3 language model. `None` keeps all of them.
        vlm_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Dtype the InternVL3 backbone is loaded in.
        max_text_length (`int`, *optional*, defaults to 1024):
            Maximum token length for the tokenized (image placeholders + instruction) prompt. Longer
            prompts are right-truncated.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to request FlashAttention in the InternVL3 backbone.
        action_head (`str`, *optional*, defaults to `"flowmatching"`):
            Identifier of the action-generation head architecture.
        embed_dim (`int`, *optional*, defaults to 896):
            Dimensionality of the fused vision-language token embeddings consumed by the action head.
        hidden_dim (`int`, *optional*, defaults to 1024):
            Hidden width of the action head's transformer layers.
        state_hidden_dim (`int`, *optional*, defaults to 1024):
            Hidden width of the state encoder inside the action head.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads in the action head's transformer layers.
        num_layers (`int`, *optional*, defaults to 8):
            Number of transformer layers in the action head.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability applied inside the action head.
        num_inference_timesteps (`int`, *optional*, defaults to 32):
            Number of integration steps used to sample an action chunk from the flow-matching head.
        num_categories (`int`, *optional*, defaults to 1):
            Number of embodiment categories the action head conditions on.
        return_cls_only (`bool`, *optional*, defaults to `False`):
            Whether the action head is conditioned on a single pooled VL token (the last non-padding token
            of the causal decoder) instead of the full fused token sequence.
        enable_gradient_checkpointing (`bool`, *optional*, defaults to `True`):
            Whether to enable gradient checkpointing on the VLM backbone to reduce memory usage.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether gradient checkpointing uses the reentrant autograd variant.
        finetune_vlm (`bool`, *optional*):
            Whether the whole VLM backbone is trainable. Defaulted from `training_stage` unless set
            explicitly with `apply_training_stage_defaults=False`. Must agree with the union of
            `finetune_language_model` and `finetune_vision_model` when those are set explicitly.
        finetune_language_model (`bool`, *optional*):
            Whether the VLM's language branch is trainable. Defaulted from `training_stage` unless set
            explicitly with `apply_training_stage_defaults=False`.
        finetune_vision_model (`bool`, *optional*):
            Whether the VLM's vision branch is trainable. Defaulted from `training_stage` unless set
            explicitly with `apply_training_stage_defaults=False`.
        finetune_action_head (`bool`, *optional*):
            Whether the flow-matching action head is trainable. Defaulted from `training_stage` unless set
            explicitly with `apply_training_stage_defaults=False`.
        apply_training_stage_defaults (`bool`, *optional*, defaults to `True`):
            Whether to reapply the `training_stage` defaults to the `finetune_*` flags after loading a
            checkpoint config, so a stage-2 run cannot silently inherit a stage-1 checkpoint's frozen-VLM
            flags. Set `False` to keep explicit finetuning flags.
        task_field (`str`, *optional*, defaults to `"task"`):
            Batch key holding the language instruction(s) passed to the VLM.
        embodiment_id_field (`str`, *optional*):
            Batch key holding an explicit per-sample embodiment id. Falls back to `"embodiment_id"`, then
            to `default_embodiment_id`, when unset or absent from the batch.
        default_embodiment_id (`int`, *optional*, defaults to 0):
            Embodiment id used when the batch carries none. Must be in `[0, num_categories)`.
        rtc_config (`RTCConfig`, *optional*):
            Real-Time Chunking guidance for asynchronous inference. `None` disables RTC.
            `lerobot-rollout --inference.type=rtc` sets this and calls `init_rtc_processor()`.
        optimizer_lr (`float`, *optional*, defaults to 1e-05):
            Learning rate used to build the default `AdamWConfig` optimizer preset.
        optimizer_betas (`tuple[float, float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam beta coefficients for the default optimizer preset.
        optimizer_eps (`float`, *optional*, defaults to 1e-08):
            Adam epsilon for the default optimizer preset.
        optimizer_weight_decay (`float`, *optional*, defaults to 1e-05):
            Weight decay applied to the decayed parameter group in the default optimizer preset.
        optimizer_grad_clip_norm (`float`, *optional*, defaults to 1.0):
            Gradient-norm clipping threshold for the default optimizer preset.
        scheduler_warmup_steps (`int`, *optional*, defaults to 300):
            Number of warmup steps for the default cosine-annealing-with-warmup scheduler preset.
    """

    training_stage: str = "stage1"
    # When True and the policy runs on CUDA, EVO1 wraps its own forward passes (training and
    # inference) in a bfloat16 autocast block, so its numerics do not depend on the dtype of any
    # outer autocast context opened by lerobot-train/lerobot-eval.
    use_amp: bool = True

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 24
    max_action_dim: int = 24
    max_views: int = 3
    image_resolution: tuple[int, int] = (448, 448)
    empty_cameras: int = 0
    postprocess_action_dim: int | None = None
    binarize_gripper: bool = False
    gripper_index: int = 6
    gripper_threshold: float = 0.5
    gripper_below_threshold_value: float = 1.0
    gripper_above_threshold_value: float = -1.0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    vlm_model_name: str = "OpenGVLab/InternVL3-1B-hf"
    vlm_num_layers: int | None = 14
    vlm_dtype: str = "bfloat16"
    # Max token length for tokenizing the (image placeholders + instruction) prompt. Prompts longer
    # than this are right-truncated, so raise it for tasks with long language instructions or many views.
    max_text_length: int = 1024
    use_flash_attn: bool = True
    action_head: str = "flowmatching"
    embed_dim: int = 896
    hidden_dim: int = 1024
    state_hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.0
    num_inference_timesteps: int = 32
    num_categories: int = 1
    # When True, the action head is conditioned on a single pooled VL token (the last non-padding
    # token of the causal decoder) instead of the full fused token sequence.
    return_cls_only: bool = False
    enable_gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    finetune_vlm: bool | None = None
    finetune_language_model: bool | None = None
    finetune_vision_model: bool | None = None
    finetune_action_head: bool | None = None
    # Reapply stage defaults after loading checkpoint configs so stage2 cannot
    # accidentally inherit the frozen VLM flags stored by a stage1 checkpoint.
    apply_training_stage_defaults: bool = True

    task_field: str = "task"
    embodiment_id_field: str | None = None
    default_embodiment_id: int = 0

    # Real-Time Chunking guidance for asynchronous inference (lerobot-rollout --inference.type=rtc
    # sets this and calls init_rtc_processor()); None disables RTC.
    rtc_config: RTCConfig | None = None

    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 300

    def __post_init__(self):
        """Resolve `device` (see [`~configs.PreTrainedConfig.__post_init__`]), then validate this config. Validates the VLM backbone/tokenizer configuration."""
        super().__post_init__()
        if self.training_stage not in {"stage1", "stage2"}:
            raise ValueError(
                f"Unsupported EVO1 training_stage '{self.training_stage}', expected 'stage1' or 'stage2'"
            )

        if self.apply_training_stage_defaults:
            stage_defaults = {
                "stage1": {
                    "finetune_vlm": False,
                    "finetune_language_model": False,
                    "finetune_vision_model": False,
                    "finetune_action_head": True,
                },
                "stage2": {
                    "finetune_vlm": True,
                    "finetune_language_model": True,
                    "finetune_vision_model": True,
                    "finetune_action_head": True,
                },
            }[self.training_stage]
            for flag_name, default_value in stage_defaults.items():
                current_value = getattr(self, flag_name)
                if current_value is not None and current_value != default_value:
                    logger.warning(
                        "EVO1 %s=%s is overridden by training_stage=%s default %s. "
                        "Set apply_training_stage_defaults=false to keep explicit finetuning flags.",
                        flag_name,
                        current_value,
                        self.training_stage,
                        default_value,
                    )
                setattr(self, flag_name, default_value)
        elif self.training_stage == "stage1":
            if self.finetune_vlm is None:
                self.finetune_vlm = False
            if self.finetune_language_model is None:
                self.finetune_language_model = False
            if self.finetune_vision_model is None:
                self.finetune_vision_model = False
            if self.finetune_action_head is None:
                self.finetune_action_head = True
        elif self.training_stage == "stage2":
            has_explicit_branch_flags = any(
                flag is not None for flag in (self.finetune_language_model, self.finetune_vision_model)
            )
            if not has_explicit_branch_flags:
                # An explicit finetune_vlm decides both branches; otherwise stage2 defaults to a
                # full-VLM finetune.
                vlm_finetune = self.finetune_vlm if self.finetune_vlm is not None else True
                self.finetune_vlm = vlm_finetune
                self.finetune_language_model = vlm_finetune
                self.finetune_vision_model = vlm_finetune
            elif self.finetune_vlm is None:
                self.finetune_vlm = bool(self.finetune_language_model or self.finetune_vision_model)
            if self.finetune_action_head is None:
                self.finetune_action_head = True

        if self.finetune_vlm is None:
            self.finetune_vlm = False
        if self.finetune_language_model is None:
            self.finetune_language_model = False
        if self.finetune_vision_model is None:
            self.finetune_vision_model = False
        if self.finetune_action_head is None:
            self.finetune_action_head = False

        branch_vlm = self.finetune_language_model or self.finetune_vision_model
        if self.finetune_vlm != branch_vlm:
            raise ValueError(
                "Inconsistent EVO1 finetune config: "
                f"finetune_vlm={self.finetune_vlm} but "
                f"(finetune_language_model or finetune_vision_model)={branch_vlm}. "
                "When branch-level flags are used, finetune_vlm must match their effective union."
            )

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )
        if len(self.image_resolution) != 2 or self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError(
                "EVO1 currently expects a square image_resolution because InternVL3 preprocessing "
                f"uses a scalar image_size, got {self.image_resolution}."
            )
        if not 0 <= self.default_embodiment_id < self.num_categories:
            raise ValueError(
                f"default_embodiment_id ({self.default_embodiment_id}) must be in "
                f"[0, num_categories={self.num_categories})"
            )

    def validate_features(self) -> None:
        """See [`~configs.PreTrainedConfig.validate_features`]."""
        if self.input_features is None:
            self.input_features = {}
        if self.output_features is None:
            self.output_features = {}

        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            if key not in self.input_features:
                self.input_features[key] = PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, *self.image_resolution),
                )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

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
        return CosineAnnealingWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.observation_delta_indices`]."""
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        """See [`~configs.PreTrainedConfig.action_delta_indices`]."""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """See [`~configs.PreTrainedConfig.reward_delta_indices`]."""
        return None
