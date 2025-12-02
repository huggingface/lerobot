# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("wall_x")
@dataclass
class WallXConfig(PreTrainedConfig):
    """
    Configuration class for Wall-X policy.

    Wall-X is based on Qwen2.5-VL with action prediction capabilities using flow matching.
    It supports cross-embodiment robotic control through unified action representations.

    This config supports multi-modal learning with vision, language, and action data.
    """

    # ==================== Model and Paths Configuration ====================
    # Logging
    log_name: str = "wall_x_training"
    log_project: str = "vla_training"
    model_type: str = "wall-oss"

    # Pretrained model paths
    pretrained_wallx_path: str | None = None  # Path to pretrained Wall-X model
    save_path: str | None = None  # Path to save checkpoints
    processor_path: str | None = None  # Path to processor (defaults to pretrained_wallx_path)
    action_tokenizer_path: str | None = None  # Path to action tokenizer (for FAST mode)

    # Tokenizer settings
    use_fast_tokenizer: bool = False  # True: train FAST, False: train Flow

    # ==================== Profiling Configuration ====================
    profile: bool = False
    profile_save_path: str | None = None
    profile_wait_iters: int = 10
    profile_warmup_iters: int = 5
    profile_active_iters: int = 2

    # ==================== Training Hyperparameters ====================
    num_warmup_steps: int = 100
    num_training_steps: int = 64000000
    learning_rate: float = 5e-5
    min_lr: float = 5e-5
    num_epoch: int = 100
    gradient_accumulation_steps: int = 32
    batch_size_per_gpu: int = 8
    padding_side: str = "left"
    epoch_save_interval: int = 10

    # Training optimization
    fsdp2: bool = False
    torch_compile: bool = False

    # ==================== Input / Output Structure ====================
    n_obs_steps: int = 1
    chunk_size: int = 32  # action_horizon in wall-x
    n_action_steps: int = 32

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Action dimension - wall-x uses hardcoded 20
    max_action_dim: int = 20
    max_state_dim: int = 20  # For proprioception

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] | None = None  # wall-x uses Qwen processor

    # Tokenizer
    tokenizer_max_length: int = 256

    # ==================== Model Architecture ====================
    vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    load_vlm_weights: bool = True

    # Vision config
    vision_config: dict = field(default_factory=lambda: {
        "depth": 32,
        "hidden_size": 3584,
        "hidden_act": "silu",
        "intermediate_size": 3420,
        "num_heads": 16,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 112,
        "out_hidden_size": 3584,
    })

    # Language model config
    hidden_size: int = 3584  # 8192 for 7B model
    intermediate_size: int = 18944  # 29568 for 7B model
    num_hidden_layers: int = 36  # 80 for 7B model
    num_attention_heads: int = 28  # 64 for 7B model
    num_key_value_heads: int = 4  # 8 for 7B model
    vocab_size: int = 152064

    # ==================== Action Prediction ====================
    # Action prediction mode: "flow" or "fast"
    prediction_mode: str = "flow"

    # Flow matching parameters
    noise_scheduler: dict = field(default_factory=lambda: {
        "beta_alpha": 1.5,  # Beta distribution concentration1
        "beta_beta": 1.0,   # Beta distribution concentration0
        "s": 0.999,         # Scaling factor for time
    })

    # Decoding parameters
    num_inference_timesteps: int = 10  # Number of ODE solver steps
    ode_solver_method: str = "euler"  # ODE solver method

    # ==================== Robot Configuration ====================
    # Degrees of freedom configuration - defines action space
    dof_config: dict = field(default_factory=lambda: {
        "left_ee_pos": 3,
        "left_ee_rot": 3,
        "left_gripper": 1,
        "right_ee_pos": 3,
        "right_ee_rot": 3,
        "right_gripper": 1,
    })

    # Proprioception configuration (typically mirrors dof_config)
    agent_pos_config: dict = field(default_factory=lambda: {
        "left_ee_pos": 3,
        "left_ee_rot": 3,
        "left_gripper": 1,
        "right_ee_pos": 3,
        "right_ee_rot": 3,
        "right_gripper": 1,
    })

    # Customized robot configuration
    enable_customized_robot_config: bool = False
    customized_robot_config: dict = field(default_factory=lambda: {
        "name": "",
        "customized_dof_config": {},
        "customized_agent_pos_config": {},
    })

    # Normalization statistics path
    norm_stats_path: str | None = None

    # ==================== MoE Configuration ====================
    num_experts: int = 4
    attention_moe: bool = False
    mlp_moe: bool = False

    # ==================== Finetuning Settings ====================
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False  # wall-x trains more components
    train_action_head: bool = True

    # Cache
    use_cache: bool = True

    # ==================== Optimizer Presets ====================
    optimizer_lr: float = 2e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-6

    # ==================== Dataset Configuration ====================
    # Dataset-specific normalization statistics
    action_statistics: dict = field(default_factory=dict)

    # Data configuration
    data_config: dict = field(default_factory=lambda: {
        "use_lerobot": True,
        "lerobot_config": {
            "repo_id": "",
            "root": None,
            "episodes": None,
            "image_transforms": None,
            "delta_timestamps": None,
            "tolerance_s": 1e-4,
            "revision": None,
            "force_cache_sync": False,
            "download_videos": True,
            "video_backend": None,
        },
        "action_horizon": 32,
        "train_test_split": 0.95,
        "obs_action_keys": [],
        "predict_action_keys": [],
        "resolution": {
            "face_view": 256,
            "left_wrist_view": 256,
            "right_wrist_view": 256,
            "move1_view": 256,
            "move2_view": 256,
            "top_view": 256,
            "wall_view": 256,
            "multi_modal": 256,
        },
    })

    # ==================== Resume Configuration ====================
    resume_config: dict | None = field(default_factory=lambda: None)

    def __post_init__(self):
        super().__post_init__()

        # Input validation
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.prediction_mode not in ["flow", "fast"]:
            raise ValueError(
                f"prediction_mode must be 'flow' or 'fast', got {self.prediction_mode}"
            )

        # Validate dof_config total doesn't exceed max_action_dim
        total_dof = sum(self.dof_config.values())
        if total_dof > self.max_action_dim:
            raise ValueError(
                f"Total DOF ({total_dof}) exceeds max_action_dim ({self.max_action_dim})"
            )

        # Sync prediction_mode with use_fast_tokenizer
        if self.use_fast_tokenizer:
            self.prediction_mode = "fast"
        else:
            self.prediction_mode = "flow"

    def get_train_config(self) -> dict:
        """
        Extract the complete train_config dictionary matching the YAML training configuration format.

        This method constructs the full train_config from WallXConfig fields, suitable for
        training scripts and Qwen2_5_VLMoEForAction.from_pretrained.

        Returns:
            dict: Complete training configuration matching YAML structure.
        """
        # Build customized_robot_config
        if self.enable_customized_robot_config and self.customized_robot_config:
            customized_robot_config = {
                "name": self.customized_robot_config.get("name", ""),
                "customized_dof_config": self.customized_robot_config.get(
                    "customized_dof_config", self.dof_config
                ),
                "customized_agent_pos_config": self.customized_robot_config.get(
                    "customized_agent_pos_config", self.agent_pos_config
                ),
            }
        else:
            customized_robot_config = {
                "name": self.data_config.get("lerobot_config", {}).get("repo_id", ""),
                "customized_dof_config": self.dof_config,
                "customized_agent_pos_config": self.agent_pos_config,
            }

        train_config = {
            # Model and paths configuration
            "log_name": self.log_name,
            "log_project": self.log_project,
            "model_type": self.model_type,
            "pretrained_wallx_path": self.pretrained_wallx_path,
            "save_path": self.save_path,
            "use_fast_tokenizer": self.use_fast_tokenizer,
            "action_tokenizer_path": self.action_tokenizer_path,

            # Profiling configuration
            "profile": self.profile,
            "profile_save_path": self.profile_save_path,
            "profile_wait_iters": self.profile_wait_iters,
            "profile_warmup_iters": self.profile_warmup_iters,
            "profile_active_iters": self.profile_active_iters,

            # Training hyperparameters
            "num_warmup_steps": self.num_warmup_steps,
            "num_training_steps": self.num_training_steps,
            "learning_rate": self.learning_rate,
            "min_lr": self.min_lr,
            "num_epoch": self.num_epoch,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "batch_size_per_gpu": self.batch_size_per_gpu,
            "padding_side": self.padding_side,
            "epoch_save_interval": self.epoch_save_interval,

            # Training optimization
            "FSDP2": self.fsdp2,
            "torch_compile": self.torch_compile,

            # Robot configuration
            "dof_config": self.dof_config,
            "agent_pos_config": self.agent_pos_config,

            # Normalization stats
            "norm_stats_path": self.norm_stats_path,

            # Customized robot config
            "enable_customized_robot_config": self.enable_customized_robot_config,
            "customized_robot_config": customized_robot_config,

            # Resume configuration
            "resume": self.resume_config,

            # Data configuration
            "data": self.data_config,
        }

        return train_config

    def get_dataload_config(self) -> dict:
        """
        Extract data loading configuration from config.

        Returns:
            dict: Data loading configuration for preprocessing.
        """
        return {
            "action_horizon": self.data_config.get("action_horizon", self.chunk_size),
            "train_test_split": self.data_config.get("train_test_split", 0.95),
            "split_seed": 42,
            "predict_action_keys": self.data_config.get("predict_action_keys", []),
            "obs_action_keys": self.data_config.get("obs_action_keys", []),
            "resolution": self.data_config.get("resolution", {}),
            "priority_order": None,
            "max_length": self.tokenizer_max_length,
        }

    def get_lerobot_config(self) -> dict:
        """
        Extract LeRobot dataset configuration.

        Returns:
            dict: LeRobot dataset configuration.
        """
        return self.data_config.get("lerobot_config", {})

    @classmethod
    def from_yaml_dict(cls, yaml_dict: dict) -> "WallXConfig":
        """
        Create a WallXConfig from a YAML configuration dictionary.

        Args:
            yaml_dict: Dictionary loaded from YAML training config file.

        Returns:
            WallXConfig instance with values from YAML.
        """
        config_kwargs = {}

        # Model and paths
        if "log_name" in yaml_dict:
            config_kwargs["log_name"] = yaml_dict["log_name"]
        if "log_project" in yaml_dict:
            config_kwargs["log_project"] = yaml_dict["log_project"]
        if "model_type" in yaml_dict:
            config_kwargs["model_type"] = yaml_dict["model_type"]
        if "pretrained_wallx_path" in yaml_dict:
            config_kwargs["pretrained_wallx_path"] = yaml_dict["pretrained_wallx_path"]
        if "save_path" in yaml_dict:
            config_kwargs["save_path"] = yaml_dict["save_path"]
        if "use_fast_tokenizer" in yaml_dict:
            config_kwargs["use_fast_tokenizer"] = yaml_dict["use_fast_tokenizer"]
        if "action_tokenizer_path" in yaml_dict:
            config_kwargs["action_tokenizer_path"] = yaml_dict["action_tokenizer_path"]

        # Profiling
        if "profile" in yaml_dict:
            config_kwargs["profile"] = yaml_dict["profile"]
        if "profile_save_path" in yaml_dict:
            config_kwargs["profile_save_path"] = yaml_dict["profile_save_path"]
        if "profile_wait_iters" in yaml_dict:
            config_kwargs["profile_wait_iters"] = yaml_dict["profile_wait_iters"]
        if "profile_warmup_iters" in yaml_dict:
            config_kwargs["profile_warmup_iters"] = yaml_dict["profile_warmup_iters"]
        if "profile_active_iters" in yaml_dict:
            config_kwargs["profile_active_iters"] = yaml_dict["profile_active_iters"]

        # Training hyperparameters
        if "num_warmup_steps" in yaml_dict:
            config_kwargs["num_warmup_steps"] = yaml_dict["num_warmup_steps"]
            config_kwargs["scheduler_warmup_steps"] = yaml_dict["num_warmup_steps"]
        if "num_training_steps" in yaml_dict:
            config_kwargs["num_training_steps"] = yaml_dict["num_training_steps"]
            config_kwargs["scheduler_decay_steps"] = yaml_dict["num_training_steps"]
        if "learning_rate" in yaml_dict:
            config_kwargs["learning_rate"] = yaml_dict["learning_rate"]
            config_kwargs["optimizer_lr"] = yaml_dict["learning_rate"]
        if "min_lr" in yaml_dict:
            config_kwargs["min_lr"] = yaml_dict["min_lr"]
            config_kwargs["scheduler_decay_lr"] = yaml_dict["min_lr"]
        if "num_epoch" in yaml_dict:
            config_kwargs["num_epoch"] = yaml_dict["num_epoch"]
        if "gradient_accumulation_steps" in yaml_dict:
            config_kwargs["gradient_accumulation_steps"] = yaml_dict["gradient_accumulation_steps"]
        if "batch_size_per_gpu" in yaml_dict:
            config_kwargs["batch_size_per_gpu"] = yaml_dict["batch_size_per_gpu"]
        if "padding_side" in yaml_dict:
            config_kwargs["padding_side"] = yaml_dict["padding_side"]
        if "epoch_save_interval" in yaml_dict:
            config_kwargs["epoch_save_interval"] = yaml_dict["epoch_save_interval"]

        # Training optimization
        if "FSDP2" in yaml_dict:
            config_kwargs["fsdp2"] = yaml_dict["FSDP2"]
        if "torch_compile" in yaml_dict:
            config_kwargs["torch_compile"] = yaml_dict["torch_compile"]

        # Robot configuration
        if "dof_config" in yaml_dict:
            config_kwargs["dof_config"] = yaml_dict["dof_config"]
        if "agent_pos_config" in yaml_dict:
            config_kwargs["agent_pos_config"] = yaml_dict["agent_pos_config"]

        # Normalization stats
        if "norm_stats_path" in yaml_dict:
            config_kwargs["norm_stats_path"] = yaml_dict["norm_stats_path"]

        # Customized robot config
        if "enable_customized_robot_config" in yaml_dict:
            config_kwargs["enable_customized_robot_config"] = yaml_dict["enable_customized_robot_config"]
        if "customized_robot_config" in yaml_dict:
            config_kwargs["customized_robot_config"] = yaml_dict["customized_robot_config"]

        # Resume config
        if "resume" in yaml_dict:
            config_kwargs["resume_config"] = yaml_dict["resume"]

        # Data configuration
        if "data" in yaml_dict:
            data = yaml_dict["data"]
            data_config = {
                "use_lerobot": data.get("use_lerobot", True),
                "action_horizon": data.get("action_horizon", 32),
                "train_test_split": data.get("train_test_split", 0.95),
                "obs_action_keys": data.get("obs_action_keys", []),
                "predict_action_keys": data.get("predict_action_keys", []),
                "resolution": data.get("resolution", {}),
            }
            if "lerobot_config" in data:
                data_config["lerobot_config"] = data["lerobot_config"]
            config_kwargs["data_config"] = data_config

            # Set chunk_size from action_horizon
            if "action_horizon" in data:
                config_kwargs["chunk_size"] = data["action_horizon"]
                config_kwargs["n_action_steps"] = data["action_horizon"]

        return cls(**config_kwargs)

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
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
