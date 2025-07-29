#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging

import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.processor.pipeline import EnvTransition, RobotProcessor, TransitionIndex


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_processor(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs,
) -> tuple[RobotProcessor, RobotProcessor]:
    """Make a processor instance for a given policy type.

    This function creates the appropriate processor configuration based on the policy type.
    Each policy type has its own processor with specific preprocessing steps.

    Args:
        policy_cfg: The config of the policy to create a processor for (e.g., "act", "diffusion", etc.)
        pretrained_path: Optional path to load a pretrained processor from. If provided, loads
            the processor from this path instead of creating a new one.
        **kwargs: Additional keyword arguments passed to the processor creation.

    Returns:
        Tuple of (input_processor, output_processor) for the policy.

    Raises:
        NotImplementedError: If the policy type doesn't have a processor implemented.
    """
    if pretrained_path:
        # Load a pretrained processor
        # TODO(azouitine): Handle this case.
        raise NotImplementedError("Loading a pretrained processor is not implemented.")

    # Create a new processor based on policy type
    if policy_cfg.type == "tdmpc":
        from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_processor

        processors = make_tdmpc_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "diffusion":
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_processor

        processors = make_diffusion_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "act":
        from lerobot.policies.act.processor_act import make_act_processor

        processors = make_act_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "vqbet":
        from lerobot.policies.vqbet.processor_vqbet import make_vqbet_processor

        processors = make_vqbet_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "pi0":
        from lerobot.policies.pi0.processor_pi0 import make_pi0_processor

        processors = make_pi0_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "pi0fast":
        from lerobot.policies.pi0fast.processor_pi0fast import make_pi0fast_processor

        processors = make_pi0fast_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "sac":
        from lerobot.policies.sac.processor_sac import make_sac_processor

        processors = make_sac_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "reward_classifier":
        from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor

        processors = make_classifier_processor(policy_cfg, **kwargs)

    elif policy_cfg.type == "smolvla":
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_processor

        processors = make_smolvla_processor(policy_cfg, **kwargs)

    else:
        raise NotImplementedError(f"Processor for policy type '{policy_cfg.type}' is not implemented.")

    # Helper hook function to detect NaNs in observation
    def nan_detection_hook(step_idx: int, transition: EnvTransition) -> None:
        observation = transition[TransitionIndex.OBSERVATION]
        if observation is not None:
            for key, value in observation.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    logging.warning(f"NaN detected in observation key '{key}' after step {step_idx}: {value}")

    # Attach the hook to all returned processors
    if isinstance(processors, RobotProcessor):
        processors = (processors,)  # Wrap single processor in tuple for consistency
    for processor in processors:
        processor.register_after_step_hook(nan_detection_hook)

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
