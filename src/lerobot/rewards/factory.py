#!/usr/bin/env python

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

import importlib
import logging
from typing import Any

import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .classifier.configuration_classifier import RewardClassifierConfig
from .pretrained import PreTrainedRewardModel
from .sarm.configuration_sarm import SARMConfig


def get_reward_model_class(name: str) -> type[PreTrainedRewardModel]:
    """
    Retrieves a reward model class by its registered name.

    This function uses dynamic imports to avoid loading all reward model classes into
    memory at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the reward model. Supported names are "reward_classifier",
              "sarm".

    Returns:
        The reward model class corresponding to the given name.

    Raises:
        ValueError: If the reward model name is not recognized.
    """
    if name == "reward_classifier":
        from lerobot.rewards.classifier.modeling_classifier import Classifier

        return Classifier
    elif name == "sarm":
        from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel

        return SARMRewardModel
    else:
        try:
            return _get_reward_model_cls_from_name(name=name)
        except Exception as e:
            raise ValueError(f"Reward model type '{name}' is not available.") from e


def make_reward_model_config(reward_type: str, **kwargs) -> RewardModelConfig:
    """
    Instantiates a reward model configuration object based on the reward type.

    This factory function simplifies the creation of reward model configuration objects
    by mapping a string identifier to the corresponding config class.

    Args:
        reward_type: The type of the reward model. Supported types include
                     "reward_classifier", "sarm".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `RewardModelConfig` subclass.

    Raises:
        ValueError: If the `reward_type` is not recognized.
    """
    if reward_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    elif reward_type == "sarm":
        return SARMConfig(**kwargs)
    else:
        try:
            config_cls = RewardModelConfig.get_choice_class(reward_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Reward model type '{reward_type}' is not available.") from e


def make_reward_model(cfg: RewardModelConfig, **kwargs) -> PreTrainedRewardModel:
    """
    Instantiate a reward model from its configuration.

    Args:
        cfg: The configuration for the reward model to be created. If
             `cfg.pretrained_path` is set, the model will be loaded with weights
             from that path.
        **kwargs: Additional keyword arguments forwarded to the model constructor
            (e.g., ``dataset_stats``, ``dataset_meta``).

    Returns:
        An instantiated and device-placed reward model.
    """
    reward_cls = get_reward_model_class(cfg.type)

    kwargs["config"] = cfg

    if cfg.pretrained_path:
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        reward_model = reward_cls.from_pretrained(**kwargs)
    else:
        reward_model = reward_cls(**kwargs)

    reward_model.to(cfg.device)
    assert isinstance(reward_model, torch.nn.Module)

    return reward_model


def make_reward_pre_post_processors(
    reward_cfg: RewardModelConfig,
    **kwargs,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create pre- and post-processor pipelines for a given reward model.

    Each reward model type has a dedicated factory function for its processors.

    Args:
        reward_cfg: The configuration of the reward model for which to create processors.
        **kwargs: Additional keyword arguments passed to the processor factory
            (e.g., ``dataset_stats``, ``dataset_meta``).

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        ValueError: If a processor factory is not implemented for the given reward
            model configuration type.
    """
    # Create a new processor based on reward model type
    if isinstance(reward_cfg, RewardClassifierConfig):
        from lerobot.rewards.classifier.processor_classifier import make_classifier_processor

        return make_classifier_processor(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(reward_cfg, SARMConfig):
        from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors

        return make_sarm_pre_post_processors(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
            dataset_meta=kwargs.get("dataset_meta"),
        )

    else:
        try:
            processors = _make_processors_from_reward_model_config(
                config=reward_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(
                f"Processor for reward model type '{reward_cfg.type}' is not implemented."
            ) from e
        return processors


def _get_reward_model_cls_from_name(name: str) -> type[PreTrainedRewardModel]:
    """Get reward model class from its registered name using dynamic imports.

    This is used as a helper function to import reward models from 3rd party lerobot
    plugins.

    Args:
        name: The name of the reward model.

    Returns:
        The reward model class corresponding to the given name.
    """
    if name not in RewardModelConfig.get_known_choices():
        raise ValueError(
            f"Unknown reward model name '{name}'. "
            f"Available reward models: {RewardModelConfig.get_known_choices()}"
        )

    config_cls = RewardModelConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention. "
            f"Make sure it ends with 'Config'!"
        )

    cls_name = model_name + "RewardModel"
    module_path = config_cls.__module__.replace("configuration_", "modeling_")

    module = importlib.import_module(module_path)
    reward_cls = getattr(module, cls_name)
    return reward_cls


def _make_processors_from_reward_model_config(
    config: RewardModelConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a reward model configuration using dynamic imports.

    This is used as a helper function to import processor factories from 3rd party
    lerobot reward model plugins.

    Args:
        config: The reward model configuration object.
        dataset_stats: Dataset statistics for normalization.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """
    reward_type = config.type
    function_name = f"make_{reward_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace("configuration_", "processor_")
    logging.debug(
        f"Instantiating reward pre/post processors using function '{function_name}' "
        f"from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
