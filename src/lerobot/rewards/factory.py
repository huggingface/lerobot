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
from copy import copy
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs.rewards import RewardModelConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .classifier.configuration_classifier import RewardClassifierConfig
from .pretrained import PreTrainedRewardModel
from .robometer.configuration_robometer import RobometerConfig
from .sarm.configuration_sarm import SARMConfig
from .topreward.configuration_topreward import TOPRewardConfig

if TYPE_CHECKING:
    from .scoring import FrameScorer


def get_reward_model_class(name: str) -> type[PreTrainedRewardModel]:
    """
    Retrieves a reward model class by its registered name.

    Resolution is convention-based: the draccus-registered config class of ``name`` is
    looked up, its ``configuration_*`` module path is rewritten to ``modeling_*``, and the
    ``PreTrainedRewardModel`` subclass whose ``config_class`` is that config is returned.
    The modeling module is only imported at call time, keeping heavy optional dependencies
    lazy. This works for both built-in reward models and third-party lerobot plugins
    (anything registered via ``@RewardModelConfig.register_subclass``).

    Args:
        name: The registered name of the reward model (e.g. "sarm", "robometer").

    Returns:
        The reward model class corresponding to the given name.

    Raises:
        ValueError: If the reward model name is not registered or has no reward model class.
        ImportError: If the reward model's optional dependencies are not installed.
    """
    if name not in RewardModelConfig.get_known_choices():
        raise ValueError(
            f"Unknown reward model name '{name}'. "
            f"Available reward models: {RewardModelConfig.get_known_choices()}"
        )

    config_cls = RewardModelConfig.get_choice_class(name)

    module = _import_sibling_reward_module(config_cls, "modeling")
    if module is None:
        raise ValueError(f"Reward model class for '{name}' is not implemented.")
    for obj in vars(module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, PreTrainedRewardModel)
            and getattr(obj, "config_class", None) is config_cls
        ):
            return obj
    raise ValueError(
        f"Reward model class with 'config_class = {config_cls.__name__}' not found in '{module.__name__}'. "
        f"Reward models must expose a 'PreTrainedRewardModel' subclass with that 'config_class' "
        f"in the sibling 'modeling_*' module."
    )


def make_reward_model_config(reward_type: str, **kwargs: Any) -> RewardModelConfig:
    """
    Instantiates a reward model configuration object based on the reward type.

    This factory function simplifies the creation of reward model configuration objects
    by mapping a string identifier to the corresponding config class.

    Args:
        reward_type: A registered reward model type, such as
                     "reward_classifier", "sarm", "robometer" or "topreward".
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `RewardModelConfig` subclass.

    Raises:
        ValueError: If the `reward_type` is not recognized.
    """
    try:
        config_cls = RewardModelConfig.get_choice_class(reward_type)
    except Exception as e:
        raise ValueError(f"Reward model type '{reward_type}' is not available.") from e
    return config_cls(**kwargs)


def make_reward_model(
    cfg: RewardModelConfig, pretrained_path: str | Path | None = None, **kwargs
) -> PreTrainedRewardModel:
    """
    Instantiate a reward model from its configuration.

    Args:
        cfg: The configuration for the reward model to be created. If
             `cfg.pretrained_path` is set, the model will be loaded with weights
             from that path.
        pretrained_path: Load the weights from here instead of `cfg.pretrained_path`, which keeps
            naming the model this one was fine-tuned from once it is built. Used when resuming.
        **kwargs: Additional keyword arguments forwarded to the model constructor
            (e.g., ``dataset_stats``, ``dataset_meta``).

    Returns:
        An instantiated and device-placed reward model.
    """
    reward_cls = get_reward_model_class(cfg.type)

    kwargs["config"] = cfg
    # As in `make_policy`: the weight source while building, the parent again afterwards.
    parent_path = cfg.pretrained_path
    if pretrained_path is not None:
        cfg.pretrained_path = str(pretrained_path)

    if cfg.pretrained_path:
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        kwargs["revision"] = cfg.pretrained_revision
        reward_model = reward_cls.from_pretrained(**kwargs)
    else:
        reward_model = reward_cls(**kwargs)
    if pretrained_path is not None:
        cfg.pretrained_path = parent_path
        reward_model.config.pretrained_path = parent_path

    reward_model.to(cfg.device)
    return reward_model


def make_frame_scorer(model: PreTrainedRewardModel, *, batch_size: int = 32) -> "FrameScorer":
    """
    Create the standard offline frame scorer for a reward model.

    The scorer is built by the ``make_<type>_frame_scorer`` function of the model's
    ``scoring_<type>`` module, found next to its configuration module. Model-specific
    scoring options, such as the frames sampled for each prediction, are read from the
    model's config.

    Args:
        model: The reward model the scorer will run.
        batch_size: Number of predictions computed together. Scorers that cannot batch
            their inputs ignore it.

    Returns:
        A frame scorer that can be passed to :func:`lerobot.rewards.scoring.score_dataset`.

    Raises:
        ValueError: If no frame scorer is implemented for the model's type.
    """
    return _make_frame_scorer_from_reward_model_config(
        config=model.config, model=model, batch_size=batch_size
    )


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
    trust_remote_code = kwargs.get("trust_remote_code", reward_cfg.trust_remote_code)
    if trust_remote_code != reward_cfg.trust_remote_code:
        reward_cfg = copy(reward_cfg)
        reward_cfg.trust_remote_code = trust_remote_code

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
    elif isinstance(reward_cfg, RobometerConfig):
        from lerobot.rewards.robometer.processor_robometer import make_robometer_pre_post_processors

        return make_robometer_pre_post_processors(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(reward_cfg, TOPRewardConfig):
        from lerobot.rewards.topreward.processor_topreward import make_topreward_pre_post_processors

        return make_topreward_pre_post_processors(
            config=reward_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
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


def _import_sibling_reward_module(config_cls: type[RewardModelConfig], prefix: str) -> ModuleType | None:
    """Import a config class' sibling ``{prefix}_*`` module, or None when that module does not exist."""
    module_path = config_cls.__module__.replace("configuration_", f"{prefix}_")
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name == module_path:
            # The sibling module itself does not exist for this reward model type. A missing optional
            # dependency inside an existing module propagates unchanged instead, so its
            # actionable install hint stays visible.
            return None
        raise


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


def _make_frame_scorer_from_reward_model_config(
    config: RewardModelConfig,
    model: PreTrainedRewardModel,
    batch_size: int,
) -> "FrameScorer":
    """Create a frame scorer from a reward model configuration using dynamic imports.

    This is used as a helper function to import frame-scorer factories from built-in
    reward models and 3rd party lerobot reward model plugins.

    Args:
        config: The reward model configuration object.
        model: The reward model the scorer will run.
        batch_size: Number of predictions computed together.

    Returns:
        The frame scorer returned by ``make_<type>_frame_scorer``.

    Raises:
        ValueError: If the scoring module or its factory function does not exist.
    """
    module = _import_sibling_reward_module(type(config), "scoring")
    function = getattr(module, f"make_{config.type}_frame_scorer", None)
    if function is None:
        raise ValueError(f"Frame scorer for reward model type '{config.type}' is not implemented.")
    return function(model=model, batch_size=batch_size)
