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

from __future__ import annotations

import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import torch

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.envs import EnvConfig, env_to_policy_features
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.types import PolicyAction
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.feature_utils import dataset_to_policy_features

from .evo1.configuration_evo1 import Evo1Config
from .groot.configuration_groot import GrootConfig
from .pretrained import PreTrainedPolicy
from .utils import validate_visual_features_consistency


def _reconnect_relative_absolute_steps(
    preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
) -> None:
    """Wire AbsoluteActionsProcessorStep.relative_step to the RelativeActionsProcessorStep after deserialization.

    After a policy is loaded from disk, the preprocessor and postprocessor are reconstructed
    independently from their configs. AbsoluteActionsProcessorStep needs a live reference to
    the RelativeActionsProcessorStep so it can read the cached state at inference time.
    That reference is not serializable, so we re-establish it here after loading.
    """
    relative_step = next((s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep)), None)
    if relative_step is None:
        return
    for step in postprocessor.steps:
        if isinstance(step, AbsoluteActionsProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    Resolution is convention-based: the draccus-registered config class of ``name`` is
    looked up, its ``configuration_*`` module path is rewritten to ``modeling_*``, and
    the ``<X>Policy`` class is imported from there. The modeling module is only imported
    at call time, keeping heavy optional dependencies lazy. This works for both built-in
    policies and third-party lerobot plugins (anything registered via
    ``@PreTrainedConfig.register_subclass``).

    Args:
        name: The registered name of the policy (e.g. "act", "diffusion", "pi0").
    Returns:
        The policy class corresponding to the given name.

    Raises:
        ValueError: If the policy name is not registered.
        ImportError: If the policy's optional dependencies are not installed.
    """
    return _get_policy_cls_from_policy_name(name=name)


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The registered type of the policy (any name registered via
                     ``@PreTrainedConfig.register_subclass``, e.g. "act", "diffusion", "pi0").
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    try:
        config_cls = PreTrainedConfig.get_choice_class(policy_type)
    except Exception as e:
        raise ValueError(f"Policy type '{policy_type}' is not available.") from e
    return config_cls(**kwargs)


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None
    dataset_meta: Any | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    pretrained_revision: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        ValueError: If no processor factory exists for the given policy configuration type.
    """
    if pretrained_path:
        if isinstance(policy_cfg, GrootConfig):
            from .groot.processor_groot import make_groot_pre_post_processors_from_pretrained

            return make_groot_pre_post_processors_from_pretrained(
                config=policy_cfg,
                pretrained_path=pretrained_path,
                dataset_stats=kwargs.get("dataset_stats"),
                dataset_meta=kwargs.get("dataset_meta"),
                preprocessor_overrides=kwargs.get("preprocessor_overrides"),
                postprocessor_overrides=kwargs.get("postprocessor_overrides"),
                preprocessor_config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                postprocessor_config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
            )

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
            revision=pretrained_revision,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            revision=pretrained_revision,
        )
        _reconnect_relative_absolute_steps(preprocessor, postprocessor)
        if isinstance(policy_cfg, Evo1Config):
            from .evo1.processor_evo1 import reconcile_evo1_processors

            preprocessor, postprocessor = reconcile_evo1_processors(
                policy_cfg,
                preprocessor,
                postprocessor,
            )
        return preprocessor, postprocessor

    # Create new processors from the policy config, resolving the per-policy factory
    # function by naming convention (lazy import keeps optional dependencies optional).
    return _make_processors_from_policy_config(
        config=policy_cfg,
        dataset_stats=kwargs.get("dataset_stats"),
        dataset_meta=kwargs.get("dataset_meta"),
    )


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
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
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    # Store action feature names for relative_exclude_joints support
    if ds_meta is not None and hasattr(cfg, "action_feature_names"):
        action_names = ds_meta.features.get(ACTION, {}).get("names")
        if action_names is not None:
            cfg.action_feature_names = list(action_names)
    if ds_meta is not None:
        set_dataset_feature_metadata = getattr(cfg, "set_dataset_feature_metadata", None)
        if callable(set_dataset_feature_metadata):
            set_dataset_feature_metadata(ds_meta.features)
        cfg._runtime_dataset_meta = ds_meta

    kwargs["config"] = cfg

    # Pass dataset_stats to the policy if available (needed for some policies like SARM)
    if ds_meta is not None and hasattr(ds_meta, "stats"):
        kwargs["dataset_stats"] = ds_meta.stats

    if ds_meta is not None:
        kwargs["dataset_meta"] = ds_meta

    if not cfg.pretrained_path and cfg.use_peft:
        raise ValueError(
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        )

    if cfg.pretrained_path and not cfg.use_peft:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        kwargs["revision"] = cfg.pretrained_revision
        policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
        # of the adapter and the adapter's config contains the path to the base policy. So we need the
        # adapter config first, then load the correct policy and then apply PEFT.
        from peft import PeftConfig, PeftModel

        logging.info("Loading policy's PEFT adapter.")

        peft_pretrained_path = str(cfg.pretrained_path)
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            # This means that there's a bug or we trained a policy from scratch using PEFT.
            # It is more likely that this is a bug so we'll raise an error.
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(
            policy, peft_pretrained_path, config=peft_config, is_trainable=True
        )

    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedPolicy]:
    """Get policy class from its registered name using dynamic imports.

    Works for built-in policies and 3rd party lerobot plugins alike: the config class
    registered under ``name`` is resolved via the draccus ChoiceRegistry, and the policy
    class is imported from the sibling ``modeling_*`` module by naming convention.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name == module_path:
            # The modeling_* module itself does not exist for this policy type. A missing
            # optional dependency inside an existing module propagates unchanged instead,
            # so its actionable install hint stays visible.
            raise ValueError(f"Policy class for '{name}' is not implemented.") from e
        raise
    policy_cls = getattr(module, cls_name, None)
    if policy_cls is None:
        raise ValueError(
            f"Policy class '{cls_name}' not found in '{module_path}'. "
            f"Policies must expose '<Name>Policy' in the sibling 'modeling_*' module by naming convention."
        )
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    Resolves ``make_{type}_pre_post_processors`` from the policy's ``processor_*`` module
    by naming convention. Works for built-in policies and 3rd party lerobot plugins.

    Args:
        config: The policy configuration object.
        dataset_stats: Dataset statistics for normalization.
        dataset_meta: Dataset metadata, forwarded only to factories that declare a
            ``dataset_meta`` parameter (e.g. groot, molmoact2).
    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )  # e.g., configuration_diffusion -> processor_diffusion
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name == module_path:
            # The processor_* module itself does not exist for this policy type. A missing
            # optional dependency inside an existing module propagates unchanged instead,
            # so its actionable install hint stays visible.
            raise ValueError(f"Processor for policy type '{policy_type}' is not implemented.") from e
        raise
    function = getattr(module, function_name, None)
    if function is None:
        raise ValueError(f"Processor for policy type '{policy_type}' is not implemented.")
    call_kwargs: dict[str, Any] = {"dataset_stats": dataset_stats}
    if "dataset_meta" in inspect.signature(function).parameters:
        call_kwargs["dataset_meta"] = dataset_meta
    return function(config, **call_kwargs)
