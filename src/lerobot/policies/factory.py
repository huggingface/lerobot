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
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import torch

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.envs import EnvConfig, env_to_policy_features
from lerobot.lerobot_types import PolicyAction
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.context import ProcessorBuildContext, apply_checkpoint_rename_map
from lerobot.processor.features import apply_policy_features
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.utils.import_utils import _peft_available, require_package

from .pretrained import PreTrainedPolicy
from .utils import validate_visual_features_consistency

if TYPE_CHECKING or _peft_available:
    from peft import PeftConfig, PeftModel
else:
    PeftConfig = None
    PeftModel = None


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
        context: The preferred way to pass per-run build inputs. Supersedes `dataset_stats`,
            `dataset_meta` and the two override dicts.
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: Deprecated. Step-level overrides; values the policy config now owns
            (`device`, `rename_map`) are applied to it, the rest are ignored with a warning.
        postprocessor_overrides: Deprecated. See `preprocessor_overrides`.
        dataset_stats: Dataset statistics for normalization. Prefer `context.dataset_stats`.
        dataset_meta: Dataset metadata. Prefer `context.dataset_meta`.
    """

    context: ProcessorBuildContext | None
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
    Create pre- and post-processor pipelines for a given policy.

    There is one construction path: the policy's own factory
    (e.g. `make_tdmpc_pre_post_processors`, resolved by naming convention) builds the pipelines from
    `policy_cfg`. When `pretrained_path` is given, the checkpoint contributes only its *tensor state*
    — normalization statistics and the like — which is loaded into those freshly built steps.

    That split is the authority rule for the whole processor system:

    - **Structure and shape belong to the code.** A `--policy.*` flag reconfigures a step even when
      the checkpoint predates it, and a policy whose pipeline reshapes tensors internally (EVO1
      padding state to `max_state_dim`) keeps its own shape instead of having dataset-derived widths
      forced onto it. Deserializing structure is what previously required per-policy reconciliation
      after loading.
    - **Statistics belong to whoever supplied them.** Passing `context.dataset_stats` makes the
      dataset authoritative (the finetune case); omitting it keeps the checkpoint's saved stats (eval
      and resume). `NormalizerProcessorStep` already implements exactly this precedence via
      `_stats_explicitly_provided`.

    To deserialize a saved pipeline wholesale instead — structure included — use
    `PolicyProcessorPipeline.from_pretrained` directly.

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: Optional checkpoint whose step state should be loaded into the pipelines.
        pretrained_revision: Optional Hub revision for `pretrained_path`.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`. Prefer passing `context=ProcessorBuildContext(...)`; the
            `preprocessor_overrides`/`postprocessor_overrides` dicts are deprecated.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        ValueError: If no processor factory exists for the given policy configuration type, or if the
            checkpoint carries state for a step the policy config does not build.
    """
    context = kwargs.get("context")
    if context is None:
        context = ProcessorBuildContext.from_legacy_kwargs(dict(kwargs), policy_cfg)
    pre_filename = kwargs.get("preprocessor_config_filename") or f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    post_filename = kwargs.get("postprocessor_config_filename") or f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
    context = replace(
        context,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        pretrained_revision=pretrained_revision,
    )

    if pretrained_path and _uses_legacy_pretrained_loader(policy_cfg):
        # GR00T has not been migrated to the rule above and still deserializes its saved pipelines,
        # then repairs them. See `_uses_legacy_pretrained_loader`.
        from .groot.processor_groot import make_groot_pre_post_processors_from_pretrained

        return make_groot_pre_post_processors_from_pretrained(
            config=policy_cfg,
            pretrained_path=pretrained_path,
            revision=pretrained_revision,
            dataset_stats=context.dataset_stats,
            dataset_meta=context.dataset_meta,
            preprocessor_overrides=kwargs.get("preprocessor_overrides"),
            postprocessor_overrides=kwargs.get("postprocessor_overrides"),
            preprocessor_config_filename=pre_filename,
            postprocessor_config_filename=post_filename,
        )

    if pretrained_path:
        # Read the saved preprocessor before building, only to recover a rename map the caller did
        # not supply. Everything else about the pipeline comes from the config.
        apply_checkpoint_rename_map(
            policy_cfg, _peek_pipeline_config(pretrained_path, pre_filename, pretrained_revision)
        )

    preprocessor, postprocessor = _make_processors_from_policy_config(
        config=policy_cfg,
        context=context,
    )

    if pretrained_path:
        # Structure and shape came from the config above; only tensors come from the checkpoint.
        preprocessor.load_pretrained_state(pretrained_path, pre_filename, revision=pretrained_revision)
        postprocessor.load_pretrained_state(pretrained_path, post_filename, revision=pretrained_revision)

    return preprocessor, postprocessor


def _uses_legacy_pretrained_loader(policy_cfg: PreTrainedConfig) -> bool:
    """Whether this policy still deserializes its saved pipelines instead of rebuilding them.

    Only GR00T does. Rebuilding its pipelines from the config is not yet possible: two of its steps
    are stateful, and for checkpoints converted from a raw N1.7 release the values those steps need
    (``raw_stats``, ``modality_config``, ``video_modality_keys``) exist *only* inside the serialized
    pipeline JSON — there is no sidecar and no config field to rebuild them from. Migrating GR00T
    therefore means teaching its config to carry those values plus a reader for checkpoints that
    predate them, and that cannot be validated without the gated backbone its tests download.

    Resolved by attribute rather than by importing `GrootConfig`, so the check stays lazy and this
    module keeps its no-eager-policy-imports property.
    """
    return policy_cfg.type == "groot"


def _peek_pipeline_config(
    pretrained_path: str, config_filename: str, revision: str | None
) -> dict[str, Any] | None:
    """Load a checkpoint's serialized pipeline config, or None if it cannot be read.

    Best-effort by design: this only feeds the rename-map carry-over, and a checkpoint without a
    readable processor config is handled with a proper error by `load_pretrained_state` later.
    """
    try:
        loaded_config, _ = PolicyProcessorPipeline._load_config(
            str(pretrained_path),
            config_filename,
            {
                "force_download": False,
                "resume_download": None,
                "proxies": None,
                "token": None,
                "cache_dir": None,
                "local_files_only": False,
                "revision": revision,
            },
        )
        return loaded_config
    except Exception:  # noqa: BLE001 - advisory read; the authoritative error comes from loading state
        return None


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
    defer_weight_load: bool = False,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg (PreTrainedConfig): The configuration for the policy to be created. If
            `cfg.pretrained_path` is set, the policy will be loaded with weights from that path.
        ds_meta (LeRobotDatasetMetadata | None): Dataset metadata used to infer feature shapes and
            types. Also provides statistics for normalization layers.
        env_cfg (EnvConfig | None): Environment configuration used to infer feature shapes and
            types. One of `ds_meta` or `env_cfg` must be provided.
        rename_map (dict[str, str] | None): Optional mapping of dataset or environment feature
            keys to match expected policy feature names (e.g., `"left"` → `"camera1"`).
        defer_weight_load (bool): Build the exact policy `from_pretrained` would build — same
            config resolution, same stats-derived buffers, same device placement and eval mode —
            but skip the safetensors weight load. Used when resuming from a DCP checkpoint, whose
            sharded weights stream in after `accelerator.prepare()` (the distributed checkpoint
            engine overwrites the random init).

    Returns:
        PreTrainedPolicy: An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend combination
            (e.g., VQBeT with 'mps').
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

    if rename_map:
        features = {rename_map.get(key, key): feature for key, feature in features.items()}
        # Record it on the config so a processor pipeline built from this config later renames the
        # same keys, without the caller having to pass the map a second time.
        cfg.rename_map = dict(rename_map)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}

    # Store action feature names for relative_exclude_joints support
    if ds_meta is not None:
        raw_action_feature = next(
            (
                feature
                for raw_key, feature in ds_meta.features.items()
                if (rename_map or {}).get(raw_key, raw_key) == ACTION
            ),
            None,
        )
        action_names = raw_action_feature.get("names") if raw_action_feature is not None else None
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
        if defer_weight_load:
            # Same construction path as from_pretrained (config already resolved from the
            # checkpoint by the caller; dataset_stats/dataset_meta kwargs identical), minus the
            # weight load — parity by construction.
            policy = policy_cls(**kwargs)
            policy.eval()
        else:
            # Load a pretrained policy and override the config if needed (for example, if there
            # are inference-time hyperparameters that we want to vary).
            kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            kwargs["revision"] = cfg.pretrained_revision
            policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
        # of the adapter and the adapter's config contains the path to the base policy. So we need the
        # adapter config first, then load the correct policy and then apply PEFT.
        require_package("peft", extra="peft")

        logging.info("Loading policy's PEFT adapter.")

        peft_pretrained_path = str(cfg.pretrained_path)
        peft_config = PeftConfig.from_pretrained(
            peft_pretrained_path,
            revision=cfg.pretrained_revision,
        )

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            # This means that there's a bug or we trained a policy from scratch using PEFT.
            # It is more likely that this is a bug so we'll raise an error.
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        kwargs["revision"] = peft_config.revision
        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(
            policy,
            peft_pretrained_path,
            config=peft_config,
            revision=cfg.pretrained_revision,
            is_trainable=True,
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
    context: ProcessorBuildContext,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    Resolves ``make_{type}_pre_post_processors`` from the policy's ``processor_*`` module
    by naming convention. Works for built-in policies and 3rd party lerobot plugins.

    Two factory signatures are supported, distinguished by parameter name:

    - ``(config, context)`` — the current contract.
    - ``(config, dataset_stats=None[, dataset_meta=None])`` — the older one, still used by most
      policies. Those values are unpacked from the context, so such factories need no edit. Anything
      else a context carries (``training``, ``pretrained_path``) is unavailable to them, which is
      fine: nothing that ignores it needs it.

    Args:
        config: The policy configuration object.
        context: Per-run build inputs.

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
    parameters = inspect.signature(function).parameters
    if "context" in parameters:
        preprocessor, postprocessor = function(config, context=context)
    else:
        call_kwargs: dict[str, Any] = {"dataset_stats": context.dataset_stats}
        if "dataset_meta" in parameters:
            call_kwargs["dataset_meta"] = context.dataset_meta
        preprocessor, postprocessor = function(config, **call_kwargs)

    apply_policy_features(config, context, preprocessor, postprocessor)
    return preprocessor, postprocessor
