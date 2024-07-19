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
import inspect
import logging

from omegaconf import DictConfig, OmegaConf

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import get_safe_torch_device


def _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg):
    expected_kwargs = set(inspect.signature(policy_cfg_class).parameters)
    if not set(hydra_cfg.policy).issuperset(expected_kwargs):
        logging.warning(
            f"Hydra config is missing arguments: {set(expected_kwargs).difference(hydra_cfg.policy)}"
        )

    # OmegaConf.to_container returns lists where sequences are found, but our dataclasses use tuples to avoid
    # issues with mutable defaults. This filter changes all lists to tuples.
    def list_to_tuple(item):
        return tuple(item) if isinstance(item, list) else item

    policy_cfg = policy_cfg_class(
        **{
            k: list_to_tuple(v)
            for k, v in OmegaConf.to_container(hydra_cfg.policy, resolve=True).items()
            if k in expected_kwargs
        }
    )
    return policy_cfg


def get_policy_and_config_classes(name: str) -> tuple[Policy, object]:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy, TDMPCConfig
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy, DiffusionConfig
    elif name == "act":
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy, ACTConfig
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy, VQBeTConfig
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy(
    hydra_cfg: DictConfig, pretrained_policy_name_or_path: str | None = None, dataset_stats=None
) -> Policy:
    """Make an instance of a policy class.

    Args:
        hydra_cfg: A parsed Hydra configuration (see scripts). If `pretrained_policy_name_or_path` is
            provided, only `hydra_cfg.policy.name` is used while everything else is ignored.
        pretrained_policy_name_or_path: Either the repo ID of a model hosted on the Hub or a path to a
            directory containing weights saved using `Policy.save_pretrained`. Note that providing this
            argument overrides everything in `hydra_cfg.policy` apart from `hydra_cfg.policy.name`.
        dataset_stats: Dataset statistics to use for (un)normalization of inputs/outputs in the policy. Must
            be provided when initializing a new policy, and must not be provided when loading a pretrained
            policy. Therefore, this argument is mutually exclusive with `pretrained_policy_name_or_path`.
    """
    if not (pretrained_policy_name_or_path is None) ^ (dataset_stats is None):
        raise ValueError(
            "Exactly one of `pretrained_policy_name_or_path` and `dataset_stats` must be provided."
        )

    policy_cls, policy_cfg_class = get_policy_and_config_classes(hydra_cfg.policy.name)

    policy_cfg = _policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg)
    if pretrained_policy_name_or_path is None:
        # Make a fresh policy.
        policy = policy_cls(policy_cfg, dataset_stats)
    else:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        # TODO(alexander-soare): This hack makes use of huggingface_hub's tooling to load the policy with,
        # pretrained weights which are then loaded into a fresh policy with the desired config. This PR in
        # huggingface_hub should make it possible to avoid the hack:
        # https://github.com/huggingface/huggingface_hub/pull/2274.
        policy = policy_cls(policy_cfg)
        policy.load_state_dict(policy_cls.from_pretrained(pretrained_policy_name_or_path).state_dict())

    policy.to(get_safe_torch_device(hydra_cfg.device))

    return policy
