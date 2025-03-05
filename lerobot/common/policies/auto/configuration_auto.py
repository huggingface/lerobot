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

from collections import OrderedDict
import importlib
from lerobot.configs.policies import PreTrainedConfig
from typing import Type, Union, Dict, Optional
from pathlib import Path

POLICY_CONFIG_NAMES_MAPPING = OrderedDict(
    [
        ("act", "ACTConfig"),
    ]
)

POLICY_NAMES_MAPPING = OrderedDict(
    [
        ("act", "ACTPolicy"),
    ]
)

def model_type_to_module_name(model_type: str) -> str:
    """Convert model type to module name format."""
    return model_type.replace("-", "_")

def find_policy_type_from_config(config_dict: dict) -> str:
    """Find the policy type from a config dictionary."""
    if "type" in config_dict:
        return config_dict["type"]
    
    # Fallback: try to infer from class name
    if "policy_class" in config_dict:
        for policy_type, class_name in POLICY_CONFIG_NAMES_MAPPING.items():
            if class_name in config_dict["policy_class"]:
                return policy_type
    
    raise ValueError(
        "Could not determine policy type from config. "
        "Config must contain either 'type' or 'policy_class' field."
    )

class _LazyPolicyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
            
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        
        # Try standard import path first
        try:
            if key not in self._modules:
                self._modules[key] = importlib.import_module(
                    f"lerobot.common.policies.{module_name}.configuration_{module_name}"
                )
            return getattr(self._modules[key], value)
        except (ImportError, AttributeError):
            # Try fallback paths
            for import_path in [
                f"lerobot.policies.{module_name}",
                f"lerobot.common.policies.{module_name}",
            ]:
                try:
                    self._modules[key] = importlib.import_module(import_path)
                    if hasattr(self._modules[key], value):
                        return getattr(self._modules[key], value)
                except ImportError:
                    continue
                    
        raise ImportError(
            f"Could not find configuration class {value} for policy type {key}"
        )

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Policy Config, pick another name.")
        self._extra_content[key] = value

POLICY_CONFIG_MAPPING = _LazyPolicyConfigMapping(POLICY_CONFIG_NAMES_MAPPING)

class AutoPolicyConfig:
    """
    Factory class for automatically loading policy configurations.
    
    This class provides methods to:
    - Load pre-trained policy configurations from local files or the Hub
    - Register new policy types dynamically
    - Create policy configurations for specific policy types
    """
    
    def __init__(self):
        raise EnvironmentError(
            "AutoPolicyConfig is designed to be instantiated "
            "using the `AutoPolicyConfig.from_pretrained(TODO)` method."
        )
    
    @classmethod
    def for_policy(
        cls, 
        policy_type: str, 
        *args, 
        **kwargs
    ) -> PreTrainedConfig:
        """Create a new configuration instance for the specified policy type."""
        if policy_type in POLICY_CONFIG_MAPPING:
            config_class = POLICY_CONFIG_MAPPING[policy_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized policy identifier: {policy_type}. Should contain one of {', '.join(POLICY_CONFIG_MAPPING.keys())}"
        )
    
    @staticmethod
    def register(policy_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            policy_type (`str`): The policy type like "act" or "pi0".
            config ([`PreTrainedConfig`]): The config to register.
        """
        if issubclass(config, PreTrainedConfig) and config.policy_type != policy_type:
            raise ValueError(
                "The config you are passing has a `policy_type` attribute that is not consistent with the policy type "
                f"you passed (config has {config.policy_type} and you passed {policy_type}. Fix one of those so they "
                "match!"
            )
        POLICY_CONFIG_MAPPING.register(policy_type, config, exist_ok=exist_ok)

    @staticmethod
    def register_policy_type(policy_type: str, config_class: str, policy_class: str) -> None:
        """
        Register a new policy type with its configuration and policy classes.
        
        Args:
            policy_type (`str`): The policy type identifier (e.g., 'act')
            config_class (`str`): Name of the configuration class
            policy_class (`str`): Name of the policy class
        """
        if policy_type in POLICY_CONFIG_NAMES_MAPPING:
            raise ValueError(f"Policy type {policy_type} is already registered")
            
        POLICY_CONFIG_NAMES_MAPPING[policy_type] = config_class
        POLICY_NAMES_MAPPING[policy_type] = policy_class

    @classmethod
    def from_pretrained(
        cls,
        pretrained_policy_config_name_or_path: Union[str, Path],
        **kwargs
    ) -> PreTrainedConfig:
        """
        Instantiate a PreTrainedConfig from a pre-trained policy configuration.

        Args:
            pretrained_policy_config_name_or_path (`str` or `Path`):
                Can be either:
                    - A string with the `policy_type` of a pre-trained policy configuration listed on 
                      the Hub or locally (e.g., 'act')
                    - A path to a `directory` containing a configuration file saved 
                      using [`~PreTrainedConfig.save_pretrained`].
                    - A path or url to a saved configuration JSON `file`.
            **kwargs: Additional kwargs passed to PreTrainedConfig.from_pretrained()

        Returns:
            [`PreTrainedConfig`]: The configuration object instantiated from that pre-trained policy config.
        """
        if os.path.isdir(pretrained_policy_config_name_or_path):
            # Load from local directory
            config_dict = PreTrainedConfig.from_pretrained(pretrained_policy_config_name_or_path, **kwargs)
            policy_type = config_dict.type
        elif os.path.isfile(pretrained_policy_config_name_or_path):
            # Load from local file
            config_dict = PreTrainedConfig.from_pretrained(pretrained_policy_config_name_or_path, **kwargs)
            policy_type = config_dict.type
        else:
            # Assume it's a policy_type identifier
            policy_type = pretrained_policy_config_name_or_path
            
        if policy_type not in POLICY_CONFIG_MAPPING:
            raise ValueError(
                f"Unrecognized policy type {policy_type}. "
                f"Should be one of {', '.join(POLICY_CONFIG_MAPPING.keys())}"
            )
            
        config_class = POLICY_CONFIG_MAPPING[policy_type]
        return config_class.from_pretrained(pretrained_policy_config_name_or_path, **kwargs)

