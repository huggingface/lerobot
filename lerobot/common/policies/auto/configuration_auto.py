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

import importlib
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig

logger = logging.getLogger(__name__)

# Constants
IMPORT_PATHS = ["lerobot.common.policies.{0}.configuration_{0}"]

POLICY_IMPORT_PATHS = ["lerobot.common.policies.{0}.modeling_{0}"]


def policy_type_to_module_name(policy_type: str) -> str:
    """
    Convert policy type to module name format.

    Args:
        policy_type: The policy type identifier (e.g. 'lerobot/vqbet-pusht')

    Returns:
        str: Normalized module name (e.g. 'vqbet')

    Examples:
        >>> policy_type_to_module_name("lerobot/vqbet-pusht")
        'vqbet'
    """
    # TODO(Steven): This is a temporary solution, we should have a more robust way to handle this
    return policy_type.replace("lerobot/", "").replace("-", "_").replace("_", "").replace("pusht", "")


class _LazyPolicyConfigMapping(OrderedDict):
    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping
        self._extra_content: Dict[str, Any] = {}
        self._modules: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(f"Policy type '{key}' not found in mapping")

        value = self._mapping[key]
        module_name = policy_type_to_module_name(key)

        for import_path in IMPORT_PATHS:
            try:
                if key not in self._modules:
                    self._modules[key] = importlib.import_module(import_path.format(module_name))
                    logger.debug(f"Config module: {module_name} imported")
                if hasattr(self._modules[key], value):
                    return getattr(self._modules[key], value)
            except ImportError:
                continue

        raise ImportError(f"Could not find configuration class {value} for policy type {key}")

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Policy Config, pick another name.")
        self._extra_content[key] = value


POLICY_CONFIG_NAMES_MAPPING = OrderedDict(
    [
        ("vqbet", "VQBeTConfig"),
        ("lerobot/vqbet_pusht", "VQBeTConfig"),
    ]
)

POLICY_CONFIG_MAPPING = _LazyPolicyConfigMapping(POLICY_CONFIG_NAMES_MAPPING)


class _LazyPolicyMapping(OrderedDict):
    """
    A dictionary that lazily loads its values when they are requested.
    """

    def __init__(self, mapping: Dict[str, str]):
        self._mapping = mapping
        self._extra_content: Dict[str, Type[PreTrainedPolicy]] = {}
        self._modules: Dict[str, Any] = {}
        self._config_mapping: Dict[Type[PreTrainedConfig], Type[PreTrainedPolicy]] = {}
        self._initialized_types: set[str] = set()

    def _lazy_init_for_type(self, policy_type: str) -> None:
        """Lazily initialize mappings for a policy type if not already done."""
        if policy_type not in self._initialized_types:
            try:
                config_class = POLICY_CONFIG_MAPPING[policy_type]
                self._config_mapping[config_class] = self[policy_type]
                self._initialized_types.add(policy_type)
            except (ImportError, AttributeError, KeyError) as e:
                logger.warning(f"Could not automatically map config for policy type {policy_type}: {str(e)}")

    def __getitem__(self, key: str) -> Type[PreTrainedPolicy]:
        """Get a policy class by key with lazy loading."""
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(f"Policy type '{key}' not found in mapping")

        value = self._mapping[key]
        module_name = policy_type_to_module_name(key)

        for import_path in POLICY_IMPORT_PATHS:
            try:
                if key not in self._modules:
                    self._modules[key] = importlib.import_module(import_path.format(module_name))
                    logger.debug(
                        f"Policy module: {module_name} imported from {import_path.format(module_name)}"
                    )
                if hasattr(self._modules[key], value):
                    return getattr(self._modules[key], value)
            except ImportError:
                continue

        raise ImportError(
            f"Could not find policy class {value} for policy type {key}. "
            f"Tried paths: {[p.format(module_name) for p in POLICY_IMPORT_PATHS]}"
        )

    def register(
        self,
        key: str,
        value: Type[PreTrainedPolicy],
        config_class: Type[PreTrainedConfig],
        exist_ok: bool = False,
    ) -> None:
        """Register a new policy class with its configuration class."""
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key)}")
        if not issubclass(value, PreTrainedPolicy):
            raise TypeError(f"Value must be a PreTrainedPolicy subclass, got {type(value)}")
        if not issubclass(config_class, PreTrainedConfig):
            raise TypeError(f"Config class must be a PreTrainedConfig subclass, got {type(config_class)}")

        if key in self._mapping and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Policy, pick another name.")
        self._extra_content[key] = value
        self._config_mapping[config_class] = value

    def get_policy_for_config(self, config_class: Type[PreTrainedConfig]) -> Type[PreTrainedPolicy]:
        """Get the policy class associated with a config class."""
        # First check direct config class mapping
        if config_class in self._config_mapping:
            return self._config_mapping[config_class]

        # Try to find by policy type
        try:
            policy_type = config_class.get_type_str()
            # Check extra content first
            if policy_type in self._extra_content:
                return self._extra_content[policy_type]

            # Then check standard mapping
            if policy_type in self._mapping:
                self._lazy_init_for_type(policy_type)
                if config_class in self._config_mapping:
                    return self._config_mapping[config_class]
                return self[policy_type]
        except AttributeError:
            pass

        raise ValueError(
            f"No policy class found for config class {config_class.__name__}. "
            f"Available types: {list(self._mapping.keys()) + list(self._extra_content.keys())}"
        )


POLICY_NAMES_MAPPING = OrderedDict(
    [
        ("vqbet", "VQBeTPolicy"),
        ("lerobot/vqbet_pusht", "VQBeTPolicy"),
    ]
)

POLICY_MAPPING = _LazyPolicyMapping(POLICY_NAMES_MAPPING)


class AutoPolicyConfig:
    """
    Factory class for automatically loading policy configurations.

    This class provides methods to:
    - Load pre-trained policy configurations from local files or the Hub
    - Register new policy types dynamically
    - Create policy configurations for specific policy types
    """

    def __init__(self):
        raise OSError("AutoPolicyConfig not meant to be instantiated directly")

    @classmethod
    def for_policy(cls, policy_type: str, *args, **kwargs) -> PreTrainedConfig:
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
        if issubclass(config, PreTrainedConfig) and config.get_type_str() != policy_type:
            raise ValueError(
                "The config you are passing has a `policy_type` attribute that is not consistent with the policy type "
                f"you passed (config has {config.type} and you passed {policy_type}. Fix one of those so they "
                "match!"
            )
        POLICY_CONFIG_MAPPING.register(policy_type, config, exist_ok=exist_ok)

    @classmethod
    def from_pretrained(
        cls, pretrained_policy_config_name_or_path: Union[str, Path], **kwargs
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


class AutoPolicy:
    """
    Factory class that allows instantiating policy models from configurations.

    This class provides methods to:
    - Load pre-trained policies from configurations
    - Register new policy types dynamically
    - Create policy instances for specific configurations
    """

    def __init__(self):
        raise OSError("AutoPolicy not meant to be instantiated directly")

    @classmethod
    def from_config(cls, config: PreTrainedConfig, **kwargs) -> PreTrainedPolicy:
        """Instantiate a policy from a configuration."""
        policy_class = POLICY_MAPPING.get_policy_for_config(type(config))
        return policy_class(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_policy_name_or_path: Union[str, Path],
        *,
        config: Optional[PreTrainedConfig] = None,
        **kwargs,
    ) -> PreTrainedPolicy:
        """
        Instantiate a pre-trained policy from a configuration.

        Args:
            pretrained_policy_name_or_path: Path to pretrained weights or model identifier
            config: Optional configuration for the policy
            **kwargs: Additional arguments to pass to from_pretrained()
        """
        if config is None:
            config = AutoPolicyConfig.from_pretrained(pretrained_policy_name_or_path)

        if isinstance(config, str):
            config = AutoPolicyConfig.from_pretrained(config)

        policy_class = POLICY_MAPPING.get_policy_for_config(config)
        return policy_class.from_pretrained(pretrained_policy_name_or_path, config=config, **kwargs)

    @staticmethod
    def register(
        config_class: Type[PreTrainedConfig], policy_class: Type[PreTrainedPolicy], exist_ok: bool = False
    ):
        """
        Register a new policy class for a configuration class.

        Args:
            config_class: The configuration class
            policy_class: The policy class to register
            exist_ok: Whether to allow overwriting existing registrations
        """
        POLICY_MAPPING.register(config_class.get_type_str(), policy_class, config_class, exist_ok=exist_ok)


def main():
    """Test the AutoPolicy and AutoPolicyConfig functionality."""

    def test_error_cases():
        """Test error handling"""
        try:
            AutoPolicyConfig()
        except OSError as e:
            assert "not meant to be instantiated directly" in str(e)
        try:
            AutoPolicy()
        except OSError as e:
            assert "not meant to be instantiated directly" in str(e)

        # try:
        #     AutoPolicy.from_config("invalid_config")
        # except ValueError as e:
        #     assert "Unrecognized policy identifier" in str(e)

    logging.basicConfig(level=logging.DEBUG)

    # Test built-in policy loading
    # config = AutoPolicyConfig.from_pretrained("lerobot/vqbet_pusht")
    config = AutoPolicyConfig.for_policy("vqbet")
    policy = AutoPolicy.from_config(config)

    from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
    from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

    assert isinstance(config, VQBeTConfig)
    assert isinstance(policy, VQBeTPolicy)

    # Test policy registration
    from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
    from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

    AutoPolicyConfig.register("tdmpc", TDMPCConfig)
    AutoPolicy.register(TDMPCConfig, TDMPCPolicy)

    my_new_config = AutoPolicyConfig.for_policy("tdmpc")
    my_new_policy = AutoPolicy.from_config(my_new_config)
    assert isinstance(my_new_config, TDMPCConfig)
    assert isinstance(my_new_policy, TDMPCPolicy)

    # Run error case tests
    test_error_cases()


if __name__ == "__main__":
    main()
