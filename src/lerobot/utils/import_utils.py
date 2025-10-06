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
import importlib
import logging
import pkgutil
from typing import Any

from draccus.choice_types import ChoiceRegistry


def is_package_available(pkg_name: str, return_version: bool = False) -> tuple[bool, str] | bool:
    """Copied from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py
    Check if the package spec exists and grab its version to avoid importing a local directory.
    **Note:** this doesn't work for all packages.
    """
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)

        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            elif pkg_name == "grpc":
                package = importlib.import_module(pkg_name)
                package_version = getattr(package, "__version__", "N/A")
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        logging.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_transformers_available = is_package_available("transformers")


def make_device_from_device_class(config: ChoiceRegistry) -> Any:
    """
    Dynamically instantiates an object from its `ChoiceRegistry` configuration.

    This factory uses the module path and class name from the `config` object's
    type to locate and instantiate the correct class, passing the `config` object
    itself to the constructor.

    Args:
        config: The configuration object, an instance of a `ChoiceRegistry` subclass.

    Returns:
        An instance of the class specified by the `config` object's type.
    """
    if not isinstance(config, ChoiceRegistry):
        raise ValueError(f"Config should be an instance of `ChoiceRegistry`, got {type(config)}")

    class_module_path, class_name_type = config.__class__.__module__, config.__class__.__name__

    try:
        module = importlib.import_module(class_module_path)
    except Exception as e:
        raise ImportError(
            f"Could not import module '{class_module_path}' for device  '{class_name_type}': {e}"
        ) from e

    try:
        cls = getattr(module, class_name_type)
    except AttributeError as e:
        raise AttributeError(f"Module '{class_module_path}' has no attribute '{class_name_type}'") from e

    if not callable(cls):
        raise TypeError(f"Resolved object {cls!r} is not callable")

    try:
        return cls(config)
    except TypeError as e:
        raise TypeError(f"Failed to instantiate '{config}': {e}") from e


def register_third_party_devices() -> None:
    """
    Discover and import third-party lerobot_* plugins so they can register themselves.

    Scans top-level modules on sys.path for packages starting with
    'lerobot_robot_', 'lerobot_camera_' or 'lerobot_teleoperator_' and imports them.
    """
    prefixes = ("lerobot_robot_", "lerobot_camera_", "lerobot_teleoperator_")
    imported = []
    failed = []

    for module_info in pkgutil.iter_modules():
        name = module_info.name
        if any(name.startswith(p) for p in prefixes):
            try:
                importlib.import_module(name)
                imported.append(name)
                logging.info("Imported third-party plugin: %s", name)
            except Exception:
                logging.exception("Could not import third-party plugin: %s", name)
                failed.append(name)

    logging.debug("Third-party plugin import summary: imported=%s failed=%s", imported, failed)
