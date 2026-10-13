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

"""
Public API for lightweight, base-dependency-only utilities.

Heavy cross-cutting modules (train_utils, control_utils) have been moved
to ``lerobot.common``. ``visualization_utils`` remains here but is
intentionally NOT re-exported to avoid pulling in optional dependencies.
"""

from .constants import (
    ACTION,
    DEFAULT_FEATURES,
    DONE,
    IMAGENET_STATS,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
    OBS_STR,
    QUERY_KIND,
    QUERY_TEXT,
    REWARD,
)
from .decorators import check_if_already_connected, check_if_not_connected
from .errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .import_utils import is_package_available, register_third_party_plugins, require_package
from .utils import init_logging

__all__ = [
    # Constants
    "ACTION",
    "DEFAULT_FEATURES",
    "DONE",
    "IMAGENET_STATS",
    "OBS_ENV_STATE",
    "OBS_IMAGE",
    "OBS_IMAGES",
    "OBS_STATE",
    "OBS_STR",
    "QUERY_KIND",
    "QUERY_TEXT",
    "REWARD",
    # Device utilities
    "auto_select_torch_device",
    "get_safe_torch_device",
    "is_torch_device_available",
    # Import guards
    "is_package_available",
    "register_third_party_plugins",
    "require_package",
    # General utilities
    "ActionInterpolator",
    "init_logging",
    # Decorators
    "check_if_already_connected",
    "check_if_not_connected",
    # Errors
    "DeviceAlreadyConnectedError",
    "DeviceNotConnectedError",
]


def __getattr__(name: str):
    # Torch-backed helpers load on first use so torch-free images (simulator servers) can import utils.
    if name == "ActionInterpolator":
        from .action_interpolator import ActionInterpolator

        return ActionInterpolator
    if name in {"auto_select_torch_device", "get_safe_torch_device", "is_torch_device_available"}:
        from . import device_utils

        return getattr(device_utils, name)
    raise AttributeError(name)
