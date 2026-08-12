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

from functools import wraps

from .errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


def check_if_not_connected(func):
    """Decorate a device method to raise `DeviceNotConnectedError` if `self.is_connected` is `False`."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Call `func`, raising `DeviceNotConnectedError` first if `self` isn't connected."""
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def check_if_already_connected(func):
    """Decorate a device method to raise `DeviceAlreadyConnectedError` if `self.is_connected` is `True`."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Call `func`, raising `DeviceAlreadyConnectedError` first if `self` is already connected."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} is already connected.")
        return func(self, *args, **kwargs)

    return wrapper
