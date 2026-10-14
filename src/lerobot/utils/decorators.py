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

from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Protocol

from .errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


class _Connectable(Protocol):
    @property
    def is_connected(self) -> bool: ...


def check_if_not_connected[T: _Connectable, **P, R](
    func: Callable[Concatenate[T, P], R],
) -> Callable[Concatenate[T, P], R]:
    @wraps(func)
    def wrapper(self: T, /, *args: P.args, **kwargs: P.kwargs) -> R:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                f"{self.__class__.__name__} is not connected. Run `.connect()` first."
            )
        return func(self, *args, **kwargs)

    return wrapper


def check_if_already_connected(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.__class__.__name__} is already connected.")
        return func(self, *args, **kwargs)

    return wrapper
