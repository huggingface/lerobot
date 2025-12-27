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


class DeviceNotConnectedError(ConnectionError):
    """Exception raised when the device is not connected."""

    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):
    """Exception raised when the device is already connected."""

    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)


class IsaacLabArenaError(RuntimeError):
    """Base exception for IsaacLab Arena environment errors."""

    def __init__(self, message: str = "IsaacLab Arena error"):
        self.message = message
        super().__init__(self.message)


class IsaacLabArenaConfigError(IsaacLabArenaError):
    """Exception raised for invalid environment configuration."""

    def __init__(self, invalid: list, available: list, key_type: str = "keys"):
        msg = f"Invalid {key_type}: {invalid}. Available: {sorted(available)}"
        super().__init__(msg)
        self.invalid = invalid
        self.available = available


class IsaacLabArenaCameraKeyError(IsaacLabArenaConfigError):
    """Exception raised when camera_keys don't match available cameras."""

    def __init__(self, invalid: list, available: list):
        super().__init__(invalid, available, "camera_keys")


class IsaacLabArenaStateKeyError(IsaacLabArenaConfigError):
    """Exception raised when state_keys don't match available state terms."""

    def __init__(self, invalid: list, available: list):
        super().__init__(invalid, available, "state_keys")
