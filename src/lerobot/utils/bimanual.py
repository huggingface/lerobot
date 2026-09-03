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

import logging
from typing import Any

from lerobot.utils.decorators import check_if_already_connected
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class BimanualMixin:
    """Lifecycle delegation for bimanual robots and teleoperators.

    Concrete subclasses must populate ``self.left_arm`` and ``self.right_arm`` in
    their own ``__init__``. They retain ownership of feature dicts and the
    data-routing methods (``get_action`` / ``send_action`` / ``get_observation`` /
    ``send_feedback``), which vary per-embodiment.

    Inherit before the ``Robot`` / ``Teleoperator`` base so the mixin's methods
    take precedence in the MRO::

        class BiFooFollower(BimanualMixin, Robot): ...
    """

    left_arm: Any
    right_arm: Any

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self.left_arm.connect(calibrate)
            self.right_arm.connect(calibrate)
        except Exception:
            self._disconnect_arms(after_failed_connect=True)
            raise

    def _disconnect_arm_after_failed_connect(self, arm: Any) -> None:
        """Disconnect an arm while rolling back a failed bimanual startup."""
        arm.disconnect()

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def disconnect(self) -> None:
        self._disconnect_arms()

    def _disconnect_arms(self, *, after_failed_connect: bool = False) -> None:
        arms = (
            (("right", self.right_arm), ("left", self.left_arm))
            if after_failed_connect
            else (("left", self.left_arm), ("right", self.right_arm))
        )
        for name, arm in arms:
            try:
                if after_failed_connect:
                    self._disconnect_arm_after_failed_connect(arm)
                else:
                    arm.disconnect()
            except DeviceNotConnectedError:
                pass
            except Exception:
                logger.exception("Failed to disconnect the %s arm.", name)
