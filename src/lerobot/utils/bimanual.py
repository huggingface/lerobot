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
            self._rollback_failed_connect()
            raise

    def _disconnect_arm_for_cleanup(self, arm: Any) -> None:
        """Disconnect one arm while recovering from a failed startup.

        Override this when cleanup has to be stronger than the arm's normal
        ``disconnect()``: a motorized follower configured with
        ``disable_torque_on_disconnect=False`` still has to end up torque-free
        when the robot as a whole failed to come up.
        """
        arm.disconnect()

    def _rollback_failed_connect(self) -> None:
        """Release both arms without masking the error that triggered rollback."""
        # The right arm goes first because it may hold the half-initialized
        # resources of the attempt that just failed.
        for arm_name, arm in (("right", self.right_arm), ("left", self.left_arm)):
            self._release_arm(arm_name, arm, after_failed_connect=True)

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def disconnect(self) -> None:
        """Release both arms, continuing if one of them fails.

        Safe to call repeatedly and after a partial connection, so a caller can
        always shut the robot down in a ``finally`` block.
        """
        for arm_name, arm in (("left", self.left_arm), ("right", self.right_arm)):
            self._release_arm(arm_name, arm, after_failed_connect=False)

    def _release_arm(self, arm_name: str, arm: Any, *, after_failed_connect: bool) -> None:
        """Disconnect one arm, logging instead of raising."""
        try:
            if after_failed_connect:
                self._disconnect_arm_for_cleanup(arm)
            else:
                arm.disconnect()
        except DeviceNotConnectedError:
            # An arm that never came up, or that is already released, is the
            # normal case here. Staying quiet keeps real cleanup faults visible.
            logger.debug("The %s arm was already disconnected.", arm_name)
        except Exception:
            logger.exception("Failed to disconnect the %s arm during cleanup.", arm_name)
