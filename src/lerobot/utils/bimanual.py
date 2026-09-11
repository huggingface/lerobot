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

from typing import Any

from lerobot.utils.lifecycle import Cleanup, idempotent_connect


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

    @idempotent_connect
    def connect(self, calibrate: bool = True) -> None:
        """Connect both arms.

        An arm connected before the call is left untouched and the other one is brought
        up. If an arm fails to connect, both arms are disconnected again.
        """
        for arm in (self.left_arm, self.right_arm):
            if not arm.is_connected:
                arm.connect(calibrate)

    def calibrate(self) -> None:
        """Explicitly calibrate both arms, including arms already calibrated."""
        errors: list[Exception] = []
        try:
            self.left_arm.calibrate()
        except Exception as exc:
            exc.add_note(f"while calibrating the left arm of {type(self).__name__}")
            errors.append(exc)
        try:
            self.right_arm.calibrate()
        except Exception as exc:
            exc.add_note(f"while calibrating the right arm of {type(self).__name__}")
            errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup(f"Failed to calibrate {type(self).__name__}", errors)

    def configure(self) -> None:
        """Apply configuration to both arms, attempting both after a failure."""
        errors: list[Exception] = []
        try:
            self.left_arm.configure()
        except Exception as exc:
            exc.add_note(f"while configuring the left arm of {type(self).__name__}")
            errors.append(exc)
        try:
            self.right_arm.configure()
        except Exception as exc:
            exc.add_note(f"while configuring the right arm of {type(self).__name__}")
            errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup(f"Failed to configure {type(self).__name__}", errors)

    def disconnect(self) -> None:
        """Disconnect both arms, attempting the second even after a failure."""
        with Cleanup(self) as cleanup:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                with cleanup.step(f"the {side} arm"):
                    arm.disconnect()
