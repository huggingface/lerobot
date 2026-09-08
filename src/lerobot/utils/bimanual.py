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

    def connect(self, calibrate: bool = True) -> None:
        """Connect both arms, repairing a partial connection when possible.

        Calling this method when both arms are already connected is a no-op. If
        an arm fails to connect, arms whose connection started during this call
        are disconnected again. An arm that was connected before the call is
        left untouched.
        """
        if self.is_connected:
            return

        started: list[tuple[str, Any]] = []
        try:
            for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
                if arm.is_connected:
                    continue

                started.append((side, arm))
                try:
                    arm.connect(calibrate)
                except Exception as exc:
                    exc.add_note(f"while connecting the {side} arm of {type(self).__name__}")
                    raise
        except Exception as connect_error:
            rollback_errors: list[Exception] = []
            for side, arm in reversed(started):
                try:
                    arm.disconnect()
                except Exception as exc:
                    exc.add_note(f"while rolling back the {side} arm connection of {type(self).__name__}")
                    rollback_errors.append(exc)

            if rollback_errors:
                raise ExceptionGroup(
                    f"Failed to connect {type(self).__name__} and to fully roll back",
                    [connect_error, *rollback_errors],
                ) from None
            raise

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
        errors: list[Exception] = []
        for side, arm in (("left", self.left_arm), ("right", self.right_arm)):
            try:
                arm.disconnect()
            except Exception as exc:
                exc.add_note(f"while disconnecting the {side} arm of {type(self).__name__}")
                errors.append(exc)

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ExceptionGroup(f"Failed to disconnect {type(self).__name__}", errors)
