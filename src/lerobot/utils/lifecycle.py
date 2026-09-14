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

"""Helpers implementing the connect/disconnect contract shared by robots, teleoperators and their parts.

Every device exposes `is_connected`, `connect()` and `disconnect()`. The contract behind them is:

- `connect()` is idempotent and all-or-nothing: calling it on a connected device is a no-op, and a failed
  call leaves nothing acquired.
- `disconnect()` is idempotent and best-effort: it can be called on a device that was never connected or
  is only partially connected, and it releases every resource it can before reporting failures.

Writing that by hand means the same bookkeeping in every device. `idempotent_connect` and `Cleanup`
provide it once, so a device only lists the resources it owns.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from .errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


def idempotent_connect(connect: Callable[..., None]) -> Callable[..., None]:
    """Make a `connect()` method idempotent and all-or-nothing.

    The decorated method returns immediately when `self.is_connected` is already `True`. Otherwise the body
    runs, and if it raises, `self.disconnect()` is called so nothing acquired so far stays open, then the
    original error propagates with a note naming the device. Should that cleanup fail too, both errors are
    raised together in an `ExceptionGroup`.

    The body only has to open what the device owns. Skipping resources that are already connected lets a
    retry resume after an earlier partial failure instead of tripping on an already-open port. It relies on
    `disconnect()` being safe to call in any state, which `Cleanup` provides.

    Args:
        connect (`Callable[..., None]`):
            The `connect` method to wrap. Its owner must expose `is_connected` and `disconnect()`.

    Returns:
        `Callable[..., None]`: The wrapped method.

    Example:
        ```python
        class MyRobot(Robot):
            @idempotent_connect
            def connect(self, calibrate: bool = True) -> None:
                if not self.bus.is_connected:
                    self.bus.connect()
                for cam in self.cameras.values():
                    if not cam.is_connected:
                        cam.connect()
                self.configure()
        ```
    """

    @wraps(connect)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
        if self.is_connected:
            return
        try:
            connect(self, *args, **kwargs)
        except Exception as connect_error:
            connect_error.add_note(f"while connecting {self}")
            try:
                self.disconnect()
            except Exception as disconnect_error:
                disconnect_error.add_note(f"while rolling back the connection of {self}")
                raise ExceptionGroup(
                    f"Failed to connect {self} and to fully roll back",
                    [connect_error, disconnect_error],
                ) from None
            raise
        logger.info(f"{self} connected.")

    return wrapper


class Cleanup:
    """Run every release step even when some fail, then report what failed.

    Use it to write an idempotent, best-effort `disconnect()`. Each `step` block either runs to completion
    or records its exception, so one failing resource never prevents the others from being released. A
    `DeviceNotConnectedError` inside a step means the resource was already released and counts as success.

    Leaving the outer `with` block raises a single recorded failure as is, or an `ExceptionGroup` when
    several steps failed. When at least one step released something and none failed, the release is logged.

    Args:
        owner (`object`):
            The device being released; its string form names it in error notes and logs.
        ignore (`tuple[type[BaseException], ...]`, *optional*, defaults to `(DeviceNotConnectedError,)`):
            Exception types that mean "already released" and are silently accepted inside a step.

    Example:
        ```python
        def disconnect(self) -> None:
            with Cleanup(self) as cleanup:
                with cleanup.step("the motor bus"):
                    self.bus.disconnect()
                for name, cam in self.cameras.items():
                    with cleanup.step(f"camera '{name}'"):
                        cam.disconnect()
        ```
    """

    def __init__(
        self, owner: object, *, ignore: tuple[type[BaseException], ...] = (DeviceNotConnectedError,)
    ) -> None:
        self._owner = owner
        self._ignore = ignore
        self._errors: list[Exception] = []
        self._released = 0

    @contextmanager
    def step(self, what: str) -> Iterator[None]:
        """Run one release step, recording its failure instead of raising it.

        Args:
            what (`str`):
                What the step releases, e.g. `"the motor bus"`. Used in the note attached to a failure.

        Yields:
            `None`: Control returns to the block, whose statements run only up to the first failure.
        """
        try:
            yield
        except self._ignore:
            return
        except Exception as exc:
            exc.add_note(f"while releasing {what} of {self._owner}")
            self._errors.append(exc)
            return
        self._released += 1

    def __enter__(self) -> "Cleanup":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if exc_value is not None:
            # A statement outside any step raised: that is a programming error, not a release failure.
            return False
        if len(self._errors) == 1:
            raise self._errors[0]
        if self._errors:
            raise ExceptionGroup(f"Failed to disconnect {self._owner}", self._errors)
        if self._released:
            logger.info(f"{self._owner} disconnected.")
        return False
