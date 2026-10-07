#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Shared base for NVIDIA Isaac Teleop-backed LeRobot teleoperators.

Isaac Teleop is a multi-modal framework: a single ``TeleopSession`` can be driven by
XR controllers, hand tracking, Manus gloves, etc. Each modality is a
:class:`Teleoperator` subclass in its own ``teleop_<device>.py``.

:class:`IsaacTeleopTeleoperator` owns what those devices share — the session
lifecycle, the per-step staleness/worker-health guard, and the no-op calibration
tracking devices need. A concrete device implements :meth:`_build_pipeline` (its
retargeting graph) and :meth:`get_action` (usually via :meth:`_step`).

``isaacteleop`` is an optional NVIDIA dependency (install instructions in the example's
``README.md``); its imports are guarded behind an availability check at module top, so this
module imports without it and constructing a device fails fast with install instructions.
"""

from __future__ import annotations

import abc
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.import_utils import is_package_available

from .config_isaac_teleop import IsaacTeleopConfig

_isaacteleop_available = is_package_available("isaacteleop")

if TYPE_CHECKING or _isaacteleop_available:
    from isaacteleop.cloudxr import CloudXRLauncher
    from isaacteleop.retargeting_engine.interface import (
        ExecutionEvents,
        ExecutionState,
        GraphExecutable,
        RetargeterIO,
    )
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig
else:
    CloudXRLauncher = None
    ExecutionEvents = None
    ExecutionState = None
    GraphExecutable = None
    RetargeterIO = None
    TeleopSession = None
    TeleopSessionConfig = None

logger = logging.getLogger(__name__)

# Gripper closedness [0, 1] -> SO-101 follower motor units [0, 100] (RANGE_0_100, 100 = OPEN).
# Shared by the XR processor and leader device, which invert via ``pos = (1 - c) * SCALE``.
_GRIPPER_MOTOR_SCALE = 100.0


def _require_isaacteleop() -> None:
    """Fail fast with install pointers when the optional ``isaacteleop`` package is missing."""
    if not _isaacteleop_available:
        raise ImportError(
            "The 'isaacteleop' package is required for Isaac Teleop devices but is not "
            "installed. See examples/isaac_teleop_to_so101/README.md for install instructions."
        )


class IsaacTeleopTeleoperator(Teleoperator):
    """Abstract base for teleoperators backed by an Isaac Teleop ``TeleopSession``.

    Owns the session lifecycle and the per-step health guard; subclasses supply
    :meth:`_build_pipeline` and :meth:`get_action`.
    """

    config_class = IsaacTeleopConfig

    def __init__(self, config: IsaacTeleopConfig):
        _require_isaacteleop()
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None

    # ------------------------------------------------------------------
    # Pipeline construction (device override point)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_pipeline(self) -> GraphExecutable:
        """Build this device's retargeting pipeline (the ``GraphExecutable`` for
        ``TeleopSessionConfig.pipeline``). Called once in :meth:`connect`; its output
        keys must match what :meth:`get_action` unpacks.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle (shared)
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def is_calibrated(self) -> bool:
        return True  # Tracking devices are self-calibrating.

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        """Auto-launch the CloudXR runtime (unless opted out) and open the session.

        The CloudXR launch blocks ~30s and, on the first run, prompts on stdin for the
        EULA (accept once via ``python -m isaacteleop.cloudxr --accept-eula``). Opt out
        when CloudXR runs externally via ``config.auto_launch_cloudxr=False`` or
        ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1`` (env var wins).
        """
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        self._ensure_cloudxr_runtime()

        try:
            pipeline = self._build_pipeline()
            session_config = TeleopSessionConfig(app_name=self.config.app_name, pipeline=pipeline)
            self._session = TeleopSession(session_config)
            self._session.__enter__()
        except Exception:
            self._session = None
            try:
                self._stop_cloudxr_runtime()
            except Exception:
                logger.exception("Failed to stop CloudXR runtime during connect() rollback")
            raise
        logger.info("Isaac Teleop session started: %s", self.config.app_name)

    def disconnect(self) -> None:
        try:
            if self._session is not None:
                # Null the handle BEFORE __exit__: even a failed session teardown must not
                # wedge the device as is_connected (blocking every later connect/disconnect).
                session = self._session
                self._session = None
                session.__exit__(None, None, None)
                logger.info("Isaac Teleop session ended")
        finally:
            # Reap the CloudXR runtime even if session teardown raised, and even if no
            # session was ever established (e.g. the launcher came up but session creation
            # failed before this point); a no-op when we never launched CloudXR (opt-out /
            # externally-owned runtime), so we never stop a runtime we don't own.
            self._stop_cloudxr_runtime()

    # ------------------------------------------------------------------
    # CloudXR runtime (shared)
    # ------------------------------------------------------------------

    def _ensure_cloudxr_runtime(self) -> None:
        """Auto-launch the CloudXR runtime once, unless opted out.

        Idempotent (no-op once the launcher is up). ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH``
        is checked first and wins over ``config.auto_launch_cloudxr``. Constructing
        :class:`CloudXRLauncher` mutates the process env (``XR_RUNTIME_JSON`` etc.) and
        blocks until the runtime is ready or raises :class:`RuntimeError`.
        """
        if self._cloudxr_launcher is not None:
            return

        if os.environ.get("LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH", "").strip() == "1":
            logger.info(
                "LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1 set; skipping CloudXR auto-launch "
                "(assuming CloudXR is already running externally)"
            )
            return

        if not self.config.auto_launch_cloudxr:
            logger.info(
                "config.auto_launch_cloudxr is False; skipping CloudXR auto-launch "
                "(assuming CloudXR is already running externally)"
            )
            return

        logger.info("Launching CloudXR runtime (first run may prompt for EULA and take ~30s)...")

        self._cloudxr_launcher = CloudXRLauncher(
            install_dir=str(Path.home() / ".cloudxr"),
            env_config=self.config.cloudxr_env_file,
            accept_eula=False,
        )

    def _stop_cloudxr_runtime(self) -> None:
        """Stop the auto-launched CloudXR runtime, if any.

        Clean stop nulls the handle. On :class:`RuntimeError` the handle is RETAINED so
        the launcher's ``atexit`` hook owns the retry — a later :meth:`connect` then
        treats the retained runtime as still up and will not relaunch.
        """
        if self._cloudxr_launcher is None:
            return
        try:
            self._cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be terminated; handle retained for atexit cleanup")
        else:
            self._cloudxr_launcher = None
            logger.info("CloudXR runtime stopped")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not yet implemented.

    # ------------------------------------------------------------------
    # Stepping (shared)
    # ------------------------------------------------------------------

    def _running_events(self) -> ExecutionEvents:
        """Constant ``RUNNING`` ``ExecutionEvents`` for a device with no clutch lifecycle.

        Keeps the stream flowing; ``reset`` stays ``False``. A clutched device that needs
        a real lifecycle should build its own ``ExecutionEvents`` instead.
        """
        return ExecutionEvents(execution_state=ExecutionState.RUNNING, reset=False)

    def _step(
        self,
        *,
        execution_events: ExecutionEvents | None = None,
        external_inputs: Mapping[str, Any] | None = None,
    ) -> RetargeterIO:
        """Step the session once and return the raw pipeline outputs.

        Applies the shared guard: re-raises a retargeting-worker exception and warns on a
        stale frame. Subclasses call this from :meth:`get_action`.

        Args:
            execution_events: The ``ExecutionEvents`` driving the session this frame.
                Devices with a lifecycle (clutch) MUST pass this every frame — when
                ``None``, ``TeleopSession.step`` auto-fires ``RUNNING`` (the clutch would
                latch immediately and never stop).
            external_inputs: Per-step inputs (e.g. a static ``base_T_anchor``) in the
                ``{leaf_node_name: {output_port_name: TensorGroup}}`` shape ``step`` expects.

        Raises:
            RuntimeError: If not connected, or if the retargeting worker raised.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._session.step(
            execution_events=execution_events,
            external_inputs=external_inputs,
        )

        info = self._session.last_step_info
        if info is not None:
            if info.worker_exception is not None:
                raise RuntimeError(
                    "Isaac Teleop retargeting worker raised an exception"
                ) from info.worker_exception
            if info.frame_deadline_miss:
                logger.warning(
                    "Isaac Teleop frame deadline miss (returned_age_frames=%s)",
                    info.returned_age_frames,
                )
        return result
