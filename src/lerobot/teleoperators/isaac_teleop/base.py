#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

NVIDIA Isaac Teleop is a multi-modal teleoperation framework: a single
``TeleopSession`` can be driven by XR controllers, hand tracking, full-body
tracking, Manus gloves, foot pedals, and more. Each input modality is exposed
to LeRobot as its own :class:`Teleoperator` subclass living in its own
``teleop_<device>.py`` module (e.g. ``teleop_xr_controller`` today, ``teleop_manus``,
``teleop_hands`` later).

:class:`IsaacTeleopTeleoperator` factors out everything those device
teleoperators share — the ``TeleopSession`` lifecycle (connect / disconnect),
the per-step staleness/worker-health guard, and the no-op calibration that
tracking devices need. A concrete device only has to:

1. implement :meth:`_build_pipeline` to wire its Isaac Teleop retargeting graph
   (source nodes + retargeters) for its modality, and
2. implement :meth:`get_action` (and :attr:`action_features`), typically by
   calling :meth:`_step` and unpacking the modality-specific outputs.

The ``isaacteleop`` package is an optional, separately distributed NVIDIA
dependency (the ``isaac-teleop`` extra). All imports of it are deferred to
:meth:`connect` so this module — and the device processors — can be imported
and unit-tested without it installed.
"""

from __future__ import annotations

import abc
import logging
import os
from typing import TYPE_CHECKING, Any

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_isaac_teleop import IsaacTeleopConfig

if TYPE_CHECKING:
    from isaacteleop.cloudxr import CloudXRLauncher
    from isaacteleop.retargeting_engine.interface import GraphExecutable, RetargeterIO
    from isaacteleop.teleop_session_manager import TeleopSession

logger = logging.getLogger(__name__)


class IsaacTeleopTeleoperator(Teleoperator):
    """Abstract base for teleoperators backed by an Isaac Teleop ``TeleopSession``.

    Owns the session lifecycle and the per-step health guard shared by every
    Isaac Teleop input device. Subclasses supply the device-specific pipeline
    via :meth:`_build_pipeline` and the device-specific action unpacking via
    :meth:`get_action`. See the module docstring for the device pattern.
    """

    config_class = IsaacTeleopConfig

    def __init__(self, config: IsaacTeleopConfig):
        super().__init__(config)
        self.config: IsaacTeleopConfig = config
        self._session: TeleopSession | None = None
        self._cloudxr_launcher: CloudXRLauncher | None = None

    # ------------------------------------------------------------------
    # Pipeline construction (device override point)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _build_pipeline(self) -> GraphExecutable:
        """Build this device's Isaac Teleop retargeting pipeline.

        Returns the ``GraphExecutable`` (e.g. an ``OutputCombiner``) passed to
        ``TeleopSessionConfig.pipeline``. The base class calls this exactly once
        during :meth:`connect`. The returned pipeline's output keys must match
        what this device's :meth:`get_action` unpacks.
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
        """Start the CloudXR runtime (unless opted out) and open the Isaac Teleop session.

        By default this auto-launches the NVIDIA CloudXR runtime so operators no
        longer need to run ``python -m isaacteleop.cloudxr`` and ``source
        cloudxr.env`` by hand. The CloudXR launch is a blocking call that can take
        ~30s, and on the very first run it prompts on stdin to accept the NVIDIA
        CloudXR EULA (run ``python -m isaacteleop.cloudxr --accept-eula`` once to
        bootstrap a headless machine). Opt out of the auto-launch — when CloudXR is
        already running externally — by setting ``config.auto_launch_cloudxr=False``
        or exporting ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH=1`` (the env var takes
        precedence over the config field).
        """
        if self._session is not None:
            raise RuntimeError("Already connected. Call disconnect() first.")

        self._ensure_cloudxr_runtime()

        try:
            from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

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
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
            logger.info("Isaac Teleop session ended")

        # Reap the CloudXR runtime even if no session was ever established (e.g.
        # the launcher came up but session creation failed before this point); a
        # no-op when we never launched CloudXR (opt-out / externally-owned
        # runtime), so we never stop a runtime we don't own.
        self._stop_cloudxr_runtime()

    # ------------------------------------------------------------------
    # CloudXR runtime (shared)
    # ------------------------------------------------------------------

    def _ensure_cloudxr_runtime(self) -> None:
        """Auto-launch the CloudXR runtime once, unless the operator opted out.

        Idempotent: a no-op once the launcher is up. The
        ``LEROBOT_CLOUDXR_SKIP_AUTOLAUNCH`` env var is checked first and takes
        precedence over ``config.auto_launch_cloudxr``; when either opts out we
        assume CloudXR is already running / externally owned. This mirrors Isaac
        Lab's auto-launch (whose env var is ``ISAACLAB_CXR_SKIP_AUTOLAUNCH``); the
        two knobs are independent. Constructing :class:`CloudXRLauncher` mutates
        the current process environment (``XR_RUNTIME_JSON`` etc.) and blocks until
        the runtime is ready or raises :class:`RuntimeError`.
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

        from pathlib import Path

        from isaacteleop.cloudxr import CloudXRLauncher

        self._cloudxr_launcher = CloudXRLauncher(
            install_dir=str(Path.home() / ".cloudxr"),
            env_config=self.config.cloudxr_env_file,
            accept_eula=False,
        )

    def _stop_cloudxr_runtime(self) -> None:
        """Stop the auto-launched CloudXR runtime, if any.

        On a clean stop the handle is nulled. On a :class:`RuntimeError` (the
        runtime could not be terminated) the handle is RETAINED — the launcher's
        own ``atexit`` hook owns the retry — and a warning is logged.
        """
        if self._cloudxr_launcher is None:
            return
        try:
            self._cloudxr_launcher.stop()
        except RuntimeError:
            logger.warning("CloudXR runtime could not be terminated; handle retained for atexit cleanup")
            # handle retained — the launcher's own atexit hook owns the retry.
            # Consequence: a later connect() -> _ensure_cloudxr_runtime() sees the
            # retained handle, early-returns, and will NOT relaunch (treats the
            # retained runtime as still up).
        else:
            self._cloudxr_launcher = None
            logger.info("CloudXR runtime stopped")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # Haptic feedback not yet implemented.

    # ------------------------------------------------------------------
    # Stepping (shared)
    # ------------------------------------------------------------------

    def _step(self) -> RetargeterIO:
        """Step the session once and return the raw pipeline outputs.

        Applies the shared staleness / worker-health guard: re-raises a
        retargeting-worker exception and warns on a dropped/stale frame. Device
        subclasses call this from :meth:`get_action` and unpack the result.

        Raises:
            RuntimeError: If not connected, or if the retargeting worker raised.
        """
        if self._session is None:
            raise RuntimeError("Not connected. Call connect() first.")

        result = self._session.step()

        # ``last_step_info`` exposes whether the retargeting worker raised and
        # how old the returned frame is.
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
