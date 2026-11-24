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

"""Backend-agnostic visualization dispatch.

Selects a visualization backend at runtime via a display-mode string (e.g. a ``--display_mode`` CLI
flag) so callers never branch on the backend. The concrete implementations live in
:mod:`lerobot.utils.rerun_visualization` and :mod:`lerobot.utils.foxglove_visualization`; importing
this module does not import ``rerun`` or ``foxglove`` (each backend imports its SDK lazily behind a
``require_package`` guard).
"""

from lerobot.lerobot_types import (
    EnvTransition,
    PolicyPrediction,
    RobotAction,
    RobotObservation,
    TransitionKey,
)

from .foxglove_visualization import init_foxglove, log_foxglove_data, shutdown_foxglove
from .rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

# Visualization backends selectable at runtime via a display-mode string (e.g. a --display_mode flag).
VISUALIZATION_MODES = ("rerun", "foxglove")


def init_visualization(
    display_mode: str,
    *,
    session_name: str = "lerobot_control_loop",
    ip: str | None = None,
    port: int | None = None,
) -> None:
    """Initializes the visualization backend selected by ``display_mode``.

    For ``"rerun"``, ``ip``/``port`` point at an optional remote Rerun server. For ``"foxglove"``,
    ``ip`` is the interface to bind the WebSocket server to (``127.0.0.1`` for local only, ``0.0.0.0``
    for all interfaces) and ``port`` is its port.
    """

    if display_mode == "rerun":
        init_rerun(session_name=session_name, ip=ip, port=port)
    elif display_mode == "foxglove":
        init_foxglove(host=ip or "127.0.0.1", port=port)
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


class PredictionPlayback:
    """Shows a chunk's stacked predicted frames at the actions they belong to.

    A policy predicts once per chunk; frames it predicted for several actions arrive stacked
    ``[B, F, C, H, W]`` with the chunk index of each frame in ``observation_steps``. Each call
    is one action: the frame of the latest index reached is returned when it changes, and the
    rest of the prediction is returned as it arrives.
    """

    def __init__(self) -> None:
        self._frames: dict | None = None
        self._steps: list[int] = []
        self._action = 0
        self._shown = -1

    def __call__(self, prediction: PolicyPrediction | None) -> PolicyPrediction | None:
        """This action's prediction to display, or ``None`` when nothing new is shown."""
        if prediction:
            self._frames = None
            if "observation_steps" in prediction:
                self._frames, self._steps = (
                    dict(prediction["observation"]),
                    list(prediction["observation_steps"]),
                )
                self._action, self._shown = 0, -1
                prediction = prediction.copy()
                prediction.pop("observation")
                prediction.pop("observation_steps")
        elif self._frames is not None:
            self._action += 1
        if self._frames is None:
            return prediction or None
        frame = max((i for i, step in enumerate(self._steps) if step <= self._action), default=-1)
        if frame in (-1, self._shown):
            return prediction or None
        self._shown = frame
        shown = prediction.copy() if prediction else PolicyPrediction()
        shown["observation"] = {key: value[:, frame] for key, value in self._frames.items()}
        return shown


_playback = PredictionPlayback()


def log_visualization_data(
    display_mode: str,
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
    policy_step: EnvTransition | None = None,
) -> None:
    """Logs observation/action data and the policy's prediction to the backend selected by ``display_mode``.

    ``policy_step`` is the transition a policy returned with its action; its prediction (see
    ``PreTrainedPolicy.predict_action_chunk``) is shown, stacked predicted frames paced over the
    following calls (see :class:`PredictionPlayback`).
    """

    prediction = _playback(policy_step.get(TransitionKey.PREDICTION) if policy_step is not None else None)

    if display_mode == "rerun":
        log_rerun_data(
            observation=observation,
            action=action,
            compress_images=compress_images,
            prediction=prediction,
        )
    elif display_mode == "foxglove":
        log_foxglove_data(
            observation=observation,
            action=action,
            compress_images=compress_images,
            prediction=prediction,
        )
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")


def shutdown_visualization(display_mode: str) -> None:
    """Shuts down the backend selected by ``display_mode``."""

    if display_mode == "rerun":
        shutdown_rerun()
    elif display_mode == "foxglove":
        shutdown_foxglove()
    else:
        raise ValueError(f"Unknown display_mode '{display_mode}'. Expected one of {VISUALIZATION_MODES}.")
