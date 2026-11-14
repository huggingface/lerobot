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

"""Per-tick action dispatch for inference engines, including ones that return PyTorch policy outputs.

The permission, hold and interpolation sequence lives in :mod:`lerobot.utils.action_dispatch`, which
does not import PyTorch. These wrappers add what a PyTorch engine needs: an engine may return the
action with the policy's prediction (an ``EnvTransition``, see ``PreTrainedPolicy.select_action``);
only the action is interpolated, and the prediction goes to the caller.

``lerobot-rollout`` builds its dispatch helper on these functions and adds dataset
frames, robot-side processors and recording; a control loop outside the rollout
package calls :func:`dispatch_tick` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from lerobot.lerobot_types import PolicyPrediction
from lerobot.processor import (
    policy_output_to_transition,
    transition_to_policy_action,
    transition_to_prediction,
)
from lerobot.utils import action_dispatch
from lerobot.utils.action_dispatch import commit_action, dispatch_permitted
from lerobot.utils.cycle_timer import CycleTimer

from .base import InferenceEngine

if TYPE_CHECKING:
    from lerobot.utils.action_interpolator import ActionInterpolator

__all__ = ["commit_action", "dispatch_permitted", "dispatch_tick", "next_interpolated_action"]


def next_interpolated_action(
    engine: InferenceEngine,
    interpolator: ActionInterpolator,
    obs_frame: dict | Callable[[], dict | None] | None,
    *,
    hold: Callable[[], None] | None = None,
    timer: CycleTimer | None = None,
    on_prediction: Callable[[PolicyPrediction], None] | None = None,
) -> torch.Tensor | np.ndarray | None:
    """Return the next interpolated target, pulling one policy action when needed.

    See :func:`lerobot.utils.action_dispatch.next_interpolated_action`. A non-empty
    prediction returned with the action is handed to ``on_prediction`` on the tick
    that pulled it. A NumPy action is returned bare, carries no prediction and is
    interpolated as it is.
    """
    predictions: list[PolicyPrediction] = []
    target = action_dispatch.next_interpolated_action(
        engine,
        interpolator,
        obs_frame,
        hold=hold,
        timer=timer,
        unpack=lambda result: _policy_action(result, None if on_prediction is None else predictions.append),
    )
    # Reported only now, so a prediction is never reported for an action the interpolator refused.
    if on_prediction is not None:
        for prediction in predictions:
            on_prediction(prediction)
    return target


def dispatch_tick(
    engine: InferenceEngine,
    interpolator: ActionInterpolator,
    obs_frame: dict | Callable[[], dict | None] | None = None,
    *,
    send: Callable[[torch.Tensor | np.ndarray], None],
    hold: Callable[[], None] | None = None,
    timer: CycleTimer | None = None,
) -> torch.Tensor | np.ndarray | None:
    """One complete autonomous motor tick: gate, pull, interpolate, gate, send, record.

    See :func:`lerobot.utils.action_dispatch.dispatch_tick`.
    """
    return action_dispatch.dispatch_tick(
        engine,
        interpolator,
        obs_frame,
        send=send,
        hold=hold,
        timer=timer,
        unpack=lambda result: _policy_action(result, None),
    )


def _policy_action(
    result: Any, on_prediction: Callable[[PolicyPrediction], None] | None
) -> torch.Tensor | np.ndarray:
    if isinstance(result, np.ndarray):
        return result
    step = policy_output_to_transition(result)
    action = transition_to_policy_action(step).cpu()
    if on_prediction is not None and (prediction := transition_to_prediction(step)):
        on_prediction(prediction)
    return action
