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

"""Per-tick action dispatch that honors the engine's permission and hold contract.

A returned action is never dispatch permission on its own. Every autonomous motor
tick, including interpolation ticks and ticks that send nothing, asks the engine
before pulling an action, again before reading the interpolated target and once
more immediately before the actuator write. A denial clears interpolation, applies
the caller's local hold when one is configured and acknowledges it, so a worker can
proceed or a terminal fault can reach shutdown.

``lerobot-rollout`` builds its dispatch helper on these functions and adds dataset
frames, robot-side processors and recording; a control loop outside the rollout
package calls :func:`dispatch_tick` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext

import torch

from lerobot.lerobot_types import PolicyPrediction
from lerobot.processor import (
    policy_output_to_transition,
    transition_to_policy_action,
    transition_to_prediction,
)
from lerobot.utils import ActionInterpolator
from lerobot.utils.cycle_timer import CycleTimer

from .base import InferenceEngine


def dispatch_permitted(
    engine: InferenceEngine, interpolator: ActionInterpolator, hold: Callable[[], None] | None = None
) -> bool:
    """Ask the engine once; on denial clear interpolation, hold and acknowledge.

    ``hold`` refreshes the robot's retained position target without a camera read or
    inference. Pass ``None`` when the driver declares no local wait: the engine is
    still acknowledged so a latched fault can progress to shutdown. A hold that
    raises is never acknowledged.
    """
    if engine.dispatch_allowed():
        return True
    interpolator.reset()
    if hold is not None:
        hold()
    engine.acknowledge_hold()
    return False


def next_interpolated_action(
    engine: InferenceEngine,
    interpolator: ActionInterpolator,
    obs_frame: dict | Callable[[], dict | None] | None,
    *,
    hold: Callable[[], None] | None = None,
    timer: CycleTimer | None = None,
    on_prediction: Callable[[PolicyPrediction], None] | None = None,
) -> torch.Tensor | None:
    """Return the next interpolated target, pulling one policy action when needed.

    ``obs_frame`` is the frame handed to ``engine.get_action`` or a zero-argument
    factory for it. The factory runs only when the interpolator needs a new policy
    action, so interpolation ticks pay nothing; asynchronous engines ignore the
    frame. Returns ``None`` while denied or starved. A starved tick is counted on
    ``timer`` when one is given, and the pull is timed as its ``infer`` step.

    The engine may return the action with the policy's prediction (an ``EnvTransition``,
    see ``PreTrainedPolicy.select_action``); only the action is interpolated, and a
    non-empty prediction is handed to ``on_prediction`` on the tick that pulled it.
    """
    if not dispatch_permitted(engine, interpolator, hold):
        return None
    if interpolator.needs_new_action():
        with timer.section("infer") if timer is not None else nullcontext():
            frame = obs_frame() if callable(obs_frame) else obs_frame
            result = engine.get_action(frame)
        if result is not None:
            step = policy_output_to_transition(result)
            interpolator.add(transition_to_policy_action(step).cpu())
            if on_prediction is not None and (prediction := transition_to_prediction(step)):
                on_prediction(prediction)
    if not dispatch_permitted(engine, interpolator, hold):
        return None
    target = interpolator.get()
    if target is None and timer is not None:
        timer.note_starved_tick()
    return target


def commit_action(
    engine: InferenceEngine,
    interpolator: ActionInterpolator,
    send: Callable[[], None],
    *,
    hold: Callable[[], None] | None = None,
) -> bool:
    """Run the final gate immediately before ``send`` and record the dispatch after it.

    ``send`` writes an already prepared target to the actuators. It runs only if the
    engine still permits motion; the engine then learns that a policy command reached
    the robot. Returns whether ``send`` ran.
    """
    if not dispatch_permitted(engine, interpolator, hold):
        return False
    send()
    engine.record_dispatch()
    return True


def dispatch_tick(
    engine: InferenceEngine,
    interpolator: ActionInterpolator,
    obs_frame: dict | Callable[[], dict | None] | None = None,
    *,
    send: Callable[[torch.Tensor], None],
    hold: Callable[[], None] | None = None,
    timer: CycleTimer | None = None,
) -> torch.Tensor | None:
    """One complete autonomous motor tick: gate, pull, interpolate, gate, send, record.

    ``send`` receives the canonical target in the engine's action order and must
    apply it promptly; conversion to the driver's action format belongs inside it.
    Returns the dispatched target, or ``None`` when the tick was held or starved.
    """
    target = next_interpolated_action(engine, interpolator, obs_frame, hold=hold, timer=timer)
    if target is None:
        return None
    with timer.section("send") if timer is not None else nullcontext():
        if not commit_action(engine, interpolator, lambda: send(target), hold=hold):
            return None
    return target
