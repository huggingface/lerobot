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

These helpers import neither PyTorch nor the inference package, so a control loop
that runs a policy without PyTorch shares them with ``lerobot-rollout``.
:mod:`lerobot.inference.dispatch` wraps them for engines that return PyTorch
policy outputs.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

    from .action_interpolator import ActionInterpolator
    from .cycle_timer import CycleTimer


class ActionSource(Protocol):
    """The part of an inference engine these helpers use."""

    def dispatch_allowed(self) -> bool:
        """Whether a policy action may reach the robot now."""
        ...

    def acknowledge_hold(self) -> None:
        """Confirm that the robot was held after a denial."""
        ...

    def get_action(self, obs_frame: dict | None) -> Any:
        """The next policy action, or ``None`` while none is ready."""
        ...

    def record_dispatch(self) -> None:
        """Note that a policy action reached the robot."""
        ...


def dispatch_permitted(
    engine: ActionSource, interpolator: ActionInterpolator, hold: Callable[[], None] | None = None
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
    engine: ActionSource,
    interpolator: ActionInterpolator,
    obs_frame: dict | Callable[[], dict | None] | None,
    *,
    hold: Callable[[], None] | None = None,
    timer: CycleTimer | None = None,
    unpack: Callable[[Any], Any] | None = None,
) -> Any:
    """Return the next interpolated target, pulling one policy action when needed.

    ``obs_frame`` is the frame handed to ``engine.get_action`` or a zero-argument
    factory for it. The factory runs only when the interpolator needs a new policy
    action, so interpolation ticks pay nothing; asynchronous engines ignore the
    frame. Returns ``None`` while denied or starved. A starved tick is counted on
    ``timer`` when one is given, and the pull is timed as its ``infer`` step.

    ``unpack`` turns what the engine returned into the action to interpolate, for an
    engine that returns more than the action; by default the result is the action.
    """
    if not dispatch_permitted(engine, interpolator, hold):
        return None
    if interpolator.needs_new_action():
        with timer.section("infer") if timer is not None else nullcontext():
            frame = obs_frame() if callable(obs_frame) else obs_frame
            result = engine.get_action(frame)
        if result is not None:
            interpolator.add(result if unpack is None else unpack(result))
    if not dispatch_permitted(engine, interpolator, hold):
        return None
    target = interpolator.get()
    if target is None and timer is not None:
        timer.note_starved_tick()
    return target


def commit_action(
    engine: ActionSource,
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
    engine: ActionSource,
    interpolator: ActionInterpolator,
    obs_frame: dict | Callable[[], dict | None] | None = None,
    *,
    send: Callable[[np.ndarray], None],
    hold: Callable[[], None] | None = None,
    timer: CycleTimer | None = None,
    unpack: Callable[[Any], Any] | None = None,
) -> Any:
    """One complete autonomous motor tick: gate, pull, interpolate, gate, send, record.

    ``send`` receives the canonical target in the engine's action order and must
    apply it promptly; conversion to the driver's action format belongs inside it.
    Returns the dispatched target, or ``None`` when the tick was held or starved.
    """
    target = next_interpolated_action(engine, interpolator, obs_frame, hold=hold, timer=timer, unpack=unpack)
    if target is None:
        return None
    with timer.section("send") if timer is not None else nullcontext():
        if not commit_action(engine, interpolator, lambda: send(target), hold=hold):
            return None
    return target
