"""
UR10-specific processor steps.

Adds the env-pipeline gripper-penalty term. The generic `GripperPenaltyProcessorStep` in
`lerobot.processor.hil_processor` is wired for SO101's continuous-position gripper and
expects `raw_joint_positions[GRIPPER_KEY]` in complementary_data — neither matches our
discrete tri-state gripper or how UR10 telemetry flows. We need a UR10-native equivalent
that reads gripper state from `observation.state[-1]` and gripper command from `action[-1]`.

Note on yaw-enabled action layouts:
    UR10RobotEnv produces actions of shape (4,) or (5,) depending on `use_yaw`. The
    gripper is always the LAST element when `use_gripper=True`, so this step's
    `action[-1]` indexing is correct regardless of whether yaw is enabled.

Penalty semantics (matches `gym_hil/wrappers/hil_wrappers.py::GripperPenaltyWrapper` and
`lerobot/processor/hil_processor.py::GripperPenaltyProcessorStep`):

    Penalize gripper STATE CHANGES — every toggle costs `penalty`. The policy is thus
    encouraged to commit to a state and only toggle when reward justifies it.

    Concretely:
        action[-1]              : 0 = close, 1 = stay, 2 = open
        observation.state[-1]   : 0.0 = closed, 1.0 = open

        Penalty fires when:
          - action == close AND state == open    (toggle: open  → closed)
          - action == open  AND state == closed  (toggle: closed → open)
        Otherwise penalty = 0.0. `stay` and redundant commands (close-when-closed,
        open-when-open) are free.

The result lands in `complementary_data["discrete_penalty"]`, which the recording loop in
`gym_manipulator.control_loop` writes to the dataset as `complementary_info.discrete_penalty`
and which the SAC trainer adds to the per-step reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE


DISCRETE_PENALTY_KEY = "discrete_penalty"


def _scalar(t: Any) -> float:
    """Coerce a torch scalar / 0-d tensor / python number to float."""
    if isinstance(t, torch.Tensor):
        return float(t.detach().cpu().item())
    return float(t)


@dataclass
@ProcessorStepRegistry.register("ur10_gripper_penalty_processor")
class UR10GripperPenaltyProcessorStep(ProcessorStep):
    """Penalize gripper state-change commands (toggles).

    Encoding (matches the UR10 driver and gamepad teleop):
        action[-1] : 0 = close, 1 = stay, 2 = open
        observation.state[-1] : 0.0 = closed, 1.0 = open

    A penalty of `penalty` (a small negative number, e.g. -0.02) is recorded when the
    action would CHANGE the gripper state — i.e. comparing the new command against the
    **previous step's gripper state** (not the current/post-step state):

        - action == close AND prev state == open    (toggle: open  → closed)
        - action == open  AND prev state == closed  (toggle: closed → open)

    Otherwise the recorded penalty is 0.0. `stay` is never penalized; redundant commands
    (close-when-closed, open-when-open) are free — they don't change the gripper state.

    *Why the previous state matters:* `UR10RobotEnv.step()` calls `send_gripper()` and
    then `get_observation()`. `gripper.is_open` updates immediately on send, so the
    observation we see in the same transition already reflects the new state. Comparing
    the new command against the post-step state would mark every real toggle as
    "redundant". gym-hil's `GripperPenaltyWrapper` solves this by tracking
    `self.last_gripper_pos` across calls; we mirror that here with `_last_gripper_state`.

    Matches the semantics of gym-hil's `GripperPenaltyWrapper` and lerobot's continuous
    `GripperPenaltyProcessorStep`: each toggle is a small fixed cost so the policy
    commits to a state and only toggles when there's real reward to be earned.
    """

    penalty: float = -0.02
    # Cross-call state. `init=False` keeps this out of the dataclass `__init__` signature
    # so JSON config + `register_subclass` constructors don't see it. Reset to None on
    # `reset()` (called between episodes); first call after reset never fires a penalty.
    _last_gripper_state: float | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        action = new_transition.get(TransitionKey.ACTION)
        observation = new_transition.get(TransitionKey.OBSERVATION, {}) or {}
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {})

        # Defensive: leave the penalty at 0.0 instead of crashing the pipeline if anything
        # looks unexpected. Recording is safety-critical; a bad penalty calculation must
        # never abort an episode.
        def _bail() -> EnvTransition:
            complementary_data.setdefault(DISCRETE_PENALTY_KEY, 0.0)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
            return new_transition

        state = observation.get(OBS_STATE)
        if state is None or action is None:
            return _bail()

        # `action` is a torch.Tensor of shape (4,)/(5,) before AddBatchDimensionProcessorStep,
        # or (1, N) after. The gripper command is always at index -1 (see ur10_robot.py
        # action layout), so this step is yaw-mode agnostic. `state` is (22,) or (1, 22).
        # The `< 4` floor is the minimum dim when use_gripper=True: 3 xyz + 1 gripper.
        if not isinstance(action, torch.Tensor) or not isinstance(state, torch.Tensor):
            return _bail()
        if action.ndim == 1:
            if action.shape[0] < 4:
                return _bail()
            gripper_cmd_raw = action[-1]
        elif action.ndim == 2:
            if action.shape[0] == 0 or action.shape[-1] < 4:
                return _bail()
            gripper_cmd_raw = action[0, -1]
        else:
            return _bail()

        if state.ndim == 1:
            if state.shape[0] == 0:
                return _bail()
            gripper_state_raw = state[-1]
        elif state.ndim == 2:
            if state.shape[0] == 0 or state.shape[-1] == 0:
                return _bail()
            gripper_state_raw = state[0, -1]
        else:
            return _bail()

        gripper_cmd = int(round(_scalar(gripper_cmd_raw)))
        current_gripper_state = _scalar(gripper_state_raw)

        # Compare the new command against the PREVIOUS step's gripper state. The current
        # observation already reflects the post-action state (gripper.is_open updates
        # immediately on send), so using it would never detect a toggle.
        if self._last_gripper_state is None:
            # First call after reset — no baseline to compare against. Record a 0 penalty
            # and prime the tracker for the next call. Matches gym-hil where reset()
            # captures `last_gripper_pos` BEFORE the first step.
            penalty_value = 0.0
        else:
            prev_open = self._last_gripper_state >= 0.5
            prev_closed = self._last_gripper_state < 0.5
            toggle_close = prev_open and gripper_cmd == 0
            toggle_open = prev_closed and gripper_cmd == 2
            penalty_value = float(self.penalty) if (toggle_close or toggle_open) else 0.0

        # Update tracker for next call. Use the post-step state so the next call's
        # comparison matches the gym-hil pattern (last = end-of-this-step state).
        self._last_gripper_state = current_gripper_state

        complementary_data[DISCRETE_PENALTY_KEY] = penalty_value
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def reset(self) -> None:
        # Drop the cross-episode baseline so the next first-call after env.reset()
        # doesn't compare against a stale state from the previous episode.
        self._last_gripper_state = None

    def get_config(self) -> dict[str, Any]:
        return {"penalty": self.penalty}

    def transform_features(self, features):
        return features
