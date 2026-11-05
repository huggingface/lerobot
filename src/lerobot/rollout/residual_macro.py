import numpy as np


class ResidualMacroHold:
    """
    Hold one 6-D PPO action for a fixed number of control frames.

    This class:
    - does NOT connect to the robot
    - does NOT apply residual scaling
    - does NOT run PPO itself

    It only controls when a new PPO action should be sampled.
    """

    def __init__(self, hold_frames=10):
        if isinstance(hold_frames, bool):
            raise ValueError("hold_frames must be a positive integer")

        hold_frames = int(hold_frames)

        if hold_frames <= 0:
            raise ValueError("hold_frames must be > 0")

        self.hold_frames = hold_frames

        self._held_action = None
        self._remaining = 0
        self._decision_count = 0

    def reset(self):
        self._held_action = None
        self._remaining = 0
        self._decision_count = 0

    @property
    def needs_sample(self):
        return self._remaining == 0

    @property
    def decision_count(self):
        return self._decision_count

    def set_new_action(self, action):
        if not self.needs_sample:
            raise RuntimeError(
                "Current action is still being held"
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (6,):
            raise ValueError(
                f"Expected action shape (6,), got {action.shape}"
            )

        if not np.isfinite(action).all():
            raise ValueError(
                "Action contains NaN or Inf"
            )

        self._held_action = action.copy()
        self._remaining = self.hold_frames
        self._decision_count += 1

    def next_frame(self):
        if self._held_action is None:
            raise RuntimeError(
                "A PPO action must be sampled first"
            )

        if self._remaining <= 0:
            raise RuntimeError(
                "Held action expired; sample a new action"
            )

        action = self._held_action.copy()

        self._remaining -= 1

        return action
