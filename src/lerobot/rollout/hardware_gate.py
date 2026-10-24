"""Default-deny authorization gate for SO-ARM101 action dispatch."""

from os import environ


def send_action_checked_with_approval(robot_wrapper, action):
    """Reject dispatch unless explicit hardware approval is enabled."""
    if environ.get("SO101_HARDWARE_APPROVED") != "1":
        raise RuntimeError(
            "SO101 safety gate: hardware dispatch disabled"
        )

    dispatch = getattr(robot_wrapper, "send_action_checked", None)
    if callable(dispatch):
        return dispatch(action)
    return robot_wrapper.send_action(action)
