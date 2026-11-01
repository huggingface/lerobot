"""Default-deny authorization gate for SO-ARM101 action dispatch."""

from os import environ


def send_action_checked_with_approval(robot_wrapper, action):
    """Require explicit approval and checked dispatch."""

    if environ.get("SO101_HARDWARE_APPROVED") != "1":
        raise RuntimeError(
            "SO101 safety gate: hardware dispatch disabled"
        )

    dispatch = getattr(
        robot_wrapper,
        "send_action_checked",
        None,
    )

    if not callable(dispatch):
        raise RuntimeError(
            "SO101 safety gate: checked action dispatch unavailable"
        )

    return dispatch(action)
