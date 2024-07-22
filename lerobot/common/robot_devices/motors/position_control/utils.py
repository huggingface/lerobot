import numpy as np


def physical_to_logical(
    joints: list[str],
    physical_position: np.ndarray,
    table: {},
):
    return np.array(
        [
            (table[joints[i]]["physical_to_logical"]((physical_position[i] % 4096) * 360 / 4096))
            for i in range(len(physical_position))
        ],
        dtype=np.float32,
    )


def logical_to_physical(joints: list[str], logical_position: np.ndarray, table: {}):
    return np.array(
        [
            (int(table[joints[i]]["logical_to_physical"](logical_position[i]) * 4096 / 360))
            for i in range(len(logical_position))
        ],
        dtype=np.int32,
    )


def calculate_physical_goal_with_offset_computation(
    joints: list[str],
    physical_position: np.ndarray,
    logical_goal: np.ndarray,
    table: {},
):
    """
    Take in account the current physical position of the robot and adapt the physical goal accordingly.
    """
    goal = logical_to_physical(joints, logical_goal, table)

    base = logical_to_physical(joints, physical_to_logical(joints, physical_position, table), table)

    return np.array(
        [physical_position[i] + goal[i] - base[i] for i in range(len(physical_position))],
        dtype=np.int32,
    )
