"""
Utility functions for working with Mujoco models.
copied from https://github.com/kevinzakka/mink/blob/main/mink/utils.py
"""

from typing import List

import mujoco


def get_body_body_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get immediate children bodies belonging to a given body.

    Args:
        model: Mujoco model.
        body_id: ID of body.

    Returns:
        A List containing all child body ids.
    """
    return [
        i
        for i in range(model.nbody)
        if model.body_parentid[i] == body_id and body_id != i  # Exclude the body itself.
    ]


def get_subtree_body_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all bodies belonging to subtree starting at a given body.

    Args:
        model: Mujoco model.
        body_id: ID of body where subtree starts.

    Returns:
        A List containing all subtree body ids.
    """
    body_ids: List[int] = []
    stack = [body_id]
    while stack:
        body_id = stack.pop()
        body_ids.append(body_id)
        stack += get_body_body_ids(model, body_id)
    return body_ids


def get_subtree_body_names(model: mujoco.MjModel, body_id: int) -> List[str]:
    """Get all bodies belonging to subtree starting at a given body.
    Args:
        model: Mujoco model.
        body_id: ID of body where subtree starts.

    Returns:
        A List containing all subtree body names.
    """
    return [model.body(i).name for i in get_subtree_body_ids(model, body_id)]


def get_body_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get immediate geoms belonging to a given body.

    Here, immediate geoms are those directly attached to the body and not its
    descendants.

    Args:
        model: Mujoco model.
        body_id: ID of body.

    Returns:
        A list containing all body geom ids.
    """
    geom_start = model.body_geomadr[body_id]
    geom_end = geom_start + model.body_geomnum[body_id]
    return list(range(geom_start, geom_end))


def get_subtree_geom_ids(model: mujoco.MjModel, body_id: int) -> List[int]:
    """Get all geoms belonging to subtree starting at a given body.

    Here, a subtree is defined as the kinematic tree starting at the body and including
    all its descendants.

    Args:
        model: Mujoco model.
        body_id: ID of body where subtree starts.

    Returns:
        A list containing all subtree geom ids.
    """
    geom_ids: List[int] = []
    stack = [body_id]
    while stack:
        body_id = stack.pop()
        geom_ids.extend(get_body_geom_ids(model, body_id))
        stack += get_body_body_ids(model, body_id)
    return geom_ids
