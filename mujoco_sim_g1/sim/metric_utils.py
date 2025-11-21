from typing import List, Tuple

import mujoco

from .sim_utilts import get_body_geom_ids


def check_contact(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    bodies_1: List[str] | str,
    bodies_2: List[str] | str,
    return_all_contact_bodies: bool = False,
) -> Tuple[bool, List[Tuple[str, str]]] | bool:
    """
    Finds contact between two body groups. Any geom in the body is considered to be in contact.
    Args:
        mj_model (MujocoModel): Current simulation object
        mj_data (MjData): Current simulation data
        bodies_1 (str or list of int): an individual body name or list of body names.
        bodies_2 (str or list of int): another individual body name or list of body names.
    Returns:
        bool: True if any body in @bodies_1 is in contact with any body in @bodies_2.
    """
    if isinstance(bodies_1, str):
        bodies_1 = [bodies_1]
    if isinstance(bodies_2, str):
        bodies_2 = [bodies_2]

    geoms_1 = [get_body_geom_ids(mj_model, mj_model.body(g).id) for g in bodies_1]
    geoms_1 = [g for geom_list in geoms_1 for g in geom_list]
    geoms_2 = [get_body_geom_ids(mj_model, mj_model.body(g).id) for g in bodies_2]
    geoms_2 = [g for geom_list in geoms_2 for g in geom_list]
    contact_bodies = []
    for i in range(mj_data.ncon):
        contact = mj_data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = contact.geom1 in geoms_1
        c2_in_g2 = contact.geom2 in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = contact.geom2 in geoms_1
        c1_in_g2 = contact.geom1 in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            contact_bodies.append(
                (
                    mj_model.body(mj_model.geom(contact.geom1).bodyid).name,
                    mj_model.body(mj_model.geom(contact.geom2).bodyid).name,
                )
            )
            if not return_all_contact_bodies:
                break
    if return_all_contact_bodies:
        return len(contact_bodies) > 0, set(contact_bodies)
    else:
        return len(contact_bodies) > 0


def check_height(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    geom_name: str,
    lower_bound: float = -float("inf"),
    upper_bound: float = float("inf"),
):
    """
    Checks if the height of a geom is greater than a given height.
    """
    geom_id = mj_model.geom(geom_name).id
    return (
        mj_data.geom_xpos[geom_id][2] < upper_bound and mj_data.geom_xpos[geom_id][2] > lower_bound
    )
