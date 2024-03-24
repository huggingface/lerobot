# import mujoco_py
import mujoco
import numpy as np


def apply_action(sim, action):
    if sim.model.nmocap > 0:
        pos_action, gripper_action = np.split(action, (sim.model.nmocap * 7,))
        if sim.data.ctrl is not None:
            for i in range(gripper_action.shape[0]):
                sim.data.ctrl[i] = gripper_action[i]
        pos_action = pos_action.reshape(sim.model.nmocap, 7)
        pos_delta, quat_delta = pos_action[:, :3], pos_action[:, 3:]
        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset(model, data):
    if model.nmocap > 0 and model.eq_data is not None:
        for i in range(model.eq_data.shape[0]):
            # if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                # model.eq_data[i, :] = np.array([0., 0., 0., 1., 0., 0., 0.])
                model.eq_data[i, :] = np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
    # sim.forward()
    mujoco.mj_forward(model, data)


def reset_mocap2body_xpos(sim):
    if sim.model.eq_type is None or sim.model.eq_obj1id is None or sim.model.eq_obj2id is None:
        return

    # For all weld constraints
    for eq_type, obj1_id, obj2_id in zip(
        sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id, strict=False
    ):
        # if eq_type != mujoco_py.const.EQ_WELD:
        if eq_type != mujoco.mjtEq.mjEQ_WELD:
            continue
        body2 = sim.model.body_id2name(obj2_id)
        if body2 == "B0" or body2 == "B9" or body2 == "B1":
            continue
        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id
        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
