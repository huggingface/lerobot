import numpy as np
import robosuite.utils.transform_utils as T


def rotate6d_to_axis_angle(r6d):
    """
    r6d: np.ndarray, shape (N, 6)
    return: np.ndarray, shape (N, 3), axis-angle vectors
    """
    flag = 0
    if len(r6d.shape) == 1:
        r6d = r6d[None, ...]
        flag = 1

    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]

    # b1
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)

    # b2
    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot_prod * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)

    # b3
    b3 = np.cross(b1, b2, axis=-1)

    R = np.stack([b1, b2, b3], axis=-1)  # shape: (N, 3, 3)

    axis_angle_list = []
    for i in range(R.shape[0]):
        quat = T.mat2quat(R[i])
        axis_angle = T.quat2axisangle(quat)
        axis_angle_list.append(axis_angle)

    axis_angle_array = np.stack(axis_angle_list, axis=0)  # shape: (N, 3)

    if flag == 1:
        axis_angle_array = axis_angle_array[0]

    return axis_angle_array


def mat_to_rotate6d(abs_action):
    if len(abs_action.shape) == 2:
        return np.concatenate([abs_action[:3, 0], abs_action[:3, 1]], axis=-1)
    elif len(abs_action.shape) == 3:
        return np.concatenate([abs_action[:, :3, 0], abs_action[:, :3, 1]], axis=-1)
    else:
        raise NotImplementedError
