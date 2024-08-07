#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
NOTE(YL): Adapted from:
    OpenVLA: https://github.com/openvla/openvla

Episode transforms for DROID dataset.
"""

from typing import Any, Dict

import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


def rmat_to_euler(rot_mat):
    return tfg.euler.from_rotation_matrix(rot_mat)


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def invert_rmat(rot_mat):
    return tfg.rotation_matrix_3d.inverse(rot_mat)


def rotmat_to_rot6d(mat):
    """
    Converts rotation matrix to R6 rotation representation (first two rows in rotation matrix).
    Args:
        mat: rotation matrix

    Returns: 6d vector (first two rows of rotation matrix)

    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def velocity_act_to_wrist_frame(velocity, wrist_in_robot_frame):
    """
    Translates velocity actions (translation + rotation) from base frame of the robot to wrist frame.
    Args:
        velocity: 6d velocity action (3 x translation, 3 x rotation)
        wrist_in_robot_frame: 6d pose of the end-effector in robot base frame

    Returns: 9d velocity action in robot wrist frame (3 x translation, 6 x rotation as R6)

    """
    r_frame = euler_to_rmat(wrist_in_robot_frame[:, 3:6])
    r_frame_inv = invert_rmat(r_frame)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (r_frame_inv @ velocity[:, :3][..., None])[..., 0]

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dr_ = euler_to_rmat(velocity[:, 3:6])
    dr_ = r_frame_inv @ (dr_ @ r_frame)
    dr_r6 = rotmat_to_rot6d(dr_)
    return tf.concat([vel_t, dr_r6], axis=-1)


def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))


def droid_baseact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dr_ = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]

    trajectory["action"] = tf.concat(
        (
            dt,
            dr_,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_wristact_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *wrist* frame of the robot.
    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory["action_dict"]["cartesian_velocity"], trajectory["observation"]["cartesian_position"]
    )
    trajectory["action"] = tf.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dr_ = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            dt,
            dr_,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def zero_action_filter(traj: Dict) -> bool:
    """
    Filters transitions whose actions are all-0 (only relative actions, no gripper action).
    Note: this filter is applied *after* action normalization, so need to compare to "normalized 0".
    """
    droid_q01 = tf.convert_to_tensor(
        [
            -0.7776297926902771,
            -0.5803514122962952,
            -0.5795090794563293,
            -0.6464047729969025,
            -0.7041108310222626,
            -0.8895104378461838,
        ]
    )
    droid_q99 = tf.convert_to_tensor(
        [
            0.7597932070493698,
            0.5726242214441299,
            0.7351000607013702,
            0.6705610305070877,
            0.6464948207139969,
            0.8897542208433151,
        ]
    )
    droid_norm_0_act = (
        2 * (tf.zeros_like(traj["action"][:, :6]) - droid_q01) / (droid_q99 - droid_q01 + 1e-8) - 1
    )

    return tf.reduce_any(tf.math.abs(traj["action"][:, :6] - droid_norm_0_act) > 1e-5)
