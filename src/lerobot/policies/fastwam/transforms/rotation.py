# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple
import torch

from ..utils.rotation import (
    quaternion_to_matrix, 
    matrix_to_quaternion, 
    matrix_to_rotation_6d, 
    rotation_6d_to_matrix, 
    matrix_to_rotation_9d, 
    rotation_9d_to_matrix, 
    quaternion_to_axis_angle, 
    axis_angle_to_quaternion, 
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)

class PoseRotationTransform:
    def __init__(self, rotation_type: str, category_keys: Dict[str, List[str]]):
        self.rotation_type = rotation_type
        self.category_keys = category_keys
        self.src_type, self.dst_type = self._parse_rotation_type(rotation_type)

    def forward(self, batch):
        for cat, ks in self.category_keys.items():
            if cat == "action" and "action" not in batch:
                continue

            for k in ks:
                batch[cat][k] = self._forward(batch[cat][k])

        return batch
    
    def backward(self, batch):
        for cat, ks in self.category_keys.items():
            for k in ks:
                batch[cat][k] = self._backward(batch[cat][k])

        return batch


    def _forward(self, pose):
        assert pose.shape[-1] >= 6
        if self.src_type == self.dst_type:
            return pose
        position, rotation, others = self._split_pose(pose, self.src_type)
        matrix = self._rotation_to_matrix(rotation, self.src_type)
        rotation_out = self._matrix_to_rotation(matrix, self.dst_type)
        return torch.cat([position, rotation_out, others], axis=-1)
        
    def _backward(self, pose: torch.Tensor):
        if self.src_type == self.dst_type:
            return pose
        position, rotation, others = self._split_pose(pose, self.dst_type)
        matrix = self._rotation_to_matrix(rotation, self.dst_type)
        rotation_out = self._matrix_to_rotation(matrix, self.src_type)
        return torch.cat([position, rotation_out, others], axis=-1)

    def add_noise(self, pose: torch.Tensor, std_position=0.05, std_angle=0.05):
        position, rotation, others = self._split_pose(pose, self.src_type)
        matrix = self._rotation_to_matrix(rotation, self.src_type)
        axis_angles = matrix_to_axis_angle(matrix)
        position = position + std_position * torch.randn_like(position)
        axis_angles = axis_angles + std_angle * torch.randn_like(axis_angles)
        matrix = axis_angle_to_matrix(axis_angles)
        rotation_out = self._matrix_to_rotation(matrix, self.src_type)
        return torch.cat([position, rotation_out, others], axis=-1)

    def _parse_rotation_type(self, rotation_type: str) -> Tuple[str, str]:
        if "_to_" in rotation_type:
            src, dst = rotation_type.split("_to_", 1)
        else:
            src, dst = "quaternion", rotation_type
        return src, dst

    def _split_pose(self, pose: torch.Tensor, rotation_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        rotation_dim = self._rotation_dim(rotation_type)
        position = pose[..., 0:3]
        rotation = pose[..., 3: 3 + rotation_dim]
        others = pose[..., 3 + rotation_dim:] if pose.shape[-1] > 3 + rotation_dim else torch.empty_like(position[..., :0])
        assert rotation.shape[-1] == rotation_dim, f"Expected {rotation_dim} dims for {rotation_type}"
        return position, rotation, others

    def _rotation_dim(self, rotation_type: str) -> int:
        if rotation_type == "quaternion":
            return 4
        if rotation_type == "axis_angle":
            return 3
        if rotation_type in ("rotation_6d",):
            return 6
        if rotation_type in ("rotation_9d", "matrix"):
            return 9
        if rotation_type.startswith("euler_"):
            return 3
        raise ValueError(f"Unsupported rotation type '{rotation_type}'")

    def _rotation_to_matrix(self, rotation: torch.Tensor, rotation_type: str) -> torch.Tensor:
        if rotation_type == "quaternion":
            quaternion = rotation[..., [3, 0, 1, 2]]
            return quaternion_to_matrix(quaternion)
        if rotation_type == "axis_angle":
            return axis_angle_to_matrix(rotation)
        if rotation_type == "rotation_6d":
            return rotation_6d_to_matrix(rotation)
        if rotation_type in ("rotation_9d", "matrix"):
            return rotation_9d_to_matrix(rotation)
        if rotation_type.startswith("euler_"):
            convention = rotation_type.split("_", 1)[1].upper()
            return euler_angles_to_matrix(rotation, convention)
        raise ValueError(f"Unsupported rotation type '{rotation_type}'")

    def _matrix_to_rotation(self, matrix: torch.Tensor, rotation_type: str) -> torch.Tensor:
        if rotation_type == "quaternion":
            quaternion = matrix_to_quaternion(matrix)
            return quaternion[..., [1, 2, 3, 0]]
        if rotation_type == "axis_angle":
            return matrix_to_axis_angle(matrix)
        if rotation_type == "rotation_6d":
            return matrix_to_rotation_6d(matrix)
        if rotation_type in ("rotation_9d", "matrix"):
            return matrix_to_rotation_9d(matrix)
        if rotation_type.startswith("euler_"):
            convention = rotation_type.split("_", 1)[1].upper()
            return matrix_to_euler_angles(matrix, convention)
        raise ValueError(f"Unsupported rotation type '{rotation_type}'")
    
