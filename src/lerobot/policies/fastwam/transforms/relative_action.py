from typing import List, Dict

import torch
from .rotation import quaternion_to_matrix, matrix_to_quaternion

# pose: position and quaternion in x, y, z, i, j, k, r
# mat: homogeneous transformation matrix in 4×4
# pos: position in x, y, z
# quat: quaternion in i, j, k, r


class RelativePoseTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._forward(action, state[..., -1:, :])
        
        return batch
        
    def backward(self, batch: Dict):
        for k in self.keys:
            action = batch["action"][k]
            state = batch["state"][k]
            batch["action"][k] = self._backward(action, state[..., -1:, :])

        return batch

    def _forward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._absolute_to_relative(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    def _backward(self, pose: torch.Tensor, base_pose: torch.Tensor):
        # pose & base_pose: position and quaternion in x, y, z, i, j, k, r
        assert pose.shape[-1] == 7, f"Pose shape must be (..., 7), but got {pose.shape}"
        assert base_pose.shape[-1] == 7, f"Base pose shape must be (..., 7), but got {base_pose.shape}"
        pose_matrix = self._pose_to_matrix(pose)
        base_pose_matrix = self._pose_to_matrix(base_pose)
        pose_matrix = self._relative_to_absolute(pose_matrix, base_pose_matrix)
        pose = self._matrix_to_pose(pose_matrix)
        return pose
    
    @staticmethod
    def _pose_to_matrix(pose: torch.Tensor):
        position = pose[..., 0: 3]
        quaternion = pose[..., [6, 3, 4, 5]] # (i j k r) to (r i j k)
        rotation = quaternion_to_matrix(quaternion)
        matrix = torch.zeros(pose.shape[:-1] + (4, 4), dtype=pose.dtype, device=pose.device)
        matrix[..., 0: 3, 0: 3] = rotation
        matrix[..., 0: 3, 3] = position
        matrix[..., 3, 3] = 1
        return matrix

    @staticmethod
    def _matrix_to_pose(matrix: torch.Tensor):
        position = matrix[..., 0: 3, 3] / matrix[..., 3, 3][..., None]
        rotation = matrix[..., 0: 3, 0: 3]
        quaternion = matrix_to_quaternion(rotation)
        quaternion = quaternion[..., [1, 2, 3, 0]] # (r i j k) to (i j k r)
        pose = torch.cat([position, quaternion], dim=-1)
        return pose
    @staticmethod
    def _absolute_to_relative(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return torch.linalg.inv(base_pose_matrix) @ pose_matrix

    @staticmethod
    def _relative_to_absolute(pose_matrix: torch.Tensor, base_pose_matrix: torch.Tensor):
        return base_pose_matrix @ pose_matrix
    

class RelativeJointTransform:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def forward(self, batch: Dict):
        # for close-loop eval, "action" may not be in batch
        if "action" not in batch:
            return batch
        
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] - batch["state"][k][..., :1, :]

        return batch

    def backward(self, batch: Dict):
        for k in self.keys:
            # NOTE: fixed to the first frame
            batch["action"][k] = batch["action"][k] + batch["state"][k][..., :1, :]
        
        return batch
