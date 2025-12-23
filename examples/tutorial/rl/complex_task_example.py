"""
复杂任务环境示例：抓取并放置到目标位置

这个示例展示了如何创建和配置一个更复杂的机器人操作任务。
任务包括多个阶段：接近物体、抓取、移动到目标位置、放置。
"""

import numpy as np
import gymnasium as gym
from typing import Any
from lerobot.rl.gym_manipulator import RobotEnv
from lerobot.utils.constants import OBS_STATE
from lerobot.teleoperators.utils import TeleopEvents


class ComplexPickAndPlaceEnv(RobotEnv):
    """
    复杂任务环境：抓取物体并放置到目标位置
    
    任务阶段：
    1. 接近物体（距离奖励）
    2. 抓取物体（抓取奖励）
    3. 移动到目标位置（距离奖励）
    4. 放置物体（完成奖励）
    """

    def __init__(
        self,
        robot,
        use_gripper: bool = True,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
        target_position: np.ndarray | None = None,
        object_position: np.ndarray | None = None,
    ):
        super().__init__(robot, use_gripper, display_cameras, reset_pose, reset_time_s)
        
        # 任务特定参数
        self.target_position = (
            target_position if target_position is not None 
            else np.array([0.5, 0.0, 0.3])
        )
        self.object_position = (
            object_position if object_position is not None 
            else np.array([0.3, 0.0, 0.1])
        )
        
        # 任务状态
        self.gripper_closed = False
        self.object_grasped = False
        self.task_stage = "approach"  # approach, grasp, transport, place
        
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """重置环境并初始化任务状态"""
        obs, info = super().reset(seed=seed, options=options)
        
        # 重置任务特定状态
        self.object_position = np.array([0.3, 0.0, 0.1])  # 物体初始位置
        self.gripper_closed = False
        self.object_grasped = False
        self.task_stage = "approach"
        
        # 添加任务信息
        info["task_stage"] = self.task_stage
        info["object_position"] = self.object_position.copy()
        info["target_position"] = self.target_position.copy()
        
        return obs, info
    
    def _get_end_effector_position(self, obs: dict[str, Any]) -> np.ndarray:
        """
        从观测中提取末端执行器位置
        
        注意：这需要根据你的机器人配置实现。
        可以使用前向运动学或从观测状态中直接读取。
        """
        state = obs.get(OBS_STATE, None)
        if state is not None:
            # 假设状态包含末端执行器位置（前3个元素）
            # 实际实现需要根据你的机器人配置调整
            if len(state) >= 3:
                return state[:3].astype(np.float32)
        
        # 如果没有状态信息，返回默认位置
        # 在实际应用中，应该通过前向运动学计算
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def _compute_reward(self, obs: dict[str, Any]) -> tuple[float, bool]:
        """
        计算复杂任务的奖励
        
        返回:
            reward: 当前步骤的奖励
            task_complete: 任务是否完成
        """
        reward = 0.0
        task_complete = False
        
        # 获取末端执行器位置
        ee_position = self._get_end_effector_position(obs)
        
        # 阶段1: 接近物体
        if self.task_stage == "approach":
            distance_to_object = np.linalg.norm(ee_position - self.object_position)
            
            # 距离奖励：越近奖励越高
            reward += max(0, 1.0 - distance_to_object * 10) * 0.1
            
            # 如果足够接近，进入抓取阶段
            if distance_to_object < 0.05:
                self.task_stage = "grasp"
                reward += 0.5  # 阶段完成奖励
        
        # 阶段2: 抓取物体
        elif self.task_stage == "grasp":
            distance_to_object = np.linalg.norm(ee_position - self.object_position)
            
            # 需要接近物体且夹爪闭合
            if distance_to_object < 0.03 and self.gripper_closed:
                self.object_grasped = True
                self.task_stage = "transport"
                reward += 2.0  # 成功抓取奖励
            elif distance_to_object < 0.05:
                # 接近但未抓取，给予小奖励
                reward += 0.1
            else:
                # 距离太远，给予惩罚
                reward -= 0.05
        
        # 阶段3: 运输到目标位置
        elif self.task_stage == "transport":
            if self.object_grasped:
                # 物体位置跟随末端执行器
                self.object_position = ee_position.copy()
                
                distance_to_target = np.linalg.norm(ee_position - self.target_position)
                
                # 距离奖励
                reward += max(0, 1.0 - distance_to_target * 5) * 0.1
                
                # 如果到达目标位置，进入放置阶段
                if distance_to_target < 0.05:
                    self.task_stage = "place"
                    reward += 1.0  # 到达目标奖励
            else:
                # 物体掉落，重置到抓取阶段
                self.task_stage = "grasp"
                reward -= 1.0  # 掉落惩罚
        
        # 阶段4: 放置物体
        elif self.task_stage == "place":
            distance_to_target = np.linalg.norm(ee_position - self.target_position)
            
            # 在目标位置且夹爪打开
            if distance_to_target < 0.05 and not self.gripper_closed:
                task_complete = True
                reward += 10.0  # 任务完成大奖励
            elif distance_to_target < 0.05:
                # 在目标位置但未放置
                reward += 0.5
            else:
                # 离开目标位置，给予惩罚
                reward -= 0.1
        
        return reward, task_complete
    
    def step(self, action) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """执行一步，包含复杂任务逻辑"""
        # 检查夹爪动作
        if self.use_gripper and len(action) > 3:
            gripper_action = action[-1]
            self.gripper_closed = gripper_action > 0.5
        
        # 执行基础步骤
        obs, base_reward, terminated, truncated, info = super().step(action)
        
        # 计算任务特定奖励
        task_reward, task_complete = self._compute_reward(obs)
        total_reward = base_reward + task_reward
        
        # 更新终止条件
        terminated = terminated or task_complete
        
        # 添加任务信息到 info
        info["task_stage"] = self.task_stage
        info["object_grasped"] = self.object_grasped
        info["gripper_closed"] = self.gripper_closed
        info["object_position"] = self.object_position.copy()
        info["target_position"] = self.target_position.copy()
        
        # 计算距离信息
        ee_position = self._get_end_effector_position(obs)
        if self.task_stage == "approach" or self.task_stage == "grasp":
            info["distance_to_object"] = float(np.linalg.norm(ee_position - self.object_position))
        elif self.task_stage == "transport" or self.task_stage == "place":
            info["distance_to_target"] = float(np.linalg.norm(ee_position - self.target_position))
        
        return obs, total_reward, terminated, truncated, info


# 使用示例
if __name__ == "__main__":
    """
    使用示例：如何在配置中使用这个复杂任务环境
    
    1. 修改 gym_manipulator.py 中的 make_robot_env 函数
    2. 在配置文件中指定任务类型
    """
    
    # 配置示例
    config_example = {
        "env": {
            "name": "complex_pick_and_place",
            "task": "ComplexPickAndPlace",
            "fps": 10,
            "processor": {
                "gripper": {
                    "use_gripper": True,
                    "gripper_penalty": -0.01
                },
                "reset": {
                    "control_time_s": 30.0,  # 复杂任务需要更长时间
                    "terminate_on_success": True
                }
            }
        }
    }
    
    print("复杂任务环境配置示例:")
    print(config_example)
    print("\n要使用此环境，需要:")
    print("1. 将 ComplexPickAndPlaceEnv 集成到 gym_manipulator.py")
    print("2. 在配置文件中指定任务类型")
    print("3. 根据你的机器人调整 _get_end_effector_position 方法")

