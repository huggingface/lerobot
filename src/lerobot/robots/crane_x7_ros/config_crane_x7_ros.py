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

from dataclasses import dataclass, field

from ..config import RobotConfig


@RobotConfig.register_subclass("crane_x7_ros")
@dataclass
class CraneX7ROSConfig(RobotConfig):
    # 対象のアーム関節名（ros2_control/JointTrajectoryControllerに渡す順序）
    joint_names: list[str]

    # ros2_controlのアーム用コントローラ名（spawnerで起動済みの名前）
    arm_controller_name: str = "crane_x7_arm_controller"

    # グリッパ用コントローラ名（必要なら使用）
    gripper_controller_name: str | None = "crane_x7_gripper_controller"

    # ROS ネームスペース（空文字でルート）
    namespace: str = ""

    # 単発ポイントのトラジェクトリに与えるデフォルト到達時間 [秒]
    default_time_from_start: float = 0.5

    # アクションの完了を待つ最大時間 [秒]
    action_timeout: float = 5.0

    # 観測の joint_states 受信待ちタイムアウト [秒]
    joint_state_wait_timeout: float = 3.0

    # MoveIt を用いたIKをこのRobotで扱うか（本実装では未使用のプレースホルダ）
    use_moveit_ik: bool = False

    # ジョイント値に角度[rad]を想定（true: rad / false: controller基準）
    use_radians: bool = True

    # 余白: 将来の拡張用
    extra: dict[str, object] = field(default_factory=dict)


