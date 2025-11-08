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

import threading
import time
from functools import cached_property
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_crane_x7_ros import CraneX7ROSConfig


def _join_ns(ns: str, name: str) -> str:
    if not ns:
        return f"/{name}" if not name.startswith("/") else name
    ns_clean = ns if ns.startswith("/") else f"/{ns}"
    return f"{ns_clean}/{name}" if not name.startswith("/") else f"{ns_clean}{name}"


class _SpinThread:
    def __init__(self, node: Node):
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        try:
            self._executor.shutdown()
        except Exception:
            pass


class CraneX7ROS(Robot):
    """
    CRANE-X7 を ros2_control 経由で操作する Robot 実装。

    - 観測: `/joint_states` から関節角を取得
    - 制御: `FollowJointTrajectory` アクションを `crane_x7_arm_controller` に送信
    - IK/MoveIt: demo.launch.py 側で move_group が起動済み（本クラスは関節空間制御を提供）
    """

    config_class = CraneX7ROSConfig
    name = "crane_x7_ros"

    def __init__(self, config: CraneX7ROSConfig):
        super().__init__(config)
        self.config = config
        self._node: Node | None = None
        self._spin: _SpinThread | None = None
        self._arm_client: ActionClient | None = None
        self._last_joint_state: JointState | None = None
        self._joint_state_lock = threading.Lock()

    # ========= Features =========
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_names}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._motors_ft

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ========= Lifecycle =========
    @property
    def is_connected(self) -> bool:
        if self._node is None or self._arm_client is None:
            return False
        arm_ready = self._arm_client.server_is_ready()
        with self._joint_state_lock:
            has_js = self._last_joint_state is not None
        return arm_ready and has_js

    def connect(self, calibrate: bool = True) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._node = Node("lerobot_crane_x7_ros")

        # joint_states subscriber
        def _on_js(msg: JointState) -> None:
            with self._joint_state_lock:
                self._last_joint_state = msg

        self._node.create_subscription(JointState, _join_ns(self.config.namespace, "joint_states"), _on_js, 10)

        # spin thread
        self._spin = _SpinThread(self._node)
        self._spin.start()

        # action client for arm trajectory
        arm_action_name = _join_ns(
            self.config.namespace, f"{self.config.arm_controller_name}/follow_joint_trajectory"
        )
        self._arm_client = ActionClient(self._node, FollowJointTrajectory, arm_action_name)

        # wait for servers and first joint state
        if not self._arm_client.wait_for_server(timeout_sec=self.config.action_timeout):
            raise RuntimeError("Arm FollowJointTrajectory action server not available")

        js_deadline = time.time() + self.config.joint_state_wait_timeout
        while time.time() < js_deadline:
            with self._joint_state_lock:
                if self._last_joint_state is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("joint_states not received within timeout")

        # このロボットにおけるキャリブレーションはros2_control側に依存するため常にTrue扱い
        # calibrate 引数は維持のみ
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # ros2_control 側に委譲（ここでは何もしない）
        return

    def configure(self) -> None:
        # 追加設定なし
        return

    # ========= IO =========
    def _current_positions(self) -> dict[str, float]:
        with self._joint_state_lock:
            js = self._last_joint_state
        if js is None:
            raise RuntimeError("No joint_states available")

        name_to_pos = {n: p for n, p in zip(js.name, js.position)}
        out: dict[str, float] = {}
        for name in self.config.joint_names:
            if name in name_to_pos:
                out[name] = float(name_to_pos[name])
        return out

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        positions = self._current_positions()
        return {f"{j}.pos": v for j, v in positions.items()}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        # 入力から関節名→目標値へ整形
        goal_pos = {k.removesuffix(".pos"): float(v) for k, v in action.items() if k.endswith(".pos")}

        # 安全対策: 必要なら相対制限（将来の設定拡張に合わせる場合はここで）
        # 現状は KochFollower 同等の仕組みに合わせたいときに ensure_safe_goal_position を利用
        # present = self._current_positions()
        # goal_pos = ensure_safe_goal_position({k: (gp, present.get(k, gp)) for k, gp in goal_pos.items()}, max_rel)

        # トラジェクトリ構築（単一点）
        traj = JointTrajectory()
        traj.joint_names = list(self.config.joint_names)

        point = JointTrajectoryPoint()
        # 順序に合わせて positions を並べる
        point.positions = [goal_pos.get(n, None) for n in traj.joint_names]
        if any(v is None for v in point.positions):
            missing = [n for n, v in zip(traj.joint_names, point.positions) if v is None]
            raise ValueError(f"Missing targets for joints: {missing}")
        point.time_from_start.sec = int(self.config.default_time_from_start)
        point.time_from_start.nanosec = int((self.config.default_time_from_start % 1.0) * 1e9)
        traj.points = [point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # 送信
        assert self._arm_client is not None
        send_future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_future, timeout_sec=self.config.action_timeout)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("FollowJointTrajectory goal rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future, timeout_sec=self.config.action_timeout)
        # 成否は一旦例外にせず、成功/失敗でも同じ戻り形式を返す

        return {f"{j}.pos": float(v) for j, v in zip(traj.joint_names, point.positions)}

    def disconnect(self) -> None:
        if self._spin is not None:
            self._spin.stop()
            self._spin = None
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        # rclpy は他でも使っている可能性があるためここでは shutdown しない
        # 必要に応じて呼び出し元で rclpy.shutdown() を行う


