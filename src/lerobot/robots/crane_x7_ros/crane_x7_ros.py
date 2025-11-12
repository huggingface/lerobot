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
from geometry_msgs.msg import Pose
from lerobot.cameras.utils import make_cameras_from_configs

try:
    # ROS 2 MoveIt Python API
    from moveit_commander.move_group import MoveGroupCommander  # type: ignore
except Exception:  # pragma: no cover - MoveIt 未導入環境でもimportエラーを無視
    MoveGroupCommander = None  # type: ignore


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
        self._move_group: Any | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    # ========= Features =========
    @property
    def _eef_pose_ft(self) -> dict[str, type]:
        return {
            "eef.pos.x": float,
            "eef.pos.y": float,
            "eef.pos.z": float,
            "eef.ori.x": float,
            "eef.ori.y": float,
            "eef.ori.z": float,
            "eef.ori.w": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {cam_key: (cam.height, cam.width, 3) for cam_key, cam in self.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # joint観測に加えてEEF姿勢を含める
        feats: dict[str, type | tuple] = {}
        feats.update(self._eef_pose_ft)
        feats.update(self._cameras_ft)
        return feats

    @cached_property
    def action_features(self) -> dict[str, type]:
        # アクション空間はEEF姿勢に切り替え
        return self._eef_pose_ft

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

        # MoveIt MoveGroup の初期化（EEF制御用）
        if MoveGroupCommander is None:
            raise RuntimeError(
                "moveit_commander が見つかりません。demo.launch.pyでmove_groupを起動し、"
                "Python依存関係にMoveItのコマンダAPIが導入されていることを確認してください。"
            )
        try:
            if self.config.namespace:
                self._move_group = MoveGroupCommander(self.config.moveit_group_name, ns=self.config.namespace)  # type: ignore[arg-type]
            else:
                self._move_group = MoveGroupCommander(self.config.moveit_group_name)  # type: ignore[call-arg]
        except TypeError:
            # ns引数が無いMoveItバージョン向けフォールバック
            self._move_group = MoveGroupCommander(self.config.moveit_group_name)  # type: ignore[call-arg]

        if self._move_group is None:
            raise RuntimeError("MoveGroupCommander の初期化に失敗しました。group名を確認してください。")

        try:
            self._move_group.set_planning_time(self.config.moveit_planning_time)  # type: ignore[attr-defined]
        except Exception:
            pass
        # 速度/加速度スケーリング
        try:
            self._move_group.set_max_velocity_scaling_factor(self.config.moveit_velocity_scaling)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self._move_group.set_max_acceleration_scaling_factor(self.config.moveit_acceleration_scaling)  # type: ignore[attr-defined]
        except Exception:
            pass
        if self.config.moveit_pose_reference_frame:
            try:
                self._move_group.set_pose_reference_frame(self.config.moveit_pose_reference_frame)  # type: ignore[attr-defined]
            except Exception:
                pass

        # カメラ接続
        for cam in self.cameras.values():
            try:
                cam.connect()
            except Exception as e:
                # 他カメラは生かすためにログに留める運用（ここでは例外にせず継続）
                if hasattr(self._node, "get_logger"):
                    self._node.get_logger().warn(f"Camera connect failed: {cam} ({e})")  # type: ignore[attr-defined]

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

    def _current_eef_pose(self) -> dict[str, float]:
        if self._move_group is None:
            raise RuntimeError("MoveGroup is not initialized")
        try:
            if self.config.moveit_end_effector_link:
                pose_stamped = self._move_group.get_current_pose(self.config.moveit_end_effector_link)  # type: ignore[attr-defined]
            else:
                pose_stamped = self._move_group.get_current_pose()  # type: ignore[attr-defined]
            p = pose_stamped.pose
            return {
                "eef.pos.x": float(p.position.x),
                "eef.pos.y": float(p.position.y),
                "eef.pos.z": float(p.position.z),
                "eef.ori.x": float(p.orientation.x),
                "eef.ori.y": float(p.orientation.y),
                "eef.ori.z": float(p.orientation.z),
                "eef.ori.w": float(p.orientation.w),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to obtain current EEF pose: {e}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        obs: dict[str, Any] = {}
        # EEF pose を観測にも含める
        try:
            eef = self._current_eef_pose()
            obs.update(eef)
        except Exception:
            # MoveIt未初期化や取得失敗時はEEFをスキップ（堅牢性優先）
            pass
        # カメラ画像を追加
        for cam_key, cam in self.cameras.items():
            try:
                obs[cam_key] = cam.async_read()
            except Exception:
                # 取得失敗時はスキップ
                continue
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected.")

        # まずEEF姿勢アクションを優先（期待仕様）
        eef_keys = {"eef.pos.x", "eef.pos.y", "eef.pos.z", "eef.ori.x", "eef.ori.y", "eef.ori.z", "eef.ori.w"}
        if eef_keys.issubset(action.keys()):
            if self._move_group is None:
                raise RuntimeError("MoveGroup is not initialized for EEF control")
            pose = Pose()
            pose.position.x = float(action["eef.pos.x"])
            pose.position.y = float(action["eef.pos.y"])
            pose.position.z = float(action["eef.pos.z"])
            pose.orientation.x = float(action["eef.ori.x"])
            pose.orientation.y = float(action["eef.ori.y"])
            pose.orientation.z = float(action["eef.ori.z"])
            pose.orientation.w = float(action["eef.ori.w"])

            # 目標設定と実行（IK→関節値ターゲット優先、失敗時は姿勢ターゲット）
            try:
                # 開始状態を現在値に設定
                try:
                    self._move_group.set_start_state_to_current_state()  # type: ignore[attr-defined]
                except Exception:
                    pass

                # まずIKを試みて関節値ターゲットをセット
                ik_success = False
                try:
                    if self.config.moveit_end_effector_link:
                        ik_success = bool(self._move_group.set_joint_value_target(pose, self.config.moveit_end_effector_link))  # type: ignore[attr-defined]
                    else:
                        ik_success = bool(self._move_group.set_joint_value_target(pose))  # type: ignore[attr-defined]
                except Exception:
                    ik_success = False

                if not ik_success:
                    # IKが失敗した場合は姿勢ターゲットで計画・実行
                    if self.config.moveit_end_effector_link:
                        self._move_group.set_pose_target(pose, self.config.moveit_end_effector_link)  # type: ignore[attr-defined]
                    else:
                        self._move_group.set_pose_target(pose)  # type: ignore[attr-defined]

                if self.config.moveit_end_effector_link:
                    # go()は内部でplan+execute。start_state設定とターゲットが反映される
                    pass
                success = self._move_group.go(wait=True)  # type: ignore[attr-defined]
                try:
                    self._move_group.stop()  # type: ignore[attr-defined]
                    self._move_group.clear_pose_targets()  # type: ignore[attr-defined]
                except Exception:
                    pass
                if not bool(success):
                    raise RuntimeError("MoveIt failed to execute pose goal")
            except Exception as e:
                raise RuntimeError(f"Failed to execute EEF pose action via MoveIt: {e}")

            # 実行後の到達値を返す（EEF + joint）
            out: dict[str, Any] = {}
            try:
                out.update(self._current_eef_pose())
            except Exception:
                # EEFが取得できない場合は指令値を返す
                out.update(
                    {
                        "eef.pos.x": float(pose.position.x),
                        "eef.pos.y": float(pose.position.y),
                        "eef.pos.z": float(pose.position.z),
                        "eef.ori.x": float(pose.orientation.x),
                        "eef.ori.y": float(pose.orientation.y),
                        "eef.ori.z": float(pose.orientation.z),
                        "eef.ori.w": float(pose.orientation.w),
                    }
                )
            # 併せて最新jointも返す（下流のロガー互換性のため）
            joints = self._current_positions()
            out.update({f"{j}.pos": v for j, v in joints.items()})
            return out

        # EEF以外の指定は受け付けない
        raise ValueError(
            "EEF pose keys are required: "
            "eef.pos.x, eef.pos.y, eef.pos.z, eef.ori.x, eef.ori.y, eef.ori.z, eef.ori.w"
        )

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
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass


