#!/usr/bin/env python3
"""NERO 一键遥操作+数据采集脚本 (运行在 vt conda 环境)

不依赖 ROS2，直接用 pyAgxArm SDK + OculusReader + ArmIK + OpenCVCamera，
在同一个循环里同步采集 obs + action + camera，写入 LeRobot v3.0 格式。

核心优势：
  - 动作和视频时间戳天然同步（同一个 Python 循环内获取）
  - 不需要 ROS2，避免 Python 3.10/3.12 环境冲突
  - 一键启动，VR 手柄 grip 键控制录制

用法：
  conda activate vt
  python nero_collect.py --dataset_name my_task --num_episodes 10
  python nero_collect.py --help
"""

import argparse
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ---- QuestArmTeleop scripts path ----
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "nero" / "QuestArmTeleop" / "src" / "oculus_reader" / "scripts"
if not _SCRIPTS_DIR.is_dir():
    # 尝试另一个路径
    _SCRIPTS_DIR = Path("/home/yuhang/projects/lerobot/nero/QuestArmTeleop/src/oculus_reader/scripts")
sys.path.insert(0, str(_SCRIPTS_DIR))

from arm_ik_pose_node import ArmIK
from lerobot_dataset_writer import LeRobotDatasetWriter
from oculus_reader import OculusReader
from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

# ---- Constants ----
NERO_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
STATE_DIM = 8
ACTION_DIM = 8


def _xyzrpy_to_mat(x, y, z, roll, pitch, yaw):
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    mat[:3, 3] = [x, y, z]
    return mat


_ADJ_MAT = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float,
)
_R_ADJ = _xyzrpy_to_mat(0, 0, 0, -np.pi, 0, -np.pi / 2)


def _correct_to_ros(transform, ros_to_arm_mat):
    transform = _ADJ_MAT @ transform
    transform = transform @ _R_ADJ
    transform = transform @ ros_to_arm_mat
    return transform


def _mat2xyzquat(matrix):
    pos = matrix[:3, 3]
    quat = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return pos, quat


def _calc_pose_incre(start_pose_matrix, current_pose_xyzrpy, zero_matrix):
    end_matrix = _xyzrpy_to_mat(*current_pose_xyzrpy)
    result_matrix = zero_matrix @ np.linalg.inv(start_pose_matrix) @ end_matrix
    return _mat2xyzquat(result_matrix)


# ---- Camera Manager ----
class CameraManager:
    """直接用 OpenCV 读相机，不走 ROS2 topic，避免异步延迟。"""

    def __init__(self, cameras_config: list[dict] | None = None):
        """
        cameras_config: [{"device": "/dev/video2", "name": "usb_cam_1", "width": 640, "height": 480, "fps": 30}, ...]
        """
        self._cams: dict[str, cv2.VideoCapture] = {}
        self._latest: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._threads: list[threading.Thread] = []
        self._names: list[str] = []

        if cameras_config is None:
            cameras_config = [
                {"device": "/dev/video2", "name": "usb_cam_1", "width": 640, "height": 480, "fps": 30},
                {"device": "/dev/video4", "name": "usb_cam_2", "width": 640, "height": 480, "fps": 30},
            ]

        for cam_cfg in cameras_config:
            device = cam_cfg["device"]
            name = cam_cfg["name"]
            width = cam_cfg.get("width", 640)
            height = cam_cfg.get("height", 480)
            fps = cam_cfg.get("fps", 30)

            cap = cv2.VideoCapture(device if isinstance(device, int) else device)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            if not cap.isOpened():
                print(f"[WARN] Cannot open camera '{name}' at {device}, skipping")
                continue

            self._cams[name] = cap
            self._names.append(name)
            # Warm up
            for _ in range(5):
                cap.read()
            print(f"[INFO] Camera '{name}' opened: {device} ({width}x{height}@{fps}fps)")

    def start(self):
        """启动后台采集线程。"""
        self._running = True
        for name, cap in self._cams.items():
            t = threading.Thread(target=self._capture_loop, args=(name, cap), daemon=True)
            t.start()
            self._threads.append(t)

    def _capture_loop(self, name: str, cap: cv2.VideoCapture):
        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._latest[name] = frame
            else:
                time.sleep(0.005)

    def get_latest_frames(self) -> dict[str, np.ndarray]:
        """获取所有相机的最新帧（线程安全拷贝）。"""
        with self._lock:
            return {name: frame.copy() for name, frame in self._latest.items() if frame is not None}

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        for cap in self._cams.values():
            cap.release()


# ---- NERO Arm Interface (direct SDK, no ROS2) ----
class NeroArm:
    """直接用 pyAgxArm SDK 控制 NERO 机械臂。"""

    def __init__(self, channel: str = "can0", interface: str = "socketcan",
                 speed_percent: int = 50, firmware_version: str = "default"):
        self._channel = channel
        self._interface = interface
        self._speed_percent = speed_percent

        arm_config = create_agx_arm_config(
            robot=ArmModel.NERO,
            comm="can",
            firmeware_version=firmware_version,
            interface=interface,
            channel=channel,
            auto_set_motion_mode=True,
            enable_joint_limits=True,
        )
        self._arm = AgxArmFactory.create_arm(arm_config)
        self._effector = None
        self._last_targets = [0.0] * 7

    def connect(self):
        print("[INFO] Connecting to NERO arm...")
        self._arm.connect()
        print("[INFO] Enabling NERO arm...")
        while not self._arm.enable():
            time.sleep(0.01)
        print("[INFO] NERO arm enabled.")
        self._effector = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._arm.set_speed_percent(self._speed_percent)

        # 读取当前位置作为初始 target
        joint_msg = self._arm.get_joint_angles()
        if joint_msg is not None and joint_msg.msg is not None:
            self._last_targets = [float(v) for v in list(joint_msg.msg)[:7]]
        print("[INFO] NERO arm connected.")

    def get_observation(self) -> np.ndarray:
        """读取 7 个关节角度 + gripper 状态 → float32 [8]"""
        joint_msg = self._arm.get_joint_angles()
        if joint_msg is not None and joint_msg.msg is not None:
            joint_vals = list(joint_msg.msg)
        else:
            joint_vals = [0.0] * 7

        obs = []
        for i in range(7):
            obs.append(float(joint_vals[i]) if i < len(joint_vals) else 0.0)

        # Gripper
        gripper_val = 0.0
        if self._effector is not None:
            try:
                gripper_msg = self._effector.get_gripper_ctrl_states()
                if gripper_msg is not None and gripper_msg.msg is not None:
                    gripper_val = float(gripper_msg.msg.value)
            except Exception:
                pass
        obs.append(gripper_val)
        return np.array(obs, dtype=np.float32)

    def send_action(self, action: np.ndarray):
        """发送动作 [j1..j7, gripper]"""
        joint_targets = list(action[:7])

        # 安全检查：和当前实际位置比较
        present_obs = self._arm.get_joint_angles()
        if present_obs is not None and present_obs.msg is not None:
            present_vals = [float(v) for v in list(present_obs.msg)[:7]]
            # 如果差距太大，用实际值
            for i in range(7):
                if abs(joint_targets[i]) < 1e-6 and abs(present_vals[i]) > 0.01:
                    joint_targets[i] = present_vals[i]
            base = list(present_vals)
        else:
            base = list(self._last_targets)

        for i in range(7):
            if abs(action[i]) > 1e-6:
                base[i] = joint_targets[i]

        self._arm.move_j(base)
        self._last_targets = list(base)

        # Gripper
        if self._effector is not None and len(action) > 7:
            try:
                self._effector.move_gripper_deg(float(action[7]))
            except Exception:
                pass

    def disconnect(self):
        try:
            self._arm.disable()
        except Exception:
            pass
        try:
            self._arm.disconnect()
        except Exception:
            pass
        print("[INFO] NERO arm disconnected.")


# ---- Quest VR Teleop (direct, no ROS2) ----
class QuestTeleop:
    """直接用 OculusReader + ArmIK，不走 ROS2。"""

    def __init__(self, urdf_path: str, package_dirs: str | None = None,
                 ip_address: str | None = None,
                 ros_to_arm_rpy: list[float] | None = None,
                 hand: str = "right",
                 start_button: str = "A", stop_button: str = "B",
                 trigger_axis: str = "rightTrig"):
        self._ros_to_arm_mat = _xyzrpy_to_mat(0.0, 0.0, 0.0, *(ros_to_arm_rpy or [-1.5708, 0.0, 0.0]))
        self._hand_key = "r" if hand == "right" else "l"
        self._start_button = start_button
        self._stop_button = stop_button
        self._trigger_axis = trigger_axis

        self._oculus_reader = None
        self._arm_ik = None
        self._teleop_active = False
        self._start_pose_matrix = np.eye(4)
        self._zero_matrix = np.eye(4)
        self._current_pose_xyzrpy = None

        # 延迟初始化
        self._urdf_path = urdf_path
        self._package_dirs = package_dirs
        self._ip_address = ip_address

    def connect(self):
        print("[INFO] Connecting to Quest VR...")
        self._oculus_reader = OculusReader(ip_address=self._ip_address, run=True)
        time.sleep(0.5)
        print("[INFO] Quest VR connected.")

        print("[INFO] Initializing ArmIK solver...")
        self._arm_ik = ArmIK(
            urdf_path=self._urdf_path,
            package_dirs=[self._package_dirs] if self._package_dirs else [],
            locked_joints=["gripper_base_joint", "gripper_joint1", "gripper_joint2"],
            ee_parent_joint="joint7",
            ee_frame_name="ee",
            tool_pre_rot_rpy=[0.0, 0.0, 0.0],
            tool_translation_xyz=[0.1755, 0.0, -0.0235],
            collision_pairs_flat=[],
            w_pos=2.0, w_ori=2.0, w_reg=0.01, w_smooth=0.1,
            ipopt_max_iter=50, ipopt_tol=1e-4,
            enable_visualization=False,
        )
        print(f"[INFO] ArmIK initialized: nq={self._arm_ik.nq}, joints={self._arm_ik.active_joint_names()}")

    def get_action(self, robot_obs: np.ndarray | None = None) -> np.ndarray | None:
        """获取遥操作动作 → float32 [8] (j1..j7, gripper)，无动作返回 None"""
        transforms, buttons = self._oculus_reader.get_transformations_and_buttons()
        if not transforms or not buttons:
            return None

        raw_transform = transforms.get(self._hand_key)
        if raw_transform is None:
            return None

        # 坐标变换
        corrected = _correct_to_ros(raw_transform, self._ros_to_arm_mat)
        xyz = corrected[:3, 3]
        rpy = Rotation.from_matrix(corrected[:3, :3]).as_euler("xyz")
        self._current_pose_xyzrpy = np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]])

        # 按钮状态机
        if buttons.get(self._start_button, False):
            if not self._teleop_active:
                print("[TELEOP] Started")
            self._start_pose_matrix = _xyzrpy_to_mat(*self._current_pose_xyzrpy)
            self._teleop_active = True

        if buttons.get(self._stop_button, False):
            if self._teleop_active:
                print("[TELEOP] Stopped")
            self._teleop_active = False
            self._zero_matrix = _xyzrpy_to_mat(*self._current_pose_xyzrpy)
            self._start_pose_matrix = self._zero_matrix

        # Gripper
        trigger_raw = buttons.get(self._trigger_axis, [0.0])
        trigger_value = float(trigger_raw[0]) if isinstance(trigger_raw, (list, tuple)) and trigger_raw else 0.0
        gripper_value = min(trigger_value, 1.0) * 100.0  # NERO gripper: 0-100

        if not self._teleop_active:
            return None

        # 计算 delta pose
        _xyz, _quat = _calc_pose_incre(
            self._start_pose_matrix, self._current_pose_xyzrpy, self._zero_matrix
        )

        # IK 求解
        target_pose = np.eye(4)
        target_pose[:3, 3] = _xyz
        target_pose[:3, :3] = Rotation.from_quat(_quat).as_matrix()

        # 同步机器人当前状态给 IK solver
        if robot_obs is not None and len(robot_obs) >= 7:
            q_current = [float(robot_obs[i]) for i in range(7)]
            if len(q_current) == self._arm_ik.nq:
                self._arm_ik.sync_state(q_current)

        try:
            joint_angles = self._arm_ik.solve(target_pose)
        except Exception as e:
            print(f"[WARN] IK failed: {e}")
            return None

        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for i in range(7):
            action[i] = float(joint_angles[i])
        action[7] = gripper_value
        return action

    def disconnect(self):
        if self._oculus_reader is not None:
            self._oculus_reader.stop()
        print("[INFO] Quest teleop disconnected.")


# ---- Main Collection Loop ----
def collect(
    urdf_path: str,
    package_dirs: str,
    dataset_name: str = "nero_teleop",
    output_dir: str = "/home/yuhang/datasets",
    task_description: str = "pick and place",
    fps: int = 30,
    num_episodes: int = 50,
    episode_time_s: float = 60.0,
    reset_time_s: float = 10.0,
    can_channel: str = "can0",
    speed_percent: int = 50,
    quest_ip: str | None = None,
    cameras: list[dict] | None = None,
    no_camera: bool = False,
):
    """一键遥操作 + 数据采集主函数。"""

    # 1. 初始化机械臂
    arm = NeroArm(channel=can_channel, speed_percent=speed_percent)
    arm.connect()

    # 2. 初始化 VR 遥操作
    teleop = QuestTeleop(
        urdf_path=urdf_path,
        package_dirs=package_dirs,
        ip_address=quest_ip,
    )
    teleop.connect()

    # 3. 初始化相机
    cam_manager = None
    cam_names = []
    if not no_camera:
        cam_manager = CameraManager(cameras)
        cam_names = cam_manager.names
        if cam_names:
            cam_manager.start()
            print(f"[INFO] Cameras started: {cam_names}")
        else:
            print("[WARN] No cameras available, recording without video")

    # 4. 构建数据集特征
    state_names = [f"joint{i+1}" for i in range(7)] + ["gripper"]
    action_names = [f"joint{i+1}" for i in range(7)] + ["gripper_cmd"]
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": state_names},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": action_names},
    }
    for cam_name in cam_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"],
        }

    # 5. 创建数据集 writer
    writer = LeRobotDatasetWriter(
        root_dir=output_dir,
        repo_id=dataset_name,
        features=features,
        robot_type="nero",
        fps=fps,
        video_encoding=bool(cam_names),
    )
    print(f"[INFO] Dataset: {output_dir}/{dataset_name}")

    # 6. 控制循环
    stop_event = threading.Event()

    def _signal_handler(sig, frame):
        print("\n[INFO] Interrupt received, stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, _signal_handler)

    print("\n" + "=" * 60)
    print("  NERO 数据采集系统")
    print("=" * 60)
    print(f"  数据集:     {dataset_name}")
    print(f"  目标集数:   {num_episodes}")
    print(f"  每集时长:   {episode_time_s}s")
    print(f"  采集频率:   {fps} Hz")
    print(f"  相机:       {cam_names if cam_names else '无'}")
    print(f"  录制控制:   VR 右手柄 Grip 键切换录制开/关")
    print(f"  遥操作:     A键=开始, B键=停止, 扳机=夹爪")
    print("=" * 60)
    print()

    try:
        recorded_episodes = 0
        recording = False
        episode_start_time = 0.0
        frame_count = 0
        prev_grip_pressed = False

        while not stop_event.is_set() and recorded_episodes < num_episodes:
            loop_start = time.perf_counter()

            # ---- 读取机器人状态（obs）----
            obs = arm.get_observation()

            # ---- 读取 VR 遥操作动作（action）----
            action = teleop.get_action(robot_obs=obs)

            # ---- 发送动作到机器人 ----
            if action is not None:
                arm.send_action(action)

            # ---- 读取相机帧（和 obs/action 同步！）----
            cam_frames = {}
            if cam_manager is not None:
                cam_frames = cam_manager.get_latest_frames()

            # ---- 录制控制（VR Grip 键）----
            transforms, buttons = teleop._oculus_reader.get_transformations_and_buttons()
            grip_value = 0.0
            if buttons:
                raw = buttons.get("rightGrip", 0.0)
                if isinstance(raw, (list, tuple)):
                    grip_value = float(raw[0]) if raw else 0.0
                elif isinstance(raw, bool):
                    grip_value = 1.0 if raw else 0.0
                else:
                    grip_value = float(raw)

            grip_pressed = grip_value >= 0.5
            if grip_pressed and not prev_grip_pressed:
                if recording:
                    # 停止录制
                    recording = False
                    ep_meta = writer.end_episode()
                    ep_idx = ep_meta.get("episode_index", recorded_episodes)
                    print(f"\n[REC] Episode {ep_idx} stopped, {frame_count} frames captured")
                    recorded_episodes += 1
                else:
                    # 开始录制
                    ep_idx = writer.start_episode(task_description)
                    recording = True
                    episode_start_time = time.monotonic()
                    frame_count = 0
                    print(f"\n[REC] Episode {ep_idx} recording...")
            prev_grip_pressed = grip_pressed

            # ---- 写入数据（同步！obs+action+camera 同一时刻）----
            if recording:
                # 只在有有效 action 时写入
                if action is not None:
                    timestamp = time.monotonic() - episode_start_time
                    images = cam_frames if cam_frames else None
                    writer.add_frame(
                        state=obs,
                        action=action,
                        timestamp=timestamp,
                        images=images,
                    )
                    frame_count += 1

                    # 检查是否超时
                    if time.monotonic() - episode_start_time >= episode_time_s:
                        recording = False
                        ep_meta = writer.end_episode()
                        ep_idx = ep_meta.get("episode_index", recorded_episodes)
                        print(f"\n[REC] Episode {ep_idx} timeout ({episode_time_s}s), {frame_count} frames")
                        recorded_episodes += 1

            # ---- 频率控制 ----
            dt = time.perf_counter() - loop_start
            sleep_time = max(1.0 / fps - dt, 0.0)
            time.sleep(sleep_time)

            # ---- 状态显示 ----
            loop_ms = (time.perf_counter() - loop_start) * 1000
            hz = 1000.0 / loop_ms if loop_ms > 0 else 0
            status = "REC" if recording else "IDLE"
            print(f"\r  [{status}] {hz:.0f}Hz | ep:{recorded_episodes}/{num_episodes} | frames:{frame_count}  ", end="", flush=True)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if recording:
            writer.end_episode()
        writer.finalize()
        print(f"\n[INFO] Dataset finalized: {output_dir}/{dataset_name}")
        print(f"[INFO] Total episodes: {recorded_episodes}")

        teleop.disconnect()
        arm.disconnect()
        if cam_manager is not None:
            cam_manager.stop()


def main():
    parser = argparse.ArgumentParser(description="NERO 一键遥操作 + 数据采集")
    parser.add_argument("--dataset_name", type=str, default="nero_teleop",
                        help="数据集名称 (default: nero_teleop)")
    parser.add_argument("--output_dir", type=str, default="/home/yuhang/datasets",
                        help="数据集输出目录 (default: /home/yuhang/datasets)")
    parser.add_argument("--task", type=str, default="pick and place",
                        help="任务描述 (default: 'pick and place')")
    parser.add_argument("--fps", type=int, default=30,
                        help="采集频率 Hz (default: 30)")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="录制集数 (default: 50)")
    parser.add_argument("--episode_time_s", type=float, default=60.0,
                        help="每集最大时长 秒 (default: 60)")
    parser.add_argument("--can_channel", type=str, default="can0",
                        help="CAN 通道 (default: can0)")
    parser.add_argument("--speed_percent", type=int, default=50,
                        help="机械臂速度百分比 (default: 50)")
    parser.add_argument("--quest_ip", type=str, default=None,
                        help="Quest VR IP (None=USB)")
    parser.add_argument("--no_camera", action="store_true",
                        help="不使用相机")
    parser.add_argument("--urdf_path", type=str,
                        default="/home/yuhang/projects/lerobot/nero/QuestArmTeleop/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_with_gripper_description.urdf",
                        help="NERO URDF 路径")
    parser.add_argument("--package_dirs", type=str,
                        default="/home/yuhang/projects/lerobot/nero/QuestArmTeleop/src/agx_arm_ros/src/agx_arm_description",
                        help="URDF package 目录")
    args = parser.parse_args()

    collect(
        urdf_path=args.urdf_path,
        package_dirs=args.package_dirs,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        task_description=args.task,
        fps=args.fps,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time_s,
        can_channel=args.can_channel,
        speed_percent=args.speed_percent,
        quest_ip=args.quest_ip,
        no_camera=args.no_camera,
    )


if __name__ == "__main__":
    main()
