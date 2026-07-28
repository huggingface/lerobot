from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from examples.picklift_v3.camera_profile import (
    camera_profile,
    canonicalize_front,
)

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class SyntheticBackend:
    def __init__(self, cfg: dict | None = None) -> None:
        self.index = 0
        self.camera_profile_id = cfg["camera_profile_id"] if cfg is not None else "synthetic_front_640x480_v1"

    def connect(self) -> None:
        pass

    def read_pre_action(self) -> tuple[np.ndarray, np.ndarray]:
        state = np.arange(6, dtype=np.float32) + self.index / 100
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[..., 0] = self.index % 255
        return state, image

    def requested_action(self) -> np.ndarray:
        return np.arange(6, dtype=np.float32) + 0.5

    def send_action(self, action: np.ndarray) -> np.ndarray:
        self.index += 1
        return np.clip(action, -100, 100).astype(np.float32)

    def preview_frame(self) -> np.ndarray:
        return self.read_pre_action()[1]

    def close(self) -> None:
        pass


@dataclass
class RelativeRebaser:
    offset: np.ndarray | None = None

    def initialize(self, leader: np.ndarray, follower: np.ndarray) -> np.ndarray:
        leader = np.asarray(leader, dtype=np.float32)
        follower = np.asarray(follower, dtype=np.float32)
        if leader.shape != (6,) or follower.shape != (6,):
            raise ValueError("relative rebase requires two six-joint vectors")
        self.offset = follower - leader
        return self.apply(leader)

    def apply(self, leader: np.ndarray) -> np.ndarray:
        if self.offset is None:
            raise RuntimeError("relative rebase used before initialization")
        leader = np.asarray(leader, dtype=np.float32)
        if leader.shape != (6,):
            raise ValueError("leader action must contain six joints")
        return (leader + self.offset).astype(np.float32)


class RealSO101Backend:
    def __init__(self, cfg: dict):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SO101Follower
        from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO101Leader

        camera = OpenCVCameraConfig(
            index_or_path=cfg["camera_device"],
            width=camera_profile(cfg["camera_profile_id"])["source"]["width"],
            height=camera_profile(cfg["camera_profile_id"])["source"]["height"],
            fps=camera_profile(cfg["camera_profile_id"])["source"]["fps"],
            fourcc=camera_profile(cfg["camera_profile_id"])["source"]["fourcc"],
        )
        self.robot = SO101Follower(
            SO101FollowerConfig(
                port=cfg["follower_port"],
                id=cfg["robot_id"],
                cameras={"front": camera},
                use_degrees=True,
                max_relative_target=cfg.get("max_relative_target", 5.0),
            )
        )
        self.leader = SO101Leader(
            SO101LeaderConfig(port=cfg["leader_port"], id=cfg["leader_id"], use_degrees=True)
        )
        self.rebaser = RelativeRebaser()
        self.alignment_mode = cfg["alignment_mode"]
        self.startup_hold_s = float(cfg["startup_hold_s"])
        self.camera_profile_id = cfg["camera_profile_id"]

    def _set_follower_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        self.robot.bus.sync_write("Torque_Enable", value, normalize=False, num_retry=2)
        actual = self.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
        if any(int(state) != value for state in actual.values()):
            raise RuntimeError(f"follower torque verification failed: expected {value}, got {actual}")

    def connect(self) -> None:
        # Robot.connect() configures hardware before this collector establishes a
        # no-jump goal. Open the buses explicitly, latch the present follower pose
        # as its goal, and only then enable torque.
        self.robot.bus.connect(handshake=True)
        self.leader.bus.connect(handshake=True)
        try:
            for camera in self.robot.cameras.values():
                camera.connect()
            follower_raw = self.robot.bus.sync_read("Present_Position", normalize=False)
            self.robot.bus.sync_write("Goal_Position", follower_raw, normalize=False)
            follower = self._read_follower_state()
            leader = self._read_leader_state()
            if self.alignment_mode == "relative_rebase":
                initial_command = self.rebaser.initialize(leader, follower)
                if not np.allclose(initial_command, follower, atol=1e-5):
                    raise RuntimeError("relative rebase failed zero-jump invariant")
            self._set_follower_torque(True)
            time.sleep(self.startup_hold_s)
        except BaseException:
            if self.robot.bus.is_connected:
                try:
                    self._set_follower_torque(False)
                finally:
                    self.robot.bus.disconnect(disable_torque=False)
            if self.leader.bus.is_connected:
                self.leader.bus.disconnect(disable_torque=False)
            raise

    def _read_follower_state(self) -> np.ndarray:
        state = self.robot.bus.sync_read("Present_Position")
        return np.asarray([state[j] for j in JOINTS], dtype=np.float32)

    def _read_leader_state(self) -> np.ndarray:
        action = self.leader.bus.sync_read("Present_Position")
        return np.asarray([action[j] for j in JOINTS], dtype=np.float32)

    def read_pre_action(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.robot.get_observation()
        state = np.asarray([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        image = canonicalize_front(np.asarray(obs["front"]), self.camera_profile_id)
        return state, image

    def requested_action(self) -> np.ndarray:
        leader = self._read_leader_state()
        return self.rebaser.apply(leader) if self.alignment_mode == "relative_rebase" else leader

    def send_action(self, action: np.ndarray) -> np.ndarray:
        requested = {f"{joint}.pos": float(action[i]) for i, joint in enumerate(JOINTS)}
        sent = self.robot.send_action(requested)
        return np.asarray([sent[f"{j}.pos"] for j in JOINTS], dtype=np.float32)

    def preview_frame(self) -> np.ndarray:
        return canonicalize_front(
            np.asarray(self.robot.cameras["front"].read_latest()),
            self.camera_profile_id,
        )

    def close(self) -> None:
        if self.robot.bus.is_connected:
            try:
                self._set_follower_torque(False)
            finally:
                self.robot.bus.disconnect(disable_torque=False)
        for camera in self.robot.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        if self.leader.bus.is_connected:
            self.leader.bus.disconnect(disable_torque=False)
