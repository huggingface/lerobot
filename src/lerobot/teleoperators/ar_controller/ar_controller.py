#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import numpy as np
import threading
import time
from datetime import datetime
from pathlib import Path

import zmq

import lerobot
from lerobot.model.kinematics import RobotKinematics

from scipy.spatial.transform import Rotation as R

from ..teleoperator import Teleoperator
from .config_ar_controller import ARControllerConfig

logger = logging.getLogger(__name__)

lerobot_path = Path(lerobot.__file__).parent
urdf_path = str(lerobot_path / "model" / "SO101" / "so101_new_calib.urdf")
ROBOT_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
ARM_LABELS = ("left", "right")


class ARController(Teleoperator):

    config_class = ARControllerConfig
    name = "ar_controller"

    def __init__(self, config: ARControllerConfig):
        super().__init__(config)
        self.config = config
        self._active_arms = ARM_LABELS if self.config.bi_controller else ("right",)

        self.context = zmq.Context()

        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.linger = 0
        self.rep_socket.bind("tcp://*:1111")

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.linger = 0
        self.sub_socket.bind("tcp://*:1112")
        self.sub_socket.subscribe("hand_data")

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.linger = 0
        self.pub_socket.bind("tcp://*:1113")

        self.running = False
        self.robot_data: dict = {}

        self._kinematics: dict[str, RobotKinematics] = {}
        self._state_lock = threading.Lock()
        default_joint_state = [0.0] * (len(ROBOT_JOINT_NAMES) - 1)
        self._arm_state = {
            "left": {
                "joints": default_joint_state.copy(),
                "jaw": 0.0,
            },
            "right": {
                "joints": default_joint_state.copy(),
                "jaw": 0.0,
            },
        }

        # Pre-initialize kinematics for active arms.
        for arm in self._active_arms:
            self._kinematics[arm] = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=ROBOT_JOINT_NAMES[:-1],
            )

        self.server_thread: threading.Thread | None = None

    @property
    def action_features(self) -> dict[str, type]:
        if self.config.bi_controller:
            return {
                **{f"left_{joint}.pos": float for joint in ROBOT_JOINT_NAMES[:-1]},
                "left_gripper.pos": float,
                **{f"right_{joint}.pos": float for joint in ROBOT_JOINT_NAMES[:-1]},
                "right_gripper.pos": float,
            }
        return {
            **{f"{joint}.pos": float for joint in ROBOT_JOINT_NAMES[:-1]},
            "gripper.pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return self.running
    
    def connect(self, calibrate: bool = True) -> None:
        """Start the server in a separate thread"""
        if self.running:
            logger.info("ARController already connected.")
            return

        self.running = True

        self.server_thread = threading.Thread(target=self._run_server, name="ARControllerServer", daemon=True)
        self.server_thread.start()

        logger.info("ARController connected and started sockets.")

    def disconnect(self) -> None:
        logger.info("ARController stopping...")
        self.running = False

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

        # Close all sockets gracefully
        for socket in (self.sub_socket, self.pub_socket, self.rep_socket):
            try:
                socket.close()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                logger.debug("Error closing socket: %s", exc, exc_info=True)

        self.context.term()
        logger.info("ARController stopped.")

    def _run_server(self):
        """Main server loop"""
        poller = zmq.Poller()
        poller.register(self.sub_socket, zmq.POLLIN)
        poller.register(self.rep_socket, zmq.POLLIN)

        while self.running:
            try:
                events = dict(poller.poll(1000))

                if self.rep_socket in events and events[self.rep_socket] & zmq.POLLIN:
                    self._handle_handshake()

                if self.sub_socket in events and events[self.sub_socket] & zmq.POLLIN:
                    topic = self.sub_socket.recv_string()
                    message = self.sub_socket.recv_string()
                    self._handle_message(topic, message)

            except zmq.ContextTerminated:
                break
            except Exception as e:
                logger.exception("ARController server error: %s", e)
                break

    def _handle_handshake(self):
        try:
            request = self.rep_socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return

        if request == "HELLO":
            self.rep_socket.send_string("READY")
            logger.info("[%s] Handshake acknowledged", datetime.now().strftime('%H:%M:%S'))
        else:
            self.rep_socket.send_string("UNKNOWN")

    def _handle_message(self, topic, message):
        """Handle incoming message from Unity"""
        try:
            logger.debug("[%s] Received from topic '%s'", datetime.now().strftime('%H:%M:%S'), topic)
            
            hand_data = json.loads(message)

            right_hand = hand_data.get("rightHand")
            if right_hand:
                self._process_hand(right_hand, "right")

            if self.config.bi_controller:
                left_hand = hand_data.get("leftHand")
                if left_hand:
                    self._process_hand(left_hand, "left")

            self._send_robot_data()

        except json.JSONDecodeError:
            logger.warning("Invalid JSON received: %s", message)
        except Exception as e:
            logger.exception("Error handling message: %s", e)

    def _pose_matrix(self, position, quaternion):
        """Create a 4x4 pose matrix from position and quaternion"""
        rot = R.from_quat(quaternion)
        pose = np.eye(4)
        pose[0:3, 0:3] = rot.as_matrix()
        pose[0:3, 3] = position
        return pose

    def _unity_to_robot(self, pos_unity, quat_unity):
        T = np.array([[ 0, 0, 1],
                      [-1, 0, 0],
                      [ 0, 1, 0]], dtype=float)
        pos_robot = T @ pos_unity

        pos_robot[0] = pos_robot[0] + 0.0981274
        pos_robot[1] = pos_robot[1] + 0.000218121
        pos_robot[2] = pos_robot[2] - 0.0079

        R_u = R.from_quat(quat_unity).as_matrix()
        R_TOOL = R.from_quat([0, 0.707107, 0, 0.707107]).as_matrix()
        R_r = T @ R_u @ T.T @ R_TOOL
        quat_robot = R.from_matrix(R_r).as_quat()

        return pos_robot, quat_robot

    def _process_hand(self, hand_payload, arm: str):
        if not hand_payload:
            return
        try:
            pos_unity = np.array([
                hand_payload.get("x"),
                hand_payload.get("y"), 
                hand_payload.get("z")
            ], dtype=float)

            q_unity = np.array([
                hand_payload.get("rx"),
                hand_payload.get("ry"), 
                hand_payload.get("rz"),
                hand_payload.get("rw")
            ], dtype=float)

            n = np.linalg.norm(q_unity)
            q_unity = q_unity / n if n > 0 else np.array([0,0,0,1], float)

            pos_robot, quat_robot = self._unity_to_robot(pos_unity, q_unity)

            ee_pose = self._pose_matrix(
                position=pos_robot.tolist(),
                quaternion=quat_robot.tolist()
            )

            solver = self._kinematics.get(arm)
            if solver is None:
                solver = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name="gripper_frame_link",
                    joint_names=ROBOT_JOINT_NAMES[:-1],
                )
                self._kinematics[arm] = solver

            with self._state_lock:
                current = np.array(self._arm_state[arm]["joints"], dtype=float)

            try:
                joint_targets = solver.inverse_kinematics(
                    current_joint_pos=current,
                    desired_ee_pose=ee_pose,
                    position_weight=1.0,
                    orientation_weight=0.1
                )
                jt = np.array(joint_targets, dtype=float)

            except Exception as exc:
                logger.warning("IK failed for %s arm: %s", arm, exc)
                return

            jaw_angle = float(hand_payload.get("jawAngle", 0.0))

            with self._state_lock:
                self._arm_state[arm]["joints"] = jt.tolist()
                self._arm_state[arm]["jaw"] = jaw_angle

        except Exception as exc:
            logger.exception("Failed to process %s hand data: %s", arm, exc)
            return
    
    def _send_robot_data(self):
        try:
            with self._state_lock:
                left_joints = list(self._arm_state["left"]["joints"])
                left_jaw = float(self._arm_state["left"]["jaw"])
                right_joints = list(self._arm_state["right"]["joints"])
                right_jaw = float(self._arm_state["right"]["jaw"])

            now = time.time()
            self.robot_data = {
                "robots": {
                    "left_robot": {
                        "id": "SO101_left",
                        "type": "SO-101",
                        "joints": {
                            "position": left_joints,
                        },
                        "jaw": {
                            "position": [left_jaw],
                        }
                    },
                    "right_robot": {
                        "id": "SO101_right",
                        "type": "SO-101",
                        "joints": {
                            "position": right_joints
                        },
                        "jaw": {
                            "position": [right_jaw]
                        }
                    }
                },
                "timestamp": {
                    "secs": int(now),
                    "nsecs": int((now % 1) * 1e9)
                },
                "session_id": "test_session_001"
            }

            robot_json = json.dumps(self.robot_data)
            self.pub_socket.send_string("robot_data", zmq.SNDMORE)
            self.pub_socket.send_string(robot_json)

        except Exception as e:
            logger.exception("Error preparing robot data: %s", e)
            return
        
    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def setup_motors(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        with self._state_lock:
            if self.config.bi_controller:
                left = self._arm_state["left"]
                right = self._arm_state["right"]
                action = {
                    **{
                        f"left_{joint}.pos": value
                        for joint, value in zip(ROBOT_JOINT_NAMES[:-1], left["joints"])
                    },
                    "left_gripper.pos": left["jaw"],
                    **{
                        f"right_{joint}.pos": value
                        for joint, value in zip(ROBOT_JOINT_NAMES[:-1], right["joints"])
                    },
                    "right_gripper.pos": right["jaw"],
                }
            else:
                right = self._arm_state["right"]
                action = {
                    **{f"{joint}.pos": value for joint, value in zip(ROBOT_JOINT_NAMES[:-1], right["joints"])},
                    "gripper.pos": right["jaw"],
                }
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    
