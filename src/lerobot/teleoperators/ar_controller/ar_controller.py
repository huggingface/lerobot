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

import logging
import time
from datetime import datetime
import json
import numpy as np
import time
import threading
import zmq

from pathlib import Path
import lerobot
from lerobot.model.kinematics import RobotKinematics

from scipy.spatial.transform import Rotation as R

from ..teleoperator import Teleoperator
from .config_ar_controller import ARControllerConfig

logger = logging.getLogger(__name__)

lerobot_path = Path(lerobot.__file__).parent
urdf_path = str(lerobot_path / "model" / "SO101" / "so101_new_calib.urdf")
ROBOT_JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']

class ARController(Teleoperator):

    config_class = ARControllerConfig
    name = "ar_controller"

    def __init__(self, config: ARControllerConfig):
        super().__init__(config)
        self.config = config

        print("bi_controller:", self.config.bi_controller)

        if not self.config.bi_controller:
            self.context = zmq.Context()

            self.rep_socket = self.context.socket(zmq.REP)
            self.rep_socket.linger = 0
            self.rep_socket.bind("tcp://*:1111")

            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.linger = 0
            self.sub_socket.bind(f"tcp://*:1112")
            self.sub_socket.subscribe("hand_data")

            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.linger = 0
            self.pub_socket.bind("tcp://*:1113")

            self.running = False
            self.robot_data = {}

            self.kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name="gripper_frame_link",
                joint_names=ROBOT_JOINT_NAMES[:-1]
            )

        self.joint_positions = [0.0] * (len(ROBOT_JOINT_NAMES) - 1)
        self.jaw_position = 0.0

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}
    
    @property
    def is_connected(self) -> bool:
        return True
    
    def connect(self, calibrate: bool = True) -> None:
        """Start the server in a separate thread"""
        self.running = True

        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        print(f"[ARController] Connected and started sockets")

    def disconnect(self) -> None:
        print(f"[ARController] Stopping...")
        self.running = False
        # Close all sockets gracefully
        try:
            self.sub_socket.close()
            self.pub_socket.close()
            self.rep_socket.close()
        except Exception as e:
            print(f"[ARController] Error closing sockets: {e}")
        
        # Wait for server thread to finish
        if hasattr(self, 'server_thread') and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)
        
        self.context.term()
        print("[ARController] stopped.")

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
                print(f"[ARController] Server error: {e}")
                break

    def _handle_handshake(self):
        try:
            request = self.rep_socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            return

        if request == "HELLO":
            self.rep_socket.send_string("READY")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Handshake acknowledged")
        else:
            self.rep_socket.send_string("UNKNOWN")

    def _handle_message(self, topic, message):
        """Handle incoming message from Unity"""
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received from topic '{topic}':")
            
            hand_data = json.loads(message)

            right_hand = hand_data.get("rightHand")
            self._process_hand(right_hand)
            self._send_robot_data()

        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")
        except Exception as e:
            print(f"Error handling message: {e}")

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

    def _process_hand(self, hand_payload):
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

            try:
                seed = np.array(self.joint_positions, dtype=float)
                joint_targets = self.kinematics.inverse_kinematics(
                    current_joint_pos=seed,
                    desired_ee_pose=ee_pose,
                    position_weight=1.0,
                    orientation_weight=0.1
                )
                jt = np.array(joint_targets, dtype=float)

            except Exception as exc:
                print(f"IK failed: {exc}")
                return

            jaw_angle = float(hand_payload.get("jawAngle", 0.0))

            self.joint_positions = jt.tolist()
            self.jaw_position = jaw_angle

        except Exception as exc:
            print(f"Failed to process hand data: {exc}")
            return
    
    def _send_robot_data(self):
        try:
            self.robot_data = {
                "robots": {
                    "left_robot": {
                        "id": "SO101_left",
                        "type": "SO-101",
                        "joints": {
                            "position": [0.0] * (len(ROBOT_JOINT_NAMES) - 1)
                        },
                        "jaw": {
                            "position": [0.0]
                        }
                    },
                    "right_robot": {
                        "id": "SO101_right",
                        "type": "SO-101",
                        "joints": {
                            "position": self.joint_positions
                        },
                        "jaw": {
                            "position": [self.jaw_position]
                        }
                    }
                },
                "timestamp": {
                    "secs": int(time.time()),
                    "nsecs": int((time.time() % 1) * 1e9)
                },
                "session_id": "test_session_001"
            }

            robot_json = json.dumps(self.robot_data)
            self.pub_socket.send_string("robot_data", zmq.SNDMORE)
            self.pub_socket.send_string(robot_json)

        except Exception as e:
            print(f"Error preparing robot data: {e}")
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
        action = {f"{motor}.pos": val for motor, val in zip(ROBOT_JOINT_NAMES[:-1], self.joint_positions)}
        action["gripper.pos"] = self.jaw_position
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    