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

import logging

logger = logging.getLogger(__name__)

try:
    from rclpy import qos
    from rclpy.callback_groups import CallbackGroup
    from rclpy.node import Node

    ROS2_AVAILABLE = True
except ImportError as e:
    logger.info(f"ROS2 dependencies not available: {e}")
    ROS2_AVAILABLE = False


class MoveIt2Servo:
    """
    Python interface for MoveIt2 Servo.
    """

    def __init__(
        self,
        node: "Node",
        frame_id: str,
        callback_group: "CallbackGroup",
    ):
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 is not available")

        self._node = node
        self._frame_id = frame_id
        self._enabled = False

        from geometry_msgs.msg import TwistStamped
        from moveit_msgs.srv import ServoCommandType
        from std_srvs.srv import SetBool

        self._twist_pub = node.create_publisher(
            TwistStamped,
            "/servo_node/delta_twist_cmds",
            qos.QoSProfile(
                durability=qos.QoSDurabilityPolicy.VOLATILE,
                reliability=qos.QoSReliabilityPolicy.RELIABLE,
                history=qos.QoSHistoryPolicy.KEEP_ALL,
            ),
            callback_group=callback_group,
        )
        self._pause_srv = node.create_client(
            SetBool, "/servo_node/pause_servo", callback_group=callback_group
        )
        self._cmd_type_srv = node.create_client(
            ServoCommandType, "/servo_node/switch_command_type", callback_group=callback_group
        )
        self._twist_msg = TwistStamped()
        self._enable_req = SetBool.Request(data=False)
        self._disable_req = SetBool.Request(data=True)
        self._twist_type_req = ServoCommandType.Request(command_type=ServoCommandType.Request.TWIST)

    def enable(self, wait_for_server_timeout_sec=1.0) -> bool:
        if not self._pause_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logger.warning("Pause service not available.")
            return False
        if not self._cmd_type_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logger.warning("Command type service not available.")
            return False
        result = self._pause_srv.call(self._enable_req)
        if not result or not result.success:
            logger.error(f"Enable failed: {getattr(result, 'message', '')}")
            self._enabled = False
            return False
        cmd_result = self._cmd_type_srv.call(self._twist_type_req)
        if not cmd_result or not cmd_result.success:
            logger.error("Switch to TWIST command type failed.")
            self._enabled = False
            return False
        logger.info("MoveIt Servo enabled.")
        self._enabled = True
        return True

    def disable(self, wait_for_server_timeout_sec=1.0) -> bool:
        if not self._pause_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logger.warning("Pause service not available.")
            return False
        result = self._pause_srv.call(self._disable_req)
        self._enabled = not (result and result.success)
        return bool(result and result.success)

    def servo(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), enable_if_disabled=True):
        if not self._enabled and enable_if_disabled and not self.enable():
            logger.warning("Dropping servo command because MoveIt2 Servo is not enabled.")
            return

        self._twist_msg.header.frame_id = self._frame_id
        self._twist_msg.header.stamp = self._node.get_clock().now().to_msg()
        self._twist_msg.twist.linear.x = float(linear[0])
        self._twist_msg.twist.linear.y = float(linear[1])
        self._twist_msg.twist.linear.z = float(linear[2])
        self._twist_msg.twist.angular.x = float(angular[0])
        self._twist_msg.twist.angular.y = float(angular[1])
        self._twist_msg.twist.angular.z = float(angular[2])
        self._twist_pub.publish(self._twist_msg)
