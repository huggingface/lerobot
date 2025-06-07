import logging
from typing import TYPE_CHECKING

from lerobot.common.errors import DeviceNotConnectedError

from .config_moveit2 import MoveIt2InterfaceConfig
from .moveit2_servo import MoveIt2Servo

if TYPE_CHECKING:
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
    from rclpy.executors import Executor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState


class MoveIt2Interface:
    """Class to interface with a MoveIt2 manipulator."""

    def __init__(self, config: MoveIt2InterfaceConfig):
        self.config = config
        self.robot_node: Node | None = None
        self.moveit2_servo: MoveIt2Servo | None = None
        self.gripper_action_client: ActionClient | None = None
        self._last_joint_state_msg = None
        self.is_connected = False

    def connect(self, executor: "Executor") -> None:
        from control_msgs.action import GripperCommand
        from rclpy import qos
        from rclpy.action import ActionClient
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        self.robot_node = Node(self.config.planning_group + "_control_node", namespace=self.config.namespace)
        self.moveit2_servo = MoveIt2Servo(
            node=self.robot_node,
            frame_id=self.config.base_link,
            callback_group=ReentrantCallbackGroup(),
        )
        self.gripper_action_client = ActionClient(
            self.robot_node,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
            callback_group=ReentrantCallbackGroup(),
        )
        self.joint_state_sub = self.robot_node.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            qos_profile=qos.QoSProfile(
                durability=qos.QoSDurabilityPolicy.VOLATILE,
                reliability=qos.QoSReliabilityPolicy.BEST_EFFORT,
                history=qos.QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            ),
            callback_group=ReentrantCallbackGroup(),
        )
        executor.add_node(self.robot_node)
        self.is_connected = True

        self._goal_msg = GripperCommand.Goal()

    def servo(self, linear, angular, normalize: bool = True) -> None:
        if not self.moveit2_servo:
            raise DeviceNotConnectedError(
                "MoveIt2Interface is not connected. You need to call `connect()`."
            )

        if normalize:
            linear = [
                v * self.config.max_linear_velocity for v in linear
            ]
            angular = [
                v * self.config.max_angular_velocity for v in angular
            ]
        self.moveit2_servo.servo(linear=linear, angular=angular)

    def send_gripper_command(self, position: float, normalize: bool = True) -> bool:
        """
        Send a command to the gripper to move to a specific position.
        Args:
            position (float): The target position for the gripper (0=open, 1=closed).
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.gripper_action_client:
            raise RuntimeError("MoveIt2Interface is not connected. You need to call `connect()`.")

        if not self.gripper_action_client.wait_for_server(timeout_sec=1.0):
            logging.error("Gripper action server not available")
            return False

        if normalize:
            # Map normalized position (0=open, 1=closed) to actual gripper joint position
            open_pos = self.config.gripper_open_position
            closed_pos = self.config.gripper_close_position
            gripper_goal = open_pos + position * (closed_pos - open_pos)
        else:
            gripper_goal = position

        self._goal_msg.command.position = gripper_goal
        if not (resp := self.gripper_action_client.send_goal(self._goal_msg)):
            logging.error("Failed to send gripper command")
            return False
        result: GripperCommand.Result = resp.result
        if result.reached_goal:
            return True
        logging.error(
            f"Gripper did not reach goal. stalled: {result.stalled}, "
            f"effort: {result.effort}, position: {result.position}"
        )
        return False

    @property
    def joint_state(self) -> "JointState | None":
        """
        Get the last received joint state message.
        Returns:
            JointState: The last joint state message received, or None if no message has been received.
        """
        return self._last_joint_state_msg

    def _joint_state_callback(self, msg):
        self._last_joint_state_msg = msg

    def disconnect(self):
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        if self.moveit2_servo:
            self.moveit2_servo = None
        if self.gripper_action_client:
            self.gripper_action_client.destroy()
            self.gripper_action_client = None
        self.is_connected = False
