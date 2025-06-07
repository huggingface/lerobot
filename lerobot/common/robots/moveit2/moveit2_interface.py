import logging
import threading
import time

from lerobot.common.errors import DeviceNotConnectedError

from .config_moveit2 import MoveIt2InterfaceConfig
from .moveit2_servo import MoveIt2Servo

logger = logging.getLogger(__name__)

try:
    import rclpy
    from control_msgs.action import GripperCommand
    from rclpy import qos
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    ROS2_AVAILABLE = True
except ImportError as e:
    logger.info(f"ROS2 dependencies not available: {e}")
    ROS2_AVAILABLE = False


class MoveIt2Interface:
    """Class to interface with a MoveIt2 manipulator."""

    def __init__(self, config: MoveIt2InterfaceConfig):
        self.config = config
        self.robot_node: Node | None = None
        self.gripper_action_client: ActionClient | None = None
        self.executor: MultiThreadedExecutor | None = None
        self._last_joint_state: dict[str, dict[str, float]] | None = None
        self.moveit2_servo: MoveIt2Servo | None = None
        self.executor_thread: threading.Thread | None = None
        self.is_connected = False

    def connect(self) -> None:
        if not ROS2_AVAILABLE:
            raise ImportError("ROS2 is not available")

        if not rclpy.ok():
            rclpy.init()

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
        self._goal_msg = GripperCommand.Goal()
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

        # Create and start the executor in a separate thread
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.robot_node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        time.sleep(3)  # Give some time to connect to services and receive messages

        self.is_connected = True

    def servo(self, linear, angular, normalize: bool = True) -> None:
        if not self.moveit2_servo:
            raise DeviceNotConnectedError("MoveIt2Interface is not connected. You need to call `connect()`.")

        if normalize:
            linear = [v * self.config.max_linear_velocity for v in linear]
            angular = [v * self.config.max_angular_velocity for v in angular]
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
            logger.error("Gripper action server not available")
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
            logger.error("Failed to send gripper command")
            return False
        result = resp.result  # type: ignore  # ROS2 types available at runtime
        if result.reached_goal:
            return True
        logger.error(
            f"Gripper did not reach goal. stalled: {result.stalled}, "
            f"effort: {result.effort}, position: {result.position}"
        )
        return False

    @property
    def joint_state(self) -> dict[str, dict[str, float]] | None:
        """Get the last received joint state."""
        return self._last_joint_state

    def _joint_state_callback(self, msg: "JointState") -> None:
        self._last_joint_state = self._last_joint_state or {}
        positions = {}
        velocities = {}
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for joint_name in self.config.arm_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in joint state.")
            positions[joint_name] = msg.position[idx]
            velocities[joint_name] = msg.velocity[idx]

        idx = name_to_index.get(self.config.gripper_joint_name)
        if idx is None:
            raise ValueError(f"Gripper joint '{self.config.gripper_joint_name}' not found in joint state.")
        positions[self.config.gripper_joint_name] = msg.position[idx]
        velocities[self.config.gripper_joint_name] = msg.velocity[idx]

        self._last_joint_state["position"] = positions
        self._last_joint_state["velocity"] = velocities

    def disconnect(self):
        if self.joint_state_sub:
            self.joint_state_sub.destroy()
            self.joint_state_sub = None
        if self.gripper_action_client:
            self.gripper_action_client.destroy()
            self.gripper_action_client = None
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        if self.moveit2_servo:
            self.moveit2_servo = None

        if self.executor:
            self.executor.shutdown()
            self.executor = None
        if self.executor_thread:
            self.executor_thread.join()
            self.executor_thread = None

        self.is_connected = False
