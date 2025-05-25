import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional
from unittest.mock import Mock

# import rclpy
import torch

# from control_msgs.action import GripperCommand
# from geometry_msgs.msg import TwistStamped
# from moveit_msgs.srv import ServoCommandType
# from rclpy import qos
# from rclpy.action import ActionClient
# from rclpy.callback_groups import CallbackGroup, ReentrantCallbackGroup
# from rclpy.executors import Executor, MultiThreadedExecutor
# from rclpy.node import Node
# from rclpy.subscription import Subscription
# from sensor_msgs.msg import JointState, Joy
# from std_srvs.srv import SetBool
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import (
    JoystickListenerConfig,
    MoveIt2ManipulatorConfig,
    MoveIt2ManipulatorRobotConfig,
)
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

if TYPE_CHECKING:
    from control_msgs.action import GripperCommand
    from rclpy.action import ActionClient
    from rclpy.callback_groups import CallbackGroup
    from rclpy.executors import Executor
    from rclpy.node import Node
    from rclpy.subscription import Subscription
    from sensor_msgs.msg import JointState, Joy


class MoveIt2Servo:
    """
    Python interface for MoveIt2 Servo.
    """

    def __init__(
        self,
        node: "Node",
        frame_id: str,
        callback_group: Optional["CallbackGroup"] = None,
    ):
        self._node = node
        self._frame_id = frame_id
        self._enabled = False

        from geometry_msgs.msg import TwistStamped
        from moveit_msgs.srv import ServoCommandType
        from rclpy import qos
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
            logging.warning("Pause service not available.")
            return False
        if not self._cmd_type_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logging.warning("Command type service not available.")
            return False
        result = self._pause_srv.call(self._enable_req)
        if not result or not result.success:
            logging.error(f"Enable failed: {getattr(result, 'message', '')}")
            self._enabled = False
            return False
        cmd_result = self._cmd_type_srv.call(self._twist_type_req)
        if not cmd_result or not cmd_result.success:
            logging.error("Switch to TWIST command type failed.")
            self._enabled = False
            return False
        logging.info("MoveIt Servo enabled.")
        self._enabled = True
        return True

    def disable(self, wait_for_server_timeout_sec=1.0) -> bool:
        if not self._pause_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logging.warning("Pause service not available.")
            return False
        result = self._pause_srv.call(self._disable_req)
        self._enabled = not (result and result.success)
        return bool(result and result.success)

    def servo(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), enable_if_disabled=True):
        if not self._enabled and enable_if_disabled and not self.enable():
            logging.warning("Dropping servo command because MoveIt2 Servo is not enabled.")
            return

        self._twist_msg.header.frame_id = self._frame_id
        self._twist_msg.header.stamp = self._node.get_clock().now().to_msg()
        self._twist_msg.twist.linear.x = linear[0]
        self._twist_msg.twist.linear.y = linear[1]
        self._twist_msg.twist.linear.z = linear[2]
        self._twist_msg.twist.angular.x = angular[0]
        self._twist_msg.twist.angular.y = angular[1]
        self._twist_msg.twist.angular.z = angular[2]
        self._twist_pub.publish(self._twist_msg)


class Moveit2Manipulator:
    """Class for a MoveIt2 manipulator."""

    def __init__(self, config: MoveIt2ManipulatorConfig):
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

    def servo(self, linear, angular):
        if not self.moveit2_servo:
            raise RobotDeviceNotConnectedError(
                "MoveIt2Manipulator is not connected. You need to call `connect()`."
            )
        self.moveit2_servo.servo(linear=linear, angular=angular)

    def send_gripper_command(self, position: float) -> bool:
        """
        Send a command to the gripper to move to a specific position.
        Args:
            position (float): The target position for the gripper (0=open, 1=closed).
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.gripper_action_client:
            raise RuntimeError("MoveIt2Manipulator is not connected. You need to call `connect()`.")

        if not self.gripper_action_client.wait_for_server(timeout_sec=1.0):
            logging.error("Gripper action server not available")
            return False

        # Map normalized position (0=open, 1=closed) to actual gripper joint position
        open_pos = self.config.gripper_open_position
        closed_pos = self.config.gripper_close_position
        gripper_goal = open_pos + position * (closed_pos - open_pos)
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


class JoystickListener:
    """
    Class to listen to joystick messages.
    This class is used to listen to joystick messages and store the last message received.
    """

    def __init__(self, config: JoystickListenerConfig):
        self.config = config
        self._node: Node | None = None
        self._joy_sub: Subscription | None = None
        self._last_joy_msg: Joy | None = None
        self.is_connected = False

    def connect(self, executor: "Executor") -> None:
        """
        Connect to the joystick listener by creating a ROS2 node and subscription.
        """
        if self.config.mock:
            self._last_joy_msg = Mock(axes=[0.0] * 6, buttons=[0] * 12)
            return

        from rclpy.node import Node
        from sensor_msgs.msg import Joy

        self._node = Node("joystick_listener_node", namespace=self.config.namespace)
        self._joy_sub = self._node.create_subscription(Joy, "/joy", self._joy_callback, 10)
        executor.add_node(self._node)
        self.is_connected = True

    def _joy_callback(self, msg: "Joy"):
        self._last_joy_msg = msg

    def get_last_joy_msg(self) -> "Joy | None":
        """
        Get the last joystick message received.
        Returns:
            Joy: The last joystick message received, or None if no message has been received.
        """
        return self._last_joy_msg

    def disconnect(self):
        """
        Disconnect the joystick listener by destroying the node and subscription.
        """
        self.is_connected = False
        if self.config.mock:
            return

        if self._joy_sub:
            self._joy_sub.destroy()
            self._joy_sub = None
        if self._node:
            self._node.destroy_node()
            self._node = None


class MockMoveit2Manipulator:
    """Mock version of Moveit2Manipulator for testing and mock mode."""

    def __init__(self, config: MoveIt2ManipulatorConfig):
        self.config = config

        joint_names = self.config.arm_joint_names + [self.config.gripper_joint_name]
        self._last_joint_state_msg = Mock(
            position=[0.0] * len(joint_names),
            velocity=[0.0] * len(joint_names),
        )
        self._last_joint_state_msg.configure_mock(name=joint_names)
        self.is_connected = False

    def connect(self, executor):
        self.is_connected = True

    def servo(self, linear, angular):
        pass

    def send_gripper_command(self, position: float) -> bool:
        return True

    @property
    def joint_state(self):
        return self._last_joint_state_msg

    def disconnect(self):
        self.is_connected = False

    @property
    def motors(self) -> List[str]:
        """
        Get the list of motors for the mock manipulator.
        Returns:
            List[str]: The list of motors, which is empty for the mock manipulator.
        """
        return [self.config.gripper_joint_name] + self.config.arm_joint_names


class MoveIt2ManipulatorRobot:
    """
    Class for a ROS2 manipulator robot.
    """

    def __init__(self, config: MoveIt2ManipulatorRobotConfig):
        """
        Initialize the ROS2 manipulator.
        """
        self.config = config
        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.leader_arms = {name: JoystickListener(cfg) for name, cfg in self.config.leader_arms.items()}
        if self.config.mock:
            self.follower_arms = {
                name: MockMoveit2Manipulator(cfg) for name, cfg in self.config.follower_arms.items()
            }
        else:
            self.follower_arms = {
                name: Moveit2Manipulator(cfg) for name, cfg in self.config.follower_arms.items()
            }

        self.executor: Executor | None = None
        self.executor_thread: threading.Thread | None = None

        self.logs = {}
        self.is_connected = False

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = []
        for arm_name in self.config.leader_arms:
            action_names.extend(
                [
                    f"{arm_name}_linear_vel_x",
                    f"{arm_name}_linear_vel_y",
                    f"{arm_name}_linear_vel_z",
                    f"{arm_name}_angular_vel_x",
                    f"{arm_name}_angular_vel_y",
                    f"{arm_name}_angular_vel_z",
                    f"{arm_name}_gripper_pos",
                ]
            )
        state_names = []
        for arm_name, arm in self.config.follower_arms.items():
            for joint_name in arm.arm_joint_names:
                state_names.append(f"{arm_name}_{joint_name}_pos")
                # state_names.append(f"{arm_name}_{joint_name}_vel")
            state_names.append(f"{arm_name}_{arm.gripper_joint_name}_pos")
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available = []
        for name, arm in self.leader_arms.items():
            # don't include joysticks as available arms
            if isinstance(arm, JoystickListener):
                continue
            available.append(get_arm_id(name, "leader"))
        for name in self.follower_arms:
            available.append(get_arm_id(name, "follower"))
        return available

    def connect(self):
        """
        Connect to the robot by establishing communication with it.
        """
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if self.config.mock:
            logging.info("Using mock configuration, no connection to robot.")
            self.is_connected = True
            return

        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        rclpy.init()
        # Create and start the executor in a separate thread
        self.executor = MultiThreadedExecutor()
        for arm in self.follower_arms.values():
            arm.connect(self.executor)
        for arm in self.leader_arms.values():
            arm.connect(self.executor)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        time.sleep(3)  # Give some time to connect to services and receive messages

        self.is_connected = True
        logging.info("MoveIt2ManipulatorRobot is connected.")

    def _get_leader_state(self, name: str) -> list[float]:
        """ Get the joystick input for the leader arm.
        On an Xbox controller, the mapping is as follows:
        - Left joystick: linear velocity in X and Y axes
        - LT: -Z linear velocity
        - RT: +Z linear velocity
        - Right joystick angular velocity in X and Y axes
        - LB: -Z angular velocity
        - RB: +Z angular velocity
        - A: close gripper

        Args:
            name (str): The name of the leader arm to get the joystick input for.
        """
        last_joy_msg = self.leader_arms[name].get_last_joy_msg()
        if last_joy_msg is None:
            logging.warning("No joystick message received yet.")
            return [0.0] * 7

        s = self.config.max_linear_velocity
        lin_x = -s * last_joy_msg.axes[0]
        lin_y = s * last_joy_msg.axes[1]

        lin_z = 0.0
        # map the input range from [-1, 1] to [0, +/-1]
        if last_joy_msg.axes[2] < 1:
            lin_z = 0.5 * s * (last_joy_msg.axes[2] - 1)
        elif last_joy_msg.axes[5] < 1:
            lin_z = -0.5 * s * (last_joy_msg.axes[5] - 1)

        s = self.config.max_angular_velocity
        ang_x = -s * last_joy_msg.axes[4]
        ang_y = -s * last_joy_msg.axes[3]
        ang_z = 0.0
        if last_joy_msg.buttons[4] > 0:
            ang_z = s
        elif last_joy_msg.buttons[5] > 0:
            ang_z = -s

        # By default, the gripper is open. Press A to close the gripper
        gripper_pos = last_joy_msg.buttons[0]

        return [
            lin_x,
            lin_y,
            lin_z,
            ang_x,
            ang_y,
            ang_z,
            gripper_pos,
        ]

    def _get_arm_observation(self) -> torch.Tensor:
        positions = []
        for name, arm in self.follower_arms.items():
            if not arm.joint_state:
                raise RobotDeviceNotConnectedError(
                    f"MoveIt2Manipulator for arm '{name}' is not connected. You need to run `robot.connect()`."
                )

            name_to_index = {name: i for i, name in enumerate(arm.joint_state.name)}
            for joint_name in arm.config.arm_joint_names:
                idx = name_to_index.get(joint_name)
                if idx is None:
                    raise ValueError(f"Joint '{joint_name}' not found in joint state.")
                positions.append(arm.joint_state.position[idx])

            idx = name_to_index.get(arm.config.gripper_joint_name)
            if idx is None:
                raise ValueError(f"Gripper joint '{arm.config.gripper_joint_name}' not found in joint state.")
            positions.append(arm.joint_state.position[idx])
        return torch.tensor(positions)

    def _get_cam_observation(self) -> dict[str, torch.Tensor]:
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            image = self.cameras[name].async_read()
            images[name] = torch.from_numpy(image)
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        return images

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Perform a teleop step by reading joystick inputs and sending actions to the robot.
        Args:
            record_data (bool): Whether to record data during the teleop step.
                If True, returns observations and actions.
                If False, don't return anything.
        Returns:
            None or tuple: If `record_data` is True, returns a tuple containing the observations
                and actions.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "MoveIt2ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        action = []
        for name in self.leader_arms:
            action.append(torch.Tensor(self._get_leader_state(name)))
        action = torch.cat(action)
        action = self.send_action(action)

        # Early exit when recording data is not requested
        if not record_data:
            return

        obs_dict = self.capture_observation()
        action_dict = {"action": torch.tensor(action)}
        return obs_dict, action_dict

    def capture_observation(self) -> dict[str, torch.Tensor]:
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "MoveIt2ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        obs_dict = {}
        obs_dict["observation.state"] = self._get_arm_observation()
        images = self._get_cam_observation()
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.

        Returns:
            torch.Tensor: the action sent to the robot.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "MoveIt2ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        action_sent = []
        for arm in self.follower_arms.values():
            linear = torch.clamp(
                action[from_idx : from_idx + 3],
                -self.config.max_linear_velocity,
                self.config.max_linear_velocity,
            )
            angular = torch.clamp(
                action[from_idx + 3 : from_idx + 6],
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )
            gripper_pos = action[from_idx + 6]

            action_sent.append(linear)
            action_sent.append(angular)
            action_sent.append(gripper_pos.unsqueeze(0))

            arm.servo(linear=tuple(linear.tolist()), angular=tuple(angular.tolist()))
            arm.send_gripper_command(float(gripper_pos.item()))
            from_idx += 7  # Move the index to the next arm
        return torch.cat(action_sent)

    def print_logs(self):
        logging.info(self.logs)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "MoveIt2ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for arm in self.follower_arms.values():
            arm.disconnect()
        for arm in self.leader_arms.values():
            arm.disconnect()
        if self.executor:
            self.executor.shutdown()
            self.executor = None
        if self.executor_thread:
            self.executor_thread.join()
            self.executor_thread = None
        if not self.config.mock:
            import rclpy

            rclpy.shutdown()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
