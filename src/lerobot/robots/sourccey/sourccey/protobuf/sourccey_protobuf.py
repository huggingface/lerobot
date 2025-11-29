import logging
from typing import Any

import cv2
import numpy as np

from .generated import sourccey_pb2

logger = logging.getLogger(__name__)


class SourcceyProtobuf:
    """Handles protobuf conversion for Sourccey robot actions and observations."""

    def __init__(self):
        pass

    def action_to_protobuf(self, action: dict[str, Any]) -> sourccey_pb2.SourcceyRobotAction:
        """Convert action dictionary to protobuf SourcceyRobotAction message."""
        try:
            robot_action = sourccey_pb2.SourcceyRobotAction()

            # Process left arm action
            left_target_positions = sourccey_pb2.MotorJoint()
            left_target_positions.shoulder_pan = float(action.get("left_shoulder_pan.pos", 0.0))
            left_target_positions.shoulder_lift = float(action.get("left_shoulder_lift.pos", 0.0))
            left_target_positions.elbow_flex = float(action.get("left_elbow_flex.pos", 0.0))
            left_target_positions.wrist_flex = float(action.get("left_wrist_flex.pos", 0.0))
            left_target_positions.wrist_roll = float(action.get("left_wrist_roll.pos", 0.0))
            left_target_positions.gripper = float(action.get("left_gripper.pos", 0.0))
            robot_action.left_arm_target_joints.CopyFrom(left_target_positions)

            # Process right arm action
            right_target_positions = sourccey_pb2.MotorJoint()
            right_target_positions.shoulder_pan = float(action.get("right_shoulder_pan.pos", 0.0))
            right_target_positions.shoulder_lift = float(action.get("right_shoulder_lift.pos", 0.0))
            right_target_positions.elbow_flex = float(action.get("right_elbow_flex.pos", 0.0))
            right_target_positions.wrist_flex = float(action.get("right_wrist_flex.pos", 0.0))
            right_target_positions.wrist_roll = float(action.get("right_wrist_roll.pos", 0.0))
            right_target_positions.gripper = float(action.get("right_gripper.pos", 0.0))
            robot_action.right_arm_target_joints.CopyFrom(right_target_positions)

            # Process base action
            base_action = sourccey_pb2.BaseVelocity()
            base_action.x_vel = float(action.get("x.vel", 0.0))
            base_action.y_vel = float(action.get("y.vel", 0.0))
            base_action.theta_vel = float(action.get("theta.vel", 0.0))
            base_action.z_vel = float(action.get("z.vel", 0.0))
            robot_action.base_target_velocity.CopyFrom(base_action)
            
            # Per-arm flags
            if "untorque_left" in action:
                try:
                    robot_action.untorque_left = bool(action.get("untorque_left", False))
                except AttributeError:
                    pass
            if "untorque_right" in action:
                try:
                    robot_action.untorque_right = bool(action.get("untorque_right", False))
                except AttributeError:
                    pass

            return robot_action

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: python src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert action to protobuf: {e}")
            raise

    def observation_to_protobuf(self, observation: dict[str, Any]) -> sourccey_pb2.SourcceyRobotState:
        """Convert observation dictionary to protobuf SourcceyRobotState message."""
        try:
            msg = sourccey_pb2.SourcceyRobotState()

            # Set left arm motor positions
            left_motor_pos = msg.left_arm_joints
            left_motor_pos.shoulder_pan = observation.get("left_shoulder_pan.pos", 0.0)
            left_motor_pos.shoulder_lift = observation.get("left_shoulder_lift.pos", 0.0)
            left_motor_pos.elbow_flex = observation.get("left_elbow_flex.pos", 0.0)
            left_motor_pos.wrist_flex = observation.get("left_wrist_flex.pos", 0.0)
            left_motor_pos.wrist_roll = observation.get("left_wrist_roll.pos", 0.0)
            left_motor_pos.gripper = observation.get("left_gripper.pos", 0.0)

            # Set right arm motor positions
            right_motor_pos = msg.right_arm_joints
            right_motor_pos.shoulder_pan = observation.get("right_shoulder_pan.pos", 0.0)
            right_motor_pos.shoulder_lift = observation.get("right_shoulder_lift.pos", 0.0)
            right_motor_pos.elbow_flex = observation.get("right_elbow_flex.pos", 0.0)
            right_motor_pos.wrist_flex = observation.get("right_wrist_flex.pos", 0.0)
            right_motor_pos.wrist_roll = observation.get("right_wrist_roll.pos", 0.0)
            right_motor_pos.gripper = observation.get("right_gripper.pos", 0.0)

            # Set base velocity
            base_vel = msg.base_velocity
            base_vel.x_vel = observation.get("x.vel", 0.0)
            base_vel.y_vel = observation.get("y.vel", 0.0)
            base_vel.theta_vel = observation.get("theta.vel", 0.0)
            base_vel.z_vel = observation.get("z.vel", 0.0)

            # Process cameras - convert numpy arrays to CameraImage messages
            for cam_key, cam_data in observation.items():
                if isinstance(cam_data, np.ndarray):
                    camera = sourccey_pb2.CameraImage()
                    camera.name = cam_key
                    # Encode as JPEG and store raw bytes
                    _, encoded_img = cv2.imencode('.jpg', cam_data)
                    camera.image_data = encoded_img.tobytes()
                    msg.cameras.append(camera)

            return msg

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: python src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert observation to protobuf: {e}")
            raise

    def protobuf_to_action(self, action_msg: sourccey_pb2.SourcceyRobotAction) -> dict[str, Any]:
        """Convert protobuf action to internal format."""
        try:
            action = {}

            # Convert left arm action
            left_motor_pos = action_msg.left_arm_target_joints
            action.update({
                "left_shoulder_pan.pos": left_motor_pos.shoulder_pan,
                "left_shoulder_lift.pos": left_motor_pos.shoulder_lift,
                "left_elbow_flex.pos": left_motor_pos.elbow_flex,
                "left_wrist_flex.pos": left_motor_pos.wrist_flex,
                "left_wrist_roll.pos": left_motor_pos.wrist_roll,
                "left_gripper.pos": left_motor_pos.gripper,
            })

            # Convert right arm action
            right_motor_pos = action_msg.right_arm_target_joints
            action.update({
                "right_shoulder_pan.pos": right_motor_pos.shoulder_pan,
                "right_shoulder_lift.pos": right_motor_pos.shoulder_lift,
                "right_elbow_flex.pos": right_motor_pos.elbow_flex,
                "right_wrist_flex.pos": right_motor_pos.wrist_flex,
                "right_wrist_roll.pos": right_motor_pos.wrist_roll,
                "right_gripper.pos": right_motor_pos.gripper,
            })

            # Convert base action
            base_vel = action_msg.base_target_velocity
            action.update({
                "x.vel": base_vel.x_vel,
                "y.vel": base_vel.y_vel,
                "theta.vel": base_vel.theta_vel,
                "z.vel": base_vel.z_vel,
            })

            # Per-arm flags from protobuf
            action["untorque_left"] = bool(getattr(action_msg, "untorque_left", False))
            action["untorque_right"] = bool(getattr(action_msg, "untorque_right", False))

            return action

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: python src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert protobuf to action: {e}")
            raise

    def protobuf_to_observation(self, robot_state: sourccey_pb2.SourcceyRobotState) -> dict[str, Any]:
        """Convert protobuf SourcceyRobotState message to observation dictionary."""
        try:
            observation = {}

            # Process left arm state
            left_motor_pos = robot_state.left_arm_joints
            observation["left_shoulder_pan.pos"] = left_motor_pos.shoulder_pan
            observation["left_shoulder_lift.pos"] = left_motor_pos.shoulder_lift
            observation["left_elbow_flex.pos"] = left_motor_pos.elbow_flex
            observation["left_wrist_flex.pos"] = left_motor_pos.wrist_flex
            observation["left_wrist_roll.pos"] = left_motor_pos.wrist_roll
            observation["left_gripper.pos"] = left_motor_pos.gripper

            # Process right arm state
            right_motor_pos = robot_state.right_arm_joints
            observation["right_shoulder_pan.pos"] = right_motor_pos.shoulder_pan
            observation["right_shoulder_lift.pos"] = right_motor_pos.shoulder_lift
            observation["right_elbow_flex.pos"] = right_motor_pos.elbow_flex
            observation["right_wrist_flex.pos"] = right_motor_pos.wrist_flex
            observation["right_wrist_roll.pos"] = right_motor_pos.wrist_roll
            observation["right_gripper.pos"] = right_motor_pos.gripper

            # Process base velocity
            base_vel = robot_state.base_velocity
            observation["x.vel"] = base_vel.x_vel
            observation["y.vel"] = base_vel.y_vel
            observation["theta.vel"] = base_vel.theta_vel
            observation["z.vel"] = base_vel.z_vel

            # Process cameras from the cameras list
            for camera in robot_state.cameras:
                if camera.image_data:
                    try:
                        nparr = np.frombuffer(camera.image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            observation[camera.name] = frame
                    except Exception as e:
                        logger.warning(f"Failed to decode camera image {camera.name}: {e}")

            return observation

        except ImportError as e:
            logger.error(f"Failed to import protobuf modules: {e}")
            logger.error("Run the protobuf setup script first: python src/lerobot/robots/sourccey/sourccey/protobuf/compile.py")
            raise
        except Exception as e:
            logger.error(f"Failed to convert protobuf to observation: {e}")
            raise
