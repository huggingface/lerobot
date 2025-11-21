import argparse
import pathlib
from pathlib import Path
import threading
from threading import Lock, Thread
from typing import Dict

import mujoco
import mujoco.viewer
import numpy as np
try:
    import rclpy
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    print("Warning: rclpy not found. Camera image publishing will be disabled.")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import yaml

from .image_publish_utils import ImagePublishProcess
from .metric_utils import check_contact, check_height
from .sim_utilts import get_subtree_body_names
from .unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

GR00T_WBC_ROOT = Path(__file__).resolve().parent.parent  # Points to mujoco_sim_g1/


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = None,
        onscreen: bool = False,
        offscreen: bool = False,
    ):
        # Avoid mutable default argument gotcha
        if camera_configs is None:
            camera_configs = {}
        
        # global_view is only set up for this specifc scene for now.
        if config["ROBOT_SCENE"] == "gr00t_wbc/control/robot_model/model_data/g1/scene_29dof.xml":
            camera_configs["global_view"] = {
                "height": 400,
                "width": 400,
            }
        self.config = config
        self.env_name = env_name
        self.num_body_dof = self.config["NUM_JOINTS"]
        self.num_hand_dof = self.config["NUM_HAND_JOINTS"]
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.config["motor_effort_limit_list"])
        self.camera_configs = camera_configs
        
        # Debug: print camera config
        if len(camera_configs) > 0:
            print(f"✓ DefaultEnv initialized with {len(camera_configs)} camera(s): {list(camera_configs.keys())}")

        # Thread safety lock
        self.reward_lock = Lock()

        # Unitree bridge will be initialized by the simulator
        self.unitree_bridge = None

        # Store display mode
        self.onscreen = onscreen

        # Initialize scene (defined in subclasses)
        self.init_scene()
        self.last_reward = 0

        # Setup offscreen rendering if needed
        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        
        # Image publishing subprocess (initialized separately)
        self.image_publish_process = None

    def init_scene(self):
        """Initialize the default robot scene"""
        self.mj_model = mujoco.MjModel.from_xml_path(
            str(pathlib.Path(GR00T_WBC_ROOT) / self.config["ROBOT_SCENE"])
        )
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            # viewer camera
            self.viewer.cam.azimuth = 120  # Horizontal rotation in degrees
            self.viewer.cam.elevation = -30  # Vertical tilt in degrees
            self.viewer.cam.distance = 2.0  # Distance from camera to target
            self.viewer.cam.lookat = np.array([0, 0, 0.5])  # Point the camera is looking at

        # Note that the actuator order is the same as the joint order in the mujoco model.
        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.config["NUM_JOINTS"], \
            f"Expected {self.config['NUM_JOINTS']} body joints, got {len(self.body_joint_index)}"
        # Hand joints are optional (some models don't have hands)
        if self.config.get("NUM_HAND_JOINTS", 0) > 0:
            expected_hands = self.config["NUM_HAND_JOINTS"]
            if len(self.left_hand_index) != expected_hands or len(self.right_hand_index) != expected_hands:
                print(f"Warning: Expected {expected_hands} hand joints, got left={len(self.left_hand_index)}, right={len(self.right_hand_index)}")
                print("Continuing without hands...")

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

    def init_renderers(self):
        # Initialize camera renderers
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer
    
    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        """Start image publishing subprocess using ZMQ"""
        # Use spawn method for better GIL isolation, or configured method
        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()
        print(f"✓ Started image publishing subprocess on ZMQ port {camera_port}")

    def compute_body_torques(self) -> np.ndarray:
        """Compute body torques based on the current robot state"""
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + 7 - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + 6 - 1]
                        )
                    )
        return body_torques

    def compute_hand_torques(self) -> np.ndarray:
        """Compute hand torques based on the current robot state"""
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + 7 - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + 6 - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + 7 - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + 6 - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        """Compute body joint positions based on the current command"""
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        """Compute hand joint positions based on the current command"""
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        """Prepare observation dictionary from the current robot state"""
        obs = {}
        obs["floating_base_pose"] = self.mj_data.qpos[:7]
        obs["floating_base_vel"] = self.mj_data.qvel[:6]
        obs["floating_base_acc"] = self.mj_data.qacc[:6]
        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]
        obs["secondary_imu_vel"] = self.mj_data.cvel[self.torso_index]
        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + 7 - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + 6 - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + 6 - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + 7 - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + 6 - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + 6 - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                # Get Cartesian pose and velocity of the band_attached_link
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],  # link position in world
                        self.mj_data.xquat[
                            self.band_attached_link
                        ],  # link quaternion in world [w,x,y,z]
                        np.zeros(6),  # placeholder for velocity
                    ]
                )

                # Get velocity in world frame
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,  # 0 for world frame
                )

                # Reorder velocity from [ang, lin] to [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                # explicitly resetting the force when the band is not enabled
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        # self.check_self_collision()

    def kinematics_step(self):
        """
        Run kinematics only: compute the qpos of the robot and directly set the qpos.
        For debugging purposes.
        """
        if self.unitree_bridge is not None:
            self.unitree_bridge.PublishLowState(self.prepare_obs())
            if self.unitree_bridge.joystick:
                self.unitree_bridge.PublishWirelessController()

        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                # Get Cartesian pose and velocity of the band_attached_link
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],  # link position in world
                        self.mj_data.xquat[
                            self.band_attached_link
                        ],  # link quaternion in world [w,x,y,z]
                        np.zeros(6),  # placeholder for velocity
                    ]
                )

                # Get velocity in world frame
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,  # 0 for world frame
                )

                # Reorder velocity from [ang, lin] to [lin, ang]
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()

                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                # explicitly resetting the force when the band is not enabled
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)

        body_qpos = self.compute_body_qpos()  # (num_body_dof,)
        hand_qpos = self.compute_hand_qpos()  # (num_hand_dof * 2,)

        self.mj_data.qpos[self.body_joint_index + 7 - 1] = body_qpos
        self.mj_data.qpos[self.left_hand_index + 7 - 1] = hand_qpos[: self.num_hand_dof]
        self.mj_data.qpos[self.right_hand_index + 7 - 1] = hand_qpos[self.num_hand_dof :]

        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        mujoco.mj_comPos(self.mj_model, self.mj_data)

    def apply_perturbation(self, key):
        """Apply perturbation to the robot"""
        # Add velocity perturbations in body frame
        perturbation_x_body = 0.0  # forward/backward in body frame
        perturbation_y_body = 0.0  # left/right in body frame
        if key == "up":
            perturbation_x_body = 1.0  # forward
        elif key == "down":
            perturbation_x_body = -1.0  # backward
        elif key == "left":
            perturbation_y_body = 1.0  # left
        elif key == "right":
            perturbation_y_body = -1.0  # right

        # Transform body frame velocity to world frame using MuJoCo's rotation
        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]  # [w, x, y, z] quaternion

        # Use MuJoCo's robust quaternion rotation (handles invalid quaternions automatically)
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        # Apply to base linear velocity in world frame
        self.mj_data.qvel[0] += vel_world[0]  # world X velocity
        self.mj_data.qvel[1] += vel_world[1]  # world Y velocity

        # Update dynamics after velocity change
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        """Calculate reward. Should be implemented by subclasses."""
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        """Thread-safe way to get the last calculated reward."""
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        """Set the unitree bridge from the simulator"""
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        """Get privileged observation. Should be implemented by subclasses."""
        return {}

    def update_render_caches(self):
        """Update render cache and shared memory for subprocess."""
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()
        
        # Update shared memory if image publishing process is available
        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band is not None:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        """Check if the robot has fallen"""
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        """Check for self-collision of the robot"""
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)


class CubeEnv(DefaultEnv):
    """Environment with a cube object for pick and place tasks"""

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
    ):
        # Override the robot scene
        config = config.copy()  # Create a copy to avoid modifying the original
        config["ROBOT_SCENE"] = "gr00t_wbc/control/robot_model/model_data/g1/pnp_cube_43dof.xml"
        super().__init__(config, "cube", {}, onscreen, offscreen)

    def update_reward(self):
        """Calculate reward based on gripper contact with cube and cube height"""
        right_hand_body = [
            "right_hand_thumb_2_link",
            "right_hand_middle_1_link",
            "right_hand_index_1_link",
        ]
        gripper_cube_contact = check_contact(
            self.mj_model, self.mj_data, right_hand_body, "cube_body"
        )
        cube_lifted = check_height(self.mj_model, self.mj_data, "cube", 0.85, 2.0)

        with self.reward_lock:
            self.last_reward = gripper_cube_contact & cube_lifted


class BoxEnv(DefaultEnv):
    """Environment with a box object for manipulation tasks"""

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
    ):
        # Override the robot scene
        config = config.copy()  # Create a copy to avoid modifying the original
        config["ROBOT_SCENE"] = "gr00t_wbc/control/robot_model/model_data/g1/lift_box_43dof.xml"
        super().__init__(config, "box", {}, onscreen, offscreen)

    def reward(self):
        """Calculate reward based on gripper contact with cube and cube height"""
        left_hand_body = [
            "left_hand_thumb_2_link",
            "left_hand_middle_1_link",
            "left_hand_index_1_link",
        ]
        right_hand_body = [
            "right_hand_thumb_2_link",
            "right_hand_middle_1_link",
            "right_hand_index_1_link",
        ]
        gripper_box_contact = check_contact(self.mj_model, self.mj_data, left_hand_body, "box_body")
        gripper_box_contact &= check_contact(
            self.mj_model, self.mj_data, right_hand_body, "box_body"
        )
        box_lifted = check_height(self.mj_model, self.mj_data, "box", 0.92, 2.0)

        print("gripper_box_contact: ", gripper_box_contact, "box_lifted: ", box_lifted)

        with self.reward_lock:
            self.last_reward = gripper_box_contact & box_lifted
            return self.last_reward


class BottleEnv(DefaultEnv):
    """Environment with a cylinder object for manipulation tasks"""

    def __init__(
        self,
        config: Dict[str, any],
        onscreen: bool = False,
        offscreen: bool = False,
    ):
        # Override the robot scene
        config = config.copy()  # Create a copy to avoid modifying the original
        config["ROBOT_SCENE"] = "gr00t_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml"
        camera_configs = {
            "egoview": {
                "height": 400,
                "width": 400,
            },
        }
        super().__init__(
            config, "cylinder", camera_configs, onscreen, offscreen
        )

        self.bottle_body = self.mj_model.body("bottle_body")
        self.bottle_geom = self.mj_model.geom("bottle")

        if self.viewer is not None:
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewer.cam.fixedcamid = self.mj_model.camera("egoview").id

    def update_reward(self):
        """Calculate reward based on gripper contact with cylinder and cylinder height"""
        pass

    def get_privileged_obs(self):
        obs_pos = self.mj_data.xpos[self.bottle_body.id]
        obs_quat = self.mj_data.xquat[self.bottle_body.id]
        return {"bottle_pos": obs_pos, "bottle_quat": obs_quat}


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(self, config: Dict[str, any], env_name: str = "default", **kwargs):
        self.config = config
        self.env_name = env_name

        # Initialize ROS 2 node (optional, only if rclpy is available)
        if HAS_RCLPY:
            if not rclpy.ok():
                rclpy.init()
                self.node = rclpy.create_node("sim_mujoco")
                self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
                self.thread.start()
            else:
                self.thread = None
                executor = rclpy.get_global_executor()
                self.node = executor.get_nodes()[0]  # will only take the first node
        else:
            self.node = None
            self.thread = None

        # Set update frequencies
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)

        # Create the appropriate environment based on name
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        elif env_name == "pnp_cube":
            self.sim_env = CubeEnv(config, **kwargs)
        elif env_name == "lift_box":
            self.sim_env = BoxEnv(config, **kwargs)
        elif env_name == "pnp_bottle":
            self.sim_env = BottleEnv(config, **kwargs)
        else:
            raise ValueError(f"Invalid environment name: {env_name}")

        # Initialize the DDS communication layer - should be safe to call multiple times

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            # If it fails because it's already initialized, that's okay
            print(f"Note: Channel factory initialization attempt: {e}")

        # Initialize the unitree bridge and pass it to the environment
        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        # Initialize additional components
        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        # Create simulation thread
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()
    
    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        """Start the image publish subprocess"""
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        """Initialize subscribers. Can be overridden by subclasses."""
        pass

    def init_publisher(self):
        """Initialize publishers. Can be overridden by subclasses."""
        pass

    def init_unitree_bridge(self):
        """Initialize the unitree SDK bridge"""
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        import time
        sim_cnt = 0
        last_time = time.time()

        print(f"Starting simulation loop. Viewer: {self.sim_env.viewer is not None}")
        
        try:
            while (
                self.sim_env.viewer and self.sim_env.viewer.is_running()
            ) or self.sim_env.viewer is None:
                # Run simulation step
                self.sim_env.sim_step()

                # Update viewer at viewer rate
                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                # Calculate reward at reward rate
                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                # Update render caches at image rate
                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Sleep to maintain correct rate (simple timing without ROS)
                elapsed = time.time() - last_time
                sleep_time = max(0, self.sim_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()

                sim_cnt += 1
            
            print(f"Loop exited. Viewer running: {self.sim_env.viewer.is_running() if self.sim_env.viewer else 'No viewer'}")
        except KeyboardInterrupt:
            # User pressed Ctrl+C - exit cleanly
            print("Keyboard interrupt received")
            pass
        except Exception as e:
            print(f"Exception in simulation loop: {e}")
            import traceback
            traceback.print_exc()
            self.close()

    def __del__(self):
        """Clean up resources when simulator is deleted"""
        self.close()

    def reset(self):
        """Reset the simulation. Can be overridden by subclasses."""
        self.sim_env.reset()

    def close(self):
        """Close the simulation. Can be overridden by subclasses."""
        try:
            # Close viewer
            if hasattr(self.sim_env, "viewer") and self.sim_env.viewer is not None:
                self.sim_env.viewer.close()

            # Shutdown ROS (if available)
            if HAS_RCLPY and rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        obs = self.sim_env.get_privileged_obs()
        # TODO: add ros2 topic to get privileged obs
        return obs

    def handle_keyboard_button(self, key):
        # Only handles keyboard buttons for default env.
        if self.env_name == "default":
            self.sim_env.handle_keyboard_button(key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument(
        "--config",
        type=str,
        default="./gr00t_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml",
        help="config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])

    simulation = BaseSimulator(config)
    simulation.start_as_thread()
