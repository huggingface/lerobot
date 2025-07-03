# so100_sim.py -----------------------------------------------------------
import mujoco
import mujoco.viewer
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from functools import cached_property

from lerobot.robots import Robot, RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.mujoco.configuration_mujoco import MuJoCoCameraConfig
from lerobot.motors.mujoco_bus.mujoco_joint_bus import MuJoCoJointBus

from .config_so101_mujoco import SO101MuJoCoConfig

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False



class SO101MuJoCo(Robot):
    """SO100 simulated robot using MuJoCo physics."""
    config_class = SO101MuJoCoConfig
    name = "so101_mujoco"

    def __init__(self, config: SO101MuJoCoConfig | str):
        if isinstance(config, str):
            config = SO101MuJoCoConfig(mjcf_path=config, id="so100_sim")
        
        super().__init__(config)
        self.config = config
        
        # Initialize MuJoCo
        self.m = mujoco.MjModel.from_xml_path(self.config.mjcf_path)
        self.d = mujoco.MjData(self.m)
        self._init_simulation()
        
        # Initialize actuators
        self.actuators = {"arm": MuJoCoJointBus(self.m, self.d, self.config.joint_names)}
        
        # Initialize cameras
        self._init_cameras()
        
        # Initialize viewer and state
        self.viewer = None
        self._is_connected = False
        self._is_calibrated = self.config.start_calibrated
        self._rerun_initialized = False

    def _init_simulation(self):
        """Initialize simulation with proper physics and home position."""
        # Set to home position if keyframe exists
        if self.m.nkey > 0 and len(self.m.key_qpos[0]) == self.m.nq:
            self.d.qpos[:] = self.m.key_qpos[0]
        else:
            # Default home position for robot joints
            home_pos = [0, -1.57079, 1.57079, 0, 0, 0]
            for i, pos in enumerate(home_pos):
                if i < len(self.d.qpos):
                    self.d.qpos[i] = pos
            
            # Initialize cube if it exists (additional DOFs beyond robot joints)
            if self.m.nq > len(home_pos):
                cube_start = len(home_pos)
                if self.config.randomize_cube_position:
                    # Randomize cube position within specified radius
                    cube_pos = self._get_random_cube_position()
                else:
                    # Use base position
                    cube_pos = self.config.cube_base_position
                
                self.d.qpos[cube_start:cube_start+3] = cube_pos  # position
                self.d.qpos[cube_start+3:cube_start+7] = [1, 0, 0, 0]  # quaternion
        
        mujoco.mj_forward(self.m, self.d)

    def _get_random_cube_position(self) -> list[float]:
        """Generate a random cube position within the specified radius of the base position."""
        base_x, base_y, base_z = self.config.cube_base_position
        radius = self.config.cube_randomization_radius
        
        # Generate random angle and distance
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius)
        
        # Calculate new position
        new_x = base_x + distance * np.cos(angle)
        new_y = base_y + distance * np.sin(angle)
        new_z = base_z  # Keep z-coordinate fixed to keep cube on table
        
        return [new_x, new_y, new_z]

    def randomize_cube_position(self):
        """Randomize the cube position within the specified radius."""
        if not self._is_connected:
            raise RuntimeError("Robot must be connected to randomize cube position")
        
        # Find cube DOFs (they come after robot joint DOFs)
        home_pos = [0, -1.57079, 1.57079, 0, 0, 0]
        cube_start = len(home_pos)
        
        print(f"Debug: Total DOFs in model: {self.m.nq}")
        print(f"Debug: Robot joints: {len(home_pos)}, Cube start index: {cube_start}")
        
        if self.m.nq > len(home_pos):
            if self.config.randomize_cube_position:
                # Get new random position
                cube_pos = self._get_random_cube_position()
                print(f"Debug: Generated random cube position: {cube_pos}")
                
                # Check current cube position before changing
                current_cube_pos = self.d.qpos[cube_start:cube_start+3].copy()
                print(f"Debug: Current cube position: {current_cube_pos}")
                
                # Set the new position
                self.d.qpos[cube_start:cube_start+3] = cube_pos
                
                # Reset cube velocity
                if hasattr(self.d, 'qvel') and len(self.d.qvel) > cube_start + 6:
                    self.d.qvel[cube_start:cube_start+6] = 0  # 3 linear + 3 angular velocities
                
                # Forward the simulation to update positions
                mujoco.mj_forward(self.m, self.d)
                
                # Let physics settle with a few simulation steps
                for _ in range(50):
                    mujoco.mj_step(self.m, self.d)
                
                # Check final cube position after settling
                final_cube_pos = self.d.qpos[cube_start:cube_start+3].copy()
                print(f"Debug: Final cube position after settling: {final_cube_pos}")
                
                # Update viewer if active
                if self.viewer is not None:
                    self.viewer.sync()
                    
                print(f"Cube randomized to position: {cube_pos}")
            else:
                print("Debug: Cube randomization is disabled in config")
        else:
            print(f"Debug: No cube found in model (nq={self.m.nq} <= robot_joints={len(home_pos)})")

    def _init_cameras(self):
        """Initialize cameras using `make_cameras_from_configs`."""
        if self.config.cameras:
            # Build cameras from provided configs
            cameras = make_cameras_from_configs(self.config.cameras)
        else:
            # When no camera configs are provided, create a sensible default MuJoCo camera config
            default_cfg = MuJoCoCameraConfig(
                model=self.m,
                data=self.d,
                fps=30,
                width=480,
                height=640,
                cam="top",
            )
            cameras = make_cameras_from_configs({"top": default_cfg})

        self.cameras = cameras

    @property
    def observation_features(self) -> dict:
        """Define observation features for the SO100 arm."""
        # Individual joint and camera features
        features: dict[str, Any] = {}
        for joint in self.config.joint_names:
            features[f"{joint}.pos"] = float

        # Camera features - MuJoCo produces (width, height, 3) shaped images
        for cam_key, cam in self.cameras.items():
            # Provide default 0 when width/height is None to satisfy type checkers
            w = cam.width if cam.width is not None else 0
            h = cam.height if cam.height is not None else 0
            features[cam_key] = (w, h, 3)

        return features

    @cached_property  
    def action_features(self) -> dict:
        """Define action features for the SO100 arm."""
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the robot."""
        if self._is_connected:
            return
            
        print("Connecting robot components...")
        
        # Connect components
        for bus in self.actuators.values():
            bus.connect()
        for cam in self.cameras.values():
            cam.connect()
            
        print("Components connected, initializing viewer...")
        
        # Initialize viewer if requested
        if self.config.show_viewer:
            try:
                # Try passive viewer first
                print("Launching MuJoCo viewer...")
                self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
                print("Viewer launched successfully")
            except Exception as e:
                print(f"Failed to launch passive viewer: {e}")
                print("Trying to continue without viewer...")
                self.viewer = None
            
        # Initialize Rerun if enabled
        if self.config.enable_rerun:
            self._init_rerun()
            
        self._is_connected = True
        print("Robot connected successfully")
        
        if calibrate and not self._is_calibrated:
            self.calibrate()

    def calibrate(self) -> None:
        """Calibrate the robot (instant for simulation)."""
        if not self._is_connected:
            raise RuntimeError("Robot must be connected before calibration")
        self._is_calibrated = True

    def configure(self) -> None:
        """Configure the robot (no-op for simulation)."""
        if not self._is_connected:
            raise RuntimeError("Robot must be connected before configuration")

    def get_observation(self) -> dict[str, Any]:
        """Get observation from the robot."""
        if not self._is_connected:
            raise RuntimeError("Robot is not connected")

        # Get joint positions as individual observations 
        joint_positions = self.actuators["arm"].read_positions()
        observations = {}
        for i, joint_name in enumerate(self.config.joint_names):
            pos = joint_positions[i]
            pos_value = pos.item() if hasattr(pos, 'item') else pos
            # Convert from radians (MuJoCo) to degrees (expected by trained policy)
            observations[f"{joint_name}.pos"] = np.degrees(pos_value)
        
        # Add camera observations
        for cam_key, cam in self.cameras.items():
            observations[cam_key] = cam.async_read()
        
        # Log to Rerun if enabled
        if self.config.enable_rerun and RERUN_AVAILABLE and self._rerun_initialized:
            self._log_to_rerun(observations, "observation")
        
        return observations

    def _map_dataset_action_to_sim(self, dataset_action: dict[str, float]) -> list[float]:
        """Map dataset joint actions to simulation joint positions with offsets."""
        sim_positions = [0.0] * len(self.config.joint_names)
        
        for dataset_joint, (sim_joint, offset) in self.config.joint_mapping.items():
            if dataset_joint in dataset_action:
                # Find the index of this joint in our simulation
                try:
                    sim_idx = self.config.joint_names.index(sim_joint)
                    sim_positions[sim_idx] = dataset_action[dataset_joint] + offset
                except ValueError:
                    print(f"Warning: Simulation joint '{sim_joint}' not found in joint_names")
                    
        return sim_positions

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to the robot."""
        if not self._is_connected:
            raise RuntimeError("Robot is not connected")
        
        # Extract joint positions from action - always assume degrees
        joint_keys = [f"{joint}.pos" for joint in self.config.joint_names]
        if all(key in action for key in joint_keys):
            # Individual joint actions format (simulation joint names)
            arm_positions_deg = [action[key] for key in joint_keys]
        elif "arm.pos" in action:
            # Array format (simulation joint order)
            arm_positions_deg = action["arm.pos"]
            if isinstance(arm_positions_deg, np.ndarray):
                arm_positions_deg = arm_positions_deg.tolist()
        elif any(key in self.config.joint_mapping for key in action.keys()):
            # Dataset joint names format - map to simulation joints with offsets
            arm_positions_deg = self._map_dataset_action_to_sim(action)
        else:
            # Fallback: use current positions (get in degrees, convert to list)
            current_rad = self.actuators["arm"].read_positions()
            arm_positions_deg = [np.degrees(pos) for pos in current_rad]
        
        # Convert to radians for MuJoCo simulation
        arm_positions_rad = [np.radians(deg) for deg in arm_positions_deg]
        
        # Send to actuators (MuJoCo expects radians)
        self.actuators["arm"].write_positions(arm_positions_rad)
        
        # Step simulation
        self.step()
        
        # Log to Rerun if enabled - log in degrees
        if self.config.enable_rerun and RERUN_AVAILABLE and self._rerun_initialized:
            action_to_log = {}
            for i, joint_name in enumerate(self.config.joint_names):
                if i < len(arm_positions_deg):
                    action_to_log[f"{joint_name}.pos"] = arm_positions_deg[i]
            self._log_to_rerun(action_to_log, "action")
        
        # Return sent action in degrees (consistent with observations)
        sent_action = {}
        for i, joint_name in enumerate(self.config.joint_names):
            if i < len(arm_positions_deg):
                sent_action[f"{joint_name}.pos"] = arm_positions_deg[i]
        return sent_action

    def step(self):
        """Advance the simulation."""
        for _ in range(self.config.n_substeps):
            mujoco.mj_step(self.m, self.d)
        
        if self.viewer is not None:
            self.viewer.sync()

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        if not self._is_connected:
            return
            
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
        for cam in self.cameras.values():
            cam.disconnect()
            
        if self.config.enable_rerun and RERUN_AVAILABLE and self._rerun_initialized:
            rr.rerun_shutdown()
            
        self._is_connected = False

    def reset_to_dataset_positions(self, dataset_positions: dict[str, float]):
        """Reset robot to dataset joint configuration (with automatic mapping and offsets)."""
        if not self._is_connected:
            raise RuntimeError("Robot must be connected to set joint positions")
        
        # Map dataset positions to simulation positions
        sim_positions_deg = self._map_dataset_action_to_sim(dataset_positions)
        sim_positions_rad = [np.radians(deg) for deg in sim_positions_deg]
        
        # Set target positions using actuator control
        self.actuators["arm"].write_positions(sim_positions_rad)
        
        # Let the physics settle with proper collision detection
        # Run simulation steps to allow contacts and constraints to be resolved
        for _ in range(100):  # More steps for proper settling
            mujoco.mj_step(self.m, self.d)
        
        # Update viewer
        if self.viewer is not None:
            self.viewer.sync()

    def reset_to_joint_positions(self, positions: list[float]):
        """Reset robot to specific joint configuration."""
        if not self._is_connected:
            raise RuntimeError("Robot must be connected to set joint positions")
        if len(positions) != len(self.config.joint_names):
            raise ValueError(f"Expected {len(self.config.joint_names)} positions, got {len(positions)}")

        # Set target positions using actuator control (positions assumed to be in radians)
        self.actuators["arm"].write_positions(positions)
        
        # Let the physics settle with proper collision detection
        # Run simulation steps to allow contacts and constraints to be resolved
        for _ in range(100):  # More steps for proper settling
            mujoco.mj_step(self.m, self.d)
        
        # Update viewer
        if self.viewer is not None:
            self.viewer.sync()

    def _init_rerun(self):
        """Initialize Rerun for logging."""
        if not RERUN_AVAILABLE:
            print("Rerun is not available (not installed)")
            return
            
        try:
            print(f"Initializing Rerun session: {self.config.rerun_session_name}")
            rr.init(self.config.rerun_session_name)
            rr.spawn(memory_limit="10%")
            self._rerun_initialized = True
            print("Rerun initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Rerun: {e}")
            self._rerun_initialized = False

    def _log_to_rerun(self, data: dict[str, Any], prefix: str):
        """Log data to Rerun."""
        if not self._rerun_initialized:
            return
            
        try:
            for key, value in data.items():
                # Organize data better - put everything under "robot" with clear separation
                if prefix == "action":
                    entity_path = f"robot/actions/{key}"
                elif prefix == "observation":
                    if key == "top":  # Camera image
                        entity_path = f"robot/camera/{key}"
                    else:  # Joint positions
                        entity_path = f"robot/joints/{key}"
                else:
                    entity_path = f"robot/{prefix}/{key}"
                
                if isinstance(value, (int, float)):
                    rr.log(entity_path, rr.Scalar(value))
                elif isinstance(value, np.ndarray):
                    if value.ndim == 3 and value.shape[2] == 3:  # RGB image
                        rr.log(entity_path, rr.Image(value))
                    else:
                        rr.log(entity_path, rr.Tensor(value))
                elif isinstance(value, list):
                    rr.log(entity_path, rr.Tensor(np.array(value)))
                        
        except Exception as e:
            print(f"Failed to log to Rerun: {e}")
