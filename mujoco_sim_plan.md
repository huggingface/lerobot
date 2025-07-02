# MuJoCo Simulation Environment Plan for Koch Screwdriver Follower

## Overview
This plan outlines the development of a MuJoCo simulation environment for the Koch screwdriver follower robot, which features 5 position-controlled joints and 1 velocity-controlled screwdriver motor.

## Phase 1: Robot Model Development

### 1.1 MJCF Model Structure
Create `koch_screwdriver_follower.xml`:

```xml
<mujoco model="koch_screwdriver_follower">
  <compiler angle="radian" meshdir="meshes/"/>
  
  <option timestep="0.002" gravity="0 0 -9.81">
    <flag contact="enable" energy="enable"/>
  </option>

  <asset>
    <!-- STL/OBJ meshes for visual appearance -->
    <mesh name="base" file="base.stl"/>
    <mesh name="shoulder" file="shoulder.stl"/>
    <mesh name="upper_arm" file="upper_arm.stl"/>
    <mesh name="forearm" file="forearm.stl"/>
    <mesh name="wrist" file="wrist.stl"/>
    <mesh name="screwdriver" file="screwdriver.stl"/>
    
    <!-- Materials -->
    <material name="robot" rgba="0.7 0.7 0.7 1"/>
    <material name="tool" rgba="0.3 0.3 0.8 1"/>
  </asset>

  <worldbody>
    <!-- Base -->
    <body name="base" pos="0 0 0">
      <geom type="mesh" mesh="base" material="robot"/>
      
      <!-- Joint 1: shoulder_pan (full rotation) -->
      <body name="shoulder_pan" pos="0 0 0.1">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" 
               range="-3.14159 3.14159" damping="0.1"/>
        <geom type="mesh" mesh="shoulder" material="robot"/>
        
        <!-- Joint 2: shoulder_lift -->
        <body name="shoulder_lift" pos="0 0 0.05">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" 
                 range="-1.57 1.57" damping="0.1"/>
          <geom type="mesh" mesh="upper_arm" material="robot"/>
          
          <!-- Joint 3: elbow_flex -->
          <body name="elbow_flex" pos="0.2 0 0">
            <joint name="elbow_flex" type="hinge" axis="0 1 0" 
                   range="-1.57 1.57" damping="0.05"/>
            <geom type="mesh" mesh="forearm" material="robot"/>
            
            <!-- Joint 4: wrist_flex -->
            <body name="wrist_flex" pos="0.15 0 0">
              <joint name="wrist_flex" type="hinge" axis="0 1 0" 
                     range="-1.57 1.57" damping="0.05"/>
              
              <!-- Joint 5: wrist_roll (full rotation) -->
              <body name="wrist_roll" pos="0.05 0 0">
                <joint name="wrist_roll" type="hinge" axis="1 0 0" 
                       range="-3.14159 3.14159" damping="0.05"/>
                
                <!-- Joint 6: screwdriver (continuous rotation) -->
                <body name="screwdriver_mount" pos="0.05 0 0">
                  <joint name="screwdriver" type="hinge" axis="1 0 0" 
                         range="-inf inf" damping="0.01"/>
                  <geom name="screwdriver_tool" type="mesh" 
                        mesh="screwdriver" material="tool"
                        friction="2.0 0.005 0.0001"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Work objects -->
    <body name="workpiece" pos="0.4 0 0.1">
      <geom name="wood_block" type="box" size="0.1 0.1 0.05" 
            rgba="0.6 0.4 0.2 1" friction="1.0 0.005 0.0001"/>
      <!-- Screw holes -->
      <body name="screw_hole_1" pos="0.03 0.03 0.025">
        <geom name="hole_1" type="cylinder" size="0.003 0.025" 
              rgba="0.2 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- Position actuators for joints 1-5 -->
    <position name="shoulder_pan_actuator" joint="shoulder_pan" 
              kp="100" kd="10" ctrlrange="-3.14159 3.14159"/>
    <position name="shoulder_lift_actuator" joint="shoulder_lift" 
              kp="100" kd="10" ctrlrange="-1.57 1.57"/>
    <position name="elbow_flex_actuator" joint="elbow_flex" 
              kp="50" kd="5" ctrlrange="-1.57 1.57"/>
    <position name="wrist_flex_actuator" joint="wrist_flex" 
              kp="50" kd="5" ctrlrange="-1.57 1.57"/>
    <position name="wrist_roll_actuator" joint="wrist_roll" 
              kp="50" kd="5" ctrlrange="-3.14159 3.14159"/>
    
    <!-- Velocity actuator for screwdriver -->
    <velocity name="screwdriver_actuator" joint="screwdriver" 
              kv="10" ctrlrange="-10 10"/>
  </actuator>

  <sensor>
    <!-- Joint position sensors -->
    <jointpos name="shoulder_pan_pos" joint="shoulder_pan"/>
    <jointpos name="shoulder_lift_pos" joint="shoulder_lift"/>
    <jointpos name="elbow_flex_pos" joint="elbow_flex"/>
    <jointpos name="wrist_flex_pos" joint="wrist_flex"/>
    <jointpos name="wrist_roll_pos" joint="wrist_roll"/>
    
    <!-- Joint velocity sensors -->
    <jointvel name="screwdriver_vel" joint="screwdriver"/>
    
    <!-- Force/torque sensor for screwdriver -->
    <torque name="screwdriver_torque" joint="screwdriver"/>
  </sensor>
</mujoco>
```

### 1.2 Mesh Generation
Options for obtaining meshes:
1. **CAD from GitHub**: Check Koch robot repositories for STL/STEP files
2. **Simplified primitives**: Use cylinders/boxes for initial development
3. **CAD modeling**: Create simplified meshes in Blender/FreeCAD
4. **URDF conversion**: If URDF exists, convert using `mjcf_from_urdf`

## Phase 2: Environment Implementation

### 2.1 Base Environment Class

```python
# lerobot/common/envs/koch_screwdriver_env.py
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from typing import Dict, Any, Optional

class KochScrewdriverEnv(gym.Env):
    """MuJoCo environment for Koch screwdriver follower robot."""
    
    def __init__(
        self,
        task: str = "screw_insertion",
        render_mode: Optional[str] = None,
        control_freq: int = 50,
        sim_freq: int = 500,
        episode_length: int = 1000,
        cameras: Dict[str, Dict] = None,
    ):
        super().__init__()
        
        # Load MuJoCo model
        model_path = Path(__file__).parent / "assets" / "koch_screwdriver_follower.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        # Control settings
        self.control_freq = control_freq
        self.sim_freq = sim_freq
        self.frame_skip = sim_freq // control_freq
        self.episode_length = episode_length
        self.step_count = 0
        
        # Task configuration
        self.task = task
        self._setup_task()
        
        # Define action space (5 positions + 1 velocity)
        self.action_space = spaces.Box(
            low=np.array([-3.14, -1.57, -1.57, -1.57, -3.14, -10.0]),
            high=np.array([3.14, 1.57, 1.57, 1.57, 3.14, 10.0]),
            dtype=np.float32
        )
        
        # Define observation space
        self._setup_observation_space(cameras)
        
        # Rendering
        self.render_mode = render_mode
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None
            
        # Camera setup for observations
        self.cameras = cameras or {
            "wrist_camera": {"pos": [0.05, 0, 0.02], "parent": "wrist_roll"},
            "side_camera": {"pos": [0.5, 0.5, 0.5], "target": [0.3, 0, 0.1]}
        }
        self._setup_cameras()
    
    def _setup_task(self):
        """Configure task-specific parameters."""
        if self.task == "screw_insertion":
            self.target_hole_pos = np.array([0.43, 0.03, 0.125])
            self.screw_length = 0.03
            self.success_threshold = 0.005
        elif self.task == "screwdriver_rotation":
            self.target_rotations = 5.0  # Number of full rotations
            self.rotation_tolerance = 0.1
    
    def _setup_observation_space(self, cameras):
        """Define observation space including proprioception and images."""
        # Proprioceptive observations
        proprio_dim = 6  # 5 joint positions + 1 velocity
        
        if cameras:
            # Add image observations
            cam_spaces = {}
            for cam_name, cam_config in cameras.items():
                h = cam_config.get("height", 480)
                w = cam_config.get("width", 640)
                cam_spaces[cam_name] = spaces.Box(
                    low=0, high=255, shape=(h, w, 3), dtype=np.uint8
                )
            
            self.observation_space = spaces.Dict({
                "proprio": spaces.Box(low=-np.inf, high=np.inf, 
                                     shape=(proprio_dim,), dtype=np.float32),
                **cam_spaces
            })
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions (home position)
        self.data.qpos[0] = 0.0    # shoulder_pan
        self.data.qpos[1] = -0.5   # shoulder_lift
        self.data.qpos[2] = 1.0    # elbow_flex
        self.data.qpos[3] = -0.5   # wrist_flex  
        self.data.qpos[4] = 0.0    # wrist_roll
        self.data.qpos[5] = 0.0    # screwdriver
        
        # Reset task-specific elements
        if self.task == "screw_insertion":
            # Randomize screw hole position slightly
            offset = self.np_random.uniform(-0.02, 0.02, size=2)
            self.target_hole_pos[:2] += offset
        
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def step(self, action):
        # Apply actions
        # Position control for first 5 joints
        self.data.ctrl[:5] = action[:5]
        # Velocity control for screwdriver
        self.data.ctrl[5] = action[5]
        
        # Simulate
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward
        reward = self._compute_reward(obs, action)
        
        # Check termination
        terminated = self._check_success()
        truncated = self.step_count >= self.episode_length
        
        # Additional info
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get current observation."""
        # Joint positions (first 5 joints)
        joint_pos = self.data.qpos[:5].copy()
        
        # Screwdriver velocity
        screw_vel = self.data.qvel[5:6].copy()
        
        proprio = np.concatenate([joint_pos, screw_vel])
        
        if self.cameras:
            obs = {"proprio": proprio}
            for cam_name in self.cameras:
                obs[cam_name] = self._render_camera(cam_name)
            return obs
        else:
            return proprio
    
    def _render_camera(self, camera_name):
        """Render image from specified camera."""
        # Get camera ID
        cam_id = self.model.camera(camera_name).id
        
        # Render image
        renderer = mujoco.Renderer(self.model)
        renderer.update_scene(self.data, camera=cam_id)
        pixels = renderer.render()
        
        return pixels
    
    def _compute_reward(self, obs, action):
        """Task-specific reward computation."""
        if self.task == "screw_insertion":
            # Get screwdriver tip position
            screw_tip_id = self.model.body("screwdriver_mount").id
            screw_tip_pos = self.data.xpos[screw_tip_id].copy()
            
            # Distance to hole
            dist_to_hole = np.linalg.norm(screw_tip_pos - self.target_hole_pos)
            
            # Alignment reward (screwdriver should be vertical)
            screw_mat = self.data.xmat[screw_tip_id].reshape(3, 3)
            alignment = screw_mat[2, 2]  # z-axis alignment
            
            # Rotation reward when close to hole
            rotation_reward = 0.0
            if dist_to_hole < 0.02:
                screw_rot = self.data.qpos[5]
                rotation_reward = 0.1 * abs(self.data.qvel[5])
            
            # Torque penalty for current limiting simulation
            torque = self.data.sensordata[
                self.model.sensor("screwdriver_torque").id
            ]
            torque_penalty = 0.01 * abs(torque)
            
            reward = (
                -dist_to_hole * 10.0 +          # Distance penalty
                alignment * 2.0 +                # Alignment bonus
                rotation_reward -                # Rotation bonus when close
                torque_penalty                   # Torque penalty
            )
            
        elif self.task == "screwdriver_rotation":
            # Reward for rotating screwdriver
            rotation_vel = abs(self.data.qvel[5])
            reward = rotation_vel * 0.1
        
        return reward
    
    def _check_success(self):
        """Check if task is successfully completed."""
        if self.task == "screw_insertion":
            screw_tip_id = self.model.body("screwdriver_mount").id
            screw_tip_pos = self.data.xpos[screw_tip_id].copy()
            dist_to_hole = np.linalg.norm(screw_tip_pos - self.target_hole_pos)
            
            # Check if screwdriver is in the hole and has rotated
            if dist_to_hole < self.success_threshold:
                total_rotation = abs(self.data.qpos[5])
                if total_rotation > 2 * np.pi:  # At least one full rotation
                    return True
        
        return False
    
    def _get_info(self):
        """Get additional info for logging."""
        info = {
            "step_count": self.step_count,
            "joint_positions": self.data.qpos[:6].copy(),
            "joint_velocities": self.data.qvel[:6].copy(),
        }
        
        if self.task == "screw_insertion":
            screw_tip_id = self.model.body("screwdriver_mount").id
            screw_tip_pos = self.data.xpos[screw_tip_id].copy()
            info["distance_to_hole"] = np.linalg.norm(
                screw_tip_pos - self.target_hole_pos
            )
            info["screwdriver_torque"] = self.data.sensordata[
                self.model.sensor("screwdriver_torque").id
            ]
        
        return info
    
    def close(self):
        if self.viewer:
            self.viewer.close()
```

### 2.2 Clutch Simulation

```python
class SoftwareClutchMixin:
    """Mixin to simulate the software clutch behavior."""
    
    def __init__(self, current_limit=800, clutch_ratio=0.85, cooldown_s=1.0):
        self.current_limit = current_limit
        self.clutch_ratio = clutch_ratio
        self.cooldown_s = cooldown_s
        self.clutch_engaged = False
        self.clutch_release_time = 0.0
    
    def _apply_clutch_sim(self, action, torque):
        """Simulate clutch behavior based on torque."""
        current_time = self.data.time
        
        # Convert torque to approximate current units
        # This is a rough approximation - tune based on real robot data
        simulated_current = abs(torque) * 100
        
        threshold_on = self.current_limit * self.clutch_ratio
        threshold_off = self.current_limit * (self.clutch_ratio * 0.6)
        
        # Check cooldown
        if self.clutch_engaged and current_time < self.clutch_release_time:
            return 0.0  # Zero velocity command
        
        # Release clutch after cooldown
        if self.clutch_engaged and current_time >= self.clutch_release_time:
            self.clutch_engaged = False
        
        # Engage clutch on high current
        if not self.clutch_engaged and simulated_current >= threshold_on:
            self.clutch_engaged = True
            self.clutch_release_time = current_time + self.cooldown_s
            return 0.0
        
        return action
```

## Phase 3: Integration with LeRobot

### 3.1 Environment Registration

```python
# lerobot/common/envs/__init__.py
from .koch_screwdriver_env import KochScrewdriverEnv

# Register environments
register(
    id="KochScrewdriver-v0",
    entry_point="lerobot.common.envs:KochScrewdriverEnv",
    max_episode_steps=1000,
    kwargs={"task": "screw_insertion"}
)
```

### 3.2 Config File

```yaml
# lerobot/configs/env/koch_screwdriver.yaml
env:
  type: KochScrewdriver-v0
  task: screw_insertion
  control_freq: 50
  sim_freq: 500
  episode_length: 1000
  cameras:
    wrist_camera:
      height: 480
      width: 640
      pos: [0.05, 0, 0.02]
      parent: wrist_roll
    side_camera:
      height: 480
      width: 640
      pos: [0.5, 0.5, 0.5]
      target: [0.3, 0, 0.1]
```

## Phase 4: Real2Sim Calibration

### 4.1 System Identification
1. **Record real robot trajectories** with various loads
2. **Measure motor parameters**: gear ratios, damping, friction
3. **Calibrate dynamics**: Use optimization to match sim to real

### 4.2 Calibration Script

```python
# scripts/calibrate_koch_sim.py
import numpy as np
from scipy.optimize import minimize
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def calibrate_simulation(real_dataset_path, sim_env):
    """Calibrate simulation parameters to match real robot."""
    
    # Load real robot data
    dataset = LeRobotDataset(real_dataset_path)
    
    # Parameters to optimize
    # [kp_gains, kd_gains, damping, friction]
    initial_params = np.array([
        100, 100, 50, 50, 50,      # kp for each joint
        10, 10, 5, 5, 5,           # kd for each joint
        0.1, 0.1, 0.05, 0.05, 0.05, 0.01,  # damping
        1.0, 0.005, 0.0001         # friction coefficients
    ])
    
    def objective(params):
        """Minimize difference between sim and real trajectories."""
        # Apply parameters to sim
        apply_params_to_sim(sim_env, params)
        
        total_error = 0.0
        for episode in dataset.episodes:
            # Reset sim
            sim_env.reset()
            
            # Play episode in sim
            for action in episode.actions:
                sim_obs, _, _, _, _ = sim_env.step(action)
                real_obs = episode.observations[t]
                
                # Compute error
                error = np.mean((sim_obs - real_obs)**2)
                total_error += error
        
        return total_error
    
    # Optimize
    result = minimize(objective, initial_params, method='L-BFGS-B')
    
    return result.x
```

## Phase 5: Advanced Features

### 5.1 Haptic Feedback Simulation
- Simulate force feedback based on screwdriver torque
- Model contact forces during screw insertion
- Provide feedback signal matching real robot

### 5.2 Randomization for Sim2Real
```python
def randomize_dynamics(env):
    """Apply domain randomization for better sim2real transfer."""
    # Randomize masses
    for i in range(env.model.nbody):
        env.model.body_mass[i] *= np.random.uniform(0.9, 1.1)
    
    # Randomize friction
    for i in range(env.model.ngeom):
        env.model.geom_friction[i] *= np.random.uniform(0.8, 1.2)
    
    # Randomize actuator gains
    for i in range(env.model.nu):
        env.model.actuator_gainprm[i, 0] *= np.random.uniform(0.9, 1.1)
```

## Implementation Timeline

### Week 1-2: Basic Model
- Create simplified MJCF model
- Implement basic environment class
- Test position/velocity control

### Week 3-4: Task Implementation
- Implement screw insertion task
- Add reward functions
- Create evaluation metrics

### Week 5-6: Calibration
- Collect real robot data
- Run system identification
- Fine-tune dynamics

### Week 7-8: Integration & Testing
- Integrate with LeRobot training
- Test sim2real transfer
- Document and refine

## Resources Needed

1. **Robot CAD/Meshes**: From Koch robot GitHub or create simplified versions
2. **Real Robot Data**: Record trajectories for calibration
3. **MuJoCo License**: For development and training
4. **Compute**: For running calibration optimization

## Success Metrics

1. **Trajectory Matching**: <5cm average position error vs real robot
2. **Task Success**: >80% screw insertion success in sim
3. **Sim2Real Transfer**: >50% success when policy transfers to real
4. **Control Frequency**: Maintain 50Hz control rate
5. **Visual Fidelity**: Camera images suitable for vision-based policies 