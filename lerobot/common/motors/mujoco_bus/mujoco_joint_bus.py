import mujoco
import numpy as np
from lerobot.common.motors.feetech import FeetechMotorsBus

class MuJoCoJointBus(FeetechMotorsBus):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 joint_names: list[str]):
        # do *not* call super().__init__  (it opens /dev/ttyUSB…)
        self.m, self.d = model, data
        self.joint_names = joint_names
        self.joint_addr = np.array([self.m.joint(name).qposadr[0] for name in joint_names])
        
        # Map joint names to actuator indices - more robust approach
        self.actuator_addr = []
        for joint_name in joint_names:
            joint_id = self.m.joint(joint_name).id
            actuator_idx = None
            
            # Method 1: Find actuator by transmission target
            for i in range(self.m.nu):
                # Check actuator transmission
                if hasattr(self.m, 'actuator_trnid') and self.m.actuator_trnid[i, 0] == joint_id:
                    actuator_idx = i
                    break
            
            # Method 2: Find by name matching (fallback)
            if actuator_idx is None:
                for i in range(self.m.nu):
                    actuator_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if actuator_name:
                        # Check for exact name match or partial match
                        if (actuator_name == joint_name or 
                            joint_name.lower() in actuator_name.lower() or
                            actuator_name.lower() in joint_name.lower()):
                            actuator_idx = i
                            break
            
            # Method 3: If still not found, try index matching (last resort)
            if actuator_idx is None and len(joint_names) <= self.m.nu:
                joint_idx = joint_names.index(joint_name)
                if joint_idx < self.m.nu:
                    actuator_idx = joint_idx
                    print(f"Warning: Using index matching for joint '{joint_name}' -> actuator {actuator_idx}")
            
            if actuator_idx is None:
                # Print debug info
                print(f"Available actuators:")
                for i in range(self.m.nu):
                    act_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    print(f"  {i}: {act_name}")
                raise ValueError(f"Could not find actuator for joint '{joint_name}'")
            
            self.actuator_addr.append(actuator_idx)
        
        self.actuator_addr = np.array(self.actuator_addr)
        print(f"Joint to actuator mapping: {dict(zip(joint_names, self.actuator_addr))}")

    def connect(self):    # nothing to open
        pass

    def disconnect(self):
        pass

    # API mirror – positions in radians
    def read_positions(self):
        return self.d.qpos[self.joint_addr].copy()

    def write_positions(self, q: np.ndarray):
        # Use control inputs instead of directly setting positions
        # This allows MuJoCo to handle physics, collisions, and constraints
        self.d.ctrl[self.actuator_addr] = q
        
        # Remove direct qpos setting - let MuJoCo physics handle it
        # self.d.qpos[self.joint_addr] = q
        