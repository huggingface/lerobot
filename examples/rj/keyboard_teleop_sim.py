#!/usr/bin/env python
"""Keyboard teleoperation for MetaWorld/LIBERO simulation environments."""

import numpy as np
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv, MetaworldEnv
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig
import cv2

def main():
    # Choose your environment
    # Option 1: MetaWorld (use V3 tasks - see metaworld_config.json for available tasks)
    env_cfg = MetaworldEnv(
        task="metaworld-reach-v3",  # Use V3 tasks: reach-v3, push-v3, pick-place-v3, etc.
        obs_type="pixels",
    )
    
    # Option 2: LIBERO (uncomment to use instead)
    # env_cfg = LiberoEnv(
    #     task="libero_spatial",
    #     camera_name="top",
    # )
    
    # Create environment
    envs_dict = make_env(env_cfg, n_envs=1)
    vec_env = list(envs_dict.values())[0][0]  # Get the vectorized environment
    env = vec_env.envs[0]  # Get the first (and only) single environment from vector env
    
    # Create keyboard teleoperator
    keyboard_config = KeyboardEndEffectorTeleopConfig(use_gripper=True)
    keyboard = KeyboardEndEffectorTeleop(keyboard_config)
    keyboard.connect()
    
    print("Keyboard controls:")
    print("  Arrow keys: Move end-effector (up/down/left/right)")
    print("  Shift: Move up/down in Z")
    print("  Ctrl: Control gripper")
    print("  ESC: Exit")
    print("\nStarting teleoperation...")
    
    try:
        obs, info = env.reset()
        
        while True:
            # Get keyboard input (deltas)
            keyboard_action = keyboard.get_action()
            
            # Convert keyboard deltas to action
            # MetaWorld: 4D [delta_x, delta_y, delta_z, gripper] - uses relative actions
            # LIBERO: 7D [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper] - uses relative actions
            action_dim = env.action_space.shape[0]
            
            # Create action array
            action = np.zeros(action_dim, dtype=np.float32)
            
            if action_dim == 4:  # MetaWorld
                # Scale deltas appropriately (MetaWorld expects actions in [-1, 1])
                scale = 0.1  # Adjust this to control movement speed
                action[0] = keyboard_action.get("delta_x", 0.0) * scale
                action[1] = keyboard_action.get("delta_y", 0.0) * scale
                action[2] = keyboard_action.get("delta_z", 0.0) * scale
                # Map gripper: 0=close, 1=stay, 2=open -> -1=close, 0=stay, 1=open
                gripper_val = keyboard_action.get("gripper", 1.0)
                action[3] = (gripper_val - 1.0)  # Maps 0->-1, 1->0, 2->1
                
            elif action_dim == 7:  # LIBERO
                scale = 0.1
                action[0] = keyboard_action.get("delta_x", 0.0) * scale
                action[1] = keyboard_action.get("delta_y", 0.0) * scale
                action[2] = keyboard_action.get("delta_z", 0.0) * scale
                # Keep orientation deltas at 0 (or you could map additional keys)
                action[3:6] = 0.0
                # Map gripper: 0=close, 1=stay, 2=open -> -1=close, 0=stay, 1=open
                gripper_val = keyboard_action.get("gripper", 1.0)
                action[6] = (gripper_val - 1.0)  # Maps 0->-1, 1->0, 2->1
            
            # Clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display image
            if "pixels" in obs:
                img = obs["pixels"]
                if isinstance(img, dict):
                    img = list(img.values())[0]  # Get first camera view
                cv2.imshow("Simulation", img)
                cv2.waitKey(1)
            
            # Reset on episode end
            if terminated or truncated:
                obs, info = env.reset()
                print("Episode ended. Resetting...")
            
            # Check for exit
            if not keyboard.is_connected:
                break
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        keyboard.disconnect()
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()