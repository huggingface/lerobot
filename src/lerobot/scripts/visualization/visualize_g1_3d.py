import asyncio
import os
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
import numpy as np

from vuer import Vuer, VuerSession
from vuer.schemas import Urdf, DefaultScene, Fog, AmbientLight, DirectionalLight
from lerobot.robots.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

# Paths
ASSETS_PATH = "/home/breno/I2CA/prometheus/src/xr_teleoperate/assets/g1"
URDF_FILE = "g1_body29_hand14.urdf"

# Global robot instance
robot = None
running = True

def parse_joint_limits(urdf_path):
    """Parse URDF to get joint effort limits."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    limits = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        limit = joint.find('limit')
        if limit is not None:
            effort = float(limit.get('effort', 0))
            limits[name] = effort
    return limits

def connect_robot():
    global robot
    try:
        print("Attempting to connect to Unitree G1 Dex3...")
        config = UnitreeG1Dex3Config()
        config.is_simulation = False
        robot = UnitreeG1Dex3(config)
        try:
            robot.connect()
            # Check if hand connected (new architecture uses _left_hand_state)
            if hasattr(robot, '_left_hand_state') and robot._left_hand_state is not None:
                print("✓ Connected to Dex3 Hands!")
            else:
                print("⚠ Robot connected but Dex3 Hands NOT available (hand state is None)")
        except Exception as e:
            print(f"UnitreeG1Dex3 connect failed: {e}")
            print("Falling back to standard UnitreeG1 (body only)...")
            robot = UnitreeG1(config)
            robot.connect()
        print("Robot Connected!")
    except Exception as e:
        print(f"Failed to initialize robot: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

async def main():
    global running
    
    # 1. Setup Vuer
    app = Vuer(
        static_root=ASSETS_PATH,
        queries=dict(
            grid=True, 
            ambient=True
        ),
    )

    # 2. Parse Limits for Torque Viz (Normalization)
    joint_limits = parse_joint_limits(os.path.join(ASSETS_PATH, URDF_FILE))
    print(f"Loaded {len(joint_limits)} joint limits.")

    @app.spawn(start=True)
    async def session(sess: VuerSession):
        sess.set.grid = True
        
        # Setup Scene
        sess.upsert([
            DefaultScene(
                up=[0, 0, 1],
                aspect=1.77,
            ),
            AmbientLight(intensity=0.5),
            DirectionalLight(intensity=1, position=[5, 5, 5]),
            # Load Robot
            Urdf(
                src=URDF_FILE,
                jointValues={},
                key="robot",
                position=[0, 0, 1.0], # Raise it up a bit
                rotation=[0, 0, 0],   # Upright z-up standard? URDF seems z-up.
            ),
        ])

        print("Vuer Scene initialized. Waiting for robot data...")

        while running:
            if robot and robot.is_connected:
                obs = robot.get_observation()
                
                # Filter for joints
                joint_values = {}
                # torque_ratios = {} # For visualization later if supported
                
                for key, value in obs.items():
                    if key.endswith('.q'):
                        joint_name = key[:-2] # remove .q
                        # Remove potentially prepended robot name/prefix if mismatch?
                        # Using exact match for now.
                        joint_values[joint_name] = value

                # Update Robot Pose
                # Vuer Urdf component accepts jointValues dict {name: angle}
                sess.upsert(
                    Urdf(
                        key="robot",
                        jointValues=joint_values,
                        src=URDF_FILE, # Need to re-specify src usually? Or partial update?
                        # Vuer usually supports partial updates if we use the same key. 
                        # But simpler to re-emit the component with new props.
                        position=[0, 0, 0.95],
                    )
                )

            await asyncio.sleep(0.033) # 30hz update

    # Start Robot connection in separate thread (or just before, it blocks?)
    # robot.connect() blocks? check. user code says connect() blocks until state received.
    # So we run it before app.run, or in a thread. 
    # Thread is safer to not block the event loop?
    # Actually Vuer runs in asyncio event loop. connect_robot is blocking sync code.
    # So run connect_robot in thread.
    
    t = threading.Thread(target=connect_robot)
    t.start()
    
    # Wait a bit for connection?
    # No, the loop handles it.
    
    # Keep main thread alive? Vuer app.run blocks?
    # app.run defaults to keep_alive=True?
    # No, we called app.spawn(start=True).
    # We need to await app.run if we want to block here?
    # Actually `app.run()` is what we call.
    # But we defined `async def main`. 
    # Let's avoid async main entry point if using app.run() which is typical.
    # But Vuer usually is: app = Vuer(); @app.spawn...; app.run()
    pass

if __name__ == '__main__':
    # Initial setup
    
    # Run robot thread
    t = threading.Thread(target=connect_robot, daemon=True)
    t.start()
    
    # Run Vuer
    app = Vuer(
        static_root=ASSETS_PATH,
        # open=True # Auto open browser
    )
    
    joint_limits = parse_joint_limits(os.path.join(ASSETS_PATH, URDF_FILE))

    @app.spawn(start=True)
    async def session(sess: VuerSession):
        sess.upsert([
            DefaultScene(up=[0, 0, 1]),
            AmbientLight(intensity=0.5),
            DirectionalLight(intensity=1.0, position=[5, 5, 5]),
            Urdf(
                src="http://localhost:8012/static/" + URDF_FILE,
                key="robot",
                position=[0, 0, 1.0],
                rotation=[1.57, 0, 0], 
            ),
        ])

        while True:
            if robot and robot.is_connected:
                obs = robot.get_observation()
                
                joint_values = {}
                max_torque_ratio = 0
                max_torque_joint = ""

                # Joint Name Mapping
                # SDK uses "kLeftHipPitch", URDF uses "left_hip_pitch_joint"
                # We need to convert.
                
                import re

                def to_snake_case(name):
                    # Remove leading 'k' if present
                    if name.startswith('k') and name[1].isupper():
                        name = name[1:]
                    # Convert CamelCase to snake_case
                    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

                for key, value in obs.items():
                    if key.endswith('.q'):
                        sdk_name = key[:-2]
                        
                        # Hand joints already use URDF naming (e.g., left_hand_thumb_0_joint)
                        if '_hand_' in sdk_name:
                            urdf_name = sdk_name  # Already correct format
                        # Special cases for body
                        elif sdk_name == "kLeftAnkleRoll": urdf_name = "left_ankle_roll_joint"
                        elif sdk_name == "kRightAnkleRoll": urdf_name = "right_ankle_roll_joint"
                        # Generic Mapping for body joints (SDK uses CamelCase)
                        else:
                            urdf_name = to_snake_case(sdk_name) + "_joint"

                        joint_values[urdf_name] = value
                    
                    if key.endswith('.tau'):
                        sdk_name = key[:-4]
                        torque = abs(value)
                        limit = joint_limits.get(to_snake_case(sdk_name) + "_joint", 50.0)
                        if limit > 0:
                            ratio = torque / limit
                            if ratio > max_torque_ratio:
                                max_torque_ratio = ratio
                                max_torque_joint = sdk_name

                sess.upsert(
                    Urdf(
                        key="robot",
                        jointValues=joint_values,
                        src="http://localhost:8012/static/" + URDF_FILE,
                        position=[0, 0, 1.0], 
                        rotation=[-1.57, 0, 0],  # -90 deg X rotation for Z-up -> Y-up
                    )
                )
                
                # Overlay for torque
                # We can't easily add 2D overlay via schemas yet? 
                # Use print for now, or check if we can upsert HTML?
                # Vuer schemas usually have some UI elements.
                # Assuming not, just log it. 
                print(f"Max Torque: {max_torque_joint} = {max_torque_ratio*100:.1f}%")

            await asyncio.sleep(0.05) # 20hz

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("Exiting...")
        if robot: robot.disconnect()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    print("Starting Vuer server on https://localhost:8012")
    app.run()
