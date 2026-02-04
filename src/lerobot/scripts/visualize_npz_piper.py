#!/usr/bin/env python3
import argparse
import time
import signal
import sys
import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import numpy as np
    import mujoco
    import mujoco.viewer
except ImportError as e:
    print(f"[visualize_npz_piper] Error: Missing dependency: {e}")
    sys.exit(1)

def signal_handler(sig, frame):
    print("\n[visualize_npz_piper] Interrupted by user.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def get_piper_xml_path():
    # Try to find the XML relative to this script or in standard locations
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2] 
    
    candidates = [
        repo_root / "assets/xml/agilex_piper/piper.xml",
        Path("assets/xml/agilex_piper/piper.xml"),
        Path("/mnt/Data/workspace/robot/lerobot_piper/assets/xml/agilex_piper/piper.xml")
    ]
    
    for p in candidates:
        if p.exists():
            return p.absolute()
    return None

def synthesize_dual_piper_xml(base_xml_path):
    """
    Synthesize a MuJoCo XML with two Piper robots (left and right).
    """
    try:
        tree = ET.parse(base_xml_path)
        root = tree.getroot()
        
        # Extract assets
        assets = []
        for asset in root.findall("asset"):
            assets.append(ET.tostring(asset, encoding="unicode"))
            
        defaults = []
        for d in root.findall("default"):
            defaults.append(ET.tostring(d, encoding="unicode"))
            
        # Extract worldbody content (the robot definition)
        wb = root.find("worldbody")
        robot_body_content = ""
        for child in wb:
            tag = child.tag
            name = child.get("name", "")
            if tag == "light" or name == "floor":
                continue
            robot_body_content += ET.tostring(child, encoding="unicode")

        assets_dir = base_xml_path.parent / "assets"
        
        # Helper to inject prefix into names
        def inject_prefix(content, prefix):
            c = content
            c = re.sub(r'name="([^"]+)"', f'name="{prefix}\\1"', c)
            c = re.sub(r'joint="([^"]+)"', f'joint="{prefix}\\1"', c)
            c = re.sub(r'body1="([^"]+)"', f'body1="{prefix}\\1"', c)
            c = re.sub(r'body2="([^"]+)"', f'body2="{prefix}\\1"', c)
            c = re.sub(r'target="([^"]+)"', f'target="{prefix}\\1"', c)
            return c

        left_body = inject_prefix(robot_body_content, "left_")
        right_body = inject_prefix(robot_body_content, "right_")
        
        # Define positions
        left_pos = "0 0.25 0"
        right_pos = "0 -0.25 0"
        
        xml_str = f"""
        <mujoco model="piper_dual">
            <compiler angle="radian" meshdir="{assets_dir}"/>
            <option integrator="implicitfast"/>
            
            {''.join(defaults)}
            
            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
                <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            </asset>
            
            {''.join(assets)}
            
            <worldbody>
                <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
                <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
                
                <body name="left_root" pos="{left_pos}">
                    {left_body}
                </body>
                
                <body name="right_root" pos="{right_pos}">
                    {right_body}
                </body>
            </worldbody>
        </mujoco>
        """
        return xml_str
    except Exception as e:
        print(f"Error synthesizing XML: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Visualize Piper robot actions from an NPZ file.")
    parser.add_argument("npz_path", type=str, help="Path to the .npz file containing the action sequence.")
    parser.add_argument("--fps", type=float, default=60.0, help="Playback FPS.")
    parser.add_argument("--key", type=str, default=None, help="Key in the NPZ file to use. If None, uses the first available.")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)

    try:
        data_npz = np.load(npz_path)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        sys.exit(1)

    print(f"[visualize_npz_piper] keys found in NPZ: {list(data_npz.keys())}")

    # Determine which array to use
    motion_data = None
    if args.key:
        if args.key in data_npz:
            motion_data = data_npz[args.key]
        else:
            print(f"Error: Key '{args.key}' not found in NPZ file. Available keys: {list(data_npz.keys())}")
            sys.exit(1)
    else:
        # Use simple heuristic: prefer 'action' or 'state' or 'arr_0'
        candidates = ['action', 'smoothed_action', 'observation.state', 'state', 'arr_0']
        for c in candidates:
            if c in data_npz:
                motion_data = data_npz[c]
                print(f"[visualize_npz_piper] selected key: '{c}'")
                break
        
        if motion_data is None:
            # Fallback to the first key
            first_key = list(data_npz.keys())[0]
            motion_data = data_npz[first_key]
            print(f"[visualize_npz_piper] selected key: '{first_key}'")

    if motion_data is None:
        print("Error: No data found in NPZ.")
        sys.exit(1)

    print(f"[visualize_npz_piper] Data shape: {motion_data.shape}")
    
    if len(motion_data.shape) != 2:
        print("Error: Data must be 2D array (N, dims).")
        sys.exit(1)
        
    num_frames, dims = motion_data.shape
    
    # Check for dual arm
    is_dual = False
    if dims == 14:
        is_dual = True
        print("[visualize_npz_piper] Detected Dual Arm configuration (14 dims).")
    else:
        print(f"[visualize_npz_piper] Detected Single Arm configuration ({dims} dims).")

    xml_path = get_piper_xml_path()
    if not xml_path:
        print("Error: piper.xml not found")
        sys.exit(1)

    if is_dual:
        xml_string = synthesize_dual_piper_xml(xml_path)
        if not xml_string:
            sys.exit(1)
        model = mujoco.MjModel.from_xml_string(xml_string)
    else:
        model = mujoco.MjModel.from_xml_path(str(xml_path))

    data = mujoco.MjData(model)

    state = {"paused": False, "offset": 0, "step": 0}

    def key_callback(keycode):
        if keycode == 32: state["paused"] = not state["paused"] # Space
        elif keycode == 263: state["paused"] = True; state["step"] = -1 # Left Arrow
        elif keycode == 262: state["paused"] = True; state["step"] = 1 # Right Arrow
        elif keycode == 256: sys.exit(0) # ESC

    print(f"[visualize_npz_piper] Starting playback... ({num_frames} frames)")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        
        frame_dur = 1.0 / args.fps
        
        left_joint_ids = []
        left_gripper_ids = []
        right_joint_ids = []
        right_gripper_ids = []
        
        if is_dual:
            for i in range(1, 7):
                left_joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i}"))
                right_joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint{i}"))
            
            left_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint7"))
            left_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint8"))
            right_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint7"))
            right_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint8"))

        while viewer.is_running():
            step_start = time.time()
            if state["step"] != 0:
                state["offset"] += state["step"]
                state["step"] = 0
            
            if state["offset"] >= num_frames: state["offset"] = 0
            if state["offset"] < 0: state["offset"] = 0
            
            idx = state["offset"]
            q = motion_data[idx]
            
            # Update MuJoCo
            if is_dual:
                # 14 dims: 7 left, 7 right
                left_q = q[0:7]
                right_q = q[7:14]
                
                # Apply Left Arm (0-5)
                for i, jid in enumerate(left_joint_ids):
                    if jid != -1: data.qpos[model.jnt_qposadr[jid]] = left_q[i]
                
                # Apply Left Gripper (6) -> joint7 (pos), joint8 (neg)
                lg = left_q[6]
                if left_gripper_ids[0] != -1: data.qpos[model.jnt_qposadr[left_gripper_ids[0]]] = lg
                if left_gripper_ids[1] != -1: data.qpos[model.jnt_qposadr[left_gripper_ids[1]]] = -lg

                # Apply Right Arm (0-5)
                for i, jid in enumerate(right_joint_ids):
                    if jid != -1: data.qpos[model.jnt_qposadr[jid]] = right_q[i]
                
                # Apply Right Gripper (6) -> joint7 (pos), joint8 (neg)
                rg = right_q[6]
                if right_gripper_ids[0] != -1: data.qpos[model.jnt_qposadr[right_gripper_ids[0]]] = rg
                if right_gripper_ids[1] != -1: data.qpos[model.jnt_qposadr[right_gripper_ids[1]]] = -rg

            else:
                 # Single arm fallback
                 # assume standard order 0-6
                 data.qpos[:len(q)] = q
                 if len(q) >=7:
                     # If gripper is the 7th element (index 6)
                     # In Piper xml often joint7 and joint8 are gripper fingers
                     # Just simplified logic here
                     pass

            mujoco.mj_forward(model, data)
            viewer.sync()
            
            if not state["paused"]:
                state["offset"] += 1
            
            elapsed = time.time() - step_start
            sleep_time = frame_dur - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    main()
