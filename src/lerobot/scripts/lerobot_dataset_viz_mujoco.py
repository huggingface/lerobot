#!/usr/bin/env python3
import argparse
import re
import signal
import sys
import time
import xml.etree.ElementTree as ET  # nosec
from pathlib import Path

from tqdm import tqdm

try:
    import cv2
    import numpy as np
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as e:
    print(f"[visualize_piper] Error: Missing dependency: {e}")
    sys.exit(1)

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("[visualize_piper] Error: 'mujoco' python binding is not installed.")
    print("Please install it: pip install mujoco")
    sys.exit(1)


def signal_handler(sig, frame):
    print("\n[visualize_piper] Interrupted by user.")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_piper_xml_path():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]

    candidates = [
        repo_root / "assets/xml/agilex_piper/piper.xml",
        Path("assets/xml/agilex_piper/piper.xml"),
        Path("/mnt/Data/workspace/robot/lerobot_piper/assets/xml/agilex_piper/piper.xml"),
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
        tree = ET.parse(base_xml_path)  # nosec
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
            # We want to exclude lights or floor if they are in the base xml,
            # but usually base xml has the robot chain.
            # We'll skip light and floor if we detect them, or just take everything
            # and wrap it in a body.
            tag = child.tag
            name = child.get("name", "")
            if tag == "light" or name == "floor":
                continue
            robot_body_content += ET.tostring(child, encoding="unicode")

        assets_dir = base_xml_path.parent / "assets"

        # Helper to inject prefix into names
        def inject_prefix(content, prefix):
            c = content
            # Regex to replace name="foo" with name="prefix_foo"
            # Be careful not to break standard names if any, but usually we want to prefix all joints/bodies.
            c = re.sub(r'name="([^"]+)"', f'name="{prefix}\\1"', c)
            c = re.sub(r'joint="([^"]+)"', f'joint="{prefix}\\1"', c)
            c = re.sub(r'body1="([^"]+)"', f'body1="{prefix}\\1"', c)
            c = re.sub(r'body2="([^"]+)"', f'body2="{prefix}\\1"', c)
            c = re.sub(r'target="([^"]+)"', f'target="{prefix}\\1"', c)
            return c

        left_body = inject_prefix(robot_body_content, "left_")
        right_body = inject_prefix(robot_body_content, "right_")

        # Define positions (approximate for now)
        # Left robot at y=-0.2, Right at y=0.2 seems reasonable for a bimanual setup
        # or -0.3 and 0.3.
        left_pos = "0 0.25 0"
        right_pos = "0 -0.25 0"

        xml_str = f"""
        <mujoco model="piper_dual">
            <compiler angle="radian" meshdir="{assets_dir}"/>
            <option integrator="implicitfast"/>

            {"".join(defaults)}

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
                <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300" mark="edge" markrgb="0.8 0.8 0.8"/>
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
            </asset>

            {"".join(assets)}

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--repo-id", type=str, default="local/dataset")
    args = parser.parse_args()

    print(f"[visualize_piper] Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(root=args.dataset, repo_id=args.repo_id)

    if dataset.meta.episodes is None or len(dataset.meta.episodes) == 0:
        print("[visualize_piper] Error: No episodes found.")
        return

    episode_meta = dataset.meta.episodes[args.episode]
    start_idx = episode_meta["dataset_from_index"]
    end_idx = episode_meta["dataset_to_index"]
    length = end_idx - start_idx
    print(f"[visualize_piper] Episode {args.episode}: {length} frames")

    # Determine if dual arm
    # Check features
    # 'observation.state' shape (14,) -> Dual
    # 'observation.state' shape (7,) or (6,) -> Single

    is_dual = False
    if "observation.state" in dataset.features:
        shape = dataset.features["observation.state"]["shape"]
        if shape[0] == 14:
            is_dual = True
            print("[visualize_piper] Detected Dual Arm configuration (14 dims).")
        else:
            print(f"[visualize_piper] Detected Single Arm configuration ({shape[0]} dims).")

    xml_path = get_piper_xml_path()
    if not xml_path:
        print("Error: piper.xml not found")
        return

    if is_dual:
        xml_string = synthesize_dual_piper_xml(xml_path)
        if not xml_string:
            return
        model = mujoco.MjModel.from_xml_string(xml_string)
    else:
        model = mujoco.MjModel.from_xml_path(str(xml_path))

    data = mujoco.MjData(model)

    # Prepare Camera Window
    camera_keys = [k for k in dataset.features if k.startswith("observation.images.")]
    if camera_keys:
        try:
            cv2.namedWindow("Cameras", cv2.WINDOW_NORMAL)
        except Exception as e:
            print(f"Warning: Could not create window: {e}")

    state = {"paused": False, "offset": 0, "step": 0}

    def key_callback(keycode):
        if keycode == 32:
            state["paused"] = not state["paused"]
        elif keycode == 263:
            state["paused"] = True
            state["step"] = -1
        elif keycode == 262:
            state["paused"] = True
            state["step"] = 1
        elif keycode == 256:
            sys.exit(0)

    pbar = tqdm(total=length, desc="[visualize_piper] Playback", unit="frame")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.lookat = np.array([0.0, 0.0, 0.2])
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        frame_dur = 1.0 / args.hz

        # Pre-calculate joint addresses if possible
        # For dual arm:
        # Left: left_joint1..6, left_joint7/8
        # Right: right_joint1..6, right_joint7/8

        left_joint_ids = []
        left_gripper_ids = []
        right_joint_ids = []
        right_gripper_ids = []

        if is_dual:
            for i in range(1, 7):
                left_joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i}"))
                right_joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_joint{i}"))

            # Grippers
            left_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint7"))
            left_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint8"))
            right_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint7"))
            right_gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_joint8"))

        else:
            # Single arm logic (fallback)
            pass

        while viewer.is_running():
            step_start = time.time()
            if state["step"] != 0:
                state["offset"] += state["step"]
                state["step"] = 0

            if state["offset"] >= length:
                state["offset"] = 0
            if state["offset"] < 0:
                state["offset"] = 0

            # Update pbar
            if pbar.n != state["offset"]:
                pbar.n = state["offset"]
                pbar.refresh()

            idx = start_idx + state["offset"]
            item = dataset[idx]

            # Update MuJoCo
            if "observation.state" in item and is_dual:
                q = item["observation.state"]
                if isinstance(q, torch.Tensor):
                    q = q.numpy()

                # q should be 14 dims: 7 left, 7 right
                left_q = q[0:7]
                right_q = q[7:14]

                # Apply Left Arm (0-5)
                for i, jid in enumerate(left_joint_ids):
                    if jid != -1:
                        data.qpos[model.jnt_qposadr[jid]] = left_q[i]

                # Apply Left Gripper (6) -> joint7 (pos), joint8 (neg)
                lg = left_q[6]
                if left_gripper_ids[0] != -1:
                    data.qpos[model.jnt_qposadr[left_gripper_ids[0]]] = lg
                if left_gripper_ids[1] != -1:
                    data.qpos[model.jnt_qposadr[left_gripper_ids[1]]] = -lg

                # Apply Right Arm (0-5)
                for i, jid in enumerate(right_joint_ids):
                    if jid != -1:
                        data.qpos[model.jnt_qposadr[jid]] = right_q[i]

                # Apply Right Gripper (6) -> joint7 (pos), joint8 (neg)
                rg = right_q[6]
                if right_gripper_ids[0] != -1:
                    data.qpos[model.jnt_qposadr[right_gripper_ids[0]]] = rg
                if right_gripper_ids[1] != -1:
                    data.qpos[model.jnt_qposadr[right_gripper_ids[1]]] = -rg

            elif "observation.state" in item and not is_dual:
                # Single arm fallback
                q = item["observation.state"]
                if isinstance(q, torch.Tensor):
                    q = q.numpy()
                # assume standard order 0-6
                data.qpos[: len(q)] = q
                if len(q) >= 7:
                    data.qpos[7] = -q[6]

            mujoco.mj_forward(model, data)
            viewer.sync()

            # Cameras
            images = []
            for k in camera_keys:
                if k in item:
                    img = item[k]
                    if isinstance(img, torch.Tensor):
                        img = img.permute(1, 2, 0).numpy()

                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8)

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        img,
                        k.replace("observation.images.", ""),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    images.append(img)

            if images:
                # Grid Layout
                n_images = len(images)
                # Compute grid dimensions
                # e.g., 1->1x1, 2->1x2, 3->2x2, 4->2x2, 5->2x3 or 3x2
                n_cols = int(np.ceil(np.sqrt(n_images)))
                n_rows = int(np.ceil(n_images / n_cols))

                # Determine cell size (use max size found)
                max_h = max(i.shape[0] for i in images)
                max_w = max(i.shape[1] for i in images)

                # Create canvas
                canvas = np.zeros((n_rows * max_h, n_cols * max_w, 3), dtype=np.uint8)

                for i, img in enumerate(images):
                    r = i // n_cols
                    c = i % n_cols

                    # Resize if necessary to fit cell exactly or just center it?
                    # Let's resize/pad to match cell size
                    h, w = img.shape[:2]

                    # Simple resize for now to fill
                    if h != max_h or w != max_w:
                        img = cv2.resize(img, (max_w, max_h))

                    y_start = r * max_h
                    x_start = c * max_w
                    canvas[y_start : y_start + max_h, x_start : x_start + max_w] = img

                cv2.imshow("Cameras", canvas)
                cv2.waitKey(1)

            if not state["paused"]:
                state["offset"] += 1

            elapsed = time.time() - step_start
            sleep_time = frame_dur - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    pbar.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
