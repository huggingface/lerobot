import numpy as np
import requests
import cv2
import os
import glob
from pathlib import Path

# --- CONFIGURATION ---
INFER_URL = "http://127.0.0.1:8080/infer"
IMAGE_DIR = "test_rollout"

# Updated Format: [Frame, x, y, z, qx, qy, qz, qw, gripper]
obs_samples = [
    [0,   0.0129, -0.2385, 0.0873, 0.5277, 0.4697, 0.4754, 0.5244, 0.0000],
    [27,  0.0136, -0.2385, 0.0869, 0.5273, 0.4702, 0.4759, 0.5237, -0.0227],
    [55,  0.0633, -0.2283, 0.0388, 0.5179, 0.4794, 0.4807, 0.5204, -0.0903],
    [83,  0.1048, -0.2492, 0.0366, 0.5180, 0.4787, 0.4736, 0.5274, -0.0903],
    [110, 0.1161, -0.2663, 0.0371, 0.5158, 0.4804, 0.4757, 0.5261, -0.0903],
    [138, 0.1212, -0.2754, 0.0380, 0.5213, 0.4802, 0.4703, 0.5258, -0.0903],
    [166, 0.1246, -0.2801, 0.0406, 0.5183, 0.4802, 0.4723, 0.5270, -0.0608],
    [193, 0.1244, -0.2803, 0.0407, 0.5182, 0.4807, 0.4719, 0.5270, -0.0490],
    [221, 0.0362, -0.2032, 0.0888, 0.5259, 0.4705, 0.4625, 0.5367, -0.0490],
    [249, 0.0255, -0.2070, 0.1012, 0.5226, 0.4684, 0.4858, 0.5210, -0.0490]
]

action_samples = [
    [0,   0.0200, -0.2373, 0.0834, 0.5000, 0.5000, 0.5000, 0.5000,  0.0000],
    [27,  0.0255, -0.2360, 0.0768, 0.5000, 0.5000, 0.5000, 0.5000, -0.0227],
    [55,  0.0738, -0.2257, 0.0276, 0.5000, 0.5000, 0.5000, 0.5000, -0.0903],
    [83,  0.1150, -0.2463, 0.0333, 0.5000, 0.5000, 0.5000, 0.5000, -0.0903],
    [110, 0.1257, -0.2645, 0.0336, 0.5000, 0.5000, 0.5000, 0.5000, -0.0903],
    [138, 0.1316, -0.2708, 0.0362, 0.5000, 0.5000, 0.5000, 0.5000, -0.0903],
    [166, 0.1330, -0.2729, 0.0368, 0.5000, 0.5000, 0.5000, 0.5000, -0.0608],
    [193, 0.1218, -0.2555, 0.0395, 0.5000, 0.5000, 0.5000, 0.5000, -0.0490],
    [221, 0.0321, -0.1905, 0.0892, 0.5000, 0.5000, 0.5000, 0.5000, -0.0490],
    [249, 0.0294, -0.2074, 0.1043, 0.5000, 0.5000, 0.5000, 0.5000, -0.0490]
]

def find_image(frame_idx):
    """Finds image file by frame index even if filename has timestamps."""
    pattern = os.path.join(IMAGE_DIR, f"episode_000_frame_{frame_idx:04d}*.png")
    files = glob.glob(pattern)
    return files[0] if files else None

def test_inference():
    print(f"{'Frame':<8} | {'Ground Truth (XYZG)':<34} | {'Predicted (XYZG)':<34} | {'Error (mm)':<10}")
    print("-" * 105)

    for i in range(len(obs_samples)):
        frame_idx = int(obs_samples[i][0])
        
        # Extract [x, y, z, gripper] - Indices 1, 2, 3, 8
        state_4d = [obs_samples[i][1], obs_samples[i][2], obs_samples[i][3], obs_samples[i][8]]
        
        # Truth Action [x, y, z, gripper] - Indices 1, 2, 3, 8
        truth_4d = np.array([action_samples[i][1], action_samples[i][2], action_samples[i][3], action_samples[i][8]])

        # Robust image loading
        img_path = find_image(frame_idx)
        if img_path is None:
            print(f"Frame {frame_idx:04}: No image matching pattern found in {IMAGE_DIR}")
            continue
        
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224))

        payload = {
            "wrist_image": img_resized.flatten().tolist(),
            "proprio": state_4d,
            "shape_wrist": [224, 224, 3],
            "reset": True if i == 0 else False
        }

        try:
            resp = requests.post(INFER_URL, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                
                # Reshape prediction
                actions_flat = np.array(data["actions"])
                actions_shape = tuple(data["shape"])
                predictions = actions_flat.reshape(actions_shape)
                
                # Select step: Diffusion policy usually returns (16, 4), take first step [0]
                pred_4d = predictions[0] if predictions.ndim > 1 else predictions

                # Calculate L2 Distance in XYZ (mm)
                error_pos = np.linalg.norm(truth_4d[:3] - pred_4d[:3]) * 1000 

                truth_str = f"[{truth_4d[0]:.4f}, {truth_4d[1]:.4f}, {truth_4d[2]:.4f}, {truth_4d[3]:.4f}]"
                pred_str = f"[{pred_4d[0]:.4f}, {pred_4d[1]:.4f}, {pred_4d[2]:.4f}, {pred_4d[3]:.4f}]"
                
                print(f"{frame_idx:<8} | {truth_str:<34} | {pred_str:<34} | {error_pos:.2f}")
            else:
                print(f"Server Error {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")

if __name__ == "__main__":
    test_inference()