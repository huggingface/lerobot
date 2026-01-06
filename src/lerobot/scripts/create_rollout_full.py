import numpy as np
import requests
import torch
import cv2
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- CONFIGURATION ---
INFER_URL = "http://127.0.0.1:8080/infer"
# The actual folder where the data lives
DATASET_FULL_PATH = "dageorge1111/v122_coffee_pod"
EPISODE_TO_TEST = 0
STEP_INTERVAL = 1

def test_inference_from_dataset():
    # Construct paths correctly
    repo_id = DATASET_FULL_PATH  # "v110_coffee_pod_sade"
    root = DATASET_FULL_PATH   # "/home/daniel-kovari/my_local_datasets"
    
    # Sanity check: Ensure meta/info.json is where we expect
    # meta_check = DATASET_FULL_PATH / "meta" / "info.json"
    # if not meta_check.exists():
    #     raise FileNotFoundError(f"Missing metadata at {meta_check}. Check your folder structure!")

    print(f"Loading local dataset: {repo_id}")
    print(f"Root directory: {root}")

    # Initialize Dataset
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root
    )
   
    # Get the frame indices for the requested episode
    from_idx = dataset.meta.episodes[EPISODE_TO_TEST]["dataset_from_index"]
    to_idx = dataset.meta.episodes[EPISODE_TO_TEST]["dataset_to_index"]
    
    print(f"Rolling out Episode {EPISODE_TO_TEST} ({to_idx - from_idx} frames)")
    print(f"{'Frame':<8} | {'Ground Truth (XYZG)':<34} | {'Predicted (XYZG)':<34} | {'Error (mm)':<10}")
    print("-" * 105)

    for i in range(from_idx, to_idx, STEP_INTERVAL):
        item = dataset[i]
        
        # Extract 4D State (x, y, z, gripper)
        state_4d = item["observation.state"].flatten().tolist()
        truth_4d = item["action"].flatten().numpy()

        # Image processing
        img_tensor = item["observation.images.gripper"]
        
        # LeRobot images are typically (C, H, W). Convert to (H, W, C)
        if img_tensor.dtype == torch.float32:
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
            
        # The inference server expects BGR (OpenCV default)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_bgr, (224, 224))

        payload = {
            "wrist_image": img_resized.flatten().tolist(),
            "proprio": state_4d,
            "shape_wrist": [224, 224, 3],
            "reset": True if i == from_idx else False
        }

        try:
            resp = requests.post(INFER_URL, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                
                # Diffusion returns (Horizon, Dim), take index 0
                predictions = np.array(data["actions"]).reshape(data["shape"])
                pred_4d = predictions[0] if predictions.ndim > 1 else predictions

                # Calculate Error in XYZ (mm)
                error_pos = np.linalg.norm(truth_4d[:3] - pred_4d[:3]) * 1000 

                truth_str = f"[{truth_4d[0]:.4f}, {truth_4d[1]:.4f}, {truth_4d[2]:.4f}, {truth_4d[3]:.4f}]"
                pred_str = f"[{pred_4d[0]:.4f}, {pred_4d[1]:.4f}, {pred_4d[2]:.4f}, {pred_4d[3]:.4f}]"
                
                print(f"{i - from_idx:<8} | {truth_str:<34} | {pred_str:<34} | {error_pos:.2f}")
            else:
                print(f"Server Error {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"Error processing frame {i}: {e}")

if __name__ == "__main__":
    test_inference_from_dataset()