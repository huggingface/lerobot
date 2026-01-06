import numpy as np
import requests
import torch
import cv2
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INFER_URL = "http://127.0.0.1:8080/infer"
# The actual folder where the data lives
DATASET_FULL_PATH = "dageorge1111/v122_coffee_pod"
EPISODE_TO_TEST = 0
STEP_INTERVAL = 1

def test_inference_and_plot():
    # Construct paths correctly
    repo_id = DATASET_FULL_PATH
    root = DATASET_FULL_PATH
    
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
    
    actual_actions = []
    predicted_actions = []
    frame_indices = []

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

                # Collect data
                actual_actions.append(truth_4d[:3])
                predicted_actions.append(pred_4d[:3])
                frame_indices.append(i - from_idx)
                
                # Optional: Print progress
                if (i - from_idx) % 10 == 0:
                   print(f"Processed frame {i - from_idx}")

            else:
                print(f"Server Error {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"Error processing frame {i}: {e}")

    # Convert to numpy arrays
    actual_actions = np.array(actual_actions)
    predicted_actions = np.array(predicted_actions)
    frame_indices = np.array(frame_indices)

    if len(actual_actions) == 0:
        print("No data collected to plot.")
        return

    # Calculate mean deviation
    diff = np.linalg.norm(actual_actions - predicted_actions, axis=1)
    mean_deviation = np.mean(diff)
    print(f"Mean XYZ deviation: {mean_deviation:.4f}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    components = ['X', 'Y', 'Z']
    
    for dim, (ax, comp) in enumerate(zip(axes, components)):
        ax.plot(frame_indices, actual_actions[:, dim], label='Ground Truth', color='blue')
        ax.plot(frame_indices, predicted_actions[:, dim], label='Predicted', color='orange', linestyle='--')
        ax.set_ylabel(f'{comp} Action')
        ax.legend()
        ax.grid(True)
        ax.set_title(f'{comp} Component')

    axes[-1].set_xlabel('Frame Index')
    plt.suptitle(f'Predicted vs Actual Actions (Episode {EPISODE_TO_TEST})\nMean Deviation: {mean_deviation:.4f}')
    
    output_file = f'rollout_viz_episode_{EPISODE_TO_TEST}.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    test_inference_and_plot()
