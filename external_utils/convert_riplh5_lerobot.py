import h5py
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Configuration
REPO_ID = "lerobot/my_custom_dataset"
DATASET_NAME = "my_custom_dataset"
ORIG_DATASET_PATH = Path("/home/razarcon/datasets/test_lerobot_conversion")
FPS = 5
ROOT_DIR = Path(f"data/{DATASET_NAME}")  # Where the dataset will be created locally

# Define your features match your HDF5 content
# Note: 'task' is not defined here but is required in add_frame()
FEATURES = {
    "action": {
        "dtype": "float64",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_action"],
    },
    "observation.state": {
        "dtype": "float64",
        "shape": (8,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper_pos"],
    },
    "observation.images.camera_external": {
        "dtype": "video",
        "shape": (360, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera_side": {
        "dtype": "video",
        "shape": (360, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera_wrist": {
        "dtype": "video",
        "shape": (360, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    }
}

def main():
    # 1. Create the empty LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=ROOT_DIR,
        use_videos=True, # Encode images to MP4
    )
    
    # 2. Iterate over all HDF5 files in the directory
    h5_files = sorted(list(ORIG_DATASET_PATH.glob("*.h5")))
    print(f"Found {len(h5_files)} episodes in {ORIG_DATASET_PATH}")
    
    for h5_path in h5_files:
        print(f"Processing {h5_path.name}...")
        with h5py.File(h5_path, "r") as f:
            # Extract hdf5 data
            actions = np.concatenate([f['arm_action'], np.expand_dims(f['gripper_action'],axis=1)], axis=1)
            observation_states = np.concatenate([f['joint_pos'], np.expand_dims(f['gripper_state'], axis=1)], axis=1)
            camera_external_imgs = f['rgb_frames'][:, 0, :]
            camera_wrist_imgs = f['rgb_frames'][:, 1, :]
            camera_side_imgs = f['rgb_frames'][:, 2, :]
            
            num_frames = observation_states.shape[0]
            
            # 3. Iterate over frames
            for i in range(num_frames):
                frame = {
                    "action": actions[i], 
                    "observation.state": observation_states[i],
                    "observation.images.camera_external": camera_external_imgs[i],
                    "observation.images.camera_side": camera_wrist_imgs[i],
                    "observation.images.camera_wrist": camera_side_imgs[i],
                    "task": "Demo Task",
                }
                
                # 4. Add frame to buffer
                dataset.add_frame(frame)
            
            # 5. Save the completed episode (one file = one episode)
            dataset.save_episode()

    # 6. Finalize (closes writers and saves metadata)
    dataset.finalize()
    print(f"Dataset created at: {dataset.root}")

if __name__ == "__main__":
    main()