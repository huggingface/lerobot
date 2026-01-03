import os
import sys
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Fix paths to find lerobot source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the specific implementation class
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def plot_trajectories(ckpt_dir, dataset_path, episode_idx=0, num_samples=30):
    ckpt_dir = Path(ckpt_dir).expanduser()
    print(f"\n--- Loading Checkpoint: {ckpt_dir.parent.name} ---")

    # 1. Load the Policy directly from the implementation class
    try:
        # from_pretrained is the standard HF-style way to load model.safetensors + config.json
        policy = DiffusionPolicy.from_pretrained(ckpt_dir)
        policy.to("cuda")
        policy.eval()
        print(f"✓ {policy.__class__.__name__} loaded.")
    except Exception as e:
        print(f"× Error loading policy: {e}")
        return

    # 2. Load Dataset
    dataset = LeRobotDataset(repo_id="dageorge1111/v122_coffee_pod")

    # 3. Pull one frame
    frame_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    batch = dataset[frame_idx]
    
    input_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            input_batch[k] = v.unsqueeze(0).to("cuda")

    # 4. Probe the Vector Field
    print(f"Sampling {num_samples} trajectories...")
    all_trajectories = []
    
    # Diffusion Policy expects a sequence of observations: (Batch, Sequence, Features)
    # We take our single frame and repeat it to fill the 'n_obs_steps' requirement
    n_obs_steps = policy.config.n_obs_steps
    seq_batch = {}
    for k, v in input_batch.items():
        # v is (1, ...), we need (1, n_obs_steps, ...)
        # We unsqueeze to add the sequence dimension then repeat/expand
        seq_batch[k] = v.unsqueeze(1).repeat(1, n_obs_steps, *([1] * (v.ndim - 1)))

    # Handle the image stacking if your config uses multiple cameras (standard in LeRobot)
    from lerobot.utils.constants import OBS_IMAGES
    if policy.config.image_features:
        seq_batch[OBS_IMAGES] = torch.stack([seq_batch[key] for key in policy.config.image_features], dim=-4)

    for _ in range(num_samples):
        with torch.no_grad():
            # 1. Generate the global conditioning vector from the observations
            global_cond = policy.diffusion._prepare_global_conditioning(seq_batch)
            
            # 2. Run the iterative denoising process to get the FULL horizon
            # shape: (1, horizon, action_dim)
            trajectory = policy.diffusion.conditional_sample(batch_size=1, global_cond=global_cond)
            
            all_trajectories.append(trajectory.cpu().numpy()[0])

    # 5. Visualization
    plt.figure(figsize=(10, 6))
    for traj in all_trajectories:
        # traj shape is [Horizon, Action_Dim]
        # We plot the first two dimensions of the action (likely X/Y or Joint 0/1)
        plt.plot(traj[:, 0], traj[:, 1], color='cyan', alpha=0.2)
        plt.scatter(traj[-1, 0], traj[-1, 1], color='red', s=5, alpha=0.5)

    plt.title(f"Action Horizon Distribution: {ckpt_dir.parent.name}")
    plt.xlabel("Action Dim 0")
    plt.ylabel("Action Dim 1")
    plt.grid(True, alpha=0.3)
    
    out_name = f"field_{ckpt_dir.parent.name}.png"
    plt.savefig(out_name)
    print(f"✓ Saved to {out_name}")

if __name__ == "__main__":
    dataset_root = "~/my_local_datasets/v110_coffee_pod_sade"
    
    # Path to the actual weights/config folder
    base_path = "../../outputs/train/v122_final/checkpoints"
    checkpoints = [
        f"{base_path}/080000/pretrained_model",
        f"{base_path}/100000/pretrained_model"
    ]

    for cp in checkpoints:
        if Path(cp).expanduser().exists():
            plot_trajectories(cp, dataset_root)
        else:
            print(f"Skipping {cp}: Path not found.")
