from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# Output directory for saving the trained model
output_directory = Path("outputs/train/my_smolvla")
output_directory.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")  # or "cuda" or "cpu"

# Replace with your custom dataset repo_id (e.g., "${HF_USER}/mydataset")
dataset_id = "lerobot/svla_so100_pickplace"

# Model configuration
# Option 1: Load from a pretrained SmolVLA checkpoint (recommended for fine-tuning)
pretrained_model_id = "lerobot/smolvla_base"
load_from_pretrained = True  # Set to False to train from scratch

# Option 2: Train from scratch (only if you have a large dataset and computational resources)
# This will initialize the model with a pretrained VLM backbone but random action expert
# load_from_pretrained = False

# Load dataset metadata to get features and statistics
print(f"Loading dataset metadata from {dataset_id}...")
dataset_metadata = LeRobotDatasetMetadata(dataset_id)

if load_from_pretrained:
    print(f"Loading pretrained model from {pretrained_model_id}...")
    policy = SmolVLAPolicy.from_pretrained(pretrained_model_id)

    # Create rename map to match dataset keys to model's expected keys
    rename_map = {
        "observation.images.top": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    
    # Create preprocessor and postprocessor with dataset statistics
    # This is important for normalizing inputs/outputs to match your dataset
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=pretrained_model_id,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": rename_map},
            "normalizer_processor": {
                "stats": dataset_metadata.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset_metadata.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
else:
    print("Initializing new SmolVLA model from scratch...")
    # Note: Training from scratch requires careful configuration
    # The input/output features must match your dataset structure
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    cfg = SmolVLAConfig(input_features=input_features, output_features=output_features)
    cfg.load_vlm_weights = True  # Load pretrained VLM backbone
    policy = SmolVLAPolicy(cfg)
    
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

# Move policy to the specified device
policy.train()
policy.to(device)


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Convert delta indices to delta timestamps based on dataset FPS."""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


# SmolVLA expects action sequences of length chunk_size (default 50)
# and single observation frames (n_obs_steps=1)
delta_timestamps = {
    "action": make_delta_timestamps(policy.config.action_delta_indices, dataset_metadata.fps),
}

# Add delta timestamps for image features that actually exist in the dataset
dataset_image_keys = [k for k in dataset_metadata.features.keys() if "image" in k.lower()]
delta_timestamps |= {
    k: make_delta_timestamps(policy.config.observation_delta_indices, dataset_metadata.fps)
    for k in dataset_image_keys
}

# Add delta timestamp for state if present
if "observation.state" in dataset_metadata.features:
    delta_timestamps["observation.state"] = make_delta_timestamps(
        policy.config.observation_delta_indices, dataset_metadata.fps
    )

# Load the dataset with appropriate delta timestamps
print(f"Loading dataset {dataset_id}...")
dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

# Training configuration
batch_size = 64  # Adjust based on your GPU memory (64 works well for most GPUs)
training_steps = 20000  # Number of training steps (increase for better performance)
log_freq = 100  # Log every N steps

# Create optimizer and scheduler using SmolVLA's preset configurations
optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
lr_scheduler = policy.config.get_scheduler_preset().build(optimizer, num_training_steps=training_steps)

# Create dataloader for offline training
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device.type == "cuda",
    drop_last=True,
    num_workers=4,  # Adjust based on your system
)

print(f"\nStarting training for {training_steps} steps...")
print(f"Batch size: {batch_size}")
print(f"Device: {device}")
print(f"Output directory: {output_directory}")
print("-" * 80)

# Training loop
step = 0
done = False
while not done:
    for batch in dataloader:
        # Preprocess the batch (normalization, tokenization, etc.)
        batch = preprocessor(batch)
        
        # Forward pass: compute loss
        loss, output_dict = policy.forward(batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Logging
        if step % log_freq == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Step: {step}/{training_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e}"
            )
        
        step += 1
        if step >= training_steps:
            done = True
            break

print("-" * 80)
print("Training completed!")

# Save the trained model
print(f"\nSaving model to {output_directory}...")
policy.save_pretrained(output_directory)
preprocessor.save_pretrained(output_directory)
postprocessor.save_pretrained(output_directory)
print("Model saved successfully!")

# Optional: Push to Hugging Face Hub
# Uncomment and update with your Hugging Face username
push_to_hub = False  # Set to True to push to Hub
hub_repo_id = "YOUR_HF_USERNAME/my_smolvla_so101"  # Replace with your repo ID

if push_to_hub:
    print(f"\nPushing model to Hugging Face Hub: {hub_repo_id}...")
    policy.push_to_hub(hub_repo_id)
    preprocessor.push_to_hub(hub_repo_id)
    postprocessor.push_to_hub(hub_repo_id)
    print(f"Model pushed to: https://huggingface.co/{hub_repo_id}")

print("\n" + "=" * 80)
print("Training complete! Next steps:")
print("1. Test the model with: examples/tutorial/smolvla/using_smolvla_example.py")
print(f"2. Update model_id in the script to: {output_directory}")
print("3. Deploy on your SO101 robot!")
print("=" * 80)