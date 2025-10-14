import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

# Device to use for training
device = "mps"  # or "cuda", or "cpu"

# Load the dataset used for training
repo_id = "lerobot/example_hil_serl_dataset"
dataset = LeRobotDataset(repo_id)

# Configure the policy to extract features from the image frames
camera_keys = dataset.meta.camera_keys

config = RewardClassifierConfig(
    num_cameras=len(camera_keys),
    device=device,
    # backbone model to extract features from the image frames
    model_name="microsoft/resnet-18",
)

# Make policy, preprocessor, and optimizer
policy = make_policy(config, ds_meta=dataset.meta)
optimizer = config.get_optimizer_preset().build(policy.parameters())
preprocessor, _ = make_pre_post_processors(policy_cfg=config, dataset_stats=dataset.meta.stats)


classifier_id = "fracapuano/reward_classifier_hil_serl_example"

# Instantiate a dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    total_accuracy = 0
    for batch in dataloader:
        # Preprocess the batch and move it to the correct device.
        batch = preprocessor(batch)

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += output_dict["accuracy"]

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")

print("Training finished!")

# You can now save the trained policy.
policy.push_to_hub(classifier_id)
