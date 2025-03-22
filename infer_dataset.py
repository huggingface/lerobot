import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(repo_id="aloha_bear", root="/home/shalin/lerobot/root/aloha_bear")

ds_meta = dataset.meta
print(f"Total episodes: {ds_meta.total_episodes}")
print(f"Frames per second: {ds_meta.fps}")
print(f"Robot type: {ds_meta.robot_type}")
print(f"Camera keys: {ds_meta.camera_keys}")


# Load the first frame from the dataset
first_frame = dataset[0]

# Display available keys in the dataset
print("Available keys in the frame:", first_frame.keys())

policy = PI0Policy.from_pretrained("lerobot/pi0")
policy.config.input_features["observation.images.hand_image"] = PolicyFeature(
    type=FeatureType.VISUAL,
    shape=(3, 480, 640),  # optionally specify shape
)

policy.config.output_features["action"] = PolicyFeature(
    type=FeatureType.ACTION,
    shape=(8,),
)
policy.config.normalization_mapping["ACTION"] = NormalizationMode.IDENTITY


# Prepare observation for the policy running in Pytorch
state = first_frame["observation.state"]
image = first_frame["observation.images.hand_image"]
task = first_frame["task"]
action = first_frame["action"]


state = state.to(torch.float32)
image = image.to(torch.float32) / 255

print("state.shape", state.shape)
print("image.shape", image.shape)
print("action.shape", action.shape)
print("task", task)

# Send data tensors from CPU to GPU
state = state.to("cuda", non_blocking=True)
image = image.to("cuda", non_blocking=True)

# Add extra (empty) batch dimension, required to forward the policy
state = state.unsqueeze(0)
image = image.unsqueeze(0)

print("DIR(policy):", dir(policy))  # see what attributes exist
if hasattr(policy, "config"):
    print("policy.config =", policy.config)
else:
    print("policy has no attribute 'config'")

# 1) Figure out how many tokens the policy sees
with torch.no_grad():
    # The policy has a language_tokenizer(...) function
    encoding = policy.language_tokenizer(task, return_tensors="pt", padding=False, truncation=True)
    # shape is [1, n_tokens]; for "pick fruit" maybe n_tokens=2 or 3
    n_tokens = encoding["input_ids"].shape[1]

print(f"Text '{task}' => {n_tokens} tokens")


# Create the policy input dictionary
observation = {
    "task": "p",
    "observation.state": state,
    "observation.images.hand_image": image,
}

# Predict the next action with respect to the current observation
action = policy.select_action(observation)

# Prepare the action for the environment
numpy_action = action.squeeze(0).to("cpu").numpy()



# Print the resulting action
print("Predicted action:", action)

