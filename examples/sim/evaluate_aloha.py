from pathlib import Path

import gym_aloha
import gymnasium as gym
import imageio
import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.dact.modeling_dact_a import DACTPolicyA
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

output_directory = Path("examples/sim/eval")

env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
observation, info = env.reset()
frames = []
# Select your device
device = "cuda"

pretrained_policy_path = "lerobot/act_aloha_sim_transfer_cube_human"

policy = ACTPolicy.from_pretrained(pretrained_policy_path)
policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01, chunk_size=1000)


for _ in range(1000):
    # Preprocess observation to LeRobot format (handles image format conversion)
    processed_observation = preprocess_observation(observation)

    # Move to the same device as the policy
    processed_observation = {k: v.to(device) for k, v in processed_observation.items()}

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(processed_observation)
    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()
    observation, reward, terminated, truncated, info = env.step(numpy_action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave(output_directory / "rollout.mp4", np.stack(frames), fps=25)
