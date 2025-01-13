# import os
import random
import time
# from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
# import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from lerobot.common.policies.sac.modeling_sac import SACPolicy
from lerobot.common.policies.sac.configuration_sac import SACConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.factory import make_dataset
import wandb
import torch.nn.functional as F  # noqa: N812

class SB3toLerobotBatch:
    def __init__(self, rb: ReplayBuffer, device: str = "cpu"):
        self.rb = rb
        self.device = device

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        # Get sample from SB3 buffer
        data = self.rb.sample(batch_size)
        
        # Convert to the format expected by lerobot forward function
        batch = {
            # Stack current and next observations for the horizon
            "observation.state": torch.stack([
                torch.as_tensor(data.observations).to(self.device),
                torch.as_tensor(data.next_observations).to(self.device)
            ], dim=1),
            
            # Stack current actions (and dummy next actions)
            "action": torch.as_tensor(data.actions).unsqueeze(1).to(self.device),
            
            # Rewards and dones don't need stacking since they're only used for next step
            "next.reward": torch.as_tensor(data.rewards).unsqueeze(1).to(self.device),
            "next.done": torch.as_tensor(data.dones, dtype=torch.bool).unsqueeze(1).to(self.device)
        }
        
        return batch

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, 
                f"videos/{run_name}",
                episode_trigger=lambda x: x % 100 == 0,  # Record every 100 episodes
                step_trigger=None,  # Don't trigger on steps
                video_length=1000,  # Record up to 1000 steps per episode
                name_prefix="rl-video"
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# TODO(lilkm)
config = {
    "buffer_size": int(1e6),
    "batch_size": int(256),
    "seed": 1,
    "total_timesteps": int(500_000),
    "learning_starts": int(5e3),
    "wandb_project_name": "lilkm-test",
    "wandb_entity":None,
    "torch_deterministic": True,
    "cuda": True,
    "env_id": "Hopper-v5",
    "capture_video": True,
    "policy_frequency": int(2),
    "exp_name": "continuous",
}

run_name = f"{config['env_id']}__{config['exp_name']}__{config['seed']}__{int(time.time())}"

wandb.init(
    project=config["wandb_project_name"],
    entity=config["wandb_entity"],
    sync_tensorboard=True,
    config=config,
    name=run_name,
    monitor_gym=True,
    save_code=True,
)

writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.backends.cudnn.deterministic = config["torch_deterministic"]

device = torch.device("cuda" if torch.cuda.is_available() and config["cuda"] else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv([make_env(config["env_id"], config["seed"], 0, config["capture_video"], run_name)])
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

max_action = float(envs.single_action_space.high[0])

cfg = SACConfig()
policy = SACPolicy(cfg)

policy = policy.to(device)

actor_optimizer = optim.Adam(policy.actor.parameters(), policy.config.actor_lr)
critic_optimizer = optim.Adam(policy.critic_ensemble.parameters(), policy.config.critic_lr)
temperature_optimizer = optim.Adam([policy.log_alpha], policy.config.temperature_lr)

envs.single_observation_space.dtype = np.float32
rb = ReplayBuffer(
    config["buffer_size"],
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False,
)

# Create wrapper
rb_wrapper = SB3toLerobotBatch(rb, device="cuda")

start_time = time.time()

# TRY NOT TO MODIFY: start the game
obs, _ = envs.reset(seed=config["seed"])
for global_step in range(config["total_timesteps"]):
    # ALGO LOGIC: put action logic here
    if global_step < config["learning_starts"]:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        actions = policy.select_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    if "final_info" in infos:
        for info in infos["final_info"]:
            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            break

    # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > config["learning_starts"]:
        batch = rb_wrapper.sample(config["batch_size"])

        # Update critics every step
        critics_loss, critics_info = policy.compute_critic_loss(batch)
        critic_optimizer.zero_grad()
        critics_loss.backward()
        critic_optimizer.step()

        if global_step % config["policy_frequency"] == 0:  # TD 3 Delayed update support
            for _ in range(config["policy_frequency"]):
                # Update actor first
                actor_loss, actor_info = policy.compute_actor_loss(batch)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Then update temperature with updated actor
                temperature_loss, temperature_info = policy.compute_temperature_loss(batch)
                temperature_optimizer.zero_grad()
                temperature_loss.backward()
                temperature_optimizer.step()

                # Update temperature value
                policy.temperature = policy.log_alpha.exp().item()

        policy.update()  # Update target networks

        if global_step % 100 == 0:
            writer.add_scalar("losses/critics_loss", critics_loss.item(), global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/temperature_loss", temperature_loss.item(), global_step)
            # writer.add_scalar("losses/loss", loss.item(), global_step)

            writer.add_scalar("output/mean_q_predicts", critics_info["mean_q_predicts"], global_step)
            writer.add_scalar("output/min_q_predicts", critics_info["min_q_predicts"], global_step)
            writer.add_scalar("output/max_q_predicts", critics_info["max_q_predicts"], global_step)
            writer.add_scalar("output/temperature", policy.temperature, global_step)
            writer.add_scalar("output/mean_log_probs", actor_info["mean_log_probs"], global_step)
            writer.add_scalar("output/min_log_probs", actor_info["min_log_probs"], global_step)
            writer.add_scalar("output/max_log_probs", actor_info["max_log_probs"], global_step)
            writer.add_scalar("output/td_target_mean", critics_info["td_target_mean"], global_step)
            writer.add_scalar("output/td_target_max", critics_info["td_target_max"], global_step)
            writer.add_scalar("output/action_mean", actor_info["action_mean"], global_step)
            writer.add_scalar("output/entropy", actor_info["entropy"], global_step)
            
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

envs.close()
writer.close()