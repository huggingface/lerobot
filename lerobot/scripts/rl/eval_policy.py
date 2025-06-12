from __future__ import annotations

import logging
import time

import torch

from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig

# Generic policy handling
from lerobot.scripts.rl.gym_manipulator import make_robot_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_policy_from_cfg(train_cfg: TrainRLServerPipelineConfig):
    """Instantiate a policy (pretrained or scratch) using the common factory."""

    # Ensure the policy config carries the pretrained path if provided via CLI shortcut
    # In TrainRLServerPipelineConfig validation, policy.pretrained_path is normally set
    # If not, try to fallback to legacy `pretrained_policy_name_or_path` field
    if getattr(train_cfg.policy, "pretrained_path", None) is None:
        legacy_field = getattr(train_cfg, "pretrained_policy_name_or_path", None)
        if legacy_field is not None:
            train_cfg.policy.pretrained_path = legacy_field  # type: ignore[attr-defined]

    policy = make_policy(cfg=train_cfg.policy, env_cfg=train_cfg.env)
    policy.eval()
    return policy


def _evaluate_once(env, policy) -> float:
    """Run a single episode and return the cumulative reward."""
    obs, info = env.reset()
    episode_reward = 0.0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        with torch.inference_mode():
            action = policy.select_action(batch=obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += float(reward)
    return episode_reward


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):  # noqa: D401 – function is a CLI entry point
    """Evaluate a pretrained policy for a fixed number of episodes and report rewards.

    The configuration cfg must include at least:
        • `pretrained_policy_name_or_path`: Path or repo id to the policy to evaluate.
        • `device`: Torch device to execute the policy on (e.g. "cuda", "cpu").
        • `num_episodes`: Number of evaluation episodes (defaults to 10 if absent).

    All other parameters (robot, teleop, wrapper, etc.) are handled by the standard
    `make_robot_env` factory.
    """

    # Validate presence of pretrained path (or allow scratch)
    if (
        getattr(cfg.policy, "pretrained_path", None) is None
        and getattr(cfg, "pretrained_policy_name_or_path", None) is None
    ):
        raise ValueError(
            "A pretrained policy path must be provided via cfg.policy.pretrained_path or pretrained_policy_name_or_path."
        )

    num_episodes: int = getattr(cfg, "num_episodes", 10)

    set_seed(getattr(cfg, "seed", 42))

    env_cfg = cfg.env
    if env_cfg is None:
        raise ValueError("The evaluation config must contain an 'env' section.")

    logger.info("Creating robot environment …")
    env = make_robot_env(env_cfg)

    logger.info("Building policy …")
    policy = _make_policy_from_cfg(cfg)

    rewards: list[float] = []
    start_time = time.perf_counter()

    for ep in range(num_episodes):
        episode_reward = _evaluate_once(env, policy)
        rewards.append(episode_reward)
        logger.info("Episode %d | reward = %.3f", ep, episode_reward)

    duration = time.perf_counter() - start_time
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    logger.info(
        "Finished %d episodes in %.1fs (%.2f avg FPS)",
        num_episodes,
        duration,
        (num_episodes / duration) if duration > 0 else float("inf"),
    )
    logger.info("Average reward over %d episodes: %.3f", num_episodes, avg_reward)

    # Print to stdout for convenience when piping/redirecting output
    print({"average_reward": avg_reward, "rewards": rewards})


if __name__ == "__main__":
    main()
