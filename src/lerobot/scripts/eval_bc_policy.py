"""Standalone autonomous eval for a (BC-pretrained) SAC policy.

Mirrors the actor.py pipeline (env_processor + action_processor) so the
reward returned per step is the SARM-shaped delta, and `terminated=True`
means the SARM success threshold fired. Reports success rate over N eps.

Usage:
    uv run python -m lerobot.scripts.eval_bc_policy \
        --config_path=src/lerobot/rl/sim_assembling_sarm_hilserl_rabc_v3_train.json \
        --pretrained=outputs/bc_pretrain_v1/last \
        --n-episodes=20 \
        [--deterministic]
"""

import argparse
import logging
import sys

import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.processor import TransitionKey


def parse_aux_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--gripper-sample-temp",
        type=float,
        default=0.0,
        help="Softmax temperature for gripper action sampling. 0 = argmax (default).",
    )
    ap.add_argument(
        "--gripper-hysteresis",
        type=int,
        default=0,
        help="Require argmax discrete class to be the same for K consecutive frames "
        "before switching the held gripper command. 0 = no hysteresis.",
    )
    ap.add_argument(
        "--gripper-fixed",
        type=int,
        default=-1,
        help="Override discrete critic with a fixed gripper class (0=CLOSE, 1=STAY, 2=OPEN). "
        "-1 = use the policy. Useful as a sanity check.",
    )
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def select_action_from_transition(
    transition,
    policy: SACPolicy,
    deterministic: bool,
    input_features,
    gripper_temp: float = 0.0,
    gripper_state=None,
) -> torch.Tensor:
    obs = transition[TransitionKey.OBSERVATION]
    obs_dict = {k: v for k, v in obs.items() if k in input_features}
    with torch.no_grad():
        if deterministic:
            _, _, mean = policy.actor(obs_dict)
            cont = torch.tanh(mean) if policy.actor.use_tanh_squash else mean
        else:
            cont, _, _ = policy.actor(obs_dict)
        if policy.config.num_discrete_actions is not None:
            if gripper_state is not None and gripper_state.get("fixed", -1) >= 0:
                discrete = torch.full_like(cont[..., :1], float(gripper_state["fixed"]))
            else:
                q = policy.discrete_critic_forward(obs_dict, use_target=False)
                if gripper_temp > 0.0:
                    probs = torch.softmax(q / gripper_temp, dim=-1)
                    proposed = torch.multinomial(probs, num_samples=1).float()
                else:
                    proposed = q.argmax(dim=-1, keepdim=True).float()
                hyst_k = (gripper_state or {}).get("hysteresis_k", 0)
                if hyst_k > 0 and gripper_state is not None:
                    cur = gripper_state.get("held")
                    cnt = gripper_state.get("count", 0)
                    last = gripper_state.get("last_proposed")
                    p = int(proposed.item())
                    if last is None or p != last:
                        cnt = 1
                    else:
                        cnt += 1
                    gripper_state["last_proposed"] = p
                    gripper_state["count"] = cnt
                    if cur is None:
                        cur = p  # initial
                    elif p != cur and cnt >= hyst_k:
                        cur = p
                        cnt = 0
                    gripper_state["held"] = cur
                    discrete = torch.full_like(proposed, float(cur))
                else:
                    discrete = proposed
            action = torch.cat([cont, discrete], dim=-1)
        else:
            action = cont
    return action.squeeze(0)


@torch.no_grad()
def run_eval(
    env,
    env_proc,
    action_proc,
    policy: SACPolicy,
    n_episodes: int,
    deterministic: bool,
    gripper_temp: float = 0.0,
    gripper_hysteresis: int = 0,
    gripper_fixed: int = -1,
) -> dict:
    rewards: list[float] = []
    successes: list[int] = []
    episode_lens: list[int] = []
    input_features = list(policy.config.input_features.keys())

    for ep in range(n_episodes):
        obs, info = env.reset()
        env_proc.reset()
        action_proc.reset()
        complementary_data = (
            {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
        )
        transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
        transition = env_proc(data=transition)

        ep_reward = 0.0
        ep_len = 0
        gripper_state = {"hysteresis_k": gripper_hysteresis, "fixed": gripper_fixed}
        while True:
            action = select_action_from_transition(
                transition, policy, deterministic, input_features, gripper_temp, gripper_state
            )
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_proc,
                action_processor=action_proc,
            )
            ep_reward += float(transition[TransitionKey.REWARD])
            ep_len += 1
            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                success = bool(transition[TransitionKey.DONE] and not transition[TransitionKey.TRUNCATED])
                break
        rewards.append(ep_reward)
        successes.append(int(success))
        episode_lens.append(ep_len)
        logging.info("ep %d: success=%s len=%d reward=%.3f", ep, success, ep_len, ep_reward)

    return {
        "n_episodes": n_episodes,
        "deterministic": deterministic,
        "success_rate": float(np.mean(successes)),
        "n_success": int(sum(successes)),
        "mean_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "mean_len": float(np.mean(episode_lens)),
    }


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    aux = main._aux  # set below
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg.env.teleop = None
    env, _teleop = make_robot_env(cfg.env)

    cfg.policy.use_torch_compile = False
    cfg.policy.pretrained_path = aux.pretrained
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.to(cfg.policy.device)
    policy.eval()

    env_proc, action_proc = make_processors(env, teleop_device=None, cfg=cfg.env, device=cfg.policy.device)

    results = run_eval(
        env,
        env_proc,
        action_proc,
        policy,
        aux.n_episodes,
        aux.deterministic,
        gripper_temp=aux.gripper_sample_temp,
        gripper_hysteresis=aux.gripper_hysteresis,
        gripper_fixed=aux.gripper_fixed,
    )
    logging.info("=" * 60)
    logging.info("BC EVAL RESULTS")
    logging.info("=" * 60)
    for k, v in results.items():
        logging.info("%s: %s", k, v)


if __name__ == "__main__":
    main._aux = parse_aux_args()
    main()
