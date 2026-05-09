"""Standalone autonomous eval for ACT / Diffusion Policy in gym_manipulator.

Mirrors `eval_bc_policy.py` (SAC variant) but uses the standard
`policy.select_action(batch)` interface so it works for any LeRobot
PreTrainedPolicy (ACT, Diffusion, VQBeT, …). Reward returned per step is the
SARM-shaped delta/dense, `terminated=True` means SARM threshold fired.

Usage:
    uv run python -m lerobot.scripts.eval_chunk_policy \\
        --config_path=src/lerobot/rl/sim_assembling_sarm_hilserl_rabc_v6_train.json \\
        --pretrained=outputs/act_v1/checkpoints/last/pretrained_model \\
        --n-episodes=20 \\
        --policy-type=act
"""

import argparse
import logging
import sys

import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig  # noqa: F401  # register draccus type so cfg parses
from lerobot.processor import TransitionKey
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401


def parse_aux_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--policy-type", type=str, required=True, choices=["act", "diffusion"])
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--task", type=str, default="Two-stage assembly")
    ap.add_argument(
        "--temporal-ensemble-coeff",
        type=float,
        default=None,
        help="ACT only: enable temporal ensembling at eval (e.g. 0.01); forces n_action_steps=1.",
    )
    ap.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override the policy's n_action_steps (per-step replan = 1).",
    )
    ap.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="If set, write per-episode rollout MP4s here (front+wrist side-by-side).",
    )
    ap.add_argument("--cnn-ckpt", type=str, default=None,
                    help="Path to CNN binary success classifier (best.pt). Scores P(succ) per frame.")
    ap.add_argument("--cnn-thr", type=float, default=0.5,
                    help="P(succ) threshold for ep-level success (≥1 frame above => success).")
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _to_batch(obs_dict, device, task_str):
    batch = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            t = v
            if t.ndim == 3:  # CHW image — add batch dim
                t = t.unsqueeze(0)
            elif t.ndim == 1:  # state vector
                t = t.unsqueeze(0)
            batch[k] = t.to(device)
    batch["task"] = [task_str]
    return batch


@torch.no_grad()
def _save_video(frames: list, path, fps: int = 20) -> None:
    if not frames:
        return
    import imageio.v2 as imageio
    imageio.mimsave(str(path), frames, fps=fps, codec="libx264", quality=8)


def _frame_from_obs(obs_dict) -> "np.ndarray | None":
    """Stack front + wrist images side-by-side as HWC uint8 frame."""
    import torch as _torch
    f = obs_dict.get("observation.images.front")
    w = obs_dict.get("observation.images.wrist")
    if f is None or w is None:
        return None
    def _to_uint8_hwc(t):
        if isinstance(t, _torch.Tensor):
            t = t.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.dtype != _torch.uint8:
                t = (t.float() * 255.0).clamp(0, 255).byte()
            arr = t.permute(1, 2, 0).numpy()
            return arr
        arr = np.asarray(t)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[0] in (1, 3) and arr.shape[-1] != 3:  # CHW
            arr = arr.transpose(1, 2, 0)
        if arr.dtype != np.uint8:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return arr
    return np.concatenate([_to_uint8_hwc(f), _to_uint8_hwc(w)], axis=1)


def run_eval(
    env,
    env_proc,
    action_proc,
    policy: PreTrainedPolicy,
    n_episodes: int,
    task: str,
    device: str,
    video_dir=None,
    cnn_ckpt: str = None,
    cnn_thr: float = 0.5,
) -> dict:
    rewards: list[float] = []
    successes: list[int] = []
    episode_lens: list[int] = []
    cnn_successes: list[int] = []
    cnn_max_probs: list[float] = []

    if video_dir is not None:
        from pathlib import Path as _P
        video_dir = _P(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    cnn_model = None
    cnn_tx = None
    if cnn_ckpt is not None:
        import torchvision.models as tvm
        import torch.nn as nn
        from torchvision import transforms

        class _CNNCls(nn.Module):
            def __init__(self):
                super().__init__()
                bb = tvm.resnet18(weights=None)
                bb.fc = nn.Linear(bb.fc.in_features, 2)
                self.net = bb

            def forward(self, x):
                return self.net(x)

        cnn_model = _CNNCls().to(device).eval()
        cnn_model.load_state_dict(torch.load(cnn_ckpt, map_location=device, weights_only=True))
        cnn_tx = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logging.info(f"CNN classifier loaded from {cnn_ckpt}, threshold={cnn_thr}")

    for ep in range(n_episodes):
        obs, info = env.reset()
        env_proc.reset()
        action_proc.reset()
        policy.reset()  # clears action queue / temporal ensembler

        complementary_data = (
            {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
        )
        transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
        transition = env_proc(data=transition)

        ep_reward = 0.0
        ep_len = 0
        max_step_r = 0.0
        max_cum_r = 0.0
        cnn_max_p = 0.0
        cnn_success = False
        ep_frames: list = []
        while True:
            obs_dict = transition[TransitionKey.OBSERVATION]
            if video_dir is not None:
                fr = _frame_from_obs(obs_dict)
                if fr is not None:
                    ep_frames.append(fr)
            if cnn_model is not None:
                front = obs_dict.get("observation.images.front")
                if front is not None:
                    if front.ndim == 3:
                        ft = front.unsqueeze(0)
                    else:
                        ft = front
                    if ft.dtype == torch.uint8:
                        ft = ft.float() / 255.0
                    ft = cnn_tx(ft.to(device))
                    with torch.no_grad():
                        logits = cnn_model(ft)
                        p_succ = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
                    cnn_max_p = max(cnn_max_p, p_succ)
                    if p_succ >= cnn_thr:
                        cnn_success = True
            batch = _to_batch(obs_dict, device, task)
            action = policy.select_action(batch)  # (1, action_dim)
            action = action.squeeze(0)
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_proc,
                action_processor=action_proc,
            )
            r = float(transition[TransitionKey.REWARD])
            ep_reward += r
            max_step_r = max(max_step_r, r)
            max_cum_r = max(max_cum_r, ep_reward)
            ep_len += 1
            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                success = bool(transition[TransitionKey.DONE] and not transition[TransitionKey.TRUNCATED])
                break
        rewards.append(ep_reward)
        successes.append(int(success))
        episode_lens.append(ep_len)
        cnn_successes.append(int(cnn_success))
        cnn_max_probs.append(cnn_max_p)
        logging.info("ep %d: success=%s len=%d reward=%.3f max_step_r=%.3f max_cum=%.3f cnn_succ=%s cnn_max=%.3f",
                     ep, success, ep_len, ep_reward, max_step_r, max_cum_r, cnn_success, cnn_max_p)
        if video_dir is not None and ep_frames:
            tag = "succ" if success else "fail"
            out = video_dir / f"ep{ep:02d}_{tag}_len{ep_len}_bestR{max_cum_r:.2f}.mp4"
            _save_video(ep_frames, out, fps=20)
            logging.info("  -> %s", out)

    out = {
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(successes)),
        "n_success": int(sum(successes)),
        "mean_reward": float(np.mean(rewards)),
        "max_reward": float(np.max(rewards)),
        "min_reward": float(np.min(rewards)),
        "mean_len": float(np.mean(episode_lens)),
    }
    if cnn_model is not None:
        out["cnn_success_rate"] = float(np.mean(cnn_successes))
        out["cnn_n_success"] = int(sum(cnn_successes))
        out["cnn_max_prob_mean"] = float(np.mean(cnn_max_probs))
        out["cnn_max_prob_max"] = float(np.max(cnn_max_probs))
    return out


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    aux = main._aux
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # gym_manipulator imports deferred — at-import-time these pull in policy
    # configs whose registry mutations confuse draccus on the SAC train cfg.
    from lerobot.rl.gym_manipulator import (
        create_transition as _create_transition,
        make_processors as _make_processors,
        make_robot_env as _make_robot_env,
        step_env_and_process_transition as _step,
    )
    globals().update(
        create_transition=_create_transition,
        make_processors=_make_processors,
        make_robot_env=_make_robot_env,
        step_env_and_process_transition=_step,
    )

    cfg.env.teleop = None
    env, _teleop = _make_robot_env(cfg.env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Imports deferred to runtime so the at-import registry mutations do not
    # confuse draccus when parsing the (SAC) train cfg passed via --config_path.
    if aux.policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as policy_cls
    elif aux.policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as policy_cls
    else:
        raise ValueError(aux.policy_type)
    # Load the saved policy config via the parent registry (PreTrainedConfig)
    # so draccus uses the registry's `type` discriminator to pick the right
    # subclass — calling from_pretrained on the leaf subclass directly fails
    # because subclasses do not declare `type` as a dataclass field.
    from lerobot.configs.policies import PreTrainedConfig
    pretrained_config = PreTrainedConfig.from_pretrained(aux.pretrained)
    if aux.temporal_ensemble_coeff is not None and aux.policy_type == "act":
        pretrained_config.temporal_ensemble_coeff = aux.temporal_ensemble_coeff
        pretrained_config.n_action_steps = 1  # required when temporal ensembling
        logging.info(
            "Eval override: temporal_ensemble_coeff=%s, n_action_steps=1",
            aux.temporal_ensemble_coeff,
        )
    elif aux.n_action_steps is not None:
        pretrained_config.n_action_steps = aux.n_action_steps
        logging.info("Eval override: n_action_steps=%d", aux.n_action_steps)
    policy = policy_cls.from_pretrained(aux.pretrained, config=pretrained_config)
    policy.to(device)
    policy.eval()

    env_proc, action_proc = _make_processors(env, teleop_device=None, cfg=cfg.env, device=device)

    results = run_eval(
        env, env_proc, action_proc, policy, aux.n_episodes, aux.task, device,
        video_dir=aux.video_dir, cnn_ckpt=aux.cnn_ckpt, cnn_thr=aux.cnn_thr,
    )
    logging.info("=" * 60)
    logging.info("CHUNK-POLICY EVAL RESULTS (%s)", aux.policy_type)
    logging.info("=" * 60)
    for k, v in results.items():
        logging.info("%s: %s", k, v)


if __name__ == "__main__":
    main._aux = parse_aux_args()
    main()
