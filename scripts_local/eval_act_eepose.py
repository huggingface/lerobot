"""Eval an ACT policy trained on the eepose state space (8-D xyz+quat+grip).

Sim env outputs 15-D legacy state [joints(7), ee_pos(3), ee_quat(4), grip_qpos(1)]
when record_gripper_width=False. This script forces that mode and slices state
[7:15] post-env_proc to produce the 8-D vector the policy was trained on.

Usage:
    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl uv run python scripts_local/eval_act_eepose.py \\
        --config_path=src/lerobot/rl/sim_3stage_act_v4_eval_env.json \\
        --pretrained=outputs/act_v4_eepose_v11/checkpoints/080000/pretrained_model \\
        --n-episodes=20 --task='Three-stage assembly' \\
        --video-dir=outputs/act_v4_eepose_v11_rollouts
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.sac.configuration_sac import SACConfig  # noqa: F401
from lerobot.processor import TransitionKey
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401


def parse_aux_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--video-dir", type=str, default=None)
    ap.add_argument("--state-slice", type=str, default="7:15",
                    help="python-style slice on env observation.state, e.g. '7:15' for ee_pose+grip")
    aux, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return aux


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig) -> None:
    aux = main._aux
    slice_lo, slice_hi = (int(x) for x in aux.state_slice.split(":"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Force env to emit 15-D legacy state
    if cfg.env.processor and cfg.env.processor.gripper:
        cfg.env.processor.gripper.record_gripper_width = False

    cfg.env.teleop = None

    from lerobot.rl.gym_manipulator import (
        create_transition,
        make_processors,
        make_robot_env,
        step_env_and_process_transition,
    )

    env, _ = make_robot_env(cfg.env)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.policies import PreTrainedConfig

    pretrained_cfg = PreTrainedConfig.from_pretrained(aux.pretrained)
    pretrained_cfg.device = device
    policy = ACTPolicy.from_pretrained(aux.pretrained, config=pretrained_cfg).to(device).eval()

    env_proc, action_proc = make_processors(env, teleop_device=None, cfg=cfg.env, device=device)

    # Wrap env_proc so observation.state -> state[lo:hi]
    base_env_proc = env_proc

    class _Sliced:
        def __init__(self, base):
            self.base = base
        def reset(self):
            self.base.reset()
        def __call__(self, data=None, **kw):
            if data is None and "data" not in kw and len(kw) == 0:
                pass
            t = self.base(data) if data is not None else self.base(**kw)
            obs = t[TransitionKey.OBSERVATION]
            s = obs.get("observation.state")
            if s is not None:
                if isinstance(s, torch.Tensor):
                    obs["observation.state"] = s[..., slice_lo:slice_hi].contiguous()
                else:
                    obs["observation.state"] = np.asarray(s)[..., slice_lo:slice_hi]
            return t

    env_proc = _Sliced(base_env_proc)

    # Inline the loop from eval_chunk_policy.run_eval
    rewards, successes, lens = [], [], []
    out_dir = Path(aux.video_dir) if aux.video_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    def _to_uint8_hwc(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.dtype != torch.uint8:
                t = (t.float() * 255.0).clamp(0, 255).byte()
            return t.permute(1, 2, 0).numpy()
        a = np.asarray(t)
        if a.ndim == 4: a = a[0]
        if a.shape[0] in (1, 3) and a.shape[-1] != 3: a = a.transpose(1, 2, 0)
        if a.dtype != np.uint8: a = (a * 255).clip(0, 255).astype(np.uint8)
        return a

    for ep in range(aux.n_episodes):
        obs, info = env.reset()
        env_proc.reset()
        action_proc.reset()
        policy.reset()
        complementary_data = {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
        transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
        transition = env_proc(data=transition)

        ep_r = 0.0; ep_len = 0; max_r = 0.0; frames = []
        while True:
            obs_dict = transition[TransitionKey.OBSERVATION]
            if out_dir is not None:
                f = obs_dict.get("observation.images.front")
                w = obs_dict.get("observation.images.wrist")
                if f is not None and w is not None:
                    frames.append(np.concatenate([_to_uint8_hwc(f), _to_uint8_hwc(w)], axis=1))
            batch = {}
            for k, v in obs_dict.items():
                if isinstance(v, torch.Tensor):
                    t = v
                    if t.ndim == 3: t = t.unsqueeze(0)
                    elif t.ndim == 1: t = t.unsqueeze(0)
                    batch[k] = t.to(device)
            batch["task"] = [aux.task]
            with torch.no_grad():
                action = policy.select_action(batch)
            transition = step_env_and_process_transition(env=env, transition=transition, action=action,
                                                         env_processor=env_proc, action_processor=action_proc)
            r = float(transition[TransitionKey.REWARD]); ep_r += r; max_r = max(max_r, r); ep_len += 1
            if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
                success = bool(transition[TransitionKey.DONE] and not transition[TransitionKey.TRUNCATED])
                break
        rewards.append(ep_r); successes.append(int(success)); lens.append(ep_len)
        logging.info("ep %d: success=%s len=%d reward=%.3f max_step_r=%.3f", ep, success, ep_len, ep_r, max_r)
        if out_dir and frames:
            tag = "succ" if success else "fail"
            out = out_dir / f"ep{ep:02d}_{tag}_len{ep_len}_maxR{max_r:.2f}.mp4"
            imageio.mimsave(str(out), frames, fps=20, codec="libx264", quality=8)

    print(f"\nsuccess_rate={np.mean(successes):.3f} mean_reward={np.mean(rewards):.3f} max_reward={np.max(rewards):.3f}")


if __name__ == "__main__":
    main._aux = parse_aux_args()
    main()
