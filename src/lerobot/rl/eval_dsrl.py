#!/usr/bin/env python
"""Quick DSRL / vanilla DP eval with teleop stage marking + video recording.

Usage:
  uv run python -m lerobot.rl.eval_dsrl --mode vanilla --n_episodes 10
  uv run python -m lerobot.rl.eval_dsrl --mode dsrl --dsrl_ckpt outputs/dsrl_hilserl_v5/checkpoints/000500/pretrained_model --n_episodes 10
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class VideoWriter:
    def __init__(self, path, fps=20):
        self.path = str(path)
        self.fps = fps
        self.writer = None

    @staticmethod
    def _to_frame(render_out):
        if isinstance(render_out, np.ndarray) and render_out.ndim == 3:
            return render_out
        if isinstance(render_out, dict):
            imgs = [v for v in render_out.values()
                    if isinstance(v, np.ndarray) and v.ndim == 3]
            if imgs:
                return np.concatenate(imgs, axis=1)
        return None

    def _write(self, frame):
        if frame is None:
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h),
            )
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame
        self.writer.write(bgr)

    def add_frame(self, env):
        """Frame from env.render() — render cameras, NOT policy input."""
        try:
            inner = env
            while hasattr(inner, "env"):
                inner = inner.env
            if hasattr(inner, "render"):
                self._write(self._to_frame(inner.render()))
        except Exception:
            pass

    @staticmethod
    def _obs_to_frame(observation):
        """Compose the policy's actual input images (observation.images.*) into a frame.

        Handles CHW float [0,1] or HWC uint8; stacks cameras side-by-side.
        """
        keys = sorted(k for k in observation if k.startswith("observation.images."))
        imgs = []
        for k in keys:
            v = observation[k]
            if hasattr(v, "detach"):
                v = v.detach().cpu().float().numpy()
            v = np.asarray(v)
            while v.ndim > 3:  # drop batch/temporal dims
                v = v[0]
            if v.ndim != 3:
                continue
            if v.shape[0] in (1, 3) and v.shape[0] < v.shape[-1]:  # CHW → HWC
                v = np.transpose(v, (1, 2, 0))
            if v.dtype != np.uint8:
                vmax = v.max() if v.size else 1.0
                v = (v * 255.0 if vmax <= 1.0 + 1e-6 else v).clip(0, 255).astype(np.uint8)
            if v.shape[2] == 1:
                v = np.repeat(v, 3, axis=2)
            imgs.append(v)
        if not imgs:
            return None
        h = max(i.shape[0] for i in imgs)
        imgs = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in imgs]
        return np.concatenate(imgs, axis=1)

    def add_obs_frame(self, observation):
        """Frame from the policy's actual observation images."""
        try:
            self._write(self._obs_to_frame(observation))
        except Exception:
            pass

    def close(self):
        if self.writer is not None:
            self.writer.release()


def parse_train_config(config_path: str):
    """Parse the training JSON config via draccus decode (no CLI arg pollution)."""
    import json as _json
    from lerobot.configs.parser import load_plugin
    load_plugin("lerobot_dsrl")
    import lerobot.teleoperators.gamepad.configuration_gamepad  # noqa: F401 — register 'gamepad'
    from draccus.parsers.decoding import decode
    from lerobot.configs.train import TrainRLServerPipelineConfig
    with open(config_path) as f:
        raw = _json.load(f)
    return decode(TrainRLServerPipelineConfig, raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "dsrl"], required=True)
    parser.add_argument("--dp_path", default="outputs/dp_v5_destale_t30/checkpoints/100000/pretrained_model")
    parser.add_argument("--dsrl_ckpt", default=None)
    parser.add_argument("--config", default="src/lerobot/rl/sim_dsrl_v5_hilserl_a6000_train.json")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--policy_view", action="store_true",
                        help="Record video from the policy's actual input images (observation.images.*) instead of env render cameras.")
    args = parser.parse_args()

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = args.mode
        if args.mode == "dsrl" and args.dsrl_ckpt:
            ckpt_name = Path(args.dsrl_ckpt).parent.name
            tag = f"dsrl_{ckpt_name}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"outputs/eval_dsrl/{tag}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(exist_ok=True)
    logging.info("Output dir: %s", out_dir)

    # Parse config (same path as actor/learner)
    cfg = parse_train_config(args.config)
    device = args.device
    fps = cfg.env.fps or 20
    input_features = set(cfg.policy.input_features.keys())

    # Build env + processors
    from lerobot.rl.gym_manipulator import make_robot_env, make_processors, step_env_and_process_transition
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.converters import create_transition
    from lerobot.teleoperators.utils import TeleopEvents

    env, teleop = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop, cfg.env, device)

    # Load policy
    if args.mode == "vanilla":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = DiffusionPolicy.from_pretrained(args.dp_path)
        policy.to(device).eval()
        pre, post = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=args.dp_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        logging.info("Loaded vanilla DP from %s", args.dp_path)

        _dp_img_size = tuple(next(iter(policy.config.image_features.values())).shape[1:])

        def get_action(obs):
            resized = {}
            for k, v in obs.items():
                if v.ndim >= 3 and v.shape[-2:] != _dp_img_size:
                    v = torch.nn.functional.interpolate(
                        v.unsqueeze(0).float() if v.ndim == 3 else v.float(),
                        size=_dp_img_size, mode="bilinear", align_corners=False,
                    )
                    if v.ndim == 4 and v.shape[0] == 1:
                        v = v.squeeze(0)
                resized[k] = v
            batch = pre(resized)
            action = policy.select_action(batch)
            if post is not None:
                action = post(action)
            return action

        def reset_policy():
            policy.reset()

    elif args.mode == "dsrl":
        assert args.dsrl_ckpt, "--dsrl_ckpt required for dsrl mode"
        import safetensors.torch
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors as _make_dp_pp
        from lerobot_dsrl.configuration_dsrl import DSRLConfig
        from lerobot_dsrl.modeling_dsrl import DSRLPolicy

        import dataclasses as _dc
        _dsrl_field_names = {f.name for f in _dc.fields(DSRLConfig)} - {"type"}
        dsrl_cfg = DSRLConfig(**{k: getattr(cfg.policy, k) for k in _dsrl_field_names
                                  if hasattr(cfg.policy, k)})
        dsrl_policy = DSRLPolicy(dsrl_cfg)

        sd = safetensors.torch.load_file(f"{args.dsrl_ckpt}/model.safetensors")
        for prefix, module in [
            ("noise_actor.", dsrl_policy.noise_actor),
            ("encoder.", dsrl_policy.encoder),
            ("critic.", dsrl_policy.critic),
            ("critic_target.", dsrl_policy.critic_target),
        ]:
            module.load_state_dict({k.removeprefix(prefix): v for k, v in sd.items() if k.startswith(prefix)})
        if "log_alpha" in sd:
            dsrl_policy.log_alpha.data.copy_(sd["log_alpha"])
        logging.info("Loaded DSRL checkpoint from %s", args.dsrl_ckpt)

        dp = DiffusionPolicy.from_pretrained(args.dp_path)
        dp.to(device)
        dp_pre, dp_post = _make_dp_pp(
            policy_cfg=dp.config, pretrained_path=args.dp_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        dsrl_policy.load_diffusion_policy(dp, preprocessor=dp_pre, postprocessor=dp_post)
        dsrl_policy.to(device).eval()
        logging.info("Attached frozen DP from %s", args.dp_path)

        def get_action(obs):
            return dsrl_policy.select_action(obs)

        def reset_policy():
            dsrl_policy.reset()

    # Eval loop
    results = []
    for ep in range(args.n_episodes):
        obs, info = env.reset()
        env_processor.reset()
        action_processor.reset()
        if teleop is not None and hasattr(teleop, "reset_episode_state"):
            teleop.reset_episode_state()
        reset_policy()

        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)

        vid = VideoWriter(vid_dir / f"ep{ep:02d}.mp4", fps=fps)

        ep_reward = 0.0
        ep_steps = 0
        ep_stages = 0
        ep_success = False
        stage_times = []
        done = False

        while not done:
            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in input_features
            }
            action = get_action(observation)

            new_transition = step_env_and_process_transition(
                env=env, transition=transition, action=action,
                env_processor=env_processor, action_processor=action_processor,
            )

            if args.policy_view:
                vid.add_obs_frame(observation)
            else:
                vid.add_frame(env)

            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)
            tr_info = new_transition.get(TransitionKey.INFO, {})

            ep_reward += float(reward)
            ep_steps += 1
            if tr_info.get(TeleopEvents.STAGE_ADVANCE, False):
                ep_stages += 1
                stage_times.append(ep_steps)
            if tr_info.get(TeleopEvents.SUCCESS, False):
                ep_success = True

            done = done or truncated
            transition = new_transition

            time.sleep(max(0, 1.0 / fps - 0.01))

        vid.close()

        results.append({
            "episode": ep, "reward": round(ep_reward, 1), "steps": ep_steps,
            "stages": ep_stages, "success": ep_success,
            "stage_times": stage_times,
        })
        logging.info(
            "[EVAL] ep=%d reward=%.1f steps=%d stages=%d success=%s stage_times=%s",
            ep, ep_reward, ep_steps, ep_stages, ep_success, stage_times,
        )

    # Summary
    n = len(results)
    succ = sum(r["success"] for r in results)
    avg_reward = sum(r["reward"] for r in results) / n
    avg_stages = sum(r["stages"] for r in results) / n
    avg_steps = sum(r["steps"] for r in results) / n

    summary = {
        "mode": args.mode,
        "checkpoint": args.dsrl_ckpt or args.dp_path,
        "n_episodes": n,
        "success_rate": f"{succ}/{n} ({100*succ/n:.0f}%)",
        "avg_reward": round(avg_reward, 1),
        "avg_stages": round(avg_stages, 1),
        "avg_steps": round(avg_steps, 0),
        "episodes": results,
    }

    report_path = out_dir / "results.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = out_dir / "results.txt"
    with open(txt_path, "w") as f:
        f.write(f"EVAL: {args.mode} | {args.dsrl_ckpt or args.dp_path}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Success rate: {succ}/{n} ({100*succ/n:.0f}%)\n")
        f.write(f"Avg reward:   {avg_reward:.1f}\n")
        f.write(f"Avg stages:   {avg_stages:.1f}\n")
        f.write(f"Avg steps:    {avg_steps:.0f}\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            status = "SUCCESS" if r["success"] else "FAIL"
            f.write(f"  ep{r['episode']:02d}: {status:7s} reward={r['reward']:7.1f} "
                    f"steps={r['steps']:4d} stages={r['stages']} "
                    f"stage_times={r['stage_times']}\n")

    print(f"\n{'='*60}")
    print(f"EVAL SUMMARY ({args.mode}, {args.dsrl_ckpt or args.dp_path})")
    print(f"  Episodes:     {n}")
    print(f"  Success rate: {succ}/{n} ({100*succ/n:.0f}%)")
    print(f"  Avg reward:   {avg_reward:.1f}")
    print(f"  Avg stages:   {avg_stages:.1f}")
    print(f"  Avg steps:    {avg_steps:.0f}")
    print(f"  Results:      {report_path}")
    print(f"  Videos:       {vid_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
