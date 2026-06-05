#!/usr/bin/env python
"""Evaluate ONE LIBERO task with the remote vllm policy and visualize it live in Rerun.

Reuses the exact same ``VllmPolicy`` (and ``examples/vllm_policy/`` config) that
``lerobot-eval`` uses, so the rollout matches a real eval episode — but for a single task,
with per-step camera/state/action streaming to Rerun using **lerobot's own Rerun helpers**
(no dependency on ``libero.libero.utils.rerun_utils``).

Prereq: the vLLM OpenPI server is serving the matching suite checkpoint on 127.0.0.1:8000.

Usage (run from the lerobot repo root):
    # default: serve a Rerun web viewer
    python examples/vllm_policy/scripts/eval_visualize.py --benchmark libero_object --task-id 0 --open-browser
    # save a recording instead
    python examples/vllm_policy/scripts/eval_visualize.py --benchmark libero_object --task-id 0 --save /tmp/run.rrd --no-keep-alive
    # native desktop viewer
    python examples/vllm_policy/scripts/eval_visualize.py --benchmark libero_object --task-id 0 --spawn
"""

from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np
import torch

# LIBERO neutralizes EGL on macOS at import; keep this before robosuite use.
from libero.libero import benchmark as libero_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

import lerobot.policies  # noqa: E402,F401  (register built-in policies)
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES  # noqa: E402
from lerobot.utils.visualization_utils import log_rerun_data  # noqa: E402

from lerobot.policies.vllm.modeling_vllm import VllmPolicy  # noqa: E402

LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

log = logging.getLogger("eval-visualize")


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite (x,y,z,w) quaternion -> axis-angle (matches Isaac-GR00T libero_env)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if den < 1e-8:
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_batch(raw_obs: dict, task_lang: str, device: str) -> dict:
    """Raw LIBERO obs -> lerobot policy batch (mirrors preprocess + LiberoProcessorStep)."""
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    axisangle = quat2axisangle(raw_obs["robot0_eef_quat"]).astype(np.float32)
    gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    state = np.concatenate([eef_pos, axisangle, gripper]).astype(np.float32)  # (8,)

    def to_chw(name: str) -> torch.Tensor:
        im = np.asarray(raw_obs[name])[::-1, ::-1]  # 180° flip (== LiberoProcessorStep)
        t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device)

    return {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
        "observation.images.image": to_chw("agentview_image"),
        "observation.images.image2": to_chw("robot0_eye_in_hand_image"),
        "task": [task_lang],
    }


def _init_rerun(args) -> str | None:
    """Initialize Rerun in one of four modes; return the web URL if serving, else None.

    Modes (mutually exclusive in priority order):
      --save PATH  → write a .rrd recording (no live viewer)
      --spawn      → native desktop Rerun viewer (`rr.spawn`)
      default      → serve gRPC sink + embeddable web viewer
    """
    import rerun as rr

    rr.init("lerobot_vllm_libero_eval")

    if args.save:
        rr.save(args.save)
        return None
    if args.spawn:
        rr.spawn()
        return None
    server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
    rr.serve_web_viewer(open_browser=args.open_browser, web_port=args.web_port, connect_to=server_uri)
    web_url = f"http://localhost:{args.web_port}"
    log.info("Rerun web viewer: %s", web_url)
    log.info("gRPC sink:         %s", server_uri)
    return web_url


def _log_step(t: int, raw_obs: dict, action: np.ndarray, reward: float, done: bool, success: bool) -> None:
    """Log one rollout step using lerobot's Rerun helpers + temporal image logging.

    `log_rerun_data` logs images with `static=True`, which strips the timeline — fine for a
    one-shot dataset preview but wrong for rollout playback. We log images ourselves with
    `rr.set_time` set first so they advance with the step counter.
    """
    import rerun as rr

    rr.set_time("step", sequence=int(t))

    # Temporal image logs (use lerobot's canonical OBS_IMAGES path).
    for cam_key, raw_key in (("image", "agentview_image"), ("image2", "robot0_eye_in_hand_image")):
        img = np.asarray(raw_obs[raw_key])[::-1, ::-1]  # match LiberoProcessorStep flip
        rr.log(f"{OBS_IMAGES}.{cam_key}", rr.Image(img))

    # Scalars / state / action via lerobot's standard helper (auto-namespaces observation.* / action.*).
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    axisangle = quat2axisangle(raw_obs["robot0_eef_quat"]).astype(np.float32)
    gripper = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[:2]
    obs_log = {
        "state": np.concatenate([eef_pos, axisangle, gripper]).astype(np.float32),
        "success": float(bool(success)),
    }
    # Name action components explicitly so the viewer shows action.dx ... action.gripper
    # rather than action.action_0 ... action.action_6.
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    action_log = {
        "dx": float(a[0]), "dy": float(a[1]), "dz": float(a[2]),
        "droll": float(a[3]), "dpitch": float(a[4]), "dyaw": float(a[5]),
        "gripper": float(a[6]),
    }
    log_rerun_data(observation=obs_log, action=action_log)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-id", type=int, default=0)
    p.add_argument("--policy-path", default="examples/vllm_policy")
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-steps-wait", type=int, default=10, help="settle steps after reset")
    p.add_argument("--render-size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    # Rerun
    p.add_argument("--grpc-port", type=int, default=9876)
    p.add_argument("--web-port", type=int, default=9090)
    p.add_argument("--open-browser", action="store_true")
    p.add_argument("--spawn", action="store_true", help="native desktop viewer instead of web")
    p.add_argument("--save", default=None, help="write a .rrd recording to this path instead of serving")
    p.add_argument("--no-keep-alive", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    suite = libero_benchmark.get_benchmark_dict()[args.benchmark]()
    task = suite.get_task(args.task_id)
    log.info("[%s][%d] %s", args.benchmark, args.task_id, task.language)

    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_heights=args.render_size,
        camera_widths=args.render_size,
    )

    # Policy (same config dir as lerobot-eval).
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.n_action_steps = args.n_action_steps
    cfg.device = args.device
    policy = VllmPolicy(cfg)
    policy.eval()

    web_url = _init_rerun(args)

    # Reset + set the bundled init state, then settle the scene.
    env.reset()
    init_states = suite.get_task_init_states(args.task_id)
    obs = env.set_init_state(init_states[args.init_id % len(init_states)])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    policy.reset()
    _log_step(0, obs, np.zeros(7, dtype=np.float32), reward=0.0, done=False, success=False)

    success = False
    for t in range(args.max_steps):
        batch = build_batch(obs, task.language, args.device)
        with torch.inference_mode():
            action = policy.select_action(batch)[0].cpu().numpy().astype(np.float32)
        obs, reward, done, info = env.step(action)
        success = bool(env.check_success())
        _log_step(t + 1, obs, action, float(reward), bool(done), success)
        if success:
            log.info("SUCCESS at step %d", t + 1)
            break
    else:
        log.info("finished %d steps without success", args.max_steps)

    env.close()

    if args.save:
        log.info("saved recording to %s (open with: rerun %s)", args.save, args.save)
        return 0
    if args.spawn:
        # Native viewer is its own process; just hand control back.
        return 0
    if args.no_keep_alive:
        return 0
    # Keep the web viewer up so the user can scrub the timeline.
    log.info("[rerun] serving %s — press Ctrl-C to stop", web_url)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
