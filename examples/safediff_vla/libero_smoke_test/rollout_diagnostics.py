#!/usr/bin/env python
"""LIBERO / LIBERO-Plus smoke-test rollout + video-based verification.

Loads a checkpoint produced by `train_smoke.py`, rolls it out on the SAME single task
(`adapter.SMOKE_TEST_TASK_TEXT` -> suite/task_id) in the real `libero`/`libero_plus` gym env, and
dumps per-step-annotated contact-sheet frames (not just raw video) so the numeric action decoding
under test -- gripper polarity, xyz motion direction, rotation direction, camera routing -- can be
checked directly against what the arm visibly does, per the protocol.

For `--policy safediff_vla`, this also wires in the LIBERO camera-key override
(`adapter.EVAL_CAMERA_NAME_MAPPING`) and the gripper-collapse `ObservationProcessorStep`
(`adapter.GripperCollapseObservationStep`) so the eval-side env observation matches what the
checkpoint was trained on (see `adapter.py`'s docstrings for why each is needed).

This does NOT touch `find_grasp_target` or connect any GroundedGrasp architecture -- it only
rolls out the plain `temporal_decoder` ("TemporalDecoder-SinCos-PadFix") or SmolVLA checkpoints
`train_smoke.py` produces.

Usage:
    uv run python examples/safediff_vla/libero_smoke_test/rollout_diagnostics.py \
        --dataset libero --policy smolvla \
        --checkpoint outputs/train/libero_smoke_test/libero_smolvla_smoke/checkpoints/001500/pretrained_model \
        --n-episodes 3 --output-dir outputs/libero_smoke_test/rollouts/libero_smolvla
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from adapter import (
    EVAL_CAMERA_NAME_MAPPING,
    SMOKE_TEST_SUITE,
    SMOKE_TEST_SUITE_TASK_ID,
    SMOKE_TEST_TASK_TEXT,
    GripperCollapseObservationStep,
    LIBERO_PLUS_SMOKE_ACTION_SEMANTICS,
    LIBERO_SMOKE_ACTION_SEMANTICS,
)

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.configs import LiberoPlusEnv as LiberoPlusEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.safediff_vla.configuration_safediff_vla import SafeDiffVLAConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device

CONFIG_CLS = {"smolvla": SmolVLAConfig, "safediff_vla": SafeDiffVLAConfig}
SEMANTICS = {"libero": LIBERO_SMOKE_ACTION_SEMANTICS, "libero_plus": LIBERO_PLUS_SMOKE_ACTION_SEMANTICS}


def build_env_cfg(dataset: str):
    camera_name_mapping = EVAL_CAMERA_NAME_MAPPING[dataset]
    common = dict(
        task=SMOKE_TEST_SUITE,
        task_ids=[SMOKE_TEST_SUITE_TASK_ID],
        obs_type="pixels_agent_pos",
        observation_height=256,
        observation_width=256,
        init_states=True,
        hard_reset=True,
        control_mode="relative",
        camera_name_mapping=camera_name_mapping,
    )
    if dataset == "libero":
        return LiberoEnvConfig(fps=10, **common)
    return LiberoPlusEnvConfig(fps=20, **common)


def annotate_frame(image: np.ndarray, lines: list[str]) -> Image.Image:
    img = Image.fromarray(image).convert("RGB")
    if img.width < 256:
        img = img.resize((256, 256), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + 14 * len(lines) + 6), "black")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((2, img.height + 2 + 14 * i), line, fill="yellow")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--policy", choices=("smolvla", "safediff_vla"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--n-frames-per-episode", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    semantics = SEMANTICS[args.dataset]

    env_cfg = build_env_cfg(args.dataset)
    envs = make_env(env_cfg, n_envs=args.n_episodes)
    vec_env = envs[SMOKE_TEST_SUITE][SMOKE_TEST_SUITE_TASK_ID]

    policy_cfg = CONFIG_CLS[args.policy].from_pretrained(args.checkpoint)
    device = get_safe_torch_device(policy_cfg.device)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    if args.policy == "safediff_vla":
        env_preprocessor.steps = list(env_preprocessor.steps) + [GripperCollapseObservationStep()]

    print(
        f"[rollout] dataset={args.dataset} policy={args.policy} task={SMOKE_TEST_TASK_TEXT!r} "
        f"suite={SMOKE_TEST_SUITE} task_id={SMOKE_TEST_SUITE_TASK_ID} n_episodes={args.n_episodes}"
    )

    # `return_observations=True` crashes today (rollout() tries to torch.stack a non-tensor obs
    # key across the time axis) -- sidestep it entirely and capture frames the same way
    # `lerobot_eval.py`'s own `eval_policy`/`render_frame` does: a `render_callback` invoked after
    # reset and after every step. `LiberoEnv.render()` already flips the image for visualization
    # (`image[::-1, ::-1]`) -- keep that as-is, do not flip again.
    rendered_frames: list[np.ndarray] = []

    def render_frame(env) -> None:
        import gymnasium as gym

        if isinstance(env, gym.vector.SyncVectorEnv):
            batch = np.stack([env.envs[i].render() for i in range(env.num_envs)])
        else:
            batch = np.stack(env.call("render")[: env.num_envs])
        rendered_frames.append(batch)

    with torch.no_grad():
        rollout_data = rollout(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(range(args.n_episodes)),
            return_observations=False,
            render_callback=render_frame,
        )

    action = rollout_data[ACTION].cpu().numpy()  # [B, T, 7]
    done = rollout_data["done"].cpu().numpy()  # [B, T] cumulative
    success = rollout_data["success"].cpu().numpy()  # [B, T]
    frames = np.stack(rendered_frames, axis=1)  # [B, T+1, H, W, C] (index 0 = post-reset frame)

    log_path = out_dir / "step_log.txt"
    with open(log_path, "w") as log:
        log.write(f"task={SMOKE_TEST_TASK_TEXT!r} suite={SMOKE_TEST_SUITE} task_id={SMOKE_TEST_SUITE_TASK_ID}\n")
        log.write(
            f"candidate semantics: translation_scale={semantics.translation_scale_candidate} "
            f"rotation_scale={semantics.rotation_scale_candidate} "
            f"rotation_composition={semantics.rotation_composition} "
            f"gripper_open_is_negative_candidate={semantics.gripper_open_is_negative_candidate} "
            f"(confidence={semantics.gripper_polarity_confidence})\n\n"
        )

        for b in range(action.shape[0]):
            ep_len = int(np.argmax(done[b])) + 1 if done[b].any() else action.shape[1]
            ep_success = bool(success[b, :ep_len].any())
            log.write(f"=== episode {b} (len={ep_len}, success={ep_success}) ===\n")
            frame_idxs = sorted(set(np.linspace(0, ep_len - 1, args.n_frames_per_episode).astype(int).tolist()))
            for t in range(ep_len):
                a = action[b, t]
                grip = a[6]
                predicted_state = "CLOSE" if (grip > 0) == (not semantics.gripper_open_is_negative_candidate) else "OPEN"
                d_xyz = a[0:3] * np.asarray(semantics.translation_scale_candidate)
                line = (
                    f"t={t:03d} action_xyz={a[0:3].round(3).tolist()} pred_dxyz_m={d_xyz.round(4).tolist()} "
                    f"action_rot={a[3:6].round(3).tolist()} action_grip={grip:+.3f} "
                    f"candidate_gripper={predicted_state}"
                )
                if t in frame_idxs:
                    log.write(line + "  <-- FRAME SAVED\n")
                else:
                    log.write(line + "\n")

                if t in frame_idxs:
                    # frames[b, t] is the render AFTER t previous steps (t=0 == post-reset, before
                    # any action) -- pair it with the action about to be taken at this step.
                    img = frames[b, t]
                    annotated = annotate_frame(
                        img,
                        [
                            f"ep{b} t={t} render (post-reset if t=0)",
                            f"a_xyz={a[0:3].round(2).tolist()} pred_d={d_xyz.round(3).tolist()}m",
                            f"a_rot={a[3:6].round(2).tolist()} a_grip={grip:+.2f} -> {predicted_state}",
                        ],
                    )
                    fname = f"ep{b}_t{t:03d}_render.png"
                    annotated.save(out_dir / fname)
            log.write(f"episode {b} final success={ep_success}\n\n")

    print(f"[rollout] wrote step log + frames to {out_dir}")
    print(f"[rollout] per-episode success: {[bool(success[b].any()) for b in range(action.shape[0])]}")


if __name__ == "__main__":
    main()
