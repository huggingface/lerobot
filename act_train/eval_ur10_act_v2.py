"""Closed-loop inference of a trained ACT policy on the UR10 — v2 (RC10-style).

Differences from ``eval_ur10_act.py``
=====================================
v1 evaluated a delta-action policy: the policy output a `[dx, dy, dz, dyaw,
gripper_cmd]` vector that the env's `step()` accumulated into the streaming
target. That layout collapsed at fine-alignment phases (see
``act_train/audit_dataset.py`` for the forensic trace).

v2 evaluates an absolute-target policy: the policy outputs a `[x.pos, y.pos,
z.pos, yaw.pos, gripper.pos]` dict (matching the v2 dataset's action schema),
which is fed directly to ``env.set_act_target(...)`` — no delta accumulation,
no action gain, no gripper-state-to-command translation in the script (all of
that logic lives inside ``set_act_target``).

The loop body is therefore dramatically simpler — it's basically:

    obs   = env.get_act_observation()
    batch = preprocess(build_inference_frame(obs, …))
    act   = postprocess(policy.select_action(batch))
    env.set_act_target(make_robot_action(act, dataset_features))

…mirroring ``act_train/act_using_example.py`` (RC10's working eval script)
beat-for-beat.

Usage
=====
    python act_train/eval_ur10_act_v2.py
"""

from __future__ import annotations

import json
import logging
import time

import draccus
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import ur10 as _ur10_register  # noqa: F401
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_processors, make_robot_env
from lerobot.utils.constants import OBS_STATE
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from _ur10_reset import auto_reset_to_home  # sibling module in act_train/

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable ---------------------------------------------------------------
MODEL_DIR = "outputs/act/ur10/pcb_act_3cams_yaw_v2_60eps_30eps2/step_10000"
DATASET_REPO_ID = "local/pcb_act_3cams_yaw_v2_60eps"   # for dataset stats (normalization)
CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams_yaw_v2.json"
NUM_EPISODES = 30
EPISODE_TIME_S = 30      # safety upper bound; user ends earlier via gamepad
RESET_TIME_S = 10        # total between-episode budget (motion + hold-at-home)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s (matches env.reset's moveL)
FPS = 30
# -------------------------------------------------------------------------------


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    dt = 1.0 / FPS

    # ---- policy + processors ------------------------------------------------
    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    policy: ACTPolicy = ACTPolicy.from_pretrained(MODEL_DIR)
    policy.eval()
    device = torch.device(policy.config.device)
    policy.to(device)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=MODEL_DIR, dataset_stats=metadata.stats
    )
    logger.info("Policy loaded from %s on %s", MODEL_DIR, device)
    logger.info(
        "ACT config: chunk_size=%d, n_action_steps=%d, temporal_ensemble_coeff=%s",
        policy.config.chunk_size,
        policy.config.n_action_steps,
        policy.config.temporal_ensemble_coeff,
    )

    # ---- env + processors --------------------------------------------------
    # We DO use env_processor here so the policy sees the same cropped/resized
    # images the training dataset was built with (the cropping config lives in
    # cfg.processor.image_preprocessing — applied by ImageCropResizeProcessorStep).
    # We don't use the returned `obs` directly; instead we override the state
    # slot with the v2 11-D ACT state from env.get_act_observation() before
    # passing to the policy. This keeps train/eval frame-for-frame consistent.
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, _action_processor = make_processors(env, teleop_device, cfg.env, str(device))
    obs, _info = env.reset()
    env_processor.reset()


    def _build_obs_for_policy() -> dict:
        """Pull a single observation through env_processor (cropped images +
        17-D state), then swap the state for the v2 11-D ACT state. Returns
        the obs dict ready to feed into ``policy.select_action`` after
        normalization."""
        raw = env._augment_observation(env.robot.get_observation())
        tr = env_processor(create_transition(
            observation=raw, info={TeleopEvents.IS_INTERVENTION: False},
        ))
        v2_state = env.get_act_observation()["agent_pos"]
        # env_processor's AddBatchDimensionProcessorStep emits (1, D); match it.
        tr[TransitionKey.OBSERVATION][OBS_STATE] = (
            torch.from_numpy(v2_state.copy()).unsqueeze(0).to(device).float()
        )
        return {
            k: v for k, v in tr[TransitionKey.OBSERVATION].items()
            if k in policy.config.input_features
        }

    episode_idx = 0
    episode_step = 0
    episode_start = time.perf_counter()
    logger.info(
        "Inference at %d Hz for %d episodes. Triangle/Cross = success/fail; Ctrl+C to exit.",
        FPS, NUM_EPISODES,
    )
    logger.info("--- Episode %d ---", episode_idx + 1)

    try:
        while episode_idx < NUM_EPISODES:
            t0 = time.perf_counter()

            # 1. Pull the observation through env_processor (cropped/resized
            #    images) and swap the state for the v2 11-D ACT state — see
            #    _build_obs_for_policy above for the rationale. Already batched
            #    on the right device.
            obs_batch = _build_obs_for_policy()

            # 2. Normalize, predict, unnormalize.
            obs_batch = preprocess(obs_batch)
            with torch.no_grad():
                action_tensor = policy.select_action(obs_batch)
            action_tensor = postprocess(action_tensor)

            # 4. Convert the policy's output tensor into a named dict matching the
            #    dataset's action schema (`{x.pos, y.pos, z.pos, yaw.pos, gripper.pos}`).
            action_dict = make_robot_action(action_tensor, metadata.features)

            # 5. Drive the robot directly with the absolute target. All bounds /
            #    rotation / gripper-translation logic lives inside set_act_target,
            #    so this script stays minimal and intent-revealing.
            env.set_act_target(action_dict)

            episode_step += 1
            if episode_step % 10 == 0:
                logger.info(
                    "  step %d  target=[x=%+.4f y=%+.4f z=%+.4f yaw=%+.4f g=%.2f]",
                    episode_step,
                    float(action_dict.get("x.pos", 0.0)),
                    float(action_dict.get("y.pos", 0.0)),
                    float(action_dict.get("z.pos", 0.0)),
                    float(action_dict.get("yaw.pos", 0.0)),
                    float(action_dict.get("gripper.pos", 0.0)),
                )

            # Episode termination: client-side time limit (the env's TimeLimit
            # processor isn't in this loop because we're not going through
            # env_processor at all in the v2 eval). Gamepad-driven success/fail
            # via teleop events still works because make_robot_env connects the
            # gamepad — we read its events here.
            truncated = episode_step >= int(EPISODE_TIME_S * FPS)
            success = False
            terminate = False
            if teleop_device is not None:
                events = teleop_device.get_teleop_events()
                success = bool(events.get(TeleopEvents.SUCCESS, False))
                terminate = bool(events.get(TeleopEvents.TERMINATE_EPISODE, False))
            done = success or terminate

            if done or truncated:
                ep_time = time.perf_counter() - episode_start
                status = "SUCCESS" if success else ("TERMINATED" if terminate else "TIMEOUT")
                logger.info(
                    "Episode %d %s after %d steps (%.1fs)",
                    episode_idx + 1, status, episode_step, ep_time,
                )
                episode_idx += 1
                if episode_idx >= NUM_EPISODES:
                    break

                # Between episodes: bypass servoStop via interpolated set_target_pose.
                # phase-1.5 fix inside auto_reset_to_home zeroes env.target_yaw, so the
                # next episode's first commanded yaw is 0 (no stale-state snap).
                auto_reset_to_home(env, dt, RESET_TIME_S, RESET_SPEED_MPS, FPS)
                env_processor.reset()  # drop any per-episode state inside the image pipeline
                policy.reset()  # clear ACT chunk queue + temporal ensembler if any
                episode_step = 0
                episode_start = time.perf_counter()
                logger.info("--- Episode %d ---", episode_idx + 1)

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        logger.info("Inference stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Inference failed")
    finally:
        logger.info("Completed %d episodes", episode_idx)
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleop disconnect failed")


if __name__ == "__main__":
    main()
