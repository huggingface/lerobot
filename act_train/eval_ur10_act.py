"""Closed-loop inference of a trained ACT policy on the UR10.

Mirrors ``act_train/act_using_example.py`` but routes through ``UR10RobotEnv``
(``make_robot_env`` + ``make_processors`` from ``gym_manipulator``) instead of
the direct-robot-`send_action` pattern. UR10 doesn't have ``send_action``; the
env's ``step()`` takes the 4-D ``[dx, dy, dz, gripper]`` action and accumulates
it into the streaming target.

The gamepad stays connected during eval as a safety override (Triangle / Cross
to end an episode if the policy misbehaves; intervention substitutes the policy
action with the user's gamepad input).

Usage:
    python act_train/eval_ur10_act.py
"""

from __future__ import annotations

import json
import logging
import time

import draccus
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import ur10 as _ur10_register  # noqa: F401
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.robot_utils import precise_sleep
from lerobot.teleoperators.utils import TeleopEvents

from _ur10_reset import auto_reset_to_home  # sibling module in act_train/

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable ---------------------------------------------------------------
MODEL_DIR = "outputs/act/ur10/usb_insertion_state/last"
DATASET_REPO_ID = "local/ur10_act_usb_insertion_state"   # for dataset stats
CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams.json"
NUM_EPISODES = 20
EPISODE_TIME_S = 25      # safety upper bound; user ends earlier via gamepad
RESET_TIME_S = 7         # total between-episode budget (motion + hold-at-home)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s (matches env.reset's moveL)
FPS = 10
# -------------------------------------------------------------------------------


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    dt = 1.0 / FPS

    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    policy: ACTPolicy = ACTPolicy.from_pretrained(MODEL_DIR)
    policy.eval()
    device = torch.device(policy.config.device)
    policy.to(device)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=MODEL_DIR, dataset_stats=metadata.stats
    )
    logger.info("Policy loaded from %s on %s", MODEL_DIR, device)

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(
        env, teleop_device, cfg.env, str(device)
    )

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    episode_idx = 0
    episode_step = 0
    episode_reward = 0.0
    episode_start = time.perf_counter()
    logger.info(
        "Inference at %d Hz for %d episodes. Triangle / Cross = success / fail; "
        "Ctrl+C to exit.",
        FPS, NUM_EPISODES,
    )
    logger.info("--- Episode %d ---", episode_idx + 1)

    try:
        while episode_idx < NUM_EPISODES:
            t0 = time.perf_counter()

            # Build the policy input: only the features ACT was trained on. Already
            # on device + batched because env_processor ran DeviceMove + AddBatchDim.
            obs_batch = {
                k: v
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in policy.config.input_features
            }

            # Normalize → predict → unnormalize.
            obs_batch = preprocess(obs_batch)
            with torch.no_grad():
                action = policy.select_action(batch=obs_batch)  # (1, action_dim)
            action = postprocess(action)
            if action.ndim > 1:
                action = action.squeeze(0)

            # Translate policy's gripper STATE prediction → env.step's COMMAND encoding.
            # Dataset action[3] is in {0.0=closed, 1.0=open} (state encoding from the
            # translation script); after unnormalize the policy outputs ~[0, 1]. We
            # compare against the gripper's currently-commanded state (read from
            # observation.state[-1] — ur10_robot.py:472 keeps that flag in sync with
            # send_gripper) and emit a transition command only when the desired state
            # differs. UR10RobotEnv.step interprets {0=CLOSE, 1=STAY, 2=OPEN}.
            if action.numel() >= 4:
                a = action.detach().cpu().float().numpy()
                obs_state = transition[TransitionKey.OBSERVATION][OBS_STATE]
                if isinstance(obs_state, torch.Tensor):
                    obs_state = obs_state.detach().cpu().float().numpy()
                current_state = float(np.asarray(obs_state).reshape(-1)[-1])  # last dim
                predicted_state = 1.0 if float(a[3]) > 0.5 else 0.0
                if predicted_state == current_state:
                    cmd_int, cmd_name = 1, "STAY"
                elif predicted_state > current_state:
                    cmd_int, cmd_name = 2, "OPEN"   # closed → open
                else:
                    cmd_int, cmd_name = 0, "CLOSE"  # open → closed
                # Rewrite action[3] to the discrete command env.step expects.
                action = torch.tensor(
                    [float(a[0]), float(a[1]), float(a[2]), float(cmd_int)],
                    device=action.device, dtype=action.dtype,
                )
                logger.info(
                    "  step %d  action=[dx=%+.3f dy=%+.3f dz=%+.3f g_raw=%+.3f "
                    "→ desired=%s, current=%s] → gripper=%s",
                    episode_step, float(a[0]), float(a[1]), float(a[2]), float(a[3]),
                    "OPEN" if predicted_state == 1.0 else "CLOSED",
                    "OPEN" if current_state == 1.0 else "CLOSED",
                    cmd_name,
                )

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            reward = float(transition.get(TransitionKey.REWARD, 0.0))
            done = bool(transition.get(TransitionKey.DONE, False))
            truncated = bool(transition.get(TransitionKey.TRUNCATED, False))
            episode_reward += reward
            episode_step += 1

            # Client-side episode bound — fires before the env's TimeLimit so eval
            # cycles aren't hostage to control_time_s in the JSON (which is sized
            # for training, not bench-eval).
            if not done and not truncated and episode_step >= int(EPISODE_TIME_S * FPS):
                truncated = True

            if episode_step % 10 == 0:
                logger.info("  step %d  reward=%.2f", episode_step, episode_reward)

            if done or truncated:
                ep_time = time.perf_counter() - episode_start
                status = "SUCCESS" if episode_reward > 0 else "DONE"
                logger.info(
                    "Episode %d %s: reward=%.2f steps=%d time=%.1fs done=%s trunc=%s",
                    episode_idx + 1, status, episode_reward, episode_step,
                    ep_time, done, truncated,
                )
                episode_idx += 1
                if episode_idx >= NUM_EPISODES:
                    break

                # Auto-drive home without env.reset() → never calls servoStop. The
                # streaming thread keeps tracking targets at 200 Hz the whole time.
                # Reset budget splits into motion (at RESET_SPEED_MPS) + hold-at-home;
                # use the hold window to reposition the test object between trials.
                auto_reset_to_home(env, dt, RESET_TIME_S, RESET_SPEED_MPS, FPS)
                env_processor.reset()
                action_processor.reset()
                policy.reset()  # clear ACT chunk queue + temporal ensembler.
                transition = env_processor(create_transition(
                    observation=env.robot.get_observation(),
                    info={TeleopEvents.IS_INTERVENTION: False},
                ))
                episode_step = 0
                episode_reward = 0.0
                episode_start = time.perf_counter()
                logger.info("--- Episode %d ---", episode_idx + 1)

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        logger.info("Inference stopped by user.")
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
