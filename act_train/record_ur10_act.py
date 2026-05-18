"""Record UR10 demonstrations for offline ACT training.

Drives the arm via gamepad (configured in ``ur10_env_3cams.json``) and writes
each episode to a ``LeRobotDataset``. Re-uses the existing UR10 env layer
(`make_robot_env`, `make_processors` from `gym_manipulator`) so we inherit:
  - the 3-camera RealSense hardware-reset preflight,
  - cropped + resized 128x128 images,
  - the 16-D relative-tcp_xyz observation,
  - the streaming-thread control + reset-recovery path.

The recorded action is the 4-D vector ``[dx, dy, dz, gripper_state]``. The
Cartesian deltas come straight from the teleop, but the gripper component is
recorded as the integrated STATE (``0.0=closed, 1.0=open``, read from
``observation.state[-1]`` after each env.step) -- NOT the raw teleop COMMAND
encoding (``{0=CLOSE, 1=STAY, 2=OPEN}``). State encoding matches RC10's ACT
dataset convention and avoids the 96%-STAY collapse that pure regression on
the command labels suffers from.

Usage:
    python act_train/record_ur10_act.py

Tune the constants at the top of the file. End an episode early via the gamepad
(Triangle = success, Cross = fail); Square / Circle (or whatever the teleop
maps to RERECORD_EPISODE) re-records the current episode.
"""

from __future__ import annotations

import json
import logging
import time

import draccus
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.robots import ur10 as _ur10_register  # noqa: F401  # registers UR10RobotConfig
from lerobot.rl.gym_manipulator import (
    GymManipulatorConfig,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from _ur10_reset import auto_reset_to_home  # sibling module in act_train/

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable ---------------------------------------------------------------
CONFIG_PATH = "src/lerobot/rl/ur10_env_2cams.json"
REPO_ID = "local/usb_insertion_act_2cams"
TASK_DESCRIPTION = "usb_insertion_act_2cams"
NUM_EPISODES = 70
EPISODE_TIME_S = 20      # truncates the episode if the user doesn't end it
RESET_TIME_S = 7         # total between-episode budget (motion + hold-at-home)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s (matches env.reset's moveL)
FPS = 10                 # must equal cfg.env.fps
DEVICE = "cpu"           # env-processor lives on CPU; only the policy needs GPU

# UX feedback (same as RC10 record scripts):
USE_TTS    = True        # log_say() TTS announcements ("recording episode 1 of 50", "reset", ...).
USE_RERUN  = True        # init_rerun() spawns the Rerun viewer + log_rerun_data each step.
RERUN_EVERY_N_STEPS = 1  # set >1 to throttle viewer updates if bandwidth is tight.
# -------------------------------------------------------------------------------


def _build_features(transition_obs: dict, use_gripper: bool) -> dict:
    """Build the LeRobotDataset feature dict from an actual processed observation.

    Shapes are read from the live transition so they match exactly what ACT will
    see at inference time (after crop/resize + AddBatchDim + DeviceMove).
    """
    action_dim = 4 if use_gripper else 3
    features: dict[str, dict] = {
        ACTION: {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": (
                ["delta_x", "delta_y", "delta_z", "gripper"]
                if use_gripper
                else ["delta_x", "delta_y", "delta_z"]
            ),
        },
    }
    for key, val in transition_obs.items():
        if "image" in key and isinstance(val, torch.Tensor):
            shape = tuple(val.squeeze(0).shape)  # drop batch dim → (C, H, W)
            features[key] = {
                "dtype": "video",
                "shape": shape,
                "names": ["channels", "height", "width"],
            }
        elif key == OBS_STATE and isinstance(val, torch.Tensor):
            features[key] = {
                "dtype": "float32",
                "shape": tuple(val.squeeze(0).shape),
                "names": None,
            }
    return features


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    dt = 1.0 / FPS

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, DEVICE)

    use_gripper = (
        cfg.env.processor.gripper.use_gripper
        if cfg.env.processor.gripper is not None
        else True
    )
    action_dim = 4 if use_gripper else 3
    neutral_action = torch.zeros(action_dim, dtype=torch.float32)
    if use_gripper:
        neutral_action[3] = 1.0  # "stay"

    # First reset and one processed observation so we can size the dataset features.
    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    features = _build_features(transition[TransitionKey.OBSERVATION], use_gripper)
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        robot_type="ur10",
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
    )
    logger.info(
        "Recording %d episodes (≤%.0f s each) at %d Hz into %s",
        NUM_EPISODES, EPISODE_TIME_S, FPS, REPO_ID,
    )
    logger.info("Hold the gamepad intervention button to drive; Triangle / Cross to end episode.")

    if USE_RERUN:
        init_rerun(session_name=f"ur10_record_{TASK_DESCRIPTION}")

    if USE_TTS:
        log_say(f"Recording episode 1 of {NUM_EPISODES}")

    max_episode_steps = int(EPISODE_TIME_S * FPS)
    episode_idx = 0
    episode_step = 0
    global_step = 0
    episode_start = time.perf_counter()

    try:
        while episode_idx < NUM_EPISODES:
            step_start = time.perf_counter()

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            terminated = bool(transition.get(TransitionKey.DONE, False))
            truncated = bool(transition.get(TransitionKey.TRUNCATED, False))
            if not terminated and not truncated and episode_step + 1 >= max_episode_steps:
                truncated = True

            # The action stored in transition is the post-intervention action (i.e. the
            # gamepad's input substituted in via InterventionActionProcessorStep). That
            # is the action that *actually drove the robot* this step.
            comp = transition[TransitionKey.COMPLEMENTARY_DATA]
            action_to_record = comp.get("teleop_action", transition[TransitionKey.ACTION])
            if isinstance(action_to_record, torch.Tensor):
                action_to_record = action_to_record.detach().cpu().float()
            else:
                action_to_record = torch.tensor(action_to_record, dtype=torch.float32)
            if action_to_record.ndim > 1:
                action_to_record = action_to_record.squeeze(0)

            # Rewrite the gripper component from COMMAND encoding ({0, 1, 2}, where ~96%
            # of frames are STAY=1) to STATE encoding ({0.0=closed, 1.0=open}, naturally
            # balanced across an episode). Source of truth is `observation.state[-1]` on
            # the post-step transition: ur10_robot.py:472 reads `float(self.gripper.is_open)`
            # fresh on every get_observation, and env.step() has just called send_gripper(),
            # so the flag reflects this step's resulting gripper state. Matches RC10's
            # action-encoding convention and the schema train/eval expect downstream.
            if action_to_record.numel() >= 4:
                obs_state = transition[TransitionKey.OBSERVATION][OBS_STATE]
                if isinstance(obs_state, torch.Tensor):
                    obs_state = obs_state.detach().cpu().float()
                gripper_state = float(np.asarray(obs_state).reshape(-1)[-1])
                action_to_record[3] = gripper_state

            # Pull observation tensors (drop batch dim — dataset expects (C, H, W) / (D,)).
            frame: dict = {ACTION: action_to_record, "task": TASK_DESCRIPTION}
            for k, v in transition[TransitionKey.OBSERVATION].items():
                if isinstance(v, torch.Tensor) and k in features:
                    frame[k] = v.squeeze(0).detach().cpu()
            dataset.add_frame(frame)
            episode_step += 1
            global_step += 1

            if USE_RERUN and (episode_step % RERUN_EVERY_N_STEPS == 0):
                # Match RC10 ACT's working pattern (record_loop, lerobot_record.py:416-419):
                # no rr.set_time calls, compress_images=False. Custom timelines confuse
                # Rerun 0.26's TimeSeriesView auto-pick and result in graphs that don't
                # render lines. log_rerun_data already handles the X axis via the implicit
                # log_time / log_tick timelines.
                rr_obs = {
                    k: v.numpy() for k, v in frame.items()
                    if k not in (ACTION, "task") and isinstance(v, torch.Tensor)
                }
                rr_action = {ACTION: action_to_record.numpy()}
                log_rerun_data(observation=rr_obs, action=rr_action, compress_images=False)

            if episode_step % 10 == 0:
                logger.info("  step %d", episode_step)

            if terminated or truncated:
                ep_time = time.perf_counter() - episode_start
                rerecord = transition[TransitionKey.INFO].get(
                    TeleopEvents.RERECORD_EPISODE, False
                )
                if rerecord:
                    logger.info(
                        "Re-recording episode %d (%.1fs, %d steps)",
                        episode_idx + 1, ep_time, episode_step,
                    )
                    dataset.clear_episode_buffer()
                    if USE_TTS:
                        log_say(f"Re-recording episode {episode_idx + 1}")
                else:
                    logger.info(
                        "Saving episode %d (%.1fs, %d steps)",
                        episode_idx + 1, ep_time, episode_step,
                    )
                    dataset.save_episode()
                    episode_idx += 1
                    if USE_TTS:
                        log_say(f"Saved episode {episode_idx}")

                if episode_idx >= NUM_EPISODES:
                    break

                if USE_TTS:
                    log_say("Reset the environment")
                # Auto-drive home without env.reset() → never calls servoStop. The
                # streaming thread keeps tracking targets at 200 Hz the whole time.
                # Reset budget splits into motion (at RESET_SPEED_MPS) + hold-at-home.
                auto_reset_to_home(env, dt, RESET_TIME_S, RESET_SPEED_MPS, FPS)
                env_processor.reset()
                action_processor.reset()
                transition = env_processor(create_transition(
                    observation=env.robot.get_observation(),
                    info={TeleopEvents.IS_INTERVENTION: False},
                ))
                episode_step = 0
                episode_start = time.perf_counter()
                logger.info("--- Episode %d ---", episode_idx + 1)
                if USE_TTS:
                    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            precise_sleep(max(dt - (time.perf_counter() - step_start), 0.0))

    except KeyboardInterrupt:
        logger.info("Recording stopped by user.")
    except Exception:
        logger.exception("Recording failed")
    finally:
        try:
            dataset.finalize()
        except Exception:
            logger.exception("dataset.finalize failed")
        try:
            env.close()
        except Exception:
            logger.exception("env.close failed")
        if teleop_device is not None:
            try:
                teleop_device.disconnect()
            except Exception:
                logger.exception("teleop disconnect failed")
        if USE_TTS:
            log_say(f"Done. Recorded {episode_idx} episodes.", blocking=True)
        logger.info("Done. %d episodes in %s.", episode_idx, REPO_ID)


if __name__ == "__main__":
    main()
