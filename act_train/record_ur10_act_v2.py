"""Record UR10 demonstrations for offline ACT training — v2 (RC10-style).

Why v2 exists
=============
The original ``record_ur10_act.py`` records the gamepad's per-step delta vector
``[dx, dy, dz, dyaw, gripper_state]`` as the dataset's action. The audit
(``act_train/audit_dataset.py``) revealed that this representation degrades
catastrophically when paired with bang-bang teleop + the streaming-thread's
servoL smoothing: ~80% of recorded action frames are exactly zero while the
wrist is physically still moving from earlier impulses. L1 regression on this
data learns "output ≈ 0" at the most important phases of the task (fine
alignment near the goal). The trained policy then stops moving once the wrist
arrives near the target.

The v2 fix mirrors the working RC10 ACT pipeline (``RC10FollowerCut`` in
``src/lerobot/robots/rc10_follower/rc10_follower.py`` —
``record_dataset_ps4_joystick.py`` + ``act_training_example.py`` +
``act_using_example.py``). Both pipelines drive the robot via the same env
delta path, but the v2 dataset stores the env's *resulting absolute target
pose* as the action. With absolute targets:

- Every frame's action label is meaningful (the current commanded pose), never
  zero except by coincidence.
- The label is dense, not sparse — there are no "wrist drifts while action=0"
  frames.
- L1 regression converges to "predict the commanded target", which the policy
  can execute decisively at inference.

Action / observation schema (matches RC10FollowerCut)
=====================================================
Recorded action (per frame, 5 floats):
  {x.pos, y.pos, z.pos, yaw.pos, gripper.pos}
  - x.pos, y.pos, z.pos : absolute base-frame target [m]
  - yaw.pos             : target wrist yaw OFFSET from R_home [rad]
  - gripper.pos         : target gripper state {0.0 = closed, 1.0 = open}

Recorded observation.state (per frame, 11 floats):
  [joint_pos(6), tcp_x, tcp_y, tcp_z, tcp_yaw_offset, gripper]
  See ``UR10RobotEnv.get_act_observation`` for the per-slot semantics.

Recorded images: one (3, 128, 128) RGB tensor per camera (post-crop / resize).

Teleop / streaming details
==========================
The wrist is still driven via the gamepad → ``env.step(delta_action)`` path,
which already does the bounds clipping + yaw composition. After each step the
env's internal ``target_xyz`` and ``target_yaw`` reflect the resulting absolute
target — that's what we record.

Between episodes we use ``auto_reset_to_home`` (same as v1) to bypass
``servoStop``. The phase-1.5 fix that zeroes ``env.target_yaw`` between episodes
is required for v2 too (otherwise the recorded absolute yaw target would carry
the previous episode's accumulated offset into the new episode).

Usage
=====
    python act_train/record_ur10_act_v2.py

Tune the constants at the top. End an episode early via the gamepad (Triangle
= success, Cross = fail); the RERECORD_EPISODE event re-records the current
episode (Square / Circle, depending on your gamepad mapping).
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
CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams_yaw_v2.json"
REPO_ID = "local/pcb_act_3cams_yaw_v2_1"
TASK_DESCRIPTION = "pcb_act_3cams_yaw_v2_1"
NUM_EPISODES = 60
EPISODE_TIME_S = 30      # truncates the episode if the user doesn't end it
RESET_TIME_S = 10        # total between-episode budget (motion + hold-at-home)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s (matches env.reset's moveL)
FPS = 30                 # must equal cfg.env.fps; matches RC10's working pipeline
DEVICE = "cpu"           # env-processor lives on CPU; only the policy needs GPU

# UX feedback
USE_TTS = True
USE_RERUN = True
RERUN_EVERY_N_STEPS = 1
# -------------------------------------------------------------------------------


# Names of the absolute-target action dimensions. Order matters — `make_robot_action`
# at eval time uses this list to construct the dict from the policy's output tensor.
ACTION_NAMES_V2: list[str] = ["x.pos", "y.pos", "z.pos", "yaw.pos", "gripper.pos"]


def _build_features_v2(transition_obs: dict) -> dict:
    """Build the LeRobotDataset feature schema for the v2 ACT pipeline.

    Action schema is fixed (RC10-style 5-D absolute target pose). Observation
    schemas (state shape + image shapes) are read from the live transition so
    they match exactly what the policy will see at inference (after crop/resize
    + AddBatchDim + DeviceMove already applied by env_processor).
    """
    features: dict[str, dict] = {
        ACTION: {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES_V2),),
            "names": ACTION_NAMES_V2,
        },
    }
    for key, val in transition_obs.items():
        if "image" in key and isinstance(val, torch.Tensor):
            features[key] = {
                "dtype": "video",
                "shape": tuple(val.squeeze(0).shape),  # (C, H, W) after dropping batch dim
                "names": ["channels", "height", "width"],
            }
        elif key == OBS_STATE and isinstance(val, torch.Tensor):
            features[key] = {
                "dtype": "float32",
                "shape": tuple(val.squeeze(0).shape),
                "names": None,
            }
    return features


def _override_state_with_v2(env, transition: dict) -> dict:
    """Swap the transition's HIL-SERL 17-D state for the v2 11-D ACT state.

    `step_env_and_process_transition` builds the transition's `observation.state`
    from ``UR10RobotEnv._augment_observation`` — the HIL-SERL relative-coords
    view ``[joint_pos(6), joint_vel(6), tcp_xyz_rel(3), yaw_offset(1),
    gripper(1)]`` (17-D). The v2 ACT dataset wants the absolute view from
    ``UR10RobotEnv.get_act_observation`` — ``[joint_pos(6), tcp_x_abs,
    tcp_y_abs, tcp_z_abs, yaw_offset, gripper]`` (11-D). We replace just the
    state slot; the cropped/resized `observation.images.*` produced by
    ``env_processor`` (ImageCropResizeProcessorStep) are kept as-is because
    they're already inference-ready.

    Must be called BEFORE the dataset schema is built (so features get sized
    11-D, not 17-D) AND before each frame is saved.
    """
    v2_obs = env.get_act_observation()
    # Add a batch dim to match the (1, D) shape env_processor's
    # AddBatchDimensionProcessorStep emits, so the dataset's add_frame sees a
    # consistent tensor layout across frames.
    state_tensor = torch.from_numpy(v2_obs["agent_pos"].copy()).unsqueeze(0).float()
    transition[TransitionKey.OBSERVATION][OBS_STATE] = state_tensor
    return transition


def _absolute_target_action(env, *, use_yaw: bool, use_gripper: bool) -> torch.Tensor:
    """Read the env's current commanded absolute target pose + gripper state.

    This is what gets *stored in the dataset* for the current frame — the env's
    target reflects the cumulative effect of all gamepad deltas applied so far,
    yielding a dense, meaningful action signal (vs. the sparse bang-bang deltas
    that doom the v1 pipeline). When ``use_yaw=False`` the yaw slot is forced
    to 0.0; when ``use_gripper=False`` the gripper slot mirrors the current
    state (the dataset shape stays fixed, the value is just static).
    """
    target_xyz = env.target_xyz
    if target_xyz is None:
        # Shouldn't happen after env.reset(), but defend against it anyway.
        target_xyz = np.array(env.robot.get_current_tcp()[:3], dtype=np.float32)
    target_yaw = float(env.target_yaw) if (use_yaw and env.target_yaw is not None) else 0.0
    gripper_state = float(env.robot.gripper.is_open)
    return torch.tensor(
        [
            float(target_xyz[0]),
            float(target_xyz[1]),
            float(target_xyz[2]),
            target_yaw,
            gripper_state,
        ],
        dtype=torch.float32,
    )


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
    ik_cfg = cfg.env.processor.inverse_kinematics
    use_yaw = bool(getattr(ik_cfg, "use_yaw", False)) if ik_cfg else False
    if not use_yaw:
        logger.warning(
            "use_yaw is False in the JSON; the v2 dataset will still have a yaw.pos "
            "column but it will be constant 0.0. Recommend enabling yaw for ACT training."
        )

    # Neutral action drives the EXISTING delta-action env.step path while the
    # operator teleoperates via gamepad. Same shape as v1 — the env wrapper
    # handles bounds + yaw composition the same way.
    action_dim_delta = 3 + int(use_yaw) + int(use_gripper)
    neutral_action = torch.zeros(action_dim_delta, dtype=torch.float32)
    if use_gripper:
        neutral_action[-1] = 1.0  # STAY

    # First reset so we can size the dataset features from a real processed obs.
    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = env_processor(create_transition(observation=obs, info=info))
    # Override the env_processor's 17-D HIL-SERL state with the v2 11-D ACT state
    # BEFORE building the dataset feature schema (otherwise the schema would lock
    # observation.state to (17,) and the per-frame override would shape-mismatch).
    transition = _override_state_with_v2(env, transition)

    features = _build_features_v2(transition[TransitionKey.OBSERVATION])
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
    logger.info(
        "v2 action schema: %s   (absolute target poses, NOT deltas)", ACTION_NAMES_V2
    )
    logger.info("Hold the gamepad intervention button to drive; Triangle / Cross to end episode.")

    if USE_RERUN:
        init_rerun(session_name=f"ur10_record_v2_{TASK_DESCRIPTION}")

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

            # Drive the wrist via the existing delta-action env step. The teleop
            # produces the delta (intervention path), env.step accumulates it into
            # env.target_xyz / env.target_yaw, sends to robot, returns the next obs.
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            # Swap in the v2 11-D ACT state before saving — see _override_state_with_v2
            # for the rationale. Cropped images are kept from env_processor.
            transition = _override_state_with_v2(env, transition)

            terminated = bool(transition.get(TransitionKey.DONE, False))
            truncated = bool(transition.get(TransitionKey.TRUNCATED, False))
            if not terminated and not truncated and episode_step + 1 >= max_episode_steps:
                truncated = True

            # ---- v2 action: store the env's resulting ABSOLUTE target -------
            # This is the key v2 deviation from the legacy script. The legacy
            # script stored the gamepad's per-step delta (sparse impulses); here
            # we store the dense absolute target the env is currently chasing.
            action_to_record = _absolute_target_action(
                env, use_yaw=use_yaw, use_gripper=use_gripper
            )

            # Pull observation tensors (drop batch dim — dataset expects (C, H, W) / (D,)).
            frame: dict = {ACTION: action_to_record, "task": TASK_DESCRIPTION}
            for k, v in transition[TransitionKey.OBSERVATION].items():
                if isinstance(v, torch.Tensor) and k in features:
                    frame[k] = v.squeeze(0).detach().cpu()
            dataset.add_frame(frame)
            episode_step += 1
            global_step += 1

            if USE_RERUN and (global_step % RERUN_EVERY_N_STEPS == 0):
                # log_rerun_data expects dict-shaped inputs; calling it with a raw
                # tensor blows up on `if action:` (multi-element tensors raise
                # "Boolean value ... ambiguous"). Mirror the v1 wrapping pattern
                # (record_ur10_act.py:240-245): pass observation tensors as a
                # numpy dict, action as a single-key {ACTION: array} dict.
                rr_obs = {
                    k: v.numpy() for k, v in frame.items()
                    if k not in (ACTION, "task") and isinstance(v, torch.Tensor)
                }
                rr_action = {ACTION: action_to_record.numpy()}
                log_rerun_data(observation=rr_obs, action=rr_action, compress_images=False)

            if terminated or truncated:
                ep_time = time.perf_counter() - episode_start
                rerecord = transition[TransitionKey.INFO].get(
                    TeleopEvents.RERECORD_EPISODE, False
                )
                success = transition[TransitionKey.INFO].get(TeleopEvents.SUCCESS, False)

                if rerecord:
                    logger.info(
                        "Re-recording episode %d (%.1fs)", episode_idx + 1, ep_time
                    )
                    dataset.clear_episode_buffer()
                    if USE_TTS:
                        log_say(f"Re-recording episode {episode_idx + 1}")
                else:
                    logger.info(
                        "Episode %d %s after %d steps (%.1fs)",
                        episode_idx + 1,
                        "SUCCESS" if success else "DONE",
                        episode_step,
                        ep_time,
                    )
                    dataset.save_episode()
                    episode_idx += 1

                if episode_idx >= NUM_EPISODES:
                    break

                # Between-episode reset bypasses `servoStop` (the well-known wedge
                # site). Phase-1.5 fix inside auto_reset_to_home zeroes
                # `env.target_yaw` — without that, the next episode's recorded
                # absolute yaw target would carry over.
                auto_reset_to_home(env, dt, RESET_TIME_S, RESET_SPEED_MPS, FPS)
                env_processor.reset()
                action_processor.reset()
                transition = env_processor(create_transition(
                    observation=env._augment_observation(env.robot.get_observation()),
                    info={TeleopEvents.IS_INTERVENTION: False},
                ))
                # Swap in the v2 11-D ACT state so the first frame of episode N+1
                # carries the v2 schema's state, not the env_processor's 17-D one.
                transition = _override_state_with_v2(env, transition)
                episode_step = 0
                episode_start = time.perf_counter()
                if USE_TTS:
                    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

            # Maintain FPS timing.
            precise_sleep(max(dt - (time.perf_counter() - step_start), 0.0))

    except KeyboardInterrupt:
        logger.info("Recording stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Recording failed")
    finally:
        try:
            dataset.finalize()
            logger.info("Dataset finalized → %s", REPO_ID)
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
        logger.info("Recorded %d episodes total", episode_idx)


if __name__ == "__main__":
    main()
