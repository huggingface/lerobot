"""Closed-loop inference of a trained ACT policy on the UR10.

Mirrors ``act_train/act_using_example.py`` but routes through ``UR10RobotEnv``
(``make_robot_env`` + ``make_processors`` from ``gym_manipulator``) instead of
the direct-robot-`send_action` pattern. UR10 doesn't have ``send_action``; the
env's ``step()`` takes either the 4-D ``[dx, dy, dz, gripper]`` action or the
5-D ``[dx, dy, dz, dyaw, gripper]`` action (when ``use_yaw`` is set in the JSON
config) and accumulates the deltas into the streaming target. Gripper is always
the LAST element — this script indexes it as ``action[-1]`` so the same code
path handles both layouts.

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

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
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
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep
from lerobot.teleoperators.utils import TeleopEvents

from _ur10_reset import auto_reset_to_home  # sibling module in act_train/
from record_ur10_act import _build_features  # sibling module in act_train/

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -- user-tunable ---------------------------------------------------------------
MODEL_DIR = "outputs/act/ur10/pcb_act_3cams_yaw2/last"
DATASET_REPO_ID = "local/pcb_act_3cams_yaw"   # for dataset stats
CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams_yaw.json"
NUM_EPISODES = 30
EPISODE_TIME_S = 30      # safety upper bound; user ends earlier via gamepad
RESET_TIME_S = 7         # total between-episode budget (motion + hold-at-home)
RESET_SPEED_MPS = 0.1    # auto-reset linear velocity, m/s (matches env.reset's moveL)
FPS = 10

# HG-DAgger: collect human-corrected eval frames into a new dataset so we can
# fine-tune ACT against its own failure modes. The recorded action is the
# post-intervention action (= human's command when intervening, ACT's command
# otherwise), gripper already in STATE encoding (read from observation.state[-1]
# in the per-step block below) so the dataset is directly trainable.
RECORD_DAGGER_DATASET = False
DAGGER_REPO_ID = "local/ur10_act_usb_insertion_dagger_v1"
DAGGER_TASK = "usb_insertion"

# Override the trained model's temporal_ensemble_coeff at inference. Larger
# values weight recent predictions more (sharper, more reactive); the trained
# default is 0.01 (heavy smoothing). Set to None to leave the trained value
# untouched. Only takes effect if the model was trained with ensembling on
# (i.e. cfg.temporal_ensemble_coeff is not None and n_action_steps == 1).
EVAL_TEMPORAL_ENSEMBLE_COEFF =  None

# Inference-time action gain on the Cartesian dx/dy/dz components (gripper
# left untouched). ACT replays demonstration step sizes; if the demos showed
# slow careful alignment (e.g. via gamepad micro-corrections), the policy
# inherits sub-mm per-step deltas that stall progress near the target. A
# multiplicative gain of 1.5-3× amplifies these without retraining. Clipped
# to ±1 (env's normalized action range) so peak-speed approach motions don't
# overshoot. Set to 1.0 to disable.
EVAL_ACTION_GAIN: float = 1.0
# -------------------------------------------------------------------------------


def main() -> None:
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    dt = 1.0 / FPS

    # Yaw / gripper flags drive both the dataset feature schema and the per-step gripper
    # translation. Read use_yaw via getattr so an older JSON without the key still parses.
    ik_cfg = cfg.env.processor.inverse_kinematics
    use_yaw = bool(getattr(ik_cfg, "use_yaw", False)) if ik_cfg else False
    use_gripper = (
        cfg.env.processor.gripper.use_gripper
        if cfg.env.processor.gripper is not None
        else True
    )
    # Number of continuous dims at the front of the action vector. Gripper (if present)
    # follows at index -1. Layout: [dx, dy, dz, (dyaw,) (gripper,)].
    cont_dims = 3 + int(use_yaw)

    metadata = LeRobotDatasetMetadata(DATASET_REPO_ID)
    policy: ACTPolicy = ACTPolicy.from_pretrained(MODEL_DIR)
    policy.eval()
    device = torch.device(policy.config.device)
    policy.to(device)
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=MODEL_DIR, dataset_stats=metadata.stats
    )
    logger.info("Policy loaded from %s on %s", MODEL_DIR, device)

    # Optional inference-time override of temporal_ensemble_coeff. The ensembler
    # holds the precomputed `ensemble_weights = exp(-coeff * arange(chunk_size))`
    # tensor — changing only `policy.config.temporal_ensemble_coeff` would have no
    # effect, so we rebuild the ensembler. Skipped silently if the model was
    # trained without ensembling (config field is None).
    if (
        EVAL_TEMPORAL_ENSEMBLE_COEFF is not None
        and policy.config.temporal_ensemble_coeff is not None
    ):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        old_coeff = policy.config.temporal_ensemble_coeff
        policy.config.temporal_ensemble_coeff = EVAL_TEMPORAL_ENSEMBLE_COEFF
        policy.temporal_ensembler = ACTTemporalEnsembler(
            EVAL_TEMPORAL_ENSEMBLE_COEFF, policy.config.chunk_size
        )
        logger.info(
            "Override temporal_ensemble_coeff: %g → %g (sharper / more recency-biased)",
            old_coeff, EVAL_TEMPORAL_ENSEMBLE_COEFF,
        )

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(
        env, teleop_device, cfg.env, str(device)
    )

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    # HG-DAgger dataset: create against the SAME feature schema record_ur10_act.py
    # uses, sized from the live transition so shapes are guaranteed to match what
    # add_frame() will receive. Disabled by default — toggle RECORD_DAGGER_DATASET.
    dagger_dataset: LeRobotDataset | None = None
    dagger_features: dict = {}
    if RECORD_DAGGER_DATASET:
        # Match the recording schema exactly — same use_gripper / use_yaw flags so an
        # ACT model fine-tuned on the DAgger set sees the same action layout it saw
        # during the original record run.
        dagger_features = _build_features(
            transition[TransitionKey.OBSERVATION], use_gripper=use_gripper, use_yaw=use_yaw,
        )
        dagger_dataset = LeRobotDataset.create(
            repo_id=DAGGER_REPO_ID,
            fps=FPS,
            features=dagger_features,
            robot_type="ur10",
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
        )
        logger.info(
            "DAgger recording ON → %s (%d episodes @ %d Hz)",
            DAGGER_REPO_ID, NUM_EPISODES, FPS,
        )

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

            # Inference-time action gain on the continuous dims (xyz + yaw if enabled),
            # leaving gripper untouched — it's a state prediction that gets discretized
            # below. Clamp to env's normalized action range [-1, 1] so peak-speed
            # approach motions are bounded at full speed instead of overshooting.
            if EVAL_ACTION_GAIN != 1.0 and action.numel() >= cont_dims:
                action[:cont_dims] = (action[:cont_dims] * EVAL_ACTION_GAIN).clamp(-1.0, 1.0)

            # Translate policy's gripper STATE prediction → env.step's COMMAND encoding.
            # Dataset action[-1] is in {0.0=closed, 1.0=open} (state encoding from the
            # translation script); after unnormalize the policy outputs ~[0, 1]. We
            # compare against the gripper's currently-commanded state (read from
            # observation.state[-1] — ur10_robot.py keeps that flag in sync with
            # send_gripper) and emit a transition command only when the desired state
            # differs. UR10RobotEnv.step interprets {0=CLOSE, 1=STAY, 2=OPEN}.
            #
            # Yaw mode: the continuous dxyz (+dyaw) precede the gripper. action[-1] is
            # the gripper regardless of layout; we rebuild the action by passing the
            # leading `cont_dims` continuous values through unchanged and replacing the
            # last element with the discretized command.
            if use_gripper and action.numel() >= cont_dims + 1:
                a = action.detach().cpu().float().numpy()
                obs_state = transition[TransitionKey.OBSERVATION][OBS_STATE]
                if isinstance(obs_state, torch.Tensor):
                    obs_state = obs_state.detach().cpu().float().numpy()
                current_state = float(np.asarray(obs_state).reshape(-1)[-1])  # last dim
                predicted_state = 1.0 if float(a[-1]) > 0.5 else 0.0
                if predicted_state == current_state:
                    cmd_int, cmd_name = 1, "STAY"
                elif predicted_state > current_state:
                    cmd_int, cmd_name = 2, "OPEN"   # closed → open
                else:
                    cmd_int, cmd_name = 0, "CLOSE"  # open → closed
                # Rewrite the last element to the discrete command env.step expects;
                # keep the leading continuous dims intact.
                new_vals = [float(a[i]) for i in range(cont_dims)] + [float(cmd_int)]
                action = torch.tensor(new_vals, device=action.device, dtype=action.dtype)
                if use_yaw:
                    logger.info(
                        "  step %d  action=[dx=%+.3f dy=%+.3f dz=%+.3f dyaw=%+.3f g_raw=%+.3f "
                        "→ desired=%s, current=%s] → gripper=%s",
                        episode_step,
                        float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[-1]),
                        "OPEN" if predicted_state == 1.0 else "CLOSED",
                        "OPEN" if current_state == 1.0 else "CLOSED",
                        cmd_name,
                    )
                else:
                    logger.info(
                        "  step %d  action=[dx=%+.3f dy=%+.3f dz=%+.3f g_raw=%+.3f "
                        "→ desired=%s, current=%s] → gripper=%s",
                        episode_step,
                        float(a[0]), float(a[1]), float(a[2]), float(a[-1]),
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

            # HG-DAgger per-step recording. Mirrors record_ur10_act.py: pull the
            # post-intervention action from complementary_data — that's the action
            # that actually drove the robot (human's command when intervening,
            # ACT's translated command otherwise). Gripper is written in STATE
            # encoding ({0.0=closed, 1.0=open}, read from observation.state[-1])
            # so the dataset is directly trainable without a post-step translation.
            if dagger_dataset is not None:
                comp = transition[TransitionKey.COMPLEMENTARY_DATA]
                action_to_record = comp.get("teleop_action", transition[TransitionKey.ACTION])
                if isinstance(action_to_record, torch.Tensor):
                    action_to_record = action_to_record.detach().cpu().float()
                else:
                    action_to_record = torch.tensor(action_to_record, dtype=torch.float32)
                if action_to_record.ndim > 1:
                    action_to_record = action_to_record.squeeze(0)

                # Overwrite gripper command with the post-step gripper STATE — same
                # source of truth as record_ur10_act.py. Gripper is always the LAST
                # element (action[-1]) in both 4-D and 5-D yaw layouts.
                if use_gripper:
                    obs_state_raw = transition[TransitionKey.OBSERVATION][OBS_STATE]
                    if isinstance(obs_state_raw, torch.Tensor):
                        obs_state_raw = obs_state_raw.detach().cpu().float()
                    gripper_state = float(np.asarray(obs_state_raw).reshape(-1)[-1])
                    action_to_record[-1] = gripper_state

                dagger_frame: dict = {ACTION: action_to_record, "task": DAGGER_TASK}
                for k, v in transition[TransitionKey.OBSERVATION].items():
                    if isinstance(v, torch.Tensor) and k in dagger_features:
                        dagger_frame[k] = v.squeeze(0).detach().cpu()
                dagger_dataset.add_frame(dagger_frame)

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
                if dagger_dataset is not None:
                    dagger_dataset.save_episode()
                    logger.info("  dagger: saved episode %d", episode_idx + 1)
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
                # Go through `env._augment_observation` so the post-reset obs carries
                # the yaw slot when use_yaw=True (matches env.reset()/env.step()).
                # Without this wrap, episode N+1's first transition is raw 16-D from
                # the driver and the 17-D normalizer raises a shape mismatch.
                transition = env_processor(create_transition(
                    observation=env._augment_observation(env.robot.get_observation()),
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
        if dagger_dataset is not None:
            try:
                dagger_dataset.finalize()
                logger.info("DAgger dataset finalized → %s", DAGGER_REPO_ID)
            except Exception:
                logger.exception("dagger_dataset.finalize failed")
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
