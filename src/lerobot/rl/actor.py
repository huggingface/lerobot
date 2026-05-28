#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from functools import lru_cache
from queue import Empty

import grpc
import numpy as np
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import rc10 as _rc10_register  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

# Main entry point


def _apply_nstep_returns(transitions: list, n: int, gamma: float) -> list:
    """Rewrite each transition to its n-step return version.

    For transition t: reward → Σ_{k<m} γ^k r_{t+k}, next_state → s_{t+m},
    done → terminal-within-window, and stores γ^m in complementary_info["discount_n"]
    so the critic bootstraps with the correct per-transition discount.
    m = min(n, steps remaining to episode end / terminal).
    """
    T = len(transitions)
    out = []
    for t in range(T):
        R = 0.0
        disc = 1.0
        done_n = False
        last_idx = t
        for k in range(n):
            idx = t + k
            if idx >= T:
                break
            tr = transitions[idx]
            R += disc * float(tr["reward"])
            disc *= gamma
            last_idx = idx
            if bool(tr["done"]):
                done_n = True
                break
        ci = dict(transitions[t].get("complementary_info") or {})
        ci["discount_n"] = float(disc)  # γ^m
        out.append(
            Transition(
                state=transitions[t]["state"],
                action=transitions[t]["action"],
                reward=R,
                next_state=transitions[last_idx]["next_state"],
                done=done_n,
                truncated=transitions[t]["truncated"],
                complementary_info=ci,
            )
        )
    return out


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    # Actor never writes to output_dir (only the learner does), but it shares cfg
    # with the learner under HIL-SERL. Skip the "output_dir already exists" check
    # so re-launching the actor against an existing run doesn't crash. The learner
    # still gets the protective check because it doesn't set this env var.
    os.environ.setdefault("LEROBOT_SKIP_OUTPUT_DIR_CHECK", "1")
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    # DSRL: attach frozen diffusion policy + its preprocessor for action generation
    if getattr(cfg.policy, "type", None) == "dsrl_ext":
        dp_path = getattr(cfg.policy, "diffusion_pretrained_path", None)
        if dp_path:
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from lerobot.policies.factory import make_pre_post_processors as _make_dp_pp

            dp = DiffusionPolicy.from_pretrained(pretrained_name_or_path=dp_path)
            dp.to(cfg.policy.device)
            dp_pre, dp_post = _make_dp_pp(
                policy_cfg=dp.config, pretrained_path=dp_path,
                preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
            )
            policy.load_diffusion_policy(dp, preprocessor=dp_pre, postprocessor=dp_post)
            logging.info("[ACTOR] DSRL: loaded frozen diffusion policy + preprocessor from %s", dp_path)
        else:
            logging.warning("[ACTOR] DSRL: no diffusion_pretrained_path — actor will output raw noise")

    # Diagnostic: the resolved policy class MUST match cfg.policy.type. If
    # cfg says qc_ext but `policy` ends up SACPolicy, something is wrong
    # (plugin not loaded → draccus silently fell back, OR cfg.policy.type
    # was overridden via CLI). This print survives buffering + tee.
    print(
        f"[ACTOR-DBG] policy class = {type(policy).__name__} "
        f"({type(policy).__module__}); cfg.policy.type = {cfg.policy.type!r}; "
        f"has load_actor_state_dict = {hasattr(policy, 'load_actor_state_dict')}",
        flush=True,
    )
    logging.info(
        "[ACTOR] policy class=%s module=%s cfg.policy.type=%s "
        "has_load_actor_state_dict=%s",
        type(policy).__name__,
        type(policy).__module__,
        cfg.policy.type,
        hasattr(policy, "load_actor_state_dict"),
    )

    # CRITICAL: load the TOP-level policy's pre/post-processors and apply
    # them around every `policy.select_action` call. Without this, QC (and
    # any other plugin policy trained on normalized obs) receives raw
    # uint8 images / un-normalized state and outputs garbage. See
    # `feedback_eval_preprocessor_critical.md` — same failure class that
    # bit residual-SAC, fixed there via `_base_act` (which wraps the
    # FROZEN BASE policy's processor, NOT the top-level policy's).
    #
    # Distinct from `_base_policy._pre_processor` (residual base for ACT):
    # this is the TOP-level policy's processor, loaded from the same
    # `pretrained_path` that `make_policy` consumed. SAC training from
    # scratch (no pretrained_path) still gets fresh processors — no
    # back-compat break.
    from lerobot.policies.factory import make_pre_post_processors as _make_pp
    try:
        _policy_pre_processor, _policy_post_processor = _make_pp(
            policy_cfg=cfg.policy,
            pretrained_path=getattr(cfg.policy, "pretrained_path", None),
        )
        logging.info(
            "[ACTOR] policy pre/post-processor loaded (pretrained_path=%s)",
            getattr(cfg.policy, "pretrained_path", None),
        )
    except Exception as _e:
        # SAC from-scratch path may not have pretrained processors yet.
        # Continue without; the existing behavior (no preprocessing in
        # select_action wrap) is preserved.
        logging.warning(
            "[ACTOR] could not build policy pre/post-processors (%s) — "
            "continuing with un-wrapped select_action; this is OK for SAC-from-scratch "
            "but WILL produce garbage for any policy trained on normalized obs.", _e,
        )
        _policy_pre_processor = None
        _policy_post_processor = None

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()
    # Reset chunk-aware policies (e.g. QC) so any stale action queue from a
    # prior run is cleared. SAC's reset() is a no-op, so back-compat safe.
    if hasattr(policy, "reset"):
        policy.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # Residual mode: cache a frozen base policy + action_dim to attach
    # `observation.base_action` onto every observation flowing into the
    # buffer. ACT manages its own chunk queue across calls.
    _residual_mode = getattr(cfg.policy, "residual_mode", False)
    _base_policy = getattr(policy, "_base_policy", None) if _residual_mode else None
    def _base_act(obs):
        # ACT base needs the saved preprocessor (normalize images + state)
        # before select_action; otherwise raw uint8 → garbage actions.
        pre = getattr(_base_policy, "_pre_processor", None)
        b = pre(obs) if pre is not None else obs
        return _base_policy.select_action(b)

    if _residual_mode and _base_policy is not None:
        _base_policy.reset()
        _ba = _base_act(transition[TransitionKey.OBSERVATION])
        transition[TransitionKey.OBSERVATION]["observation.base_action"] = _ba

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Tracks IS_INTERVENTION across steps for edge detection. Used by chunk-aware
    # policies (QC) to flush their internal action queue at intervention onset.
    # SAC ignores this; remains False forever.
    prev_intervention_flag = False
    # Tracks whether the episode ended with a Triangle (manual SUCCESS) flag.
    # Set True on any frame with info[TeleopEvents.SUCCESS]=True. If False at
    # episode end, transitions are DISCARDED (not pushed to learner).
    episode_success = False
    # Counter of how many stage-advance button presses landed this episode.
    # Each press fires +stage_advance_bonus reward in the SARM step.
    episode_stage_advances = 0
    # Discard counter for telemetry.
    discarded_episodes = 0
    # Rolling buffers for episode-level moving averages (last N completed eps).
    _ROLLING_N = 5
    rolling_rewards: list[float] = []
    rolling_successes: list[int] = []
    rolling_stage_advances: list[int] = []
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0
    # Track SARM progress per episode (independent of reward shaping mode).
    last_sarm_progress = 0.0
    max_sarm_progress = 0.0
    # Per-episode gripper diagnostics (residual mode only): we want to confirm
    # the residual is actually using its gripper dim. Tracks running sums of
    # base / combined / residual gripper values across the episode and counts
    # how many steps the residual flipped the post-deadband band.
    grip_base_sum = 0.0
    grip_combined_sum = 0.0
    grip_residual_abs_sum = 0.0
    grip_residual_signed_sum = 0.0
    grip_flip_count = 0  # frac of steps where deadband band(base) != band(combined)

    policy_timer = TimerManager("Policy inference", log=False)

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # Time policy inference and check if it meets FPS requirement
        with policy_timer:
            # Extract observation from transition for policy.
            # CRITICAL: normalize via the TOP-level policy's preprocessor
            # before select_action; otherwise QC (or any policy trained
            # on normalized obs) outputs garbage. See
            # `feedback_eval_preprocessor_critical.md`.
            if _policy_pre_processor is not None:
                _batch = _policy_pre_processor(observation)
            else:
                _batch = observation
            action = policy.select_action(batch=_batch)
            if _policy_post_processor is not None:
                action = _policy_post_processor(action)
        policy_fps = policy_timer.fps_last

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # Residual-mode gripper diagnostics: track per-step base / combined /
        # residual gripper to confirm the residual head actually moves it.
        if _residual_mode:
            try:
                _act_full = action.detach().cpu().numpy().flatten() if hasattr(action, "detach") else np.asarray(action).flatten()
                _base_full = observation["observation.base_action"].detach().cpu().numpy().flatten()
                _g_base = float(_base_full[-1])
                _g_combined = float(_act_full[-1])
                _g_residual = _g_combined - _g_base
                grip_base_sum += _g_base
                grip_combined_sum += _g_combined
                grip_residual_signed_sum += _g_residual
                grip_residual_abs_sum += abs(_g_residual)
                # Deadband band classification: -1 if a < -db, +1 if a > db, else 0.
                _db = (
                    cfg.env.processor.gripper.continuous_gripper_deadband
                    if cfg.env.processor.gripper is not None
                    else 0.33
                )
                _band_base = -1 if _g_base < -_db else (1 if _g_base > _db else 0)
                _band_comb = -1 if _g_combined < -_db else (1 if _g_combined > _db else 0)
                if _band_base != _band_comb:
                    grip_flip_count += 1
            except Exception:
                pass

        # Visualization: pass action arrows to the underlying mujoco env so the
        # passive viewer renders them on top of the simulation. In residual
        # mode we draw two arrows (base direction in green, residual delta in
        # red); otherwise one arrow showing the executed action.
        try:
            inner_env = online_env
            while hasattr(inner_env, "env"):
                inner_env = inner_env.env
            if hasattr(inner_env, "set_action_arrows") and hasattr(inner_env, "get_tcp_position"):
                tcp = np.asarray(inner_env.get_tcp_position(), dtype=np.float64).flatten()
                # Visible-arrow length tuning: typical normed action mag ~ 0.5,
                # we want arrow ~5cm. Scale = 0.10 m / unit.
                arrow_scale = 0.10
                act_np = action.detach().cpu().numpy().flatten() if hasattr(action, "detach") else np.asarray(action).flatten()
                act_np = act_np.astype(np.float64)
                arrows: list = []
                if _residual_mode:
                    base_np = observation["observation.base_action"].detach().cpu().numpy().flatten().astype(np.float64)
                    end_base = tcp + arrow_scale * base_np[:3]
                    arrows.append((tcp, end_base, (0.0, 1.0, 0.2, 0.9)))  # base = green
                    res_np = act_np[:3] - base_np[:3]
                    if np.linalg.norm(res_np) > 1e-4:
                        end_res = end_base + arrow_scale * 3.0 * res_np  # 3x to make small residual visible
                        arrows.append((end_base, end_res, (1.0, 0.2, 0.0, 0.9)))  # residual = red
                else:
                    end_act = tcp + arrow_scale * act_np[:3]
                    arrows.append((tcp, end_act, (0.0, 0.6, 1.0, 0.9)))  # policy action = blue
                inner_env.set_action_arrows(arrows)
        except Exception as _viz_err:
            # Visualization is best-effort; don't kill training on render bugs.
            if interaction_step < 5:
                logging.warning("[ACTOR] action arrow viz failed: %s", _viz_err)

        # Use the new step function
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        # Residual mode: attach next base_action to next observation. Done
        # *after* env_processor so the obs is in the canonical (CHW float)
        # form expected by the base policy. ACT will pop from its chunk queue.
        if _residual_mode and _base_policy is not None:
            _ba = _base_act(new_transition[TransitionKey.OBSERVATION])
            new_transition[TransitionKey.OBSERVATION]["observation.base_action"] = _ba

        # Display camera feeds if configured
        display_cameras = (
            cfg.env.processor.observation.display_cameras
            if cfg.env.processor.observation is not None
            else False
        )
        if display_cameras:
            import matplotlib.pyplot as plt

            obs_dict = new_transition.get(TransitionKey.OBSERVATION, {})
            image_keys = sorted([k for k in obs_dict if "image" in k and isinstance(obs_dict[k], torch.Tensor)])

            if image_keys and not hasattr(act_with_policy, "_cam_fig"):
                plt.ion()
                fig, axes = plt.subplots(1, len(image_keys), figsize=(4 * len(image_keys), 4))
                if len(image_keys) == 1:
                    axes = [axes]
                img_plots = []
                for ax, key in zip(axes, image_keys):
                    img = obs_dict[key].squeeze(0).cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    im = ax.imshow(img)
                    ax.set_title(key.replace("observation.images.", ""))
                    ax.axis("off")
                    img_plots.append(im)
                fig.tight_layout()
                act_with_policy._cam_fig = fig
                act_with_policy._cam_plots = img_plots
                act_with_policy._cam_keys = image_keys
                plt.show(block=False)
                plt.pause(0.001)
            elif image_keys and hasattr(act_with_policy, "_cam_fig"):
                for im, key in zip(act_with_policy._cam_plots, act_with_policy._cam_keys):
                    img = obs_dict[key].squeeze(0).cpu().permute(1, 2, 0).numpy()
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                    im.set_data(img)
                act_with_policy._cam_fig.canvas.draw_idle()
                act_with_policy._cam_fig.canvas.flush_events()

        # Extract values from processed transition
        next_observation = {
            k: v
            for k, v in new_transition[TransitionKey.OBSERVATION].items()
            if k in cfg.policy.input_features
        }

        # Teleop action is the action that was executed in the environment
        # It is either the action from the teleop device or the action from the policy
        executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # Check for intervention from transition info
        intervention_info = new_transition[TransitionKey.INFO]
        _is_interv_now = bool(intervention_info.get(TeleopEvents.IS_INTERVENTION, False))
        # Flush chunk queue on BOTH intervention edges for chunk-aware policies
        # (e.g. QC). Rising edge (False -> True): user just grabbed control;
        # drop the autonomous chunk so the next teleop action is applied
        # cleanly. Falling edge (True -> False): user released; drop the
        # stale chunk that was generated mid-intervention against outdated
        # state, force regeneration from the current (post-intervention)
        # observation. Gated by cfg flag so SAC default behavior is unchanged.
        _edge_rising = _is_interv_now and not prev_intervention_flag
        _edge_falling = (not _is_interv_now) and prev_intervention_flag
        if (
            (_edge_rising or _edge_falling)
            and getattr(cfg.policy, "flush_chunk_on_intervention", False)
            and hasattr(policy, "reset")
        ):
            policy.reset()
        prev_intervention_flag = _is_interv_now
        if _is_interv_now:
            episode_intervention = True
            episode_intervention_steps += 1
        # Latch manual SUCCESS (Triangle). Used to gate the learner push at
        # episode end: only successful episodes contribute to training.
        if intervention_info.get(TeleopEvents.SUCCESS, False):
            episode_success = True
        # Stage-advance counter (each gamepad stage-button press during the ep).
        if intervention_info.get(TeleopEvents.STAGE_ADVANCE, False):
            episode_stage_advances += 1
        # Track SARM progress (set by SARMRewardProcessorStep regardless of reward_mode).
        sp = intervention_info.get("sarm_progress")
        if sp is not None:
            try:
                last_sarm_progress = float(sp)
                if last_sarm_progress > max_sarm_progress:
                    max_sarm_progress = last_sarm_progress
            except (TypeError, ValueError):
                pass

        complementary_info = {
            "discrete_penalty": torch.tensor(
                [new_transition[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty", 0.0)]
            ),
            # Propagate intervention flag to learner so the dual-buffer mode
            # can route teleop frames into the offline (intervention-only)
            # replay buffer. Without this, all transitions land in the
            # online buffer regardless of whether the human took control.
            # Use the enum's .value (string) — torch.save/load with
            # weights_only=True can strip non-tensor objects like Enum keys.
            TeleopEvents.IS_INTERVENTION.value: torch.tensor(
                [1 if bool(_is_interv_now) else 0], dtype=torch.int8
            ),
        }
        # DSRL: store the noise vector used to generate the action
        if hasattr(policy, "_last_noise") and policy._last_noise is not None:
            complementary_info["noise"] = policy._last_noise.squeeze(0)

        # Create transition for learner (convert to old format)
        list_transition_to_send_to_learner.append(
            Transition(
                state=observation,
                action=executed_action,
                reward=reward,
                next_state=next_observation,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
        )

        # Update transition for next iteration
        transition = new_transition

        if done or truncated:
            logging.info(
                f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode} "
                f"success={episode_success} steps={episode_total_steps}"
            )

            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

            # DSRL n-step returns: rewrite each transition's reward/next_state/done
            # to n-step versions before pushing. We have the full episode here.
            _dsrl_n = int(getattr(cfg.policy, "n_step", 1) or 1)
            if (
                getattr(cfg.policy, "type", None) == "dsrl_ext"
                and _dsrl_n > 1
                and len(list_transition_to_send_to_learner) > 0
            ):
                list_transition_to_send_to_learner = _apply_nstep_returns(
                    list_transition_to_send_to_learner,
                    n=_dsrl_n,
                    gamma=float(getattr(cfg.policy, "discount", 0.99)),
                )

            if len(list_transition_to_send_to_learner) > 0:
                push_transitions_to_transport_queue(
                    transitions=list_transition_to_send_to_learner,
                    transitions_queue=transitions_queue,
                )
                list_transition_to_send_to_learner = []

            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # Calculate intervention rate
            intervention_rate = 0.0
            if episode_total_steps > 0:
                intervention_rate = episode_intervention_steps / episode_total_steps

            # Per-episode gripper diagnostics (residual mode).
            grip_metrics = {}
            if _residual_mode and episode_total_steps > 0:
                grip_metrics = {
                    "Gripper base mean": grip_base_sum / episode_total_steps,
                    "Gripper combined mean": grip_combined_sum / episode_total_steps,
                    "Gripper residual mean": grip_residual_signed_sum / episode_total_steps,
                    "Gripper residual abs mean": grip_residual_abs_sum / episode_total_steps,
                    "Gripper band-flip rate": grip_flip_count / episode_total_steps,
                }

            # Rolling moving avgs over last N completed episodes.
            rolling_rewards.append(float(sum_reward_episode))
            rolling_successes.append(int(episode_success))
            rolling_stage_advances.append(int(episode_stage_advances))
            for _buf in (rolling_rewards, rolling_successes, rolling_stage_advances):
                if len(_buf) > _ROLLING_N:
                    _buf.pop(0)
            _roll_reward = sum(rolling_rewards) / max(len(rolling_rewards), 1)
            _roll_success = sum(rolling_successes) / max(len(rolling_successes), 1)
            _roll_stage = sum(rolling_stage_advances) / max(len(rolling_stage_advances), 1)

            # Send episodic reward to the learner
            interactions_queue.put(
                python_object_to_bytes(
                    {
                        "Episodic reward": sum_reward_episode,
                        "Episode terminal SARM progress": last_sarm_progress,
                        "Episode max SARM progress": max_sarm_progress,
                        "Interaction step": interaction_step,
                        "Episode intervention": int(episode_intervention),
                        "Episode success": int(episode_success),
                        "Episode stage advances": episode_stage_advances,
                        "Episode discarded total": discarded_episodes,
                        "Intervention rate": intervention_rate,
                        "Rolling reward (last 5)": _roll_reward,
                        "Rolling success (last 5)": _roll_success,
                        "Rolling stage advances (last 5)": _roll_stage,
                        **grip_metrics,
                        **stats,
                    }
                )
            )

            # Reset intervention counters and environment
            sum_reward_episode = 0.0
            episode_intervention = False
            episode_success = False
            episode_stage_advances = 0
            episode_intervention_steps = 0
            episode_total_steps = 0
            last_sarm_progress = 0.0
            max_sarm_progress = 0.0
            grip_base_sum = 0.0
            grip_combined_sum = 0.0
            grip_residual_signed_sum = 0.0
            grip_residual_abs_sum = 0.0
            grip_flip_count = 0

            # Reset environment and processors
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()
            # Reset main policy (chunk-aware policies need to flush their
            # action queue at episode boundaries). SAC.reset() is no-op.
            if hasattr(policy, "reset"):
                policy.reset()
            prev_intervention_flag = False

            # Process initial observation
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

            # Residual mode: clear ACT's chunk queue and attach fresh base_action.
            if _residual_mode and _base_policy is not None:
                _base_policy.reset()
                _ba = _base_act(transition[TransitionKey.OBSERVATION])
                transition[TransitionKey.OBSERVATION]["observation.base_action"] = _ba

        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            precise_sleep(max(1 / cfg.env.fps - dt_time, 0.0))


#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions


def update_policy_parameters(policy: SACPolicy, parameters_queue: Queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        # Plugins (e.g. QCPolicy) with no monolithic `.actor` submodule opt
        # into this path by defining `load_actor_state_dict(payload)`. The
        # payload shape is plugin-defined (typically a {name: state_dict}
        # nested dict, mirroring `get_actor_state_dict`). SAC keeps the
        # untouched `policy.actor.load_state_dict(...)` path below.
        if hasattr(policy, "load_actor_state_dict"):
            policy.load_actor_state_dict(actor_state_dict)
        else:
            policy.actor.load_state_dict(actor_state_dict)

        # Load discrete critic if present
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


#  Utilities functions


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
