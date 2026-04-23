#!/usr/bin/env python
"""Find sim-env workspace bounds by teleoperating through the task workspace.

Sim counterpart to lerobot-panda's find_workspace_bounds.py. Uses the same
GymManipulatorConfig schema as lerobot.rl.gym_manipulator, builds the env
and teleop, and tracks min/max end-effector position during a teleop
session. On exit, prints the bounds with a small margin in a form that
can be pasted into the env JSON or passed on the command line.

Stop with Ctrl+C. See docs/port/2026-04-22-sim-hilserl-commands.md for
usage examples.
"""

import logging
import signal
import time
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.rl.gym_manipulator import (
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.constants import OBS_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_stop = False


def _handle_sigint(_sig, _frame):
    global _stop
    _stop = True


@dataclass
class FindWorkspaceBoundsConfig:
    """Top-level config used by ``parser.wrap`` for this script."""

    env: HILSerlRobotEnvConfig
    device: str = "cpu"
    # 0 means forever (Ctrl+C to stop)
    duration_s: float = 0.0
    # Fraction of range to pad the reported bounds
    margin: float = 0.05
    # Start / end indices of EE xyz inside the flat observation.state vector.
    # sim_assembling adapter emits [joint_pos(7), ee_pos(3), ee_quat(4), gripper(1)]
    ee_pos_start: int = 7
    ee_pos_end: int = 10


def _read_ee_xyz(transition: dict, lo: int, hi: int) -> np.ndarray:
    obs = transition[TransitionKey.OBSERVATION.value]
    state = obs.get(OBS_STATE)
    if state is None:
        raise KeyError(f"{OBS_STATE!r} missing from observation. Keys: {list(obs)}")
    arr = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
    arr = arr.reshape(-1)
    return arr[lo:hi].copy()


@parser.wrap()
def main(cfg: FindWorkspaceBoundsConfig) -> None:
    signal.signal(signal.SIGINT, _handle_sigint)

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("\n" + "=" * 60)
    print("  SIM WORKSPACE BOUNDS FINDER")
    print("=" * 60)
    print("  Hold R1 + move with left/right sticks to visit the full workspace.")
    print("  Cover every corner you want the policy to be allowed to reach.")
    print(f"  Stop with Ctrl+C{' (auto-stops in %.1f s)' % cfg.duration_s if cfg.duration_s > 0 else ''}.\n")

    fps = cfg.env.fps
    dt = 1.0 / float(fps)

    # Prime the pipeline with a reset.
    obs, info = env.reset()
    transition = create_transition(observation=obs, info=info or {}, complementary_data={})
    transition = env_processor(transition)

    # Size neutral action from the env's action space (works for 4D or 5D).
    act_dim = int(env.action_space.shape[0])
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True
    neutral_action = torch.zeros(act_dim, dtype=torch.float32)
    if use_gripper:
        neutral_action[-1] = 1.0

    ee_min = np.full(3, np.inf)
    ee_max = np.full(3, -np.inf)
    n_samples = 0
    t_start = time.perf_counter()

    try:
        while not _stop:
            step_start = time.perf_counter()

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )

            ee = _read_ee_xyz(transition, int(cfg.ee_pos_start), int(cfg.ee_pos_end))
            ee_min = np.minimum(ee_min, ee)
            ee_max = np.maximum(ee_max, ee)
            n_samples += 1

            if n_samples % max(1, int(fps) * 2) == 0:
                print(
                    f"\r  n={n_samples:5d}  "
                    f"min=[{ee_min[0]:+.4f} {ee_min[1]:+.4f} {ee_min[2]:+.4f}]  "
                    f"max=[{ee_max[0]:+.4f} {ee_max[1]:+.4f} {ee_max[2]:+.4f}]",
                    end="",
                    flush=True,
                )

            if cfg.duration_s > 0 and (time.perf_counter() - t_start) >= cfg.duration_s:
                break

            # Respect env fps only in realtime mode. Fast mode doesn't need extra sleep.
            if getattr(cfg.env, "realtime", True):
                precise_sleep(max(dt - (time.perf_counter() - step_start), 0.0))
    except (KeyboardInterrupt, EOFError):
        pass

    print("\n\n" + "=" * 60)
    print("  WORKSPACE BOUNDS RESULTS")
    print("=" * 60)
    print(f"  samples:    {n_samples}")
    print(f"  observed min: [{ee_min[0]:.4f}, {ee_min[1]:.4f}, {ee_min[2]:.4f}]")
    print(f"  observed max: [{ee_max[0]:.4f}, {ee_max[1]:.4f}, {ee_max[2]:.4f}]")
    rng = ee_max - ee_min
    print(f"  range:        [{rng[0]:.4f}, {rng[1]:.4f}, {rng[2]:.4f}]")
    print(f"  centre:       [{(ee_min[0]+ee_max[0])/2:.4f}, {(ee_min[1]+ee_max[1])/2:.4f}, {(ee_min[2]+ee_max[2])/2:.4f}]")

    pad = rng * float(cfg.margin)
    b_min = ee_min - pad
    b_max = ee_max + pad
    print(f"\n  suggested (with {cfg.margin*100:.0f} pct margin):")
    print(f"    min: [{b_min[0]:.4f}, {b_min[1]:.4f}, {b_min[2]:.4f}]")
    print(f"    max: [{b_max[0]:.4f}, {b_max[1]:.4f}, {b_max[2]:.4f}]")
    print("")
    print("  paste into your env JSON under env.processor.inverse_kinematics:")
    print("    \"inverse_kinematics\": {")
    print(f"        \"end_effector_bounds\": {{")
    print(f"            \"min\": [{b_min[0]:.4f}, {b_min[1]:.4f}, {b_min[2]:.4f}],")
    print(f"            \"max\": [{b_max[0]:.4f}, {b_max[1]:.4f}, {b_max[2]:.4f}]")
    print(f"        }}")
    print("    }")
    print("=" * 60)

    if teleop_device is not None:
        try:
            teleop_device.disconnect()
        except Exception:
            pass
    env.close()


if __name__ == "__main__":
    main()
