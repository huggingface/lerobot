"""Sequential multi-model ACT orchestration on UR10 (PCB-then-top assembly).

Drives the UR10 through a sequence of ACT v2 policies on a single env. Each
stage runs until the operator signals via PS4 gamepad:

  Triangle (SUCCESS)         → advance to the next stage
  Cross / Square (FAILURE / RERECORD) → restart the pipeline from stage 0

After the LAST stage completes (regardless of outcome), the pipeline restarts
from stage 0 as well. This matches the user-described flow for the PCB
assembly task:

  home → M1 (place PCB onto bottom enclosure) → user verdict
       └─ if SUCCESS → home → M2 (place top enclosure onto PCB) → user verdict
                              └─ any verdict → restart from home → M1
       └─ if FAILURE → restart from home → M1

Why ``py_trees``
================
The pipeline is a `py_trees.composites.Sequence` with `memory=True`. When a
memory Sequence returns SUCCESS or FAILURE, py_trees re-initialises its
children on the next tick — i.e. the Sequence automatically restarts from
child 0. So the entire restart loop is captured by a plain
``while True: tree.tick()`` without any custom decorator.

Adding a third stage (e.g. labelling) = append one ``Stage(...)`` to the
``STAGES`` list. Adding branching (e.g. "if PCB rotation is wrong, run a
re-pose policy before M2") = wrap the relevant subtree in a
``py_trees.composites.Selector``. Adding parallel monitoring (e.g. a watchdog
that aborts on a force spike) = wrap in ``py_trees.composites.Parallel``.

Install
=======
    pip install py_trees

(Not added to pyproject.toml — kept as an optional dep until we commit to
this scaffolding for the longer term.)

Usage
=====
    python act_train/eval_ur10_act_v2_pipeline.py

Tune the ``STAGES`` list and constants at the top of the file. End the run
with Ctrl+C.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import draccus
import py_trees
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.robots import rc10 as _rc10_register  # noqa: F401  # registers RC10 cfg subclass
from lerobot.robots import ur10 as _ur10_register  # noqa: F401  # registers UR10 cfg subclass
from lerobot.rl.gym_manipulator import GymManipulatorConfig, make_processors, make_robot_env
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.robot_utils import precise_sleep

from _ur10_reset import auto_reset_to_home  # sibling module in act_train/

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# User-tunable configuration
# =============================================================================

@dataclass(frozen=True)
class Stage:
    """One ACT policy in the pipeline.

    Attributes:
        name: Operator-facing label (shown in the tree banner + per-step logs).
        model_dir: Path to the checkpoint directory containing ``model.safetensors``,
            ``config.json``, etc. (the output of ``policy.save_pretrained``).
        dataset_repo_id: The dataset the policy was trained on — used to load
            normalization statistics via ``LeRobotDatasetMetadata``.
    """
    name: str
    model_dir: str
    dataset_repo_id: str


STAGES: list[Stage] = [
    Stage(
        name="M1 — PCB onto bottom enclosure",
        model_dir="outputs/act/ur10/pcb_act_3cams_yaw_v2_60eps_30eps2/step_10000",
        dataset_repo_id="local/pcb_act_3cams_yaw_v2_60eps",
    ),
    Stage(
        name="M2 — top enclosure onto PCB",
        model_dir="outputs/act/ur10/pcb_act_3cams_yaw_v2_60eps_30eps_top/last",
        dataset_repo_id="local/pcb_act_3cams_yaw_v2_60eps_top",
    ),
]

CONFIG_PATH = "src/lerobot/rl/ur10_env_3cams_yaw_v2.json"
EPISODE_TIME_S = 60          # per-stage safety timeout (s); auto-fail if no user input by then
RESET_TIME_S = 5            # auto_reset_to_home budget per HomeReset
RESET_SPEED_MPS = 0.1        # auto-reset linear velocity, m/s
FPS = 30                     # must equal cfg.env.fps
DEVICE = "cuda"              # torch device for policy inference

# Show the tree state under each tick (verbose). Set True only when debugging the BT,
# otherwise it spams the log at FPS.
TREE_SNAPSHOT_PER_TICK = False
# =============================================================================


# =============================================================================
# Custom py_trees behaviours
# =============================================================================

class HomeReset(py_trees.behaviour.Behaviour):
    """Drive the wrist to the home pose via ``auto_reset_to_home`` (blocking).

    The function takes the full ``RESET_TIME_S`` budget (motion + hold). We
    don't try to do anything "tickable" here — the moment py_trees enters this
    node we issue the reset, wait for it to complete, then return SUCCESS.

    Side effects (from ``auto_reset_to_home``):
      - Wrist physically moved to home (xyz + fixed_rx/ry/rz).
      - ``env.target_xyz`` latched to achieved home.
      - ``env.target_yaw`` zeroed (phase-1.5 fix — required so the next policy
        run doesn't compose a stale yaw offset on its first step).
      - ``env.robot.capture_baselines()`` re-anchors the per-episode xyz +
        rotation baselines used by the v2 11-D observation.
      - Gripper opened.
    """

    def __init__(self, name: str, env, dt: float, reset_time_s: float,
                 reset_speed_mps: float, fps: int):
        super().__init__(name=name)
        self.env = env
        self.dt = dt
        self.reset_time_s = reset_time_s
        self.reset_speed_mps = reset_speed_mps
        self.fps = fps
        self._done = False

    def initialise(self) -> None:
        # Called every time py_trees (re-)enters this node. Reset our latch so
        # the reset call runs once per entry, not once for the whole script.
        self._done = False
        self.logger.info("home reset: driving to home")

    def update(self) -> py_trees.common.Status:
        if not self._done:
            auto_reset_to_home(
                self.env, self.dt, self.reset_time_s, self.reset_speed_mps, self.fps
            )
            self._done = True
            self.logger.info("home reset: at home, baselines re-anchored")
        return py_trees.common.Status.SUCCESS


class RunACTPolicy(py_trees.behaviour.Behaviour):
    """Run one ACT policy until the operator presses a gamepad button.

    Loads the policy + processors ONCE at construction (so all stages are
    GPU-resident before the pipeline starts ticking). ``initialise()`` resets
    per-run counters and the policy's internal chunk queue / temporal
    ensembler. Each ``update()`` runs one inference step (~1/FPS seconds).

    Returns:
        SUCCESS  — operator pressed Triangle (TeleopEvents.SUCCESS).
        FAILURE  — operator pressed Cross or Square (TeleopEvents.TERMINATE_EPISODE
                   set; we don't distinguish FAILURE from RERECORD in deployment).
        FAILURE  — episode timeout (operator forgot to press a button).
        RUNNING  — otherwise.
    """

    def __init__(
        self,
        name: str,
        model_dir: str,
        dataset_repo_id: str,
        env,
        env_processor,
        teleop_device,
        device: torch.device,
        episode_time_s: float,
        fps: int,
    ):
        super().__init__(name=name)

        # Eager load — fail loudly at construction if the checkpoint or stats
        # are missing, rather than mid-pipeline.
        self.policy: ACTPolicy = ACTPolicy.from_pretrained(model_dir)
        self.policy.eval()
        self.policy.to(device)
        self.metadata = LeRobotDatasetMetadata(dataset_repo_id)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_dir,
            dataset_stats=self.metadata.stats,
        )

        # Refs we don't own — provided by the caller, shared across stages.
        self.env = env
        self.env_processor = env_processor
        self.teleop_device = teleop_device
        self.device = device
        self.episode_time_s = episode_time_s
        self.fps = fps

        # Per-run state (re-initialised in initialise()).
        self._step = 0
        self._start_time = 0.0
        # Cache the set of obs keys the policy actually expects — saves
        # rebuilding the filter dict every step.
        self._expected_keys = set(self.policy.config.input_features.keys())

    def initialise(self) -> None:
        self._step = 0
        self._start_time = time.perf_counter()
        self.policy.reset()  # clear ACT chunk queue + temporal ensembler
        # py_trees' self.logger.info() takes a single pre-formatted string (NOT
        # the printf-style %-args that stdlib logging supports). Use f-strings
        # consistently to keep this script working with py_trees' Logger.
        self.logger.info(f"--- Activating policy: {self.name} ---")
        self.logger.info(
            f"  chunk_size={self.policy.config.chunk_size}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"temporal_ensemble_coeff={self.policy.config.temporal_ensemble_coeff}"
        )

    def update(self) -> py_trees.common.Status:
        # 1. Read the policy's input from the env.
        #    - Images come through env_processor (crop + resize + batch + device).
        #    - State is overridden with the v2 11-D ACT vector from
        #      get_act_observation, mirroring how the training dataset was built
        #      (see act_train/record_ur10_act_v2.py::_override_state_with_v2).
        obs_batch = self._build_obs_for_policy()
        obs_batch = self.preprocess(obs_batch)

        # 2. Predict → unnormalize.
        with torch.no_grad():
            action_tensor = self.policy.select_action(obs_batch)
        action_tensor = self.postprocess(action_tensor)

        # 3. Drive the robot.
        action_dict = make_robot_action(action_tensor, self.metadata.features)
        self.env.set_act_target(action_dict)
        self._step += 1

        # 4. Periodic per-step log (every 30 steps = 1 s).
        if self._step % 30 == 0:
            tx = float(action_dict.get("x.pos", 0.0))
            ty = float(action_dict.get("y.pos", 0.0))
            tz = float(action_dict.get("z.pos", 0.0))
            tyaw = float(action_dict.get("yaw.pos", 0.0))
            tg = float(action_dict.get("gripper.pos", 0.0))
            self.logger.info(
                f"  step {self._step}  target=[x={tx:+.4f} y={ty:+.4f} "
                f"z={tz:+.4f} yaw={tyaw:+.4f} g={tg:.2f}]"
            )

        # 5. Check operator input + safety timeout.
        events = self.teleop_device.get_teleop_events()
        if events.get(TeleopEvents.SUCCESS, False):
            elapsed = time.perf_counter() - self._start_time
            self.logger.info(
                f"{self.name}: SUCCESS (Triangle) at step {self._step} ({elapsed:.1f}s)"
            )
            return py_trees.common.Status.SUCCESS
        if events.get(TeleopEvents.TERMINATE_EPISODE, False):
            # TERMINATE_EPISODE fires for both FAILURE (Cross) and RERECORD
            # (Square). In deployment both mean "this run is bad; restart" —
            # we don't distinguish.
            elapsed = time.perf_counter() - self._start_time
            self.logger.info(
                f"{self.name}: FAILURE / RERECORD at step {self._step} ({elapsed:.1f}s)"
            )
            return py_trees.common.Status.FAILURE
        if self._step >= int(self.episode_time_s * self.fps):
            self.logger.warning(
                f"{self.name}: timeout after {self._step} steps "
                f"({self._step / self.fps:.1f}s) — operator didn't press a button; restarting"
            )
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING

    def _build_obs_for_policy(self) -> dict:
        """Pull one fresh observation tailored to the policy's input schema.

        Identical pattern to ``eval_ur10_act_v2.py::_build_obs_for_policy``:
          1. Raw obs via ``env._augment_observation(env.robot.get_observation())`` —
             the env's 17-D HIL-SERL state + raw camera frames.
          2. Run through ``env_processor`` for the image-crop + resize +
             batch + device-move pipeline. State is still 17-D at this point.
          3. Replace the state slot with the v2 11-D ACT state from
             ``env.get_act_observation``. Cropped images are kept untouched.
          4. Filter to the policy's declared input features so we don't pass
             extra keys the normalizer doesn't expect.
        """
        raw = self.env._augment_observation(self.env.robot.get_observation())
        tr = self.env_processor(create_transition(
            observation=raw, info={TeleopEvents.IS_INTERVENTION: False},
        ))
        v2_state = self.env.get_act_observation()["agent_pos"]
        # env_processor's AddBatchDimensionProcessorStep emits shape (1, D);
        # match it so dataset / normalizer indexing stays uniform across frames.
        tr[TransitionKey.OBSERVATION][OBS_STATE] = (
            torch.from_numpy(v2_state.copy()).unsqueeze(0).to(self.device).float()
        )
        return {
            k: v for k, v in tr[TransitionKey.OBSERVATION].items()
            if k in self._expected_keys
        }


# =============================================================================
# Tree assembly
# =============================================================================

def build_tree(
    env,
    env_processor,
    teleop_device,
    device: torch.device,
    stages: list[Stage] = STAGES,
) -> py_trees.composites.Sequence:
    """Assemble the pipeline as a memory-Sequence: [Home, M1, Home, M2, ...].

    A ``memory=True`` Sequence remembers which child is currently RUNNING
    across ticks. When a child returns SUCCESS the Sequence advances to the
    next child; when one returns FAILURE the Sequence aborts and returns
    FAILURE. On any terminal status, py_trees calls ``initialise()`` on the
    Sequence at the next tick — which restarts it from child 0. That's
    exactly the user-described restart loop, with no custom decorator needed.
    """
    children: list[py_trees.behaviour.Behaviour] = []
    for stage in stages:
        children.append(HomeReset(
            name=f"home_reset_pre :: {stage.name}",
            env=env,
            dt=1.0 / FPS,
            reset_time_s=RESET_TIME_S,
            reset_speed_mps=RESET_SPEED_MPS,
            fps=FPS,
        ))
        children.append(RunACTPolicy(
            name=stage.name,
            model_dir=stage.model_dir,
            dataset_repo_id=stage.dataset_repo_id,
            env=env,
            env_processor=env_processor,
            teleop_device=teleop_device,
            device=device,
            episode_time_s=EPISODE_TIME_S,
            fps=FPS,
        ))
    return py_trees.composites.Sequence(
        name="PCB assembly pipeline",
        memory=True,
        children=children,
    )


# =============================================================================
# Main
# =============================================================================

def _validate_stages_exist(stages: list[Stage]) -> None:
    """Fail fast at startup if any stage's checkpoint dir is missing."""
    missing: list[str] = []
    for stage in stages:
        if not Path(stage.model_dir).exists():
            missing.append(f"  - {stage.name}: {stage.model_dir}")
    if missing:
        raise FileNotFoundError(
            "Cannot launch pipeline — the following model directories are missing:\n"
            + "\n".join(missing)
            + "\n\nIf a model is still training, wait for the 'last/' checkpoint "
            "to appear (or update STAGES at the top of this script to point "
            "elsewhere)."
        )


def main() -> None:
    _validate_stages_exist(STAGES)

    # ---- env + processors + teleop -----------------------------------------
    with open(CONFIG_PATH) as f:
        raw_cfg = json.load(f)
    cfg = draccus.decode(GymManipulatorConfig, raw_cfg)
    assert cfg.env.fps == FPS, (
        f"FPS constant ({FPS}) must match cfg.env.fps ({cfg.env.fps})"
    )

    device = torch.device(DEVICE)

    env, teleop_device = make_robot_env(cfg.env)
    env_processor, _action_processor = make_processors(env, teleop_device, cfg.env, str(device))
    env.reset()
    env_processor.reset()

    # ---- build the tree ----------------------------------------------------
    root = build_tree(env, env_processor, teleop_device, device)
    tree = py_trees.trees.BehaviourTree(root=root)
    if TREE_SNAPSHOT_PER_TICK:
        tree.add_visitor(py_trees.visitors.SnapshotVisitor())

    # ---- show the pipeline structure to the operator -----------------------
    print()
    print("=" * 72)
    print("PIPELINE STRUCTURE")
    print("=" * 72)
    print(py_trees.display.unicode_tree(root))
    print("=" * 72)
    print("Gamepad: Triangle = SUCCESS (advance / restart) | "
          "Cross or Square = FAILURE (restart from M1)")
    print("Stage timeout: %.0f s. Ctrl+C to exit." % EPISODE_TIME_S)
    print("=" * 72)
    print()

    # ---- run forever; the Sequence auto-restarts on terminal status --------
    dt = 1.0 / FPS
    n_pipeline_completions = 0
    last_root_status = py_trees.common.Status.INVALID
    try:
        while True:
            t0 = time.perf_counter()
            tree.tick()

            # Track pipeline completions (root SUCCESS or FAILURE) for the operator's
            # awareness — a FAILURE means M1 was aborted or M2 hit any outcome; either
            # way, the next tick restarts from M1.
            if (
                root.status != last_root_status
                and root.status in (py_trees.common.Status.SUCCESS, py_trees.common.Status.FAILURE)
            ):
                n_pipeline_completions += 1
                logger.info(
                    "Pipeline iteration #%d finished with status %s — restarting from stage 0",
                    n_pipeline_completions, root.status.name,
                )
            last_root_status = root.status

            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user (Ctrl+C).")
    except Exception:
        logger.exception("Pipeline crashed")
    finally:
        logger.info("Completed %d pipeline iterations", n_pipeline_completions)
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
