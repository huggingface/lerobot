# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""A complete third-party rollout strategy: an unattended patrol that labels its own data.

``lerobot-rollout`` dispatches ``--strategy.type=<name>`` through an open registry, so a
strategy that lives entirely outside LeRobot is a first-class citizen: it is handed the
dataset, the inference backend, cadence reporting, ``--interactive`` and the text-query
channel without a single edit to LeRobot.  This module is the smallest *interesting*
example of one, in two classes:

* :class:`PatrolStrategyConfig` — registered as ``patrol``.  Its ClassVars and hooks
  *declare* what the engine must arrange on the strategy's behalf (a dataset, streaming
  encoding, one extra dataset column).  Nothing outside the strategy ever checks its
  concrete type, which is why a third party gets the same treatment as a built-in.
* :class:`PatrolStrategy` — the real-time control loop: run the policy in fixed-length
  laps and record every frame, tagged with the lap it belongs to.

Both live in this single module; the package ``__init__`` next door only re-exports them,
and it is the ``PatrolStrategy`` re-export that lets the factory find the class.  That
file explains why the example has to be a package and not a lone ``.py`` file.

The tagging is the part that is *impossible* to bolt on from outside: ``lap`` becomes a
real column of the recorded ``LeRobotDataset`` only because ``extra_dataset_features()``
is merged into the features the dataset is *created* with.  ``--resume`` cannot add a
column — a resumed dataset's schema is whatever its on-disk metadata says — so resuming
only works on a dataset that already carries ``lap``; ``build_rollout_context`` checks
that up front and refuses otherwise.  A patrol of a shelf, a conveyor or a corridor
drifts as it repeats, so the lap index is what lets you slice the result afterwards —
train on ``lap <= 3``, audit the late laps, or compare the same waypoint across passes.
It is not recoverable from ``episode_index``: episodes hold ``laps_per_episode`` laps
each, and an ``--interactive`` segment can stop mid-lap.

Running it
----------
This in-repo copy, no installation, from the repository root::

    PYTHONPATH=. lerobot-rollout \\
        --strategy.discover_packages_path=examples.rollout.custom_strategy \\
        --strategy.type=patrol \\
        --strategy.laps_per_episode=3 \\
        --policy.path=${HF_USER}/my_policy \\
        --robot.type=so100_follower --robot.port=/dev/ttyACM0 \\
        --dataset.repo_id=${HF_USER}/rollout_patrol \\
        --dataset.episode_time_s=30 \\
        --task="patrol the shelf" --duration=600

``--strategy.discover_packages_path`` imports that *package* and every module in it before
draccus parses the CLI, which is what makes ``--strategy.type=patrol`` and the
``--strategy.*`` flags below exist at all.  ``PYTHONPATH=.`` is needed because a console
script puts its own ``bin/`` directory on ``sys.path``, not the current directory.

As a real distribution, drop the flag entirely: name the package (and the distribution)
``lerobot_strategy_patrol`` — the ``lerobot_strategy_`` prefix with **underscores**, since
the distribution name is imported verbatim, so a hyphenated one matches no prefix and is
skipped with a warning — and ``register_third_party_plugins()`` imports it for you before
the CLI is parsed::

    pip install lerobot_strategy_patrol
    lerobot-rollout --strategy.type=patrol ...   # nothing else changes

Because this strategy declares ``supports_interactive``, it also runs under
``--interactive=true``, where ``run()`` is called once per ``/start`` … ``/stop`` segment
and the laps keep counting across them.

See ``docs/source/bring_your_own_rollout_strategies.mdx`` for the full contract.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rollout import (
    InferenceEngine,
    RolloutContext,
    RolloutStrategy,
    RolloutStrategyConfig,
    safe_push_to_hub,
    send_next_action,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import build_dataset_frame

logger = logging.getLogger(__name__)

# The name of the extra dataset column.  Declared once, used by both the config hook and
# the recording call, so the two can never drift apart.
LAP = "lap"


@RolloutStrategyConfig.register_subclass("patrol")
@dataclass
class PatrolStrategyConfig(RolloutStrategyConfig):
    """Unattended autonomous patrol, recording every frame tagged with its lap index."""

    # --- Capability declarations -------------------------------------------------------
    # ``RolloutConfig.__post_init__`` (validation) and ``build_rollout_context`` (wiring)
    # read these instead of checking the strategy's type, so they hold for a third-party
    # strategy exactly as they do for a built-in.
    supports_interactive: ClassVar[bool] = True  # run() is restartable — see PatrolStrategy
    records_data: ClassVar[bool] = True  # ctx.data.dataset is created for us
    requires_dataset: ClassVar[bool] = True  # --dataset.repo_id becomes mandatory
    # requires_teleop stays False: a patrol has no human in the loop.

    # --- Fields: each one becomes a ``--strategy.<field>`` CLI flag --------------------
    # Seconds of robot motion per lap.  None resolves to --dataset.episode_time_s below.
    lap_duration_s: float | None = None
    # Laps per recorded episode.  Keep it > 1 so the ``lap`` column carries information
    # that ``episode_index`` does not.
    laps_per_episode: int = 5

    def requires_streaming_encoding(self) -> bool:
        """Always: this strategy calls ``add_frame`` from inside the timed control loop.

        Without streaming encoding, finishing an episode encodes its video on the control
        thread — the loop misses its deadline for seconds, the robot hitches, and the
        dataset's real cadence ends up nowhere near its declared fps.  Declaring it here
        makes ``RolloutConfig`` force ``--dataset.streaming_encoding=true`` on our behalf.
        """
        return True

    def extra_dataset_features(self) -> dict[str, dict]:
        """Add the ``lap`` column to the recorded dataset.

        ``build_rollout_context`` merges this on top of the features derived from the
        robot and the policy — a name that collides with one of those is an error — and
        creates the dataset with the merged schema.  On ``--resume`` it cannot do that:
        ``LeRobotDataset.resume`` takes no features and reads its schema from the
        dataset's on-disk metadata, so resuming requires a dataset that *already* has
        ``lap``, and the context raises when the existing one does not.

        The declaration is a contract in both directions: ``validate_frame`` rejects a
        frame that is missing ``lap`` just as hard as one carrying an unknown key, so
        *every* recorded frame must supply it.
        """
        return {LAP: {"dtype": "int64", "shape": (1,), "names": None}}

    def resolve_defaults(self, dataset_cfg: DatasetRecordConfig | None) -> None:
        """Fill ``lap_duration_s`` from ``--dataset.episode_time_s`` when it was left unset.

        Called once, after the capability checks have passed, which is the only point at
        which the dataset config is known to exist.  Raise ``ValueError`` for anything
        that cannot be resolved — the CLI then fails before any hardware is touched.
        Only the dataset config's *own* flags are readable here: ``RolloutConfig`` runs
        this hook before it loads the policy and before it propagates ``--task`` into
        ``dataset.single_task``, so that field is still empty at this point whenever the
        user set the task with ``--task`` rather than ``--dataset.single_task``.
        """
        if self.laps_per_episode < 1:
            raise ValueError(f"--strategy.laps_per_episode must be >= 1, got {self.laps_per_episode}")
        if self.lap_duration_s is None:
            # ``requires_dataset`` guarantees a dataset config here; be explicit anyway,
            # so a future capability change fails loudly instead of crashing on None.
            if dataset_cfg is None:
                raise ValueError(
                    "patrol needs --strategy.lap_duration_s or --dataset.episode_time_s to be set"
                )
            self.lap_duration_s = dataset_cfg.episode_time_s
            logger.info("lap_duration_s not set — using --dataset.episode_time_s=%s", self.lap_duration_s)
        if self.lap_duration_s <= 0:
            raise ValueError(f"--strategy.lap_duration_s must be > 0, got {self.lap_duration_s}")


class PatrolStrategy(RolloutStrategy):
    """Autonomous laps with always-on recording, every frame tagged with its lap index.

    ``setup()`` is inherited: the base implementation attaches and starts the inference
    engine, which is what makes ``self.engine`` and ``self.interpolator`` usable.  An
    override must call *exactly one* of ``super().setup(ctx)`` or ``self._init_engine(ctx)``
    first — never both, since ``_init_engine`` starts the engine.

    ``run()`` is restartable, as ``supports_interactive`` promises: it never finalizes the
    dataset (that is ``teardown()``'s job), saves at most a partial tail episode when a
    segment ends, and binds no keyboard or stdin listener — stdin belongs to the
    interactive command prompt.
    """

    config: PatrolStrategyConfig

    def __init__(self, config: PatrolStrategyConfig) -> None:
        super().__init__(config)
        # Cross-segment state lives on the instance, never in ``run()`` locals: under
        # ``--interactive`` the controller calls ``run()`` once per /start…/stop segment,
        # and both the lap label and the episode's fill level must survive that.  The
        # ``CycleTimer`` is deliberately the other way round — see ``run()``.
        self._lap = 1
        self._laps_in_episode = 0

    def run(self, ctx: RolloutContext) -> None:
        """Patrol until ``--duration`` expires, shutdown is requested, or the segment ends."""
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        features = ctx.data.dataset_features
        engine = self.engine
        interpolator = self.interpolator
        # ``resolve_defaults`` guaranteed this is a positive float.
        lap_duration_s = self.config.lap_duration_s

        # One timer per ``run()``, never hoisted onto the instance: its start-up exemption
        # is what absorbs the interpolator that ``reset_control_state()`` re-primes at
        # every /start, and each segment gets its own cadence report.
        timer = CycleTimer(cfg.fps, interpolator.multiplier, report=ctx.runtime.cadence_report)

        # Mandatory: async backends start paused, and the interactive controller pauses the
        # engine again at the end of every segment.
        engine.resume()

        start_time = time.perf_counter()
        lap_start = start_time
        logger.info(
            "Patrol started (lap=%.0fs, %d laps/episode, from lap %d)",
            lap_duration_s,
            self.config.laps_per_episode,
            self._lap,
        )

        try:
            while not ctx.runtime.shutdown_event.is_set():
                timer.tick(new_cycle=interpolator.needs_new_action())

                if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break

                # Observe every tick so the action post-processor sees a fresh frame; the
                # helper throttles the expensive part (processors + notify_observation) to
                # the ticks on which the interpolator actually needs a new action.
                with timer.section("observe"):
                    obs = robot.get_observation()
                with timer.section("process_obs"):
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                if self._handle_warmup(cfg.use_torch_compile, timer):
                    continue

                action_dict = send_next_action(obs_processed, obs, ctx, interpolator, timer)

                if action_dict is not None:
                    with timer.section("telemetry"):
                        self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                    # Record once per interpolation cycle so the dataset cadence matches
                    # its declared fps; interpolated ticks only command the robot.
                    if interpolator.emitted_policy_action:
                        with timer.section("record"):
                            self._record_frame(dataset, features, obs_processed, action_dict, engine)

                if (time.perf_counter() - lap_start) >= lap_duration_s:
                    self._finish_lap(dataset, timer)
                    # Restart the lap clock *after* a possible episode save, so the save
                    # does not eat into the next lap.
                    lap_start = time.perf_counter()

                # Service the text-query channel (/vqa answers, /autosteer turns) at the
                # end of the tick, after the frame is recorded: a multi-second generation
                # must not land between this tick's observation and its ``add_frame``, nor
                # sit inside the action path.  No-op when nothing is queued.
                with timer.section("query"):
                    engine.pump_query(obs_processed)

                timer.wait()
        finally:
            logger.info("Patrol segment ended (lap %d)", self._lap)
            # Report the cadence before touching the dataset, so a failing save cannot
            # swallow this segment's summary.
            timer.log_run_summary()
            self._save_partial_episode(dataset)

    def _record_frame(
        self,
        dataset: LeRobotDataset,
        features: dict,
        obs_processed: dict,
        action_dict: dict,
        engine: InferenceEngine,
    ) -> None:
        """Append one frame, carrying every column the config declared."""
        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
        frame = {
            **obs_frame,
            **action_frame,
            # The extra column, on *every* frame (``validate_frame`` rejects a missing
            # key) and as an ndarray of exactly the declared dtype and shape — a bare
            # Python int is rejected too.
            LAP: np.array([self._lap], dtype=np.int64),
            # ``dispatched_task``, not ``engine.task``: it is the instruction that produced
            # the action just sent, so a /subtask typed mid-chunk cannot mislabel actions
            # still queued from the previous one.
            "task": engine.dispatched_task,
        }
        dataset.add_frame(frame)

    def _finish_lap(self, dataset: LeRobotDataset, timer: CycleTimer) -> None:
        """Close the current lap and rotate the episode every ``laps_per_episode`` laps."""
        logger.info("Lap %d complete", self._lap)
        self._lap += 1
        self._laps_in_episode += 1
        if self._laps_in_episode < self.config.laps_per_episode:
            return

        # Reset before the early return below, so a rotation that saved nothing still
        # starts a fresh count of ``laps_per_episode`` instead of retrying every lap.
        self._laps_in_episode = 0
        if not dataset.has_pending_frames():
            # ``save_episode`` rejects an empty buffer, and a lap can legitimately record
            # nothing — a lap shorter than the warmup, or an async backend starved for its
            # whole duration.  Rotating on nothing is a no-op, not an error.
            logger.warning("No frames recorded in the last %d lap(s)", self.config.laps_per_episode)
            return

        dataset.save_episode()
        logger.info("Episode saved (total: %d)", dataset.num_episodes)
        # ``save_episode`` blocks for a good fraction of a second inside the timed loop
        # body.  That is episode finalisation, not the steady-state cadence: report the
        # episode, then drop the partial group and the gap the save opened.
        timer.log_episode_summary(f"episode {dataset.num_episodes}")
        timer.restart()

    def _save_partial_episode(self, dataset: LeRobotDataset) -> None:
        """Commit the segment's unfinished episode so no recorded frame is lost.

        Saving a partial tail is allowed in ``run()``; *finalizing* the dataset is not —
        the next ``/start`` would find it closed.
        """
        if not dataset.has_pending_frames():
            return
        logger.info("Saving the segment's final (partial) episode")
        dataset.save_episode()
        self._laps_in_episode = 0

    def teardown(self, ctx: RolloutContext) -> None:
        """Finalize the dataset, optionally push it, then stop inference and disconnect."""
        cfg = ctx.runtime.cfg
        dataset = ctx.data.dataset
        if dataset is not None:
            logger.info("Finalizing dataset (%d episodes)", dataset.num_episodes)
            dataset.finalize()
            if cfg.dataset is not None and cfg.dataset.push_to_hub:
                safe_push_to_hub(dataset, tags=cfg.dataset.tags, private=cfg.dataset.private)

        self._teardown_hardware(ctx.hardware, return_to_initial_position=cfg.return_to_initial_position)
        logger.info("Patrol strategy teardown complete (%d laps)", self._lap - 1)
