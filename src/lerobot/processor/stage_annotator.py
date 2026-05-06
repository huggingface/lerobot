#!/usr/bin/env python
"""In-loop multi-stage annotation for SARM dataset recording.

During teleoperation the operator presses the gamepad stage-advance button to
advance the dataset to the NEXT stage. Stage 0 auto-starts at frame 0; each
subsequent press marks the start frame of the next stage in ``stage_names``.

Unlike the lerobot-panda version (pynput keyboard listener), this port reads
the advance signal from ``info[TeleopEvents.STAGE_ADVANCE]`` which is set by
the gamepad teleop's ``get_teleop_events()`` + ``AddTeleopEventsAsInfoStep``
upstream. Keeps the step dependency-free; works on any teleop that exposes
the flag.

On episode end the caller invokes :meth:`flush_episode_annotation` to get
``(subtask_names, subtask_start_frames, subtask_end_frames)`` arrays
compatible with upstream SARM's sparse / dual annotation mode.

The step is a no-op when ``stage_names`` is empty/None.
"""

import logging
from dataclasses import dataclass, field

from lerobot.processor import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.teleoperators.utils import TeleopEvents

logger = logging.getLogger(__name__)


@dataclass
@ProcessorStepRegistry.register("stage_annotator")
class StageAnnotatorProcessorStep(ProcessorStep):
    """Record multi-stage annotations via teleop button presses.

    Stage convention matches upstream SARM sparse / dual annotation mode:
    each episode outputs ``subtask_names``, ``subtask_start_frames``,
    ``subtask_end_frames`` arrays with entries only for stages the operator
    explicitly reached. Stages the operator never pressed are omitted.
    """

    stage_names: list[str] = field(default_factory=list)

    _pending_advances: int = field(default=0, init=False, repr=False)
    _frame_counter: int = field(default=0, init=False, repr=False)
    # {stage_idx: start_frame}. Stage 0 auto-added at reset().
    _stage_starts: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    # Frames of the partial last-entered stage that got folded into the
    # previous (completed) stage's extension during the last flush. Used by
    # the temporal_proportions writer to avoid inflating the completed
    # stage's frame_count with partial-stage frames.
    _last_extension_frames: int = field(default=0, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.stage_names:
            return transition

        new_transition = transition.copy()

        # Drain one advance per True info flag. Upstream gamepad already
        # debounces via JOYBUTTONDOWN (one event per press), so we trust the
        # flag is a one-shot.
        info_in = new_transition.get(TransitionKey.INFO, {}) or {}
        if info_in.get(TeleopEvents.STAGE_ADVANCE, False):
            self._pending_advances += 1

        stage_started_this_frame: int | None = None
        pending = self._pending_advances
        self._pending_advances = 0

        for _ in range(pending):
            next_stage = (max(self._stage_starts.keys()) + 1) if self._stage_starts else 0
            if next_stage >= len(self.stage_names):
                logger.warning(
                    "[STAGE] advance ignored — already at last stage '%s' (%d/%d configured)",
                    self.stage_names[-1],
                    len(self.stage_names),
                    len(self.stage_names),
                )
                continue
            self._stage_starts[next_stage] = self._frame_counter
            stage_started_this_frame = next_stage
            logger.info(
                "[STAGE] entered '%s' (stage %d) at frame %d",
                self.stage_names[next_stage],
                next_stage,
                self._frame_counter,
            )

        current_stage = max(self._stage_starts.keys()) if self._stage_starts else 0

        info = dict(info_in)
        info["stage_index"] = current_stage
        info["stage_name"] = (
            self.stage_names[current_stage] if current_stage < len(self.stage_names) else None
        )
        if stage_started_this_frame is not None:
            info["stage_started_this_frame"] = stage_started_this_frame
        new_transition[TransitionKey.INFO] = info

        self._frame_counter += 1
        return new_transition

    def reset(self) -> None:
        """Called at episode boundary. Stage 0 auto-starts at frame 0."""
        self._pending_advances = 0
        self._frame_counter = 0
        self._stage_starts = {}
        self._last_extension_frames = 0
        if self.stage_names:
            self._stage_starts[0] = 0

    def transform_features(self, features, **kwargs):
        return features

    # ------------------------------------------------------------------
    # Caller API (used by control_loop on episode end)
    # ------------------------------------------------------------------

    def flush_episode_annotation(self, episode_succeeded: bool = False):
        """Return (names, start_frames, end_frames) for the just-finished episode.

        - Only includes stages the operator actually entered.
        - If the operator DID NOT reach the last configured stage, OR reached
          it but ``episode_succeeded=False``, the last-entered stage is
          treated as partial: it is dropped from the annotation and the
          previous (completed) stage's end is extended to cover the
          partial-stage frames. This makes SARM ``find_stage_and_tau`` clip
          τ to 1 through those frames → progress stays flat at breakpoint[K]
          of the last completed stage (== "τ=0 within partial stage K").
        - End-of-stage K = start-of-stage (K+1) minus 1; last reported stage
          ends at ``_frame_counter - 1``.
        - Returns ``(None, None, None)`` if annotation disabled, nothing
          was recorded, or the episode had zero completed stages (stuck in
          stage 0 with no success signal).
        """
        self._last_extension_frames = 0

        if not self.stage_names or not self._stage_starts:
            return None, None, None

        ordered_idxs = sorted(self._stage_starts.keys())
        reached_final = ordered_idxs[-1] == len(self.stage_names) - 1
        is_partial = (not reached_final) or (not episode_succeeded)

        if is_partial:
            completed = ordered_idxs[:-1]
            if not completed:
                # No stage was actually completed (operator never advanced
                # past stage 0, or never reached final with success).
                return None, None, None
            partial_idx = ordered_idxs[-1]
            self._last_extension_frames = max(
                0, self._frame_counter - self._stage_starts[partial_idx]
            )
            reported_idxs = completed
        else:
            reported_idxs = ordered_idxs

        names = [self.stage_names[i] for i in reported_idxs]
        starts = [self._stage_starts[i] for i in reported_idxs]

        ends: list[int] = []
        for i, idx in enumerate(reported_idxs):
            if i + 1 < len(reported_idxs):
                ends.append(self._stage_starts[reported_idxs[i + 1]] - 1)
            else:
                ends.append(max(0, self._frame_counter - 1))

        return names, starts, ends

    @property
    def frame_count(self) -> int:
        return self._frame_counter

    @property
    def current_stage(self) -> int:
        return max(self._stage_starts.keys()) if self._stage_starts else 0

    def extension_frame_count(self) -> int:
        """Frames of the last-entered partial stage folded into the previous
        stage's extension at the last :meth:`flush_episode_annotation` call.

        Zero if the episode completed all stages (no partial) or if there
        were no completed stages (flush returned None). The temporal
        proportions writer subtracts this from the last reported stage's
        frame_count so partial-stage frames do not inflate any stage's
        duration statistic.
        """
        return self._last_extension_frames
