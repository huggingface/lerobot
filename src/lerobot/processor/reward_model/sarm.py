"""SARM reward model processor step for HIL-SERL.

Wraps ``lerobot.policies.sarm.modeling_sarm.SARMRewardModel`` so its
continuous progress signal (∈ [0, 1]) can drive SAC's reward / termination.

At inference SARM needs a bidirectional window of frames around the query
frame (see ``SARMConfig.observation_delta_indices``). In a live control loop
we only have the past, so this step keeps a rolling deque of the last
``n_obs_steps`` observations and builds the window by replicating the current
frame into the future slots. SARM is queried at its default target index
(the middle of the window), i.e. the current frame.

Reward modes (``reward_mode``):

- ``"binary"`` — match the CNN contract: write ``success_reward`` and
  terminate only when ``progress >= success_threshold``.
- ``"dense"`` — write ``r_t = progress_t`` every step.
- ``"delta"`` — potential-based shaping: ``r_t = progress_t − progress_{t-1}``.

Raw progress is always written to ``INFO['sarm_progress']``.

``task`` MUST match the task text the checkpoint was trained on.
"""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.processor.reward_model.base import (
    BaseRewardProcessorStep,
    RewardModelConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SARMRewardConfig(RewardModelConfig):
    type: str = "sarm"
    success_threshold: float = 0.9
    pretrained_path: str | None = None
    device: str = "cpu"
    task: str = "peg_lifting"
    head_mode: str = "sparse"
    # "binary" | "dense" | "delta"
    reward_mode: str = "binary"
    # Dataset repo_id whose meta.stats feeds the normalizer.
    stats_dataset_repo_id: str | None = None
    # Run SARM every N actor steps; hold previous progress in between.
    eval_every_n_steps: int = 1
    # If True: print a per-step progress bar + predicted stage to stdout. Costs
    # one extra synchronous SARM forward every `verbose_every_n_steps` env steps.
    verbose: bool = False
    verbose_every_n_steps: int = 10
    # Extra reward added on the step that first crosses success_threshold (any
    # reward_mode). Helpful for sparse-success signal in residual RL on top of
    # dense progress shaping. 0.0 = disabled.
    success_terminal_bonus: float = 0.0
    # When True, ignore terminate_on_success at the SARM step (no auto-success
    # by threshold). Reward / bonus still fires. Termination must come from
    # an external source (e.g. gamepad SUCCESS button).
    disable_threshold_termination: bool = False
    # When True, run SARM in "sync" mode: shift positive observation_delta_indices
    # to non-positive (past). Output progress for frame t-max_future_delta using
    # window built only from past frames in the ring buffer. Adds latency =
    # max_future_delta frames * dt (~1.75s for paperfull). Matches training-time
    # offline-eval distribution (which sees real future frames). Default False
    # = legacy async behavior (replicates current frame for future deltas).
    sync_inference: bool = False
    # Path to JSONL log file. If set, every verbose SARM step appends
    # {step, stage_idx, stage_name, stage_conf, progress, delta_indices, ts}.
    # Used to diagnose teleop vs eval distribution gap.
    log_jsonl_path: str | None = None

    def __post_init__(self) -> None:
        allowed = {"binary", "dense", "delta"}
        if self.reward_mode not in allowed:
            raise ValueError(
                f"SARMRewardConfig.reward_mode must be one of {allowed}, got {self.reward_mode!r}"
            )


@dataclass
class SARMRewardProcessorStep(BaseRewardProcessorStep):
    """Reward step wrapping ``SARMRewardModel`` for progress-based success detection."""

    config: SARMRewardConfig = field(default_factory=SARMRewardConfig)

    def __post_init__(self) -> None:
        self._model = None
        self._preprocess: PolicyProcessorPipeline | None = None
        # Legacy single key (first image_key for back-compat refs); multi-cam uses _image_keys.
        self._image_key: str | None = None
        self._image_keys: list[str] = []
        self._state_key: str | None = None
        self._delta_indices: list[int] | None = None
        self._center_idx: int | None = None
        # Multi-cam: dict[key] -> deque. Legacy code may still inspect _image_buf as first cam.
        self._image_bufs: dict[str, deque] = {}
        self._image_buf: deque | None = None  # legacy alias for first cam buffer
        self._state_buf: deque | None = None
        self._last_progress: float = 0.0
        self._prev_progress: float = 0.0
        self._step_counter: int = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sarm")
        self._pending_future = None

        if self.config.pretrained_path is None:
            return

        if self.config.type == "sarm_ext":
            from lerobot_policy_sarm.modeling_sarm import SARMRewardModel
            from lerobot_policy_sarm.processor_sarm import (
                make_sarm_ext_pre_post_processors as make_sarm_pre_post_processors,
            )
        else:
            from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
            from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

        path = Path(self.config.pretrained_path)
        logger.info("Loading SARM reward model from %s onto %s", path, self.config.device)
        self._model = SARMRewardModel.from_pretrained(str(path))
        self._model.config.device = self.config.device
        self._model.to(self.config.device).eval()

        # Multi-cam aware: prefer config.image_keys (list) if set, else [image_key].
        cfg_keys = getattr(self._model.config, "image_keys", None)
        if cfg_keys:
            self._image_keys = list(cfg_keys)
        else:
            self._image_keys = [self._model.config.image_key]
        self._image_key = self._image_keys[0]  # legacy
        self._state_key = self._model.config.state_key

        self._delta_indices = list(self._model.config.observation_delta_indices)
        n_obs = self._model.config.n_obs_steps
        # Centre slot = the obs frame at delta=0 (i.e., "current frame").
        # In bidirectional mode this is at index n_obs//2; in epstart_anchor mode the
        # first slot is the ep_start anchor and delta=0 is at index 1.
        if getattr(self._model.config, "epstart_anchor", False):
            self._center_idx = 1
        else:
            self._center_idx = n_obs // 2

        # SYNC inference: shift positive (future) deltas to non-positive (past).
        # Effect: output progress is for frame `t - max_future_delta`, using past
        # frames from the ring buffer. Matches the offline sync eval distribution.
        if getattr(self.config, "sync_inference", False):
            max_future_delta = max((d for d in self._delta_indices if d > 0), default=0)
            if max_future_delta > 0:
                self._delta_indices = [d - max_future_delta for d in self._delta_indices]
                logging.info(
                    "SARM sync_inference: shifted delta_indices by -%d → %s (latency=%.2fs at 20fps)",
                    max_future_delta, self._delta_indices, max_future_delta / 20.0,
                )

        max_back = max(-min(self._delta_indices), 0)
        self._image_bufs = {k: deque(maxlen=max_back + 1) for k in self._image_keys}
        self._image_buf = self._image_bufs[self._image_keys[0]]  # legacy alias
        self._state_buf = deque(maxlen=max_back + 1)

        stats_repo = self.config.stats_dataset_repo_id
        if stats_repo is None:
            try:
                import json

                with open(path / "train_config.json") as f:
                    stats_repo = json.load(f).get("dataset", {}).get("repo_id")
            except Exception as e:  # noqa: BLE001
                logger.warning("SARM: could not read train_config.json for stats: %s", e)
                stats_repo = None

        dataset_stats = None
        dataset_meta = None
        if stats_repo is not None:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset

                logger.info("SARM: loading stats from %s", stats_repo)
                stats_ds = LeRobotDataset(stats_repo)
                if not stats_ds.meta.stats:
                    raise RuntimeError(
                        f"dataset {stats_repo!r} has empty meta.stats — typically happens when it was "
                        f"produced by split_dataset.py without a follow-up stats pass."
                    )
                dataset_stats = stats_ds.meta.stats
                dataset_meta = stats_ds.meta
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "SARM: failed to load stats dataset %s: %s. State normalization will be identity, "
                    "SARM outputs likely near-zero. Set stats_dataset_repo_id to a dataset with "
                    "populated meta.stats (e.g. the source dataset you're relabeling).",
                    stats_repo,
                    e,
                )

        self._model.config.device = self.config.device
        self._preprocess, _ = make_sarm_pre_post_processors(
            config=self._model.config,
            dataset_stats=dataset_stats,
            dataset_meta=dataset_meta,
        )
        for step in getattr(self._preprocess, "steps", []):
            if hasattr(step, "eval"):
                step.eval()
            if hasattr(step, "_encode_text_clip"):
                _original_encode = step._encode_text_clip
                _text_cache: dict[str, torch.Tensor] = {}

                def _cached_encode(text, batch_size, _orig=_original_encode, _cache=_text_cache):
                    if text not in _cache:
                        _cache[text] = _orig(text, 1)
                    return _cache[text].expand(batch_size, -1)

                step._encode_text_clip = _cached_encode
                logger.info("SARM: CLIP text encoding cached")

        logger.info(
            "SARM loaded: image_keys=%s, state_key=%s, n_obs_steps=%d, frame_gap=%d, center_idx=%d, task=%r",
            self._image_keys,
            self._state_key,
            n_obs,
            self._model.config.frame_gap,
            self._center_idx,
            self.config.task,
        )

    def reset(self) -> None:
        if self._pending_future is not None:
            self._pending_future.cancel()
            self._pending_future = None
        for buf in self._image_bufs.values():
            buf.clear()
        if self._state_buf is not None:
            self._state_buf.clear()
        self._last_progress = 0.0
        self._prev_progress = 0.0
        self._step_counter = 0

    def _build_window_from_snapshot(
        self,
        image_snap,
        state_snap: list,
    ):
        """Build per-cam window + state window.

        image_snap may be:
          - legacy list (single cam) → returns (Tensor, state_stack)
          - dict[key]→list (multi cam) → returns (dict[key]→Tensor, state_stack)
        """
        assert self._delta_indices is not None

        if isinstance(image_snap, dict):
            first_key = next(iter(image_snap))
            ref_snap = image_snap[first_key]
        else:
            ref_snap = image_snap

        current_state = state_snap[-1] if state_snap else None

        # Determine per-delta index into snapshot
        per_delta_back_idx: list[int] = []
        for delta in self._delta_indices:
            if delta >= 0:
                per_delta_back_idx.append(0)  # current
            else:
                back = -delta
                per_delta_back_idx.append(back if back < len(ref_snap) else len(ref_snap) - 1)

        def pick(snap: list, back: int) -> torch.Tensor:
            return snap[-1] if back == 0 else (
                snap[-(back + 1)] if back < len(snap) else snap[0]
            )

        states: list[torch.Tensor | None] = []
        for back in per_delta_back_idx:
            st = current_state if back == 0 else (
                state_snap[-(back + 1)] if back < len(state_snap) else (state_snap[0] if state_snap else current_state)
            )
            states.append(st)

        if any(s is None for s in states):
            state_stack = None
        else:
            state_stack = torch.stack(states, dim=0)  # type: ignore[arg-type]

        if isinstance(image_snap, dict):
            per_cam = {}
            for key, snap in image_snap.items():
                per_cam[key] = torch.stack([pick(snap, b) for b in per_delta_back_idx], dim=0)
            return per_cam, state_stack
        else:
            image_stack = torch.stack([pick(ref_snap, b) for b in per_delta_back_idx], dim=0)
            return image_stack, state_stack

    def _push_obs_to_buffer(self, observation: dict[str, Any]) -> bool:
        current_state = observation.get(self._state_key)

        # Buffer prewarm: when the buffer is empty (first push of an episode),
        # replicate the observation max_back times so sync-mode windows can
        # build a full window immediately instead of clamping to the same
        # frame for ~35 steps. Triggered by sync_inference flag or by any
        # config that needs a deep history.
        prewarm = (
            self._image_bufs[self._image_keys[0]].__len__() == 0
            and getattr(self.config, "sync_inference", False)
            and self._delta_indices is not None
        )

        pushed_any = False
        for key in self._image_keys:
            cur_img = observation.get(key)
            if cur_img is None or not isinstance(cur_img, torch.Tensor):
                return False
            if cur_img.ndim == 4 and cur_img.shape[0] == 1:
                cur_img = cur_img.squeeze(0)
            cpu_img = cur_img.detach().cpu()
            if prewarm:
                # Fill buffer up to its maxlen with copies of the very first frame.
                pad = (self._image_bufs[key].maxlen or 1) - 1
                for _ in range(min(pad, 100)):  # cap to avoid mishaps if maxlen is huge
                    self._image_bufs[key].append(cpu_img)
            self._image_bufs[key].append(cpu_img)
            pushed_any = True

        if isinstance(current_state, torch.Tensor) and current_state.ndim == 2 and current_state.shape[0] == 1:
            current_state = current_state.squeeze(0)
        cpu_state = current_state.detach().cpu() if isinstance(current_state, torch.Tensor) else None
        if prewarm:
            pad = (self._state_buf.maxlen or 1) - 1
            for _ in range(min(pad, 100)):
                self._state_buf.append(cpu_state)
        self._state_buf.append(cpu_state)
        return pushed_any

    def _snapshot_buffers(self):
        """Returns (image_snap, state_snap). image_snap is dict[key]→list for multi-cam;
        single-cam returns a plain list for legacy callers."""
        state_snap = list(self._state_buf)
        if len(self._image_keys) == 1:
            return list(self._image_bufs[self._image_keys[0]]), state_snap
        return {k: list(v) for k, v in self._image_bufs.items()}, state_snap

    def _compute_progress_from_buffer(
        self,
        image_snap=None,
        state_snap: list | None = None,
    ) -> float:
        if self._model is None:
            return 0.0

        if image_snap is None:
            image_snap, state_snap = self._snapshot_buffers()
        # Empty snapshot: dict all-empty or plain empty list
        if not image_snap:
            return 0.0
        if isinstance(image_snap, dict) and not any(image_snap.values()):
            return 0.0

        image_out, state_stack = self._build_window_from_snapshot(
            image_snap=image_snap,
            state_snap=state_snap,
        )

        batch: dict[str, Any] = {
            "task": self.config.task,
            "index": 0,
            "episode_index": 0,
        }
        if isinstance(image_out, dict):
            batch.update(image_out)
        else:
            batch[self._image_keys[0]] = image_out
        if state_stack is not None:
            batch[self._state_key] = state_stack

        with torch.inference_mode():
            processed = self._preprocess(batch)
            progress = self._model.calculate_rewards(
                text_embeddings=processed["text_features"],
                video_embeddings=processed["video_features"],
                state_features=processed.get("state_features"),
                lengths=processed.get("lengths"),
                frame_index=self._center_idx,
                return_all_frames=False,
                return_stages=False,
                head_mode=self.config.head_mode,
            )

        if isinstance(progress, torch.Tensor):
            progress = float(progress.detach().cpu().reshape(-1)[0].item())
        else:
            progress = float(progress.reshape(-1)[0])
        return max(0.0, min(1.0, progress))

    def _run_model(self, observation: dict[str, Any]) -> float:
        if not self._push_obs_to_buffer(observation):
            logger.warning("SARM reward: observation missing image key %r", self._image_key)
            return 0.0
        return self._compute_progress_from_buffer()

    def compute_reward(self, observation: dict[str, Any]) -> float:
        progress = self._run_model(observation)
        self._last_progress = progress
        return progress

    def _drain_pending(self) -> None:
        if self._pending_future is not None:
            try:
                self._last_progress = self._pending_future.result()
            except Exception as e:
                logger.warning("SARM background eval failed: %s", e)
            self._pending_future = None

    def _verbose_print(self) -> None:
        """Sync-compute progress + stage and print a one-line progress bar."""
        if self._model is None:
            return
        img_snap, state_snap = self._snapshot_buffers()
        if not img_snap or (isinstance(img_snap, dict) and not any(img_snap.values())):
            return
        image_out, state_stack = self._build_window_from_snapshot(img_snap, state_snap)
        batch: dict[str, Any] = {"task": self.config.task, "index": 0, "episode_index": 0}
        if isinstance(image_out, dict):
            batch.update(image_out)
        else:
            batch[self._image_keys[0]] = image_out
        if state_stack is not None:
            batch[self._state_key] = state_stack
        try:
            with torch.inference_mode():
                processed = self._preprocess(batch)
                progress, stage_probs = self._model.calculate_rewards(
                    text_embeddings=processed["text_features"],
                    video_embeddings=processed["video_features"],
                    state_features=processed.get("state_features"),
                    lengths=processed.get("lengths"),
                    frame_index=self._center_idx,
                    return_all_frames=False,
                    return_stages=True,
                    head_mode=self.config.head_mode,
                )
        except Exception as e:  # noqa: BLE001
            logger.debug("SARM verbose forward failed: %s", e)
            return
        if isinstance(progress, torch.Tensor):
            prog = float(progress.detach().cpu().reshape(-1)[0].item())
        else:
            prog = float(progress.reshape(-1)[0])
        prog = max(0.0, min(1.0, prog))

        import numpy as _np
        sp = stage_probs.detach().cpu().numpy() if isinstance(stage_probs, torch.Tensor) else _np.asarray(stage_probs)
        if sp.ndim == 3:
            sp = sp[0, self._center_idx]
        elif sp.ndim == 2:
            sp = sp[self._center_idx]
        stage_idx = int(sp.argmax())
        stage_conf = float(sp[stage_idx])
        attr = f"{self.config.head_mode}_subtask_names"
        names = list(getattr(self._model.config, attr, None) or ["task"])
        stage_name = names[stage_idx] if stage_idx < len(names) else f"stage_{stage_idx}"

        bar_w = 30
        filled = int(prog * bar_w)
        bar = "▓" * filled + "░" * (bar_w - filled)
        print(f"[SARM step={self._step_counter:5d}] |{bar}| {prog:.3f}  stage={stage_name}({stage_conf:.2f})", flush=True)
        # Optional JSONL debug log
        log_path = getattr(self.config, "log_jsonl_path", None)
        if log_path:
            import json as _json, time as _time, os as _os
            _os.makedirs(_os.path.dirname(log_path) or ".", exist_ok=True)
            entry = {
                "ts": _time.time(),
                "step": int(self._step_counter),
                "progress": float(prog),
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
                "stage_conf": float(stage_conf),
                "stage_probs": [float(x) for x in sp.tolist()],
                "delta_indices": list(self._delta_indices) if self._delta_indices else None,
                "buffer_len": len(self._image_bufs.get(self._image_keys[0], [])) if self._image_bufs else 0,
                "gt_stage_idx": getattr(self, "_gt_stage_idx", None),
                "gt_stage_name": getattr(self, "_gt_stage_name", None),
                "gt_stage_started_this_frame": getattr(self, "_gt_stage_started_this_frame", None),
            }
            with open(log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        # Capture GT stage from upstream StageAnnotatorProcessorStep (if any)
        # so the JSONL log can include user-pressed stage advances.
        info_in = new_transition.get(TransitionKey.INFO, {}) or {}
        self._gt_stage_idx = info_in.get("stage_index")
        self._gt_stage_name = info_in.get("stage_name")
        self._gt_stage_started_this_frame = info_in.get("stage_started_this_frame")

        self._step_counter += 1
        if self.config.verbose and (self._step_counter % max(1, self.config.verbose_every_n_steps) == 0):
            self._verbose_print()
        is_terminal = (
            new_transition.get(TransitionKey.DONE, False)
            or new_transition.get(TransitionKey.TRUNCATED, False)
        )

        self._push_obs_to_buffer(observation)

        if is_terminal:
            self._drain_pending()
            progress = self._compute_progress_from_buffer()
            self._last_progress = progress
        else:
            if self._pending_future is not None and self._pending_future.done():
                try:
                    self._last_progress = self._pending_future.result()
                except Exception as e:
                    logger.warning("SARM background eval failed: %s", e)
                self._pending_future = None

            should_eval = (
                self.config.eval_every_n_steps <= 1
                or self._step_counter % self.config.eval_every_n_steps == 1
            )
            if self._pending_future is None and should_eval and self._model is not None:
                img_snap, state_snap = self._snapshot_buffers()
                self._pending_future = self._executor.submit(
                    self._compute_progress_from_buffer, img_snap, state_snap
                )

            progress = self._last_progress

        reward = new_transition.get(TransitionKey.REWARD, 0.0) or 0.0
        terminated = new_transition.get(TransitionKey.DONE, False) or False

        mode = self.config.reward_mode
        if mode == "dense":
            reward = progress
        elif mode == "delta":
            reward = progress - self._prev_progress

        if progress >= self.config.success_threshold:
            if mode == "binary":
                reward = self.config.success_reward
            if self.terminate_on_success and not self.config.disable_threshold_termination:
                terminated = True

        # Terminal bonus is fired only when the human SUCCESS button is pressed
        # (TeleopEvents.SUCCESS in info), independent of whether the SARM
        # threshold was hit. Avoids accidentally rewarding "stayed near goal".
        if self.config.success_terminal_bonus != 0.0:
            info_local = new_transition.get(TransitionKey.INFO, {}) or {}
            try:
                from lerobot.teleoperators.utils import TeleopEvents

                _success_key = TeleopEvents.SUCCESS
            except Exception:
                _success_key = "success"
            if info_local.get(_success_key, False):
                reward = float(reward) + float(self.config.success_terminal_bonus)

        self._prev_progress = progress

        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        info = new_transition.get(TransitionKey.INFO, {}) or {}
        info["sarm_progress"] = progress
        new_transition[TransitionKey.INFO] = info
        return new_transition
