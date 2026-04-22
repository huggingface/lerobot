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
        self._image_key: str | None = None
        self._state_key: str | None = None
        self._delta_indices: list[int] | None = None
        self._center_idx: int | None = None
        self._image_buf: deque | None = None
        self._state_buf: deque | None = None
        self._last_progress: float = 0.0
        self._prev_progress: float = 0.0
        self._step_counter: int = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sarm")
        self._pending_future = None

        if self.config.pretrained_path is None:
            return

        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel
        from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

        path = Path(self.config.pretrained_path)
        logger.info("Loading SARM reward model from %s onto %s", path, self.config.device)
        self._model = SARMRewardModel.from_pretrained(str(path))
        self._model.config.device = self.config.device
        self._model.to(self.config.device).eval()

        self._image_key = self._model.config.image_key
        self._state_key = self._model.config.state_key

        self._delta_indices = list(self._model.config.observation_delta_indices)
        n_obs = self._model.config.n_obs_steps
        self._center_idx = n_obs // 2

        max_back = max(-min(self._delta_indices), 0)
        self._image_buf = deque(maxlen=max_back + 1)
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
            "SARM loaded: image_key=%s, state_key=%s, n_obs_steps=%d, frame_gap=%d, center_idx=%d, task=%r",
            self._image_key,
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
        if self._image_buf is not None:
            self._image_buf.clear()
        if self._state_buf is not None:
            self._state_buf.clear()
        self._last_progress = 0.0
        self._prev_progress = 0.0
        self._step_counter = 0

    def _build_window_from_snapshot(
        self,
        image_snap: list,
        state_snap: list,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert self._delta_indices is not None
        current_image = image_snap[-1]
        current_state = state_snap[-1] if state_snap else None

        images: list[torch.Tensor] = []
        states: list[torch.Tensor | None] = []
        for delta in self._delta_indices:
            if delta >= 0:
                images.append(current_image)
                states.append(current_state)
            else:
                back_idx = -delta
                if back_idx < len(image_snap):
                    past_img = image_snap[-(back_idx + 1)]
                    past_state = state_snap[-(back_idx + 1)]
                else:
                    past_img = image_snap[0]
                    past_state = state_snap[0] if state_snap else current_state
                images.append(past_img)
                states.append(past_state)

        image_stack = torch.stack(images, dim=0)
        if any(s is None for s in states):
            state_stack = None
        else:
            state_stack = torch.stack(states, dim=0)  # type: ignore[arg-type]
        return image_stack, state_stack

    def _push_obs_to_buffer(self, observation: dict[str, Any]) -> bool:
        current_image = observation.get(self._image_key)
        current_state = observation.get(self._state_key)
        if current_image is None or not isinstance(current_image, torch.Tensor):
            return False

        if current_image.ndim == 4 and current_image.shape[0] == 1:
            current_image = current_image.squeeze(0)
        if isinstance(current_state, torch.Tensor) and current_state.ndim == 2 and current_state.shape[0] == 1:
            current_state = current_state.squeeze(0)

        self._image_buf.append(current_image.detach().cpu())
        if isinstance(current_state, torch.Tensor):
            self._state_buf.append(current_state.detach().cpu())
        else:
            self._state_buf.append(None)
        return True

    def _snapshot_buffers(self) -> tuple[list, list]:
        return list(self._image_buf), list(self._state_buf)

    def _compute_progress_from_buffer(
        self,
        image_snap: list | None = None,
        state_snap: list | None = None,
    ) -> float:
        if self._model is None:
            return 0.0

        if image_snap is None:
            image_snap, state_snap = self._snapshot_buffers()
        if not image_snap:
            return 0.0

        image_stack, state_stack = self._build_window_from_snapshot(
            image_snap=image_snap,
            state_snap=state_snap,
        )

        batch: dict[str, Any] = {
            self._image_key: image_stack,
            "task": self.config.task,
            "index": 0,
            "episode_index": 0,
        }
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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return new_transition

        self._step_counter += 1
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
            if self.terminate_on_success:
                terminated = True

        self._prev_progress = progress

        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        info = new_transition.get(TransitionKey.INFO, {}) or {}
        info["sarm_progress"] = progress
        new_transition[TransitionKey.INFO] = info
        return new_transition
