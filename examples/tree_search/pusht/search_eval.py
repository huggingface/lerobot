#!/usr/bin/env python
"""Tree-search evaluation for PushT policies.

This script intentionally keeps the implementation PushT-specific. It uses the
same LeRobot factories as `lerobot-eval` for policy loading and processors, but
it owns the simulator loop so it can snapshot the current PushT state, roll out
hypothetical action chunks, score the resulting states, then restore the real
state before committing an action.

Example:

```bash
uv run python examples/tree_search/pusht/search_eval.py \
  --policy.path=aadarshram/act_pusht \
  --policy.device=cuda \
  --policy.use_amp=false \
  --episodes=10 \
  --num-candidates=8 \
  --depth=2 \
  --chunk-size=8
```
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs import (
    PushtEnv,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed


LOGGER = logging.getLogger("pusht_tree_search")
ONE_STEP_FURTHER_COVERAGE_DROP = 0.05


@dataclass(frozen=True)
class PlannerConfig:
    policy_path: str
    policy_device: str
    policy_use_amp: bool
    episodes: int
    seed: int | None
    output_dir: Path
    chunk_size: int
    execute_steps: int
    depth: int
    beam_width: int
    num_candidates: int
    noise_std: float
    noise_mode: str
    score_mode: str
    max_steps: int | None
    render_videos: int
    video_overlay: bool
    log_every_steps: int
    dump_frames: bool
    plot_policy_trace: bool
    dump_search_images: bool
    one_step_further: bool


@dataclass
class PushTSnapshot:
    elapsed_steps: int | None
    agent_position: np.ndarray
    agent_velocity: np.ndarray
    block_position: np.ndarray
    block_velocity: np.ndarray
    block_angle: float
    block_angular_velocity: float
    last_action: np.ndarray | None
    n_contact_points: int
    rng_state: dict[str, Any]


@dataclass
class SimRollout:
    observation: Mapping[str, Any]
    state: PushTSnapshot
    rewards: list[float]
    max_reward: float
    sum_reward: float
    final_reward: float
    coverage: float
    success: bool
    terminated: bool
    truncated: bool
    action_count: int
    agent_positions: list[np.ndarray]


@dataclass
class SearchNode:
    state: PushTSnapshot
    observation: Mapping[str, Any]
    depth: int
    env_step: int
    root_actions: np.ndarray | None
    root_candidate_index: int | None
    score: float
    success: bool


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int | None
    sum_reward: float
    max_reward: float
    success: bool
    steps: int
    alternative_selection_count: int
    selection_count: int
    asr: float
    video_path: str | None


@dataclass
class SearchChunkTrace:
    candidate_index: int
    points: list[np.ndarray]
    score: float
    is_original: bool
    is_selected: bool = False


def annotate_frame(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    annotated = np.ascontiguousarray(frame.copy())
    if annotated.dtype != np.uint8:
        annotated = np.clip(annotated, 0, 255).astype(np.uint8)

    pad = 8
    line_height = 22
    box_height = pad * 2 + line_height * len(lines)
    box_width = min(annotated.shape[1], 360)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    for line_ix, line in enumerate(lines):
        y = pad + 16 + line_ix * line_height
        cv2.putText(
            annotated,
            line,
            (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def write_frame_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ascontiguousarray(frame)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write frame image: {path}")


def maybe_annotate_frame(frame: np.ndarray, lines: Sequence[str], *, enabled: bool) -> np.ndarray:
    if not enabled:
        return np.ascontiguousarray(frame.copy())
    return annotate_frame(frame, lines)


def save_search_debug_image(
    *,
    path: Path,
    frame: np.ndarray,
    traces: Sequence[SearchChunkTrace],
    episode_index: int,
    env_step: int,
) -> None:
    if not traces:
        return

    image = np.ascontiguousarray(frame.copy())
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]
    other_color = (170, 170, 170)
    original_color = (20, 120, 255)
    selected_color = (30, 220, 30)

    def to_pixel(point: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(float(point[0]) / 512.0 * width, 0, width - 1))
        y = int(np.clip(float(point[1]) / 512.0 * height, 0, height - 1))
        return x, y

    sorted_traces = sorted(traces, key=lambda trace: trace.candidate_index)
    for trace in sorted_traces:
        color = original_color if trace.is_original else other_color
        draw_trace_dots(image, trace.points, color=color, radius=2, thickness=-1, to_pixel=to_pixel)

    for trace in sorted_traces:
        if trace.is_selected:
            draw_trace_dots(
                image,
                trace.points,
                color=selected_color,
                radius=2,
                thickness=2,
                to_pixel=to_pixel,
            )

    selected = next((trace for trace in sorted_traces if trace.is_selected), None)
    selected_label = "none" if selected is None else str(selected.candidate_index)
    image = annotate_frame(
        image,
        [
            f"episode={episode_index} step={env_step}",
            f"chosen={selected_label} original=0 candidates={len(traces)}",
            "green=chosen outline blue=original gray=other",
        ],
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write search debug image: {path}")
    save_search_trace_json(
        path=path.with_suffix(".json"),
        traces=sorted_traces,
        episode_index=episode_index,
        env_step=env_step,
        width=width,
        height=height,
        to_pixel=to_pixel,
    )


def save_search_trace_json(
    *,
    path: Path,
    traces: Sequence[SearchChunkTrace],
    episode_index: int,
    env_step: int,
    width: int,
    height: int,
    to_pixel: Callable[[np.ndarray], tuple[int, int]],
) -> None:
    payload = {
        "episode_index": episode_index,
        "env_step": env_step,
        "image_width": width,
        "image_height": height,
        "coordinate_frame": "image_pixels_xy_from_top_left",
        "world_coordinate_frame": "pusht_world_xy_0_512",
        "traces": [
            {
                "candidate_index": trace.candidate_index,
                "score": trace.score,
                "is_original": trace.is_original,
                "is_selected": trace.is_selected,
                "pixel_points": [[int(x), int(y)] for x, y in [to_pixel(point) for point in trace.points]],
                "world_points": [
                    [float(point[0]), float(point[1])] for point in trace.points
                ],
            }
            for trace in traces
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def save_policy_trace_debug(
    *,
    path: Path,
    frame: np.ndarray,
    actions: np.ndarray,
    rollout: SimRollout,
    episode_index: int,
    env_step: int,
    coverage_before: float,
    coverage_drop: float,
) -> None:
    image = np.ascontiguousarray(frame.copy())
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]

    def to_pixel(point: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(float(point[0]) / 512.0 * width, 0, width - 1))
        y = int(np.clip(float(point[1]) / 512.0 * height, 0, height - 1))
        return x, y

    action_points = [np.asarray(action, dtype=np.float32).copy() for action in actions[: rollout.action_count]]
    agent_points = rollout.agent_positions

    draw_trace_dots(
        image,
        action_points,
        color=(255, 80, 220),
        radius=5,
        thickness=-1,
        to_pixel=to_pixel,
    )
    draw_trace_dots(
        image,
        agent_points,
        color=(20, 220, 255),
        radius=5,
        thickness=2,
        to_pixel=to_pixel,
    )

    image = annotate_frame(
        image,
        [
            f"policy trace episode={episode_index} step={env_step}",
            f"coverage {coverage_before:.3f}->{rollout.coverage:.3f} drop={coverage_drop:.3f}",
            "magenta=policy actions cyan=sim agent",
        ],
    )

    write_frame_png(path, image)
    save_policy_trace_json(
        path=path.with_suffix(".json"),
        actions=action_points,
        agent_points=agent_points,
        rollout=rollout,
        episode_index=episode_index,
        env_step=env_step,
        width=width,
        height=height,
        coverage_before=coverage_before,
        coverage_drop=coverage_drop,
        to_pixel=to_pixel,
    )


def save_policy_trace_json(
    *,
    path: Path,
    actions: Sequence[np.ndarray],
    agent_points: Sequence[np.ndarray],
    rollout: SimRollout,
    episode_index: int,
    env_step: int,
    width: int,
    height: int,
    coverage_before: float,
    coverage_drop: float,
    to_pixel: Callable[[np.ndarray], tuple[int, int]],
) -> None:
    payload = {
        "episode_index": episode_index,
        "env_step": env_step,
        "image_width": width,
        "image_height": height,
        "coordinate_frame": "image_pixels_xy_from_top_left",
        "world_coordinate_frame": "pusht_world_xy_0_512",
        "coverage_before": coverage_before,
        "coverage_after": rollout.coverage,
        "coverage_drop": coverage_drop,
        "score": rollout.coverage,
        "reward_sum": rollout.sum_reward,
        "max_reward": rollout.max_reward,
        "success": rollout.success,
        "action_count": rollout.action_count,
        "action_trace": {
            "pixel_points": [[int(x), int(y)] for x, y in [to_pixel(point) for point in actions]],
            "world_points": [[float(point[0]), float(point[1])] for point in actions],
        },
        "simulated_agent_trace": {
            "pixel_points": [[int(x), int(y)] for x, y in [to_pixel(point) for point in agent_points]],
            "world_points": [[float(point[0]), float(point[1])] for point in agent_points],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def draw_trace_dots(
    image: np.ndarray,
    points: Sequence[np.ndarray],
    *,
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
    to_pixel: Callable[[np.ndarray], tuple[int, int]],
) -> None:
    if not points:
        return
    pixels = [to_pixel(point) for point in points]
    for start, end in zip(pixels, pixels[1:], strict=False):
        cv2.line(image, start, end, color=color, thickness=2, lineType=cv2.LINE_AA)
    for point_ix, point in enumerate(pixels):
        cv2.circle(image, point, radius, color, thickness=thickness, lineType=cv2.LINE_AA)
        if point_ix == len(pixels) - 1:
            cv2.circle(image, point, radius + 2, color, thickness=2, lineType=cv2.LINE_AA)


class PushTStateAdapter:
    """Snapshot and restore the mutable state of `gym_pusht.envs.pusht.PushTEnv`."""

    def __init__(self, base_env: gym.Env) -> None:
        self.base_env = base_env
        self.unwrapped = base_env.unwrapped

    def snapshot(self) -> PushTSnapshot:
        return PushTSnapshot(
            elapsed_steps=getattr(self.base_env, "_elapsed_steps", None),
            agent_position=np.asarray(self.unwrapped.agent.position, dtype=np.float64).copy(),
            agent_velocity=np.asarray(self.unwrapped.agent.velocity, dtype=np.float64).copy(),
            block_position=np.asarray(self.unwrapped.block.position, dtype=np.float64).copy(),
            block_velocity=np.asarray(self.unwrapped.block.velocity, dtype=np.float64).copy(),
            block_angle=float(self.unwrapped.block.angle),
            block_angular_velocity=float(self.unwrapped.block.angular_velocity),
            last_action=(
                None
                if self.unwrapped._last_action is None
                else np.asarray(self.unwrapped._last_action, dtype=np.float64).copy()
            ),
            n_contact_points=int(self.unwrapped.n_contact_points),
            rng_state=copy.deepcopy(self.unwrapped.np_random.bit_generator.state),
        )

    def restore(self, state: PushTSnapshot) -> Mapping[str, Any]:
        if state.elapsed_steps is not None:
            self.base_env._elapsed_steps = int(state.elapsed_steps)

        self.unwrapped.agent.position = state.agent_position.tolist()
        self.unwrapped.agent.velocity = state.agent_velocity.tolist()
        self.unwrapped.block.position = state.block_position.tolist()
        self.unwrapped.block.velocity = state.block_velocity.tolist()
        self.unwrapped.block.angle = float(state.block_angle)
        self.unwrapped.block.angular_velocity = float(state.block_angular_velocity)
        self.unwrapped._last_action = None if state.last_action is None else state.last_action.copy()
        self.unwrapped.n_contact_points = int(state.n_contact_points)
        self.unwrapped.np_random.bit_generator.state = copy.deepcopy(state.rng_state)

        # Let pymunk synchronize cached transforms without advancing time.
        self.unwrapped.space.step(0.0)
        return self.unwrapped.get_obs()

    def step(
        self, action: np.ndarray
    ) -> tuple[Mapping[str, Any], float, bool, bool, Mapping[str, Any]]:
        observation, reward, terminated, truncated, info = self.base_env.step(action)
        return observation, float(reward), bool(terminated), bool(truncated), dict(info)

    def render(self) -> np.ndarray:
        return self.base_env.render()


class PolicyChunker:
    """Policy inference wrapper that returns env-ready action chunks."""

    def __init__(
        self,
        *,
        policy: torch.nn.Module,
        env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
        device: torch.device,
        use_amp: bool,
    ) -> None:
        self.policy = policy
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.use_amp = use_amp

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict_chunk(self, observation: Mapping[str, Any], *, horizon: int) -> np.ndarray:
        batch = preprocess_observation(dict(observation))
        batch["task"] = [""]
        batch = self.env_preprocessor(batch)
        batch = self.preprocessor(batch)

        amp_context = torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext()
        with torch.inference_mode(), amp_context:
            if hasattr(self.policy, "predict_action_chunk"):
                actions = self.policy.predict_action_chunk(batch)
            else:
                actions = self.policy.select_action(batch).unsqueeze(1)

        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        actions = actions[:, :horizon]
        actions = self.postprocessor(actions)
        actions = self.env_postprocessor({ACTION: actions})[ACTION]

        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(
                f"Expected action chunk shape (1, horizon, action_dim), got {tuple(actions.shape)}"
            )
        return actions[0].detach().cpu().numpy().astype(np.float32, copy=True)


class CandidateGenerator:
    def __init__(
        self,
        *,
        action_space: gym.Space,
        num_candidates: int,
        noise_std: float,
        noise_mode: str,
        rng: np.random.Generator,
    ) -> None:
        self.low = np.asarray(action_space.low, dtype=np.float32)
        self.high = np.asarray(action_space.high, dtype=np.float32)
        self.num_candidates = num_candidates
        self.noise_std = noise_std
        self.noise_mode = noise_mode
        self.rng = rng

    def make(self, policy_chunk: np.ndarray) -> list[np.ndarray]:
        clipped = np.clip(policy_chunk, self.low, self.high).astype(np.float32)
        candidates = [clipped]
        for _ in range(1, self.num_candidates):
            if self.noise_mode == "iid":
                noise = self.rng.normal(0.0, self.noise_std, size=policy_chunk.shape)
            elif self.noise_mode == "chunk":
                noise = self.rng.normal(0.0, self.noise_std, size=(1, policy_chunk.shape[-1]))
            else:
                chunk_noise = self.rng.normal(0.0, self.noise_std, size=(1, policy_chunk.shape[-1]))
                iid_noise = self.rng.normal(0.0, self.noise_std * 0.25, size=policy_chunk.shape)
                noise = chunk_noise + iid_noise
            candidates.append(np.clip(policy_chunk + noise, self.low, self.high).astype(np.float32))
        return candidates


class RolloutScorer:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def score(self, rollout: SimRollout) -> float:
        if rollout.success:
            return 1.0
        if self.mode == "coverage":
            return float(rollout.coverage)
        if self.mode == "sum_reward":
            return float(rollout.sum_reward)
        if self.mode == "max_reward":
            return float(rollout.max_reward)
        if self.mode == "final_reward":
            return float(rollout.final_reward)
        raise ValueError(f"Unsupported score mode: {self.mode}")


class BeamTreePlanner:
    def __init__(
        self,
        *,
        adapter: PushTStateAdapter,
        policy: PolicyChunker,
        candidates: CandidateGenerator,
        scorer: RolloutScorer,
        cfg: PlannerConfig,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.candidates = candidates
        self.scorer = scorer
        self.cfg = cfg
        self.last_root_traces: list[SearchChunkTrace] = []

    def choose(
        self, observation: Mapping[str, Any], *, env_step: int, max_steps: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        root_state = self.adapter.snapshot()
        self.last_root_traces = []
        root = SearchNode(
            state=root_state,
            observation=observation,
            depth=0,
            env_step=env_step,
            root_actions=None,
            root_candidate_index=None,
            score=0.0,
            success=False,
        )
        frontier = [root]
        best: SearchNode | None = None
        expanded = 0

        for _depth in range(self.cfg.depth):
            next_frontier: list[SearchNode] = []
            for node in frontier:
                if node.success or node.env_step >= max_steps:
                    next_frontier.append(node)
                    continue
                children = self._expand(node, max_steps=max_steps)
                expanded += 1
                next_frontier.extend(children)
                for child in children:
                    if best is None or child.score > best.score:
                        best = child
            if not next_frontier:
                break
            frontier = sorted(next_frontier, key=lambda node: node.score, reverse=True)[: self.cfg.beam_width]
            if any(node.success for node in frontier):
                break

        self.adapter.restore(root_state)
        if best is None or best.root_actions is None:
            fallback = self.policy.predict_chunk(observation, horizon=self.cfg.chunk_size)
            return fallback, {"reason": "fallback_policy_chunk", "expanded_nodes": expanded}

        for trace in self.last_root_traces:
            trace.is_selected = trace.candidate_index == best.root_candidate_index

        return best.root_actions, {
            "reason": "beam_search",
            "best_score": best.score,
            "best_depth": best.depth,
            "selected_candidate_index": best.root_candidate_index,
            "expanded_nodes": expanded,
            "selected_chunk_len": int(best.root_actions.shape[0]),
        }

    def _expand(self, node: SearchNode, *, max_steps: int) -> list[SearchNode]:
        self.adapter.restore(node.state)
        policy_chunk = self.policy.predict_chunk(node.observation, horizon=self.cfg.chunk_size)
        action_candidates = self.candidates.make(policy_chunk)

        children: list[SearchNode] = []
        for candidate_index, actions in enumerate(action_candidates):
            self.adapter.restore(node.state)
            rollout = rollout_actions(
                adapter=self.adapter,
                actions=actions,
                start_step=node.env_step,
                max_steps=max_steps,
            )
            score = self.scorer.score(rollout)
            if node.root_actions is None:
                self.last_root_traces.append(
                    SearchChunkTrace(
                        candidate_index=candidate_index,
                        points=rollout.agent_positions,
                        score=score,
                        is_original=candidate_index == 0,
                    )
                )
            children.append(
                SearchNode(
                    state=rollout.state,
                    observation=rollout.observation,
                    depth=node.depth + 1,
                    env_step=node.env_step + rollout.action_count,
                    root_actions=actions if node.root_actions is None else node.root_actions,
                    root_candidate_index=(
                        candidate_index if node.root_candidate_index is None else node.root_candidate_index
                    ),
                    score=score,
                    success=rollout.success,
                )
            )

        self.adapter.restore(node.state)
        return children


def rollout_actions(
    *,
    adapter: PushTStateAdapter,
    actions: np.ndarray,
    start_step: int,
    max_steps: int,
) -> SimRollout:
    rewards: list[float] = []
    observation: Mapping[str, Any] | None = None
    info: Mapping[str, Any] = {}
    terminated = False
    truncated = False
    success = False
    agent_positions: list[np.ndarray] = []
    max_actions = min(len(actions), max(0, max_steps - start_step))

    for action in actions[:max_actions]:
        observation, reward, terminated, truncated, info = adapter.step(action)
        rewards.append(float(reward))
        pos_agent = info.get("pos_agent", np.asarray(adapter.unwrapped.agent.position))
        agent_positions.append(np.asarray(pos_agent, dtype=np.float32).copy())
        success = success or bool(info.get("is_success", False))
        if terminated or truncated:
            break

    if observation is None:
        observation = adapter.unwrapped.get_obs()

    return SimRollout(
        observation=observation,
        state=adapter.snapshot(),
        rewards=rewards,
        max_reward=max(rewards) if rewards else 0.0,
        sum_reward=float(sum(rewards)),
        final_reward=float(rewards[-1]) if rewards else 0.0,
        coverage=float(info.get("coverage", 0.0)),
        success=success,
        terminated=terminated,
        truncated=truncated,
        action_count=len(rewards),
        agent_positions=agent_positions,
    )


def current_coverage(adapter: PushTStateAdapter) -> float:
    return float(adapter.unwrapped._get_coverage())


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def parse_args() -> PlannerConfig:
    parser = argparse.ArgumentParser(description="Evaluate a PushT policy with beam tree search.")
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--policy.device", dest="policy_device", default="cuda")
    parser.add_argument("--policy.use_amp", dest="policy_use_amp", type=str_to_bool, default=False)
    parser.add_argument("--episodes", "--eval.n_episodes", dest="episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("outputs/tree_search/pusht"),
    )
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=8)
    parser.add_argument("--execute-steps", "--execute_steps", dest="execute_steps", type=int, default=1)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam-width", "--beam_width", dest="beam_width", type=int, default=4)
    parser.add_argument("--num-candidates", "--num_candidates", dest="num_candidates", type=int, default=8)
    parser.add_argument("--noise-std", "--noise_std", dest="noise_std", type=float, default=12.0)
    parser.add_argument(
        "--noise-mode",
        "--noise_mode",
        dest="noise_mode",
        choices=["chunk", "iid", "mixed"],
        default="mixed",
    )
    parser.add_argument(
        "--score-mode",
        "--score_mode",
        dest="score_mode",
        choices=["coverage", "sum_reward", "max_reward", "final_reward"],
        default="coverage",
    )
    parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--render-videos", "--render_videos", dest="render_videos", type=int, default=1)
    parser.add_argument(
        "--video-overlay",
        "--video_overlay",
        dest="video_overlay",
        type=str_to_bool,
        default=True,
        help="Overlay Search On/Off, step, reward, and coverage text on rollout videos and dumped frames.",
    )
    parser.add_argument(
        "--log-every-steps",
        "--log_every_steps",
        dest="log_every_steps",
        type=int,
        default=10,
        help="Print episode progress every N committed environment steps. Use 1 for detailed progress.",
    )
    parser.add_argument(
        "--dump-frames",
        "--dump_frames",
        dest="dump_frames",
        type=str_to_bool,
        default=False,
        help="Write rollout frames as PNGs under OUTPUT_DIR/frames/episode_XXX/.",
    )
    parser.add_argument(
        "--plot-policy-trace",
        "--plot_policy_trace",
        dest="plot_policy_trace",
        action="store_true",
        help=(
            "Save PNG/JSON traces for the raw policy chunk at each decision point under "
            "OUTPUT_DIR/policy_frames/episode_XXX/."
        ),
    )
    parser.add_argument(
        "--dump-search-images",
        "--dump_search_images",
        dest="dump_search_images",
        type=str_to_bool,
        default=False,
        help="Write one PNG per planning call with root action-chunk dot trajectories.",
    )
    parser.add_argument(
        "--one-step-further",
        "--one_step_further",
        dest="one_step_further",
        action="store_true",
        help=(
            "Before running search, simulate the raw policy chunk. Only search if final coverage drops "
            f"by more than {ONE_STEP_FURTHER_COVERAGE_DROP:.2f}."
        ),
    )

    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive.")
    if args.execute_steps <= 0:
        parser.error("--execute-steps must be positive.")
    if args.depth <= 0:
        parser.error("--depth must be positive.")
    if args.beam_width <= 0:
        parser.error("--beam-width must be positive.")
    if args.num_candidates <= 0:
        parser.error("--num-candidates must be positive.")
    if args.log_every_steps <= 0:
        parser.error("--log-every-steps must be positive.")

    return PlannerConfig(**vars(args))


def load_policy_stack(
    cfg: PlannerConfig, env_cfg: PushtEnv
) -> tuple[PolicyChunker, dict[str, dict[int, gym.vector.VectorEnv]]]:
    policy_overrides = [
        f"--device={cfg.policy_device}",
        f"--use_amp={str(cfg.policy_use_amp).lower()}",
    ]
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path, cli_overrides=policy_overrides)
    policy_cfg.pretrained_path = Path(cfg.policy_path)

    device = get_safe_torch_device(policy_cfg.device, log=True)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": {}},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    return (
        PolicyChunker(
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=device,
            use_amp=policy_cfg.use_amp,
        ),
        envs,
    )


def run_episode(
    *,
    episode_index: int,
    episode_seed: int | None,
    base_env: gym.Env,
    action_source: PolicyChunker,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    video_dir: Path,
    frame_dir: Path,
    policy_frame_dir: Path,
    search_image_dir: Path,
) -> EpisodeResult:
    adapter = PushTStateAdapter(base_env)
    scorer = RolloutScorer(cfg.score_mode)
    candidates = CandidateGenerator(
        action_space=base_env.action_space,
        num_candidates=cfg.num_candidates,
        noise_std=cfg.noise_std,
        noise_mode=cfg.noise_mode,
        rng=rng,
    )
    planner = BeamTreePlanner(
        adapter=adapter,
        policy=action_source,
        candidates=candidates,
        scorer=scorer,
        cfg=cfg,
    )

    action_source.reset()
    observation, _ = base_env.reset(seed=episode_seed)
    max_steps = cfg.max_steps or int(base_env._max_episode_steps)
    LOGGER.info(
        "episode=%s seed=%s started max_steps=%s",
        episode_index,
        episode_seed,
        max_steps,
    )

    frames: list[np.ndarray] = []
    should_render_video = episode_index < cfg.render_videos
    should_capture_rollout_frame = should_render_video or cfg.dump_frames
    frame_index = 0
    if should_capture_rollout_frame:
        frame = maybe_annotate_frame(
            adapter.render(),
            [
                "Search: Off",
                f"episode={episode_index} step=0/{max_steps}",
                "reset",
            ],
            enabled=cfg.video_overlay,
        )
        if should_render_video:
            frames.append(frame)
        if cfg.dump_frames:
            write_frame_png(frame_dir / f"episode_{episode_index:03d}" / f"frame_{frame_index:05d}.png", frame)
        frame_index += 1

    rewards: list[float] = []
    success = False
    terminated = False
    truncated = False
    env_step = 0
    alternative_selection_count = 0
    selection_count = 0

    while env_step < max_steps and not (terminated or truncated or success):
        root_state = adapter.snapshot()
        search_frame: np.ndarray | None = None
        should_search = True
        policy_rollout: SimRollout | None = None
        policy_chunk: np.ndarray | None = None
        coverage_before = current_coverage(adapter)

        if cfg.one_step_further or cfg.plot_policy_trace:
            policy_chunk = action_source.predict_chunk(observation, horizon=cfg.chunk_size)
            policy_rollout = rollout_actions(
                adapter=adapter,
                actions=policy_chunk,
                start_step=env_step,
                max_steps=max_steps,
            )
            adapter.restore(root_state)
            coverage_drop = coverage_before - policy_rollout.coverage
            if cfg.plot_policy_trace:
                save_policy_trace_debug(
                    path=(
                        policy_frame_dir
                        / f"episode_{episode_index:03d}"
                        / f"policy_trace_step_{env_step:05d}.png"
                    ),
                    frame=adapter.render(),
                    actions=policy_chunk,
                    rollout=policy_rollout,
                    episode_index=episode_index,
                    env_step=env_step,
                    coverage_before=coverage_before,
                    coverage_drop=coverage_drop,
                )

        if cfg.one_step_further:
            assert policy_chunk is not None
            assert policy_rollout is not None
            coverage_drop = coverage_before - policy_rollout.coverage
            should_search = coverage_drop > ONE_STEP_FURTHER_COVERAGE_DROP

            if not should_search:
                action_chunk = policy_chunk
                plan_info = {
                    "reason": "one_step_further_policy",
                    "best_score": scorer.score(policy_rollout),
                    "selected_candidate_index": 0,
                    "expanded_nodes": 0,
                    "coverage_before": coverage_before,
                    "policy_coverage_after": policy_rollout.coverage,
                    "coverage_drop": coverage_drop,
                }
            else:
                search_frame = adapter.render() if cfg.dump_search_images else None
                action_chunk, plan_info = planner.choose(
                    observation, env_step=env_step, max_steps=max_steps
                )
                plan_info.update(
                    {
                        "coverage_before": coverage_before,
                        "policy_coverage_after": policy_rollout.coverage,
                        "coverage_drop": coverage_drop,
                    }
                )
        else:
            search_frame = adapter.render() if cfg.dump_search_images else None
            action_chunk, plan_info = planner.choose(observation, env_step=env_step, max_steps=max_steps)

        adapter.restore(root_state)
        if should_search and cfg.dump_search_images and search_frame is not None:
            search_image_path = (
                search_image_dir / f"episode_{episode_index:03d}" / f"step_{env_step:05d}.png"
            )
            save_search_debug_image(
                path=search_image_path,
                frame=search_frame,
                traces=planner.last_root_traces,
                episode_index=episode_index,
                env_step=env_step,
            )

        commit_count = min(cfg.execute_steps, len(action_chunk), max_steps - env_step)
        selection_count += 1
        selected_candidate = plan_info.get("selected_candidate_index")
        if selected_candidate is not None and int(selected_candidate) != 0:
            alternative_selection_count += 1
        if env_step == 0 or env_step % cfg.log_every_steps == 0:
            LOGGER.info(
                "episode=%s step=%s/%s planning_done reason=%s best_score=%s selected=%s "
                "expanded_nodes=%s coverage_drop=%s asr=%.2f",
                episode_index,
                env_step,
                max_steps,
                plan_info.get("reason"),
                plan_info.get("best_score"),
                plan_info.get("selected_candidate_index"),
                plan_info.get("expanded_nodes"),
                plan_info.get("coverage_drop"),
                alternative_selection_count / selection_count * 100.0,
            )
        for action_ix, action in enumerate(action_chunk[:commit_count]):
            observation, reward, terminated, truncated, info = adapter.step(action)
            rewards.append(float(reward))
            success = success or bool(info.get("is_success", False))
            env_step += 1
            if should_capture_rollout_frame:
                frame = maybe_annotate_frame(
                    adapter.render(),
                    [
                        f"Search: {'On' if should_search and action_ix == 0 else 'Off'}",
                        f"episode={episode_index} step={env_step}/{max_steps}",
                        f"reward={reward:.3f} coverage={float(info.get('coverage', 0.0)):.3f}",
                    ],
                    enabled=cfg.video_overlay,
                )
                if should_render_video:
                    frames.append(frame)
                if cfg.dump_frames:
                    write_frame_png(
                        frame_dir / f"episode_{episode_index:03d}" / f"frame_{frame_index:05d}.png",
                        frame,
                    )
                frame_index += 1
            if env_step >= max_steps or terminated or truncated or success:
                break
        if env_step == 0 or env_step % cfg.log_every_steps == 0 or terminated or truncated or success:
            LOGGER.info(
                "episode=%s step=%s/%s reward=%.4f max_reward=%.4f coverage=%.4f success=%s",
                episode_index,
                env_step,
                max_steps,
                rewards[-1] if rewards else 0.0,
                max(rewards) if rewards else 0.0,
                float(info.get("coverage", 0.0)) if "info" in locals() else 0.0,
                success,
            )

    video_path = None
    if should_render_video and frames:
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(video_dir / f"episode_{episode_index:03d}.mp4")
        write_video(video_path, np.asarray(frames), fps=int(base_env.unwrapped.metadata["render_fps"]))

    return EpisodeResult(
        episode_index=episode_index,
        seed=episode_seed,
        sum_reward=float(sum(rewards)),
        max_reward=float(max(rewards) if rewards else 0.0),
        success=success,
        steps=env_step,
        alternative_selection_count=alternative_selection_count,
        selection_count=selection_count,
        asr=(alternative_selection_count / selection_count * 100.0) if selection_count else 0.0,
        video_path=video_path,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    register_third_party_plugins()

    env_cfg = PushtEnv()
    per_decision_rollouts = sum(cfg.beam_width**level for level in range(cfg.depth))
    approximate_sim_steps = per_decision_rollouts * cfg.num_candidates * cfg.chunk_size
    LOGGER.info(
        "Loading policy and PushT env. Search cost is roughly %s simulated env steps per real decision "
        "(depth=%s beam_width=%s num_candidates=%s chunk_size=%s).",
        approximate_sim_steps,
        cfg.depth,
        cfg.beam_width,
        cfg.num_candidates,
        cfg.chunk_size,
    )
    action_source, envs = load_policy_stack(cfg, env_cfg)
    vector_env = envs["pusht"][0]
    if not isinstance(vector_env, gym.vector.SyncVectorEnv) or vector_env.num_envs != 1:
        raise ValueError("PushT tree search requires a single SyncVectorEnv.")
    base_env = vector_env.envs[0]

    LOGGER.info("Starting PushT tree-search eval: %s", cfg)
    rng = np.random.default_rng(cfg.seed)
    started_at = time.time()
    results: list[EpisodeResult] = []
    try:
        for episode_index in range(cfg.episodes):
            episode_seed = None if cfg.seed is None else cfg.seed + episode_index
            result = run_episode(
                episode_index=episode_index,
                episode_seed=episode_seed,
                base_env=base_env,
                action_source=action_source,
                cfg=cfg,
                rng=rng,
                video_dir=cfg.output_dir / "videos",
                frame_dir=cfg.output_dir / "frames",
                policy_frame_dir=cfg.output_dir / "policy_frames",
                search_image_dir=cfg.output_dir / "search_images",
            )
            results.append(result)
            LOGGER.info(
                "episode=%s seed=%s sum_reward=%.3f max_reward=%.3f success=%s steps=%s",
                result.episode_index,
                result.seed,
                result.sum_reward,
                result.max_reward,
                result.success,
                result.steps,
            )
    finally:
        close_envs(envs)

    elapsed_s = time.time() - started_at
    total_alternative_selections = sum(result.alternative_selection_count for result in results)
    total_selections = sum(result.selection_count for result in results)
    payload = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "per_episode": [asdict(result) for result in results],
        "aggregated": {
            "avg_sum_reward": float(np.mean([result.sum_reward for result in results])) if results else 0.0,
            "avg_max_reward": float(np.mean([result.max_reward for result in results])) if results else 0.0,
            "pc_success": float(np.mean([result.success for result in results]) * 100.0) if results else 0.0,
            "asr": (
                float(total_alternative_selections / total_selections * 100.0)
                if total_selections
                else 0.0
            ),
            "alternative_selection_count": total_alternative_selections,
            "selection_count": total_selections,
            "n_episodes": len(results),
            "eval_s": elapsed_s,
            "eval_ep_s": elapsed_s / len(results) if results else 0.0,
            "video_paths": [result.video_path for result in results if result.video_path is not None],
        },
    }

    metrics_path = cfg.output_dir / "eval_info.json"
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["aggregated"], indent=2))
    LOGGER.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
