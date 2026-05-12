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
from collections.abc import Mapping, Sequence
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
    log_every_steps: int


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


@dataclass
class SearchNode:
    state: PushTSnapshot
    observation: Mapping[str, Any]
    depth: int
    env_step: int
    root_actions: np.ndarray | None
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
    video_path: str | None


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

    def choose(
        self, observation: Mapping[str, Any], *, env_step: int, max_steps: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        root_state = self.adapter.snapshot()
        root = SearchNode(
            state=root_state,
            observation=observation,
            depth=0,
            env_step=env_step,
            root_actions=None,
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

        return best.root_actions, {
            "reason": "beam_search",
            "best_score": best.score,
            "best_depth": best.depth,
            "expanded_nodes": expanded,
            "selected_chunk_len": int(best.root_actions.shape[0]),
        }

    def _expand(self, node: SearchNode, *, max_steps: int) -> list[SearchNode]:
        self.adapter.restore(node.state)
        policy_chunk = self.policy.predict_chunk(node.observation, horizon=self.cfg.chunk_size)
        action_candidates = self.candidates.make(policy_chunk)

        children: list[SearchNode] = []
        for actions in action_candidates:
            self.adapter.restore(node.state)
            rollout = rollout_actions(
                adapter=self.adapter,
                actions=actions,
                start_step=node.env_step,
                max_steps=max_steps,
            )
            score = self.scorer.score(rollout)
            children.append(
                SearchNode(
                    state=rollout.state,
                    observation=rollout.observation,
                    depth=node.depth + 1,
                    env_step=node.env_step + rollout.action_count,
                    root_actions=actions if node.root_actions is None else node.root_actions,
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
    max_actions = min(len(actions), max(0, max_steps - start_step))

    for action in actions[:max_actions]:
        observation, reward, terminated, truncated, info = adapter.step(action)
        rewards.append(float(reward))
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
    )


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
        "--log-every-steps",
        "--log_every_steps",
        dest="log_every_steps",
        type=int,
        default=10,
        help="Print episode progress every N committed environment steps. Use 1 for detailed progress.",
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
    should_render = episode_index < cfg.render_videos
    if should_render:
        frames.append(
            annotate_frame(
                adapter.render(),
                [
                    "Search: On",
                    f"episode={episode_index} step=0/{max_steps}",
                    "reset",
                ],
            )
        )

    rewards: list[float] = []
    success = False
    terminated = False
    truncated = False
    env_step = 0

    while env_step < max_steps and not (terminated or truncated or success):
        root_state = adapter.snapshot()
        action_chunk, plan_info = planner.choose(observation, env_step=env_step, max_steps=max_steps)
        adapter.restore(root_state)

        commit_count = min(cfg.execute_steps, len(action_chunk), max_steps - env_step)
        if env_step == 0 or env_step % cfg.log_every_steps == 0:
            LOGGER.info(
                "episode=%s step=%s/%s planning_done reason=%s best_score=%s expanded_nodes=%s",
                episode_index,
                env_step,
                max_steps,
                plan_info.get("reason"),
                plan_info.get("best_score"),
                plan_info.get("expanded_nodes"),
            )
        for action_ix, action in enumerate(action_chunk[:commit_count]):
            observation, reward, terminated, truncated, info = adapter.step(action)
            rewards.append(float(reward))
            success = success or bool(info.get("is_success", False))
            env_step += 1
            if should_render:
                frames.append(
                    annotate_frame(
                        adapter.render(),
                        [
                            f"Search: {'On' if action_ix == 0 else 'Off'}",
                            f"episode={episode_index} step={env_step}/{max_steps}",
                            f"reward={reward:.3f} coverage={float(info.get('coverage', 0.0)):.3f}",
                        ],
                    )
                )
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
    if should_render and frames:
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
    payload = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "per_episode": [asdict(result) for result in results],
        "aggregated": {
            "avg_sum_reward": float(np.mean([result.sum_reward for result in results])) if results else 0.0,
            "avg_max_reward": float(np.mean([result.max_reward for result in results])) if results else 0.0,
            "pc_success": float(np.mean([result.success for result in results]) * 100.0) if results else 0.0,
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
