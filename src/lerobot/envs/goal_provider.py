"""Environment-specific goal observation extraction for latent-space planning.

Each provider returns a goal observation dict in the same format that
`preprocess_observation` expects (i.e. the raw numpy arrays keyed by the
env's native keys).  The eval rollout can then pass the result through the
same preprocessor pipeline before calling ``policy.set_goal``.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


class BaseGoalProvider(ABC):
    """Return a goal observation dict for each env in a VectorEnv."""

    @abstractmethod
    def get_goal_obs(self, vec_env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        """Produce goal observations matching the env's native observation format.

        Args:
            vec_env: A Gymnasium VectorEnv whose underlying single envs support
                the provider's goal extraction logic.

        Returns:
            Dict of numpy arrays with the same keys and shapes as ``env.reset()``
            but batched: each array has shape ``(num_envs, *feature_shape)``.
        """
        ...


def _render_pusht_goal_scene(base) -> np.ndarray:
    """Render a goal-achieved image for a single PushT env without touching physics.

    Replicates the env's internal ``_draw()`` / ``_get_img()`` pipeline but draws
    the T-block at the goal pose directly using the same ``get_goal_pose_body()``
    helper the env uses for the green goal ghost — bypassing ``space.debug_draw()``.
    This avoids the need to move the block body or step the physics engine.

    Returns:
        (H, W, 3) uint8 numpy array at the env's observation resolution.
    """
    import pygame
    import pymunk.pygame_util

    # Exact colors used by the env's debug_draw (inspected from shape.color).
    _BLOCK_COLOR = (119, 136, 153, 255)   # slate-grey
    _AGENT_COLOR = (65, 105, 225, 255)    # royal-blue
    _AGENT_RADIUS = 15

    screen = pygame.Surface((512, 512))
    screen.fill((255, 255, 255))
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Build a virtual body at the goal pose (CoG = 0, same as the env's ghost body).
    goal_body = base.get_goal_pose_body(base.goal_pose)

    # 1. Green goal ghost — identical to _draw().
    for shape in base.block.shapes:
        pts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
        pts = [pymunk.pygame_util.to_pygame(p, screen) for p in pts]
        pygame.draw.polygon(screen, pygame.Color("LightGreen"), pts + [pts[0]])

    # 2. Block drawn at goal position using the same goal_body transform.
    for shape in base.block.shapes:
        pts = [goal_body.local_to_world(v) for v in shape.get_vertices()]
        pts = [pymunk.pygame_util.to_pygame(p, screen) for p in pts]
        pygame.draw.polygon(screen, pygame.Color(*_BLOCK_COLOR), pts + [pts[0]])

    # 3. Agent at its current (live) position.
    agent_px = pymunk.pygame_util.to_pygame(base.agent.position, screen)
    pygame.draw.circle(screen, pygame.Color(*_AGENT_COLOR), agent_px, _AGENT_RADIUS)

    # Rescale to observation resolution — same as _get_img().
    return base._get_img(screen, base.observation_width, base.observation_height)


class PushTGoalProvider(BaseGoalProvider):
    """Goal provider for gym-pusht environments.

    Renders what the env looks like when the task is solved (block at goal pose)
    without moving any physics bodies or stepping the simulation.
    """

    def get_goal_obs(self, vec_env: gym.vector.VectorEnv) -> dict[str, np.ndarray]:
        images: list[np.ndarray] = []
        states: list[np.ndarray] = []

        if not hasattr(vec_env, "envs"):
            raise RuntimeError(
                "PushTGoalProvider requires a SyncVectorEnv with an .envs attribute."
            )

        for env in vec_env.envs:
            base = env.unwrapped
            img = _render_pusht_goal_scene(base)
            images.append(img)
            states.append(np.array(base.agent.position, dtype=np.float64))

        return {
            "pixels": np.stack(images, axis=0),    # (B, H, W, C)
            "agent_pos": np.stack(states, axis=0),  # (B, 2)
        }


def make_goal_provider(env_type: str) -> BaseGoalProvider:
    """Return the appropriate BaseGoalProvider for the given env type.

    Args:
        env_type: String env type matching those used in LeRobot eval configs
            (e.g. ``"pusht"``).

    Raises:
        ValueError: If no provider is registered for the given env type.
    """
    if env_type == "pusht":
        return PushTGoalProvider()
    raise ValueError(
        f"No goal provider implemented for env_type={env_type!r}. "
        "Supported: ['pusht']"
    )
