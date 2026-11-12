# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/sampler.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Timestep grids, guidance and noise of the reverse-flow sampler.

The sampler moves a noise sample from t = 1 to t = 0 along the predicted
velocity field (:meth:`praxis.model.PraxisModel.sample_window_joint`). This
module holds no neural network code.

Conventions:

* Noising is ``x_t = (1 - t) * x0 + t * eps``. The models predict the
  velocity ``v = eps - x0``.
* One Euler step is ``x + dt * v`` with ``dt = t_next - t_cur``. ``dt`` is
  negative.
* The window sampler draws its noise with :func:`seeded_normal` on the CPU
  and steps on the float64 grid of :func:`window_time_grid`.
* :func:`sampler_timesteps`, :func:`action_substep_grid` and
  :func:`sampler_num_evals` validate the sampler settings of a server
  session.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import torch

SOLVERS: tuple[str, ...] = ("euler", "heun")


def sampler_timesteps(num_steps: int, t_shift: float = 1.0) -> torch.Tensor:
    """Return the descending float32 grid ``(num_steps + 1,)`` from 1 to 0.

    The grid is ``u = linspace(1, 0, num_steps + 1)`` warped by the flow shift
    ``t = s * u / (1 + (s - 1) * u)``. ``s = 1`` returns ``u`` unchanged. The
    end points are exactly 1 and 0 for every ``s``.
    """
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    s = float(t_shift)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(f"t_shift must be a finite float > 0, got {t_shift!r}")
    u = torch.linspace(1.0, 0.0, num_steps + 1)
    if s == 1.0:
        return u
    return s * u / (1.0 + (s - 1.0) * u)


def action_substep_grid(
    num_steps: int, action_steps: int | None, t_shift: float = 1.0
) -> tuple[int, torch.Tensor]:
    """Return ``(m, ts)`` for expert-only sub-steps.

    The expert takes ``m = action_steps // num_steps`` Euler steps per backbone
    step on the finer grid ``ts = sampler_timesteps(action_steps, t_shift)``.
    The sub-steps read the taps of one backbone call again. ``action_steps``
    ``None`` or equal to ``num_steps`` gives ``m = 1`` (the joint sampler).
    """
    num_steps = int(num_steps)
    if action_steps is None or int(action_steps) == num_steps:
        return 1, sampler_timesteps(num_steps, t_shift)
    action_steps = int(action_steps)
    if action_steps < num_steps or action_steps % num_steps != 0:
        raise ValueError(f"action_steps={action_steps} must be a positive multiple of num_steps={num_steps}")
    return action_steps // num_steps, sampler_timesteps(action_steps, t_shift)


def sampler_num_evals(num_steps: int, solver: str = "euler", action_steps: int | None = None) -> int:
    """Return the number of backbone calls per chunk and validate the knobs.

    ``euler`` makes one call per step. ``heun`` makes a predictor call and a
    corrector call per step. The last step to t = 0 stays Euler, so heun makes
    ``2 * num_steps - 1`` calls. Heun refuses expert-only sub-steps.
    """
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if solver == "euler":
        return num_steps
    if action_steps is not None and int(action_steps) != num_steps:
        raise ValueError(
            "solver='heun' has no action_steps sub-stepping form (action_steps="
            f"{action_steps}, num_steps={num_steps}); use solver='euler' or "
            "action_steps == num_steps"
        )
    if num_steps == 1:
        logging.getLogger(__name__).warning(
            "solver=heun at num_steps=1 is plain euler (the final step to t=0 "
            "takes no corrector); num_evals=1"
        )
    return 2 * num_steps - 1


def guidance_combine(v_cond: torch.Tensor, v_uncond: torch.Tensor, scale: float) -> torch.Tensor:
    """Return the guided velocity ``v_u + scale * (v_c - v_u)`` in float32.

    ``scale == 1.0`` returns ``v_cond`` itself, with no cast and no
    arithmetic, so a sampler without guidance keeps its exact bits.
    """
    if float(scale) == 1.0:
        return v_cond
    return v_uncond.float() + float(scale) * (v_cond.float() - v_uncond.float())


def check_guidance(cfg_scale: float, cfg_video_scale: float) -> bool:
    """Validate the two guidance scales. Return True if guidance is on.

    Guidance is off when both scales are 1.0. A scale must be finite and not
    negative. A scale of 0.0 gives the unconditional velocity.
    """
    for name, value in (("cfg_scale", cfg_scale), ("cfg_video_scale", cfg_video_scale)):
        v = float(value)
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"{name} must be a finite float >= 0, got {value!r}")
    return not (float(cfg_scale) == 1.0 and float(cfg_video_scale) == 1.0)


# ---------------------------------------------------------------------------
# The window joint sampler of the checkpoint
# ---------------------------------------------------------------------------

#: Salt of the video noise and of the action noise (see :func:`seeded_normal`).
VIDEO_SALT = 0
ACTION_SALT = 1


def window_time_grid(num_steps: int, shift: float = 1.0) -> list[float]:
    """Return the descending grid of the window sampler as Python floats.

    The grid is ``u = linspace(1, 0, num_steps + 1)`` in float64, warped by
    ``t = s * u / (1 + (s - 1) * u)``. ``shift`` 1 keeps ``u``. The values
    stay float64 (:func:`sampler_timesteps` rounds them to float32).
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    ts = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    if shift != 1.0:
        ts = shift * ts / (1.0 + (shift - 1.0) * ts)
    return [float(t) for t in ts]


def seeded_normal(shape: Sequence[int], seed: int, salt: int) -> torch.Tensor:
    """Return float32 standard normal noise drawn on the CPU.

    The generator seed is ``(seed * 1000003 + salt) & (2**63 - 1)``. The
    window sampler draws the video noise with salt :data:`VIDEO_SALT` and the
    action noise with salt :data:`ACTION_SALT`, each from its own generator.
    """
    s = (int(seed) * 1000003 + int(salt)) & 0x7FFFFFFFFFFFFFFF
    gen = torch.Generator().manual_seed(s)
    return torch.randn(tuple(shape), generator=gen)
