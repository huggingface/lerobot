# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Training-free *latent* time-advance: translate the cube in feature space.

This is the latent twin of :func:`shift.shift_cube_in_frame`. Instead of editing
RGB pixels and letting the policy re-encode, it advances the cube directly on the
vision encoder's **patch-token grid** and hands the time-advanced tokens to the
rest of the policy. The grid here is the encoder's *pre-shuffle* feature map
(row-major ``H*W`` patch tokens), so the warp is independent of any downstream
token-resampling (e.g. SmolVLM's connector pixel-shuffle).

This realises, for the conveyor's near-rigid cube motion, the
predict-future-patch-tokens idea of latent world models (DINO-WM, LaDi-WM, AHEAD;
see ``docs/overhead-predictor.md``) with an *analytic* dynamics: a constant
optical-flow translation rather than a learned latent dynamics network. The flow
source is pluggable -- the analytic per-cube velocity from ``CubePredictor`` is
the default; a dense optical-flow estimator can later supply ``offset_px`` without
touching this module or its RTC integration.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812


def mask_to_token_grid(
    pixel_mask: torch.Tensor, grid_hw: tuple[int, int], threshold: float = 0.0
) -> torch.Tensor:
    """Down-sample a pixel mask to a ``grid_hw`` patch grid by area coverage.

    Args:
        pixel_mask: ``(H, W)`` boolean (or {0,1}) cube mask in the *same* image
            space the encoder sees (i.e. after the policy's resize/pad).
        grid_hw: ``(gh, gw)`` target patch-grid size.
        threshold: a patch is marked as cube when its fractional cube coverage is
            ``> threshold``. ``0.0`` means "any overlap counts".

    Returns a ``(gh, gw)`` boolean tensor.
    """
    gh, gw = grid_hw
    m = torch.as_tensor(pixel_mask, dtype=torch.float32)[None, None]
    pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]
    return pooled > threshold


def warp_token_grid(
    tokens: torch.Tensor,
    grid_hw: tuple[int, int],
    token_mask: torch.Tensor,
    offset_tokens: tuple[float, float],
    fill_token: torch.Tensor | None = None,
) -> torch.Tensor:
    """Translate the masked tokens by ``offset_tokens`` on a row-major patch grid.

    Mirrors :func:`shift.shift_cube_in_frame` but in feature space: erase the cube
    tokens from their current grid cells (filling the hole with a background
    token), then re-write the cube tokens shifted by ``offset_tokens``. Cells that
    would land outside the grid are dropped.

    Args:
        tokens: ``(B, N, D)`` or ``(N, D)`` patch tokens, ``N == gh * gw``,
            laid out row-major (token index ``= row * gw + col``).
        grid_hw: ``(gh, gw)`` patch-grid size.
        token_mask: ``(gh, gw)`` boolean cube mask on the patch grid.
        offset_tokens: ``(dx, dy)`` translation in patch units (``dx`` is columns
            / x, ``dy`` is rows / y). Rounded to the nearest integer.
        fill_token: optional ``(D,)`` or ``(B, D)`` fill for vacated cells. When
            ``None``, the per-feature median over the non-cube tokens is used (the
            feature-space analogue of the background-color fill in the pixel edit).

    Returns the warped tokens with the same shape/dtype/device as ``tokens``.
    """
    squeeze = tokens.dim() == 2
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens of shape (B, N, D) or (N, D), got {tuple(tokens.shape)}")

    bsz, n, d = tokens.shape
    gh, gw = grid_hw
    token_mask = torch.as_tensor(token_mask, dtype=torch.bool, device=tokens.device)
    if gh * gw != n:
        raise ValueError(f"grid {grid_hw} does not match token count {n}")
    if token_mask.shape != (gh, gw):
        raise ValueError(f"token_mask shape {tuple(token_mask.shape)} does not match grid {grid_hw}")

    dx = int(round(offset_tokens[0]))
    dy = int(round(offset_tokens[1]))
    if (dx == 0 and dy == 0) or not bool(token_mask.any()):
        return tokens.squeeze(0) if squeeze else tokens

    grid = tokens.view(bsz, gh, gw, d)
    out = grid.clone()

    # Background fill for the erased cube cells: per-feature median of non-cube tokens.
    if fill_token is None:
        bg = grid[:, ~token_mask]  # (B, n_bg, D)
        # Per-feature median of the non-cube tokens (mean fallback if all cube).
        fill = bg.median(dim=1).values if bg.shape[1] > 0 else grid.reshape(bsz, -1, d).mean(dim=1)
    else:
        fill = fill_token if fill_token.dim() == 2 else fill_token.unsqueeze(0).expand(bsz, -1)
    n_cube = int(token_mask.sum())
    out[:, token_mask] = fill.unsqueeze(1).expand(bsz, n_cube, d).to(out.dtype)

    # Re-write cube tokens at the shifted cells, dropping out-of-grid targets.
    ys, xs = torch.nonzero(token_mask, as_tuple=True)  # rows (y), cols (x)
    tys, txs = ys + dy, xs + dx
    inside = (tys >= 0) & (tys < gh) & (txs >= 0) & (txs < gw)
    out[:, tys[inside], txs[inside], :] = grid[:, ys[inside], xs[inside], :]

    out = out.view(bsz, n, d)
    return out.squeeze(0) if squeeze else out


def warp_token_grid_by_flow(
    tokens: torch.Tensor,
    grid_hw: tuple[int, int],
    flow_tokens: torch.Tensor,
    motion_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance each patch token along *its own* flow on a row-major patch grid.

    Unlike :func:`warp_token_grid` (a single rigid translation of a masked blob),
    this applies a dense, per-patch displacement field -- the feature-space analogue
    of warping an image by an optical-flow field. Implemented as a backward bilinear
    sample (``grid_sample``): each target cell ``q`` is filled from source position
    ``q - flow[q]``, so feature content moves along ``+flow``.

    Args:
        tokens: ``(B, N, D)`` or ``(N, D)`` patch tokens, ``N == gh * gw``, row-major.
        grid_hw: ``(gh, gw)`` patch-grid size.
        flow_tokens: ``(gh, gw, 2)`` per-patch displacement ``(dx, dy)`` in *patch*
            units (already scaled to the desired time-advance horizon).
        motion_mask: optional ``(gh, gw)`` boolean. Where ``False``, the original
            token is kept (static regions are left untouched so flow noise on the
            background/idle arm does not perturb them). ``None`` warps every cell.

    Returns the warped tokens with the same shape/dtype/device as ``tokens``.
    """
    squeeze = tokens.dim() == 2
    if squeeze:
        tokens = tokens.unsqueeze(0)
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens of shape (B, N, D) or (N, D), got {tuple(tokens.shape)}")

    bsz, n, d = tokens.shape
    gh, gw = grid_hw
    if gh * gw != n:
        raise ValueError(f"grid {grid_hw} does not match token count {n}")
    flow = torch.as_tensor(flow_tokens, dtype=tokens.dtype, device=tokens.device)
    if flow.shape != (gh, gw, 2):
        raise ValueError(f"flow_tokens shape {tuple(flow.shape)} does not match grid {grid_hw} + 2")

    feat = tokens.view(bsz, gh, gw, d).permute(0, 3, 1, 2)  # (B, D, gh, gw)

    # Base sampling coords in grid_sample's normalised [-1, 1] space (align_corners).
    ys = torch.linspace(-1.0, 1.0, gh, dtype=feat.dtype, device=feat.device)
    xs = torch.linspace(-1.0, 1.0, gw, dtype=feat.dtype, device=feat.device)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")  # (gh, gw)
    step_x = 2.0 / (gw - 1) if gw > 1 else 0.0  # normalised units per patch column
    step_y = 2.0 / (gh - 1) if gh > 1 else 0.0
    samp_x = base_x - flow[..., 0] * step_x  # backward sample -> content moves +flow
    samp_y = base_y - flow[..., 1] * step_y
    coords = torch.stack([samp_x, samp_y], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)

    warped = F.grid_sample(feat, coords, mode="bilinear", padding_mode="border", align_corners=True)
    warped = warped.permute(0, 2, 3, 1)  # (B, gh, gw, D)

    if motion_mask is not None:
        mm = torch.as_tensor(motion_mask, dtype=torch.bool, device=feat.device)
        out = torch.where(mm[None, ..., None], warped, tokens.view(bsz, gh, gw, d))
    else:
        out = warped

    out = out.reshape(bsz, n, d)
    return out.squeeze(0) if squeeze else out


def make_token_warp_fn(
    pixel_mask: torch.Tensor,
    offset_px: tuple[float, float],
    image_hw: tuple[int, int],
    mask_threshold: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a ``tokens -> tokens`` warp closure for one camera's patch tokens.

    The returned callable is meant to run inside the policy's image encoder, on the
    pre-shuffle patch-token grid (``(B, N, D)`` with square ``N``). It derives the
    patch-grid side from ``N`` at call time and projects ``pixel_mask`` / ``offset_px``
    -- both expressed in the encoder-input image space ``image_hw`` -- onto that grid,
    so it adapts to whatever resolution the encoder uses. A non-square ``N`` (rare,
    e.g. variable-resolution encoders) is left unchanged.

    Args:
        pixel_mask: ``(H, W)`` boolean cube mask in encoder-input image space.
        offset_px: ``(dx, dy)`` cube displacement in encoder-input pixels.
        image_hw: ``(H, W)`` of the encoder-input image (matches ``pixel_mask``).
        mask_threshold: forwarded to :func:`mask_to_token_grid`.
    """
    img_h, img_w = image_hw

    def warp(tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[-2]
        side = int(round(n**0.5))
        if side * side != n:
            return tokens  # non-square grid: skip rather than guess the layout
        token_mask = mask_to_token_grid(pixel_mask, (side, side), mask_threshold)
        offset_tokens = (offset_px[0] * side / img_w, offset_px[1] * side / img_h)
        return warp_token_grid(tokens, (side, side), token_mask, offset_tokens)

    return warp


def make_flow_token_warp_fn(
    flow_px: torch.Tensor,
    image_hw: tuple[int, int],
    motion_threshold: float = 0.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a dense per-patch flow warp closure for one camera's patch tokens.

    ``flow_px`` is the *time-advance* displacement field (already scaled to the lead
    horizon, e.g. ``per_frame_flow * lead_s / dt``) in encoder-input pixels. The
    closure pools it onto the patch grid derived from ``N`` at call time, converts
    pixels to patch units, optionally builds a motion mask from the per-patch flow
    magnitude, and applies :func:`warp_token_grid_by_flow`. A non-square ``N`` is
    left unchanged.

    Args:
        flow_px: ``(H, W, 2)`` displacement field ``(dx, dy)`` in encoder-input pixels.
        image_hw: ``(H, W)`` of the encoder-input image (matches ``flow_px``).
        motion_threshold: patch-unit flow magnitude below which a patch is treated as
            static and left untouched. ``0`` warps every patch.
    """
    img_h, img_w = image_hw
    flow = torch.as_tensor(flow_px, dtype=torch.float32)  # (H, W, 2)

    def warp(tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[-2]
        side = int(round(n**0.5))
        if side * side != n:
            return tokens  # non-square grid: skip rather than guess the layout
        # Pool the (H, W, 2) field onto the (side, side) patch grid.
        pooled = F.adaptive_avg_pool2d(flow.permute(2, 0, 1).unsqueeze(0), (side, side))
        pooled = pooled[0].permute(1, 2, 0)  # (side, side, 2)
        flow_tokens = torch.stack(
            (pooled[..., 0] * side / img_w, pooled[..., 1] * side / img_h), dim=-1
        ).to(device=tokens.device)
        motion_mask = None
        if motion_threshold > 0:
            motion_mask = flow_tokens.norm(dim=-1) > motion_threshold
        return warp_token_grid_by_flow(tokens, (side, side), flow_tokens, motion_mask)

    return warp
