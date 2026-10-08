# Copyright 2026 Black Forest Labs. All rights reserved.
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
"""Observation window -> the exact DiT inputs, plus the flow-matching loss.

Vendored from black-forest-labs/flux-action (``src/flux_action/processing/packing.py``). Policy settings
(camera layout, canvas, action dim, chunk, fps, action scale and loss weights) live in Flux3Config and
are passed explicitly. Constants here describe the VAE, token format and augmentation implementation.

Pipeline for one training window (``chunk + 1`` frames: 1 conditioning + ``chunk`` predicted):

    cameras uint8 (n_cams, T, 3, H, W)
      -> materialize_video      (augment, compose one canvas, [-1, 1])
      -> encode_video           (video VAE, pad to 45 frames, crop the latent to the content)
      -> pack_video             (latent frame 0 = x_video_cond, frames 1.. = x_video)
    state[s], action[s .. s+chunk-1]
      -> pack_actions           (x action_scale, audio-style ids on the shared 10 ms clock)
    caption
      -> text context           (see text_encoder.text_context), pack_text
    -> sample_timesteps + add_noise (one t per sample, shared by video and action)
    -> build_forward_kwargs / flow_loss
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.v2.functional as tvf
from einops import repeat
from torch import Tensor

from ..utils import compose_grid, rational_time_shift
from .positional import batched_prc_audio, batched_prc_txt, batched_prc_vid, times_to_ids

if TYPE_CHECKING:
    from ..configuration_flux3 import Flux3Config

# ---- VAE and token format ------------------------------------------------------------------
LATENT_CHANNELS = 96
SPATIAL_DOWNSAMPLE = 32
TEMPORAL_DOWNSAMPLE = 4
VAE_CHUNK_FRAMES = 45  # the video VAE encodes 45-frame chunks
VAE_PAD_VALUE = -1.0  # short clips are padded with black to the chunk length
CAPTION_SEPARATOR = " | "  # Cosmos3-DROID stores several paraphrases of one instruction in a single string
VEC_DIM = 768
AUG_CROP_FRACTION = 0.95
AUG_BRIGHTNESS, AUG_CONTRAST, AUG_SATURATION, AUG_HUE = 0.3, 0.4, 0.5, 0.08

CAMERA_LAYOUTS = ("droid", "single", "side_by_side", "grid")


def latent_frames(num_frames: int, temporal_downsample: int = TEMPORAL_DOWNSAMPLE) -> int:
    return 1 + (num_frames - 1) // temporal_downsample


def padded_chunk_length(num_frames: int, chunk: int, overlap: int = 1) -> int:
    """Smallest ``T' >= num_frames`` satisfying ``T' = chunk + n * (chunk - overlap)``."""
    stride = chunk - overlap
    n = max(0, -(-(num_frames - chunk) // stride))  # ceil division
    return chunk + n * stride


def latent_hw(content_hw: tuple[int, int]) -> tuple[int, int]:
    """Latent grid kept after the VAE: ceil(content / 32); DROID composite 540x640 -> 17x20."""
    h, w = content_hw
    return (-(-h // SPATIAL_DOWNSAMPLE), -(-w // SPATIAL_DOWNSAMPLE))


def select_caption(caption: str, training: bool) -> str:
    """One paraphrase of a stored caption: a random one in training (the reference recipe), else the first."""
    alternatives = caption.split(CAPTION_SEPARATOR)
    if training and len(alternatives) > 1:
        return alternatives[int(torch.randint(len(alternatives), ()))]
    return alternatives[0]


# ---- 1. pixels ------------------------------------------------------------------------------
def sample_augmentation(generator: torch.Generator | None = None, *, camera_hw: tuple[int, int]) -> dict:
    """Draw the Cosmos crop + color-jitter parameters for one window.

    One draw is shared by all cameras and all frames (so the model cannot read the augmentation as
    motion). Draw order mirrors torchvision's ColorJitter: crop top, crop left, function permutation,
    then the factors.
    """
    h, w = camera_hw
    crop_h, crop_w = int(h * AUG_CROP_FRACTION), int(w * AUG_CROP_FRACTION)
    top = int(torch.randint(0, h - crop_h + 1, (), generator=generator))
    left = int(torch.randint(0, w - crop_w + 1, (), generator=generator))
    order = [int(i) for i in torch.randperm(4, generator=generator).tolist()]

    def uniform(lo: float, hi: float) -> float:
        return float(torch.empty(1).uniform_(lo, hi, generator=generator).item())

    return {
        "crop_top": top,
        "crop_left": left,
        "crop_height": crop_h,
        "crop_width": crop_w,
        "color_fn_order": order,
        "brightness_factor": uniform(max(0.0, 1 - AUG_BRIGHTNESS), 1 + AUG_BRIGHTNESS),
        "contrast_factor": uniform(max(0.0, 1 - AUG_CONTRAST), 1 + AUG_CONTRAST),
        "saturation_factor": uniform(max(0.0, 1 - AUG_SATURATION), 1 + AUG_SATURATION),
        "hue_factor": uniform(-AUG_HUE, AUG_HUE),
    }


def apply_augmentation(x: Tensor, a: dict) -> Tensor:
    """``x`` float ``(N, 3, H, W)`` in ``[0, 1]`` -> same shape, cropped/resized back + color jitter."""
    h, w = x.shape[-2:]
    x = tvf.crop(x, top=a["crop_top"], left=a["crop_left"], height=a["crop_height"], width=a["crop_width"])
    x = tvf.resize(x, size=[h, w], antialias=True)
    for fn in a["color_fn_order"]:
        if fn == 0:
            x = tvf.adjust_brightness(x, a["brightness_factor"])
        elif fn == 1:
            x = tvf.adjust_contrast(x, a["contrast_factor"])
        elif fn == 2:
            x = tvf.adjust_saturation(x, a["saturation_factor"])
        else:
            x = tvf.adjust_hue(x, a["hue_factor"])
    return x


def compose_canvas(cams: Tensor, layout: str, canvas_hw: tuple[int, int]) -> Tensor:
    """``cams`` float ``(n_cams, T, 3, H, W)`` in ``[0, 1]`` -> canvas ``(3, T, Hc, Wc)`` in ``[-1, 1]``.

    ``droid``: three 360x640 cameras [wrist, left, right] -> wrist full-res on top, exteriors half-res
    side by side below (540x640) -> reflect-pad right/bottom to the canvas.
    ``single``: one camera, resized to the canvas. ``side_by_side``: two cameras, each half the width.
    ``grid``: any number of cameras in a near-square grid (``compose_grid``).
    What binds a checkpoint is that training and deployment use the same layout and canvas.
    """
    n_cams, t, c, h, w = cams.shape
    ch, cw = canvas_hw
    if layout == "droid":
        if n_cams != 3:
            raise ValueError(f"droid layout needs 3 cameras, got {n_cams}")
        wrist, left, right = cams.unbind(0)
        half = (h // 2, w // 2)
        left = F.interpolate(left, size=half, mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=half, mode="bilinear", align_corners=False)
        composite = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)  # (T, 3, 540, 640)
        pad_right, pad_bottom = cw - composite.shape[-1], ch - composite.shape[-2]
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError(
                f"canvas {canvas_hw} smaller than the DROID composite {tuple(composite.shape[-2:])}"
            )
        canvas = F.pad(composite, (0, pad_right, 0, pad_bottom), mode="reflect")
    elif layout == "side_by_side":
        if n_cams != 2 or cw % 2:
            raise ValueError("side_by_side requires two cameras and an even canvas width")
        canvas = torch.cat(
            [
                F.interpolate(cam, size=(ch, cw // 2), mode="bilinear", align_corners=False, antialias=True)
                for cam in cams
            ],
            dim=-1,
        )
    elif layout == "single":
        if n_cams != 1:
            raise ValueError(f"single layout needs exactly one camera, got {n_cams}")
        canvas = F.interpolate(cams[0], size=(ch, cw), mode="bilinear", align_corners=False, antialias=True)
    elif layout == "grid":
        canvas = compose_grid(cams, canvas_hw)
    else:
        raise ValueError(f"unknown camera layout {layout!r}; choose from {CAMERA_LAYOUTS}")
    return canvas.permute(1, 0, 2, 3).contiguous().mul_(2.0).sub_(1.0)


def materialize_video(
    cameras: Tensor,
    augmentation: dict | None,
    device: torch.device | str,
    *,
    layout: str,
    canvas_hw: tuple[int, int],
) -> Tensor:
    """``cameras`` uint8 or float ``(n_cams, T, 3, H, W)`` -> canvas ``(3, T, Hc, Wc)`` in ``[-1, 1]``.

    Deterministic given ``augmentation`` (None = no augmentation, the inference path). uint8 is
    scaled by 255; float input is taken as already in ``[0, 1]``.
    """
    if cameras.ndim != 5:
        raise ValueError(f"cameras must be (n_cams, T, 3, H, W), got {tuple(cameras.shape)}")
    n_cams, t, c, h, w = cameras.shape
    x = cameras.to(device=device, non_blocking=True).reshape(-1, c, h, w)
    x = x.float().div_(255.0) if cameras.dtype == torch.uint8 else x.float()
    if augmentation is not None:
        x = apply_augmentation(x, augmentation)
    return compose_canvas(x.reshape(n_cams, t, c, h, w), layout, canvas_hw)


# ---- 2. video VAE ---------------------------------------------------------------------------
@torch.no_grad()
def encode_video(
    vae,
    video: Tensor,
    latent_hw: tuple[int, int],
    *,
    chunk_frames: int = VAE_CHUNK_FRAMES,
    pad_value: float = VAE_PAD_VALUE,
) -> Tensor:
    """``(3, T, Hc, Wc)`` in ``[-1, 1]`` -> latents ``(1, 96, latent_frames(T), *latent_hw)``.

    The clip is padded at the END with black to the VAE's 45-frame chunk, the whole padded canvas is
    encoded (the VAE sees the mirrored borders), then the padded latent frames and the padded latent
    columns/rows are dropped.
    """
    c, t, h, w = video.shape
    total = chunk_frames if t < chunk_frames else padded_chunk_length(t, chunk_frames)
    if total > t:
        pad = torch.full((c, total - t, h, w), pad_value, dtype=video.dtype, device=video.device)
        video = torch.cat([video, pad], dim=1)
    lat = vae.encode(video[None].to(torch.bfloat16))  # (1, 96, latent_frames(total), H/32, W/32)
    lat = lat[:, :, : latent_frames(t), : latent_hw[0], : latent_hw[1]]
    return lat.clone()  # leave the VAE's inference_mode so autograd may consume it


@torch.no_grad()
def encode_single_frame(vae, frame: Tensor, latent_hw: tuple[int, int]) -> Tensor:
    """``(3, Hc, Wc)`` -> ``(1, 96, 1, *latent_hw)``: the inference-time conditioning latent.

    Only the current frame is supplied. The production VAE repeats it to a 45-frame chunk,
    encodes that chunk and retains its first latent frame (the reference inference behavior).
    """
    lat = vae.encode(frame[None, :, None].to(torch.bfloat16))  # (1, 96, 1, H/32, W/32)
    return lat[..., : latent_hw[0], : latent_hw[1]].clone()


# ---- 3. token packing (position ids on one 10 ms clock) -------------------------------------
def video_time_ids(
    n_latent: int,
    first_latent: int,
    batch: int,
    *,
    fps: float,
    temporal_downsample: int = TEMPORAL_DOWNSAMPLE,
) -> Tensor:
    """Latent frame ``i`` sits at time ``i * 4 / fps`` s (frame ``i*4`` of the window)."""
    seconds = torch.arange(first_latent, first_latent + n_latent).float() * temporal_downsample / fps
    return times_to_ids(repeat(seconds, "t -> b t", b=batch))


def pack_video(latents: Tensor, *, fps: float) -> dict[str, Tensor]:
    """``(B, 96, n, h, w)`` -> ``x_video_cond`` (frame 0) + ``x_video`` (frames 1..) with ids."""
    b, _, n, _, _ = latents.shape
    latents = latents.to(torch.bfloat16)
    cond, ids_c = batched_prc_vid(latents[:, :, :1], video_time_ids(1, 0, b, fps=fps))
    pred, ids_p = batched_prc_vid(latents[:, :, 1:], video_time_ids(n - 1, 1, b, fps=fps))
    return {"x_video_cond": cond, "x_video_cond_ids": ids_c, "x_video": pred, "x_video_ids": ids_p}


def pack_actions(
    state: Tensor,
    actions: Tensor | None,
    action_times_s: Tensor | None,
    modality: str,
    *,
    scale: float,
    targets: bool = True,
) -> dict[str, Tensor]:
    """``state (B, 1, D)``, ``actions (B, K, D)``, ``times (B, K)`` s since the frame -> scaled tokens + ids.

    Audio-style packing: ids ``(t, 0, 0, 0)``. The state token sits at ``t = 0``, each action at the
    frame it produces (DROID: ``k / 15`` s -> ids 6, 13, 20, 26, ...).
    """
    b = state.shape[0]
    x_cond, ids_cond = batched_prc_audio(
        (state * scale).transpose(1, 2), times_to_ids(torch.zeros(b, 1, device=state.device))
    )
    result = {f"x_{modality}_cond": x_cond.float(), f"x_{modality}_cond_ids": ids_cond}
    if not targets:
        return result
    if actions is None or action_times_s is None:
        raise ValueError("Action targets and their times are required when targets=True")
    x_act, ids_act = batched_prc_audio(
        (actions * scale).transpose(1, 2), times_to_ids(action_times_s.to(actions.device))
    )
    result.update({f"x_{modality}": x_act.to(torch.bfloat16), f"x_{modality}_ids": ids_act})
    return result


def default_action_times(batch: int, chunk: int, fps: float) -> Tensor:
    """Action ``k`` (0-based) produces frame ``k + 1``: times ``(k + 1) / fps``."""
    return repeat((torch.arange(chunk).float() + 1) / fps, "t -> b t", b=batch)


def pack_text(ctx: Tensor, vec_dim: int = VEC_DIM) -> dict[str, Tensor]:
    """``ctx (B, L, 20480)`` -> ctx ids on the l axis (t = h = w = 0), zero ctx timesteps, zero vector."""
    _, ctx_ids = batched_prc_txt(ctx)
    b = ctx.shape[0]
    return {
        "ctx": ctx,
        "ctx_ids": ctx_ids.to(ctx.device),
        "timesteps_ctx": torch.zeros(ctx.shape[:2], device=ctx.device),
        "vector": torch.zeros(b, vec_dim, device=ctx.device),
    }


# ---- 4. flow matching -----------------------------------------------------------------------
def sample_timesteps(
    batch: int,
    generator: torch.Generator | None = None,
    *,
    width: float,
    shift: float,
) -> Tensor:
    """Training timestep: ``logit(t) ~ Logistic(0, width)``, then the rational shift.

    ONE ``t`` per sample, shared by the video tokens and the action tokens.
    """
    eps = torch.finfo(torch.float32).eps
    u = torch.rand(batch, generator=generator).clamp(eps, 1 - eps)
    t = torch.sigmoid(width * (torch.log(u) - torch.log1p(-u)))
    return rational_time_shift(t, shift)


def add_noise(x0: Tensor, t: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """``x_t = t * eps + (1 - t) * x0``; velocity target ``eps - x0``. ``t`` is ``(B,)``.

    Everything is computed in ``x0.dtype`` after rounding ``t`` and ``eps`` to it: the reference packs
    latents, noise and timesteps in bfloat16 and noises in that dtype, so a bf16 window sees ``t`` on
    the bf16 grid (spacing 2^-8 just below 1; ``t > 1 - 2^-9`` becomes exactly 1).
    """
    eps_ = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(
        device=x0.device, dtype=x0.dtype
    )
    tt = t.to(device=x0.device, dtype=x0.dtype).view(-1, *([1] * (x0.ndim - 1)))
    x_t = tt * eps_ + (1 - tt) * x0
    return x_t, eps_ - x0


def build_forward_kwargs(
    video: dict[str, Tensor],
    action: dict[str, Tensor],
    text: dict[str, Tensor],
    t: Tensor,
    modality: str,
    generator: torch.Generator | None = None,
    *,
    action_timesteps: Tensor | None = None,
    conditioning_noise_max: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Assemble one training call: ``(forward_kwargs, flow_targets)``.

    Conditioning streams (``x_video_cond``, ``x_<modality>_cond``) stay clean at timestep 0. Predicted
    streams (``x_video``, ``x_<modality>``) get the same ``t`` (or ``action_timesteps`` for the action
    stream), rounded to the latent dtype by :func:`add_noise`; the timesteps handed to the model are fp32
    tensors holding that rounded value.
    """
    ak = f"x_{modality}"
    t_action = t if action_timesteps is None else action_timesteps
    xv, tv = add_noise(video["x_video"], t, generator)
    xa, ta = add_noise(action[ak], t_action, generator)
    dev = xv.device
    tt = t.to(device=dev, dtype=xv.dtype).float()  # the rounded value the noising used
    tt_action = t_action.to(device=dev, dtype=xa.dtype).float()
    kw = {
        "x_video": xv,
        "x_video_ids": video["x_video_ids"].to(dev),
        "x_video_timesteps": tt[:, None] * torch.ones(xv.shape[:2], device=dev),
        ak: xa,
        f"{ak}_ids": action[f"{ak}_ids"].to(dev),
        f"{ak}_timesteps": tt_action[:, None] * torch.ones(xa.shape[:2], device=dev),
        "x_video_cond": video["x_video_cond"],
        "x_video_cond_ids": video["x_video_cond_ids"].to(dev),
        "x_video_cond_timesteps": torch.zeros(video["x_video_cond"].shape[:2], device=dev),
        f"{ak}_cond": action[f"{ak}_cond"],
        f"{ak}_cond_ids": action[f"{ak}_cond_ids"].to(dev),
        f"{ak}_cond_timesteps": torch.zeros(action[f"{ak}_cond"].shape[:2], device=dev),
        **text,
    }
    if conditioning_noise_max:
        cond_t = torch.rand(t.shape, generator=generator) * conditioning_noise_max
        kw["x_video_cond"], _ = add_noise(video["x_video_cond"], cond_t, generator)
        cond_tt = cond_t.to(device=dev, dtype=kw["x_video_cond"].dtype).float()
        kw["x_video_cond_timesteps"] = cond_tt[:, None].expand(video["x_video_cond"].shape[:2])
    targets = {"x_video": tv, ak: ta}
    return kw, targets


def flow_loss(
    pred: dict[str, Tensor],
    targets: dict[str, Tensor],
    modality: str,
    action_weight: float,
    video_weight: float,
    *,
    reduction: str,
    channel_weights: list[float] | None = None,
) -> dict[str, Tensor]:
    """Configured joint-token or per-modality mean, with per-modality MSEs returned for logging.

    Weights depend on the reduction: DROID's joint-token action weight of 50 yields a per-modality
    ratio of ``32 * 50 / 2720 = 0.588``. It must not be reused unchanged with per-modality pooling.
    """
    ak = f"x_{modality}"
    video = ((pred["x_video"].float() - targets["x_video"].float()) ** 2).mean(-1)  # (B, Nv)
    squared_error = (pred[ak].float() - targets[ak].float()).square()
    action = squared_error.mean(-1)  # (B, Na)
    weighted_action = action
    if channel_weights is not None:
        weights = torch.tensor(channel_weights, device=action.device, dtype=torch.float32)
        weights = weights.square() / weights.square().mean()
        weighted_action = (squared_error * weights).mean(-1)
    if reduction == "modalities":
        total = video_weight * video.mean() + action_weight * weighted_action.mean()
    elif reduction == "joint_tokens":
        total = (video_weight * video.sum() + action_weight * weighted_action.sum()) / (
            video.numel() + action.numel()
        )
    else:
        raise ValueError(f"unknown loss reduction {reduction}")
    return {"loss": total, "video_mse": video.mean().detach(), "action_mse": action.mean().detach()}


def token_budget(chunk: int, latent_hw: tuple[int, int], text_tokens: int) -> dict[str, int]:
    """Token counts of one window (for sanity checks and batch sizing)."""
    hw = latent_hw[0] * latent_hw[1]
    n_pred = latent_frames(chunk + 1) - 1
    return {
        "x_video_cond": hw,
        "x_video": n_pred * hw,
        "x_action_cond": 1,
        "x_action": chunk,
        "ctx_min": text_tokens,
    }


@torch.no_grad()
def pack_history_video(
    vae,
    videos: Tensor,
    history: int,
    snapshots: int,
    fps: float,
    hw: tuple[int, int],
    *,
    targets: bool = True,
) -> dict[str, Tensor]:
    """Independent history snapshots and future clip, matching the task-LoRA VAE boundary."""
    b = videos.shape[0]
    tokens, ids = [], []
    indices = (
        [history - 1]
        if snapshots == 1
        else [round(j * (history - 1) / (snapshots - 1)) for j in range(snapshots)]
    )
    for i in indices:
        lat = torch.cat([vae.encode_task(v[None, :, i : i + 1].to(torch.bfloat16)).clone() for v in videos])
        lat = lat[..., : hw[0], : hw[1]]
        tok, pos = batched_prc_vid(lat, times_to_ids(torch.full((b, 1), i / fps)))
        tokens.append(tok)
        ids.append(pos)
    result = {"x_video_cond": torch.cat(tokens, 1), "x_video_cond_ids": torch.cat(ids, 1)}
    if targets:
        lat = torch.cat([vae.encode_task(v[None, :, history:].to(torch.bfloat16)).clone() for v in videos])
        lat = lat[..., : hw[0], : hw[1]]
        tok, pos = batched_prc_vid(lat, video_time_ids(lat.shape[2], latent_frames(history), b, fps=fps))
        result.update(x_video=tok, x_video_ids=pos)
    return result


def pack_history_actions(
    states: Tensor,
    past_actions: Tensor | None,
    actions: Tensor | None,
    modality: str,
    fps: float,
    scale: float,
    *,
    targets: bool = True,
) -> dict[str, Tensor]:
    """Unscaled state history, optionally concatenated with past actions, on the conditioning layer."""
    b, history, _ = states.shape
    values = states if past_actions is None else torch.cat([past_actions, states], -1)
    times = ((torch.arange(history).float() - (history - 1)) / fps).expand(b, -1)
    cond, cond_ids = batched_prc_audio(
        values.transpose(1, 2), times_to_ids(times), torch.full((b, 1), -1, dtype=torch.long)
    )
    result = {f"x_{modality}_cond": cond, f"x_{modality}_cond_ids": cond_ids}
    if not targets:
        return result
    if actions is None:
        raise ValueError("Action targets are required when targets=True")
    action, ids = batched_prc_audio(
        (actions * scale).transpose(1, 2),
        times_to_ids((torch.arange(actions.shape[1]).float() / fps).expand(b, -1)),
    )
    result.update({f"x_{modality}": action.to(torch.bfloat16), f"x_{modality}_ids": ids})
    return result


class Packer(NamedTuple):
    """Token layout shared by training and inference.

    ``conditioning`` identifies the existing frame/history input and processor contract. The six
    functions own head width, video/action tokens and sampling clocks. ``pack_video`` takes batched,
    already composed canvases ``(B, 3, T, H, W)`` so augmentation order and VAE batching stay unchanged.
    ``pack_actions`` takes states ``(B, H, D)``, optional past actions of that shape, and targets
    ``(B, K, D)``. With ``targets=False``, both packers return only conditioning tokens and ids;
    action targets can be omitted. Both return the named tensors consumed by the DiT.

    Register a variant made with ``FRAME._replace(...)`` / ``HISTORY._replace(...)`` for a different
    token layout within the same input contract. Select it with ``config.packer``; registration must
    run before loading its config. New processor/window semantics require their own support too.
    """

    conditioning: str
    conditioning_channels: Callable[[Flux3Config], int]
    predicted_latent_frames: Callable[[Flux3Config], int]
    predicted_video_times: Callable[[Flux3Config, int], Tensor]
    action_times: Callable[[Flux3Config, int], Tensor]
    pack_video: Callable[..., dict[str, Tensor]]
    pack_actions: Callable[..., dict[str, Tensor]]


PACKERS: dict[str, Packer] = {}


def register_packer(name: str, packer: Packer) -> Packer:
    """Register a token layout without replacing an existing checkpoint's layout."""
    if name in PACKERS:
        raise ValueError(f"packer {name!r} is already registered")
    if not isinstance(packer, Packer) or packer.conditioning not in ("frame", "history"):
        raise ValueError("a Packer must declare frame or history conditioning")
    PACKERS[name] = packer
    return packer


def build_packer(config: Flux3Config) -> Packer:
    """Select an explicit packer or default to the configured conditioning scheme."""
    name = config.packer or config.conditioning
    if name not in PACKERS:
        raise ValueError(f"unknown packer {name!r}; registered: {sorted(PACKERS)}")
    return PACKERS[name]


def frame_conditioning_channels(config: Flux3Config) -> int:
    return config.action_dim


def frame_predicted_latent_frames(config: Flux3Config) -> int:
    return latent_frames(config.window_frames) - 1


def frame_predicted_video_times(config: Flux3Config, batch: int) -> Tensor:
    return video_time_ids(
        frame_predicted_latent_frames(config), 1, batch, fps=config.video_position_fps or config.fps
    )


def frame_action_times(config: Flux3Config, batch: int) -> Tensor:
    return default_action_times(batch, config.chunk_size, config.fps)


@torch.no_grad()
def frame_pack_video(config: Flux3Config, vae: Any, videos: Tensor, *, targets: bool) -> dict[str, Tensor]:
    video_fps = config.video_position_fps or config.fps
    if targets:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=videos.device.type == "cuda"):
            latents = torch.cat([encode_video(vae, v, config.latent_hw) for v in videos])
            return pack_video(latents, fps=video_fps)
    # Preserve the reference inference encoder's repeated-frame behavior, without autocast.
    latents = torch.cat([encode_single_frame(vae, v[:, 0], config.latent_hw) for v in videos])
    cond, ids = batched_prc_vid(
        latents.to(torch.bfloat16), video_time_ids(1, 0, latents.shape[0], fps=video_fps)
    )
    return {"x_video_cond": cond, "x_video_cond_ids": ids}


def frame_pack_actions(
    config: Flux3Config,
    states: Tensor,
    past_actions: Tensor | None,
    actions: Tensor | None = None,
    *,
    targets: bool = True,
) -> dict[str, Tensor]:
    return pack_actions(
        states,
        actions,
        frame_action_times(config, states.shape[0]) if targets else None,
        config.action_modality,
        scale=config.action_scale,
        targets=targets,
    )


def history_conditioning_channels(config: Flux3Config) -> int:
    return config.action_dim * (2 if config.condition_on_past_actions else 1)


def history_predicted_latent_frames(config: Flux3Config) -> int:
    return latent_frames(config.chunk_size)


def history_predicted_video_times(config: Flux3Config, batch: int) -> Tensor:
    return video_time_ids(
        history_predicted_latent_frames(config),
        latent_frames(config.n_obs_steps),
        batch,
        fps=config.video_position_fps or config.fps,
    )


def history_action_times(config: Flux3Config, batch: int) -> Tensor:
    return (torch.arange(config.chunk_size).float() / config.fps).expand(batch, -1)


@torch.no_grad()
def history_pack_video(config: Flux3Config, vae: Any, videos: Tensor, *, targets: bool) -> dict[str, Tensor]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=videos.device.type == "cuda"):
        return pack_history_video(
            vae,
            videos,
            config.n_obs_steps,
            config.history_snapshots,
            config.video_position_fps or config.fps,
            config.latent_hw,
            targets=targets,
        )


def history_pack_actions(
    config: Flux3Config,
    states: Tensor,
    past_actions: Tensor | None,
    actions: Tensor | None = None,
    *,
    targets: bool = True,
) -> dict[str, Tensor]:
    return pack_history_actions(
        states,
        past_actions,
        actions,
        config.action_modality,
        config.fps,
        config.action_scale,
        targets=targets,
    )


FRAME = register_packer(
    "frame",
    Packer(
        "frame",
        frame_conditioning_channels,
        frame_predicted_latent_frames,
        frame_predicted_video_times,
        frame_action_times,
        frame_pack_video,
        frame_pack_actions,
    ),
)
HISTORY = register_packer(
    "history",
    Packer(
        "history",
        history_conditioning_channels,
        history_predicted_latent_frames,
        history_predicted_video_times,
        history_action_times,
        history_pack_video,
        history_pack_actions,
    ),
)
