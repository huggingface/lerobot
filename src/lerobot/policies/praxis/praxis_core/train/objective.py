# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/train/objective.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Training objective of the praxis policy: the loss of one training window.

The objective is the one of the single-arm checkpoint (backbone
layout ``chunk_causal``). A window holds K chunks of F latents (K = 4, F = 6).
The backbone reads ``[registers | anchor | text | noised | clean]``
(``VideoBackbone.forward_chunk_causal``): the K noised chunks, each with its
own timestep, and the clean copies of chunks 0 to K-2 at timestep 0. Noised
chunk k attends the clean chunks before it, so one window trains every
context depth. The clean copies get no loss. The action expert reads the
encoder taps (``run_expert_on_taps_k``) and denoises the action tokens of all
K chunks in one pass.

The steps follow the original training code. ``Draws`` holds all random
values of one window:

* Video noise, two stages. First the window gets one timestep
  ``t0 = s * u0 / (1 + (s - 1) * u0)`` (``s`` is ``video_timestep_shift``)
  and one noise ``eps0`` (bf16): ``x_t = (1 - t0) * x0 + t0 * eps0`` in
  float32, rounded to bf16. The target is ``eps0 - x0`` in bf16. Then
  ``x0`` and ``eps`` are recovered from ``x_t``, the bf16 ``t0`` and the
  target in float32: ``x0' = x_t - t0 * v``, ``eps' = x0' + v``. Each chunk
  k gets its own timestep ``t_k`` (the same law, rounded to bf16) and is
  noised again: ``(1 - t_k) * x0' + t_k * eps'`` in float32, rounded to
  bf16. The clean copies are ``x0'`` in bf16. The target stays
  ``eps0 - x0``.
* Text dropout. With the probability ``text_dropout`` the text of the window
  becomes the encoded blank caption of the checkpoint config (``text_uncond``,
  for example 1 token). This trains the unconditional branch of text
  guidance.
* Proprio dropout. With the probability ``proprio_dropout`` the proprio of
  all chunks becomes zero.
* Actions. Each chunk gets a uniform action timestep (independent of the
  video), ``x_t = (1 - t) * x0 + t * eps`` and the target ``eps - x0``, in
  float32.
* RoPE offsets. The rotary positions move by four random offsets: the
  encoder anchor, the encoder text, the encoder video (the clean copies move
  with it) and the decoder video. Each offset is uniform below
  ``FRAME_STRIDE - span``. ``span`` is the position span of the input. In
  eval mode the offsets are zero, as at inference.

The loss of a window is::

    loss = video_weight * L_video + action_weight * L_action
           + moe_seq_balance_coef * L_balance

* ``L_video``: the squared error ``(v - (eps0 - x0))**2`` in bf16, summed and
  divided by the number of elements (one mean over all chunks and tokens).
* ``L_action``: the squared error of the action velocity in float32, the mean
  over the channels and then over the tokens. A window with a non-finite
  value or a value above ``action_guard`` gets an action loss of exactly 0
  (the original training code drops such a row).
* ``L_balance``: the sequence balance loss of the MoE router (see
  ``praxis.train.moe``), the mean over the MoE layers. Only in train mode.

The terms add in float32. The MoE routing biases change after each optimizer
step (``praxis.train.moe.update_expert_biases``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.policies.praxis.praxis_core.action_expert import run_expert_on_taps_k
from lerobot.policies.praxis.praxis_core.backbone import FRAME_STRIDE, ROW_STRIDE
from lerobot.policies.praxis.praxis_core.model import PraxisModel
from lerobot.policies.praxis.praxis_core.robot_io import (
    NormStats,
    contiguous_chunk_geometry,
    pack_action_tokens,
    proprio_normalize,
    q99_normalize,
)
from lerobot.policies.praxis.praxis_core.train.moe import take_sequence_balance_losses, track_routing
from lerobot.policies.praxis.praxis_core.vae import Autoencoder

if TYPE_CHECKING:
    from lerobot.policies.praxis.praxis_core.train.data import TrainingWindow

__all__ = [
    "EXPERT_FLEX_BACKWARD_OPTIONS",
    "Draws",
    "EncodedWindow",
    "FinetuneModel",
    "ObjectiveSettings",
    "TextEncode",
    "encode_window",
    "shifted_timesteps",
]


@dataclass
class ObjectiveSettings:
    """Settings of the training objective (the ``objective`` section of a run).

    The defaults are the values of ``configs/finetune_droid.yaml``.
    ``moe_balance_speed`` is the step of the routing bias update. When it is
    above 0, the sequence balance loss uses the expert picks without the
    routing bias. ``action_guard`` is the largest
    action loss of a window that trains (larger values give 0).
    ``moe_dispatch_chunk_size`` is the token chunk of the expert dispatch of
    the MoE layers in training (the checkpoint config holds the serve
    value).

    ``fixed_draws`` true gives a window the same random values each time it
    comes back. The trainer then seeds the draws from the seed and the
    ``window_id`` of the window, not from the step. Use it only for the
    overfit check on a fixed set of windows. With new draws in each step, the
    flow-matching loss cannot go to 0.
    """

    video_weight: float = 1.0
    action_weight: float = 1.0
    video_timestep_shift: float = 5.0
    text_dropout: float = 0.1
    proprio_dropout: float = 0.1
    moe_balance_speed: float = 1e-3
    moe_seq_balance_coef: float = 1e-4
    action_guard: float = 5.0
    moe_dispatch_chunk_size: int = 65536
    fixed_draws: bool = False

    def __post_init__(self) -> None:
        # A value from a YAML file can be a string ("1e-4" is a string in
        # YAML 1.1). Convert each number to a float.
        for f in fields(self):
            if f.name not in ("fixed_draws", "moe_dispatch_chunk_size"):
                setattr(self, f.name, float(getattr(self, f.name)))
        self.moe_dispatch_chunk_size = int(self.moe_dispatch_chunk_size)
        if self.moe_dispatch_chunk_size < 1:
            raise ValueError("moe_dispatch_chunk_size must be >= 1")
        if not isinstance(self.fixed_draws, bool):
            raise ValueError(f"fixed_draws must be true or false: {self.fixed_draws!r}")


@dataclass
class EncodedWindow:
    """The encoded inputs of one training window (K chunks, A action dims).

    ``text`` is ``(1, S, 2048)`` in bf16, before the text dropout. ``anchor``
    is the 1-frame latent ``(1, C, 1, h, w)``. ``chunks`` holds the K clean
    chunk latents ``(1, C, L, h, w)``. ``actions`` holds the normalized action
    tokens ``(1, K * L, 4 * A)`` float32, chunk-major. ``proprio`` holds the
    normalized boundary poses ``(1, K, A)`` float32. ``window_id`` is the
    index of the window in a fixed set of windows (the overfit check), or
    None. With ``fixed_draws``, it sets the random values of the window.
    """

    text: Tensor
    anchor: Tensor
    chunks: list[Tensor]
    actions: Tensor
    proprio: Tensor
    window_id: int | None = None


@dataclass
class Draws:
    """The random values of one training window, in draw order.

    ``text_u``, ``video_u0`` and ``proprio_u`` are ``(1,)`` float32 uniform
    values in [0, 1). A dropout value below the dropout probability drops the
    input. ``video_noise`` is the window noise ``eps0`` ``(1, C, K * L, h, w)``
    in bf16. ``video_u`` and ``action_u`` are ``(K,)`` uniform values.
    ``action_noise`` is ``(1, K * L, 4 * A)`` float32. ``rope_shifts`` holds
    the offsets ``(encoder anchor, encoder text, encoder video, decoder
    video)``.
    """

    text_u: Tensor
    video_u0: Tensor
    video_noise: Tensor
    video_u: Tensor
    rope_shifts: tuple[int, int, int, int]
    action_u: Tensor
    action_noise: Tensor
    proprio_u: Tensor


#: The backward template parameters of the compiled flex attention of the
#: action expert in training. Pinned, so the backward
#: kernel compiles without autotuning.
EXPERT_FLEX_BACKWARD_OPTIONS: dict[str, int] = {
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_M2": 64,
    "bwd_BLOCK_N1": 64,
    "bwd_BLOCK_N2": 32,
}


def shifted_timesteps(u: Tensor, shift: float) -> Tensor:
    """Return ``shift * u / (1 + (shift - 1) * u)``: the shifted timesteps."""
    return shift * u / (1.0 + (shift - 1.0) * u)


def _fixed_compile_cache_order() -> None:
    """Make each compiled function use its first compiled variant that fits.

    By default ``torch.compile`` moves the variant of the last call to the
    front of its cache. The flex attention of the backbone gets a static
    variant for the first sequence length and a dynamic variant later. The
    two variants give different gradient bits. This function turns the
    reordering off.
    """
    set_lru = getattr(torch._C._dynamo.eval_frame, "_set_lru_cache", None)
    if set_lru is not None:
        set_lru(False)


class FinetuneModel(nn.Module):
    """A praxis policy with the training objective of the chunk-causal layout.

    ``forward(batch, generator)`` draws the random values of the window from
    the CPU generator (``draw``). Then it returns the loss and the metrics
    (``loss``). All draws happen before the first block runs, so a recompute
    of a block (activation checkpointing) gives the same values.

    ``text_uncond`` is the encoded blank caption ``(1, S0, 2048)`` (the
    ``text_encoder.blank_caption`` of the checkpoint config). The text
    dropout needs it; set it with ``set_text_uncond``. The policy has the
    ``chunk_causal`` backbone and the trunk-rope expert (the only layout).

    The constructor installs the routing hooks of the MoE layers
    (``praxis.train.moe.track_routing``). It also fixes the cache order of
    the compiled functions of the process (``_fixed_compile_cache_order``).
    """

    def __init__(
        self,
        policy: PraxisModel,
        settings: ObjectiveSettings,
        text_uncond: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.settings = settings
        self.text_uncond: Tensor | None = None
        if text_uncond is not None:
            self.set_text_uncond(text_uncond)
        self.moe_layers = track_routing(
            policy,
            seq_balance=settings.moe_seq_balance_coef > 0.0,
            unbiased_picks=settings.moe_balance_speed > 0.0,
        )
        for layer in self.moe_layers:
            layer.dispatch_chunk_size = settings.moe_dispatch_chunk_size
        expert = policy.action_expert
        if expert.flex_kernel_options is not None:
            # The pinned backward block sizes. The forward keys of
            # the checkpoint config stay as they are.
            expert.flex_kernel_options = {
                **expert.flex_kernel_options,
                **EXPERT_FLEX_BACKWARD_OPTIONS,
            }
        _fixed_compile_cache_order()

    def set_text_uncond(self, text_uncond: Tensor) -> None:
        """Set the encoded blank caption ``(1, S0, D)`` of the text dropout."""
        if text_uncond.dim() != 3 or text_uncond.shape[0] != 1:
            raise ValueError(f"text_uncond must be (1, S, D), got {text_uncond.shape}")
        self.text_uncond = text_uncond.detach().to("cpu", torch.bfloat16)

    def blocks(self) -> list[nn.Module]:
        """Return the transformer blocks of both towers and of the action expert."""
        backbone = self.policy.backbone
        return [
            *backbone.encoder.core.blocks,
            *backbone.decoder.core.blocks,
            *self.policy.action_expert.blocks,
        ]

    def _text_dropped(self, text_u: Tensor) -> bool:
        return self.training and bool(text_u < self.settings.text_dropout)

    def draw(self, batch: EncodedWindow, generator: torch.Generator) -> Draws:
        """Draw the random values of one window from ``generator`` (CPU).

        The order is the order of the ``Draws`` fields. The text dropout value
        comes first: the span of the text offset depends on the text that the
        window uses.
        """
        text_u = torch.rand(1, generator=generator)
        video_u0 = torch.rand(1, generator=generator)
        x_shape = list(batch.chunks[0].shape)
        x_shape[2] = sum(int(c.shape[2]) for c in batch.chunks)
        video_noise = torch.randn(tuple(x_shape), generator=generator)
        kk = len(batch.chunks)
        video_u = torch.rand(kk, generator=generator)
        n_text = int(batch.text.shape[1])
        if self._text_dropped(text_u):
            n_text = int(self._uncond().shape[1])
        h, w = int(batch.anchor.shape[-2]), int(batch.anchor.shape[-1])
        ph, pw = self.policy.backbone.decoder.patch_size
        spans = (
            w + h * ROW_STRIDE,
            n_text,
            w + h * ROW_STRIDE,
            w * pw + h * ph * ROW_STRIDE,
        )
        shifts = [int(torch.randint(0, FRAME_STRIDE - span, (1,), generator=generator)) for span in spans]
        action_u = torch.rand(kk, generator=generator)
        action_noise = torch.randn(tuple(batch.actions.shape), generator=generator)
        proprio_u = torch.rand(1, generator=generator)
        return Draws(
            text_u=text_u,
            video_u0=video_u0,
            video_noise=video_noise.to(torch.bfloat16),
            video_u=video_u,
            rope_shifts=(shifts[0], shifts[1], shifts[2], shifts[3]),
            action_u=action_u,
            action_noise=action_noise,
            proprio_u=proprio_u,
        )

    def _uncond(self) -> Tensor:
        if self.text_uncond is None:
            raise ValueError("the text dropout needs the encoded blank caption (set_text_uncond)")
        return self.text_uncond

    def forward(self, batch: EncodedWindow, generator: torch.Generator) -> tuple[Tensor, dict[str, Tensor]]:
        return self.loss(batch, self.draw(batch, generator))

    def video_inputs(
        self, batch: EncodedWindow, draws: Draws
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        """Return the noised window, the clean copy, the timesteps and the target.

        The two-stage noising of the module docstring, on the device of the
        policy: ``(noised (1, C, K * L, h, w), clean (1, C, (K - 1) * L, h, w)
        or None, timesteps (1, K), target (1, C, K * L, h, w))``, all in the
        model dtype.
        """
        policy, s = self.policy, self.settings
        dev, md = policy.device, policy.dtype
        kk = len(batch.chunks)
        lat = int(batch.chunks[0].shape[2])
        x0 = torch.cat([c.to(dev, md) for c in batch.chunks], dim=2)
        eps0 = draws.video_noise.to(dev, md)
        t0 = shifted_timesteps(draws.video_u0.to(dev), s.video_timestep_shift)
        t0v = t0[0].view(1, 1, 1, 1, 1)
        xt = (x0 * (1 - t0v) + eps0 * t0v).to(md)
        target = (eps0 - x0).detach()
        t0_md = t0.to(md)
        # Recover x0 and eps from the bf16 x_t, t0 and target in float32.
        z32, v32 = xt.float(), target.float()
        x0r = z32 - t0_md[0].float() * v32
        epsr = x0r + v32
        t_k = shifted_timesteps(draws.video_u.to(dev), s.video_timestep_shift).to(md)
        noised = torch.empty_like(x0r)
        for k in range(kk):
            tk = t_k[k].float()
            sl = slice(k * lat, (k + 1) * lat)
            noised[:, :, sl] = x0r[:, :, sl] * (1 - tk) + epsr[:, :, sl] * tk
        clean = x0r.to(md).detach()[:, :, : (kk - 1) * lat] if kk > 1 else None
        return noised.to(md), clean, t_k.view(1, kk), target

    def action_inputs(self, batch: EncodedWindow, draws: Draws) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the noised actions, their timesteps, the target and the proprio.

        ``(noised (1, K * L, 4A), t (1, K), target (1, K * L, 4A), proprio
        (1, K, A))``, float32 on the CPU, with the proprio dropout.
        """
        kk = len(batch.chunks)
        x0 = batch.actions.detach().to("cpu", torch.float32)
        t = draws.action_u.to(torch.float32)
        eps = draws.action_noise.to(torch.float32)
        t_tok = t.repeat_interleave(x0.shape[1] // kk)[None, :, None]
        noised = (1 - t_tok) * x0 + t_tok * eps
        target = eps - x0
        proprio = batch.proprio.detach().to("cpu", torch.float32)
        if self.training and bool(draws.proprio_u < self.settings.proprio_dropout):
            proprio = torch.zeros_like(proprio)
        return noised, t.view(1, kk), target, proprio

    def loss(self, batch: EncodedWindow, draws: Draws) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the loss of one window with the given random values.

        The metrics are the unweighted terms ``loss_video``, ``loss_action``
        (before the guard) and ``loss_moe_balance`` (train mode), and
        ``action_dropped`` (1 when the guard dropped the action loss), all
        float32 and detached. In eval mode there is no dropout and no RoPE
        offset.
        """
        policy, s = self.policy, self.settings
        dev, md = policy.device, policy.dtype
        kk = len(batch.chunks)
        noised_v, clean, t_k, target_v = self.video_inputs(batch, draws)
        noised_a, t_a, target_a, proprio = self.action_inputs(batch, draws)
        text = batch.text
        if self._text_dropped(draws.text_u):
            text = self._uncond()
        text = text.to(dev, md)
        shifts = draws.rope_shifts if self.training else (0, 0, 0, 0)

        out = policy.backbone.forward_chunk_causal(
            text,
            batch.anchor.to(dev, md),
            clean,
            noised_v,
            t_k,
            rope_shifts=shifts,
        )
        taps = out.taps
        assert out.velocity is not None
        assert taps.global_spans is not None and taps.noised_span is not None
        assert taps.rope_ids is not None
        v_a = run_expert_on_taps_k(
            policy.action_expert,
            taps.taps,
            taps.embeddings,
            taps.rope_ids,
            global_spans=taps.global_spans,
            noised_span=taps.noised_span,
            clean_span=taps.clean_span,
            n_chunks=kk,
            frames_per_chunk=taps.frames_per_chunk,
            tokens_per_frame=taps.tokens_per_frame,
            noised_actions=noised_a.to(dev),
            action_t_per_chunk=t_a.to(dev),
            proprio_per_chunk=proprio.to(dev),
        )

        err_v = (out.velocity - target_v).pow(2)
        loss_v = (0 + err_v.sum()) / err_v.numel()
        p = v_a[0].float()
        tgt = target_a[0].to(dev, torch.float32)
        loss_a = (p - tgt).square().mean(-1).mean()
        with torch.no_grad():
            keep = bool(torch.isfinite(loss_a)) and float(loss_a) <= s.action_guard
        metrics = {
            "loss_video": loss_v.detach().float(),
            "loss_action": loss_a.detach().float(),
            "action_dropped": torch.tensor(0.0 if keep else 1.0, device=dev),
        }
        if not keep:
            # An exact zero. The weights of the expert stay in the graph, so
            # that every rank reduces the same gradients.
            loss_a = (torch.nan_to_num(p, 0.0, 0.0, 0.0) * 0.0).sum()
        total = (s.video_weight * loss_v).float() + (s.action_weight * loss_a).float()
        balance = take_sequence_balance_losses(self.moe_layers)
        if balance:
            loss_b = torch.stack(balance).mean()
            total = total + s.moe_seq_balance_coef * loss_b.to(total.dtype)
            metrics["loss_moe_balance"] = loss_b.detach().float()
        return total, metrics


@dataclass
class WindowInputs:
    """Encoded inputs of one training window (see ``window_inputs``)."""

    text: Tensor  # (1, S, 2048)
    anchor: Tensor  # (1, 192, 1, h, w): 1-frame encode of frame 0
    chunks: list[Tensor]  # K x (1, 192, 6, h, w): slices of the window encode
    proprio: Tensor  # (1, K, A): normalized boundary poses
    gt_actions: list[Tensor]  # K x (6, 4 A): normalized relative action tokens


def chunk_geometry(n_chunks: int, action_chunk: int = 24) -> list[tuple[int, int]]:
    """``(offset, length)`` of each chunk in the window: chunk 0 has
    ``action_chunk - 3`` frames (it owns the anchor latent), later chunks
    ``action_chunk`` frames."""
    geom = [(0, action_chunk - 3)]
    for _ in range(1, n_chunks):
        off = geom[-1][0] + geom[-1][1]
        geom.append((off, action_chunk))
    return geom


@torch.no_grad()
def window_inputs(
    vae: Autoencoder,
    text: Tensor,
    frames: np.ndarray,
    action_rel: np.ndarray,
    proprio_abs_anchors: np.ndarray,
    stats: NormStats,
    n_chunks: int,
    action_chunk: int,
    q99_dims: np.ndarray,
) -> WindowInputs:
    """Build the encoded inputs of one window from recorded data.

    ``frames`` is the canvas ``(T, H, W, 3)`` of the window (93 frames), uint8
    or float32 in [0, 255].
    ``action_rel`` holds the ``(T, A)`` relative action targets and
    ``proprio_abs_anchors`` the ``(K, A)`` absolute boundary poses.
    ``stats`` holds the normalization of the checkpoint. ``q99_dims``
    are the action dims that use the q99 normalization (the
    ``q99_normed_dims`` of the checkpoint config).
    """
    geom = chunk_geometry(n_chunks, action_chunk)
    clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2)
    clip = clip.unsqueeze(0).float().to(vae.latent_mean.device)
    window = vae.encode(clip)
    lat = (action_chunk - 1) // 4 + 1
    chunks = [window[:, :, k * lat : (k + 1) * lat].contiguous() for k in range(n_chunks)]
    anchor = vae.encode(clip[:, :, :1])
    norm = q99_normalize(action_rel, stats.action_q01, stats.action_q99, dims=q99_dims)
    gt_actions = [
        torch.from_numpy(pack_action_tokens(norm[off : off + ln], first_chunk=(k == 0)))
        for k, (off, ln) in enumerate(geom)
    ]
    proprio = proprio_normalize(proprio_abs_anchors, stats.proprio_mean, stats.proprio_std)
    return WindowInputs(
        text=text,
        anchor=anchor,
        chunks=chunks,
        proprio=torch.from_numpy(np.ascontiguousarray(proprio))[None].float(),
        gt_actions=gt_actions,
    )


class TextEncode(Protocol):
    """The text encoder interface that ``encode_window`` uses."""

    def encode(self, prompts: Sequence[str]) -> list[Tensor]: ...


@torch.no_grad()
def encode_window(
    window: TrainingWindow,
    vae: Autoencoder,
    text_encoder: TextEncode,
    config: dict,
    device: torch.device | str,
    window_id: int | None = None,
) -> EncodedWindow:
    """Encode a ``praxis.train.data.TrainingWindow`` for ``FinetuneModel``.

    The steps are those of ``window_inputs``:

    * Video: the autoencoder encodes the whole window (``T`` canvas frames,
      uint8 or float in [0, 255]) in one pass. The latents split into the
      ``n_chunks`` chunks of 6 latents. The anchor is a separate 1-frame
      encode of frame 0.
    * Actions: the ``(T, A)`` relative rows get the q99 normalization on the
      ``q99_normed_dims`` of the config (clipped to [-1, 1]); the other dims
      stay as they are. The rows of each chunk pack into tokens of 4 rows.
      Chunk 0 repeats row 0 in its first token.
    * Proprio: the ``(K, A)`` absolute boundary poses get the mean and std
      normalization on all dims.
    * Text: ``text_encoder.encode([prompt])``. The text dropout belongs to the
      objective (``Draws.text_u``), not to this function.

    ``text_encoder`` needs only the method ``encode(prompts)``. The outputs
    go to ``device``: the latents and the text in bf16, the actions and the
    proprio in float32. ``window_id`` goes into the result unchanged.
    """
    norm, chunking = config["normalization"], config["chunking"]
    n_chunks, action_chunk = int(chunking["n_chunks"]), int(chunking["action_chunk"])
    dim = int(config["action_space"]["dim"])
    frames_t = sum(contiguous_chunk_geometry(action_chunk, n_chunks)[-1])
    if window.frames.shape[0] != frames_t:
        raise ValueError(f"window has {window.frames.shape[0]} frames, not {frames_t}")
    if window.actions_rel.shape != (frames_t, dim):
        raise ValueError(f"actions_rel {window.actions_rel.shape} != {(frames_t, dim)}")
    if window.proprio_abs.shape != (n_chunks, dim):
        raise ValueError(f"proprio_abs {window.proprio_abs.shape} != {(n_chunks, dim)}")
    stats = NormStats(
        action_q01=np.asarray(norm["action_q01"], dtype=np.float32),
        action_q99=np.asarray(norm["action_q99"], dtype=np.float32),
        proprio_mean=np.asarray(norm["proprio_mean"], dtype=np.float32),
        proprio_std=np.asarray(norm["proprio_std"], dtype=np.float32),
    )
    text = text_encoder.encode([window.prompt])[0]
    frames = window.frames.permute(0, 2, 3, 1).numpy()
    inputs = window_inputs(
        vae,
        text,
        frames,
        window.actions_rel,
        window.proprio_abs,
        stats,
        n_chunks=n_chunks,
        action_chunk=action_chunk,
        q99_dims=np.asarray(norm["q99_normed_dims"], dtype=np.int64),
    )
    dev = torch.device(device)
    return EncodedWindow(
        text=inputs.text.to(dev, torch.bfloat16),
        anchor=inputs.anchor.to(dev, torch.bfloat16),
        chunks=[c.to(dev, torch.bfloat16) for c in inputs.chunks],
        actions=torch.cat(inputs.gt_actions, dim=0)[None].to(dev, torch.float32),
        proprio=inputs.proprio.to(dev, torch.float32),
        window_id=window_id,
    )
