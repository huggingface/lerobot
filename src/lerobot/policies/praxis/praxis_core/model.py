# Vendored from runwayml/praxis-1-oss at commit 8dad1a4 plus the canvas rule of the released DROID checkpoint,
# original path praxis/model.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""The praxis policy: video backbone, action expert, autoencoder, text encoder.

One policy call denoises one chunk of the future: 6 video latents and 6 action
tokens (24 action rows at 15 Hz). The inputs are the task text, the window
anchor latent, up to three clean chunk latents of the executed past, and the
current arm pose. The backbone denoises the video. Each backbone step also
gives the hidden states ("taps") that the action expert reads to denoise the
actions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from lerobot.policies.praxis.praxis_core.action_expert import ActionExpert, run_expert_on_taps_k
from lerobot.policies.praxis.praxis_core.backbone import VideoBackbone, build_backbone
from lerobot.policies.praxis.praxis_core.checkpoint import load_checkpoint, load_config, model_variant
from lerobot.policies.praxis.praxis_core.sampler import (
    ACTION_SALT,
    VIDEO_SALT,
    check_guidance,
    guidance_combine,
    seeded_normal,
    window_time_grid,
)

LAYOUTS = ("window_joint",)

TraceFn = Callable[[str, Tensor], None]
"""Optional probe ``(name, tensor)`` of the window joint sampler."""


class ChunkCausalTaps(Protocol):
    """The encoder features of one chunk-causal backbone call.

    ``global_spans`` maps ``register``, ``anchor`` and ``text`` to their
    ``(start, length)`` token range. ``noised_span`` and ``clean_span`` are
    the ranges of the noised window and of the clean copies (None for a
    window of one chunk). ``rope_ids`` are the ``(1, S)`` rotary positions of
    the encoder sequence.
    """

    taps: list[Tensor]
    embeddings: Tensor
    rope_ids: Tensor
    global_spans: Mapping[str, tuple[int, int]]
    noised_span: tuple[int, int]
    clean_span: tuple[int, int] | None
    frames_per_chunk: int
    tokens_per_frame: int


class ChunkCausalOutput(Protocol):
    velocity: Tensor | None
    taps: ChunkCausalTaps


class ChunkCausalBackbone(Protocol):
    """A backbone with the chunk-causal sequence of the checkpoint."""

    def forward_chunk_causal(
        self,
        text: Tensor,
        anchor: Tensor,
        clean: Tensor | None,
        noised: Tensor,
        timesteps: Tensor,
        decode: bool = True,
    ) -> ChunkCausalOutput: ...


@dataclass
class SamplerSettings:
    """Sampler settings of one policy call.

    The defaults are the recommended settings of the checkpoint.

    ``layout`` is ``window_joint``, the sampler of the checkpoint
    (chunk-causal backbone). The window holds the ``d + 1`` chunks up to the
    served one, all noised at the same time and integrated jointly, video and
    actions. It needs the ``euler`` solver and no expert-only sub-steps. See
    :meth:`PraxisModel.sample_window_joint`.

    ``cfg_scale`` and ``cfg_video_scale`` are the text guidance scales of the
    action velocity and of the video velocity. 1.0 turns guidance off. The
    guided and the unguided pass run one after the other, and the unguided
    pass uses the encoded blank caption of the checkpoint config with the
    same video and proprio inputs. ``cfg_batched`` has no effect.
    """

    num_steps: int = 2
    t_shift: float = 12.5
    solver: str = "euler"
    action_steps: int | None = None
    layout: str = "window_joint"
    cfg_scale: float = 1.0
    cfg_video_scale: float = 4.0
    cfg_batched: bool = False

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}, got {self.layout!r}")
        check_guidance(self.cfg_scale, self.cfg_video_scale)
        if self.solver != "euler":
            raise ValueError("layout='window_joint' needs solver='euler'")
        if self.action_steps not in (None, self.num_steps):
            raise ValueError("layout='window_joint' has no expert-only sub-steps")

    @property
    def guided(self) -> bool:
        """True if text guidance is on (a scale is not 1.0)."""
        return check_guidance(self.cfg_scale, self.cfg_video_scale)


class PraxisModel(nn.Module):
    """Video backbone plus action expert, loaded from a praxis checkpoint."""

    def __init__(self, backbone: VideoBackbone, expert: ActionExpert, config: dict) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_expert = expert
        self.config = config

    @classmethod
    def from_config(cls, config: dict) -> PraxisModel:
        """Build the model from a checkpoint config, with uninitialized weights.

        The modules go to the current default device. Use
        ``with torch.device("meta")`` to build the model without memory.
        """
        ex = cast(dict, config["action_expert"])
        backbone = build_backbone(cast(dict, config["backbone"]))
        # The variant fields of the trunk-rope expert (rope, global order,
        # attention, head_plus_zero, ...).
        variant_kwargs = model_variant(config).expert_kwargs()
        expert = ActionExpert(
            action_patch_dim=ex["action_patch_dim"],
            proprio_patch_dim=ex["proprio_patch_dim"],
            trunk_dim=ex["trunk_dim"],
            dim=ex["dim"],
            num_layers=ex["num_layers"],
            num_heads=ex["num_heads"],
            head_dim=ex["head_dim"],
            num_future_per_chunk=ex["num_future_per_chunk"],
            max_chunks=ex["max_chunks"],
            max_action_tokens=ex["max_action_tokens"],
            tap_norm=ex["tap_norm"],
            clean_only_conditioning=ex["clean_only_conditioning"],
            **variant_kwargs,
        )
        return cls(backbone, expert, config)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str | torch.device = "cuda") -> PraxisModel:
        """Build the model from ``config.json`` and load the weights strictly."""
        with torch.device("meta"):
            model = cls.from_config(load_config(path))
        load_checkpoint(model, path, device=str(device), assign=True)
        return model.eval().requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone.encoder.registers.dtype

    @property
    def device(self) -> torch.device:
        return self.backbone.encoder.registers.device

    @torch.no_grad()
    def sample_window_joint(
        self,
        text: Tensor,
        text_uncond: Tensor | None,
        anchor: Tensor,
        cleans: list[Tensor],
        proprio_all: Tensor,
        seed: int,
        settings: SamplerSettings,
        trace: TraceFn | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Denoise the window of the chunk at depth ``d = len(cleans)`` jointly.

        The sampler of the checkpoint (layout ``window_joint``). The
        window holds ``K' = d + 1`` chunks: the clean copies of the ``d``
        context chunks and ``K'`` noised chunks, all at the same time ``t``.
        Each step makes one backbone call on the whole window and one expert
        call on its taps, and moves the video and the actions of every chunk
        with one Euler step. Only the last chunk is returned. The noised
        history chunks start from their own noise and are dropped.

        * ``text`` is ``(1, S, 2048)``. ``text_uncond`` is the encoded blank
          caption of the unguided pass. Guidance needs it. The unguided pass
          keeps every other input.
        * ``anchor`` is ``(1, 192, 1, h, w)``, each clean chunk
          ``(1, 192, 6, h, w)``. ``proprio_all`` holds the normalized boundary
          pose of each chunk ``(1, K', P)`` float32, current pose last.
        * The noise is float32 from :func:`praxis.sampler.seeded_normal` on the
          CPU: the video ``(1, 192, 6 K', h, w)`` with :data:`VIDEO_SALT` and
          the actions ``(6 K', A)`` with :data:`ACTION_SALT`.
        * The grid is :func:`praxis.sampler.window_time_grid` (float64). The
          states stay float32 and enter the models in the model dtype. The
          step is ``x + (t_next - t_cur) * v`` with a Python float step.
        * ``trace`` receives the intermediate tensors by name.

        Returns the video latent of the last chunk ``(1, 192, 6, h, w)`` and
        its normalized action tokens ``(1, 6, A)``, float32.
        """
        if settings.layout != "window_joint":
            raise ValueError(f"layout {settings.layout!r} is not 'window_joint'")
        guided = settings.guided
        if guided and text_uncond is None:
            raise ValueError("text guidance needs the encoded blank caption")
        dev, md = self.device, self.dtype
        lat = int(self.config["chunking"]["latents_per_chunk"])
        max_chunks = int(self.config["chunking"]["n_chunks"])
        d = len(cleans)
        kk = d + 1
        if kk > max_chunks:
            raise ValueError(f"depth {d} is out of range for {max_chunks} chunks")
        pro = proprio_all if proprio_all.ndim == 3 else proprio_all[None]
        if pro.shape[:2] != (1, kk):
            raise ValueError(f"proprio_all {tuple(pro.shape)} for {kk} chunks")
        a_dim = int(self.config["action_expert"]["action_patch_dim"])
        c, h, w = anchor.shape[1], anchor.shape[3], anchor.shape[4]

        def probe(name: str, t: Tensor) -> None:
            if trace is not None:
                trace(name, t)

        noise_v = seeded_normal((1, c, kk * lat, h, w), seed, VIDEO_SALT)
        probe("noise.salt0", noise_v)
        noise_a = seeded_normal((kk * lat, a_dim), seed, ACTION_SALT)
        probe("noise.salt1", noise_a)
        ts = window_time_grid(settings.num_steps, settings.t_shift)
        probe("sampler.time_grid", torch.tensor(ts, dtype=torch.float64))
        xv = noise_v.to(dev)
        xa = noise_a.to(dev)
        anchor_m = anchor.to(dev, md)
        clean = torch.cat([z.to(dev, md) for z in cleans], dim=2) if d else None
        pro = pro.to(dev, torch.float32)
        text_c = text.to(dev, md)
        text_u = None if text_uncond is None else text_uncond.to(dev, md)
        backbone = cast(ChunkCausalBackbone, self.backbone)

        def velocity(
            text_in: Tensor, x_v: Tensor, x_a: Tensor, t_cur: float, decode: bool
        ) -> tuple[Tensor | None, Tensor]:
            probe("velocity.in.xt_v", x_v)
            probe("velocity.in.xt_a", x_a)
            probe("velocity.in.t_cur", torch.tensor(t_cur, dtype=torch.float64))
            t_trunk = torch.full((1, kk), t_cur, dtype=md, device=dev)
            out = backbone.forward_chunk_causal(text_in, anchor_m, clean, x_v.to(md), t_trunk, decode=decode)
            taps = out.taps
            a = run_expert_on_taps_k(
                self.action_expert,
                taps.taps,
                taps.embeddings,
                taps.rope_ids,
                global_spans=taps.global_spans,
                noised_span=taps.noised_span,
                clean_span=taps.clean_span,
                n_chunks=kk,
                frames_per_chunk=taps.frames_per_chunk,
                tokens_per_frame=taps.tokens_per_frame,
                noised_actions=x_a[None],
                action_t_per_chunk=torch.full((1, kk), t_cur, device=dev),
                proprio_per_chunk=pro,
            ).float()[0]
            v = None if out.velocity is None else out.velocity.float()
            if v is not None:
                probe("velocity.out.video", v)
            probe("velocity.out.action", a)
            return v, a

        decode_uncond = trace is not None or float(settings.cfg_video_scale) != 1.0
        for step in range(settings.num_steps):
            t_cur, t_next = ts[step], ts[step + 1]
            probe("pass.marker", torch.tensor([step, 0]))
            v, a = velocity(text_c, xv, xa, t_cur, True)
            assert v is not None
            if guided:
                assert text_u is not None
                probe("pass.marker", torch.tensor([step, 1]))
                v_u, a_u = velocity(text_u, xv, xa, t_cur, decode_uncond)
                for kind, cond, unc, scale in (
                    ("video", v, v_u, settings.cfg_video_scale),
                    ("action", a, a_u, settings.cfg_scale),
                ):
                    if unc is None:  # the unguided video is not needed
                        continue
                    out = guidance_combine(cond, unc, scale)
                    probe(f"cfg_combine.{kind}.cond", cond)
                    probe(f"cfg_combine.{kind}.uncond", unc)
                    probe(f"cfg_combine.{kind}.out", out)
                    if kind == "video":
                        v = out
                    else:
                        a = out
            dt = t_next - t_cur
            xv = xv + dt * v
            xa = xa + dt * a
        probe("out.video", xv)
        probe("out.action", xa)
        video = xv[:, :, d * lat :].contiguous()
        actions = xa[d * lat :].contiguous()
        probe("out.video_last", video)
        probe("out.action_last", actions)
        return video, actions[None]
