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
"""FLUX 3 Action (``flux3``) policy: joint video + action denoising on the FLUX 3 trunk.

Frame conditioning uses the current camera canvas and state. History conditioning uses independently
encoded snapshots and a sequence of states, optionally concatenated with past actions. Both condition a
DiT on the task caption and jointly denoise future video and action tokens. Video is never decoded here.

Training (``forward``) encodes the observed and target video, packs action targets, samples noise levels,
and computes the configured flow-matching loss. The conditioning layout, action representation and loss
pooling are explicit config choices rather than robot-name modes.

Inference (``predict_action_chunk``) starts the unknown future streams from noise and denoises them jointly.
``select_action`` queues the configured execution horizon. Saved processors own normalization, observation
history and reconstruction of absolute commands when consecutive command deltas are used.

"""

from __future__ import annotations

import copy
import logging
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor, nn

from lerobot.policies.flux3.configuration_flux3 import Flux3Config
from lerobot.policies.flux3.f3 import (
    REQUIRED_CONTENT_STREAMS,
    VEC_DIM,
    JointSingleSeq,
    JointSingleSeqParams,
    action_dit_params,
    batched_prc_audio,
    batched_prc_vid,
    build_action_dit,
    fresh_head_state_dict,
    fresh_module_names,
    load_text_encoder,
    load_video_vae,
    packing,
    restrict_content_streams,
    sampling,
    stream_of_key,
    text_context,
    times_to_ids,
    unused_content_key,
)
from lerobot.policies.flux3.processor_flux3 import PAST_ACTIONS
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .utils import resolve_weights

logger = logging.getLogger(__name__)

TRUNK_WEIGHTS_FILENAME = "dit.safetensors"


@dataclass
class FrozenComponents:
    """Video VAE and text encoder: eval-only, never saved, never trained.

    Deliberately not an ``nn.Module``: assigning an instance to the policy registers nothing, so both
    components stay out of ``state_dict()`` / ``parameters()`` (never written to ``model.safetensors``,
    never handed to the optimizer) and ``policy.train()`` cannot undo the ``eval()`` applied here. The
    policy's ``_apply`` forwards device / dtype moves so they follow the trunk.
    """

    video_vae: Any
    text_encoder: nn.Module

    def __post_init__(self) -> None:
        for module in self.modules():
            module.eval()
            module.requires_grad_(False)

    def modules(self) -> Iterator[nn.Module]:
        """The torch modules held here (the VAE may be a thin wrapper around its ``module``)."""
        for m in (getattr(self.video_vae, "module", self.video_vae), self.text_encoder):
            if isinstance(m, nn.Module):
                yield m

    def _apply(self, fn) -> None:
        for module in self.modules():
            module._apply(fn)


class Flux3Policy(PreTrainedPolicy):
    """FLUX 3 Action policy (see the module docstring)."""

    config_class = Flux3Config
    name = "flux3"

    def __init__(
        self,
        config: Flux3Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *,
        _load_trunk_weights: bool = True,
        **kwargs: Any,
    ):
        super().__init__(config, dataset_stats)
        config.validate_features()
        # Configs can be parsed before export statistics are computed. Saved policies
        # restore their quantiles from processor state, not from the model config.
        if (
            _load_trunk_weights
            and not config.pretrained_path
            and config.conditioning == "history"
            and config.normalization_stats is None
        ):
            raise ValueError(
                "FLUX3 history conditioning requires a saved base (--policy.path) or explicit "
                "action/state quantiles in normalization_stats before building the model. "
                "Use examples/flux3/export_base.py to prepare a base; raw dataset stats cannot "
                "initialize command-delta normalization."
            )
        self.config = config
        self.dataset_stats = dataset_stats
        self.modality = config.action_modality
        self.dtype_ = getattr(torch, config.dtype)
        self.packer = packing.build_packer(config)
        self.dit_params = self._dit_params(config)
        # A full policy (including a deferred distributed restore) already owns its trunk.
        init_config = config
        if not _load_trunk_weights or config.pretrained_path:
            init_config = replace(config, trunk_weights=None)
        self.dit: JointSingleSeq = self._build_dit(init_config)
        self.dit.gradient_checkpointing = config.gradient_checkpointing
        self.frozen = FrozenComponents(self._build_video_vae(config), self._build_text_encoder(config))
        self._ctx_cache: dict[str, tuple[Tensor, Tensor]] = {}
        self._compiled_dit: Callable[..., dict[str, Tensor]] | None = None
        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        """Restore a complete policy without reopening its original initialization trunk."""
        kwargs.setdefault("strict", True)
        policy = super().from_pretrained(pretrained_name_or_path, _load_trunk_weights=False, **kwargs)
        # PEFT must reference the complete policy that supplies its frozen base weights.
        if policy.config.use_peft:
            # Preserve the adapter's processor path; fresh PEFT wrapping must still update the caller's config.
            policy.config = copy.copy(policy.config)
        policy.config.pretrained_path = str(pretrained_name_or_path)
        return policy

    # ------------------------------------------------------------------ construction hooks
    # Construct on CPU; the factory/from_pretrained device move places the DiT and frozen encoders
    # together via _apply. These loaders are never called from forward.
    def _dit_params(self, config: Flux3Config) -> JointSingleSeqParams:
        base = JointSingleSeqParams(**(config.dit_config or {}))
        base = restrict_content_streams(base, REQUIRED_CONTENT_STREAMS)
        return action_dit_params(
            base,
            config.action_modality,
            config.action_dim,
            attn_mode=config.attn_mode,
            conditioning_channels=self.packer.conditioning_channels(config),
        )

    def _build_dit(self, config: Flux3Config) -> JointSingleSeq:
        """The trainable core. ``trunk_weights`` (path or Hub id) starts a finetune from the action-pretrained trunk."""
        ckpt = resolve_weights(config.trunk_weights, TRUNK_WEIGHTS_FILENAME) if config.trunk_weights else None
        return build_action_dit(
            self.dit_params,
            ckpt,
            modality=self.modality,
            device="cpu",
            dtype=self.dtype_,
            strict_heads=False,
            head_seed=config.head_init_seed,
        )

    def _build_video_vae(self, config: Flux3Config):
        """Frozen Video VAE (needs NATTEN). Tests replace this hook with a shape-compatible fake."""
        return load_video_vae(config.video_vae_id, device="cpu", compile_model=config.compile_model)

    def _build_text_encoder(self, config: Flux3Config) -> nn.Module:
        """Load the frozen Qwen3-VL text encoder."""
        return load_text_encoder(
            config.text_encoder_id,
            device="cpu",
            compile_model=config.compile_model,
        )

    def _apply(self, fn, *args, **kwargs):
        """``.to()`` / ``.cuda()`` / accelerate moves reach the unregistered frozen components here too."""
        super()._apply(fn, *args, **kwargs)
        self.frozen._apply(fn)
        self._ctx_cache.clear()
        self._compiled_dit = None  # a moved model is compiled afresh on its next prediction
        return self

    def _inference_dit(self) -> Callable[..., dict[str, Tensor]]:
        """The DiT forward used by sampling: ``torch.compile``d once, lazily, when ``compile_model`` is set.

        Only inference uses it. Training keeps the eager forward until compilation under caption-length
        batching and gradient checkpointing has been measured.
        """
        if not self.config.compile_model:
            return self.dit
        if self._compiled_dit is None:
            self._compiled_dit = torch.compile(self.dit.forward)
        return self._compiled_dit

    # ------------------------------------------------------------------ lerobot contract
    def get_optim_params(self) -> list[dict[str, Any]]:
        """Two groups: the trunk (and LoRA adapters), and the fresh embodiment heads at their own rate."""
        head_prefixes = tuple(f"{n}." for n in fresh_module_names(self.modality))
        trunk: list[nn.Parameter] = []
        heads: list[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (heads if any(prefix in name for prefix in head_prefixes) else trunk).append(p)
        groups: list[dict[str, Any]] = [{"params": trunk}]
        if heads:
            groups.append(
                {"params": heads, "lr": self.config.optimizer_lr * self.config.optimizer_lr_heads_multiplier}
            )
        return groups

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def _get_default_peft_targets(self) -> dict[str, Any]:
        """LoRA on every attention / MLP projection of the DiT; the fresh heads train in full."""
        target_modules = r".*\.(q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out)"
        return {
            "target_modules": target_modules,
            "modules_to_save": [f"dit.{n}" for n in fresh_module_names(self.modality)],
        }

    def _validate_peft_config(self, peft_config) -> None:
        if not self.config.pretrained_path:
            raise ValueError(
                "LoRA finetuning needs a saved pretrained flux3 policy as its reloadable base. "
                "For trunk_weights initialization, first save_pretrained a base policy, then reload it "
                "with from_pretrained (or --policy.path) before enabling PEFT."
            )

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        """Strict restore after stream filtering; ``strict=False`` permits fresh embodiment heads only."""
        expected = model.state_dict()
        head_prefixes = tuple(f"dit.{n}." for n in fresh_module_names(model.modality))
        built = set(model.dit.in_channels)
        state: dict[str, Tensor] = {}
        mismatched: list[str] = []
        foreign: list[str] = []
        # Stage checkpoints on CPU to avoid a second full copy in accelerator memory during restore.
        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 (safe_open handles are not dicts)
                if unused_content_key(key, built):
                    foreign.append(key)
                    continue
                tensor = f.get_tensor(key)
                if key in expected and tuple(expected[key].shape) != tuple(tensor.shape):
                    mismatch = (
                        f"{key}: checkpoint {tuple(tensor.shape)} vs model {tuple(expected[key].shape)}"
                    )
                    if strict or not key.startswith(head_prefixes):
                        raise RuntimeError("shape mismatch in " + mismatch)
                    mismatched.append(mismatch)
                    continue
                state[key] = tensor
        if mismatched:
            logger.warning(
                "flux3: %d tensors skipped for shape mismatch (fresh heads for this action dim): %s",
                len(mismatched),
                mismatched[:4],
            )
        # Weights of content streams the policy is built without (a full generative trunk, or an older full
        # policy file opened as a lean policy) carry nothing for action prediction: drop them.
        if foreign:
            logger.info(
                "flux3: ignoring %d tensors of content streams this policy is built without: %s",
                len(foreign),
                sorted(filter(None, {stream_of_key(k) for k in foreign})),
            )
        missing = [k for k in expected.keys() - state.keys() if (strict or not k.startswith(head_prefixes))]
        if missing:
            raise RuntimeError(f"flux3 checkpoint missing {len(missing)} keys, e.g. {sorted(missing)[:6]}")
        unexpected = state.keys() - expected.keys()
        if unexpected:
            raise RuntimeError(
                f"flux3 checkpoint has {len(unexpected)} unexpected keys, e.g. {sorted(unexpected)[:6]}"
            )
        fresh_missing = expected.keys() - state.keys()
        if fresh_missing:
            emb, emb_cond = (
                model.dit.emb_in[model.modality].weight,
                model.dit.emb_in[f"{model.modality}_cond"].weight,
            )
            fresh = fresh_head_state_dict(
                emb.shape[0], model.modality, emb.shape[1], model.config.head_init_seed, emb_cond.shape[1]
            )
            state.update({k: fresh[k.removeprefix("dit.")] for k in fresh_missing})
        model.load_state_dict(state, strict=True)
        return model.to(map_location)

    # ------------------------------------------------------------------ batch helpers
    def _flip(self, x: Tensor) -> Tensor:
        """``x -> 1 - x`` on the configured gripper dims (last axis). Self-inverse."""
        dims = list(self.config.gripper_flip_dims)
        if not dims:
            return x
        x = x.clone()
        x[..., dims] = 1.0 - x[..., dims]
        return x

    def _cameras(self, batch: dict[str, Any]) -> Tensor:
        """Stack the configured cameras: ``(B, n_cams, T, 3, H, W)`` (``T = 1`` for single frames)."""
        cams = []
        for key in self.config.camera_order:
            if key not in batch:
                have = sorted(k for k in batch if isinstance(k, str) and k.startswith("observation.image"))
                raise KeyError(f"batch lacks camera {key!r}; image keys present: {have}")
            img = batch[key]
            if img.ndim == 4:
                img = img[:, None]
            if img.ndim != 5:
                raise ValueError(f"{key}: expected (B, C, H, W) or (B, T, C, H, W), got {tuple(img.shape)}")
            cams.append(img)
        if len({tuple(c.shape[1:]) for c in cams}) != 1:
            raise ValueError(
                f"cameras must share (T, C, H, W), got {[tuple(c.shape) for c in cams]}; "
                "apply the saved preprocessor before calling the policy"
            )
        return torch.stack(cams, dim=1)

    def _conditioning_inputs(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor | None]:
        """Validate the processor contract and return state/past-action tensors as (B, H, D)."""
        cfg = self.config
        state = batch[OBS_STATE].float()
        if cfg.conditioning == "frame":
            if state.ndim == 3:
                state = state[:, 0]
            return self._flip(state)[:, None], None
        if state.shape[1:] != (cfg.n_obs_steps, cfg.action_dim):
            raise ValueError("History conditioning requires the complete state history")
        past = None
        if cfg.condition_on_past_actions:
            if PAST_ACTIONS not in batch:
                raise ValueError("Apply the saved history preprocessor before calling the policy")
            past = batch[PAST_ACTIONS].float()
            if past.shape != state.shape:
                raise ValueError("Past actions must have the same (B, H, D) shape as states")
        return state, past

    @staticmethod
    def _captions(batch: dict[str, Any], batch_size: int, training: bool = False) -> list[str]:
        """Task strings for the batch, one paraphrase each (``packing.select_caption``)."""
        task = batch.get("task")
        if task is None:
            return [""] * batch_size
        if isinstance(task, str):
            task = [task] * batch_size
        task = list(task)
        if len(task) == 1 and batch_size > 1:
            task = task * batch_size
        if len(task) != batch_size:
            raise ValueError(f"got {len(task)} task strings for a batch of {batch_size}")
        return [packing.select_caption("" if t is None else str(t), training) for t in task]

    @staticmethod
    def _valid_windows(batch: dict[str, Any], batch_size: int, device: torch.device) -> Tensor:
        """Windows that reach past the episode end carry ``*_is_pad`` flags; those samples are excluded."""
        keep = torch.ones(batch_size, dtype=torch.bool, device=device)
        for key, value in batch.items():
            if (
                isinstance(key, str)
                and key.endswith("_is_pad")
                and isinstance(value, Tensor)
                and value.dtype == torch.bool
                and value.ndim >= 1
                and value.shape[0] == batch_size
            ):
                keep &= ~value.reshape(batch_size, -1).any(-1).to(device)
        return keep

    def _context(self, caption: str, device: torch.device) -> tuple[Tensor, Tensor]:
        """Text context ``(1, L, ctx_dim)`` in the DiT dtype plus its position ids, cached per caption."""
        hit = self._ctx_cache.get(caption)
        if hit is None or hit[0].device != device:
            if len(self._ctx_cache) > 256:
                self._ctx_cache.clear()
            ctx = text_context(
                self.frozen.text_encoder, caption, device, fixed_length=self.config.text_fixed_length
            )
            self._ctx_cache[caption] = (ctx.to(self.dtype_), packing.pack_text(ctx, VEC_DIM)["ctx_ids"])
        return self._ctx_cache[caption]

    def _cast_inputs(self, kwargs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Streams, context and vector in the DiT dtype; timesteps stay fp32 (bf16 collapses t near 1); ids stay int."""
        out = {}
        for k, v in kwargs.items():
            if v.is_floating_point() and not (k.endswith("_timesteps") or k == "timesteps_ctx"):
                v = v.to(self.dtype_)
            out[k] = v
        return out

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, Any]]:
        cfg = self.config
        cams = self._cameras(batch)  # (B, n_cams, T, 3, H, W)
        b, _, t = cams.shape[:3]
        if t != cfg.window_frames:
            raise ValueError(
                f"training needs {cfg.window_frames} frames per camera (from observation_delta_indices), got {t}"
            )
        device = cams.device
        state, past_actions = self._conditioning_inputs(batch)
        state = state.to(device)
        if past_actions is not None:
            past_actions = past_actions.to(device)
        actions = self._flip(batch[ACTION].float()).to(device)  # (B, chunk, D)
        if actions.shape[1:] != (cfg.chunk_size, cfg.action_dim):
            raise ValueError(
                f"actions must be (B, {cfg.chunk_size}, {cfg.action_dim}), got {tuple(actions.shape)}"
            )
        keep = self._valid_windows(batch, b, device)
        captions = self._captions(batch, b, training=self.training)
        if self.training and cfg.caption_dropout > 0:
            drop = (torch.rand(b) < cfg.caption_dropout).tolist()
            captions = ["" if d else c for c, d in zip(captions, drop, strict=True)]
        # Caption grouping needs Python indices; transfer the combined flags once for the whole batch.
        idx_keep = [i for i, valid in enumerate(keep.tolist()) if valid]
        if not idx_keep:
            zero = next(p for p in self.parameters() if p.requires_grad).sum() * 0.0
            return zero, {"video_mse": 0.0, "action_mse": 0.0, "n_valid_windows": 0}
        ctxs = {i: self._context(captions[i], device)[0] for i in idx_keep}
        groups: dict[int, list[int]] = defaultdict(
            list
        )  # the model has no attention mask: batch equal lengths
        for i in idx_keep:
            groups[ctxs[i].shape[1]].append(i)

        total = len(idx_keep)
        loss: Tensor | float = 0.0
        mse = torch.zeros(2, device=device)
        augment = self.training and cfg.augment
        camera_hw = tuple(cams.shape[-2:])
        for idxs in groups.values():
            nb = len(idxs)
            videos = torch.stack(
                [
                    packing.materialize_video(
                        cams[i],
                        packing.sample_augmentation(camera_hw=camera_hw) if augment else None,
                        device,
                        layout=cfg.camera_layout,
                        canvas_hw=cfg.canvas_hw,
                    )
                    for i in idxs
                ]
            )  # (nb, 3, T, Hc, Wc)
            video = self.packer.pack_video(cfg, self.frozen.video_vae, videos, targets=True)
            action = self.packer.pack_actions(
                cfg, state[idxs], past_actions[idxs] if past_actions is not None else None, actions[idxs]
            )
            text = packing.pack_text(torch.cat([ctxs[i] for i in idxs]), VEC_DIM)
            timesteps = packing.sample_timesteps(
                nb, width=cfg.train_timestep_width, shift=cfg.train_timestep_shift
            )
            action_timesteps = timesteps
            if cfg.separate_timesteps:
                timesteps = torch.sigmoid(torch.randn(nb) * cfg.video_logit_std + cfg.video_logit_mean)
            kwargs, targets = packing.build_forward_kwargs(
                video,
                action,
                text,
                timesteps,
                self.modality,
                action_timesteps=action_timesteps,
                conditioning_noise_max=cfg.conditioning_noise_max if self.training else 0.0,
            )
            pred = self.dit(**self._cast_inputs(kwargs))
            losses = packing.flow_loss(
                pred,
                targets,
                self.modality,
                cfg.action_loss_weight,
                cfg.video_loss_weight,
                reduction=cfg.loss_reduction,
                channel_weights=cfg.action_channel_weights,
            )
            w = nb / total
            loss = loss + losses["loss"] * w
            mse += torch.stack((losses["video_mse"], losses["action_mse"])).detach() * w
        video_mse, action_mse = mse.tolist()
        return loss, {"video_mse": video_mse, "action_mse": action_mse, "n_valid_windows": total}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _sample(self, cond: dict[str, Tensor], caption: str, seed: int) -> Tensor:
        """Joint video + action denoising from pure noise -> ``(chunk, D)`` in model units / action_scale."""
        cfg, m, mdt = self.config, self.modality, self.dtype_
        ak, ck = f"x_{m}", f"x_{m}_cond"
        device = cond["x_video_cond"].device
        n_pred = self.packer.predicted_latent_frames(cfg)
        rng = torch.Generator().manual_seed(seed)
        video_noise = torch.randn(1, packing.LATENT_CHANNELS, n_pred, *cfg.latent_hw, generator=rng)
        x_video, x_video_ids = batched_prc_vid(
            video_noise,
            self.packer.predicted_video_times(cfg, 1),
        )
        times = self.packer.action_times(cfg, 1)
        action_noise = torch.randn(1, cfg.action_dim, cfg.chunk_size, generator=rng)
        x_action, x_action_ids = batched_prc_audio(action_noise, times_to_ids(times))
        # The solver state stays fp32 (scaled joint targets would lose ~0.01 rad per bf16 round trip);
        # inputs are cast at the model boundary.
        flow = {"x_video": x_video.to(device), ak: x_action.to(device)}
        fixed = {
            "x_video_ids": x_video_ids.to(device),
            f"{ak}_ids": x_action_ids.to(device),
            "x_video_cond": cond["x_video_cond"].to(device, mdt),
            "x_video_cond_ids": cond["x_video_cond_ids"].to(device),
            "x_video_cond_timesteps": torch.zeros(1, cond["x_video_cond"].shape[1], device=device),
            ck: cond[ck].to(device, mdt),
            f"{ck}_ids": cond[f"{ck}_ids"].to(device),
            f"{ck}_timesteps": torch.zeros(1, cond[ck].shape[1], device=device),
            "vector": torch.zeros(1, VEC_DIM, device=device, dtype=mdt),
        }
        guidance = {
            "x_video": cfg.guidance_scale,
            ak: cfg.guidance_scale if cfg.guidance_scale_action is None else cfg.guidance_scale_action,
        }
        ctx_c = self._context(caption, device)
        ctx_uc = self._context("", device) if any(g != 1.0 for g in guidance.values()) else None
        dit = self._inference_dit()

        def predict(samples: dict[str, Tensor], t) -> dict[str, Tensor]:
            t = float(t) / 1000.0 if isinstance(t, Tensor) and not t.is_floating_point() else float(t)
            timesteps = {
                "x_video_timesteps": torch.full((1, samples["x_video"].shape[1]), t, device=device),
                f"{ak}_timesteps": torch.full((1, cfg.chunk_size), t, device=device),
            }
            model_in = {k: v.to(mdt) for k, v in samples.items()}
            if ctx_uc is None:  # guidance 1.0 on every stream: a single conditional pass
                ctx, ctx_ids = ctx_c
                pred = dit(
                    **model_in,
                    **fixed,
                    **timesteps,
                    ctx=ctx,
                    ctx_ids=ctx_ids,
                    timesteps_ctx=torch.zeros(ctx.shape[:2], device=device),
                )
                pred = {k: pred[k] for k in model_in}
            else:
                pred = sampling.cfg_two_pass(dit, model_in, fixed, timesteps, ctx_uc, ctx_c, guidance)
            return {k: v.float() for k, v in pred.items()}

        if cfg.sampler == "cosmos_unipc":
            out = sampling.cosmos_unipc_order2(
                flow, predict, n_steps=cfg.num_inference_steps, shift=cfg.sampler_shift
            )
        else:
            out = sampling.euler(flow, predict, n_steps=cfg.num_inference_steps, alpha=cfg.sampler_shift)
        return out[ak][0].float() / cfg.action_scale

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        """Observation batch -> ``(B, chunk_size, action_dim)`` in the (normalized) action space of the dataset."""
        if kwargs:
            raise NotImplementedError("flux3 does not implement RTC inference arguments yet")
        self.eval()
        cfg = self.config
        cams = self._cameras(batch)
        if cfg.conditioning == "frame":
            cams = cams[:, :, :1]  # frame checkpoints consume the current observation only
        elif cams.shape[2] != cfg.n_obs_steps:
            raise ValueError("History prediction needs exactly n_obs_steps image/state observations")
        states, past = self._conditioning_inputs(batch)
        captions = self._captions(batch, states.shape[0])
        chunks = []
        for i, caption in enumerate(captions):
            canvas = packing.materialize_video(
                cams[i], None, states.device, layout=cfg.camera_layout, canvas_hw=cfg.canvas_hw
            )
            cond = self.packer.pack_video(cfg, self.frozen.video_vae, canvas[None], targets=False)
            action = self.packer.pack_actions(
                cfg,
                states[i : i + 1],
                past[i : i + 1] if past is not None else None,
                targets=False,
            )
            cond.update(action)
            chunks.append(self._sample(cond, caption, cfg.inference_seed))
        return self._flip(torch.stack(chunks)).float()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        if self.config.use_relative_actions:
            raise NotImplementedError(
                "flux3 relative-action execution requires RTC, which is not implemented yet. "
                "predict_action_chunk is available for offline evaluation; training remains supported."
            )
        if kwargs:
            raise NotImplementedError("flux3 does not implement RTC inference arguments yet")
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
