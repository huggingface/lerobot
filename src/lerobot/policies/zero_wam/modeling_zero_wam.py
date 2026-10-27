# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Zero-WAM policy: in-context world-action modeling from human videos.

Training reproduces upstream ``wan_va/train.py`` (per-frame flow matching on video latents and
actions, multi-chunk prediction, ICL/text dropout, random chunk size and attention window) on
LeRobot batches: camera clips are VAE-encoded and instructions UMT5-encoded on the fly, and
human-video latents are looked up from ``config.icl_pairs_path``.

Inference reproduces the ICL path of upstream ``wan_va/wan_va_server.py``: the optional human
video is cached once per episode as persistent context, each chunk denoises video latents then
actions, and the observed keyframes plus the executed actions replace the prediction in the KV
cache before the next chunk. CFG branches are packed along the sequence, not the batch.
"""

import json
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from huggingface_hub import snapshot_download
from torch import Tensor

from lerobot.policies.lingbot_va.utils import (
    FlowMatchScheduler,
    WanVAEStreamingWrapper,
    _sample_timestep_id,
    clean_prompt,
    data_seq_to_patch,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration_zero_wam import VAE_SPATIAL_DOWNSAMPLE, ZeroWAMConfig
from .transformer import (
    ICL_CACHE_TYPE,
    OBSERVATION_CACHE_TYPE,
    PREDICTION_CACHE_TYPE,
    ICLAttentionBackend,
    WanICLTransformer3DModel,
)


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, h_shift=0, action=False) -> Tensor:
    """RoPE grid ``[4, f*h*w]`` of (frame, height, width, stream-type) ids; upstream ``get_mesh_id``."""
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h_shift, h + h_shift)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff = ff + (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    return torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)


def shift_latents_for_mcp(latents: Tensor, frame_shift: int) -> tuple[Tensor, Tensor]:
    """Shift video latents ``frame_shift`` frames into the future, padding with the last frame."""
    batch_size, _, num_frames, _, _ = latents.shape
    pad_frames = min(frame_shift, num_frames)
    shifted = torch.cat(
        [latents[:, :, pad_frames:], latents[:, :, -1:].expand(-1, -1, pad_frames, -1, -1)], dim=2
    )
    valid = (torch.arange(num_frames, device=latents.device) + frame_shift) < num_frames
    return shifted, valid.view(1, 1, num_frames, 1, 1).expand(batch_size, -1, -1, -1, -1)


def read_video_at_fps(video_path: str, fps: float) -> Tensor:
    """Decode a video as ``[T, 3, H, W]`` in [0, 1], resampled to ``fps`` (upstream ``_read_icl_video``)."""
    require_package("av", extra="dataset")
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate or 0)
        frames = torch.stack(
            [torch.from_numpy(f.to_ndarray(format="rgb24")) for f in container.decode(stream)]
        )
    total = frames.shape[0]
    if src_fps > 0:
        count = max(1, round(total / src_fps * fps))
        indices = torch.clamp((torch.arange(count) * src_fps / fps).floor().long(), 0, total - 1)
        frames = frames[torch.unique(indices)]
    return frames.permute(0, 3, 1, 2).float() / 255.0


class ZeroWAMPolicy(PreTrainedPolicy):
    """LeRobot wrapper for the Zero-WAM in-context video-action world model."""

    config_class = ZeroWAMConfig
    name = "zero_wam"
    # The streaming KV cache needs whole chunks of observed keyframes, so a mid-episode
    # instruction change applies from the next chunk instead of dropping queued actions.
    _action_queue_attrs = ()
    _fsdp_wrap_modules = ["WanICLTransformerBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: ZeroWAMConfig, **kwargs):
        require_package("diffusers", extra="zero_wam")
        require_package("transformers", extra="zero_wam")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.dtype = config.dtype
        self.frozen_dtype = config.frozen_dtype
        ICLAttentionBackend.set_backend(config.attention_backend)

        self.transformer = WanICLTransformer3DModel(
            patch_size=tuple(config.patch_size),
            num_attention_heads=config.num_attention_heads,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            action_dim=config.action_dim,
            text_dim=config.text_dim,
            freq_dim=config.freq_dim,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps,
            rope_max_seq_len=config.rope_max_seq_len,
            action_inner_dim=config.action_inner_dim,
            action_ffn_dim=config.action_ffn_dim,
            attn_window=config.attn_window,
            enable_mcp=config.enable_mcp,
            num_mcp_modules=config.num_mcp_modules,
            mcp_blocks_per_group=config.mcp_blocks_per_group,
            mcp_hidden_collect_layers=tuple(config.mcp_hidden_collect_layers),
        ).to(self.dtype)
        self.transformer.gradient_checkpointing = config.gradient_checkpointing
        # Kept only so released checkpoints load strictly; never used, so it would get no optimizer state.
        self.transformer.patch_embedding.requires_grad_(False)

        # Frozen VAE / text encoder live outside the module registry: they are neither saved in
        # model.safetensors nor moved by ``.to()``, and load lazily on first use.
        self._frozen: dict = {}
        self._text_cache: dict[str, Tensor] = {}
        self._precomputed_text: dict[str, Tensor] = {}
        self._icl_pairs: dict[int, list[dict]] | None = None
        self._train_schedulers: dict[str, FlowMatchScheduler] = {}
        self.reset()

    # ------------------------------------------------------------------ frozen components
    def _frozen_component_path(self, *components: str) -> str:
        """Resolve pinned frozen assets without downloading the transformer again."""
        cfg = self.config
        if cfg.frozen_components_revision and not Path(cfg.frozen_components_path).is_dir():
            return snapshot_download(
                cfg.frozen_components_path,
                revision=cfg.frozen_components_revision,
                allow_patterns=[f"{component}/*" for component in components],
            )
        return cfg.frozen_components_path

    def _ensure_frozen_modules(self) -> None:
        if self._frozen:
            return
        path = self._frozen_component_path("vae")
        vae = load_vae(path, torch_dtype=self.frozen_dtype, torch_device=self.config.device, subfolder="vae")
        self._frozen = {"vae": vae.eval(), "streaming_vae": WanVAEStreamingWrapper(vae)}
        if self.config.text_embedding_cache_path:
            cache = torch.load(self.config.text_embedding_cache_path, map_location="cpu", weights_only=True)
            self._precomputed_text = {k: v.to(self.frozen_dtype) for k, v in cache.items()}

    def _ensure_text_encoder(self) -> None:
        """Load UMT5 on first use; with a precomputed cache it is only needed for unseen prompts."""
        if "text_encoder" in self._frozen:
            return
        path = self._frozen_component_path("text_encoder", "tokenizer")
        self._frozen["text_encoder"] = load_text_encoder(
            path,
            torch_dtype=self.frozen_dtype,
            torch_device=self.config.text_encoder_device,
            subfolder="text_encoder",
        ).eval()
        self._frozen["tokenizer"] = load_tokenizer(path, subfolder="tokenizer")

    @property
    def _vae(self):
        return self._frozen["vae"]

    @property
    def _streaming_vae(self) -> WanVAEStreamingWrapper:
        return self._frozen["streaming_vae"]

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ encoders
    @torch.no_grad()
    def _encode_text(self, prompt: str) -> Tensor:
        """UMT5 embedding ``[1, max_sequence_length, text_dim]`` zero-padded after the prompt tokens."""
        if prompt not in self._text_cache:
            max_len = self.config.max_sequence_length
            if prompt in self._precomputed_text:
                tokens_emb = self._precomputed_text[prompt]
                embedding = tokens_emb.new_zeros(1, max_len, tokens_emb.shape[-1])
                length = min(tokens_emb.shape[0], max_len)
                embedding[0, :length] = tokens_emb[:length]
                self._text_cache[prompt] = embedding
                return embedding.to(self.config.device, self.dtype)
            self._ensure_text_encoder()
            tokens = self._frozen["tokenizer"](
                [clean_prompt(prompt)],
                padding="max_length",
                max_length=max_len,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            encoder = self._frozen["text_encoder"]
            device = next(encoder.parameters()).device
            hidden = encoder(tokens.input_ids.to(device), tokens.attention_mask.to(device)).last_hidden_state
            length = int(tokens.attention_mask.gt(0).sum())
            embedding = hidden.new_zeros(1, max_len, hidden.shape[-1])
            embedding[:, :length] = hidden[:, :length]
            self._text_cache[prompt] = embedding.to("cpu", self.frozen_dtype)
        return self._text_cache[prompt].to(self.config.device, self.dtype)

    def _normalize_vae_output(self, enc_out: Tensor) -> Tensor:
        mu, _ = torch.chunk(enc_out, 2, dim=1)
        mean = torch.tensor(self._vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)
        inv_std = (1.0 / torch.tensor(self._vae.config.latents_std, device=mu.device)).view(1, -1, 1, 1, 1)
        return ((mu.float() - mean) * inv_std).to(mu)

    def _resize(self, frames: Tensor, size: tuple[int, int]) -> Tensor:
        """``[N, T, C, H, W]`` in [0, 1] -> ``[N, C, T, h, w]`` in [-1, 1] for the VAE."""
        n, t = frames.shape[:2]
        frames = F.interpolate(frames.flatten(0, 1).float(), size=size, mode="bilinear", align_corners=False)
        frames = frames.view(n, t, *frames.shape[1:]).permute(0, 2, 1, 3, 4).contiguous()
        return (frames * 2.0 - 1.0).to(self.frozen_dtype)

    @torch.no_grad()
    def _encode_cameras(self, clips: list[Tensor], streaming: bool) -> Tensor:
        """Encode one ``[T, C, H, W]`` clip per camera and concatenate the latents on width."""
        size = (self.config.height, self.config.width)
        video = self._resize(torch.stack(clips), size).to(next(self._vae.parameters()).device)
        if streaming:
            enc_out = self._streaming_vae.encode_chunk(video)
        else:
            dist = self._vae.encode(video).latent_dist
            enc_out = torch.cat([dist.mean, dist.logvar], dim=1)
        latent = self._normalize_vae_output(enc_out)
        return torch.cat(latent.split(1, dim=0), dim=-1).to(self.config.device, self.dtype)

    # ------------------------------------------------------------------ training
    def _scheduler(self, name: str, shift: float) -> FlowMatchScheduler:
        if name not in self._train_schedulers:
            scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
            scheduler.set_timesteps(1000, training=True)
            self._train_schedulers[name] = scheduler
        return self._train_schedulers[name]

    @torch.no_grad()
    def _add_noise(self, latent, scheduler, action_mode, noisy_cond_prob, frame_shift=0):
        """Per-frame flow-matching noising of one stream (upstream ``Trainer._add_noise``)."""
        cfg = self.config
        device = latent.device
        b, _, f, h, w = latent.shape
        timesteps = scheduler.timesteps[_sample_timestep_id(f, num_train_timesteps=1000)].to(device)
        noise = torch.randn_like(latent)
        noisy = scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = scheduler.training_target(latent, noise, timesteps)
        pf, ph, pw = (1, 1, 1) if action_mode else cfg.patch_size
        grid = get_mesh_id(f // pf, h // ph, w // pw, t=int(action_mode), f_shift=frame_shift).to(device)
        if torch.rand(1).item() < noisy_cond_prob:
            cond_ids = _sample_timestep_id(
                f,
                min_timestep_bd=cfg.noisy_cond_min_timestep_bd,
                max_timestep_bd=cfg.noisy_cond_max_timestep_bd,
                num_train_timesteps=1000,
            )
            cond_timesteps = scheduler.timesteps[cond_ids].to(device)
            latent = scheduler.add_noise(latent, torch.randn_like(latent), cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)
        return {
            "timesteps": timesteps[None].repeat(b, 1),
            "noisy_latents": noisy,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(b, 1),
            "grid_id": grid[None].repeat(b, 1, 1),
        }

    def _load_icl_pairs(self) -> dict[int, list[dict]]:
        if self._icl_pairs is None:
            self._icl_pairs = {}
            if self.config.icl_pairs_path:
                for line in Path(self.config.icl_pairs_path).read_text().splitlines():
                    if line.strip():
                        pair = json.loads(line)
                        self._icl_pairs.setdefault(int(pair["episode_index"]), []).append(pair)
        return self._icl_pairs

    def _find_icl_latent(self, episode_index: int, frame_index: int) -> str | None:
        """Latent path of a human video paired with the robot segment containing ``frame_index``."""
        candidates = [
            p
            for p in self._load_icl_pairs().get(episode_index, [])
            if int(p["start_frame"]) <= frame_index < int(p["end_frame"])
        ]
        if not candidates:
            return None
        return candidates[int(torch.randint(len(candidates), (1,)).item())]["latent_path"]

    def _load_icl_latent(self, latent_path: str) -> tuple[Tensor, Tensor | None]:
        """Read a human-video latent ``.pth``: ``latent`` as ``[1, 48, F, H, W]`` plus its text embedding."""
        item = torch.load(latent_path, map_location="cpu", weights_only=True)
        latent = item["latent"]
        if latent.ndim == 2:
            f, h, w = (int(item[k]) for k in ("latent_num_frames", "latent_height", "latent_width"))
            latent = latent.view(f, h, w, -1).permute(3, 0, 1, 2)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        text_emb = item.get("text_emb")
        if torch.is_tensor(text_emb) and text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(0)
        return latent.to(self.config.device, self.dtype), text_emb

    @torch.no_grad()
    def _prepare_training_sample(self, batch: dict, i: int) -> dict:
        """Build one upstream ``_prepare_input_dict`` sample from item ``i`` of a LeRobot batch."""
        cfg = self.config
        device = cfg.device
        num_latent = cfg.train_num_latent_frames
        apf = cfg.action_per_frame

        latents = self._encode_cameras([batch[k][i] for k in cfg.obs_cam_keys], streaming=False)

        # Action target [1, action_dim, F, apf, 1]: frame 0 is zero history (the conditioning frame
        # has no preceding actions), later frames hold the actions leading to their video frames.
        used = torch.as_tensor(cfg.used_action_channel_ids, device=device)
        actions = batch[ACTION][i].to(device).float().clamp(-cfg.action_clip, cfg.action_clip)
        valid = torch.ones(actions.shape[0], dtype=torch.bool, device=device)
        if f"{ACTION}_is_pad" in batch:
            valid = ~batch[f"{ACTION}_is_pad"][i].to(device)
        full = actions.new_zeros(num_latent * apf, cfg.action_dim)
        full_mask = torch.zeros_like(full, dtype=torch.bool)
        full[apf:, used] = actions * valid[:, None]
        full_mask[:apf, used] = True
        full_mask[apf:, used] = valid[:, None]
        actions = rearrange(full, "(f n) c -> 1 c f n 1", f=num_latent).to(self.dtype)
        actions_mask = rearrange(full_mask, "(f n) c -> 1 c f n 1", f=num_latent)

        task = batch["task"][i] if isinstance(batch.get("task"), list | tuple) else batch.get("task", "")
        drop_text = torch.rand(1).item() < cfg.droptext_target
        text_emb = self._encode_text("" if drop_text else task)

        chunk_size = cfg.train_frame_chunk_size or int(torch.randint(1, cfg.max_frame_chunk_size + 1, (1,)))
        latent_dict = self._add_noise(
            latents, self._scheduler("latent", cfg.snr_shift), False, cfg.noisy_img_prob
        )
        action_dict = self._add_noise(actions, self._scheduler("action", cfg.action_snr_shift), True, 0.0)
        action_dict["actions_mask"] = actions_mask
        # Exclude padded camera targets at the end of an episode.
        camera_pad = batch.get(f"{cfg.obs_cam_keys[0]}_is_pad")
        if camera_pad is not None:
            valid_video = (~camera_pad[i].to(device))[::4].view(1, 1, num_latent, 1, 1)
            latent_dict["valid_mask"] = valid_video
        text_parts = [text_emb]
        encoder_seq = [torch.zeros(text_emb.shape[1], device=device, dtype=torch.int)]
        icl_latent_dict = None
        latent_path = None
        if "episode_index" in batch and torch.rand(1).item() >= cfg.drop_icl:
            frame = batch["frame_index"][i] if "frame_index" in batch else batch["index"][i]
            latent_path = self._find_icl_latent(int(batch["episode_index"][i]), int(frame))
        if latent_path is not None:
            icl_latent, icl_text = self._load_icl_latent(latent_path)
            _, _, fr, hh, ww = icl_latent.shape
            p = cfg.patch_size
            grid = get_mesh_id(fr // p[0], hh // p[1], ww // p[2], t=0, h_shift=cfg.icl_rope_h).to(device)
            icl_latent_dict = {
                "latent": icl_latent,
                "timesteps": torch.zeros(1, fr, device=device),
                "grid_id": grid[None],
            }
            icl_text = icl_text.to(device, self.dtype) if icl_text is not None else text_emb
            text_parts.append(icl_text)
            encoder_seq.append(torch.ones(icl_text.shape[1], device=device, dtype=torch.int))

        window = cfg.train_attn_window or int(torch.randint(4, cfg.max_attn_window + 1, (1,)))
        input_dict = {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "icl_latent_dict": icl_latent_dict,
            "text_emb": torch.cat(text_parts, dim=1),
            "encoder_seq_ids": torch.cat(encoder_seq),
            "chunk_size": chunk_size,
            "max_frame_chunk_size": cfg.max_frame_chunk_size,
            "window_size": window,
        }
        if cfg.enable_mcp:
            mcp_dicts = []
            for module_index in range(cfg.num_mcp_modules):
                shift = (1 + module_index * cfg.future_chunk_stride) * chunk_size
                shifted, valid_mask = shift_latents_for_mcp(latents, shift)
                mcp_dict = self._add_noise(
                    shifted, self._scheduler("mcp", cfg.mcp_snr_shift), False, cfg.noisy_img_prob, shift
                )
                if "valid_mask" in latent_dict:
                    shifted_valid, _ = shift_latents_for_mcp(latent_dict["valid_mask"].float(), shift)
                    valid_mask = valid_mask & shifted_valid.bool()
                mcp_dict["valid_mask"] = valid_mask
                mcp_dicts.append(mcp_dict)
            input_dict["mcp_latent_dicts"] = mcp_dicts
        return input_dict

    def _sample_loss(self, input_dict: dict) -> tuple[Tensor, dict[str, Tensor]]:
        """Upstream ``Trainer.compute_loss`` for one sample."""
        cfg = self.config
        out = self.transformer(input_dict, train_mode=True)
        latent_pred, action_pred = out[0], out[1]
        mcp_preds = out[2] if cfg.enable_mcp else []
        ld, ad = input_dict["latent_dict"], input_dict["action_dict"]
        f, h, w = ld["targets"].shape[-3:]
        latent_pred = data_seq_to_patch(cfg.patch_size, latent_pred, f, h, w, batch_size=1)
        action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=ad["targets"].shape[-3])

        def weight(name, shift, timesteps, enabled):
            if not enabled:
                return torch.ones_like(timesteps)[:, None, :, None, None]
            flat = self._scheduler(name, shift).training_weight(timesteps.flatten())
            return flat.reshape(timesteps.shape)[:, None, :, None, None]

        latent_w = weight("latent", cfg.snr_shift, ld["timesteps"], cfg.video_loss_reweight)
        latent_loss = F.mse_loss(latent_pred.float(), ld["targets"].float(), reduction="none")
        latent_mask = ld.get("valid_mask", torch.ones_like(latent_loss, dtype=torch.bool))
        latent_loss = (latent_loss * latent_w * latent_mask).sum() / latent_mask.expand_as(
            latent_loss
        ).sum().clamp_min(1)

        action_w = weight("action", cfg.action_snr_shift, ad["timesteps"], cfg.action_loss_reweight)
        action_loss = F.mse_loss(action_pred.float(), ad["targets"].float(), reduction="none")
        action_loss = (action_loss * action_w * ad["actions_mask"].float()).mean()

        mcp_loss = latent_loss.new_zeros(())
        mcp_dicts = input_dict.get("mcp_latent_dicts", []) if mcp_preds else []
        mcp_weights = cfg.mcp_loss_weights if mcp_preds else ()
        for weight_i, pred, md in zip(mcp_weights, mcp_preds, mcp_dicts, strict=True):
            mcp_frames, mcp_height, mcp_width = md["targets"].shape[-3:]
            pred = data_seq_to_patch(cfg.patch_size, pred, mcp_frames, mcp_height, mcp_width, batch_size=1)
            loss = F.mse_loss(pred.float(), md["targets"].float(), reduction="none")
            loss = loss * weight("mcp", cfg.mcp_snr_shift, md["timesteps"], cfg.video_loss_reweight)
            valid = md["valid_mask"].to(loss)
            mcp_loss = mcp_loss + weight_i * (loss * valid).sum() / valid.expand_as(loss).sum().clamp_min(1.0)
        loss = latent_loss + action_loss + mcp_loss
        return loss, {
            "latent_loss": latent_loss.detach(),
            "action_loss": action_loss.detach(),
            "mcp_loss": mcp_loss.detach(),
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Zero-WAM training loss, averaged over the (per-sample) items of the batch."""
        self._ensure_frozen_modules()
        batch_size = batch[ACTION].shape[0]
        total = None
        logs: dict[str, float] = {}
        for i in range(batch_size):
            loss, parts = self._sample_loss(self._prepare_training_sample(batch, i))
            total = loss if total is None else total + loss
            for k, v in parts.items():
                logs[k] = logs.get(k, 0.0) + float(v) / batch_size
        return total / batch_size, logs

    # ------------------------------------------------------------------ inference state
    def reset(self) -> None:
        cfg = self.config
        self._action_queue: deque = deque()
        self._obs_buffer: list[dict[str, Tensor]] = []
        self._chunk_idx = 0
        self._exec_step = 0
        self._started = False
        self._init_latent: Tensor | None = None
        self._last_actions: Tensor | None = None
        self._prompt: str | None = None
        self.last_predicted_latents: Tensor | None = None
        self._infer_scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
        self._infer_action_scheduler = FlowMatchScheduler(
            shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True
        )
        if hasattr(self, "transformer"):
            self.transformer.clear_cache("pos")
        if self._frozen:
            self._streaming_vae.clear_cache()

    def drop_queued_actions(self) -> None:
        """Queued actions stay: the next chunk picks up the new instruction."""

    def count_queued_actions(self) -> int:
        return len(self._action_queue)

    @property
    def _latent_hw(self) -> tuple[int, int]:
        cfg = self.config
        return cfg.height // VAE_SPATIAL_DOWNSAMPLE, cfg.width // VAE_SPATIAL_DOWNSAMPLE * len(
            cfg.obs_cam_keys
        )

    @property
    def _action_mask(self) -> Tensor:
        mask = torch.zeros(self.config.action_dim, dtype=torch.bool, device=self.config.device)
        mask[self.config.used_action_channel_ids] = True
        return mask

    def _batch_task(self, batch: dict | None) -> str | None:
        if batch is None or "task" not in batch:
            return None
        task = batch["task"]
        return str(task[0] if isinstance(task, list | tuple) else task)

    def _set_prompt(self, prompt: str) -> None:
        cfg = self.config
        self._prompt = prompt
        self._empty_emb = self._encode_text("")
        self._prompt_emb = self._encode_text(prompt)
        self._use_icl = bool(cfg.icl_latent_path or cfg.icl_video_path)
        self._use_icl_cfg = self._use_icl and cfg.icl_guidance_scale > 1.0
        self._text_cfg = not self._use_icl_cfg and cfg.guidance_scale > 1.0
        self._target_emb = self._empty_emb if cfg.guidance_scale < 0 else self._prompt_emb

    # ------------------------------------------------------------------ ICL context
    def _read_icl_video(self, video_path: str) -> Tensor:
        """Decode an mp4 at ``icl_fps`` into ``[1, 3, T, icl_height, icl_width]`` in [-1, 1]."""
        frames = read_video_at_fps(video_path, self.config.icl_fps)
        return self._resize(frames[None], (self.config.icl_height, self.config.icl_width))

    @torch.no_grad()
    def _cache_icl_context(self) -> None:
        """Encode the human video once and store it as persistent, bidirectional ICL cache."""
        cfg = self.config
        device = cfg.device
        text_emb = None
        if cfg.icl_latent_path:
            latent, text_emb = self._load_icl_latent(cfg.icl_latent_path)
        else:
            # Whole-clip encode, as ``encode_human_videos.py`` made the training latents: the
            # streaming encoder only accepts camera-sized keyframe chunks.
            assert cfg.icl_video_path is not None
            video = self._read_icl_video(cfg.icl_video_path).to(next(self._vae.parameters()).device)
            dist = self._vae.encode(video).latent_dist
            latent = self._normalize_vae_output(torch.cat([dist.mean, dist.logvar], dim=1)).to(
                device, self.dtype
            )
        text_emb = self._prompt_emb if text_emb is None else text_emb.to(device, self.dtype)
        p = cfg.patch_size
        grid = get_mesh_id(
            latent.shape[-3] // p[0], latent.shape[-2] // p[1], latent.shape[-1] // p[2], 0
        ).to(device)
        grid[1] += cfg.icl_rope_h
        tokens = grid.shape[1]
        self.transformer(
            {
                "latent_res_lst": {
                    "noisy_latents": latent,
                    "timesteps": torch.zeros(latent.shape[2], device=device),
                    "cache_type_ids": torch.full([tokens], ICL_CACHE_TYPE, device=device, dtype=torch.int),
                },
                "latent_grid_id": grid,
                "current_seq_ids": torch.zeros(tokens, device=device, dtype=torch.int),
                "current_frame_ids": torch.zeros(tokens, device=device, dtype=torch.int),
                "encoder_seq_ids": torch.zeros(text_emb.shape[1], device=device, dtype=torch.int),
                "text_emb": text_emb,
            },
            update_cache=1,
            cache_name="pos",
            mode="forward_latent_only",
        )

    # ------------------------------------------------------------------ streaming inference
    def _stream_input(self, data: Tensor, timestep, mode: str, cache_type: int) -> dict:
        """Upstream ``_prepare_icl_stream`` + ``_pack_icl_cfg`` for one video or action chunk."""
        cfg = self.config
        device = cfg.device
        is_video = mode == "video"
        p = cfg.patch_size
        grid = get_mesh_id(
            data.shape[-3],
            data.shape[-2] // (p[1] if is_video else 1),
            data.shape[-1] // (p[2] if is_video else 1),
            0 if is_video else 1,
            f_shift=cfg.frame_chunk_size * self._chunk_idx,
        ).to(device)
        stream = {
            "noisy_latents": data,
            "timesteps": torch.full([data.shape[2]], float(timestep), device=device),
        }
        cfg_active = is_video and (self._use_icl_cfg or self._text_cfg)
        tokens = grid.shape[1]
        if cfg_active:
            stream["noisy_latents"] = stream["noisy_latents"].repeat(1, 1, 2, 1, 1)
            stream["timesteps"] = stream["timesteps"].repeat(2)
            grid = grid.repeat(1, 2)
            seq_ids = torch.cat([torch.zeros(tokens, dtype=torch.int), torch.ones(tokens, dtype=torch.int)])
            second = self._target_emb if self._use_icl_cfg else self._empty_emb
            text_emb = torch.cat([self._target_emb, second], dim=1)
            text_len = self._target_emb.shape[1]
            encoder_seq = torch.cat(
                [torch.zeros(text_len, dtype=torch.int), torch.ones(text_len, dtype=torch.int)]
            )
        else:
            seq_ids = torch.zeros(tokens, dtype=torch.int)
            text_emb = self._target_emb
            encoder_seq = torch.zeros(text_emb.shape[1], dtype=torch.int)
        tokens = grid.shape[1]
        frame_id = 2 * self._chunk_idx + (0 if is_video else 1)
        stream["cache_type_ids"] = torch.full([tokens], cache_type, device=device, dtype=torch.int)
        return {
            "latent_res_lst" if is_video else "action_res_lst": stream,
            "latent_grid_id" if is_video else "action_grid_id": grid,
            "current_seq_ids": seq_ids.to(device),
            "current_frame_ids": torch.full([tokens], frame_id, device=device, dtype=torch.int),
            "text_emb": text_emb.to(self.dtype),
            "encoder_seq_ids": encoder_seq.to(device),
            "_cfg_active": cfg_active,
        }

    def _transformer(self, input_dict: dict, **kwargs) -> Tensor:
        input_dict = {k: v for k, v in input_dict.items() if not k.startswith("_")}
        return self.transformer(input_dict, cache_name="pos", **kwargs)

    @torch.no_grad()
    def _infer_chunk(self) -> Tensor:
        """Denoise one chunk of video latents, then actions; returns normalized ``[1, 30, F, apf, 1]``."""
        cfg = self.config
        device = cfg.device
        latent_h, latent_w = self._latent_hw
        fc = cfg.frame_chunk_size
        action_mask = self._action_mask
        latents = torch.randn(1, cfg.in_channels, fc, latent_h, latent_w, device=device, dtype=self.dtype)
        actions = torch.randn(1, cfg.action_dim, fc, cfg.action_per_frame, 1, device=device, dtype=self.dtype)
        _ = torch.randn_like(actions)[:, ~action_mask]  # keep upstream's RNG stream
        self._infer_scheduler.set_timesteps(cfg.num_inference_steps)
        self._infer_action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        video_timesteps = F.pad(self._infer_scheduler.timesteps, (0, 1), value=0)

        first = self._chunk_idx == 0
        for step, timestep in enumerate(video_timesteps):
            last = step == len(video_timesteps) - 1
            latent_input = latents.clone()
            if first:
                assert self._init_latent is not None
                latent_input[:, :, :1] = self._init_latent[:, :, :1]
            input_dict = self._stream_input(latent_input, timestep, "video", PREDICTION_CACHE_TYPE)
            if first:
                input_dict["latent_res_lst"]["timesteps"][::fc] = 0
            noise = self._transformer(input_dict, update_cache=int(last), mode="forward_latent_only")
            if last:
                continue
            if input_dict["_cfg_active"]:
                scale = cfg.icl_guidance_scale if self._use_icl_cfg else cfg.guidance_scale
                cond, uncond = noise.chunk(2, dim=1)
                noise = uncond + scale * (cond - uncond)
            noise = data_seq_to_patch(cfg.patch_size, noise, fc, latent_h, latent_w)
            latents = self._infer_scheduler.step(noise, timestep, latents, return_dict=False)
            if first:
                assert self._init_latent is not None
                latents[:, :, :1] = self._init_latent[:, :, :1]

        for timestep in self._infer_action_scheduler.timesteps:
            action_input = actions.clone()
            action_input[:, ~action_mask] = 0
            input_dict = self._stream_input(action_input, timestep, "action", PREDICTION_CACHE_TYPE)
            noise = self._transformer(input_dict, update_cache=0, mode="forward_action_only")
            noise = rearrange(noise, "1 (f n) c -> 1 c f n 1", f=fc)
            actions = self._infer_action_scheduler.step(noise, timestep, actions, return_dict=False)
        actions[:, ~action_mask] = 0
        history = actions.clone()
        if first:
            history[:, :, 0] = 0
        self._last_actions = history
        self.last_predicted_latents = latents
        return actions

    @torch.no_grad()
    def _feed_observations(self) -> None:
        """Replace the predicted chunk in the KV cache by the observed keyframes and executed actions."""
        if not self._obs_buffer or self._last_actions is None:
            raise RuntimeError("Use select_action() to supply feedback between Zero-WAM chunks")
        self.transformer.clear_prediction_cache(-1, "pos")
        clips = [torch.stack([o[k] for o in self._obs_buffer]) for k in self.config.obs_cam_keys]
        latent = self._encode_cameras(clips, streaming=True)
        if self._chunk_idx == 0:
            latent = torch.cat([self._init_latent, latent], dim=2)
        self._transformer(
            self._stream_input(latent, 0, "video", OBSERVATION_CACHE_TYPE),
            update_cache=1,
            mode="forward_latent_only",
        )
        self._transformer(
            self._stream_input(self._last_actions.to(latent), 0, "action", OBSERVATION_CACHE_TYPE),
            update_cache=1,
            mode="forward_action_only",
            clean_window_cache=True,
        )
        self._chunk_idx += 1
        self._obs_buffer = []

    def _snapshot(self, batch: dict) -> dict[str, Tensor]:
        """Current camera frames ``[C, H, W]`` in [0, 1] for the first batch item."""
        frames = {}
        for key in self.config.obs_cam_keys:
            image = batch[key]
            if image.dim() == 4 and image.shape[0] != 1:
                raise ValueError("Zero-WAM streaming inference requires batch size 1")
            frames[key] = (image[0] if image.dim() == 4 else image).detach().float()
        return frames

    def _queue_chunk(self, actions: Tensor, skip_first_frame: bool) -> None:
        used = actions[0, self.config.used_action_channel_ids]  # [n_used, F, apf, 1]
        if skip_first_frame:
            used = used[:, 1:]
        steps = used.flatten(1).transpose(0, 1).float()
        self._action_queue.extend(steps.clamp(-self.config.action_clip, self.config.action_clip))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict the next chunk of normalized actions ``[1, n_steps, n_used]`` (first call: from ``batch``)."""
        self.eval()
        self._ensure_frozen_modules()
        task = self._batch_task(batch)
        if not self._started:
            self._set_prompt(task or "")
            if self._use_icl:
                self._cache_icl_context()
            self._streaming_vae.clear_cache()
            self._init_latent = self._encode_cameras(
                [f[None] for f in self._snapshot(batch).values()], streaming=True
            )
            self._started = True
        else:
            self._feed_observations()
            if task is not None and task != self._prompt:
                self._set_prompt(task)
        skip_first = self._chunk_idx == 0
        before = len(self._action_queue)
        self._queue_chunk(self._infer_chunk(), skip_first_frame=skip_first)
        return torch.stack(list(self._action_queue)[before:])[None]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Return the next normalized action ``[1, n_used]``, predicting a new chunk when the queue empties.

        Observations after each executed action are buffered as keyframes every
        ``video_frame_stride`` steps (``4 * frame_chunk_size`` per chunk) and fed back into the
        KV cache before the next chunk, as in the upstream client.
        """
        if not self._started:
            self.predict_action_chunk(batch)
        else:
            if self._exec_step % self.config.video_frame_stride == 0:
                self._obs_buffer.append(self._snapshot(batch))
            if not self._action_queue:
                self.predict_action_chunk(batch)
                self._exec_step = 0
        self._exec_step += 1
        return self._action_queue.popleft()[None]

    # ------------------------------------------------------------------ utilities
    @torch.no_grad()
    def decode_latents(self, latents: Tensor) -> Tensor:
        """Decode normalized video latents ``[1, 48, F, h, w]`` to uint8 frames ``[T, H, W, 3]``."""
        self._ensure_frozen_modules()
        vae = self._vae
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        latents = latents.float().cpu() * std + mean
        video = vae.decode(latents.to(next(vae.parameters()).device, vae.dtype), return_dict=False)[0]
        video = ((video.float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        return video[0].permute(1, 2, 3, 0).cpu()


__all__ = ["ZeroWAMPolicy", "get_mesh_id", "read_video_at_fps", "shift_latents_for_mcp"]
