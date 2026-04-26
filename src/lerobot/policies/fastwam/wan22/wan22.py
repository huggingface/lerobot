import numpy as np
import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Any, Optional, Sequence, Union

from .helpers.loader import load_wan22_ti2v_5b_components
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .wan_video_dit import WanVideoDiT


class Wan22Core(torch.nn.Module):
    """Standalone Wan2.2-TI2V-5B core without pipeline unit graph."""

    def __init__(
        self,
        dit: WanVideoDiT,
        vae,
        text_encoder,
        tokenizer,
        device="cpu",
        torch_dtype=torch.float32,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.train_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=train_shift,
        )
        self.infer_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=infer_shift,
        )
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        redirect_common_files=True,
        dit_config: dict[str, Any] | None = None,
        train_shift: float = 5.0,
        infer_shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        if dit_config is None:
            raise ValueError("`dit_config` is required for Wan22Core.from_wan22_pretrained().")
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=dit_config,
        )
        model = cls(
            dit=components.dit,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            train_shift=train_shift,
            infer_shift=infer_shift,
            num_train_timesteps=num_train_timesteps,
        )
        model.model_paths = {
            "dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.dit.to(*args, **kwargs)
        self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        return prompt_emb.to(device=self.device), mask

    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def _model_fn(self, 
                  latents: torch.Tensor, # [B, C, T, H, W]
                  timestep: torch.Tensor, # [B] or [1] (inference mode)
                  context: torch.Tensor, # [B, L, D]
                  context_mask: Optional[torch.Tensor] = None, # [B, L]
                  action: Optional[torch.Tensor] = None, # [B, T-1, a_dim]
                  fuse_vae_embedding_in_latents=False):
        return self.dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

    def build_inputs(self, sample, tiled=False):
        video = sample["video"]
        prompt = sample["prompt"]
        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"`sample['video']` must be a torch.Tensor with shape [B, 3, T, H, W], got {type(video)}"
            )
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        if isinstance(prompt, str):
            prompt_list = [prompt]
        elif isinstance(prompt, Sequence):
            prompt_list = list(prompt)
        else:
            raise TypeError(f"`sample['prompt']` must be str or list[str], got {type(prompt)}")

        batch_size, _, num_frames, height, width = video.shape
        if len(prompt_list) != batch_size:
            raise ValueError(
                f"Prompt batch mismatch: got len(prompt)={len(prompt_list)} and video batch={batch_size}"
            )
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")

        input_video = video.to(device=self.device, dtype=self.torch_dtype)
        input_latents = self._encode_video_latents(input_video, tiled=tiled) # [B, C, Latent_T, H', W']

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.dit, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True
        context, context_mask = self.encode_prompt(prompt_list)

        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor with shape [B, T, a_dim], got {type(action)}"
                )
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] <= 0:
                raise ValueError(f"`sample['action']` temporal dimension must be positive, got {action.shape[1]}")
            if action.shape[1] % (num_frames - 1) != 0:
                raise ValueError(
                    "`sample['action']` temporal dimension must be divisible by video transitions "
                    f"({num_frames - 1}), got {action.shape[1]}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
        }

    def training_loss(self, sample, tiled=False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        action = inputs["action"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]

        # 1. Continuous timestep sampling and noise injection.
        noise = torch.randn_like(input_latents)
        timestep = self.train_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_scheduler.add_noise(input_latents, noise, timestep)
        target = self.train_scheduler.training_target(input_latents, noise, timestep)

        # 2. fix first latent
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0: 1] = inputs["first_frame_latents"]

        pred = self._model_fn(
            latents=latents, # [B, C, Latent_T, H', W']
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        if inputs["first_frame_latents"] is not None:
            pred = pred[:, :, 1:]
            target = target[:, :, 1:]
        loss_per_sample = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3, 4))
        sample_weight = self.train_scheduler.training_weight(timestep).to(
            loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        loss_total = (loss_per_sample * sample_weight).mean()
        loss_dict = {
            "loss_video": float(loss_total.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def infer(
        self,
        prompt: str,
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **kwargs
    ):
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_frames:
            raise ValueError(
                f"`num_frames` must satisfy T % 4 == 1, got {num_frames}"
            )
        
        latent_t = (num_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if action is not None:
            action = action.to(device=self.device, dtype=latents.dtype)
            if action.ndim != 2:
                raise ValueError(f"`action` must be 2D [T, a_dim], got shape {tuple(action.shape)}")
            action = action.unsqueeze(0) # [1, T, a_dim]
            if action.shape[1] % (num_frames - 1) != 0:
                raise ValueError(
                    "`action` temporal dimension must be divisible by `num_frames - 1`, "
                    f"got {action.shape[1]} vs {num_frames - 1}"
                )
        if action_cfg_scale != 1.0 and action is None:
            raise ValueError("`action_cfg_scale` != 1.0 requires non-null `action` input.")

        z = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents[:, :, 0:1] = z
        first_frame_latents = z
        fuse_flag = True

        context_posi, context_posi_mask = self.encode_prompt(prompt)
        context_nega = None
        context_nega_mask = None
        if text_cfg_scale != 1.0:
            context_nega, context_nega_mask = self.encode_prompt("" if negative_prompt is None else negative_prompt)
        action_nega = torch.zeros_like(action) if (action is not None and action_cfg_scale != 1.0) else None

        infer_timesteps, infer_deltas = self.infer_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(infer_timesteps, infer_deltas):
            timestep = step_t.unsqueeze(0).to(dtype=latents.dtype, device=self.device)
            noise_pred_posi = self._model_fn(
                latents=latents,
                timestep=timestep,
                context=context_posi,
                context_mask=context_posi_mask,
                action=action,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            noise_pred = noise_pred_posi
            if context_nega is not None:
                noise_pred_text_nega = self._model_fn(
                    latents=latents,
                    timestep=timestep,
                    context=context_nega,
                    context_mask=context_nega_mask,
                    action=action,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                noise_pred = noise_pred + (text_cfg_scale - 1.0) * (noise_pred_posi - noise_pred_text_nega)
            if action_nega is not None:
                noise_pred_action_nega = self._model_fn(
                    latents=latents,
                    timestep=timestep,
                    context=context_posi,
                    context_mask=context_posi_mask,
                    action=action_nega,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                noise_pred = noise_pred + (action_cfg_scale - 1.0) * (noise_pred_posi - noise_pred_action_nega)
            latents = self.infer_scheduler.step(noise_pred, step_delta, latents)
            latents[:, :, 0:1] = first_frame_latents

        return {"video": self._decode_latents(latents, tiled=tiled)}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "dit": self.dit.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        self.dit.load_state_dict(payload["dit"], strict=False)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
