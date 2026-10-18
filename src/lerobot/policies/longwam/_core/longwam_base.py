# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from FastWAM's model base, then split/renamed for Long-WAM.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/models/wan22/fastwam.py
# Changes: Long-WAM naming, streaming clean-observation memory, training and checkpoint integration.
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
#
# NVIDIA modifications are licensed under the Apache License, Version 2.0 (the "License");
# upstream FastWAM portions retain MIT terms in licenses/FastWAM-MIT.txt.
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
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/models/wan22/longwam_base.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

from collections.abc import Sequence
from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from lerobot.policies.longwam._core.inference_helpers import InferenceModelMixin

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = getLogger(__name__)


class LongWAMBase(InferenceModelMixin, torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: int | None = None,
        proprio_dim: int | None = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        # v0b streaming clean-observation memory (settable instance attrs; defaults reproduce v0a).
        #   num_clean_frames: number of leading latent frames that are clean (past+current obs).
        #   current_obs_idx: index of the CURRENT observation in the proprio time dimension
        #     (raw frame resolution, length num_frames-1; see build_inputs proprio convention).
        self.num_clean_frames = 1
        self.current_obs_idx = 0

        # P4 joint imagination (settable by LongWAM; defaults reproduce v0a/v0b exactly):
        #   num_imagine_frames (k): leading FUTURE latent frames the action additionally attends as
        #     imagination. k=0 -> action attends clean frames ONLY (byte-identical to v0a/v0b).
        #   imagine_timestep: the HIGH/fixed noise level (in scheduler timestep units, range
        #     [0, video_num_train_timesteps]) at which the action reads the imagined future. DECOUPLED
        #     from the full-range video-loss timestep so the action never sees a near-GT future.
        #   imagine_infer_steps: # partial-denoise steps used at inference to roll the future latent to
        #     `imagine_timestep` (train==infer high-σ target).
        self.num_imagine_frames = 0
        self.imagine_timestep = float(video_num_train_timesteps) * 0.9
        self.imagine_infer_steps = 4
        self.joint_denoise = False
        self._inference_optimization_config = None
        self._inference_optimization_prepared = False
        self._inference_optimization_state = None

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: int | None = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        skip_video_dit_load_from_pretrain: bool | None = None,
        skip_action_dit_load_from_pretrain: bool | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for LongWAMBase.from_wan22_pretrained().")
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for LongWAMBase.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for LongWAMBase.")

        legacy_skip = bool(skip_dit_load_from_pretrain)
        skip_video_pretrain = (
            legacy_skip
            if skip_video_dit_load_from_pretrain is None
            else bool(skip_video_dit_load_from_pretrain)
        )
        skip_action_pretrain = (
            legacy_skip
            if skip_action_dit_load_from_pretrain is None
            else bool(skip_action_dit_load_from_pretrain)
        )

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_video_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_action_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_action_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        if self._inference_optimization_prepared:
            state = self._inference_optimization_state
            if state is not None and state.model_quant:
                raise RuntimeError(
                    "Do not call model.to() after FourOverSix materialization; "
                    "move the BF16 model before prepare_inference_optimizations()."
                )
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
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

    @torch.no_grad()
    def encode_prompt(self, prompt: str | Sequence[str]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)  # [B, 1, D]
        if context_mask.dtype == torch.bool:
            proprio_mask = torch.ones(
                (context_mask.shape[0], 1),
                dtype=torch.bool,
                device=context_mask.device,
            )
        else:
            # A floating-point context mask is an additive attention bias. The
            # visible proprio token therefore needs a neutral zero bias, not 1.
            proprio_mask = torch.zeros(
                (context_mask.shape[0], 1),
                dtype=context_mask.dtype,
                device=context_mask.device,
            )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if video_tensor.ndim == 6:
            self._validate_three_view_rgb(
                video_tensor,
                name="video_tensor",
                expected_ndim=6,
            )
            batch_size, num_views, channels, num_frames, height, width = video_tensor.shape
            flat_video = video_tensor.reshape(
                batch_size * num_views,
                channels,
                num_frames,
                height,
                width,
            )
            encoded = self.vae.encode(
                flat_video,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if not isinstance(encoded, torch.Tensor) or encoded.ndim != 5:
                raise ValueError(
                    "Per-view VAE encoding must return a 5D tensor "
                    f"[B*V,C,T,H,W], got {type(encoded).__name__} "
                    f"with shape {getattr(encoded, 'shape', None)}."
                )
            if encoded.shape[0] != batch_size * num_views:
                raise ValueError(
                    f"Per-view VAE batch mismatch: got {encoded.shape[0]}, expected {batch_size * num_views}."
                )
            encoded = encoded.reshape(batch_size, num_views, *encoded.shape[1:])
            return torch.cat(encoded.unbind(dim=1), dim=-1)

        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        if input_image.ndim == 5:
            self._validate_three_view_rgb(
                input_image,
                name="input_image",
                expected_ndim=5,
            )
            image_video = input_image.to(device=self.device).unsqueeze(3)
            return self._encode_video_latents(
                image_video,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    @staticmethod
    def _validate_three_view_rgb(
        tensor: torch.Tensor,
        *,
        name: str,
        expected_ndim: int,
    ) -> None:
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"`{name}` must be {expected_ndim}D for three-view RGB input, "
                f"got shape {tuple(tensor.shape)}."
            )
        view_dim = 1
        channel_dim = 2
        if tensor.shape[view_dim] != 3 or tensor.shape[channel_dim] != 3:
            raise ValueError(f"`{name}` must have V=3 and C=3, got shape {tuple(tensor.shape)}.")
        if tuple(tensor.shape[-2:]) != (256, 256):
            raise ValueError(
                f"`{name}` three-view frames must be exactly 256x256, got HxW={tuple(tensor.shape[-2:])}."
            )

    @classmethod
    def _prepare_inference_image(
        cls,
        input_image: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int, bool]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)

        is_three_view = input_image.ndim == 5
        if is_three_view:
            cls._validate_three_view_rgb(
                input_image,
                name="input_image",
                expected_ndim=5,
            )
            if input_image.shape[0] != 1:
                raise ValueError(
                    f"Inference requires `input_image` batch size 1, got shape {tuple(input_image.shape)}."
                )
        elif input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                "`input_image` must have shape [1,3,H,W], [3,H,W], or "
                f"[1,3,3,256,256], got {tuple(input_image.shape)}."
            )

        height, width = map(int, input_image.shape[-2:])
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "`input_image` must be resized before infer, expected multiples of 16 "
                f"but got HxW=({height},{width})"
            )
        return input_image, height, width, is_three_view

    @classmethod
    def _prepare_inference_obs_window(
        cls,
        obs_window: torch.Tensor,
        *,
        height: int,
        width: int,
        expect_three_view: bool,
    ) -> torch.Tensor:
        is_three_view = obs_window.ndim == 6
        if is_three_view:
            cls._validate_three_view_rgb(
                obs_window,
                name="obs_window",
                expected_ndim=6,
            )
            if obs_window.shape[0] != 1:
                raise ValueError(
                    f"Inference requires `obs_window` batch size 1, got shape {tuple(obs_window.shape)}."
                )
        elif obs_window.ndim != 5 or obs_window.shape[0] != 1 or obs_window.shape[1] != 3:
            raise ValueError(
                "`obs_window` must have shape [1,3,T_sampled,H,W] or "
                f"[1,3,3,T_sampled,256,256], got {tuple(obs_window.shape)}."
            )

        if is_three_view != expect_three_view:
            raise ValueError(
                "`input_image` and `obs_window` must use the same view representation "
                "(both legacy mosaic or both three-view tensors)."
            )
        window_height, window_width = map(int, obs_window.shape[-2:])
        if window_height != height or window_width != width:
            raise ValueError(
                "`obs_window` spatial dims must match `input_image`: "
                f"obs_window HxW=({window_height},{window_width}) vs "
                f"input_image HxW=({height},{width})."
            )
        return obs_window

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(
            latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "LongWAMBase training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim == 6:
            self._validate_three_view_rgb(
                video,
                name="sample['video']",
                expected_ndim=6,
            )
            batch_size, _, _, num_frames, height, width = video.shape
        elif video.ndim == 5:
            if video.shape[1] != 3:
                raise ValueError(
                    f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
                )
            batch_size, _, num_frames, height, width = video.shape
        else:
            raise ValueError(
                "`sample['video']` must be 5D [B,3,T,H,W] or 6D "
                f"[B,3,3,T,256,256], got shape {tuple(video.shape)}"
            )

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Video spatial dims must be multiples of 16, got H={height}, W={width}")
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for LongWAMBase training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        # P4: the action chunk is DECOUPLED from num_frames (independent `action_chunk` dataset param).
        # The per-video-frame action grouping only exists in the video DiT `action_conditioned=true` path.
        # LongWAM uses action_conditioned=false (action<->video coupling is purely the MoT attention mask), so
        # the action horizon need NOT divide the video transitions -- enforce the original divisibility ONLY
        # for an action_conditioned video expert. v0a/v0b are action_conditioned=false too, and their coupled
        # chunk (action_horizon == freq*(num_frames-1)) divides anyway, so skipping it here changes nothing.
        if getattr(self.video_expert, "action_conditioned", False) and action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions "
                f"({num_frames - 1}) for an action_conditioned video expert, got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            # v0b: the first `num_clean_frames` latent frames are clean (past->current obs memory).
            # num_clean_frames=1 reproduces v0a (clean frame-0 only).
            first_frame_latents = input_latents[:, :, 0 : self.num_clean_frames]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            # Proprio time-dim convention: proprio arrives at RAW frame resolution (length
            # num_frames-1, NOT subsampled by video_sample_indices -- see RobotVideoDataset._get
            # `proprio = sample["proprio"][:-1, :]`). With past_obs_size=P the obs window is
            # [t=-P .. t=+(num_frames-1-P)] at raw frame deltas, so the CURRENT obs (t=0) sits at
            # index P. The conditioning proprio must be the current obs (not the oldest past frame),
            # so we index `current_obs_idx` (= past_obs_size, set by LongWAM). For v0a (past_obs_size=0)
            # current_obs_idx=0 -> identical to the previous `proprio[:, 0, :]`.
            proprio = proprio[:, self.current_obs_idx, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        num_imagine_frames: int = 0,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # Start from the temporal video mask. In sequential P4, future frames attend clean history and
        # earlier future frames, while all video->action entries remain False. Joint mode below replaces
        # only the future-video rows, leaving every history row unchanged and cacheable.
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> video conditioning columns.
        #   k=0 (v0a/v0b): action attends the M=num_clean_frames clean past->current obs frames ONLY
        #     (v0a: M=1 -> first-frame). No future leak; train/infer conditioning identical. This branch
        #     is BYTE-IDENTICAL to the previous mask (num_imagine_frames defaults to 0).
        #   k>0 (P4 joint imagination): action additionally attends the first k FUTURE (imagined) latent
        #     frames. Those frames are noised to a HIGH fixed sigma in training_loss (never sigma=0/GT),
        #     so opening these columns does NOT leak the clean GT future.
        k = max(0, int(num_imagine_frames))
        cond_frames = int(self.num_clean_frames) + k
        expected_cond_tokens = cond_frames * video_tokens_per_frame
        joint_denoise = bool(getattr(self, "joint_denoise", False))

        if joint_denoise and k > 0 and expected_cond_tokens != video_seq_len:
            raise ValueError(
                "Joint denoising requires exactly M+k video frames: "
                f"expected video tokens={expected_cond_tokens}, video tokens={video_seq_len}."
            )

        cond_tokens = min(expected_cond_tokens, video_seq_len)
        mask[video_seq_len:, :cond_tokens] = True

        if joint_denoise and k > 0:
            history_tokens = int(self.num_clean_frames) * video_tokens_per_frame
            if not 0 < history_tokens < video_seq_len:
                raise ValueError(
                    "Joint denoising requires non-empty history and future blocks, got "
                    f"history tokens={history_tokens}, video tokens={video_seq_len}."
                )
            # [future video | action] is one bidirectional generation block. History rows
            # retain the video expert's causal mask and cannot read either generated group.
            mask[history_tokens:video_seq_len, :] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: torch.Tensor | None,
        include_initial_video_step: bool,
        num_clean_frames: int = 1,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(
            dim=(1, 3, 4)
        )
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        # v0b: the clean region spans the first `num_clean_frames` latent frames. In SAMPLED video
        # frames that is `(num_clean_frames-1)*temporal_factor + 1` frames (the Wan VAE maps the
        # first sampled frame -> latent 0, then each subsequent group of `temporal_factor` sampled
        # frames -> one latent frame). The video loss is computed only on the noised tail (the
        # frames AFTER the clean region), so we drop that many leading sampled frames from the pad
        # mask. num_clean_frames=1 -> head_sampled_frames=1 -> identical to v0a (`image_is_pad[:, 1:]`).
        head_sampled_frames = (int(num_clean_frames) - 1) * temporal_factor + 1
        if (image_is_pad.shape[1] - head_sampled_frames) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}, "
                f"num_clean_frames={num_clean_frames} (head_sampled_frames={head_sampled_frames})."
            )

        tail_is_pad = image_is_pad[:, head_sampled_frames:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _action_flow_matching_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: torch.Tensor | None,
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        """Padding-masked, schedule-weighted flow-matching loss on the action chunk (shared by the
        k=0 single-pass path and the k>0 imagination pass; identical reduction in both)."""
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(
            dim=2
        )  # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        return (action_loss_per_sample * action_weight).mean()

    def _mot_forward(
        self,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_pre: dict,
        action_pre: dict,
        num_imagine_frames: int,
    ) -> dict:
        """Single MoT forward with an explicit `num_imagine_frames` for the action->video mask.
        Used for the P4 imagination action pass (k>0); pass `num_imagine_frames=0` to reproduce the
        v0b conditioning (action attends clean frames only)."""
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            num_imagine_frames=num_imagine_frames,
        )
        return self.mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            # v0b: overwrite the first `num_clean_frames` latent frames with the clean obs memory.
            latents[:, :, 0 : self.num_clean_frames] = inputs["first_frame_latents"]

        k = int(self.num_imagine_frames)
        joint_denoise = bool(self.joint_denoise)
        if joint_denoise:
            if inputs["first_frame_latents"] is None or k <= 0:
                raise ValueError("Joint denoising requires clean history and k>0 future frames.")
            num_latent_frames = int(input_latents.shape[2])
            if int(self.num_clean_frames) + k != num_latent_frames:
                raise ValueError(
                    "Joint denoising requires exactly M+k latent frames: "
                    f"M={self.num_clean_frames}, k={k}, video latents={num_latent_frames}."
                )

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None if joint_denoise else action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            num_clean_frames=self.num_clean_frames,
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        # Joint mode predicts independently noised video and action in one MoT pass. Sequential P4
        # keeps the original detached-imagination action pass below.
        tokens_out = self._mot_forward(
            video_tokens=video_tokens,
            action_tokens=action_tokens,
            video_pre=video_pre,
            action_pre=action_pre,
            num_imagine_frames=k if joint_denoise else 0,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        # PASS B — P4 joint imagination (k>0 only). The action reads the first k FUTURE latent frames at a
        # HIGH fixed sigma (genuine imagination, never sigma=0/GT). We build a SECOND noised version of the
        # future at `self.imagine_timestep`, DECOUPLED from PASS A's full-range video timestep, then encode
        # the video branch's per-layer K/V UNDER torch.no_grad() (full STOP-GRADIENT: the video expert
        # receives ZERO gradient from loss_action, so the action cannot drag the future prediction toward
        # GT to cheat). The action branch then attends those detached video K/V WITH gradient, exactly as
        # at inference. PASS A still owns loss_video on the full-range noised future, exactly as v0b. k=0
        # skips this block entirely (-> v0b single-pass path).
        if not joint_denoise and k > 0 and inputs["first_frame_latents"] is not None:
            num_latent_frames = int(input_latents.shape[2])
            if self.num_clean_frames + k > num_latent_frames:
                raise ValueError(
                    f"P4 imagination needs num_clean_frames(M={self.num_clean_frames})+k({k}) "
                    f"<= num_latent_frames({num_latent_frames}); increase num_frames or reduce k."
                )
            timestep_imagine = torch.full(
                (batch_size,),
                float(self.imagine_timestep),
                device=self.device,
                dtype=input_latents.dtype,
            )
            # STOP-GRADIENT: build the imagined-future video K/V with the video branch fully detached
            # from autograd. The video expert gets no gradient from the action pass.
            with torch.no_grad():
                # Independent high-sigma noise for the imagined future (fresh draw; do NOT reuse PASS A's
                # noise so the two future views are decoupled).
                noise_video_hi = torch.randn_like(input_latents)
                latents_hi = self.train_video_scheduler.add_noise(
                    input_latents, noise_video_hi, timestep_imagine
                )
                # Clean obs frames stay clean (sigma=0); only [M : M+k...] future frames are high-sigma imagined.
                latents_hi[:, :, 0 : self.num_clean_frames] = inputs["first_frame_latents"]
                video_pre_hi = self.video_expert.pre_dit(
                    x=latents_hi,
                    timestep=timestep_imagine,
                    context=context,
                    context_mask=context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                    num_clean_frames=self.num_clean_frames,
                )
                video_seq_len_hi = int(video_pre_hi["tokens"].shape[1])
                attn_mask_hi = self._build_mot_attention_mask(
                    video_seq_len=video_seq_len_hi,
                    action_seq_len=int(action_tokens.shape[1]),
                    video_tokens_per_frame=int(video_pre_hi["meta"]["tokens_per_frame"]),
                    device=video_pre_hi["tokens"].device,
                    num_imagine_frames=k,
                )
                video_kv_cache_hi = self.mot.prefill_video_cache(
                    video_tokens=video_pre_hi["tokens"],
                    video_freqs=video_pre_hi["freqs"],
                    video_t_mod=video_pre_hi["t_mod"],
                    video_context_payload={
                        "context": video_pre_hi["context"],
                        "mask": video_pre_hi["context_mask"],
                    },
                    video_attention_mask=attn_mask_hi[:video_seq_len_hi, :video_seq_len_hi],
                )
                # Detach defensively (prefill ran under no_grad, but make the constant explicit).
                video_kv_cache_hi = [
                    {"k": layer["k"].detach(), "v": layer["v"].detach()} for layer in video_kv_cache_hi
                ]
            # Action pass WITH gradient: attend the detached imagined-future video K/V + clean K/V.
            action_pre_hi = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )
            action_tokens_hi = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre_hi["tokens"],
                action_freqs=action_pre_hi["freqs"],
                action_t_mod=action_pre_hi["t_mod"],
                action_context_payload={
                    "context": action_pre_hi["context"],
                    "mask": action_pre_hi["context_mask"],
                },
                video_kv_cache=video_kv_cache_hi,
                attention_mask=attn_mask_hi,
                video_seq_len=video_seq_len_hi,
            )
            # Override the action prediction with the imagination-conditioned one (PASS A's action pred is
            # discarded; loss_video is still computed from PASS A's pred_video below).
            pred_action = self.action_expert.post_dit(action_tokens_hi, action_pre_hi)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            # v0b: drop the first `num_clean_frames` clean latent frames from the video loss (no loss
            # on the given clean obs); keep loss only on the noised future frames. M=1 -> v0a `[:, :, 1:]`.
            pred_video = pred_video[:, :, self.num_clean_frames :]
            target_video = target_video[:, :, self.num_clean_frames :]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
            num_clean_frames=self.num_clean_frames,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        loss_action = self._action_flow_matching_loss(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: torch.Tensor | None = None,
        num_clean_frames: int = 1,
        num_imagine_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_clean_frames=num_clean_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_imagine_frames=num_imagine_frames,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(
            timestep_action, dtype=first_frame_latents.dtype, device=self.device
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        use_precomputed_action_cross_kv: bool = False,
        action_cross_kv_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if bool(getattr(self, "_action_video_kv_resident_cache_enabled", False)):
            video_kv_cache = self.mot.action_video_kv_cache()
        if use_precomputed_action_cross_kv and action_cross_kv_cache is None:
            action_cross_kv_cache = self.mot.action_cross_kv_cache()
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_cross_kv_cache=(action_cross_kv_cache if use_precomputed_action_cross_kv else None),
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: torch.Tensor
        | None = None,  # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}")
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
            strict=False,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action and not torch.allclose(
            action_out, action_only_out, atol=1e-2, rtol=1e-2
        ):
            max_abs_diff = (action_out - action_only_out).abs().max().item()
            logger.warning(
                f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
            )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        obs_window: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Predict an action chunk conditioned on the clean past->current observation memory.

        v0a (num_clean_frames=1, obs_window=None): conditions on a single current-frame latent
        (encoded from `input_image`); identical to the original LongWAMBase behavior.

        v0b (num_clean_frames=M>1): conditions on the first M clean latent frames encoded from
        `obs_window` -- the already-sampled past->current observation window of shape
        [1, 3, T_sampled, H, W] with T_sampled = (M-1)*temporal_factor + 1. The Wan VAE is causal
        in time, so its first M latents depend ONLY on these past+current frames (not on any
        future frames). Encoding via the SAME `self._encode_video_latents(...)` call training uses
        therefore reproduces training's clean latents bit-for-bit -> no train/infer mismatch and no
        future-obs leak (the action attends clean frames only).
        """
        self.eval()
        # v0b uses `per_frame_causal` (clean frames are block/frame-causal); v0a uses
        # `first_frame_causal`. Both are valid here -- the action attends ONLY the leading
        # num_clean_frames frames via `_build_mot_attention_mask`, regardless of the video
        # self-attention mode.
        video_attn_mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if video_attn_mode not in ("first_frame_causal", "per_frame_causal"):
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode` in "
                f"('first_frame_causal', 'per_frame_causal'), got '{video_attn_mode}'."
            )

        input_image, height, width, input_is_three_view = self._prepare_inference_image(input_image)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if obs_window is None:
            # v0a path: single current-frame latent. num_clean_frames must be 1 here, otherwise
            # the action mask would request M>1 clean frames that the cache does not contain.
            if int(self.num_clean_frames) != 1:
                raise ValueError(
                    "`infer_action` was called without `obs_window` but `num_clean_frames`="
                    f"{self.num_clean_frames} (>1). v0b inference must pass the past->current "
                    "`obs_window` so the model can reconstruct the M clean latent frames."
                )
            clean_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        else:
            # v0b path: encode the past->current observation window through the SAME call training
            # uses (`_encode_video_latents`). The Wan VAE is causal in time, so its first M latents
            # equal training's clean latents regardless of any future frames.
            obs_window = self._prepare_inference_obs_window(
                obs_window,
                height=height,
                width=width,
                expect_three_view=input_is_three_view,
            )
            obs_window = obs_window.to(device=self.device, dtype=self.torch_dtype)
            clean_latents = self._encode_video_latents(obs_window, tiled=tiled)
            if int(clean_latents.shape[2]) != int(self.num_clean_frames):
                raise ValueError(
                    "`obs_window` VAE-encoded to "
                    f"{int(clean_latents.shape[2])} latent frames, but `num_clean_frames`="
                    f"{self.num_clean_frames}. Expected T_sampled=(M-1)*temporal_factor+1="
                    f"{(int(self.num_clean_frames) - 1) * int(self.vae.temporal_downsample_factor) + 1} "
                    f"sampled frames, got T_sampled={int(obs_window.shape[-3])}."
                )
        first_frame_latents = clean_latents
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # All M clean frames carry t=0 (they are the given clean obs memory). `pre_dit` forces the
        # first `num_clean_frames` latent-frame timesteps to 0 internally; passing a per-batch
        # zero scalar keeps the rest of the (here non-existent) noised tail consistent.
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            num_clean_frames=self.num_clean_frames,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(
            infer_timesteps_action, infer_deltas_action, strict=False
        ):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def _predict_video_noise(
        self,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        num_clean_frames: int,
        use_precomputed_video_cross_kv: bool = False,
    ) -> torch.Tensor:
        """Video-only velocity prediction over `[clean | future]` under per_frame_causal. The leading
        `num_clean_frames` latent frames are forced to t=0 inside `pre_dit`; the rest carry
        `timestep_video`. Runs the video expert's OWN block stack (not the MoT) -- no action is involved
        in the future rollout, and the per_frame_causal self-attention mask governs the future
        prediction. Used by the P4 future-imagination rollout (latent-only, no decode)."""
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_clean_frames=num_clean_frames,
            context_is_projected=use_precomputed_video_cross_kv,
        )
        x_tokens = video_pre["tokens"]
        self_attn_mask = (
            self.video_expert.build_video_to_video_mask(
                video_seq_len=x_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=x_tokens.device,
            )
            if self.video_expert.video_attention_mask_mode != "bidirectional"
            else None
        )
        for block in self.video_expert.blocks:
            x_tokens = block(
                x_tokens,
                video_pre["context"],
                video_pre["t_mod"],
                video_pre["freqs"],
                context_mask=video_pre["context_mask"],
                self_attn_mask=self_attn_mask,
                use_cached_cross_kv=use_precomputed_video_cross_kv,
            )
        return self.video_expert.post_dit(x_tokens, video_pre)

    @torch.no_grad()
    def _imagine_future_latents(
        self,
        clean_latents: torch.Tensor,
        num_imagine_frames: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        generator: torch.Generator | None,
        rand_device: str,
        sigma_shift: float | None = None,
        use_precomputed_video_cross_kv: bool = False,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Roll out `num_imagine_frames` FUTURE latent frames by PARTIAL denoise, stopping at the same
        high sigma (`self.imagine_timestep`) the action trained on. Latent-only (NO VAE decode).

        train==infer: training shows the action a future at sigma=imagine_timestep; here we denoise a
        generated future from pure noise down to exactly that sigma (a few steps), then hand the action
        the partially-denoised future. Returns the imagined future latents [B, z, k, h, w] at the target
        sigma (NOT clean), to be concatenated after the clean frames for the action pass.
        """
        b, z, M, h, w = clean_latents.shape
        future = torch.randn(
            (b, z, int(num_imagine_frames), h, w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=clean_latents.dtype)

        use_clean_prefix_wavefront = bool(getattr(self, "_video_clean_prefix_wavefront_enabled", False))
        if use_clean_prefix_wavefront:
            clean_cache = self._prefill_video_clean_prefix(
                clean_latents=clean_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                use_precomputed_video_cross_kv=use_precomputed_video_cross_kv,
            )
            clean_sequence_length = int(clean_cache[0]["k"].shape[1])
            if clean_sequence_length % int(M) != 0:
                raise RuntimeError("clean-prefix token count is not divisible by its frame count")
            tokens_per_frame = clean_sequence_length // int(M)
            total_sequence_length = (int(M) + int(num_imagine_frames)) * tokens_per_frame
            self.mot.install_video_clean_prefix_cache(
                clean_cache,
                total_sequence_length=total_sequence_length,
            )

        ntt = float(self.train_video_scheduler.num_train_timesteps)
        sigma_target = float(self.imagine_timestep) / ntt
        shift = self.infer_video_scheduler.shift if sigma_shift is None else float(sigma_shift)
        steps = max(1, int(self.imagine_infer_steps if num_inference_steps is None else num_inference_steps))
        # Partial flow-matching schedule: sigma 1.0 -> sigma_target over `steps` (mirrors
        # build_inference_schedule's phi-shift sampling, truncated at sigma_target instead of 0).
        u_steps = torch.linspace(1.0, 0.0, steps + 1, device=self.device, dtype=torch.float32)
        sigma_full = self.infer_video_scheduler._phi(u_steps, shift)  # 1.0 -> 0.0
        # Rescale so the final point lands exactly on sigma_target (keep the first point at sigma=1).
        sigma_steps = sigma_target + (sigma_full - sigma_full[-1]) * (
            (1.0 - sigma_target) / (sigma_full[0] - sigma_full[-1] + 1e-12)
        )
        for i in range(steps):
            sigma_cur = sigma_steps[i]
            delta = (sigma_steps[i + 1] - sigma_steps[i]).to(dtype=future.dtype)
            timestep_video = (sigma_cur * ntt).reshape(1).to(dtype=clean_latents.dtype, device=self.device)
            latents_video = torch.cat([clean_latents, future], dim=2)
            if use_clean_prefix_wavefront:
                pred_future = self._predict_video_noise_with_clean_prefix(
                    latents_video=latents_video,
                    timestep_video=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    num_clean_frames=int(M),
                    use_precomputed_video_cross_kv=use_precomputed_video_cross_kv,
                )
            else:
                pred_video = self._predict_video_noise(
                    latents_video=latents_video,
                    timestep_video=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                    num_clean_frames=int(M),
                    use_precomputed_video_cross_kv=use_precomputed_video_cross_kv,
                )
                pred_future = pred_video[:, :, int(M) :]
            # Step ONLY the future frames; clean frames are fixed observations.
            future = self.infer_video_scheduler.step(pred_future, delta, future)
        return future

    def _resolve_joint_ar_inference_steps(
        self,
        *,
        action_num_inference_steps: int,
        video_num_inference_steps: int | None,
    ) -> tuple[int, int]:
        """Resolve separate sequential steps and enforce one shared joint schedule."""
        action_steps = int(action_num_inference_steps)
        if action_steps <= 0:
            raise ValueError(f"Action denoising steps must be positive, got {action_steps}.")

        if bool(self.joint_denoise):
            if video_num_inference_steps is not None:
                raise ValueError(
                    "Joint video-action denoising has one shared step count; "
                    "do not set video_num_inference_steps separately."
                )
            return action_steps, action_steps

        video_steps = (
            int(self.imagine_infer_steps)
            if video_num_inference_steps is None
            else int(video_num_inference_steps)
        )
        if video_steps <= 0:
            raise ValueError(f"Video denoising steps must be positive, got {video_steps}.")
        return video_steps, action_steps

    @torch.no_grad()
    def infer_joint_ar(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        obs_window: torch.Tensor | None = None,
        video_num_inference_steps: int | None = None,
        clean_latents: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Generate action with M clean history latents and k future-video latents.

        - k=0 -> reduces EXACTLY to `infer_action` (action-only). Same args, same result.
        - sequential P4 -> partially denoise video, cache it, then denoise action.
        - joint mode -> advance video and action schedules together with bidirectional attention inside
          [future video | action]. Neither mode decodes the imagined video through the VAE.
        """
        self.eval()
        k = int(self.num_imagine_frames)
        if k <= 0:
            if video_num_inference_steps is not None:
                raise ValueError(
                    "video_num_inference_steps is only valid when k>0 future video is generated."
                )
            # Action-only: byte-identical to today's infer_action.
            return self.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                obs_window=obs_window,
            )

        video_steps, action_steps = self._resolve_joint_ar_inference_steps(
            action_num_inference_steps=num_inference_steps,
            video_num_inference_steps=video_num_inference_steps,
        )

        video_attn_mode = str(getattr(self.video_expert, "video_attention_mask_mode", ""))
        if video_attn_mode != "per_frame_causal":
            raise ValueError(
                f"`infer_joint_ar` (k>0) requires `video_attention_mask_mode='per_frame_causal'`, got '{video_attn_mode}'."
            )
        input_image, height, width, input_is_three_view = self._prepare_inference_image(input_image)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        future_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        if clean_latents is not None:
            if k <= 0 or tiled:
                raise ValueError("Preencoded latents require untiled video-action inference")
            if (
                clean_latents.ndim != 5
                or clean_latents.shape[0] != 1
                or clean_latents.shape[2] != self.num_clean_frames
            ):
                raise ValueError("Preencoded latents must have shape [1, Z, M, h, w]")
            clean_latents = clean_latents.to(device=self.device, dtype=self.torch_dtype)
        elif obs_window is None:
            if int(self.num_clean_frames) != 1:
                raise ValueError(
                    "`infer_joint_ar` (k>0) requires the past->current `obs_window` when num_clean_frames>1."
                )
            clean_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        else:
            obs_window = self._prepare_inference_obs_window(
                obs_window,
                height=height,
                width=width,
                expect_three_view=input_is_three_view,
            )
            obs_window = obs_window.to(device=self.device, dtype=self.torch_dtype)
            clean_latents = self._encode_video_latents(obs_window, tiled=tiled)
            if int(clean_latents.shape[2]) != int(self.num_clean_frames):
                raise ValueError(
                    f"`obs_window` VAE-encoded to {int(clean_latents.shape[2])} latent frames, but "
                    f"`num_clean_frames`={self.num_clean_frames}."
                )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        if bool(self.joint_denoise):
            if float(self.imagine_timestep) != 0.0:
                raise ValueError("Joint video-action denoising must run the full schedule to sigma=0.")

            b, z, num_history, latent_height, latent_width = clean_latents.shape
            future_latents = torch.randn(
                (b, z, k, latent_height, latent_width),
                generator=future_generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=self.device, dtype=clean_latents.dtype)
            video_timesteps, video_deltas = self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=video_steps,
                device=self.device,
                dtype=future_latents.dtype,
                shift_override=sigma_shift,
            )
            action_timesteps, action_deltas = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=action_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
            for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
                video_timesteps, video_deltas, action_timesteps, action_deltas, strict=False
            ):
                timestep_video = step_t_video.reshape(1).to(device=self.device, dtype=future_latents.dtype)
                timestep_action = step_t_action.reshape(1).to(device=self.device, dtype=latents_action.dtype)
                latents_video = torch.cat([clean_latents, future_latents], dim=2)
                pred_video, pred_action = self._predict_joint_noise(
                    latents_video=latents_video,
                    latents_action=latents_action,
                    timestep_video=timestep_video,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    num_clean_frames=int(num_history),
                    num_imagine_frames=k,
                )
                future_latents = self.infer_video_scheduler.step(
                    pred_video[:, :, int(num_history) :], step_delta_video, future_latents
                )
                latents_action = self.infer_action_scheduler.step(
                    pred_action, step_delta_action, latents_action
                )
            return {
                "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            }

        use_video_cache = bool(getattr(self, "_video_cross_kv_cache_enabled", False))
        video_context = context
        if use_video_cache:
            video_context = self.mot.install_video_cross_kv_cache(
                self.mot.prefill_video_cross_kv_cache(context)
            )

        # 1) Imagine k future latents (partial denoise to the training high-sigma; latent-only, NO decode).
        future_latents = self._imagine_future_latents(
            clean_latents=clean_latents,
            num_imagine_frames=k,
            context=video_context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            generator=future_generator,
            rand_device=rand_device,
            sigma_shift=sigma_shift,
            num_inference_steps=video_steps,
            use_precomputed_video_cross_kv=use_video_cache,
        )

        # 2) Build the joint video sequence [clean (sigma=0) | imagined future (sigma=imagine_timestep)].
        #    pre_dit forces the leading M clean frames to t=0; the trailing k future frames carry the
        #    high imagine timestep, EXACTLY matching the action's training-time future view.
        latents_video = torch.cat([clean_latents, future_latents], dim=2)
        timestep_video = torch.full(
            (latents_video.shape[0],),
            float(self.imagine_timestep),
            dtype=latents_video.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=video_context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            num_clean_frames=self.num_clean_frames,
            context_is_projected=use_video_cache,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_imagine_frames=k,
        )
        video_prefill = (
            self.mot.prefill_video_cache_with_clean_prefix
            if getattr(self, "_video_clean_prefix_wavefront_enabled", False)
            else self.mot.prefill_video_cache
        )
        video_kv_cache = video_prefill(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        video_kv_cache = self._stage_action_video_kv_cache(video_kv_cache)
        use_action_cache = bool(getattr(self, "_action_cross_kv_cache_enabled", False))
        if use_action_cache:
            self.mot.install_action_cross_kv_cache(self.mot.prefill_action_cross_kv_cache(context))

        # 3) Denoise the action attending [M clean + k imagined future] via the cached video KV.
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=action_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(
            infer_timesteps_action, infer_deltas_action, strict=False
        ):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
            pred_action = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                use_precomputed_action_cross_kv=use_action_cache,
            )
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        num_frames: int,
        action: torch.Tensor | None = None,
        action_horizon: int | None = None,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        negative_prompt: str | None = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None, strict=False):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "mot" in payload:
            result = self.mot.load_state_dict(payload["mot"], strict=strict)
            if strict:
                logger.info("Strictly loaded joint MoT checkpoint (video + action): %s", path)
            else:
                missing = list(getattr(result, "missing_keys", []))
                unexpected = list(getattr(result, "unexpected_keys", []))
                logger.warning(
                    "Non-strict MoT load from %s: loaded=%d missing=%d unexpected=%d; "
                    "missing keys keep their current initialization: %s%s",
                    path,
                    len(payload["mot"]) - len(unexpected),
                    len(missing),
                    len(unexpected),
                    missing[:16],
                    " ..." if len(missing) > 16 else "",
                )
                if unexpected:
                    logger.warning(
                        "Unexpected MoT keys ignored: %s%s",
                        unexpected[:16],
                        " ..." if len(unexpected) > 16 else "",
                    )
        elif strict:
            raise ValueError(
                "Strict joint checkpoint loading requires the `mot` key containing both "
                f"video and action experts: {path}"
            )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=strict)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
                if strict:
                    logger.info("Strictly loaded proprio encoder checkpoint: %s", path)
            elif strict:
                raise ValueError(
                    f"Strict joint checkpoint loading requires `proprio_encoder` weights: {path}"
                )
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
