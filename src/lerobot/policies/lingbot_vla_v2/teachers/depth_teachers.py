# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""Frozen native-depth / DINO-video teacher target extraction.

This package intentionally contains no upstream source tree and never resolves
or imports an upstream checkout. Teacher implementations must be first-party
weight-compatible modules, with weights supplied through ``align_params``.

``DepthTeacherBundle`` deliberately is not an ``nn.Module``. It is held as a
plain policy attribute so frozen teacher weights never enter optimizers,
FSDP/DDP state dicts, or saved LeRobot checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 - conventional torch alias

# True: the first-party teacher runtimes below are wired in (depth + DINO-video),
# so real-weight integration tests can gate without probing loader internals.
FIRST_PARTY_TEACHERS_READY = True


def _require_file(path: str | Path | None, name: str) -> Path:
    if not path:
        raise ValueError(f"align_params requires {name}.")
    result = Path(path).expanduser()
    if not result.is_file():
        raise FileNotFoundError(f"{name} not found: {result}")
    return result


def _freeze(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    module.requires_grad_(False)
    module.to(device=device)
    module.eval()
    return module


def _load_depth_models(params: dict, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Build first-party, weight-compatible MoGe and MoRGBD teachers.

    Both runtimes are LeRobot-maintained modules (``native_depth_models.py`` /
    ``morgbd_teacher.py``) that restore the published checkpoint weights with a
    strict local ``load_state_dict``; no upstream code is imported.
    """
    from .morgbd_teacher import MoRGBDTeacher
    from .native_depth_models import load_moge_v2_teacher

    depth_cfg = params["depth"]
    moge = load_moge_v2_teacher(
        str(_require_file(depth_cfg.get("moge_path"), "align_params.depth.moge_path")), device=device
    )
    morgbd = MoRGBDTeacher.from_pretrained(
        str(_require_file(depth_cfg.get("morgbd_path"), "align_params.depth.morgbd_path")), device=device
    )
    return _freeze(moge, device), _freeze(morgbd, device)


def _load_video_teacher(params: dict, device: torch.device) -> torch.nn.Module:
    """Build the first-party, weight-compatible DINO-video teacher.

    ``teachers/dino_video`` reimplements the published video teacher locally
    (verified bit-exact against the upstream SDPA reference on real weights);
    only the two published weight files are needed, never a repository.
    """
    from .dino_video import build_dino_video_teacher

    video_cfg = params["video"]
    _require_file(video_cfg.get("ckpt_path"), "align_params.video.ckpt_path")
    # config.yaml defaults to the checkpoint's directory when omitted.
    config_path = video_cfg.get("config_path") or str(Path(video_cfg["ckpt_path"]).parent / "config.yaml")
    _require_file(config_path, "align_params.video.config_path")
    config = dict(video_cfg)
    config["device"] = str(device)
    return _freeze(build_dino_video_teacher(config), device)


@dataclass
class DepthTeacherBundle:
    """Plain container intentionally excluded from the policy module tree."""

    moge: torch.nn.Module
    morgbd: torch.nn.Module
    video: torch.nn.Module | None
    device: torch.device

    @classmethod
    def build(cls, params: dict, device: torch.device) -> DepthTeacherBundle:
        moge, morgbd = _load_depth_models(params, device)
        video = _load_video_teacher(params, device) if params.get("use_future_video", False) else None
        return cls(moge=moge, morgbd=morgbd, video=video, device=device)

    def depth_targets(self, pil_images: torch.Tensor, num_backbone_tokens: int = 256) -> torch.Tensor:
        """Port of upstream ``get_depth_target``; returns [B, num_backbone_tokens, 1024] bf16.

        Non-square inputs (e.g. 240x320) produce a non-square patch grid
        (14x18=252 instead of 16x16=256). Pad with zeros to num_backbone_tokens
        so the distillation head's fixed query count matches.
        """
        if pil_images.ndim != 5:
            raise ValueError(f"pil_images must have shape [B,N,C,H,W], got {tuple(pil_images.shape)}")
        images = pil_images[:, :1].reshape(-1, *pil_images.shape[2:]).contiguous().float() / 255.0
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output_moge = self.moge.infer(images, resolution_level=3, num_tokens=256, apply_mask=False)
            depth_pred = output_moge["depth"].squeeze().detach().clone()
            depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
            target, _cls_token = self.morgbd.infer_feat(
                images,
                depth_pred,
                depth_down_scale=1,
                resolution_level=3,
                num_tokens=256,
                enable_depth_mask=False,
            )
        batch_size = pil_images.shape[0]
        out = target.permute(0, 2, 3, 1).reshape(batch_size, -1, target.shape[1]).to(dtype=torch.bfloat16)
        if out.shape[1] < num_backbone_tokens:
            pad = torch.zeros(
                batch_size, num_backbone_tokens - out.shape[1], out.shape[2],
                dtype=out.dtype, device=out.device,
            )
            out = torch.cat([out, pad], dim=1)
        return out

    def video_targets(
        self,
        pil_images: torch.Tensor,
        future_pil_images: torch.Tensor,
        config: dict,
        effective_fps: float | torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor | None]:
        """Port of upstream ``get_video_target`` with its return-shape contract."""
        if self.video is None:
            raise RuntimeError("DINO-video teacher was not built.")
        use_patch = bool(config.get("use_patch_loss", True))
        use_cls = bool(config.get("use_cls_loss", False))
        if not use_patch and not use_cls:
            raise ValueError("future-video alignment requires use_patch_loss or use_cls_loss.")
        num_future = int(config.get("num_future_frames", 1))
        # Preserve upstream's historical interpretation verbatim: axis 1 is the
        # canonical-camera axis after FeatureTransform, even though it is sliced
        # as ``num_future_frames``. The released recipe uses one camera/frame; do
        # not reject multi-camera or num_future_frames>1 experiments here.
        input_size = int(config.get("input_size", 256))
        current = pil_images[:, :1].reshape(-1, *pil_images.shape[2:]).contiguous().float() / 255.0
        futures = (
            future_pil_images[:, :num_future].reshape(-1, *future_pil_images.shape[2:]).contiguous().float()
            / 255.0
        )
        if current.shape[-2:] != (input_size, input_size):
            current = F.interpolate(
                current, size=(input_size, input_size), mode="bilinear", align_corners=False
            )
        if futures.shape[-2:] != (input_size, input_size):
            futures = F.interpolate(
                futures, size=(input_size, input_size), mode="bilinear", align_corners=False
            )
        mean = torch.tensor([0.485, 0.456, 0.406], device=current.device, dtype=current.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=current.device, dtype=current.dtype).view(1, 3, 1, 1)
        current = (current - mean) / std
        futures = (futures - mean) / std
        futures = futures.reshape(pil_images.shape[0], num_future, *futures.shape[1:])
        frames = [current, *[futures[:, index] for index in range(num_future)]]
        if bool(config.get("use_warmup_frame", False)):
            frames.insert(0, current.clone())
            current_index = 1
        else:
            current_index = 0
        video = torch.stack(frames, dim=2).contiguous()
        kwargs = {"return_cls": use_cls}
        return_current = bool(config.get("use_current_patch_loss", False))
        if return_current:
            kwargs["return_current"] = True
        if current_index:
            kwargs["current_index"] = current_index
        if effective_fps is not None:
            kwargs["fps"] = (
                float(effective_fps.flatten()[0]) if torch.is_tensor(effective_fps) else effective_fps
            )
        # Upstream runs the DINO-video teacher under bf16 autocast (its weights are
        # bf16 while the normalized inputs here are float32).
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            result = self.video.get_future_feature(video, **kwargs)
        if use_cls:
            if not return_current:
                patch, cls = result
                return patch.detach().to(dtype=torch.bfloat16), cls.detach().to(dtype=torch.bfloat16)
            patch, cls, current_patch, _current_cls = result
            return {
                "patch": patch.detach().to(dtype=torch.bfloat16),
                "cls": cls.detach().to(dtype=torch.bfloat16),
                "current_patch": current_patch.detach().to(dtype=torch.bfloat16),
            }
        if not return_current:
            return result.detach().to(dtype=torch.bfloat16)
        patch, current_patch = result
        return {
            "patch": patch.detach().to(dtype=torch.bfloat16),
            "cls": None,
            "current_patch": current_patch.detach().to(dtype=torch.bfloat16),
        }
