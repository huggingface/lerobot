# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import shutil
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_fastwam import FastWAMConfig

if TYPE_CHECKING:
    from .wan_components import WanCheckpointPaths


class FastWAMPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for FastWAM.

    Args:
        config (FastWAMConfig): FastWAM policy configuration.
        dataset_stats (dict[str, dict[str, Tensor]] | None): Optional LeRobot
            dataset statistics passed by the training/evaluation stack.
    """

    config_class = FastWAMConfig
    name = "fastwam"

    def __init__(
        self,
        config: FastWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        **kwargs: Any,
    ):
        skip_wan_init = bool(kwargs.pop("_skip_wan_init", False))
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.dataset_stats = dataset_stats
        if skip_wan_init:
            self.model = _build_core_model_from_architecture(config)
        else:
            self.model = self._build_core_model(config)
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: FastWAMConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> FastWAMPolicy:
        """Load FastWAM weights and local Wan components from one HF directory.

        Args:
            pretrained_name_or_path (str | Path): HF-format policy directory
                containing `config.json`, `model.safetensors`, local Wan VAE,
                local UMT5 text encoder, and tokenizer files.
            config (FastWAMConfig | None): Optional config override. When
                omitted, `config.json` is read from `pretrained_name_or_path`.
            force_download (bool): Forwarded to LeRobot's pretrained loader.
            resume_download (bool | None): Forwarded to LeRobot's pretrained loader.
            proxies (dict | None): Forwarded to LeRobot's pretrained loader.
            token (str | bool | None): Forwarded to LeRobot's pretrained loader.
            cache_dir (str | Path | None): Forwarded to LeRobot's pretrained loader.
            local_files_only (bool): Forwarded to LeRobot's pretrained loader.
            revision (str | None): Forwarded to LeRobot's pretrained loader.
            strict (bool): Whether safetensors loading should require an exact
                match between checkpoint keys and policy module keys.
            **kwargs (Any): Extra constructor arguments forwarded to
                `FastWAMPolicy`.
        """

        pretrained_path = _resolve_pretrained_directory(
            pretrained_name_or_path=pretrained_name_or_path,
            force_download=force_download,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_path)
        kwargs["_skip_wan_init"] = True
        policy = super().from_pretrained(
            pretrained_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
        policy.load_wan_components_from_pretrained(pretrained_path)
        policy.eval()
        return policy

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        _copy_wan_components_from_policy(policy=self, save_directory=save_directory)

    def get_optim_params(self) -> dict[str, Any]:
        params = (
            list(self.model.dit.parameters()) if hasattr(self.model, "dit") else list(self.model.parameters())
        )
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(list(proprio_encoder.parameters()))
        return {"params": [p for p in params if p.requires_grad]}

    def load_wan_components_from_pretrained(self, pretrained_name_or_path: str | Path) -> None:
        """Attach local Wan VAE, text encoder, and tokenizer from a HF directory.

        Args:
            pretrained_name_or_path (str | Path): Directory containing
                `Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`,
                and `google/umt5-xxl/` tokenizer files.
        """

        paths = resolve_wan_component_paths(pretrained_name_or_path)
        _load_wan_components_into_policy(policy=self, paths=paths)

    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque([], maxlen=self.config.n_action_steps)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute FastWAM training loss for a LeRobot batch.

        Args:
            batch (dict[str, Tensor]): Batch containing FastWAM-ready keys
                (`video`, `action`, `context`, `context_mask`) or LeRobot keys
                that can be adapted (`observation.images.*`, `observation.state`,
                `action`, `action_is_pad`).

        Returns:
            dict[str, Tensor]: Output dictionary containing the scalar `loss`
            key required by LeRobot and optional tensor metrics.
        """

        sample = _batch_to_training_sample(batch=batch, config=self.config)
        loss, metrics = self.model.training_loss(sample)
        output = {"loss": loss}
        output.update(_metrics_to_tensors(metrics=metrics, device=loss.device))
        return output

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **_: Any) -> Tensor:
        """Predict a chunk of actions from the current FastWAM observation.

        Args:
            batch (dict[str, Tensor]): Inference batch with `input_image` or
                image observation keys, plus `context/context_mask` or `prompt`.

        Returns:
            Tensor: Action chunk with shape `[B, action_horizon, action_dim]`.
        """

        self.eval()
        infer_kwargs = _batch_to_infer_kwargs(batch=batch, config=self.config)
        batch_size = _infer_kwargs_batch_size(infer_kwargs)
        if batch_size == 1:
            action = _action_from_model_output(self.model.infer_action(**infer_kwargs))
        else:
            action = torch.cat(
                [
                    _action_from_model_output(
                        self.model.infer_action(
                            **_slice_infer_kwargs(infer_kwargs, index=i, batch_size=batch_size)
                        )
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )
        return action.to(device=batch_device(batch), dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def _build_core_model(self, config: FastWAMConfig) -> torch.nn.Module:
        return _build_core_model_from_wan22(config)


def _resolve_pretrained_directory(
    pretrained_name_or_path: str | Path,
    *,
    force_download: bool,
    token: str | bool | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
) -> Path:
    path = Path(pretrained_name_or_path)
    if path.is_dir():
        return path

    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=str(pretrained_name_or_path),
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        token=token,
        local_files_only=local_files_only,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "Wan2.2_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "google/umt5-xxl/**",
        ],
    )
    return Path(snapshot_path)


def resolve_wan_component_paths(pretrained_name_or_path: str | Path) -> WanCheckpointPaths:
    """Resolve local Wan component paths stored beside FastWAM HF weights.

    Args:
        pretrained_name_or_path (str | Path): HF-format FastWAM directory.

    Returns:
        WanCheckpointPaths: Existing VAE, text encoder, and tokenizer paths.
        DiT shards are intentionally optional here because FastWAM HF
        checkpoints store trainable DiT weights in `model.safetensors`.
    """

    from .wan_components import resolve_wan_checkpoint_paths

    return resolve_wan_checkpoint_paths(
        pretrained_name_or_path,
        load_dit=False,
        load_text_encoder=True,
    )


def _load_wan_components_into_policy(policy: FastWAMPolicy, paths: WanCheckpointPaths) -> None:
    from .wan_components import load_wan_text_encoder, load_wan_tokenizer, load_wan_vae

    if paths.text_encoder is None or paths.tokenizer is None:
        raise FileNotFoundError("FastWAM HF checkpoint requires Wan text encoder and tokenizer sidecars.")
    dtype = _dtype_from_name(policy.config.torch_dtype)
    device = str(policy.config.device)
    policy.model.vae = load_wan_vae(paths.vae, torch_dtype=dtype, device=device)
    policy.model.text_encoder = load_wan_text_encoder(paths.text_encoder, torch_dtype=dtype, device=device)
    policy.model.tokenizer = load_wan_tokenizer(
        paths.tokenizer,
        tokenizer_max_len=int(policy.config.tokenizer_max_len),
    )
    _record_wan_component_paths(policy=policy, paths=paths)


def _record_wan_component_paths(policy: FastWAMPolicy, paths: WanCheckpointPaths) -> None:
    model_paths = dict(getattr(policy.model, "model_paths", {}) or {})
    model_paths.update(
        {
            "vae": str(paths.vae),
            "text_encoder": str(paths.text_encoder),
            "tokenizer": str(paths.tokenizer),
        }
    )
    policy.model.model_paths = model_paths


def _copy_wan_components_from_policy(policy: FastWAMPolicy, save_directory: Path) -> None:
    model_paths = getattr(policy.model, "model_paths", {}) or {}
    paths = {
        "vae": model_paths.get("vae"),
        "text_encoder": model_paths.get("text_encoder"),
        "tokenizer": model_paths.get("tokenizer"),
    }
    missing = [name for name, path in paths.items() if path is None]
    if missing:
        raise RuntimeError(
            "FastWAM save_pretrained requires local Wan component paths for "
            f"{missing}. Load or initialize the policy with local Wan VAE, text encoder, and tokenizer files."
        )
    _copy_component_path(Path(paths["vae"]), save_directory / Path(paths["vae"]).name)
    _copy_component_path(Path(paths["text_encoder"]), save_directory / Path(paths["text_encoder"]).name)
    tokenizer_source = Path(paths["tokenizer"])
    _copy_component_path(tokenizer_source, save_directory / "google" / "umt5-xxl")


def _copy_component_path(source: Path, destination: Path) -> None:
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"FastWAM component path does not exist: {source}")
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)


def _build_core_model_from_wan22(config: FastWAMConfig) -> torch.nn.Module:
    from .modular_fastwam import FastWAM

    dtype = _dtype_from_name(config.torch_dtype)
    return FastWAM.from_wan22_pretrained(
        device=config.device,
        torch_dtype=dtype,
        model_id=config.model_id,
        tokenizer_model_id=config.tokenizer_model_id,
        tokenizer_max_len=config.tokenizer_max_len,
        load_text_encoder=config.load_text_encoder,
        proprio_dim=config.proprio_dim,
        video_dit_config=config.video_dit_config,
        action_dit_config=config.action_dit_config,
        mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
        video_train_shift=float(config.video_scheduler["train_shift"]),
        video_infer_shift=float(config.video_scheduler["infer_shift"]),
        video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
        action_train_shift=float(config.action_scheduler["train_shift"]),
        action_infer_shift=float(config.action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(config.loss["lambda_video"]),
        loss_lambda_action=float(config.loss["lambda_action"]),
    )


def _build_core_model_from_architecture(config: FastWAMConfig) -> torch.nn.Module:
    from .modular_fastwam import ActionDiT, FastWAM, MoT
    from .wan_video_dit import WanVideoDiT

    dtype = _dtype_from_name(config.torch_dtype)
    video_expert = WanVideoDiT(**config.video_dit_config).to(device=config.device, dtype=dtype)
    action_expert = ActionDiT(**config.action_dit_config).to(device=config.device, dtype=dtype)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=config.mot_checkpoint_mixed_attn,
    )
    return FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=_FastWAMVAEPlaceholder(),
        text_encoder=None,
        tokenizer=None,
        text_dim=int(config.video_dit_config["text_dim"]),
        proprio_dim=config.proprio_dim,
        device=config.device,
        torch_dtype=dtype,
        video_train_shift=float(config.video_scheduler["train_shift"]),
        video_infer_shift=float(config.video_scheduler["infer_shift"]),
        video_num_train_timesteps=int(config.video_scheduler["num_train_timesteps"]),
        action_train_shift=float(config.action_scheduler["train_shift"]),
        action_infer_shift=float(config.action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(config.action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(config.loss["lambda_video"]),
        loss_lambda_action=float(config.loss["lambda_action"]),
    )


class _FastWAMVAEPlaceholder(torch.nn.Module):
    """Minimal VAE placeholder for checkpoint loading without Wan2.2 VAE.

    Args:
        temporal_downsample_factor (int): Temporal compression factor expected
            by FastWAM latent shape logic.
        upsampling_factor (int): Spatial compression factor expected by FastWAM.
        z_dim (int): Latent channel count used by Wan2.2 TI2V VAE.
    """

    temporal_downsample_factor: int = 4
    upsampling_factor: int = 8

    def __init__(self, z_dim: int = 48):
        super().__init__()
        self.model = type("VAEModelShape", (), {"z_dim": int(z_dim)})()

    def encode(self, *args, **kwargs):
        raise RuntimeError(
            "FastWAM VAE placeholder cannot encode images; load Wan2.2 VAE for image inference."
        )

    def decode(self, *args, **kwargs):
        raise RuntimeError(
            "FastWAM VAE placeholder cannot decode latents; load Wan2.2 VAE for video inference."
        )


def _batch_to_training_sample(batch: dict[str, Tensor], config: FastWAMConfig) -> dict[str, Tensor]:
    sample = dict(batch)
    if "video" not in sample:
        sample["video"] = _stack_video_from_images(batch, config)
    if "proprio" not in sample and OBS_STATE in batch:
        sample["proprio"] = batch[OBS_STATE]
    required = {"video", ACTION, "context", "context_mask"}
    missing = sorted(required - set(sample))
    if missing:
        raise KeyError(f"FastWAM training batch is missing keys: {missing}.")
    return sample


def _batch_to_infer_kwargs(batch: dict[str, Tensor], config: FastWAMConfig) -> dict[str, Any]:
    return {
        "prompt": _prompt_from_batch(batch=batch, config=config),
        "input_image": _input_image_from_batch(batch, config),
        "action_horizon": config.action_horizon,
        "proprio": batch.get("proprio", batch.get(OBS_STATE)),
        "context": batch.get("context"),
        "context_mask": batch.get("context_mask"),
        "negative_prompt": batch.get("negative_prompt", config.negative_prompt),
        "text_cfg_scale": float(batch.get("text_cfg_scale", config.text_cfg_scale)),
        "num_inference_steps": int(batch.get("num_inference_steps", config.num_inference_steps)),
        "sigma_shift": batch.get("sigma_shift", config.sigma_shift),
        "seed": batch.get("seed", config.inference_seed),
        "rand_device": batch.get("rand_device", config.rand_device),
        "tiled": bool(batch.get("tiled", config.tiled)),
    }


def _prompt_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Any:
    prompt = batch.get("prompt")
    if prompt is not None:
        return prompt

    task = batch.get("task")
    if task is None:
        return None
    if isinstance(task, str):
        return config.prompt_template.format(task=task)
    if isinstance(task, (list, tuple)):
        return [config.prompt_template.format(task=str(item)) for item in task]
    return config.prompt_template.format(task=str(task))


def _action_from_model_output(output: Any) -> Tensor:
    action = output["action"] if isinstance(output, dict) else output
    if action.ndim == 2:
        action = action.unsqueeze(0)
    return action


def _infer_kwargs_batch_size(infer_kwargs: dict[str, Any]) -> int:
    image = infer_kwargs["input_image"]
    if not isinstance(image, Tensor):
        raise TypeError(f"`input_image` must be a tensor, got {type(image).__name__}.")
    if image.ndim == 3:
        return 1
    if image.ndim == 4:
        return int(image.shape[0])
    raise ValueError(f"`input_image` must be [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")


def _slice_infer_kwargs(infer_kwargs: dict[str, Any], *, index: int, batch_size: int) -> dict[str, Any]:
    return {
        key: _slice_infer_value(value, index=index, batch_size=batch_size)
        for key, value in infer_kwargs.items()
    }


def _slice_infer_value(value: Any, *, index: int, batch_size: int) -> Any:
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return value[index]
    return value


def _dtype_from_name(name: str) -> torch.dtype:
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype `{name}`.")
    return dtype_map[name]


def _metrics_to_tensors(metrics: dict[str, Any] | None, device: torch.device) -> dict[str, Tensor]:
    if metrics is None:
        return {}
    tensor_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            tensor_metrics[key] = value.to(device=device)
        else:
            tensor_metrics[key] = torch.as_tensor(value, device=device)
    return tensor_metrics


def batch_device(batch: dict[str, Any]) -> torch.device:
    for value in batch.values():
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


def _stack_video_from_images(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    image_keys = sorted(k for k in batch if k.startswith("observation.images."))
    if not image_keys:
        raise KeyError("FastWAM batch must contain `video` or `observation.images.*` keys.")
    images = [batch[key] for key in image_keys]
    image = torch.cat(images, dim=-1) if len(images) > 1 else images[0]
    if image.ndim == 4:
        image = image.unsqueeze(2).repeat(1, 1, config.num_video_frames, 1, 1)
    if image.ndim != 5:
        raise ValueError(f"Expected image batch [B,C,H,W] or video [B,C,T,H,W], got {tuple(image.shape)}.")
    return image


def _input_image_from_batch(batch: dict[str, Tensor], config: FastWAMConfig) -> Tensor:
    if "input_image" in batch:
        return _prepare_infer_image(batch["input_image"], config)
    video = batch.get("video")
    if video is None:
        video = _stack_video_from_images(batch, config)
    if video.ndim == 5:
        return _prepare_infer_image(video[:, :, 0], config)
    if video.ndim == 4:
        return _prepare_infer_image(video, config)
    raise ValueError(f"Cannot build input image from tensor with shape {tuple(video.shape)}.")


def _prepare_infer_image(image: Tensor, config: FastWAMConfig) -> Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor [B,C,H,W] or [C,H,W], got {tuple(image.shape)}.")

    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32).div(255.0).mul(2.0).sub(1.0)
    else:
        image = image.to(dtype=torch.float32)
        image_min = float(image.detach().amin().cpu())
        image_max = float(image.detach().amax().cpu())
        if image_min >= 0.0 and image_max <= 1.0:
            image = image.mul(2.0).sub(1.0)
        elif image_max > 2.0:
            image = image.div(255.0).mul(2.0).sub(1.0)

    target_h, target_w = config.image_size
    if tuple(image.shape[-2:]) != (target_h, target_w):
        image = _center_crop_resize(image, target_h=target_h, target_w=target_w)
    return image


def _center_crop_resize(image: Tensor, *, target_h: int, target_w: int) -> Tensor:
    _, _, height, width = image.shape
    scale = max(target_h / height, target_w / width)
    resized_h = round(height * scale)
    resized_w = round(width * scale)
    image = functional.interpolate(image, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top = max((resized_h - target_h) // 2, 0)
    left = max((resized_w - target_w) // 2, 0)
    return image[:, :, top : top + target_h, left : left + target_w].contiguous()
