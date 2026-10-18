# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native train/save/load API and an ordered, per-control-tick Long-WAM runtime."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_longwam import LongWAMConfig
from .scheduling_longwam import AsyncPolicy, Schedule, SyncPolicy, SyncSchedule
from .streaming_longwam import StreamingVAE


class LongWAMPolicy(PreTrainedPolicy):
    config: LongWAMConfig
    config_class = LongWAMConfig
    name = "longwam"

    def __init__(
        self,
        config: LongWAMConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *,
        load_initial_weights: bool | None = None,
        tokenizer_assets: str | Path | None = None,
        load_backbone_assets: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        if load_initial_weights is None:
            load_initial_weights = config.pretrained_path is None
        if not load_initial_weights and tokenizer_assets is None and config.pretrained_path:
            # The factory's deferred-weight path constructs the policy directly.
            tokenizer_assets = self._checkpoint_tokenizer(
                config.pretrained_path, revision=config.pretrained_revision
            )
        self.model = self._build_core_model(
            config,
            load_initial_weights=load_initial_weights,
            tokenizer_assets=tokenizer_assets,
            load_backbone_assets=load_backbone_assets,
        )
        self._runtime: AsyncPolicy | SyncPolicy | None = None
        self._instruction: str | None = None
        for name in ("vae", "text_encoder"):
            module = getattr(self.model, name, None)
            if module is not None:
                module.requires_grad_(False).eval()

    @staticmethod
    def _build_core_model(
        config: LongWAMConfig,
        *,
        load_initial_weights: bool,
        tokenizer_assets: str | Path | None = None,
        load_backbone_assets: bool = False,
    ) -> torch.nn.Module:
        from .weights_longwam import build_core_model

        return build_core_model(
            config,
            load_initial_weights=load_initial_weights,
            tokenizer_assets=tokenizer_assets,
            load_backbone_assets=load_backbone_assets,
        )

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self:
        # A complete policy must not be initialized from Robot-S again on reload.
        kwargs["load_initial_weights"] = False
        kwargs.setdefault("strict", True)
        location = args[0] if args else kwargs.get("pretrained_name_or_path")
        if location is not None:
            options = {
                key: kwargs[key]
                for key in ("revision", "cache_dir", "token", "local_files_only", "force_download")
                if key in kwargs
            }
            kwargs["tokenizer_assets"] = cls._checkpoint_tokenizer(location, **options)
        return super().from_pretrained(*args, **kwargs)

    @staticmethod
    def _checkpoint_tokenizer(location: str | Path, **options: Any) -> Path:
        local = Path(location)
        if local.is_dir():
            return local / "tokenizer"
        from huggingface_hub import snapshot_download

        root = snapshot_download(str(location), allow_patterns=["tokenizer/*"], **options)
        return Path(root) / "tokenizer"

    def _save_pretrained(self, save_directory: Path) -> None:
        from lerobot.distributed.utils import is_main_process

        super()._save_pretrained(save_directory)
        tokenizer = getattr(self.model, "tokenizer", None)
        if is_main_process() and tokenizer is not None:
            tokenizer.tokenizer.save_pretrained(Path(save_directory) / "tokenizer")

    def _apply(self, fn: Any, recurse: bool = True) -> Self:
        result = super()._apply(fn, recurse=recurse)
        # The original core stores these in addition to Module parameters.
        parameter = next(self.model.parameters())
        self.model.device, self.model.torch_dtype = parameter.device, parameter.dtype
        return result

    def train(self, mode: bool = True) -> Self:
        if mode:
            self.close()
        super().train(mode)
        for name in ("vae", "text_encoder"):
            module = getattr(self.model, name, None)
            if module is not None:
                module.eval()
        return self

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _images(self, batch: dict[str, Any], *, temporal: bool) -> Tensor:
        images = []
        for key in self.config.camera_keys:
            image = batch[key]
            expected = 5 if temporal else 4
            if image.ndim != expected or image.shape[-3] != 3:
                raise ValueError(f"{key} must have shape {'[B,T,3,H,W]' if temporal else '[B,3,H,W]'}")
            if (
                not image.is_floating_point()
                or not torch.isfinite(image).all()
                or image.min() < 0
                or image.max() > 1
            ):
                raise ValueError("Long-WAM expects unnormalized float images in [0,1]")
            if not temporal and self.config.inference_image_preprocessing == "pil":
                # Match released inference exactly: uint8 PIL center-crop/resize,
                # then normalize in the model dtype (including BF16 rounding).
                height, width = self.config.camera_size
                resized = []
                for frame in image:
                    pixels = frame.detach().float().mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                    pil = Image.fromarray(pixels)
                    scale = max(width / pil.width, height / pil.height)
                    pil = pil.resize((round(pil.width * scale), round(pil.height * scale)), Image.BILINEAR)
                    left, top = max((pil.width - width) // 2, 0), max((pil.height - height) // 2, 0)
                    pixels = np.array(pil.crop((left, top, left + width, top + height)), copy=True)
                    resized.append(torch.from_numpy(pixels).permute(2, 0, 1))
                images.append(torch.stack(resized).to(device=image.device, dtype=self.config.dtype))
                continue
            if temporal:
                expected_steps = len(self.config.observation_delta_indices)
                if image.shape[1] != expected_steps:
                    raise ValueError(
                        f"Training needs {expected_steps} raw frames, including history and future"
                    )
                image = image[:, :: self.config.video_stride]
                b, t = image.shape[:2]
                image = image.flatten(0, 1)
            image = functional.interpolate(
                image.float(),
                size=self.config.camera_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            if temporal:
                image = image.reshape(b, t, 3, *self.config.camera_size).transpose(1, 2)
            images.append(image)
        # Camera order is explicit and never inferred by sorting feature names.
        combined = torch.cat(images, dim=-1)
        scale = 2.0 / 255.0 if not temporal and self.config.inference_image_preprocessing == "pil" else 2.0
        return combined.mul(scale).sub(1)

    def _tasks(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        task = batch.get("task")
        if isinstance(task, str):
            task = [task] * batch_size
        if (
            not isinstance(task, (list, tuple))
            or len(task) != batch_size
            or not all(isinstance(t, str) for t in task)
        ):
            raise ValueError("Provide one task string per batch element")
        return list(task)

    def _training_sample(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        video = self._images(batch, temporal=True)
        action = batch["action"]
        if action.shape != (video.shape[0], self.config.action_horizon, self.config.action_dim):
            raise ValueError("Training action shape does not match the configured horizon")
        state = batch["observation.state"]
        if state.shape != (
            video.shape[0],
            len(self.config.observation_delta_indices),
            self.config.proprio_dim,
        ):
            raise ValueError("Training state must use the same raw temporal offsets as the images")
        sample = {"video": video, "action": action, "proprio": state[:, :-1]}
        if "action_is_pad" in batch:
            sample["action_is_pad"] = batch["action_is_pad"]
        masks = [batch.get(f"{key}_is_pad") for key in self.config.camera_keys]
        if any(mask is not None for mask in masks):
            if any(mask is None for mask in masks):
                raise ValueError("Padding masks must be present for all cameras")
            valid_masks = [mask for mask in masks if mask is not None]
            if any(mask.shape != state.shape[:2] for mask in valid_masks):
                raise ValueError("Camera padding masks must match the raw temporal window")
            sample["image_is_pad"] = torch.stack(valid_masks).any(dim=0)[:, :: self.config.video_stride]
        if ("context" in batch) != ("context_mask" in batch):
            raise ValueError("context and context_mask must be supplied together")
        if "context" in batch:
            sample.update(context=batch["context"], context_mask=batch["context_mask"])
        else:
            prompts = [self.config.prompt_template.format(task=t) for t in self._tasks(batch, video.shape[0])]
            sample["context"], sample["context_mask"] = self.model.encode_prompt(prompts)
        return sample

    def forward(self, batch: dict[str, Any], reduction: str = "mean") -> tuple[Tensor, dict[str, Any]]:
        if reduction != "mean":
            raise ValueError("The reference Long-WAM loss supports reduction='mean' only")
        if self._runtime is not None:
            raise RuntimeError("Call train() to close inference before training")
        loss, metrics = self.model.training_loss(self._training_sample(batch))
        return loss, {
            key: value.detach().item() if isinstance(value, Tensor) else value
            for key, value in metrics.items()
        }

    def _ensure_runtime(self) -> AsyncPolicy | SyncPolicy:
        if self._runtime is None:
            cfg = self.config
            worker = _OrderedWorker(self)
            if cfg.execution == "async":
                assert cfg.trigger_stride is not None
                self._runtime = AsyncPolicy(
                    worker,
                    Schedule(cfg.prediction_steps, cfg.n_action_steps, cfg.trigger_stride),
                    timeout=cfg.inference_timeout,
                )
            else:
                self._runtime = SyncPolicy(
                    worker,
                    SyncSchedule(cfg.prediction_steps, cfg.n_action_steps),
                    timeout=cfg.inference_timeout,
                )
        return self._runtime

    def reset(self) -> None:
        # Match the released adapter: the next observation supplies the new task
        # and performs exactly one ordered reset, draining the old pending request.
        self._instruction = None

    def drop_queued_actions(self) -> None:
        raise ValueError(
            "Long-WAM initially requires fixed instructions within an episode; call reset() between tasks"
        )

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        raise NotImplementedError(
            "Use inline select_action on every tick; generic async/RTC is not supported"
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        if kwargs:
            raise ValueError("External noise and RTC overrides are not supported")
        if batch["observation.state"].shape != (1, self.config.proprio_dim):
            raise ValueError("Initial Long-WAM inline inference requires one environment per policy")
        task = self._tasks(batch, 1)[0]
        if self._instruction is not None and task != self._instruction:
            raise ValueError("Call reset() before changing the task")
        image = self._images(batch, temporal=False)
        if image.shape[0] != 1:
            raise ValueError("Initial Long-WAM inference requires batch_size=1")
        state = batch["observation.state"]
        if not torch.isfinite(state).all():
            raise ValueError("Non-finite robot state")
        if ("context" in batch) != ("context_mask" in batch):
            raise ValueError("context and context_mask must be supplied together")
        runtime = self._ensure_runtime()
        if self._instruction is None:
            runtime.reset(task)
            self._instruction = task
        observation = {"image": image[0].detach().clone(), "state": state.detach().clone()}
        for key in ("context", "context_mask"):
            if key in batch:
                observation[key] = batch[key].detach().clone()
        action = runtime.act(observation)
        return torch.as_tensor(action, device=state.device, dtype=state.dtype).unsqueeze(0)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._instruction = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _OrderedWorker:
    """One ordered worker; observation updates cannot overtake an inference anchor.

    The native policy owns its parameters so LeRobot can optimize and serialize
    them. A single worker thread shares that module instead of constructing a
    second model in a subprocess. Training must close this inference worker.
    """

    def __init__(self, policy: LongWAMPolicy) -> None:
        self.policy = policy
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="longwam")
        self.frames: deque[Tensor] = deque(maxlen=policy.config.past_obs_steps + 1)
        self.episode, self.last_step = 0, -1
        self.current: dict[str, Any] | None = None
        self.instruction = ""
        self._observation_error: Exception | None = None
        self._text_conditioning: tuple[Tensor, Tensor] | None = None
        self._next_anchor: int | None = None
        self._streaming = (
            StreamingVAE(
                policy.model,
                past_steps=policy.config.past_obs_steps,
                sample_stride=policy.config.video_stride,
            )
            if policy.config.streaming_vae
            else None
        )

    def _frame_history(self) -> list[tuple[int, Tensor]]:
        start = self.last_step - len(self.frames) + 1
        return list(enumerate(self.frames, start=start))

    def submit(self, operation: str, payload: dict[str, Any] | None = None) -> Future:
        # Propagate autocast across the worker-thread boundary.
        device = next(self.policy.model.parameters()).device.type
        enabled = torch.is_autocast_enabled(device)
        dtype = torch.get_autocast_dtype(device)
        return self.executor.submit(self._run, operation, payload, device, enabled, dtype)

    def notify(self, operation: str, payload: dict[str, Any]) -> None:
        self.submit(operation, payload)

    def call(self, operation: str, payload: dict[str, Any] | None = None) -> Any:
        return self.submit(operation, payload).result(timeout=self.policy.config.inference_timeout)

    def _run(
        self, operation: str, payload: dict[str, Any] | None, device: str, enabled: bool, dtype: torch.dtype
    ) -> Any:
        with torch.no_grad(), torch.autocast(device, enabled=enabled, dtype=dtype):
            try:
                if payload is None:
                    raise ValueError("Long-WAM worker operations require a payload")
                return self._handle(operation, payload)
            except Exception as exc:
                if operation == "observe":
                    self._observation_error = exc
                raise

    def _handle(self, operation: str, payload: dict[str, Any]) -> Any:
        if operation == "reset":
            self._observation_error = None
            self._text_conditioning = None
            self.episode, self.instruction = payload["episode"], payload["instruction"]
            self.last_step, self.current = -1, None
            self.frames.clear()
            self._next_anchor = None
            if self._streaming is not None:
                self._streaming.reset()
            return None
        if payload["episode"] != self.episode:
            raise RuntimeError("Request belongs to another episode")
        if operation == "observe":
            if payload["step"] != self.last_step + 1:
                raise ValueError("Observations must arrive at consecutive control steps")
            observation = payload["observation"]
            self.current, self.last_step = observation, payload["step"]
            self.frames.append(observation["image"])
            session, cfg = self._streaming, self.policy.config
            if session is not None:
                if session.anchor is None and self._next_anchor is not None:
                    session.begin(self._next_anchor, self._frame_history())
                if session.anchor is not None:
                    first = session.anchor - cfg.past_obs_steps
                    if self.last_step <= session.anchor and (self.last_step - first) % cfg.video_stride == 0:
                        session.feed(self.last_step, observation["image"])
            return None
        if self._observation_error is not None:
            raise RuntimeError("An observation update failed") from self._observation_error
        if operation != "infer" or payload["anchor"] != self.last_step:
            raise ValueError("Inference must use its exact observation anchor")
        if self.current is None:
            raise ValueError("Observe before requesting inference")
        cfg, core = self.policy.config, self.policy.model
        frames = list(self.frames)
        frames = [frames[0]] * (cfg.past_obs_steps + 1 - len(frames)) + frames
        window = torch.stack(frames[:: cfg.video_stride], dim=1).unsqueeze(0)
        kwargs = {
            "prompt": cfg.prompt_template.format(task=self.instruction),
            "input_image": frames[-1].unsqueeze(0),
            "obs_window": window if cfg.past_obs_steps else None,
            "proprio": self.current["state"],
            "action_horizon": cfg.prediction_steps,
            "num_inference_steps": cfg.num_inference_steps,
            "seed": cfg.inference_seed,
            "rand_device": "cpu",
            "text_cfg_scale": 1.0,
        }
        if self._streaming is not None:
            anchor = payload["anchor"]
            if self._streaming.anchor != anchor:
                self._streaming.begin(anchor, self._frame_history())
            kwargs["clean_latents"] = self._streaming.finish(anchor)
        for key in ("context", "context_mask"):
            if key in self.current:
                kwargs[key] = self.current[key]
        if "context" in kwargs:
            kwargs["prompt"] = None
        else:
            if self._text_conditioning is None:
                self._text_conditioning = core.encode_prompt([kwargs["prompt"]])
            assert self._text_conditioning is not None
            kwargs["context"], kwargs["context_mask"] = self._text_conditioning
            kwargs["prompt"] = None
        actions = core.infer_joint_ar(**kwargs)["action"]
        if actions.shape != (cfg.prediction_steps, cfg.action_dim):
            raise ValueError("Core returned an invalid action chunk")
        stride = cfg.trigger_stride if cfg.execution == "async" else cfg.n_action_steps
        assert stride is not None
        self._next_anchor = payload["anchor"] + stride
        return {**payload, "actions": actions.detach().float().cpu().numpy()}

    def close(self, force: bool = False) -> None:
        self.executor.shutdown(wait=True, cancel_futures=force)
        if self._streaming is not None:
            self._streaming.reset()
