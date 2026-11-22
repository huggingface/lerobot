# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""SANA-Robot-AR in LeRobot: inference and evaluation of the released checkpoints.

The model code is vendored in ``sana_robot_core``. This module drives it from LeRobot's policy interface.
"""

from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, Self

import torch
from huggingface_hub import list_repo_files, snapshot_download
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_sana_robot import SanaRobotConfig
from .sana_robot_core.ar_session import ARSession
from .sana_robot_core.memory_rtc import MemoryRTCController
from .sana_robot_core.prompt import build_prompts
from .sana_robot_core.robot80 import ROBOT_DIM
from .sana_robot_core.sana_robot_ar import SanaRobotAR
from .sana_robot_core.text_encoder import load_text_encoder
from .sana_robot_core.vae import load_video_encoder
from .sana_robot_core.video import VAE_SPATIAL_STRIDE
from .sana_robot_core.view_stream import ViewStream

TEXT_CACHE_SIZE = 64


def local_folder(path: str, revision: str | None, patterns: list[str] | None = None) -> str:
    """Return ``path`` when it is a folder, otherwise the folder its Hub repository is downloaded to."""
    if Path(path).is_dir():
        return path
    return snapshot_download(path, revision=revision, allow_patterns=patterns)


def is_released_checkpoint(path: str, revision: str | None) -> bool:
    """Return whether a folder or a Hub repository is a released SANA-Robot checkpoint: weights in a `.pth`
    file and no LeRobot `config.json`."""
    if Path(path).is_dir():
        names = [file.name for file in Path(path).iterdir()]
    else:
        names = list_repo_files(path, revision=revision)
    return "config.json" not in names and any(name.endswith(".pth") for name in names)


class SanaRobotPolicy(PreTrainedPolicy):
    """SANA-Robot-AR: a world action model that denoises the future video latents of every camera and the action
    chunk together, on a memory of the episode.

    A checkpoint holds the weights of the policy alone. The LTX-2 video VAE and the Gemma 2 2B text encoder are
    loaded from ``config.vae_path`` and ``config.text_encoder_path`` when the first action is requested.

    Call `select_action` on every control step: the memory reads a camera frame every
    ``config.video_frame_stride`` steps and a chunk is generated every ``config.n_action_steps`` steps. The policy
    runs one episode at a time (batch size 1).
    """

    config_class = SanaRobotConfig
    name = "sana_robot"
    config: SanaRobotConfig

    def __init__(self, config: SanaRobotConfig, **kwargs: Any) -> None:
        require_package("diffusers", extra="sana_robot")
        require_package("transformers", extra="sana_robot")
        super().__init__(config)
        config.validate_features()
        self.model = SanaRobotAR(
            actions_per_chunk=config.chunk_size,
            sliding_window_chunks=config.sliding_window_chunks,
            hidden_size=config.hidden_size,
            depth=config.depth,
            cross_attention_heads=config.cross_attention_heads,
            in_channels=config.in_channels,
            caption_channels=config.caption_channels,
            linear_head_dim=config.linear_head_dim,
            softmax_head_dim=config.softmax_head_dim,
            softmax_ratio=config.softmax_ratio,
            attn_res_block_size=config.attn_res_block_size,
            mlp_ratio=config.mlp_ratio,
            vae_temporal_stride=config.vae_temporal_stride,
            spatial_layout=config.spatial_layout,
        )
        self.model.requires_grad_(False)
        if config.dtype is not None:
            self.to(dtype=config.dtype)
        # A LeRobot checkpoint (`pretrained_path`) brings its own weights.
        if config.checkpoint is not None and config.pretrained_path is None:
            state = torch.load(config.checkpoint_weights(), map_location="cpu", weights_only=True, mmap=True)
            for key in ("state_dict_ema", "state_dict"):
                state = state.get(key, state)
            self.model.load_state_dict(state, strict=True)
        self._encoders: dict[str, Any] = {}
        self._text_cache: dict[str, tuple[Tensor, Tensor]] = {}
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        trust_remote_code: bool | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> Self:
        """Load a released SANA-Robot checkpoint (a `.pth` weights file, with `meta.json` and
        `normalization.json` when it is finetuned on a dataset) or a LeRobot checkpoint.

        A released checkpoint takes the defaults of `SanaRobotConfig` for what its files do not say: the
        camera keys, the view sizes, the spatial layout and the sampling settings. Pass `config` to set them.
        """
        if is_released_checkpoint(str(pretrained_name_or_path), revision):
            if config is None:
                config = SanaRobotConfig()
            if not isinstance(config, SanaRobotConfig):
                raise TypeError(f"Expected a SanaRobotConfig, got {type(config).__name__}")
            config = replace(
                config,
                checkpoint=str(pretrained_name_or_path),
                checkpoint_revision=revision,
                pretrained_path=None,
            )
            policy = cls(config, **kwargs)
            policy.to(config.device)
            policy.eval()
            return policy
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                trust_remote_code=bool(trust_remote_code),
            )
        if config.pretrained_path is None:
            config = replace(config, pretrained_path=Path(pretrained_name_or_path))
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            strict=strict,
            **kwargs,
        )

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        return list(self.parameters())

    def reset(self) -> None:
        """Start a new episode: drop the memory, the frames and the queued actions."""
        self._step = 0
        self._window: deque[tuple[Tensor, ...]] = deque(
            maxlen=self.config.chunk_size // self.config.video_frame_stride
        )
        self._action_queue: deque[Tensor] = deque()
        self._instruction: str | None = None
        self._session: ARSession | None = None
        self._controller: MemoryRTCController | None = None

    def drop_queued_actions(self) -> None:
        """Queued actions belong to the memory's clock, so a new instruction takes over at the next chunk instead."""
        return None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("SANA-Robot supports inference and evaluation in LeRobot, not training")

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        raise NotImplementedError(
            "SANA-Robot reads a frame every video_frame_stride control steps: call select_action on every step"
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        """Return the action of the current control step, [1, action dimensions], normalized.

        Args:
            batch (`dict[str, Tensor]`):
                The observation of the step after the policy's preprocessor: every camera of
                `config.camera_keys` as [1, 3, h, w] in [-1, 1] at its view size, `observation.state` as
                [1, state dimensions] normalized, and the instruction under `task`.
        """
        self.eval()
        config = self.config
        frames = self._frames(batch)
        instruction = self._task(batch)
        if self._step == 0:
            self._open(frames, instruction)
        elif self._step % config.video_frame_stride == 0:
            self._window.append(frames)
        if self._step % config.n_action_steps == 0:
            self._plan(batch[OBS_STATE], instruction)
        self._step += 1
        return self._action_queue.popleft()

    def _frames(self, batch: dict[str, Tensor]) -> tuple[Tensor, ...]:
        """Return the current frame of every camera as [3, 1, h, w]."""
        frames = []
        for key, size in zip(self.config.camera_keys, self.config.view_sizes, strict=True):
            image = batch[key]
            if image.ndim != 4 or tuple(image.shape) != (1, 3, *size):
                raise ValueError(f"{key} must be [1, 3, {size[0]}, {size[1]}], got {tuple(image.shape)}")
            frames.append(image[0].unsqueeze(1))
        return tuple(frames)

    @staticmethod
    def _task(batch: dict[str, Any]) -> str:
        task = batch["task"]
        if isinstance(task, str):
            return task
        if len(task) != 1:
            raise ValueError("SANA-Robot runs one episode at a time: use a batch size of 1")
        return task[0]

    def _load_encoders(self) -> None:
        """Load the text encoder and the video VAE on the policy's device, once."""
        parameter = next(self.model.parameters())
        if self._encoders.get("device") == parameter.device:
            return
        config = self.config
        if config.vae_path is None:
            raise ValueError("Set policy.vae_path to the LTX-2 video VAE in the diffusers layout")
        text_folder = local_folder(config.text_encoder_path, config.text_encoder_revision)
        vae_folder = local_folder(config.vae_path, config.vae_revision, ["vae/*"])
        # Building the encoders draws their initial weights: the noise of the first chunk must not depend on it.
        random_state = torch.get_rng_state()
        self._encoders = {
            "device": parameter.device,
            "text": load_text_encoder(text_folder, parameter.device, parameter.dtype),
            "stream": ViewStream(load_video_encoder(vae_folder, parameter.device, parameter.dtype)),
        }
        torch.set_rng_state(random_state)
        self._text_cache = {}

    def _text(self, instruction: str | None) -> tuple[Tensor, Tensor]:
        """Return the text embeddings and the token mask of the prompts of an instruction, encoded once."""
        key = "" if instruction is None else instruction
        if key not in self._text_cache:
            if len(self._text_cache) >= TEXT_CACHE_SIZE:
                self._text_cache.clear()
            config = self.config
            prompts = build_prompts(
                config.embodiment, config.action_mode, config.view_descriptions, instruction
            )
            self._text_cache[key] = self._encoders["text"]([prompts])
        return self._text_cache[key]

    def _data_info(self, device: torch.device) -> dict[str, Any]:
        """Return what the model reads about the cameras and the clock."""
        config = self.config
        latent_shapes = [
            [size[0] // VAE_SPATIAL_STRIDE, size[1] // VAE_SPATIAL_STRIDE] for size in config.view_sizes
        ]
        return {
            "view_latent_shape": torch.tensor(latent_shapes, dtype=torch.long, device=device)[None],
            "view_slot_ids": torch.tensor(config.view_slots, dtype=torch.long, device=device)[None],
            "view_count": len(config.view_sizes),
            "model_fps": torch.tensor([float(config.fps)], dtype=torch.float64, device=device),
            "video_frame_stride": config.video_frame_stride,
        }

    def _open(self, frames: tuple[Tensor, ...], instruction: str) -> None:
        """Open an episode at its first frames: a new session and a window of copies of the first frame."""
        self._load_encoders()
        config = self.config
        unconditional = self._text(None) if config.video_cfg_scale != 1.0 else (None, None)
        self._session = ARSession(
            self.model,
            *self._text(instruction),
            self._data_info(frames[0].device),
            config.num_inference_steps,
            config.video_flow_shift,
            config.action_flow_shift,
            config.video_cfg_scale,
            *unconditional,
        )
        self._controller = MemoryRTCController(
            self._session, self._encoders["stream"], 0, config.n_action_steps
        )
        self._instruction = instruction
        self._window.clear()
        self._window.extend([frames] * (config.chunk_size // config.video_frame_stride))

    def _plan(self, state: Tensor, instruction: str) -> None:
        """Bring the memory to the current step, generate the next chunk and queue its first actions."""
        config = self.config
        if self._session is None or self._controller is None:
            raise RuntimeError("No episode is open")
        if instruction != self._instruction:
            self._session.set_prompt(*self._text(instruction))
            self._instruction = instruction
        device = state.device
        clips = [torch.cat(view, dim=1) for view in zip(*self._window, strict=True)]
        row = torch.zeros(1, ROBOT_DIM, dtype=torch.float32, device=device)
        row[:, config.state_slots] = state.float()
        state_mask = torch.zeros(1, ROBOT_DIM, dtype=torch.bool, device=device)
        state_mask[:, config.state_slots] = True
        action_mask = torch.zeros(1, config.chunk_size, ROBOT_DIM, dtype=torch.bool, device=device)
        action_mask[:, :, config.action_slots] = True
        if self._step == 0:
            actions = self._controller.open(clips, row, state_mask, action_mask)
        else:
            actions = self._controller.generate(self._step, clips, row, state_mask, action_mask)
        actions = actions[:, : config.n_action_steps, config.action_slots].float()
        self._action_queue.extend(actions.transpose(0, 1))
