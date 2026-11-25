# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0
"""SANA-Robot-AR in LeRobot: inference, evaluation and fine-tuning.

The model code is vendored in ``sana_robot_core``. This module drives it from LeRobot's policy interface.
"""

from collections import deque
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import is_package_available, require_package

from .configuration_sana_robot import SanaRobotConfig
from .sana_robot_core.ar_session import ARSession
from .sana_robot_core.augmentation import augment_views
from .sana_robot_core.causal_gdn import CausalGatedDeltaNet
from .sana_robot_core.memory_rtc import MemoryRTCController
from .sana_robot_core.prompt import build_prompts
from .sana_robot_core.robot80 import ROBOT_DIM
from .sana_robot_core.sana_robot_ar import SanaRobotAR
from .sana_robot_core.teacher_forcing_sequence import (
    encode_teacher_forcing_latents,
    strip_view_latents,
    with_state_rows,
)
from .sana_robot_core.text_encoder import load_text_encoder
from .sana_robot_core.training_step import draw_window_noise, pad_chunks, window_flow_inputs, window_losses
from .sana_robot_core.vae import load_video_encoder
from .sana_robot_core.video import VAE_SPATIAL_STRIDE
from .sana_robot_core.view_stream import ViewStream
from .sana_robot_core.windows import window_chunk_count

TEXT_CACHE_SIZE = 64
NORMALIZATION_LAYERS = (
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.modules.instancenorm._InstanceNorm,
)


def resolve_training_backend(requested: str, device: torch.device) -> str:
    """Return how the Gated DeltaNet layers compute a training step on ``device``: `"triton"` or `"eager"`.

    Every CPU device and `"eager"` select plain PyTorch. `"auto"` selects the Triton kernels on CUDA when Triton is
    installed; `"triton"` on CUDA without Triton raises `ValueError`.
    """
    if requested == "eager" or device.type != "cuda":
        return "eager"
    if is_package_available("triton"):
        return "triton"
    if requested == "triton":
        raise ValueError("training_backend='triton' needs the triton package; use 'eager' or 'auto'")
    return "eager"


def pad_rows(values: Tensor, rows: int) -> Tensor:
    """Return ``values`` `[1, R, ...]` with zero rows appended up to ``rows`` rows."""
    return torch.cat((values, values.new_zeros(1, rows - values.shape[1], *values.shape[2:])), dim=1)


def episode_window(real: Tensor, chunk_size: int, video_frame_stride: int) -> tuple[int, int, int]:
    """Return the training window of one sample as (first row, chunks, head rows).

    A sample holds the control steps within a reach of the sampled frame, clamped to its episode. The window is
    the episode from the last chunk boundary at or before its first frame, chosen so that the sampled frame
    starts a chunk, to its end. When the episode starts or ends beyond the reach, the window is the whole chunks
    inside the reach instead, and then starts after the first frame of the episode.

    Args:
        real (`torch.Tensor`):
            Which control steps of the sample are inside the episode, `[2 * reach]`; the sampled frame is at
            index `reach`.
        chunk_size (`int`):
            Control steps per chunk.
        video_frame_stride (`int`):
            Control steps between two video frames.

    Returns:
        `tuple[int, int, int]`: The index of the window's first row in `real`, its number of chunks (0 when it
        has no chunk to train) and the rows it starts before the episode. With rows before the episode, the
        first chunk only fills the memory.
    """
    reach = real.numel() // 2
    (before, after) = (int(real[:reach].sum()), int(real[reach:].sum()))
    head_rows = 0 if before == reach else -before % chunk_size
    first_row = reach - before - head_rows
    if after == reach:
        return (first_row, (2 * reach - first_row) // chunk_size, head_rows)
    return (
        first_row,
        window_chunk_count(before + after, head_rows, chunk_size, video_frame_stride),
        head_rows,
    )


def local_folder(path: str, revision: str | None, patterns: list[str] | None = None) -> str:
    """Return ``path`` when it is a folder, otherwise the folder its Hub repository is downloaded to."""
    if Path(path).is_dir():
        return path
    return snapshot_download(path, revision=revision, allow_patterns=patterns)


class SanaRobotPolicy(PreTrainedPolicy):
    """SANA-Robot-AR: a world action model that denoises the future video latents of every camera and the action
    chunk together, on a memory of the episode.

    A checkpoint holds the weights of the policy alone. The LTX-2 video VAE and the Gemma 2 2B text encoder are
    loaded from ``config.vae_path`` and ``config.text_encoder_path`` when the first action is requested.

    Call `select_action` on every control step: the memory reads a camera frame every
    ``config.video_frame_stride`` steps and a chunk is generated every ``config.n_action_steps`` steps. The policy
    runs one episode at a time (batch size 1).

    `forward` trains on whole episodes with teacher forcing: every sample of a batch is the episode of its
    sampled frame, and the episodes of a batch go through the model as one sequence.
    """

    config_class = SanaRobotConfig
    name = "sana_robot"
    config: SanaRobotConfig
    _fsdp_wrap_modules = ["PolicyBlock"]

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
        self.model.grad_checkpointing = config.gradient_checkpointing
        self.training_backend = "eager"
        if config.dtype is not None:
            self.to(dtype=config.dtype)
        self._encoders: dict[str, Any] = {}
        self._text_cache: dict[str, tuple[Tensor, Tensor]] = {}
        self.reset()

    def get_optim_params(self) -> list[dict[str, Any]]:
        """The trainable parameters in two groups: with the optimizer's weight decay, and without it the biases
        and the weights of the normalization layers, unless ``config.optimizer_decay_biases_and_norms``. The
        RMSNorm gains are decayed."""
        exempt_ids: set[int] = set()
        if not self.config.optimizer_decay_biases_and_norms:
            for module in self.modules():
                if isinstance(module, NORMALIZATION_LAYERS):
                    exempt_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
        decayed: list[torch.nn.Parameter] = []
        exempt: list[torch.nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            is_bias = name.rsplit(".", 1)[-1] == "bias" and not self.config.optimizer_decay_biases_and_norms
            (exempt if is_bias or id(parameter) in exempt_ids else decayed).append(parameter)
        groups: list[dict[str, Any]] = [{"params": decayed}, {"params": exempt, "weight_decay": 0.0}]
        return [group for group in groups if group["params"]]

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

    def forward(self, batch: dict[str, Any], reduction: str = "mean") -> tuple[Tensor, dict | None]:
        """Return the flow-matching loss of a batch of whole-episode windows.

        Every sample is trained on the window of its episode (`episode_window`), on the teacher-forcing sequence:
        each chunk denoises its video latents and its actions at its own noise level, from its observation, the
        first frame of the window and the clean latents of the chunks before it.

        Args:
            batch (`dict[str, Any]`):
                A batch of the dataset after the policy's preprocessor, with the windows of
                `config.image_observation_delta_indices`, `config.state_observation_delta_indices` and
                `config.action_delta_indices`: every camera of `config.camera_keys` as `[B, T, 3, h, w]` in
                [-1, 1] at its view size, `observation.state` `[B, T_state, state dimensions]` and `action`
                `[B, T_action, action dimensions]` normalized, `action_is_pad` `[B, T_action]`, and the
                instructions under `task`.
            reduction (`str`, *optional*, defaults to `"mean"`):
                `"mean"` for the mean over the samples, `"none"` for the loss of every sample `[B]`.

        Returns:
            `tuple[Tensor, dict]`: The loss, the mean over the trained chunks of a window averaged over the
            windows, and its parts: `video_loss`, `action_loss` and `chunks`, the mean number of chunks of a
            window.
        """
        config = self.config
        actions = batch[ACTION]
        if actions.ndim != 3 or actions.shape[1] != 2 * config.max_episode_steps:
            raise ValueError(
                f"Training reads {2 * config.max_episode_steps} actions per sample (config.action_delta_indices), "
                f"got {tuple(actions.shape)}"
            )
        device = actions.device
        # The frozen encoders run in BF16 on a GPU, whatever the precision of the trained weights.
        self._load_encoders(torch.bfloat16 if device.type == "cuda" else None)
        self._apply_training_backend(device)
        real = ~batch[f"{ACTION}_is_pad"]
        tasks = [batch["task"]] * actions.shape[0] if isinstance(batch["task"], str) else batch["task"]
        (windows, trained, chunk_counts) = ([], [], [])
        for index in range(actions.shape[0]):
            (first_row, chunks, head_rows) = episode_window(
                real[index], config.chunk_size, config.video_frame_stride
            )
            chunk_counts.append(chunks)
            # An episode too short for one trained chunk is left out of the step.
            if chunks:
                windows.append(
                    self._training_window(batch, index, first_row, chunks, head_rows, tasks[index])
                )
                trained.append(index)
        sample_losses = actions.new_zeros(actions.shape[0], dtype=torch.float32)
        info = {"video_loss": 0.0, "action_loss": 0.0, "chunks": sum(chunk_counts) / len(chunk_counts)}
        if not windows:
            if reduction == "none":
                return (sample_losses, info)
            return (0.0 * next(self.model.parameters()).sum(), info)
        losses = self._packed_losses(windows)
        sample_losses = sample_losses.index_copy(
            0, torch.tensor(trained, device=device), losses["loss"].to(sample_losses.dtype)
        )
        info["video_loss"] = float(losses["video_loss"].detach().mean())
        info["action_loss"] = float(losses["action_loss"].detach().mean())
        if reduction == "none":
            return (sample_losses, info)
        return (sample_losses.sum() / len(windows), info)

    def _apply_training_backend(self, device: torch.device) -> None:
        """Select the kernels of the Gated DeltaNet layers for training on ``device`` (`resolve_training_backend`)."""
        backend = resolve_training_backend(self.config.training_backend, device)
        if backend != self.training_backend:
            for module in self.model.modules():
                if isinstance(module, CausalGatedDeltaNet):
                    module.fused_kernels = backend == "triton"
            self.training_backend = backend

    def _training_window(
        self, batch: dict[str, Any], index: int, first_row: int, chunks: int, head_rows: int, instruction: str
    ) -> dict[str, Tensor]:
        """Return the window of one sample, ``chunks`` chunks from ``first_row`` of its actions: its robot rows, its clean latents and its text."""
        config = self.config
        (size, stride) = (config.chunk_size, config.video_frame_stride)
        device = batch[ACTION].device
        rows = slice(first_row, first_row + chunks * size)

        action = torch.zeros(chunks * size, ROBOT_DIM, device=device)
        action[:, config.action_slots] = batch[ACTION][index, rows].float()
        action_mask = torch.zeros(chunks * size, ROBOT_DIM, dtype=torch.bool, device=device)
        action_mask[:, config.action_slots] = True
        supervised = torch.ones(chunks, dtype=torch.bool, device=device)
        supervised[0] = head_rows == 0
        inside_episode = ~batch[f"{ACTION}_is_pad"][index, rows]
        loss_mask = action_mask & inside_episode[:, None] & supervised.repeat_interleave(size)[:, None]

        first_state = first_row // size
        state = torch.zeros(chunks, ROBOT_DIM, device=device)
        state[:, config.state_slots] = batch[OBS_STATE][index, first_state : first_state + chunks].float()
        state_mask = torch.zeros(chunks, ROBOT_DIM, dtype=torch.bool, device=device)
        state_mask[:, config.state_slots] = True

        (first_frame, frames) = (first_row // stride, chunks * size // stride + 1)
        clips = [
            batch[key][index : index + 1, first_frame : first_frame + frames] for key in config.camera_keys
        ]
        if self.training and config.augmentation:
            clips = self._augment(clips, config.augmentation)
        views = [encode_teacher_forcing_latents(self._encoders["video"], clip, chunks) for clip in clips]
        without_instruction = self.training and float(torch.rand(())) < config.instruction_dropout
        (text_embeddings, text_mask) = self._text(None if without_instruction else instruction)
        return {
            "initial_state": state[:1],
            "initial_state_condition_mask": state_mask[:1],
            "chunk_states": state[None, 1:],
            "chunk_states_mask": state_mask[None, 1:],
            "action": with_state_rows(action, chunks, size)[None],
            "action_mask": with_state_rows(action_mask, chunks, size)[None],
            "action_loss_mask": with_state_rows(loss_mask, chunks, size)[None],
            "chunk_supervised": supervised[None],
            "window_latents": strip_view_latents([view[0] for view in views]),
            "observation_latents": strip_view_latents([view[1] for view in views]),
            "text_embeddings": text_embeddings,
            "text_mask": text_mask,
        }

    def _augment(self, clips: list[Tensor], settings: dict[str, Any]) -> list[Tensor]:
        """Return the frames of a window, `[1, T, 3, h, w]` in [-1, 1] per camera, after the image changes."""
        with torch.autocast(clips[0].device.type, enabled=False):
            views = augment_views(
                [clip[0].float().add(1).div(2) for clip in clips], self.config.camera_keys, settings
            )
        return [view.mul(2).sub(1)[None].to(clip.dtype) for (view, clip) in zip(views, clips, strict=True)]

    def _packed_losses(self, windows: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Return the losses of every window, `[windows]` each, from one pass of the model.

        The windows are padded to the longest one, as the original trainer's batches are, and the model computes
        them as one packed row: the chunks of all windows in one token sequence, nothing for the padding.
        """
        config = self.config
        size = config.chunk_size
        counts = [window["chunk_supervised"].shape[1] for window in windows]
        longest = max(counts)
        device = windows[0]["action"].device
        packed: dict[str, Any] = {
            **self._data_info(device, len(windows)),
            "teacher_forcing_num_chunks": longest,
            "chunk_counts": torch.tensor(counts, dtype=torch.long, device=device),
        }
        for name in ("initial_state", "initial_state_condition_mask"):
            packed[name] = torch.cat([window[name] for window in windows])
        for name, rows in (
            ("chunk_states", longest - 1),
            ("chunk_states_mask", longest - 1),
            ("action", longest * (size + 1) - 1),
            ("action_mask", longest * (size + 1) - 1),
            ("action_loss_mask", longest * (size + 1) - 1),
            ("chunk_supervised", longest),
        ):
            packed[name] = torch.cat([pad_rows(window[name], rows) for window in windows])
        chunk_frames = (windows[0]["window_latents"].shape[2] - 1) // counts[0]
        window_latents = torch.cat(
            [
                pad_chunks(window["window_latents"], (longest - count) * chunk_frames)
                for (window, count) in zip(windows, counts, strict=True)
            ]
        )
        observation_latents = torch.cat(
            [
                pad_chunks(window["observation_latents"], longest - count)
                for (window, count) in zip(windows, counts, strict=True)
            ]
        )
        noise = draw_window_noise(window_latents, longest, size, ROBOT_DIM)
        (inputs, targets) = window_flow_inputs(
            packed, window_latents, observation_latents, noise, config.timestep_weighting
        )
        output = self.model(
            text_embeddings=torch.cat([window["text_embeddings"] for window in windows]),
            text_mask=torch.cat([window["text_mask"] for window in windows]),
            **inputs,
        )
        per_window = []
        for index in range(len(windows)):
            row = slice(index, index + 1)
            per_window.append(
                window_losses(
                    {name: value[row] for (name, value) in output.items()},
                    {
                        name: value if name.endswith("_positions") else value[row]
                        for (name, value) in targets.items()
                    },
                    {name: packed[name][row] for name in ("chunk_supervised", "action_loss_mask")},
                    config.action_loss_weight,
                )
            )
        return {
            name: torch.stack([losses[name] for losses in per_window])
            for name in ("loss", "video_loss", "action_loss")
        }

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

    def _load_encoders(self, dtype: torch.dtype | None = None) -> None:
        """Load the text encoder and the video VAE on the policy's device, once: in ``dtype``, or in the precision
        of the policy's weights."""
        parameter = next(self.model.parameters())
        if self._encoders.get("device") == parameter.device:
            return
        dtype = dtype or parameter.dtype
        config = self.config
        if config.vae_path is None:
            raise ValueError("Set policy.vae_path to the LTX-2 video VAE in the diffusers layout")
        text_folder = local_folder(config.text_encoder_path, config.text_encoder_revision)
        vae_folder = local_folder(config.vae_path, config.vae_revision, ["vae/*"])
        # Building the encoders draws their initial weights: the noise of the first chunk must not depend on it.
        random_state = torch.get_rng_state()
        self._encoders = {
            "device": parameter.device,
            "text": load_text_encoder(text_folder, parameter.device, dtype),
            "video": load_video_encoder(vae_folder, parameter.device, dtype),
        }
        self._encoders["stream"] = ViewStream(self._encoders["video"])
        torch.set_rng_state(random_state)
        self._text_cache = {}

    def _text(self, instruction: str | None) -> tuple[Tensor, Tensor]:
        """Return the text embeddings and the token mask of the prompts of an instruction, encoded once."""
        key = "" if instruction is None else instruction
        cached = self._text_cache.get(key)
        # Embeddings cached while evaluating in inference mode cannot enter a training graph.
        if cached is None or (cached[0].is_inference() and not torch.is_inference_mode_enabled()):
            if len(self._text_cache) >= TEXT_CACHE_SIZE:
                self._text_cache.clear()
            config = self.config
            prompts = build_prompts(
                config.embodiment, config.action_mode, config.view_descriptions, instruction
            )
            self._text_cache[key] = self._encoders["text"]([prompts])
        return self._text_cache[key]

    def _data_info(self, device: torch.device, batch_size: int = 1) -> dict[str, Any]:
        """Return what the model reads about the cameras and the clock, for ``batch_size`` windows."""
        config = self.config
        latent_shapes = [
            [size[0] // VAE_SPATIAL_STRIDE, size[1] // VAE_SPATIAL_STRIDE] for size in config.view_sizes
        ]
        return {
            "view_latent_shape": torch.tensor(latent_shapes, dtype=torch.long, device=device).repeat(
                batch_size, 1, 1
            ),
            "view_slot_ids": torch.tensor(config.view_slots, dtype=torch.long, device=device).repeat(
                batch_size, 1
            ),
            "view_count": len(config.view_sizes),
            "model_fps": torch.full((batch_size,), float(config.fps), dtype=torch.float64, device=device),
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
