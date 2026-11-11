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

"""LeRobot policy adapter of Praxis-1, the single-arm video-action model.

One policy call denoises the next chunk of video (6 latents) and the next
24 robot commands at 15 Hz. The policy keeps, per environment of the batch,
the praxis session: a ring buffer of the last 25 observations (the executed
chunk plus the current frame), the encoded context latents and the sampler
seed. ``select_action`` pushes the incoming observation, refills the command
queue when it is empty and returns one command per call.

The model code is the vendored praxis core (``praxis_core``); this module only
translates LeRobot batches to the praxis observation contract and the praxis
commands to LeRobot actions.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import logging
import math
import os
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar, Unpack

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from lerobot.configs import PreTrainedConfig
from lerobot.lerobot_types import PolicyOutput
from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    PREDICTION,
)
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.import_utils import require_package

from ..chunk import ChunkPolicySpec
from .configuration_praxis import PRAXIS_ACTION_DIM, PRAXIS_POSE_DIM, PraxisConfig
from .praxis_core import bimanual, layers as praxis_layers
from .praxis_core.action_expert import FLEX_DYNAMIC_KERNEL_OPTIONS
from .praxis_core.backbone import _TeacherForcingMaskCache
from .praxis_core.bimanual import BimanualChunkInputs, BimanualServeSession
from .praxis_core.checkpoint import model_variant, validate_config
from .praxis_core.model import PraxisModel, SamplerSettings
from .praxis_core.robot_io import ASPECT_TOL, FPS, NormStats, contiguous_chunk_geometry, pose8_to_10d
from .praxis_core.single_arm import (
    N_CHUNKS,
    SingleArmChunkInputs,
    SingleArmServeSession,
    canvas_size,
    q99_dims,
    tile_clip,
)
from .praxis_core.train.data import TrainingWindow, droid_pose8
from .praxis_core.train.moe import update_expert_biases
from .praxis_core.train.objective import FinetuneModel, encode_window
from .praxis_core.train.optim import derive_seed
from .praxis_core.vae import Autoencoder
from .processor_praxis import droid7_to_pose8

logger = logging.getLogger(__name__)

K = TypeVar("K")

#: The observation key of the scene seed an environment may provide (see ``PraxisConfig.seed``).
OBS_SCENE_SEED = "observation.scene_seed"


@functools.lru_cache(maxsize=1)
def flash_backend_available() -> bool:
    """Return True when ``flash-attn-4`` (the ``flash_attn.cute`` package) is importable.

    This is an import check only. A probe that compiles ``flex_attention`` before the
    model's first call would make Dynamo see a second shape and select the dynamic
    kernel variant for the model, which changes the output bits (see the praxis
    README on compile cache order). With the package installed, flex attention runs
    its FlashAttention-4 CuTe backend on torch >= 2.11.
    """
    try:
        import flash_attn.cute  # noqa: F401
    except ImportError:
        return False
    return True


def flash_takes_block_masks(device: torch.device) -> bool:
    """Return False on GPUs where FlashAttention-4 has no block-sparse forward.

    The backbone's ``flash`` backend runs the flex attention masks through it. flash-attn-4
    4.0.0b36 has no block-sparse forward on compute capability 12.x (RTX PRO 6000 Blackwell,
    RTX 50 series, GB10), whatever the block size: the checkpoint's 128-token blocks fail with
    ``Block sparsity requires sparse_block_size[1]=128 to match tile_n=64``, and blocks of its
    128 x 64 tile with ``Block sparsity not supported on SM 12.0``.
    """
    return torch.cuda.get_device_capability(device)[0] != 12


def resolve_attention_backend(
    requested: str, device: torch.device
) -> tuple[praxis_layers.AttentionBackend, bool]:
    """Return ``(backbone backend, compiled flex attention)`` for the request and the device.

    ``eager`` and every CPU device select the uncompiled flex attention with
    the ``triton`` mask layout. ``auto`` selects ``flash`` when ``flash-attn-4`` is
    importable (:func:`flash_backend_available`) and takes the masks on the device
    (:func:`flash_takes_block_masks`), else ``triton``. ``flash`` on a device whose
    FlashAttention-4 takes no masks raises ``ValueError``.
    """
    if device.type != "cuda" or requested == "eager":
        return "triton", False
    if requested == "flash":
        if not flash_takes_block_masks(device):
            major, minor = torch.cuda.get_device_capability(device)
            raise ValueError(
                f"attention_backend='flash' cannot run on compute capability {major}.{minor}: "
                "FlashAttention-4 has no block-sparse forward there; use 'triton' or 'auto'"
            )
        return "flash", True
    if requested == "triton":
        return "triton", True
    flash = flash_backend_available() and flash_takes_block_masks(device)
    return ("flash" if flash else "triton"), True


#: The block of the block-sparse masks FlashAttention-4 takes on compute capability 10.x and 11.x
#: (B200, GB200, B300, GB300; Jetson Thor): its forward stages two 128-token query tiles there and
#: refuses other block sizes.
FLASH_BLACKWELL_MASK_BLOCK = 256


def mask_block_size(block: int, backend: str, device: torch.device) -> int:
    """Return the block size of the backbone's flex attention masks for the backend and the device.

    The checkpoint's ``backbone.attention.flex_block_size`` (128 for the released checkpoints),
    except for the ``flash`` backend on compute capability 10.x or 11.x, which takes the least
    multiple of both that and 256 tokens. The block size only sets the granularity of the
    block lists: every partial block applies the mask function to each token pair.
    """
    if backend != "flash" or device.type != "cuda" or block % FLASH_BLACKWELL_MASK_BLOCK == 0:
        return block
    if torch.cuda.get_device_capability(device)[0] not in (10, 11):
        return block
    return math.lcm(block, FLASH_BLACKWELL_MASK_BLOCK)


def _resolve_weight_file(spec: str, default_filename: str, **hub_kwargs: Any) -> str:
    """Return a local file for ``spec``: a path, ``repo_id:filename`` or a repo id.

    ``hub_kwargs`` are the download options of the policy load (``cache_dir``,
    ``local_files_only``, ``token``, ...), so an offline load stays offline.
    """
    if os.path.exists(spec):
        return spec
    if ":" in spec:
        repo_id, filename = spec.rsplit(":", 1)
    else:
        repo_id, filename = spec, default_filename
    return hf_hub_download(repo_id, filename, **hub_kwargs)


def _resolve_model_dir(spec: str, **hub_kwargs: Any) -> str:
    """Return a local directory for ``spec``: a path or a Hub repo id (see ``_resolve_weight_file``)."""
    if os.path.isdir(spec):
        return spec
    return snapshot_download(spec, **hub_kwargs)


def _cast_parameters(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast the floating-point parameters of ``module`` to ``dtype``; the buffers keep their dtype.

    The praxis model keeps float32 buffers (the routing biases, the rotary tables) next to its
    bfloat16 weights. Works on the meta device too.
    """
    for param in module.parameters():
        if param.is_floating_point() and param.dtype != dtype:
            param.data = param.data.to(dtype)


def _uint8_chw(image: Tensor) -> Tensor:
    """Return one LeRobot image ``(3, H, W)`` (float in [0, 1] or uint8) as uint8 ``(3, H, W)``, same device.

    LeRobot divides the uint8 camera frame by 255; the multiply and round here give the original
    bytes back.
    """
    if image.ndim == 4:
        image = image[-1]  # a frame stack: the current frame is last
    if image.ndim != 3:
        raise ValueError(f"expected an image (3, H, W), got {tuple(image.shape)}")
    if image.dtype != torch.uint8:
        image = (image.float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
    return image.detach()


def _frame_for_history(image: Tensor, tile_hw: tuple[int, int]) -> Tensor | np.ndarray:
    """A preprocessor tile stays a ``(3, H, W)`` uint8 tensor on its device; a raw frame becomes ``(H, W, 3)`` uint8 numpy.

    The tiles need no canvas work and go straight to the autoencoder, so they are not copied off
    the device; a raw frame (a policy called without the pipeline) takes the canvas rule on the CPU.
    """
    image = _uint8_chw(image)
    if tuple(image.shape[-2:]) == tuple(tile_hw):
        return image
    return image.permute(1, 2, 0).contiguous().cpu().numpy()


def _state_rows(batch: dict[str, Any], index: int, n_frames: int, n_arms: int = 1) -> np.ndarray:
    """The ``(n_frames, 8 * n_arms)`` pose rows of one environment: the state of each frame of the call.

    The state is ``(7 * n_arms,)`` (the DROID pose and gripper of each arm) or ``(8 * n_arms,)``
    (pose8 of each arm, from the pose processor step) for one frame, or a stack ``(T, ...)``. A
    stack with one camera frame per call gives its last row (the current state).
    """
    state = batch[OBS_STATE][index]
    rows = state[None] if state.ndim == 1 else state
    droid, pose = PRAXIS_ACTION_DIM * n_arms, PRAXIS_POSE_DIM * n_arms
    if rows.ndim != 2 or rows.shape[-1] not in (droid, pose):
        raise ValueError(
            f"{OBS_STATE} must have {droid} values (the DROID pose and gripper of each of the {n_arms} arm(s)) "
            f"or {pose} (the praxis pose8 of the pose processor step) per frame, got {tuple(state.shape)}"
        )
    if rows.shape[0] != n_frames:
        if n_frames != 1:
            raise ValueError(
                f"{OBS_STATE} holds {rows.shape[0]} rows but the cameras hold {n_frames} frames; "
                "a frame stack needs the state of every frame"
            )
        rows = rows[-1:]
    values = rows.detach().float().cpu().numpy()
    if values.shape[-1] == droid:
        values = np.concatenate(
            [droid7_to_pose8(values[:, 7 * i : 7 * i + 7]) for i in range(n_arms)], axis=1
        )
    return values.astype(np.float32)


def _task_of(batch: dict[str, Any], index: int) -> str:
    task = batch.get("task")
    if task is None:
        return ""
    if isinstance(task, str):
        return task.strip()
    value = task[index] if index < len(task) else ""
    return (value if isinstance(value, str) else str(value)).strip()


@contextlib.contextmanager
def _praxis_numerics():
    """Run with the numerical settings of the praxis pipeline, whatever the host has set.

    praxis runs on PyTorch defaults: no cuDNN benchmark mode (choosing convolution algorithms
    by timing changes the last bits of the VAE), TF32 allowed in cuDNN, and float32 matmuls in
    full precision (the MoE router routes in float32). ``lerobot-eval`` turns benchmark mode on
    and its ``--tf32`` flag toggles both TF32 switches at once; training scripts set their own.
    Pinning the settings here keeps the policy bitwise equal to the praxis pipeline in every
    harness and for every batch size. The host settings are restored afterwards.
    """
    cudnn = torch.backends.cudnn
    matmul = torch.backends.cuda.matmul
    saved_cudnn = (cudnn.benchmark, cudnn.allow_tf32)
    # The float32 matmul precision is one global switch (`set_float32_matmul_precision`, which
    # `matmul.allow_tf32` also moves) with per-backend switches behind it (`fp32_precision`,
    # PyTorch 2.9+). Moving the switches through different APIs leaves torch unable to report the
    # global value, so the guard reads and writes the global switch only; when the host has
    # already mixed them it saves and restores the cuBLAS switch instead.
    try:
        saved_precision: str | None = torch.get_float32_matmul_precision()
    except RuntimeError:  # the host set a backend precision the global switch cannot report
        saved_precision = None
    saved_cublas = getattr(matmul, "fp32_precision", None) if saved_precision is None else None
    cudnn.benchmark = False
    cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")  # float32 matmuls in full precision, no TF32
    try:
        yield
    finally:
        cudnn.benchmark, cudnn.allow_tf32 = saved_cudnn
        if saved_precision is not None:
            torch.set_float32_matmul_precision(saved_precision)
        elif saved_cublas is not None:
            matmul.fp32_precision = saved_cublas


def _scene_seed(batch: dict[str, Any], index: int) -> int:
    seed = batch.get(OBS_SCENE_SEED)
    if seed is None:
        return 0
    value = seed[index] if isinstance(seed, (Tensor, np.ndarray, list, tuple)) else seed
    if isinstance(value, Tensor):
        value = value.reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return int(round(float(value)))


class _FrozenParts:
    """The autoencoder and the text encoder. A plain holder: not part of the state dict."""

    def __init__(self) -> None:
        self.vae: Autoencoder | None = None
        self.text_encoder: Any | None = None


def _pose8_rows(rows: Tensor) -> np.ndarray:
    """``(T, 7)`` DROID rows or ``(T, 8)`` pose8 rows (from the pose processor step) as ``(T, 8)`` float32."""
    values = rows.detach().float().cpu().numpy()
    if values.shape[-1] == PRAXIS_POSE_DIM:
        return values.astype(np.float32)
    return droid_pose8(values[:, :6], values[:, 6])


def _frames_uint8_thwc(frames: Tensor) -> np.ndarray:
    """Convert a LeRobot frame stack ``(T, 3, H, W)`` (float in [0, 1] or uint8) to ``(T, H, W, 3)`` uint8."""
    if frames.ndim != 4:
        raise ValueError(f"expected a frame stack (T, 3, H, W), got {tuple(frames.shape)}")
    if frames.dtype != torch.uint8:
        frames = (frames.float() * 255.0).round().clamp_(0, 255).to(torch.uint8)
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()


class _TrainCounter(torch.nn.Module):
    """The number of training windows the policy has seen, as a buffer.

    It seeds the noise and dropout draws of `forward`, so it travels with the checkpoint (a buffer
    of the state dict) and with a PEFT adapter (`modules_to_save`), and a resumed run continues
    the sequence of draws instead of repeating it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("calls", torch.zeros((), dtype=torch.int64))


class _TextCache:
    """Embeddings of the frozen text encoder, cached per instruction and per token sequence.

    The caches hold the ``max_entries`` most recently added embeddings each (an evaluation suite
    has a few instructions; a server fed ever new instructions must not grow without bound).
    """

    max_entries = 256

    def __init__(self, policy: PraxisPolicy) -> None:
        self._policy = policy
        self._by_text: dict[str, Tensor] = {}
        self._by_tokens: dict[tuple[int, ...], Tensor] = {}

    @staticmethod
    def _put(cache: dict[K, Tensor], key: K, value: Tensor) -> Tensor:
        if len(cache) >= _TextCache.max_entries:
            del cache[next(iter(cache))]  # dicts keep insertion order: the oldest entry goes
        cache[key] = value
        return value

    def encode_text(self, prompt: str) -> Tensor:
        """The embedding ``(1, S, 2048)`` of an instruction string (the text encoder tokenizes it)."""
        if prompt in self._by_text:
            return self._by_text[prompt]
        return self._put(self._by_text, prompt, self._policy.text_encoder.encode([prompt])[0])

    def encode_tokens(self, token_ids: Tensor) -> Tensor:
        """The embedding of an instruction given as its token ids ``(S,)`` (the tokenizer step's output, unpadded)."""
        key = tuple(int(i) for i in token_ids.reshape(-1).tolist())
        if key in self._by_tokens:
            return self._by_tokens[key]
        embedding = self._policy.text_encoder.encode_token_ids(torch.tensor(key, dtype=torch.int64))
        return self._put(self._by_tokens, key, embedding)

    def encode_uncond(self) -> Tensor:
        """The embedding of the blank caption of the unguided pass."""
        return self.encode_text(self._policy.variant.blank_caption)

    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        """The ``TextEncode`` surface of ``train.objective.encode_window`` (cached per prompt)."""
        return [self.encode_text(prompt) for prompt in prompts]

    def clear(self) -> None:
        self._by_text.clear()
        self._by_tokens.clear()


def _finish_chunk(
    state: SingleArmServeSession | BimanualServeSession,
    inputs: SingleArmChunkInputs | BimanualChunkInputs,
    actions: Tensor,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Decode the sampled chunk and advance the session.

    Returns the normalized action rows and the ``(poses, grips)`` rows of each arm (one pair on
    the single-arm rig, two on the bimanual rig), each session with the inputs it prepared.
    """
    if isinstance(state, BimanualServeSession) and isinstance(inputs, BimanualChunkInputs):
        rows_norm, _rel = state.decode_actions(inputs, actions)
        return rows_norm, state.finish(inputs, actions)
    if isinstance(state, SingleArmServeSession) and isinstance(inputs, SingleArmChunkInputs):
        rows_norm, _rel = state.decode_actions(inputs, actions)
        return rows_norm, [state.finish(inputs, actions)]
    raise TypeError(f"{type(inputs).__name__} is not the input of a {type(state).__name__}")


class _EnvState:
    """The per-environment state: the frame history, the praxis session, the instruction and the queues."""

    def __init__(self, stack_rows: int) -> None:
        # (the frames of each camera in canvas order, the pose row) per tick
        self.history: deque[tuple[tuple[Tensor | np.ndarray, ...], np.ndarray]] = deque(maxlen=stack_rows)
        self.state: SingleArmServeSession | BimanualServeSession | None = None
        self.seed = 0
        self.task = ""
        self.text: Tensor | None = None  # the instruction embedding from the tokenizer step's tokens
        self.actions: deque[np.ndarray] = deque()
        self.frames: deque[np.ndarray] = deque()
        self.scene_seed: int = 0


class PraxisPolicy(PreTrainedPolicy):
    """Praxis-1 in LeRobot: observations in, one 15 Hz Cartesian command per call out."""

    config_class = PraxisConfig
    name = "praxis"
    _fsdp_wrap_modules = ["TransformerBlock", "ActionExpertBlock"]
    # 54 GB of bfloat16 weights, above the Hub's 50 GB file limit: `save_pretrained` writes 5 GB shards
    # (the praxis checkpoint layout; transformers itself shards at 50 GB since v5.5).
    _max_shard_size: ClassVar[int | str] = "5GB"
    _action_queue_attrs: ClassVar[tuple[str, ...]] = ()

    def __init__(self, config: PraxisConfig, **kwargs: Any) -> None:
        super().__init__(config)
        kwargs.pop("dataset_stats", None)
        kwargs.pop("dataset_meta", None)
        config.validate_features()
        self.config = config
        validate_config(config.model_config)

        device = torch.device(config.device or "cpu")
        backend, compiled = resolve_attention_backend(config.attention_backend, device)
        praxis_layers.set_flex_attention_compiled(compiled)
        model_config = copy.deepcopy(config.model_config)
        attention = model_config["backbone"]["attention"]
        attention["backend"] = backend
        attention["flex_block_size"] = mask_block_size(attention["flex_block_size"], backend, device)
        if not compiled:
            # The compiled dynamic flex attention of the expert has no CPU kernel either.
            model_config["action_expert"]["attention"] = "sdpa"
            model_config["action_expert"].pop("flex_kernel_options", None)
        self.attention_backend = backend
        self._backend_device_type = device.type
        self.model = PraxisModel.from_config(model_config)
        if config.dtype is not None:
            _cast_parameters(self.model, config.dtype)

        norm = config.model_config["normalization"]
        self.stats = NormStats(
            action_q01=np.asarray(norm["action_q01"], dtype=np.float32),
            action_q99=np.asarray(norm["action_q99"], dtype=np.float32),
            proprio_mean=np.asarray(norm["proprio_mean"], dtype=np.float32),
            proprio_std=np.asarray(norm["proprio_std"], dtype=np.float32),
        )
        self.variant = model_variant(config.model_config)
        self.sampler_settings = SamplerSettings(
            num_steps=config.num_steps,
            t_shift=config.t_shift,
            cfg_scale=config.cfg_scale,
            cfg_video_scale=config.cfg_video_scale,
        )
        self._frozen = _FrozenParts()
        self._text = _TextCache(self)
        self._stack_rows = config.chunk_size + 1
        self._envs: list[_EnvState] = []
        # The canvas rule of the checkpoint applies to raw frames only. The preprocessor's tiles must
        # not go through it again: the fit rule of an antialiased canvas (`praxis-1-droid`) is not idempotent.
        self._tile_hw: tuple[int, int] = (int(self.variant.tile_size[0]), int(self.variant.tile_size[1]))
        self._antialias = bool(self.variant.antialias)
        self._camera_keys: list[str] = list(config.image_keys)
        self._n_arms = int(config.n_arms)
        # The training objective shares `self.model`; it is kept out of the module tree so the
        # state dict and the FSDP units stay those of the policy.
        object.__setattr__(self, "_finetune", None)
        self.train_counter = _TrainCounter()
        self._hub_kwargs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Frozen parts
    # ------------------------------------------------------------------

    @property
    def vae(self) -> Autoencoder:
        if self._frozen.vae is None:
            self._load_frozen_parts()
        if self._frozen.vae is None:
            raise RuntimeError("the autoencoder did not load")
        return self._frozen.vae

    @vae.setter
    def vae(self, value: Autoencoder) -> None:
        self._frozen.vae = value

    @property
    def text_encoder(self) -> Any:
        if self._frozen.text_encoder is None:
            self._load_frozen_parts()
        return self._frozen.text_encoder

    @text_encoder.setter
    def text_encoder(self, value: Any) -> None:
        self._frozen.text_encoder = value
        self._text.clear()

    def _load_frozen_parts(self) -> None:
        """Load the Wan2.2 autoencoder and the Qwen3.5 text encoder on the model device."""
        require_package("transformers", extra="praxis")
        from .praxis_core.text_encoder import TextEncoder

        device = self.model.device
        if self._frozen.vae is None:
            path = _resolve_weight_file(self.config.vae_id, "Wan2.2_VAE.pth", **self._hub_kwargs)
            self._frozen.vae = Autoencoder.from_pretrained(path, device=device)
        if self._frozen.text_encoder is None:
            path = _resolve_model_dir(self.config.text_encoder_id, **self._hub_kwargs)
            self._frozen.text_encoder = TextEncoder(
                path, device=device, pad_to_multiple_of=self.config.text_pad_to_multiple_of
            )
            self._text.clear()

    def to(self, *args: Any, **kwargs: Any) -> PraxisPolicy:
        out = super().to(*args, **kwargs)
        device = self.model.device
        if device.type != self._backend_device_type:
            self._apply_attention_backend(device)
        if self._frozen.vae is not None:
            self._frozen.vae.to(device)
        if self._frozen.text_encoder is not None and hasattr(self._frozen.text_encoder, "model"):
            self._frozen.text_encoder.model.to(device)
            self._frozen.text_encoder.device = device
        self._text.clear()
        return out

    def _apply_attention_backend(self, device: torch.device) -> None:
        """Select the attention kernels of ``device`` on the built model (`resolve_attention_backend`).

        The constructor builds the model for ``config.device``. `to` calls this when the policy
        moves to another kind of device, so a policy built for the CPU (a checkpoint converted on
        a host without a GPU, say) runs the checkpoint's compiled flex attention once on a GPU,
        and a policy built for a GPU runs the dense CPU attention on the CPU. The compiled or
        dense flex attention is a process-wide switch: it follows the policy moved last.
        """
        backend, compiled = resolve_attention_backend(self.config.attention_backend, device)
        praxis_layers.set_flex_attention_compiled(compiled)
        for module in self.model.modules():
            if isinstance(module, praxis_layers.Attention):
                module.set_backend(backend)
        block = mask_block_size(
            self.config.model_config["backbone"]["attention"]["flex_block_size"], backend, device
        )
        self.model.config["backbone"]["attention"]["flex_block_size"] = block
        for tower in (self.model.backbone.encoder, self.model.backbone.decoder):
            if tower._tf_masks.block_size != block:
                tower._tf_masks = _TeacherForcingMaskCache(block)
        expert_config = self.config.model_config["action_expert"]
        expert = self.model.action_expert
        expert.attention = expert_config.get("attention", "sdpa") if compiled else "sdpa"
        expert.flex_kernel_options = None
        if expert.attention == "flex_dynamic":
            options = expert_config.get("flex_kernel_options")
            expert.flex_kernel_options = dict(FLEX_DYNAMIC_KERNEL_OPTIONS if options is None else options)
        self.attention_backend = backend
        self._backend_device_type = device.type

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

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
        strict: bool = False,
        load_frozen_parts: bool = True,
        **kwargs: Any,
    ) -> PraxisPolicy:
        """Load a converted praxis checkpoint: a sharded safetensors directory or Hub repo.

        The weights (54 GB in bfloat16) are read shard by shard onto the device of
        the config, into a model built on the meta device. The frozen autoencoder
        and text encoder are loaded from ``vae_id`` and ``text_encoder_id`` unless
        ``load_frozen_parts`` is False (weight-only tools such as the adapter merge). Every
        tensor of the model must be in the checkpoint (a tensor left on the meta device
        has no value); ``strict`` also refuses tensors the model does not have, which
        are otherwise logged and skipped.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        if not isinstance(config, PraxisConfig):
            raise TypeError(f"PraxisPolicy needs a PraxisConfig, got {type(config).__name__}")
        weight_files = cls._resolve_weight_files(
            str(pretrained_name_or_path),
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
        )
        # "cuda" means the current device of the rank (safetensors maps the bare string to cuda:0)
        device = torch.device(resolve_safetensors_device(config.device or "cpu"))
        with torch.device("meta"):
            policy = cls(config)
        policy._hub_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "token": token,
            "local_files_only": local_files_only,
        }
        cls._load_weights(policy, weight_files, device, strict=strict)
        policy.eval()
        if load_frozen_parts:
            policy._load_frozen_parts()
        return policy

    @classmethod
    def _load_weights(
        cls, policy: PraxisPolicy, files: list[Path], device: torch.device, strict: bool = True
    ) -> None:
        """Fill a meta-built policy from its weight files, shard by shard, onto `device`.

        Two layouts load: the converter's, whose tensor names are the praxis checkpoint's
        (``backbone.…``, ``action_expert.…``), and `save_pretrained`'s, whose names carry the
        policy's ``model.`` prefix. Every tensor of the model must be present; with ``strict``
        the checkpoint may not hold others (without it they are logged and skipped). The weights
        take ``config.dtype``; the buffers keep the checkpoint's dtype.

        Args:
            policy (PraxisPolicy): A policy built under ``torch.device("meta")``.
            files (list[Path]): One safetensors file or the shards of an index.
            device (torch.device): Where the weights go.
            strict (bool): Refuse the tensors the model does not have.
        """
        with safe_open(str(files[0]), framework="pt") as handle:
            first_key = next(iter(handle.keys()))
        target: torch.nn.Module = policy if first_key.startswith("model.") else policy.model
        # The checkpoint's dtype for the buffers (the float32 routing biases), the config's for the
        # weights: `assign=True` would otherwise keep the stored dtype whatever `config.dtype` says.
        dtype = policy.config.dtype
        parameter_names = {name for name, _ in target.named_parameters()}
        unexpected: list[str] = []
        for file in files:
            state = load_safetensors_file(str(file), device=str(device))
            if dtype is not None:
                for name, tensor in state.items():
                    if name in parameter_names and tensor.is_floating_point() and tensor.dtype != dtype:
                        state[name] = tensor.to(dtype)
            result = target.load_state_dict(state, strict=False, assign=True)
            unexpected.extend(result.unexpected_keys)
            del state
        if policy.train_counter.calls.device.type == "meta":
            # A checkpoint from before the counter, or a converted praxis checkpoint: start at zero.
            policy.train_counter.calls = torch.zeros((), dtype=torch.int64, device=device)
        missing = [name for name, tensor in target.state_dict().items() if tensor.device.type == "meta"]
        if unexpected and not strict and not missing:
            logger.warning(
                "praxis checkpoint: %d tensors the model does not have were skipped (e.g. %s)",
                len(unexpected),
                unexpected[:3],
            )
        elif missing or unexpected:
            raise RuntimeError(
                f"praxis checkpoint does not match the model: {len(missing)} missing tensors "
                f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})"
            )

    # ------------------------------------------------------------------
    # LeRobot policy surface
    # ------------------------------------------------------------------

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        return self._trainable_parameters()

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        """The parameters to train: the ones with gradients. A full fine-tune trains them all
        (`from_pretrained` leaves them trainable); PEFT freezes all but the adapters. Reading
        them does not change the model."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def reset(self) -> None:
        """Start a new episode: drop the frame histories, the sessions and the queues."""
        self._envs = []

    def drop_queued_actions(self) -> None:
        """Drop the commands not yet executed and restart the temporal context of those sessions.

        The model conditions the next chunk on the whole previous chunk having been executed, so a
        session whose chunk is cut short (a new instruction mid-rollout, for example) starts the
        next call cold, at depth 0, like the first call of an episode.
        """
        for env in self._envs:
            if env.actions and env.state is not None:
                env.state.restart()
            env.actions.clear()
            env.frames.clear()

    def count_queued_actions(self) -> int:
        return sum(len(env.actions) for env in self._envs)

    def chunk_inference_spec(self) -> ChunkPolicySpec:
        """Praxis conditions on the executed chunk, so it keeps session state between calls."""
        return ChunkPolicySpec(
            prediction_steps=self.config.chunk_size,
            execution_steps=self.config.n_action_steps,
            current_observation_only=False,
            retains_session_state=True,
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Return the fine-tuning loss of a batch of training windows and the objective's metrics.

        Each sample of the batch is one praxis training window: `window_frames` (93) consecutive
        frames of both cameras, the observed flange pose and the commanded pose of every frame,
        and the task string. `observation_delta_indices` and `action_delta_indices` make
        `LeRobotDataset` deliver them (`(batch, 93, ...)`), and `drop_n_last_frames` keeps the
        windows inside the episodes.

        The windows go through the praxis objective one by one (`praxis_core.train.objective`):
        the canvas tiles and the relative actions are built as the praxis data module builds
        them, the frozen autoencoder and text encoder encode them, and the loss is the joint
        flow-matching loss of the video and the actions plus the MoE sequence-balance term. The
        random draws of a window come from a CPU generator seeded by `seed`, the number of
        training `forward` calls so far, the rank and the position in the batch, so a run is
        reproducible; an evaluation call (`eval()`) draws from the current count without
        advancing it.

        Args:
            batch (dict[str, Tensor]): The training batch with the two camera stacks
                `(B, 93, 3, H, W)`, `observation.state` and `action` `(B, 93, 7)`, and `task`.

        Returns:
            tuple[Tensor, dict | None]: The mean loss over the windows, and the mean metrics
            (`loss_video`, `loss_action`, `loss_moe_balance`, `action_dropped`) as floats.
        """
        finetune = self._finetune_model()
        finetune.train(self.training)
        windows = self._training_windows(batch)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        device = self.model.device
        calls = int(self.train_counter.calls.item())
        losses: list[Tensor] = []
        sums: dict[str, Tensor] = {}
        # The praxis trainer runs on PyTorch defaults without autocast: the router scores in
        # float32, no TF32 in float32 matmuls. The host's autocast (`--accelerator.mixed_precision`)
        # would turn the router GEMM into bfloat16, so it is off here; the model is bfloat16 already.
        with _praxis_numerics(), torch.autocast(device_type=device.type, enabled=False):
            for index, window in enumerate(windows):
                encoded = encode_window(window, self.vae, self._text, self.config.model_config, device)
                seed = derive_seed(self.config.seed, calls, rank, index)
                loss, metrics = finetune(encoded, torch.Generator().manual_seed(seed))
                losses.append(loss)
                for key, value in metrics.items():
                    part = value.detach().float()
                    sums[key] = sums[key] + part if key in sums else part
        if self.training:
            # Training calls only: a held-out evaluation draws from the current count and leaves
            # the training sequence where it was.
            self.train_counter.calls.add_(1)
        total = torch.stack(losses).mean()
        return total, {key: float(value / len(windows)) for key, value in sums.items()}

    # LoRA fine-tuning through PEFT. The default targets are the praxis `expert` group: the
    # attention and MLP projections of the action expert (`praxis/train/lora.py` LORA_TARGETS).
    # The backbone attention output projections (`backbone_attention` group) are
    # `.*backbone\.(encoder|decoder)\.core\.blocks\.\d+\.mha\.out_proj`; pass them with
    # `--peft.target_modules`. The MoE expert weights are not nn.Linear and cannot take adapters.
    PEFT_TARGETS: ClassVar[dict[str, str]] = {
        "expert": r".*action_expert\.blocks\.\d+\.(to_q|to_kv|to_out|mlp_in|mlp_out)",
        "backbone_attention": r".*backbone\.(encoder|decoder)\.core\.blocks\.\d+\.mha\.out_proj",
    }

    # The training counter is a module of its own so a PEFT adapter saves it (`modules_to_save`).
    PEFT_SAVED_MODULES: ClassVar[tuple[str, ...]] = ("train_counter",)
    # The head modules of a rig (the six tensors the base checkpoint lacks). A fine-tune from the base
    # names them in `--peft.full_training_modules` with the counter: they train in full next to the
    # adapters, and the merge tool keeps the trained heads.
    PEFT_HEAD_MODULES: ClassVar[tuple[str, ...]] = (
        "action_expert.action_encoder.w1",
        "action_expert.embed_proprio",
        "action_expert.head",
    )

    def _get_default_peft_targets(self) -> dict[str, Any]:
        return {
            "target_modules": self.PEFT_TARGETS["expert"],
            "modules_to_save": list(self.PEFT_SAVED_MODULES),
        }

    def _validate_peft_config(self, peft_config) -> None:  # noqa: ANN001 (a peft config class)
        super()._validate_peft_config(peft_config)
        saved = set(getattr(peft_config, "modules_to_save", None) or ())
        extra = saved - set(self.PEFT_SAVED_MODULES) - set(self.PEFT_HEAD_MODULES)
        if extra:
            raise ValueError(
                "`full_training_modules` / `modules_to_save` can hold the training counter and the rig heads "
                f"{list(self.PEFT_SAVED_MODULES + self.PEFT_HEAD_MODULES)} (the heads for a fine-tune from "
                f"the base checkpoint), got {sorted(extra)}"
            )
        missing = set(self.PEFT_SAVED_MODULES) - saved
        if missing:
            raise ValueError(
                f"`modules_to_save` must hold {sorted(missing)}: the training counter travels with the "
                "adapter (leave `--peft.full_training_modules` unset to take the default)"
            )

    def _peft_active(self) -> bool:
        """Whether PEFT adapters are injected into the model (an adapter fine-tune)."""
        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
        except ImportError:  # peft is optional; without it no adapter can be injected
            return False
        return any(isinstance(module, BaseTunerLayer) for module in self.model.modules())

    def update(self) -> None:
        """After each optimizer step: move the MoE routing biases towards balanced expert loads.

        The training loop calls this once per optimizer step on every rank; the expert counts
        of the step are summed over the ranks inside. An adapter fine-tune (PEFT) leaves the
        biases as they are: they belong to the base model and an adapter checkpoint would not
        carry their updates (the praxis trainer saves full checkpoints for its LoRA runs).
        """
        if self._finetune is None:
            return
        if self._peft_active():
            if not getattr(self, "_logged_frozen_biases", False):
                logger.info("praxis: adapter fine-tune, the MoE routing biases of the base model stay fixed")
                object.__setattr__(self, "_logged_frozen_biases", True)
            return
        update_expert_biases(self.model, self.config.moe_balance_speed)

    def _finetune_model(self) -> FinetuneModel:
        """The praxis objective over `self.model`, built on the first training call."""
        if self._finetune is None:
            finetune = FinetuneModel(
                self.model, self.config.objective_settings(), text_uncond=self._text.encode_uncond()
            )
            if self.config.activation_checkpointing:
                for block in finetune.blocks():
                    block.forward = functools.partial(checkpoint, block.forward, use_reentrant=False)
            object.__setattr__(self, "_finetune", finetune)
        return self._finetune

    def _training_windows(self, batch: dict[str, Any]) -> list[TrainingWindow]:
        """Build one praxis training window per sample of the batch (the praxis data recipe)."""
        cfg = self.config
        if cfg.n_arms != 1:
            raise NotImplementedError(
                "fine-tuning is implemented for the single-arm rig; the praxis repository has no "
                "bimanual training recipe to follow yet"
            )
        frames_t = cfg.window_frames
        exterior_key, wrist_key = cfg.image_keys
        exterior, wrist = batch[exterior_key], batch[wrist_key]
        state, action = batch[OBS_STATE], batch[ACTION]
        for key, value, ndim in (
            (exterior_key, exterior, 5),
            (wrist_key, wrist, 5),
            (OBS_STATE, state, 3),
            (ACTION, action, 3),
        ):
            if ndim == 3 and value.shape[-1] not in (PRAXIS_ACTION_DIM, PRAXIS_POSE_DIM):
                raise ValueError(
                    f"{key}: rows must have {PRAXIS_ACTION_DIM} (DROID) or {PRAXIS_POSE_DIM} (pose8) values"
                )
            if value.ndim != ndim or value.shape[1] != frames_t:
                raise ValueError(
                    f"{key}: expected (batch, {frames_t}, ...) for a training window, got {tuple(value.shape)}. "
                    "The dataset must deliver observation_delta_indices / action_delta_indices frames."
                )
            pad = batch.get(f"{key}_is_pad")
            if pad is not None and bool(torch.as_tensor(pad).any()):
                raise ValueError(
                    f"{key}: a window crosses an episode end; sample with drop_n_last_frames={cfg.drop_n_last_frames}"
                )
        tasks = batch.get("task")
        if tasks is None or len(tasks) != exterior.shape[0]:
            raise ValueError("the training batch needs one 'task' string per sample")
        model_config = cfg.model_config
        chunking = model_config["chunking"]
        geometry = contiguous_chunk_geometry(int(chunking["action_chunk"]), int(chunking["n_chunks"]))
        tile_h, canvas_w = canvas_size(model_config)
        tile = (tile_h, canvas_w // 2)
        aspect_tol = float(model_config.get("video", {}).get("aspect_tol", ASPECT_TOL))
        # The canvas rule per view: a view delivered as tiles already passes through.
        antialias = (self._canvas_antialias(exterior.shape[-2:]), self._canvas_antialias(wrist.shape[-2:]))
        starts = [offset for offset, _ in geometry]
        windows: list[TrainingWindow] = []
        for b in range(exterior.shape[0]):
            clip = tile_clip(
                _frames_uint8_thwc(exterior[b]),
                _frames_uint8_thwc(wrist[b]),
                tile[0],
                tile[1],
                aspect_tol,
                antialias=antialias,
            )
            observed = _pose8_rows(state[b])
            commanded = _pose8_rows(action[b])
            actions_rel = np.concatenate(
                [pose8_to_10d(commanded[off : off + ln], anchor8=observed[off]) for off, ln in geometry],
                axis=0,
            )
            windows.append(
                TrainingWindow(
                    frames=clip[0].permute(1, 0, 2, 3).to(torch.uint8).contiguous(),
                    prompt=str(tasks[b]),
                    actions_rel=actions_rel,
                    proprio_abs=pose8_to_10d(observed[starts]),
                    meta={"batch_index": b},
                )
            )
        return windows

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return the action chunk ``(B, chunk_size, 13)`` of the current observations, on the CPU.

        The first call of a context window gives 21 commands (its first latent is the window's
        uncompressed anchor frame, which gives one command instead of four); the chunk is padded with
        its last command to ``chunk_size`` rows, the fixed ``[B, chunk_size, A]`` shape of
        LeRobot's chunk contract, which holds the last pose for the last 3 steps. The later
        calls of the window give 24 commands, the full chunk. The call advances the sessions like
        ``select_action`` does, without queueing the commands. The next chunk is conditioned
        on the executed frames only when the call receives them: pass the frames executed since
        the previous call as a stack ``(B, T, 3, H, W)`` with their states ``(B, T, 7 | 8)``
        (oldest first); a call with one frame after a chunk starts the context again.
        """
        self.eval()
        self._push_observations(batch)
        rows = []
        for env in self._envs:
            actions, _frames = self._sample_chunk(env)
            if len(actions) < self.config.chunk_size:
                actions = actions + [actions[-1]] * (self.config.chunk_size - len(actions))
            rows.append(np.stack(actions[: self.config.chunk_size]))
        return torch.from_numpy(np.stack(rows)).to(torch.float32)

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor | PolicyOutput:
        """Return one command per environment, sampling a new chunk when the queue is empty.

        The commands are ``(B, 13)`` float32 on the CPU: the praxis wire pose and the gripper
        (the command processor step converts them to the DROID 7 values). They are numpy rows
        of the session, so nothing is uploaded to the device for them.
        """
        self.eval()
        self._push_observations(batch)
        out: list[np.ndarray] = []
        frames: list[np.ndarray | None] = []
        for env in self._envs:
            if not env.actions:
                actions, predicted = self._sample_chunk(env)
                env.actions.extend(actions)
                env.frames.clear()
                if predicted is not None:
                    env.frames.extend(predicted)
            out.append(env.actions.popleft())
            frames.append(env.frames.popleft() if env.frames else None)
        action = torch.from_numpy(np.stack(out)).to(torch.float32)
        if not self.config.return_prediction or any(f is None for f in frames):
            return action
        canvas = torch.from_numpy(np.stack(frames))  # (B, 3, tile_h, n_views * tile_w) uint8
        width = canvas.shape[-1] // len(self._camera_keys)
        prediction = {
            "observation": {
                key: canvas[..., v * width : (v + 1) * width] for v, key in enumerate(self._camera_keys)
            }
        }
        return {ACTION: action, PREDICTION: prediction}

    # ------------------------------------------------------------------
    # Session mechanics
    # ------------------------------------------------------------------

    def _canvas_antialias(self, frame_hw: tuple[int, int] | torch.Size) -> bool:
        """Whether frames of this size take the antialiased canvas rule (never for tiles)."""
        return self._antialias and tuple(int(x) for x in frame_hw) != self._tile_hw

    def _new_session(self) -> SingleArmServeSession | BimanualServeSession:
        """The praxis temporal context of one environment, on the model's parts."""
        chunking = self.config.model_config["chunking"]
        if self._n_arms == 2:
            return BimanualServeSession(
                encode=self.vae.encode,
                stats=self.stats,
                device=self.model.device,
                prompt="",
                max_context_chunks=self.config.max_context_chunks,
                fps=float(chunking.get("grid_hz", FPS)),
                dims=bimanual.q99_dims(self.config.model_config),
                control_mode="none",
                exec_horizon=self.config.n_action_steps,
                n_chunks=int(chunking.get("n_chunks", N_CHUNKS)),
                tile_size=self.variant.tile_size,
                aspect_tol=self.variant.aspect_tol,
                antialias=self.variant.antialias,
            )
        return SingleArmServeSession(
            encode=self.vae.encode,
            stats=self.stats,
            device=self.model.device,
            prompt="",
            max_context_chunks=self.config.max_context_chunks,
            fps=float(chunking.get("grid_hz", FPS)),
            dims=q99_dims(self.config.model_config),
            exec_horizon=self.config.n_action_steps,
            n_chunks=int(chunking.get("n_chunks", N_CHUNKS)),
            tile_size=self.variant.tile_size,
            aspect_tol=self.variant.aspect_tol,
            antialias=self.variant.antialias,
        )

    def _push_observations(self, batch: dict[str, Tensor]) -> None:
        """Append the observations of the call to the frame history of every environment.

        The cameras hold one frame per environment ``(B, 3, H, W)``, or a stack ``(B, T, 3, H, W)``
        of the ``T`` observations since the previous call, oldest first, with the state of every
        frame ``(B, T, 7 | 8)``: a caller that executes whole chunks hands over the executed frames
        this way, and the next chunk is conditioned on them as in a call-per-frame rollout.
        The tiles of the preprocessor stay on their device (the autoencoder reads them there);
        raw frames of a policy called without the pipeline become CPU bytes for the canvas rule.
        The instruction comes as the tokenizer step's token ids, or as the ``task`` string.
        """
        cameras = [batch[key] for key in self._camera_keys]
        first = cameras[0]
        batch_size = int(first.shape[0])
        n_frames = int(first.shape[1]) if first.ndim == 5 else 1
        for key, cam in zip(self._camera_keys, cameras, strict=True):
            if (
                cam.ndim != first.ndim
                or cam.shape[0] != batch_size
                or (cam.ndim == 5 and int(cam.shape[1]) != n_frames)
            ):
                raise ValueError(
                    f"the cameras must hold the same frames, got {tuple(first.shape)} for "
                    f"{self._camera_keys[0]} and {tuple(cam.shape)} for {key}"
                )
        while len(self._envs) < batch_size:
            self._envs.append(_EnvState(self._stack_rows))
        if len(self._envs) > batch_size:
            raise ValueError(
                f"the batch has {batch_size} environments but the policy holds state for "
                f"{len(self._envs)}; call reset() when the batch changes"
            )
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)
        if isinstance(tokens, Tensor) and (tokens.ndim != 2 or tokens.shape[0] not in (1, batch_size)):
            raise ValueError(
                f"{OBS_LANGUAGE_TOKENS} must be (rows, length) with one row per environment or one row "
                f"for all {batch_size}, got {tuple(tokens.shape)}"
            )
        for i, env in enumerate(self._envs):
            poses = _state_rows(batch, i, n_frames, self._n_arms)
            for t in range(n_frames):
                frames = tuple(cam[i, t] if cam.ndim == 5 else cam[i] for cam in cameras)
                env.history.append((tuple(_frame_for_history(f, self._tile_hw) for f in frames), poses[t]))
            env.task = _task_of(batch, i)
            if isinstance(tokens, Tensor):
                # One instruction for the whole batch comes as one row (a single task string).
                row_index = i if tokens.shape[0] > 1 else 0
                ids = tokens[row_index]
                if isinstance(masks, Tensor):
                    ids = ids[masks[row_index].bool()]
                with _praxis_numerics():  # the text encoder runs with the praxis numerics too
                    env.text = self._text.encode_tokens(ids.cpu())
            if env.state is None:
                env.scene_seed = _scene_seed(batch, i)
                env.seed = self.config.seed + self.config.seed_stride * env.scene_seed
                env.state = self._new_session()

    def _observation_dict(self, env: _EnvState) -> dict[str, Any]:
        """Build the praxis observation of one environment from its frame history."""
        n_views = len(self._camera_keys)
        stacks: list[Tensor | np.ndarray] = []
        for v in range(n_views):
            frames = [row[0][v] for row in env.history]
            # (T, 3, H, W) tiles from the preprocessor, or (T, H, W, 3) raw frames
            stacks.append(torch.stack(frames) if isinstance(frames[0], Tensor) else np.stack(frames))
        poses = np.stack([row[1] for row in env.history]).astype(np.float32)
        obs: dict[str, Any] = {}
        if self._n_arms == 1:
            obs["image.exterior"], obs["image.wrist"] = stacks
            obs["robot_state.ee_pose"] = poses[:, :7]
            obs["grip"] = poses[:, 7:8]
        else:
            for view, stack in zip(bimanual.VIEWS, stacks, strict=True):
                obs[bimanual.VIEW_KEY_CANDIDATES[view][0]] = stack
            for i, side in enumerate(bimanual.SIDES):
                arm = poses[:, 8 * i : 8 * i + 8]
                obs[bimanual.pose_keys(side)[0]] = arm[:, :7]
                obs[bimanual.grip_keys(side)[0]] = arm[:, 7]
        if env.task:
            obs["task"] = env.task
        return obs

    def _sample_chunk(self, env: _EnvState) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
        """Run one praxis call for one environment. Return the command rows (and the predicted frames).

        A command row is the praxis wire pose ``[x, y, z, rotation (9, row-major)]`` followed by
        the gripper (``PRAXIS_COMMAND_DIM`` values); the command processor step turns it into the
        DROID ``[x, y, z, roll, pitch, yaw, gripper]``. The steps are those of the praxis
        session: prepare the window (the executed frames and the current frame through the
        autoencoder, the proprio), sample the chunk with the instruction embedding and the blank
        caption of the unguided pass, decode the actions against the anchor pose.
        """
        state = env.state
        if state is None:
            raise RuntimeError("the environment has no praxis session; `reset()` runs first")
        with _praxis_numerics():
            inputs = state.prepare(self._observation_dict(env))
            text = env.text if env.text is not None else self._text.encode_text(inputs.prompt)
            uncond = self._text.encode_uncond() if self.sampler_settings.guided else None
            video, actions = self.model.sample_window_joint(
                text=text,
                text_uncond=uncond,
                anchor=inputs.anchor,
                cleans=inputs.cleans,
                proprio_all=inputs.proprio_all,
                seed=env.seed + inputs.chunk_index,
                settings=self.sampler_settings,
            )
            rows_norm, arms = _finish_chunk(state, inputs, actions)  # one (poses, grips) per arm
            commands = [
                np.concatenate([np.concatenate([poses[t], [grips[t]]]) for poses, grips in arms]).astype(
                    np.float32
                )
                for t in range(len(arms[0][0]))
            ]
            predicted: list[np.ndarray] | None = None
            if self.config.return_prediction:
                latent = video
                if inputs.cleans:
                    latent = torch.cat(
                        [*inputs.cleans, latent.to(inputs.cleans[0].device, inputs.cleans[0].dtype)], dim=2
                    )
                frames = self.vae.decode(latent)  # one frame per command row, the context's rows in front
                predicted = list(frames[-len(rows_norm) :][: len(commands)].cpu().numpy())
        return commands, predicted

    def sessions(self) -> Sequence[_EnvState]:
        """The per-environment states of the current batch (for tests and diagnostics)."""
        return list(self._envs)
