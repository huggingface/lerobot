# Vendored from the praxis repository (main, with its base-checkpoint additions).
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is PraxisModel here. Keep the file in sync with the praxis repository.
"""Load praxis weights from a directory of safetensors shards.

A checkpoint directory contains these files:

- ``config.json``: the architecture hyperparameters.
- ``model.safetensors.index.json``: the map from tensor name to shard file.
- ``model-XXXXX-of-NNNNN.safetensors``: the shards.
- ``SHA256SUMS``: the checksums of the shards, and optionally of
  ``config.json`` and the index (optional).

A directory with one ``model.safetensors`` file and no index is also valid.

The tensor names start with ``backbone.encoder.``, ``backbone.decoder.`` or
``action_expert.``. All tensors are bfloat16, except the MoE routing biases
(``*.ffn.expert_bias``), which are float32. The loaders keep the stored dtype.

``load_config`` checks the config with ``validate_config``. Format version 2
loads for two rigs: ``"embodiment": "single_arm"`` (the DROID checkpoint) and
``"embodiment": "bimanual"`` (the YAM checkpoint). Both rigs run the same model
variant; only the expert head widths, the views and the normalization differ.
A config of another embodiment raises ``CheckpointError``
(``OTHER_RIG_REFUSED``), and so does a config of another model variant
(``OTHER_VARIANT_REFUSED``). The variant fields below are optional.
A missing field takes the value of the checkpoint.
``model_variant`` returns the resolved values of these fields:

- ``action_expert.absolute_positions`` (``true``): the ``future_chunk_embed``
  table. Only ``true`` is supported.
- ``action_expert.action_pos_table`` (default: ``absolute_positions``): the
  absolute action position table. Only ``false`` is supported.
- ``action_expert.future_chunk_offset``: ``"zero"`` reads the
  ``future_chunk_embed`` rows from row 0. It is the only value.
- ``action_expert.rope``: the expert RoPE on the trunk position ids, applied
  before the query and key norms (``EXPERT_TRUNK_ROPE``, the only value).
- ``action_expert.global_order`` (``["register", "anchor", "text"]``): the
  order of the global tokens. It must match the backbone layout.
- ``action_expert.attention`` (``"sdpa"``): ``"flex_dynamic"`` is compiled
  flex attention with dynamic shapes and ``flex_kernel_options``.
- ``action_expert.head_plus_zero`` (``false``): adds ``+0.0`` to the head.
- ``action_expert.layout_stamp`` (``null``): the token layout of the expert.
  The only known stamp is ``LAYOUT_STAMP_CHUNK_CAUSAL``.
- ``backbone.layout`` (``"chunk_causal"``): the training window with clean
  copies of the earlier chunks (teacher forcing). It is the only value.
- ``backbone.attention.backend`` (``"triton"``): the flex attention backend.
- ``backbone.moe.dispatch_chunk_size`` (``null``): the token chunk of the
  MoE dispatch, with the routed sum in float32.
- ``backbone.rope.ref_placement`` (``"group"``): the anchor in position block
  0 and the text in block 1. It is the only value.
- ``video.canvas`` (``"tiles"``): each view resized to one tile
  (``tile_height`` x ``tile_width``, centre crop beyond ``aspect_tol``). It
  is the only value. ``video.antialias`` (``false``) turns on the antialias
  of the bilinear resize, as in the training of the checkpoint.
- ``video.vae_encode_impl`` (``"megakernel"``): the only value, with the
  optional ``vae_megakernel_chunk_frames``.
- ``text_encoder.uncond`` (``"blank_caption"``): encodes ``blank_caption``
  for the unconditional pass. It is the only value.
- ``pose_frame``: the frame of the arm poses of the model. ``"flange"`` (the
  single-arm rig): the Franka flange pose with no turn; the session converts
  the pose of each client to it. ``"rig"`` (the bimanual rig): the poses as
  the rig reports them, in its world frame, with no conversion. Each rig
  takes only its own value; a missing field takes it.

An unknown key or value in these sections raises ``CheckpointError``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

__all__ = [
    "BASE_EMBODIMENT",
    "BASE_HEAD_TENSORS",
    "BASE_REFUSED",
    "EXPERT_TRUNK_ROPE",
    "LAYOUT_STAMP_CHUNK_CAUSAL",
    "CheckpointError",
    "ModelVariant",
    "expected_dtype",
    "load_base_config",
    "load_checkpoint",
    "load_config",
    "load_state_dict_from_dir",
    "model_variant",
    "save_checkpoint",
    "strict_load",
    "validate_base_config",
    "validate_config",
    "verify_checksums",
    "weight_map",
]

CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
CHECKSUM_NAME = "SHA256SUMS"

# The largest shard that save_checkpoint writes (bytes of tensor data).
MAX_SHARD_BYTES = 5 * 1000**3

_MAX_LISTED = 20


class CheckpointError(RuntimeError):
    """A checkpoint does not agree with its own files or with a module."""


def _listing(title: str, names: list[str]) -> str:
    head = ", ".join(names[:_MAX_LISTED])
    more = f" (+{len(names) - _MAX_LISTED} more)" if len(names) > _MAX_LISTED else ""
    return f"{title} [{len(names)}]: {head}{more}"


def load_config(path: str | Path) -> dict[str, object]:
    """Read ``config.json`` from a checkpoint directory and validate it.

    A base checkpoint (``"embodiment": "base"``) raises ``CheckpointError``
    (``BASE_REFUSED``): it has no action heads. Use ``load_base_config`` to
    read it, and ``examples/praxis/init_heads.py`` to make a policy that loads.
    """
    config = json.loads((Path(path) / CONFIG_NAME).read_text())
    if config.get("embodiment") == BASE_EMBODIMENT:
        raise CheckpointError(BASE_REFUSED)
    validate_config(config)
    return config


def load_base_config(path: str | Path) -> dict[str, object]:
    """Read and validate the ``config.json`` of a base checkpoint.

    A base checkpoint holds the backbone and the action expert without the
    tensors of one rig (``BASE_HEAD_TENSORS``). Its config has
    ``"embodiment": "base"`` and no ``action_space``, ``normalization``,
    ``pose_frame``, patch dims or camera canvas. Raises ``CheckpointError``
    for a config of another embodiment.
    """
    config = json.loads((Path(path) / CONFIG_NAME).read_text())
    validate_base_config(config)
    return config


# ---------------------------------------------------------------------------
# Config fields and validation
# ---------------------------------------------------------------------------

FORMAT_VERSION = 2
#: Action rows per action token (one token per latent frame).
FRAMES_PER_TOKEN = 4

#: The expert RoPE of the checkpoint. It is the only supported value.
EXPERT_TRUNK_ROPE: dict[str, object] = {
    "kind": "trunk",
    "dim": 64,
    "effective_dim": 64,
    "base": 26_000_000,
    "table_size": 1_000_000,
    "order": "before_norm",
}

#: The layout stamp of the expert: per-embodiment tables, the trunk
#: RoPE, the global order register, anchor, text and the chunk-causal window.
LAYOUT_STAMP_CHUNK_CAUSAL = 20260924

GLOBAL_ORDER_CHUNK_CAUSAL = ("register", "anchor", "text")
FUTURE_CHUNK_OFFSETS = ("zero",)
EXPERT_ATTENTIONS = ("sdpa", "flex_dynamic")
FLEX_KERNEL_OPTION_KEYS = ("fwd_BLOCK_M", "fwd_BLOCK_N", "num_warps", "num_stages")
BACKBONE_LAYOUTS = ("chunk_causal",)
ATTENTION_BACKENDS = ("triton", "flash")
ROPE_REF_PLACEMENTS = ("group",)
CANVASES = ("tiles",)
VAE_ENCODE_IMPLS = ("megakernel",)
TEXT_UNCONDS = ("blank_caption",)
POSE_FRAMES = ("flange", "rig")
#: The pose frame of each rig (``pose_frame`` in the config).
RIG_POSE_FRAMES: dict[str, str] = {"single_arm": "flange", "bimanual": "rig"}
#: The tile canvas of the checkpoint: the views, the tile (height,
#: width) and the aspect tolerance of the centre crop.
VIEWS = ("exterior", "wrist")
TILE_SIZE = (256, 416)
ASPECT_TOL = 0.03

#: The rigs of format version 2 and the views of each rig, in canvas order.
EMBODIMENTS = ("single_arm", "bimanual")
RIG_VIEWS: dict[str, tuple[str, ...]] = {
    "single_arm": VIEWS,
    "bimanual": ("top_left", "wrist_left", "wrist_right"),
}
#: The action row width of each rig (``action_space.dim``).
RIG_ACTION_DIMS: dict[str, int] = {"single_arm": 10, "bimanual": 20}
#: The native camera view (height, width) of each rig. The fit rule of an
#: antialiased canvas resizes this view to the tile width and trims it to the
#: tile: DROID 720x1280 -> 234x416 -> 224x416, YAM 1200x1920 -> 260x416 ->
#: 256x416.
NATIVE_VIEWS: dict[str, tuple[int, int]] = {
    "single_arm": (720, 1280),
    "bimanual": (1200, 1920),
}

#: The error of a config of another embodiment.
OTHER_RIG_REFUSED = (
    'config: only "embodiment": "single_arm" or "bimanual" is supported; use the '
    "single-arm or the bimanual checkpoint"
)

#: The embodiment of a base checkpoint: the backbone and the action expert
#: without the tensors of one rig.
BASE_EMBODIMENT = "base"

#: The tensors of one rig. A base checkpoint has none of them.
#: ``examples/praxis/init_heads.py`` adds them.
BASE_HEAD_TENSORS = (
    "action_expert.action_encoder.w1.weight",
    "action_expert.action_encoder.w1.bias",
    "action_expert.embed_proprio.weight",
    "action_expert.embed_proprio.bias",
    "action_expert.head.weight",
    "action_expert.head.bias",
)

#: The error of ``load_config`` for a base checkpoint.
BASE_REFUSED = (
    'config: "embodiment": "base" is a base checkpoint without action heads; '
    "make a policy of your rig with examples/praxis/init_heads.py "
    "(docs/source/praxis.mdx, 'Fine-tune the base checkpoint')"
)

#: The config sections of one rig. A base config has none of them.
_RIG_KEYS = ("action_space", "normalization", "pose_frame")
_RIG_EXPERT_KEYS = ("action_patch_dim", "proprio_patch_dim")
_RIG_VIDEO_KEYS = (
    "height",
    "width",
    "views",
    "tile_bounds_rows_cols",
    "tile_height",
    "tile_width",
    "antialias",
)

#: The error of a config of another model variant. ``{reason}`` names the
#: field that differs from the checkpoint.
OTHER_VARIANT_REFUSED = (
    "config: this model variant is not supported ({reason}); use the single-arm or the bimanual checkpoint"
)

_TOP_KEYS = frozenset(
    {
        "format_version",
        "model_type",
        "embodiment",
        "torch_dtype",
        "fp32_tensors",
        "backbone",
        "action_expert",
        "chunking",
        "video",
        "text_encoder",
        "action_space",
        "normalization",
        "pose_frame",
        "source",
        "heads",
    }
)
_EXPERT_KEYS = frozenset(
    {
        "dim",
        "num_layers",
        "num_heads",
        "head_dim",
        "trunk_dim",
        "action_patch_dim",
        "proprio_patch_dim",
        "num_future_per_chunk",
        "max_chunks",
        "max_action_tokens",
        "tap_norm",
        "clean_only_conditioning",
        "rope_cross",
        "rope_self",
        "rope_base",
        "absolute_positions",
        "action_pos_table",
        "future_chunk_offset",
        "rope",
        "global_order",
        "attention",
        "flex_kernel_options",
        "head_plus_zero",
        "layout_stamp",
    }
)
_BACKBONE_KEYS = frozenset(
    {
        "dim",
        "timestep_dim",
        "num_registers",
        "modulation_dtype",
        "norm_eps",
        "norm_compute_dtype",
        "attention",
        "rope",
        "moe",
        "dense_ffn_hidden",
        "encoder",
        "decoder",
        "layout",
    }
)
_ATTENTION_KEYS = frozenset(
    {
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "gated",
        "qk_norm_eps",
        "qk_norm_compute_dtype",
        "rope_after_norm",
        "flex_block_size",
        "backend",
    }
)
_ROPE_KEYS = frozenset(
    {
        "base",
        "dim",
        "effective_dim",
        "max_seq_len",
        "efficient_cache_max",
        "max_intra",
        "max_inter",
        "chunked_video_layout",
        "n_chunks",
        "time_scale",
        "n_tiles",
        "ref_placement",
    }
)
_MOE_KEYS = frozenset(
    {
        "num_experts",
        "num_routed_experts",
        "shared_expert",
        "top_k",
        "top_k_routed",
        "expert_hidden",
        "router_activation",
        "router_dtype",
        "routed_scaling_factor",
        "no_expert_slots",
        "routed_weight_layout",
        "dispatch_chunk_size",
    }
)
_VIDEO_KEYS = frozenset(
    {
        "height",
        "width",
        "views",
        "canvas",
        "layout",
        "keep_aspect",
        "antialias",
        "tile_bounds_rows_cols",
        "vae",
        "vae_encode_chunk_frames",
        "latent_channels",
        "vae_stride",
        "patch_size",
        "tile_height",
        "tile_width",
        "aspect_tol",
        "vae_encode_impl",
        "vae_megakernel_chunk_frames",
    }
)
_TEXT_KEYS = frozenset(
    {
        "model",
        "hidden_layer",
        "pad_to_multiple_of",
        "trim_padding",
        "hidden_dim",
        "uncond",
        "blank_caption",
    }
)


class ExpertVariantKwargs(TypedDict, total=False):
    """The ``ActionExpert`` keyword arguments of the variant fields."""

    rope: dict[str, int | float | str]
    global_order: tuple[str, ...]
    attention: str
    flex_kernel_options: dict[str, int]
    head_plus_zero: bool
    layout_stamp: int


@dataclass(frozen=True)
class ModelVariant:
    """The resolved variant fields of a checkpoint config (see the module doc)."""

    expert_rope: Mapping[str, int | float | str]
    global_order: tuple[str, ...]
    expert_attention: str
    flex_kernel_options: Mapping[str, int] | None
    head_plus_zero: bool
    layout_stamp: int | None
    backbone_layout: str
    attention_backend: str
    moe_dispatch_chunk_size: int | None
    rope_ref_placement: str
    canvas: str
    tile_size: tuple[int, int]
    aspect_tol: float
    antialias: bool
    vae_encode_impl: str
    vae_megakernel_chunk_frames: int | None
    text_uncond: str
    blank_caption: str
    pose_frame: str

    def expert_kwargs(self) -> ExpertVariantKwargs:
        """The ``ActionExpert`` keyword arguments of the variant fields."""
        out: ExpertVariantKwargs = {
            "rope": dict(self.expert_rope),
            "global_order": self.global_order,
            "attention": self.expert_attention,
        }
        if self.flex_kernel_options is not None:
            out["flex_kernel_options"] = dict(self.flex_kernel_options)
        out["head_plus_zero"] = self.head_plus_zero
        if self.layout_stamp is not None:
            out["layout_stamp"] = self.layout_stamp
        return out


def _section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise CheckpointError(f"config: {name} must be an object")
    return value


def _check_keys(section: Mapping[str, object], known: frozenset[str], where: str) -> None:
    unknown = sorted(set(section) - known)
    if unknown:
        raise CheckpointError(f"config: unknown keys in {where}: {unknown}")


def _choice(value: object, allowed: tuple[object, ...], where: str) -> object:
    if value not in allowed or isinstance(value, bool):
        raise CheckpointError(f"config: {where} must be one of {allowed}, got {value!r}")
    return value


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise CheckpointError(f"config: {where} must be true or false, got {value!r}")
    return value


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointError(f"config: {where} must be a positive int, got {value!r}")
    return value


def _expert_variant(ex: Mapping[str, object]) -> dict[str, Any]:
    absolute = _bool(ex.get("absolute_positions", True), "action_expert.absolute_positions")
    if not absolute:
        raise CheckpointError(OTHER_RIG_REFUSED)
    pos_table = _bool(ex.get("action_pos_table", absolute), "action_expert.action_pos_table")
    if pos_table:
        raise CheckpointError(OTHER_VARIANT_REFUSED.format(reason="action_expert.action_pos_table"))
    offset = ex.get("future_chunk_offset")
    if offset is None:
        raise CheckpointError(
            OTHER_VARIANT_REFUSED.format(reason="action_expert.future_chunk_offset 'serve'")
        )
    _choice(offset, FUTURE_CHUNK_OFFSETS, "action_expert.future_chunk_offset")
    rope = ex.get("rope")
    if rope is None:
        raise CheckpointError(
            OTHER_VARIANT_REFUSED.format(
                reason="the absolute position tables need action_expert.rope (trunk)"
            )
        )
    if rope != EXPERT_TRUNK_ROPE:
        raise CheckpointError(f"config: action_expert.rope must be {EXPERT_TRUNK_ROPE}, got {rope!r}")
    if ex.get("rope_cross") or ex.get("rope_self"):
        raise CheckpointError("config: action_expert.rope (trunk) excludes rope_cross and rope_self")
    if ex.get("head_dim", EXPERT_TRUNK_ROPE["dim"]) != EXPERT_TRUNK_ROPE["dim"]:
        raise CheckpointError("config: action_expert.rope.dim must equal action_expert.head_dim")
    order_raw = ex.get("global_order", list(GLOBAL_ORDER_CHUNK_CAUSAL))
    if not isinstance(order_raw, list | tuple):
        raise CheckpointError("config: action_expert.global_order must be a list")
    order = tuple(order_raw)
    if order != GLOBAL_ORDER_CHUNK_CAUSAL:
        raise CheckpointError(
            f"config: action_expert.global_order {list(order)} does not match the "
            "global tokens of backbone.layout='chunk_causal': "
            f"{list(GLOBAL_ORDER_CHUNK_CAUSAL)}"
        )
    attention = _choice(ex.get("attention", "sdpa"), EXPERT_ATTENTIONS, "action_expert.attention")
    options = ex.get("flex_kernel_options")
    if attention == "flex_dynamic":
        if not isinstance(options, Mapping) or not options:
            raise CheckpointError(
                "config: action_expert.attention='flex_dynamic' needs action_expert.flex_kernel_options"
            )
        _check_keys(
            options,
            frozenset(FLEX_KERNEL_OPTION_KEYS),
            "action_expert.flex_kernel_options",
        )
        for key, value in options.items():
            _positive_int(value, f"action_expert.flex_kernel_options.{key}")
    elif options is not None:
        raise CheckpointError("config: action_expert.flex_kernel_options needs attention='flex_dynamic'")
    head_plus_zero = _bool(ex.get("head_plus_zero", False), "action_expert.head_plus_zero")
    stamp = ex.get("layout_stamp")
    if stamp is not None and (stamp != LAYOUT_STAMP_CHUNK_CAUSAL or isinstance(stamp, bool)):
        raise CheckpointError(
            f"config: action_expert.layout_stamp {stamp!r} is unknown; this "
            f"version knows null and {LAYOUT_STAMP_CHUNK_CAUSAL}"
        )
    return {
        "expert_rope": dict(EXPERT_TRUNK_ROPE),
        "global_order": order,
        "expert_attention": attention,
        "flex_kernel_options": None if options is None else dict(options),
        "head_plus_zero": head_plus_zero,
        "layout_stamp": stamp,
    }


def _video_variant(video: Mapping[str, object], embodiment: str = "single_arm") -> dict[str, Any]:
    canvas = _choice(video.get("canvas", "tiles"), CANVASES, "video.canvas")
    th = _positive_int(video.get("tile_height", TILE_SIZE[0]), "video.tile_height")
    tw = _positive_int(video.get("tile_width", TILE_SIZE[1]), "video.tile_width")
    tol = video.get("aspect_tol", ASPECT_TOL)
    if isinstance(tol, bool) or not isinstance(tol, int | float):
        raise CheckpointError(f"config: video.aspect_tol must be a number, got {tol!r}")
    if not 0.0 <= float(tol) < 1.0:
        raise CheckpointError(f"config: video.aspect_tol must be in [0, 1), got {tol}")
    views = video.get("views", list(RIG_VIEWS.get(embodiment, VIEWS)))
    if not isinstance(views, list) or not views:
        raise CheckpointError("config: video.canvas='tiles' needs video.views")
    height, width = video.get("height", th), video.get("width", tw * len(views))
    if height != th or width != tw * len(views):
        raise CheckpointError(
            f"config: video tiles of {th}x{tw} for {len(views)} views need a "
            f"{th}x{tw * len(views)} canvas, got "
            f"{height}x{width}"
        )
    if video.get("keep_aspect", True) is not True:
        raise CheckpointError("config: video.canvas='tiles' needs keep_aspect=true")
    antialias = video.get("antialias", False)
    if not isinstance(antialias, bool):
        raise CheckpointError(f"config: video.antialias must be a bool, got {antialias!r}")
    nh, nw = NATIVE_VIEWS.get(embodiment, NATIVE_VIEWS["single_arm"])
    if antialias and th > tw * nh // nw:
        # The fit rule of an antialiased canvas trims the native view resized to
        # tw wide (single_arm.fit_resize_size); a taller tile cannot be cut.
        raise CheckpointError(
            f"config: video.antialias needs tiles no taller than {tw * nh // nw} "
            f"for {tw}-wide tiles of a {nh}x{nw} view, got {th}x{tw}"
        )
    bounds = video.get("tile_bounds_rows_cols")
    if bounds is not None:
        want = {v: [0, th, i * tw, (i + 1) * tw] for i, v in enumerate(views)}
        if bounds != want:
            raise CheckpointError(f"config: video.tile_bounds_rows_cols {bounds} != tiles {want}")
    impl = _choice(
        video.get("vae_encode_impl", "megakernel"),
        VAE_ENCODE_IMPLS,
        "video.vae_encode_impl",
    )
    chunk = video.get("vae_megakernel_chunk_frames")
    if chunk is not None:
        chunk = _positive_int(chunk, "video.vae_megakernel_chunk_frames")
    return {
        "canvas": canvas,
        "tile_size": (th, tw),
        "aspect_tol": float(tol),
        "antialias": antialias,
        "vae_encode_impl": impl,
        "vae_megakernel_chunk_frames": chunk,
    }


def model_variant(config: Mapping[str, object]) -> ModelVariant:
    """Return the resolved variant fields of ``config``.

    A missing field takes its default. A missing section counts as empty. The
    function raises ``CheckpointError`` on an unknown key or value, and on
    fields that do not agree with each other.
    """
    _check_keys(config, _TOP_KEYS, "the top level")
    backbone = _section(config, "backbone")
    _check_keys(backbone, _BACKBONE_KEYS, "backbone")
    attention = _section(backbone, "attention")
    _check_keys(attention, _ATTENTION_KEYS, "backbone.attention")
    rope = _section(backbone, "rope")
    _check_keys(rope, _ROPE_KEYS, "backbone.rope")
    moe = _section(backbone, "moe")
    _check_keys(moe, _MOE_KEYS, "backbone.moe")
    expert = _section(config, "action_expert")
    _check_keys(expert, _EXPERT_KEYS, "action_expert")
    video = _section(config, "video")
    _check_keys(video, _VIDEO_KEYS, "video")
    text = _section(config, "text_encoder")
    _check_keys(text, _TEXT_KEYS, "text_encoder")

    # The expert fields first: they tell a config of another rig apart.
    expert_fields = _expert_variant(expert)
    layout = str(_choice(backbone.get("layout", "chunk_causal"), BACKBONE_LAYOUTS, "backbone.layout"))
    backend = _choice(
        attention.get("backend", "triton"),
        ATTENTION_BACKENDS,
        "backbone.attention.backend",
    )
    chunk = moe.get("dispatch_chunk_size")
    if chunk is not None:
        chunk = _positive_int(chunk, "backbone.moe.dispatch_chunk_size")
    placement = _choice(
        rope.get("ref_placement", "group"),
        ROPE_REF_PLACEMENTS,
        "backbone.rope.ref_placement",
    )
    if rope.get("n_chunks") is not None:
        raise CheckpointError("config: backbone.rope.ref_placement='group' needs rope.n_chunks=null")
    uncond = _choice(text.get("uncond", "blank_caption"), TEXT_UNCONDS, "text_encoder.uncond")
    caption = text.get("blank_caption")
    if not isinstance(caption, str):
        raise CheckpointError("config: text_encoder.uncond='blank_caption' needs a blank_caption string")
    rig_frame = RIG_POSE_FRAMES.get(str(config.get("embodiment")), "flange")
    pose_frame = _choice(config.get("pose_frame", rig_frame), POSE_FRAMES, "pose_frame")
    return ModelVariant(
        **expert_fields,  # ty: ignore[invalid-argument-type]
        backbone_layout=layout,
        attention_backend=str(backend),
        moe_dispatch_chunk_size=chunk,
        rope_ref_placement=str(placement),
        **_video_variant(video, str(config.get("embodiment", "single_arm"))),
        text_uncond=str(uncond),
        blank_caption=caption,
        pose_frame=str(pose_frame),
    )


def _check_normalization(config: Mapping[str, object]) -> None:
    norm = config.get("normalization")
    if norm is None:
        return
    if not isinstance(norm, Mapping):
        raise CheckpointError("config: normalization must be an object")
    space = _section(config, "action_space")
    dim = space.get("dim")
    vectors = ("action_q01", "action_q99", "proprio_mean", "proprio_std")
    values: dict[str, list[float]] = {}
    for key in vectors:
        vec = norm.get(key)
        if not isinstance(vec, list) or (dim is not None and len(vec) != dim):
            raise CheckpointError(
                f"config: normalization.{key} must be a list of action_space.dim={dim} numbers"
            )
        if not all(isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) for x in vec):
            raise CheckpointError(f"config: normalization.{key} has a non-finite value")
        values[key] = [float(x) for x in vec]
    n = len(values["action_q01"])
    dims = norm.get("q99_normed_dims", [])
    if not isinstance(dims, list) or not all(
        isinstance(d, int) and not isinstance(d, bool) and 0 <= d < n for d in dims
    ):
        raise CheckpointError(f"config: normalization.q99_normed_dims must be dims in [0, {n})")
    for d in dims:
        if not values["action_q99"][d] > values["action_q01"][d]:
            raise CheckpointError(f"config: normalization.action_q99[{d}] must exceed action_q01[{d}]")
    if not all(s > 0 for s in values["proprio_std"]):
        raise CheckpointError("config: normalization.proprio_std must be positive")


def _check_single_arm(config: Mapping[str, object], version: object) -> None:
    """Refuse a config of another model variant.

    Both rigs run only this model variant: format version 2,
    the chunk-causal backbone, the trunk RoPE of the expert, the tile canvas
    and the pose frame of the rig (``RIG_POSE_FRAMES``).
    """
    layout = _section(config, "backbone").get("layout", "chunk_causal")
    canvas = _section(config, "video").get("canvas", "tiles")
    rig_frame = RIG_POSE_FRAMES.get(str(config.get("embodiment")), "flange")
    pose_frame = config.get("pose_frame", rig_frame)
    reason = None
    if version != FORMAT_VERSION:
        reason = f"format_version {version!r}"
    elif layout != "chunk_causal":
        reason = f"backbone.layout {layout!r}"
    elif _section(config, "action_expert").get("rope") is None:
        reason = "no action_expert.rope"
    elif canvas != "tiles":
        reason = f"canvas {canvas!r}"
    elif pose_frame != rig_frame:
        reason = f"pose_frame {pose_frame!r} (the {config.get('embodiment')} rig has {rig_frame!r})"
    if reason is not None:
        raise CheckpointError(OTHER_VARIANT_REFUSED.format(reason=reason))


def validate_config(config: Mapping[str, object]) -> None:
    """Check a checkpoint config: the rig (``EMBODIMENTS``), the format
    version, the variant fields (``model_variant``), the views of the rig and
    the normalization vectors. Raises ``CheckpointError``."""
    embodiment = config.get("embodiment")
    if embodiment not in EMBODIMENTS:
        raise CheckpointError(OTHER_RIG_REFUSED)
    version = config.get("format_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version > 2:
        raise CheckpointError(
            f"config: format_version {version!r} is unknown; this version reads {FORMAT_VERSION}"
        )
    if "heads" in config:
        raise CheckpointError("config: only a base config has the section heads")
    _check_single_arm(config, version)
    model_variant(config)
    _check_rig(config, str(embodiment))
    _check_normalization(config)


def _check_rig(config: Mapping[str, object], embodiment: str) -> None:
    """Check the views and the action widths of the rig of ``config``."""
    want = list(RIG_VIEWS[embodiment])
    views = _section(config, "video").get("views", want)
    if views != want:
        raise CheckpointError(f"config: the {embodiment} rig has the views {want}, got {views}")
    dim = _section(config, "action_space").get("dim")
    if dim is None:
        return
    if dim != RIG_ACTION_DIMS[embodiment]:
        raise CheckpointError(
            f"config: the {embodiment} rig has action_space.dim {RIG_ACTION_DIMS[embodiment]}, got {dim}"
        )
    expert = _section(config, "action_expert")
    proprio = expert.get("proprio_patch_dim", dim)
    action = expert.get("action_patch_dim", dim * FRAMES_PER_TOKEN)
    if proprio != dim or action != dim * FRAMES_PER_TOKEN:
        raise CheckpointError(
            f"config: action_space.dim {dim} needs proprio_patch_dim {dim} and "
            f"action_patch_dim {dim * FRAMES_PER_TOKEN}, got {proprio} and {action}"
        )


def validate_base_config(config: Mapping[str, object]) -> None:
    """Check the config of a base checkpoint (``"embodiment": "base"``).

    The model fields must be those of the single-arm checkpoint (format
    version 2 and the variant fields of ``model_variant``). The rig fields
    (``action_space``, ``normalization``, ``pose_frame``, the patch dims of
    the expert, the views and the tile size of the canvas) must be absent.
    Raises ``CheckpointError``.
    """
    if config.get("embodiment") != BASE_EMBODIMENT:
        raise CheckpointError(
            f'config: a base checkpoint needs "embodiment": "{BASE_EMBODIMENT}", '
            f"got {config.get('embodiment')!r}"
        )
    if config.get("format_version") != FORMAT_VERSION:
        raise CheckpointError(
            f"config: a base checkpoint needs format_version {FORMAT_VERSION}, "
            f"got {config.get('format_version')!r}"
        )
    expert = _section(config, "action_expert")
    video = _section(config, "video")
    rig = [k for k in _RIG_KEYS if k in config]
    rig += [f"action_expert.{k}" for k in _RIG_EXPERT_KEYS if k in expert]
    rig += [f"video.{k}" for k in _RIG_VIDEO_KEYS if k in video]
    if rig:
        raise CheckpointError(f"config: a base config has rig fields: {rig}")
    layout = _section(config, "backbone").get("layout", "chunk_causal")
    if layout != "chunk_causal":
        raise CheckpointError(f"config: backbone.layout {layout!r} is unknown")
    model_variant(config)


def expected_dtype(config: Mapping[str, object], name: str) -> torch.dtype:
    """The stored dtype of tensor ``name``: ``float32`` for a name that matches
    ``fp32_tensors``, else ``torch_dtype``."""
    patterns = config.get("fp32_tensors", [])
    if isinstance(patterns, list) and any(fnmatch.fnmatchcase(name, str(p)) for p in patterns):
        return torch.float32
    dtype = getattr(torch, str(config.get("torch_dtype", "bfloat16")), None)
    if not isinstance(dtype, torch.dtype):
        raise CheckpointError(f"config: unknown torch_dtype {config.get('torch_dtype')!r}")
    return dtype


def weight_map(path: str | Path) -> dict[str, str]:
    """Return the map from tensor name to shard file name."""
    root = Path(path)
    index = root / INDEX_NAME
    if index.is_file():
        return dict(json.loads(index.read_text())["weight_map"])
    single = root / SINGLE_NAME
    if single.is_file():
        with safe_open(str(single), framework="pt") as f:
            return dict.fromkeys(f.keys(), SINGLE_NAME)
    raise CheckpointError(f"{root} has no {INDEX_NAME} and no {SINGLE_NAME}")


def load_state_dict_from_dir(
    path: str | Path,
    *,
    prefix: str = "",
    strip_prefix: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load the tensors whose names start with ``prefix``.

    The function reads only the shards that hold such tensors. Each tensor
    keeps its stored dtype. With ``strip_prefix``, the returned names do not
    include ``prefix``. The function raises ``CheckpointError`` when a shard
    and the index do not list the same tensors.
    """
    root = Path(path)
    files: dict[str, list[str]] = {}
    for name, file in weight_map(root).items():
        files.setdefault(file, []).append(name)
    out: dict[str, torch.Tensor] = {}
    cut = len(prefix) if strip_prefix else 0
    for file, names in sorted(files.items()):
        wanted = [n for n in names if n.startswith(prefix)]
        if not wanted:
            continue
        with safe_open(str(root / file), framework="pt", device=str(device)) as f:
            stored = set(f.keys())
            if stored != set(names):
                raise CheckpointError(
                    f"{file} does not hold the tensors that the index lists: "
                    f"{len(stored - set(names))} extra, "
                    f"{len(set(names) - stored)} missing"
                )
            for name in wanted:
                out[name[cut:]] = f.get_tensor(name)
    if not out:
        raise CheckpointError(f"{root} has no tensor with the prefix {prefix!r}")
    return out


def strict_load(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    prefix: str = "",
    assign: bool = False,
) -> None:
    """Load ``state_dict`` into ``module`` and require an exact name match.

    ``prefix`` selects the entries for ``module`` and removes the prefix from
    their names. The function raises ``CheckpointError`` when a name is missing
    or unexpected, or when a shape is different. With ``assign=False`` the
    function copies into the tensors of the module. It then also raises on a
    dtype mismatch, because a silent cast rounds the float32 routing biases.
    With ``assign=True`` the module takes the loaded tensors, with their dtype
    and device. Use ``assign=True`` for a module on the meta device.
    """
    if prefix:
        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    expected = module.state_dict(keep_vars=True)
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    problems: list[str] = []
    if missing:
        problems.append(_listing("missing", missing))
    if unexpected:
        problems.append(_listing("unexpected", unexpected))
    shape: list[str] = []
    dtype: list[str] = []
    meta: list[str] = []
    for name in sorted(set(expected) & set(state_dict)):
        have, new = expected[name], state_dict[name]
        if tuple(have.shape) != tuple(new.shape):
            shape.append(f"{name} {tuple(have.shape)} != {tuple(new.shape)}")
        if not assign and have.dtype != new.dtype:
            dtype.append(f"{name} {have.dtype} != {new.dtype}")
        if not assign and have.is_meta:
            meta.append(name)
    if shape:
        problems.append(_listing("shape mismatch", shape))
    if dtype:
        problems.append(_listing("dtype mismatch", dtype))
    if meta:
        problems.append(_listing("meta tensors (use assign=True)", meta))
    if problems:
        raise CheckpointError(f"cannot load into {type(module).__name__}: " + "; ".join(problems))
    module.load_state_dict(dict(state_dict), strict=True, assign=assign)


def load_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    prefix: str = "",
    device: str | torch.device = "cpu",
    assign: bool = False,
) -> None:
    """Read the tensors under ``prefix`` from ``path`` into ``module``.

    The function calls ``load_state_dict_from_dir`` and then ``strict_load``.
    """
    state_dict = load_state_dict_from_dir(path, prefix=prefix, device=device)
    strict_load(module, state_dict, assign=assign)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def verify_checksums(path: str | Path) -> None:
    """Compare each listed file with its line in ``SHA256SUMS``.

    The list must hold every shard. It may also hold ``config.json`` and the
    index. The function raises ``CheckpointError`` when a shard is not in the
    list, when the list names another file, when a listed file is missing, or
    when a checksum is different.
    """
    root = Path(path)
    listed: dict[str, str] = {}
    for line in (root / CHECKSUM_NAME).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            listed[name.strip().lstrip("*")] = digest
    shards = set(weight_map(root).values())
    extra = set(listed) - shards - {CONFIG_NAME, INDEX_NAME}
    if not shards <= set(listed) or extra:
        raise CheckpointError(f"{CHECKSUM_NAME} lists {sorted(listed)}, the index uses {sorted(shards)}")
    for name, digest in sorted(listed.items()):
        if not (root / name).is_file():
            raise CheckpointError(f"{CHECKSUM_NAME} lists {name}, which is missing")
        if _sha256(root / name) != digest:
            raise CheckpointError(f"checksum mismatch for {name}")


def _natural_key(name: str) -> list[int | str]:
    """Sort key that orders ``blocks.2`` before ``blocks.10``."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def save_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, torch.Tensor],
    config: Mapping[str, object],
    *,
    max_shard_bytes: int = MAX_SHARD_BYTES,
) -> None:
    """Write a checkpoint directory in the format that the loaders read.

    The function writes ``config.json``, the shards
    ``model-XXXXX-of-NNNNN.safetensors``, ``model.safetensors.index.json``
    and ``SHA256SUMS``. The shards hold the tensors in the natural order of
    their names. A shard gets the next tensor while its size stays at or
    below ``max_shard_bytes``. Each tensor keeps its dtype. The function
    writes the files into the directory ``path.tmp`` and then renames it to
    ``path``. It raises ``CheckpointError`` when ``path`` exists.
    """
    root = Path(path)
    if root.exists():
        raise CheckpointError(f"{root} exists")
    tmp = root.with_name(root.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    for stale in tmp.iterdir():  # the files of an interrupted save
        stale.unlink()
    names = sorted(state_dict, key=_natural_key)
    shards: list[list[str]] = [[]]
    size = 0
    for name in names:
        t = state_dict[name]
        n = t.numel() * t.element_size()
        if shards[-1] and size + n > max_shard_bytes:
            shards.append([])
            size = 0
        shards[-1].append(name)
        size += n
    count = len(shards)
    files = [f"model-{i + 1:05d}-of-{count:05d}.safetensors" for i in range(count)]
    for file, keys in zip(files, shards, strict=True):
        tensors = {k: state_dict[k].detach().cpu().contiguous() for k in keys}
        save_file(tensors, str(tmp / file), metadata={"format": "pt"})
    total = sum(state_dict[k].numel() * state_dict[k].element_size() for k in names)
    index = {
        "metadata": {"total_size": total},
        "weight_map": {k: f for f, keys in zip(files, shards, strict=True) for k in keys},
    }
    (tmp / INDEX_NAME).write_text(json.dumps(index, indent=2))
    (tmp / CONFIG_NAME).write_text(json.dumps(config, indent=2))
    (tmp / CHECKSUM_NAME).write_text("".join(f"{_sha256(tmp / f)}  {f}\n" for f in files))
    tmp.rename(root)
