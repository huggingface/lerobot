# Vendored from runwayml/praxis-1-oss at commit afef2e1, original path praxis/so101.py, with the LeRobot
# reductions noted in the module docstring; import paths rewritten to the LeRobot package.
"""Robot input and output glue for the SO-100 / SO-101 rig. No neural networks.

An so101 checkpoint (``"embodiment": "so101"``) is the model variant of the
single-arm checkpoint with a 6-D joint-space head: the 5 arm joints and the
gripper of an SO-100 / SO-101 follower arm (LeRobot). It reads one or two
views. This module is the rig side of it; :mod:`praxis.single_arm` and
:mod:`praxis.bimanual` are its twins. The session is the observed-context
session of those rigs: the executed frames of each chunk go into one
contiguous stream (:class:`praxis.robot_io.ContiguousStreamContext`).

Observation contract (rig side, LeRobot's ``SOFollower`` with
``use_degrees=True``):

* ``state``: ``(T, 6)`` rows ``shoulder_pan.pos``, ``shoulder_lift.pos``,
  ``elbow_flex.pos``, ``wrist_flex.pos``, ``wrist_roll.pos`` in degrees (0 =
  the calibrated mid-range of the motor) and ``gripper.pos`` in 0..100 (the
  follower's own scale). A joint reading past :data:`MAX_JOINT_DEG` is not a
  ``use_degrees=True`` reading and is refused.
* ``image.exterior`` (view 0) and, in a 2-view session, ``image.wrist``
  (view 1): RGB uint8 ``(T, H, W, 3)``. A 1-view session reads only
  ``image.exterior``.
* Each value carries a leading time axis (a frame stack) of the same length:
  one entry per executed 15 Hz row, oldest first. The last entry is the
  current frame and state, the anchor of the new chunk. A single entry (no
  time axis) is a stack of one.
* ``task`` or ``prompt``: an optional prompt override. ``reset_context``
  restarts the context.

Model space (the 6-D joint head of training):

* A row is ``[q0 .. q4, g]``: the joints in radians (degrees x pi / 180) and
  the gripper closure ``g = 1 - (x - closed) / (open - closed)`` clipped to
  [0, 1], with ``closed`` / ``open`` the follower's ``gripper.pos`` at
  closure 1 and 0 (:class:`GripRange`). Training mapped the p1 / p99 reading
  of each table to closure 1 / 0, so "low = closed" is an assumption about
  the training data, not a rule of LeRobot: a follower whose reading grows as
  it closes gives ``closed > open``.
* The actions are the joints relative to the anchor row (``q_t - q_anchor``)
  and the absolute closure. All 6 dims are q99-normalized. The proprio is the
  absolute row at each context chunk start and at the anchor, mean/std
  normalized.
* The decode clips the normalized dims to [-1, 1], applies the inverse q99
  map, adds the anchor joints and clips the closure to [0, 1].
* The canvas is the fit rule of training on the views of the session, side
  by side (:func:`praxis.bimanual.fit_row_sizes`): each view scaled down
  (antialiased, never up) to fit 416 x 416 with its aspect kept, the views to
  one common height, each trimmed to the 32-px grid. A 640x480 camera gives a
  288x416 tile, so a 2-view canvas is 288x832 and a 1-view canvas 288x416.
  Other camera sizes give other canvases, as in training. The checkpoint's
  ``tile_height x tile_width`` is the tile of the native 480x640 view.

Output: the ``(T, 6)`` float32 absolute targets of one chunk in the units of
``state`` (the values of LeRobot's ``send_action``: degrees x 5,
``gripper.pos``), one row per action at 15 Hz. Chunk 0 of a context window
gives 21 rows. Every later chunk gives 24 rows.

LeRobot reductions of the upstream module: a view given as a ``(T, 3, H, W)`` uint8 tensor
is a tile prepared by the LeRobot preprocessor and passes through ``canvas_clip`` as is
(the fit rule ran already, on the CPU; the views must be a fixed point of it,
``is_fit_canvas``); the session runs on the chunk grid only (no ``executed_rows``: the
real-time chunking of the praxis server is not ported); ``finish`` returns the targets
array instead of the command dicts of upstream's ``rel_to_wire`` (LeRobot sends the
targets to the robot); the hooks of the praxis policy server and its boot warm-up are not
needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .bimanual import fit_row_sizes, resize_frames, trim_centre
from .robot_io import (
    ACTION_CHUNK,
    FPS,
    FRAMES_PER_TOKEN,
    ContiguousStreamContext,
    NormStats,
    action_rows,
    clip_normalized,
    latent_frames,
    proprio_normalize,
    q99_denormalize,
    text_encoder_prompt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The views of the checkpoint config, in canvas order. A 1-view session uses
#: the first one.
VIEWS: tuple[str, ...] = ("exterior", "wrist")
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ACTION_DIM = 6
N_ARM = 5
GRIP_DIM = 5
N_CHUNKS = 4
MAX_CONTEXT_CHUNKS = 3
#: The tile of the native 480x640 view (the config's tile_height x tile_width).
TILE_SIZE: tuple[int, int] = (288, 416)
NATIVE_VIEW: tuple[int, int] = (480, 640)
#: The fit rule bounds each view to this square with its aspect kept (it never
#: scales up): 480x640 -> 312x416.
RESTRICT_SIDE = 416
#: The spatial patch of the model (VAE stride 16 x patch 2): the fit rule
#: trims each view to this grid.
SPATIAL_PATCH = 32
#: All 6 dims are q99-normalized (the "joints:6" layout of training).
Q99_DIMS = np.arange(ACTION_DIM)
GRIP_DIMS = np.array([GRIP_DIM])
DEG_TO_RAD = np.pi / 180.0
#: The gripper readings at closure 1 / 0 without a measurement: the
#: hour-weighted medians of the p1 / p99 ``gripper.pos`` of the SO training
#: tables, rounded. Measure your own follower and replace them.
DEFAULT_GRIP_CLOSED = 1.4
DEFAULT_GRIP_OPEN = 42.1
#: The least distance of the two gripper readings.
MIN_GRIP_SPAN = 1.0
#: A joint reading past this many degrees is not a ``use_degrees=True``
#: reading (RANGE_M100_100 tops out at 100, a raw tick at 4095).
MAX_JOINT_DEG = 200.0

STATE_KEY = "state"
VIEW_KEYS: dict[str, str] = {"exterior": "image.exterior", "wrist": "image.wrist"}

# The so101 rig has no control law of its own: the arm keeps its position loop.
CONTROL_MODES: dict[str, dict[str, object] | None] = {"none": None}


def stack_keys(n_views: int = 2) -> tuple[str, ...]:
    """Return the frame stack keys of a session of ``n_views`` views."""
    return (*(VIEW_KEYS[v] for v in VIEWS[:n_views]), STATE_KEY)


def check_views(n_views: object) -> int:
    """Return ``n_views`` as 1 or 2, else raise ``ValueError``."""
    if isinstance(n_views, bool) or not isinstance(n_views, int | np.integer) or int(n_views) not in (1, 2):
        raise ValueError(f"views must be 1 or 2, got {n_views!r}")
    return int(n_views)


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


def canvas_size(config: dict, n_views: int = 2) -> tuple[int, int]:
    """Return the canvas ``(height, width)`` of native 480x640 views of ``config``.

    The config must declare the tile canvas of :data:`VIEWS`. Another canvas
    raises ``ValueError``.
    """
    video = config.get("video", {})
    views = tuple(video.get("views", VIEWS))
    if views != VIEWS or video.get("canvas") != "tiles":
        raise ValueError(
            f"The so101 rig supports the tile canvas of {VIEWS}. The config has "
            f"views={views} canvas={video.get('canvas')!r}."
        )
    th = int(video.get("tile_height", TILE_SIZE[0]))
    tw = int(video.get("tile_width", TILE_SIZE[1]))
    return th, check_views(n_views) * tw


def q99_dims(config: dict) -> np.ndarray:
    """Return the q99-normalized action dims of an so101 checkpoint config."""
    norm = config.get("normalization", {})
    return np.asarray(norm.get("q99_normed_dims", Q99_DIMS), dtype=np.int64)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GripRange:
    """The follower's ``gripper.pos`` at closure 1 (``closed``) and 0 (``open``).

    Training mapped the p1 / p99 reading of each table to closure 1 / 0, so
    the defaults are low = closed. That is an assumption about the training
    data, not a rule of LeRobot: a follower whose reading grows as it closes
    gives ``closed > open``, which is allowed. The two readings must differ
    by at least :data:`MIN_GRIP_SPAN`.
    """

    closed: float = DEFAULT_GRIP_CLOSED
    open: float = DEFAULT_GRIP_OPEN

    def __post_init__(self) -> None:
        if not np.isfinite([self.closed, self.open]).all():
            raise ValueError(f"non-finite grip range {self}")
        if abs(self.open - self.closed) < MIN_GRIP_SPAN:
            raise ValueError(
                f"grip_closed {self.closed} and grip_open {self.open} must differ by "
                f">= {MIN_GRIP_SPAN} (LeRobot gripper.pos, 0..100)"
            )

    def as_dict(self) -> dict[str, float]:
        return {"closed": float(self.closed), "open": float(self.open)}


def lerobot_to_model(state: np.ndarray, grip: GripRange) -> np.ndarray:
    """``(..., 6)`` LeRobot readings -> model rows (rad x 5, closure), float64."""
    x = np.asarray(state, np.float64)
    if x.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected (..., 6) SO101 state, got {x.shape}")
    out = np.empty_like(x)
    out[..., :N_ARM] = x[..., :N_ARM] * DEG_TO_RAD
    g = (x[..., GRIP_DIM] - grip.closed) / (grip.open - grip.closed)
    out[..., GRIP_DIM] = np.clip(1.0 - g, 0.0, 1.0)
    return out


def model_to_lerobot(rows: np.ndarray, grip: GripRange) -> np.ndarray:
    """Inverse of :func:`lerobot_to_model` on absolute rows. The closure is
    clipped to [0, 1] first, so the gripper target stays between ``closed``
    and ``open``."""
    x = np.asarray(rows, np.float64)
    if x.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected (..., 6) rows, got {x.shape}")
    out = np.empty_like(x)
    out[..., :N_ARM] = x[..., :N_ARM] / DEG_TO_RAD
    c = np.clip(x[..., GRIP_DIM], 0.0, 1.0)
    out[..., GRIP_DIM] = grip.open - c * (grip.open - grip.closed)
    return out


def rel_to_abs(rel: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Relative rows ``(T, 6)`` of one chunk -> absolute model rows: the joints
    ``rel + q_anchor``, the closure absolute, clipped to [0, 1]."""
    r = np.asarray(rel, np.float64)
    a = np.asarray(anchor, np.float64).reshape(ACTION_DIM)
    out = r.copy()
    out[:, :N_ARM] = r[:, :N_ARM] + a[:N_ARM]
    out[:, GRIP_DIM] = np.clip(r[:, GRIP_DIM], 0.0, 1.0)
    return out


def abs_to_rel(rows: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rel_to_abs` (the joint targets of training)."""
    r = np.asarray(rows, np.float64)
    a = np.asarray(anchor, np.float64).reshape(ACTION_DIM)
    out = r.copy()
    out[:, :N_ARM] = r[:, :N_ARM] - a[:N_ARM]
    return out


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def state_stack(obs: dict) -> np.ndarray:
    """Return the ``(T, 6)`` float64 LeRobot state stack of ``obs``.

    A missing key, a wrong shape, a value that is not finite or a joint past
    :data:`MAX_JOINT_DEG` (not ``use_degrees=True``) raises ``ValueError``.
    """
    raw = obs.get(STATE_KEY)
    if raw is None:
        raise ValueError(f"missing {STATE_KEY} (T, 6): {', '.join(JOINT_NAMES)}")
    state = np.asarray(raw, dtype=np.float64)
    if state.ndim == 1:
        state = state[None]
    if state.ndim != 2 or state.shape[1] != ACTION_DIM or len(state) < 1:
        raise ValueError(f"{STATE_KEY}: expected (T, 6), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError(f"{STATE_KEY} has values that are not finite")
    big = float(np.abs(state[:, :N_ARM]).max())
    if big > MAX_JOINT_DEG:
        raise ValueError(
            f"{STATE_KEY}: |joint| {big:.0f} is not degrees; connect the follower "
            "with use_degrees=True (joints in degrees, gripper.pos 0..100)"
        )
    return state


def view_stacks(obs: dict, n_views: int) -> list[np.ndarray | torch.Tensor]:
    """Return the stacks of the first ``n_views`` views.

    A raw view is ``(T, H, W, 3)`` uint8 numpy; a single frame gets a leading time
    axis. A ``(T, 3, H, W)`` uint8 tensor is a tile of the LeRobot preprocessor and is
    returned as is. A missing view raises ``KeyError``, a frame that is not uint8 or
    smaller than one patch raises ``ValueError``.
    """
    out: list[np.ndarray | torch.Tensor] = []
    for view in VIEWS[:n_views]:
        img = obs.get(VIEW_KEYS[view])
        if img is None:
            raise KeyError(
                f"missing {VIEW_KEYS[view]}: this session has views={n_views} ("
                + " + ".join(VIEW_KEYS[v] for v in VIEWS[:n_views])
                + ")"
            )
        if isinstance(img, torch.Tensor):
            if img.ndim != 4 or img.shape[1] != 3 or img.dtype != torch.uint8:
                raise ValueError(f"{view}: a tile stack must be (T, 3, H, W) uint8, got {tuple(img.shape)}")
            out.append(img)
            continue
        img = np.asarray(img)
        if img.ndim == 3:
            img = img[None]
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"{view}: image must be (T,H,W,3), got {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"{view}: image must be uint8, got {img.dtype}")
        if min(img.shape[1:3]) < SPATIAL_PATCH:
            raise ValueError(f"{view}: frames {img.shape[1:3]} under {SPATIAL_PATCH} px")
        out.append(img)
    return out


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def canvas_hw(view_hw: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the canvas ``(H, W)`` of views of native sizes ``view_hw``."""
    _resize, trimmed = fit_row_sizes(list(view_hw), RESTRICT_SIDE, SPATIAL_PATCH)
    return trimmed[0][0], sum(w for _, w in trimmed)


def fit_tile_sizes(view_hw: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the ``(h, w)`` tile of each view of native sizes ``view_hw`` (the fit rule)."""
    return fit_row_sizes(list(view_hw), RESTRICT_SIDE, SPATIAL_PATCH)[1]


def is_fit_canvas(view_hw: list[tuple[int, int]]) -> bool:
    """True if the fit rule leaves views of sizes ``view_hw`` as they are (no resize, no trim).

    The tiles of the fit rule are such views, so the rule does not change a tile twice: it
    is idempotent on its own output.
    """
    hw = [(int(h), int(w)) for h, w in view_hw]
    if not hw or min(min(s) for s in hw) < SPATIAL_PATCH:
        return False
    resize, trimmed = fit_row_sizes(hw, RESTRICT_SIDE, SPATIAL_PATCH)
    return resize == hw and trimmed == hw


def canvas_clip(stacks: list[np.ndarray | torch.Tensor], device: torch.device | str = "cpu") -> torch.Tensor:
    """Convert view stacks to a ``(1, 3, T, H, W)`` float32 canvas clip.

    The fit rule of training (:func:`praxis.bimanual.fit_row_sizes`): an
    antialiased resize of each view to the common height, a centre trim to
    the 32-px grid, the views side by side in session order. The values are
    integers in [0, 255]. Tensor stacks are tiles of the LeRobot preprocessor
    (the fit rule ran already) and go side by side as they are; their sizes
    must be a fixed point of the rule (:func:`is_fit_canvas`).
    """
    counts = {s.shape[0] for s in stacks}
    if len(counts) != 1:
        raise ValueError(f"the views have different frame counts {sorted(counts)}")
    if all(isinstance(s, torch.Tensor) for s in stacks):
        native = [(int(s.shape[-2]), int(s.shape[-1])) for s in stacks]
        if not is_fit_canvas(native):
            raise ValueError(
                f"tensor views of {native} are not tiles of the fit rule (it makes {fit_tile_sizes(native)})"
            )
        tiles = [s.to(device) for s in stacks if isinstance(s, torch.Tensor)]
    elif any(isinstance(s, torch.Tensor) for s in stacks):
        raise ValueError("the views must all be prepared tiles or all raw frames")
    else:
        frames = [torch.from_numpy(np.ascontiguousarray(s)).permute(0, 3, 1, 2) for s in stacks]
        native = [(int(f.shape[-2]), int(f.shape[-1])) for f in frames]
        resize, trimmed = fit_row_sizes(native, RESTRICT_SIDE, SPATIAL_PATCH)
        tiles = [
            trim_centre(resize_frames(f, *r), *t) for f, r, t in zip(frames, resize, trimmed, strict=True)
        ]
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Serve session state (model free)
# ---------------------------------------------------------------------------


@dataclass
class So101ChunkInputs:
    """Model inputs of one served chunk of the so101 rig.

    The fields mirror :class:`praxis.bimanual.BimanualChunkInputs`:
    ``boundary_poses`` holds the ``depth + 1`` absolute model rows at the
    chunk starts of the window (the anchor row last) and ``pose_now`` the
    anchor row. ``proprio`` is ``(1, 1, 6)`` and ``proprio_all`` is
    ``(1, depth + 1, 6)``. ``canvas_hw`` is the canvas of this call.
    """

    anchor: torch.Tensor
    cleans: list[torch.Tensor]
    boundary_poses: np.ndarray
    proprio: torch.Tensor
    proprio_all: torch.Tensor
    depth: int
    pose_now: np.ndarray
    chunk_index: int
    prompt: str
    canvas_hw: tuple[int, int] = (0, 0)

    @property
    def refs(self) -> list[torch.Tensor]:
        """Return the reference latents ``[anchor, *cleans]``."""
        return [self.anchor, *self.cleans]

    @property
    def video_shape(self) -> tuple[int, ...]:
        """Return the shape of the noised video chunk (6 latent frames)."""
        shape = list(self.anchor.shape)
        shape[-3] = latent_frames(ACTION_CHUNK)
        return tuple(shape)


class So101ServeSession:
    """Observed-context state machine of one so101 rig session.

    The steps and the context rules are those of
    :class:`praxis.bimanual.BimanualServeSession`: a prompt override, the
    executed rows of the previous chunk appended to the stream (the stack
    rows before the current one), the current frame encoded, then the window
    ``[anchor | d cleans]`` and the proprio. A stack that is missing or too
    short for the executed chunk restarts the session (a cold start at
    depth 0).

    ``n_views`` (1 or 2) and ``grip`` are the settings of the rig. The
    canvas is the fit rule (``antialias`` must be on), and ``tile_size`` must
    be the fit tile of the native 480x640 view (288x416). The rig has one
    control mode, ``"none"``.
    """

    def __init__(
        self,
        encode: Callable[[torch.Tensor], torch.Tensor],
        stats: NormStats,
        device: torch.device | str,
        prompt: str = "",
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
        fps: float = FPS,
        dims: np.ndarray = Q99_DIMS,
        control_mode: str = "none",
        exec_horizon: int | None = None,
        n_chunks: int = N_CHUNKS,
        tile_size: tuple[int, int] | None = None,
        antialias: bool = True,
        n_views: int = 2,
        grip: GripRange | None = None,
    ):
        if control_mode not in CONTROL_MODES:
            raise ValueError(
                f"Unknown control_mode {control_mode!r} for the so101 rig. The "
                f"known modes are {sorted(CONTROL_MODES)}."
            )
        if not antialias:
            raise ValueError("the so101 rig needs video.antialias: its canvas is the fit rule")
        th, tw = TILE_SIZE if tile_size is None else tile_size
        self.tile_size = (int(th), int(tw))
        fit = canvas_hw([NATIVE_VIEW])
        if fit != self.tile_size:
            raise ValueError(
                f"the fit rule gives {fit[0]}x{fit[1]} tiles of a "
                f"{NATIVE_VIEW[0]}x{NATIVE_VIEW[1]} view; the checkpoint config asks "
                f"{self.tile_size[0]}x{self.tile_size[1]}"
            )
        self.antialias = True
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(dims)
        self.control_mode = str(control_mode)
        self.exec_horizon = None if exec_horizon is None else int(exec_horizon)
        self.n_views = check_views(n_views)
        self.grip = grip or GripRange()
        self.max_context_chunks = max(0, min(int(max_context_chunks), int(n_chunks) - 1))
        self.stateless = self.max_context_chunks == 0 or (
            self.exec_horizon is not None and self.exec_horizon < ACTION_CHUNK
        )
        self.stream = ContiguousStreamContext(
            action_chunk=ACTION_CHUNK,
            max_context_chunks=max(self.max_context_chunks, 1),
        )
        self.chunk_index = 0
        self.restart_count = 0
        # True after a call: the next call appends the executed rows. A
        # restart makes the next call a cold start.
        self._primed = False
        self._boundary_poses: list[np.ndarray] = []
        self._cap = self.max_context_chunks + 1

    @property
    def stack_keys(self) -> tuple[str, ...]:
        """Return the frame stack keys of this session (its views and the state)."""
        return stack_keys(self.n_views)

    @property
    def canvas_size(self) -> tuple[int, int]:
        """Return the canvas ``(H, W)`` of native 480x640 views in this session."""
        return self.tile_size[0], self.n_views * self.tile_size[1]

    def reset(self) -> None:
        """Start a new episode (chunk counter back to 0, context dropped)."""
        self.chunk_index = 0
        self.restart()

    def restart(self) -> None:
        """Drop the context. The next call is a cold start at depth 0."""
        self.restart_count += 1
        self._primed = False
        self._boundary_poses = []
        self.stream.reset()

    def _clip(self, stacks: list[np.ndarray | torch.Tensor]) -> torch.Tensor:
        return canvas_clip(stacks, self.device)

    def _stacks(self, obs: dict) -> tuple[np.ndarray, list[np.ndarray | torch.Tensor]]:
        """Return the model rows ``(T, 6)`` and the view stacks of ``obs``.

        The state and the views must have the same number of entries.
        """
        state = state_stack(obs)
        stacks = view_stacks(obs, self.n_views)
        n = len(state)
        if any(len(s) != n for s in stacks):
            raise ValueError(f"stack lengths differ: {STATE_KEY} {n}, views {[len(s) for s in stacks]}")
        return lerobot_to_model(state, self.grip), stacks

    def prepare(self, obs: dict) -> So101ChunkInputs:
        """Update the context from ``obs`` and return the inputs of this chunk."""
        if bool(obs.get("reset_context") or obs.get("__reset_context__")):
            self.restart()
        override = obs.get("task") or obs.get("prompt")
        if override:
            if isinstance(override, bytes):
                override = override.decode("utf-8", "replace")
            new = str(override).strip()
            if new and new != self.prompt:
                self.prompt = new
        rows, stacks = self._stacks(obs)
        pose_now = rows[-1]
        cur_clip = self._clip([s[-1:] for s in stacks])
        if self.stateless:
            self.restart()
        elif self._primed:
            need = self.stream.expected_next_rows
            if len(rows) < need + 1:
                self.restart()
            else:
                clip = self._clip([s[-(need + 1) : -1] for s in stacks])
                self.stream.append_executed(clip, rows[-(need + 1) : -1])
        self._boundary_poses = [*self._boundary_poses, pose_now][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        d = len(cleans)
        af = self.stream.anchor_frame()
        anchor = self.encode(cur_clip) if af is None else self.encode(af)
        bp = self.stream.boundary_poses() if d else None
        if d == 0:
            poses = pose_now[None]
        elif bp is not None:
            poses = np.concatenate([bp, pose_now[None]], axis=0)
        else:
            n = min(d + 1, len(self._boundary_poses))
            poses = np.stack(self._boundary_poses[-n:], axis=0)
        poses = poses.astype(np.float32)
        p = proprio_normalize(poses, self.stats.proprio_mean, self.stats.proprio_std)
        every = torch.from_numpy(np.ascontiguousarray(p)).reshape(1, len(p), -1)
        every = every.to(self.device, torch.float32)
        self._primed = True
        return So101ChunkInputs(
            anchor=anchor,
            cleans=cleans,
            boundary_poses=poses,
            proprio=every[:, -1:],
            proprio_all=every,
            depth=d,
            pose_now=pose_now.astype(np.float64),
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
            canvas_hw=(int(cur_clip.shape[-2]), int(cur_clip.shape[-1])),
        )

    def decode_actions(
        self, inputs: So101ChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized rows and the denormalized relative rows.

        The relative rows clip the normalized ``dims`` to [-1, 1] before the
        inverse q99 map and clip the closure to [0, 1] after it.
        """
        rows = action_rows(pred_action, inputs.depth, FRAMES_PER_TOKEN, ACTION_DIM)
        rel = q99_denormalize(
            clip_normalized(rows, self.dims),
            self.stats.action_q01,
            self.stats.action_q99,
            dims=self.dims,
        )
        rel[:, GRIP_DIM] = np.clip(rel[:, GRIP_DIM], 0.0, 1.0)
        return rows, rel

    def rel_to_targets(self, rel: np.ndarray, pose_now: np.ndarray) -> np.ndarray:
        """Return the ``(T, 6)`` float32 absolute LeRobot targets of ``rel``."""
        return model_to_lerobot(rel_to_abs(rel, pose_now), self.grip).astype(np.float32)

    def finish(self, inputs: So101ChunkInputs, pred_action: torch.Tensor) -> np.ndarray:
        """Convert the sampled actions of this chunk to the ``(T, 6)`` targets and advance.

        The rows are cut to ``exec_horizon``. (Upstream returns the command dicts of
        ``rel_to_wire``; LeRobot sends the targets to the robot as they are.)
        """
        _rows, rel = self.decode_actions(inputs, pred_action)
        targets = self.rel_to_targets(rel, inputs.pose_now)
        self.chunk_index += 1
        return targets if self.exec_horizon is None else targets[: self.exec_horizon]
