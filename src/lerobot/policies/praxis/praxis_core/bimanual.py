# Vendored from the praxis repository (main, praxis/bimanual.py) with the LeRobot reductions noted
# in the module docstring; import paths rewritten to the LeRobot package.
"""Robot input and output glue for the bimanual (YAM) rig. No neural networks.

The bimanual checkpoint (``"embodiment": "bimanual"``) is the model variant of
the single-arm checkpoint with 20-D actions over two arms and a canvas of three
views. This module is the rig side of it; :mod:`praxis.single_arm` is the
single-arm twin.

Observation contract (rig side, the Positronic ``yam_bimanual`` embodiment):

* ``robot_state.left.ee_pose`` and ``robot_state.right.ee_pose``: the pose of
  each arm, 7 values ``[x, y, z, qw, qx, qy, qz]`` (quaternion scalar first),
  metres, in the world frame of the rig.
* ``grip.left`` and ``grip.right``: absolute gripper values, 0 = open.
* ``image.exterior`` (the top camera), ``image.wrist_left`` and
  ``image.wrist_right``: RGB uint8. All three views are needed.
* Each value can carry a leading time axis (a frame stack). The current
  frame is the last row.

Model space:

* pose16 = ``[L: xyz3 quat_wxyz4 grip1 | R: same]``.
* 20-D = ``[L: xyz3 rot6d6 grip1 | R: same]``. rot6d holds the first two
  columns of the rotation matrix.
* Each arm's actions are relative to that arm's pose at the start of the
  chunk (``xyz - anchor_xyz``, ``R @ R_anchor^T``). The grippers are absolute.
* Actions are q99-normalized on the xyz and grip dims of both arms
  (``normalization.q99_normed_dims`` of the config). rot6d stays identity.
  Proprio is mean/std-normalized on all 20 dims. The proprio keeps the
  gripper value of the rig (no clip), as in training.
* The decode clips the normalized xyz and grip dims to [-1, 1], the range of
  the training targets. After the inverse q99 map, it clips the grippers to
  [0, 1].
* The canvas is the tile canvas of the three views, side by side in the order
  ``top_left``, ``wrist_left``, ``wrist_right``. With ``video.antialias``
  (the YAM checkpoint), each view goes through the fit rule of training: an
  antialiased resize so that the view fits ``tile_width`` with its aspect
  kept, then a centre trim to the tile (1200x1920 -> 260x416 -> 256x416).
* The poses stay in the world frame of the rig: the model saw the rig poses
  as recorded.

Output: for each arm, the absolute poses ``[xyz3, R_flat9]`` (row-major) and
grippers of a chunk, one row per action at 15 Hz. Chunk 0 of a context window
gives 21 rows. Every later chunk gives 24 rows.

LeRobot reductions of the upstream module: a view given as a ``(T, 3, tile_h, tile_w)``
uint8 tensor is a tile prepared by the LeRobot preprocessor and passes through
``canvas_clip`` as is (the canvas rule ran already, on the CPU); the poses and grippers
of each arm are arrays (``rel20_to_pose12``) instead of the command dicts of upstream's
``rel20_to_wire`` (LeRobot's command processor step makes the robot commands); the boot
warm-up helper is not needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .robot_io import (
    ACTION_CHUNK,
    ASPECT_TOL,
    FPS,
    FRAMES_PER_TOKEN,
    ContiguousStreamContext,
    NormStats,
    action_rows,
    clip_grip,
    denormalize_rows,
    latent_frames,
    pose8_to_10d,
    proprio_normalize,
    rot6d_to_matrix,
    text_encoder_prompt,
    unit_quat_wxyz_to_matrix,
)

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

VIEWS: tuple[str, ...] = ("top_left", "wrist_left", "wrist_right")
SIDES: tuple[str, ...] = ("left", "right")
ACTION_DIM = 20
ARM_DIM = 10
N_CHUNKS = 4
MAX_CONTEXT_CHUNKS = 3

# The tile canvas: one tile (height, width) per view, side by side.
TILE_SIZE: tuple[int, int] = (256, 416)
# The native YAM camera view (height, width): the ZED views of the training
# data. The client bounds each view to ``RESTRICT_SIDE`` x ``RESTRICT_SIDE``
# with its aspect kept (1200x1920 -> 260x416) before it sends it.
NATIVE_VIEW: tuple[int, int] = (1200, 1920)
RESTRICT_SIDE = 416
# The spatial patch of the model (VAE stride 16 x patch 2): the fit rule trims
# each view to this grid.
SPATIAL_PATCH = 32
# Frames per resize call of the canvas (bounds the float32 memory).
TILE_CHUNK_FRAMES = 32

# Dims of a 20-D row that use the q99 normalization (xyz and grip, both arms).
Q99_DIMS = np.array([0, 1, 2, 9, 10, 11, 12, 19])
# The dims of a 20-D row that hold the grippers.
GRIP_DIMS = np.array([9, 19])

VIEW_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "top_left": ("image.exterior", "image.top", "image.top_left"),
    "wrist_left": ("image.wrist_left", "image.left_wrist"),
    "wrist_right": ("image.wrist_right", "image.right_wrist"),
}


def pose_keys(side: str) -> tuple[str, ...]:
    return (f"robot_state.{side}.ee_pose", f"ee_pose.{side}")


def grip_keys(side: str) -> tuple[str, ...]:
    return (f"grip.{side}", f"robot_state.{side}.grip", f"gripper.{side}")


# The frame stack keys of the rig, in the declared order.
STACK_KEYS: tuple[str, ...] = (
    *(VIEW_KEY_CANDIDATES[v][0] for v in VIEWS),
    *(pose_keys(s)[0] for s in SIDES),
    *(grip_keys(s)[0] for s in SIDES),
)

# The bimanual rig has no control law of its own: the arms keep theirs.
CONTROL_MODES: dict[str, dict[str, object] | None] = {"none": None}

# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


def canvas_size(config: dict) -> tuple[int, int]:
    """Return the canvas ``(height, width)`` of a bimanual checkpoint config.

    The config must declare the tile canvas of :data:`VIEWS`. Another canvas
    raises ``ValueError``.
    """
    video = config.get("video", {})
    views = tuple(video.get("views", VIEWS))
    canvas = video.get("canvas")
    if views != VIEWS or canvas != "tiles":
        raise ValueError(
            f"The bimanual rig supports the tile canvas of {VIEWS}. The config "
            f"has views={views} canvas={canvas!r}."
        )
    th = int(video.get("tile_height", TILE_SIZE[0]))
    tw = int(video.get("tile_width", TILE_SIZE[1]))
    return th, len(VIEWS) * tw


def q99_dims(config: dict) -> np.ndarray:
    """Return the q99-normalized action dims of a bimanual checkpoint config."""
    norm = config.get("normalization", {})
    return np.asarray(norm.get("q99_normed_dims", Q99_DIMS), dtype=np.int64)


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def _pick(obs: dict, candidates: tuple[str, ...]) -> np.ndarray:
    return np.asarray(_pick_any(obs, candidates))


def _pick_any(obs: dict, candidates: tuple[str, ...]) -> object:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"obs missing any of {candidates}; got keys={sorted(obs)}")


class StackMismatch(ValueError):  # noqa: N818 (the upstream name)
    """The pose and grip stacks of an arm have different lengths."""


def _pose8_rows(obs: dict, side: str, broadcast: bool = True) -> np.ndarray:
    """Return the ``(T, 8)`` ``[xyz, quat_wxyz, grip]`` rows of one arm.

    ``T`` is 1 when the observation has no time axis. With ``broadcast``, a
    single grip row (or pose row) is repeated to the length of the other
    stack. A quaternion that is not a unit quaternion (within 1e-2) or a value
    that is not finite raises ``ValueError``.
    """
    ee: np.ndarray = _pick(obs, pose_keys(side)).astype(np.float32)
    grip: np.ndarray = _pick(obs, grip_keys(side)).astype(np.float32)
    ee = ee.reshape(-1, ee.shape[-1]) if ee.ndim >= 2 else ee.reshape(1, -1)
    if ee.shape[1] < 7:
        raise ValueError(f"{side} ee_pose must be (..., >=7), got {ee.shape}")
    grip = grip.reshape(grip.shape[0], -1) if grip.ndim >= 2 else grip.reshape(-1, 1)
    if grip.shape[0] != ee.shape[0]:
        if broadcast and (grip.shape[0] == 1 or ee.shape[0] == 1):
            n = max(grip.shape[0], ee.shape[0])
            if ee.shape[0] == 1:
                ee = np.broadcast_to(ee[-1:], (n, ee.shape[1]))
            if grip.shape[0] == 1:
                grip = np.broadcast_to(grip[-1:], (n, grip.shape[1]))
        else:
            raise StackMismatch(f"{side}: ee_pose has {ee.shape[0]} rows but grip has {grip.shape[0]}")
    rows = np.concatenate([ee[:, :7], grip[:, :1]], axis=1).astype(np.float32)
    if not np.all(np.isfinite(rows)):
        raise ValueError(f"{side} ee_pose or grip has values that are not finite")
    norms = np.linalg.norm(rows[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-2):
        raise ValueError(f"{side} ee_pose quaternion norms {norms.min()}..{norms.max()} != 1")
    return rows


def extract_pose16(obs: dict) -> np.ndarray:
    """Return the current pose16 ``(16,)``: the last row of each arm's stacks."""
    return np.concatenate([_pose8_rows(obs, s)[-1] for s in SIDES]).astype(np.float32)


def pose16_history(obs: dict, need: int) -> np.ndarray | None:
    """Return the ``(need, 16)`` pose16 rows of the executed frames, or ``None``.

    The rows are stack rows ``[-(need + 1):-1]`` (the current row is not
    included). ``None`` means that a pose or grip key is missing, or that a
    stack is too short or shorter than the other stack of its arm. A single row is not repeated: it would make up a history. A
    pose that is not finite or not a unit quaternion raises ``ValueError``.
    """
    try:
        rows = [_pose8_rows(obs, s, broadcast=False) for s in SIDES]
    except (KeyError, StackMismatch):
        return None
    if any(r.shape[0] < need + 1 for r in rows):
        return None
    return np.concatenate([r[-(need + 1) : -1] for r in rows], axis=1).astype(np.float32)


def view_stacks(obs: dict) -> dict[str, np.ndarray]:
    """Return ``{view: (T, H, W, 3) uint8}`` in :data:`VIEWS` order.

    A single frame gets a leading time axis. A missing view raises
    ``KeyError``, a frame that is not uint8 raises ``ValueError`` (a float
    frame in [0, 1] would otherwise turn black).
    """
    out: dict[str, np.ndarray | torch.Tensor] = {}
    for view in VIEWS:
        raw = _pick_any(obs, VIEW_KEY_CANDIDATES[view])
        if isinstance(raw, torch.Tensor):
            # Tiles prepared by the LeRobot preprocessor: (T, 3, tile_h, tile_w) uint8, on any device.
            tiles = raw if raw.ndim == 4 else raw[None]
            if tiles.ndim != 4 or tiles.shape[1] != 3 or tiles.dtype != torch.uint8:
                raise ValueError(
                    f"{view}: tensor frames must be (T, 3, H, W) uint8 tiles, got {tuple(raw.shape)}"
                )
            out[view] = tiles
            continue
        img = np.asarray(raw)
        if img.ndim == 3:
            img = img[None]
        if img.ndim != 4 or img.shape[-1] != 3:
            raise ValueError(f"{view}: image must be (T,H,W,3), got {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"{view}: image must be uint8, got {img.dtype}")
        out[view] = img
    return out


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def fit_resize_size(h: int, w: int, max_side: int = RESTRICT_SIDE) -> tuple[int, int]:
    """Return ``(floor(h * s), floor(w * s))`` at ``s = min(1, max_side / h,
    max_side / w)`` in integers: the view scaled down to fit ``max_side`` with
    its aspect kept, never scaled up."""
    if h < 1 or w < 1 or max_side < 1:
        raise ValueError(f"bad fit: {h}x{w} into {max_side}")
    return min(h, max_side, (h * max_side) // w), min(w, max_side, (w * max_side) // h)


def fit_row_sizes(
    native: list[tuple[int, int]],
    max_side: int = RESTRICT_SIDE,
    patch: int = SPATIAL_PATCH,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the resize sizes and the trimmed sizes of the views of one row.

    The views share one height: the smallest fitted height of the views. A
    taller view is resized to that height with its aspect kept. Each resized
    view is then trimmed to the ``patch`` grid. This is the fit rule of the
    training canvas.
    """
    if not native:
        raise ValueError("the fit rule needs at least one view")
    fitted = [fit_resize_size(h, w, max_side) for h, w in native]
    common_h = min(rh for rh, _ in fitted)
    resize = [
        (common_h, (w * common_h) // h) if rh > common_h else (rh, rw)
        for (h, w), (rh, rw) in zip(native, fitted, strict=True)
    ]
    trimmed = []
    for rh, rw in resize:
        gh, gw = rh // patch * patch, rw // patch * patch
        if gh < patch or gw < patch:
            raise ValueError(f"a view fits to {rh}x{rw}: less than one {patch}-px patch")
        trimmed.append((gh, gw))
    return resize, trimmed


def _resize(fr: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Return the uint8 ``(T, 3, h, w)`` antialiased bilinear resize of ``fr``.

    The resize runs in float32 with ``align_corners=False``, then a round and a
    clamp to uint8. A stack already at ``h x w`` is returned as is.
    """
    if tuple(fr.shape[-2:]) == (h, w):
        return fr
    return torch.cat(
        [
            torch.nn.functional.interpolate(
                chunk.float(),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            for chunk in fr.split(TILE_CHUNK_FRAMES, dim=0)
        ]
    )


def _trim(fr: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Return the centre trim of ``fr`` to ``h x w``."""
    fh, fw = fr.shape[-2], fr.shape[-1]
    if h > fh or w > fw:
        raise ValueError(f"cannot trim {fh}x{fw} to {h}x{w}")
    y0, x0 = (fh - h) // 2, (fw - w) // 2
    return fr[..., y0 : y0 + h, x0 : x0 + w]


def canvas_clip(
    views: dict[str, np.ndarray],
    tile_h: int = TILE_SIZE[0],
    tile_w: int = TILE_SIZE[1],
    tol: float = ASPECT_TOL,
    device: torch.device | str = "cpu",
    antialias: bool = True,
) -> torch.Tensor:
    """Convert view stacks to a ``(1, 3, T, tile_h, 3 * tile_w)`` canvas clip.

    ``views`` maps each view of :data:`VIEWS` to ``(T, H, W, 3)`` uint8 frames
    with the same ``T``. With ``antialias`` each view goes through the fit rule
    (:func:`fit_row_sizes`); the result must be the tile, else ``ValueError``
    (send each view at its native aspect). Without it each view gets the
    single-arm tile rule (a centre crop to the tile aspect and one resize).
    The clip is float32 with integer values in [0, 255].
    """
    from .single_arm import _tile_view

    stacks = [views[v] for v in VIEWS]
    counts = {s.shape[0] for s in stacks}
    if len(counts) != 1:
        raise ValueError(f"the views have different frame counts {sorted(counts)}")
    if all(isinstance(s, torch.Tensor) for s in stacks):
        # Tiles prepared by the LeRobot preprocessor: the canvas rule ran already.
        for s in stacks:
            if tuple(s.shape[-2:]) != (tile_h, tile_w):
                raise ValueError(f"tensor frames must be {tile_h}x{tile_w} tiles, got {tuple(s.shape[-2:])}")
        tiles = [s.to(device) for s in stacks if isinstance(s, torch.Tensor)]
    elif any(isinstance(s, torch.Tensor) for s in stacks):
        raise ValueError("the views must all be prepared tiles or all raw frames")
    elif antialias:
        frames = [torch.from_numpy(np.ascontiguousarray(s)).permute(0, 3, 1, 2) for s in stacks]
        native = [(int(f.shape[-2]), int(f.shape[-1])) for f in frames]
        resize, trimmed = fit_row_sizes(native, RESTRICT_SIDE, SPATIAL_PATCH)
        if any(t != (tile_h, tile_w) for t in trimmed):
            raise ValueError(
                f"views of {native} fit to {trimmed}, not the {tile_h}x{tile_w} "
                "tiles of the checkpoint. praxis serves 16:10 views (for "
                "example 1920x1200 or 960x600) at their native aspect; other "
                "aspects are not supported yet"
            )
        tiles = [_trim(_resize(f, *r), tile_h, tile_w) for f, r in zip(frames, resize, strict=True)]
    else:
        tiles = [_tile_view(s, tile_h, tile_w, tol) for s in stacks]
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Proprio
# ---------------------------------------------------------------------------


def pose16_to_20d(pose16: np.ndarray, anchor16: np.ndarray | None = None) -> np.ndarray:
    """Convert pose16 rows ``(..., 16)`` to 20-D rows, one arm at a time.

    With ``anchor16`` each arm is relative to its own anchor pose.
    """
    arms = []
    for lo in (0, 8):
        anchor = None if anchor16 is None else anchor16[..., lo : lo + 8]
        arms.append(pose8_to_10d(pose16[..., lo : lo + 8], anchor))
    return np.concatenate(arms, axis=-1).astype(np.float32)


def proprio_tensors(
    poses: np.ndarray, stats: NormStats, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the expert proprio inputs of the boundary poses (float32).

    ``poses`` holds the ``K`` boundary pose16 rows, current pose last. The
    function returns the current proprio ``(1, 1, 20)`` and the proprio of
    all rows ``(1, K, 20)``.
    """
    p = proprio_normalize(pose16_to_20d(poses), stats.proprio_mean, stats.proprio_std)
    p = np.asarray(p).reshape(-1, p.shape[-1])
    every = torch.from_numpy(np.ascontiguousarray(p)).reshape(1, p.shape[0], -1)
    every = every.to(device, torch.float32)
    return every[:, -1:], every


# ---------------------------------------------------------------------------
# Action tokens -> rows -> poses
# ---------------------------------------------------------------------------


def rel20_to_pose12(rel: np.ndarray, pose16_now: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compose denormalized ``(T, 20)`` relative actions with the anchor pose of each arm.

    For each arm block ``[xyz3, rot6d6, grip1]`` (offset 0 left, 10 right):
    ``xyz_abs = xyz_rel + anchor_xyz`` in float64 and ``R_abs = R_rel @ R_anchor``,
    against that arm's current pose in ``pose16_now``. Returns one
    ``(poses (T, 12), grips (T,))`` pair per arm (left, then right), float32: the poses
    ``[xyz3, R_flat9]`` with a row-major rotation and the absolute grippers. The
    arithmetic is that of upstream's ``rel20_to_wire``, so the bits are those of its
    command dicts.
    """
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < ACTION_DIM:
        raise ValueError(f"rel actions must be (T, >={ACTION_DIM}), got {rel.shape}")
    pose16_now = np.asarray(pose16_now)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(SIDES)):
        a8 = pose16_now[8 * i : 8 * i + 8]
        anchor_xyz: np.ndarray = a8[:3].astype(np.float64)
        r_anchor = unit_quat_wxyz_to_matrix(a8[3:7])
        lo = ARM_DIM * i
        poses = np.empty((rel.shape[0], 12), dtype=np.float32)
        for t, r in enumerate(rel):
            xyz_abs = r[lo : lo + 3].astype(np.float64) + anchor_xyz
            r_abs = rot6d_to_matrix(r[lo + 3 : lo + 9]) @ r_anchor
            poses[t] = np.concatenate([xyz_abs, r_abs.reshape(-1)]).astype(np.float32)
        out.append((poses, rel[:, lo + 9].astype(np.float32)))
    return out


def decode_actions(
    pred_action: torch.Tensor,
    stats: NormStats,
    depth: int,
    dims: np.ndarray = Q99_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized rows and the relative actions of a chunk.

    The ``(1, L, 4 * 20)`` tokens unpack to 21 rows at depth 0 and 24 rows
    after. The relative actions clip ``dims`` to [-1, 1], apply the inverse
    q99 map on ``dims`` and clip the grippers to [0, 1].
    """
    rows = action_rows(pred_action, depth, FRAMES_PER_TOKEN, ACTION_DIM)
    return rows, clip_grip(denormalize_rows(rows, stats, dims), GRIP_DIMS)


# ---------------------------------------------------------------------------
# Serve session state (model free)
# ---------------------------------------------------------------------------


@dataclass
class BimanualChunkInputs:
    """Model inputs of one served chunk of the bimanual rig.

    The fields mirror :class:`praxis.single_arm.SingleArmChunkInputs` with
    pose16 rows in place of pose8 rows: ``boundary_poses`` holds the
    ``depth + 1`` pose16 rows at the chunk starts of the window (current pose
    last) and ``pose_now`` the current pose16. ``proprio`` is ``(1, 1, 20)``
    and ``proprio_all`` is ``(1, depth + 1, 20)``.
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


class BimanualServeSession:
    """Observed-context state machine of one bimanual rig session.

    The steps and the context rules are those of
    :class:`praxis.single_arm.SingleArmServeSession`: a prompt override, the
    executed frames of the previous chunk appended to the stream, the current
    frame encoded, then the window ``[anchor | d cleans]`` and the proprio.
    A view stack that is missing or too short for the executed chunk
    restarts the session (a cold start at depth 0).

    The poses stay in the world frame of the rig. The bimanual rig has one
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
        aspect_tol: float | None = None,
        antialias: bool = True,
    ):
        if control_mode not in CONTROL_MODES:
            raise ValueError(
                f"Unknown control_mode {control_mode!r} for the bimanual rig. The "
                f"known modes are {sorted(CONTROL_MODES)}."
            )
        th, tw = TILE_SIZE if tile_size is None else tile_size
        self.tile_size = (int(th), int(tw))
        self.aspect_tol = ASPECT_TOL if aspect_tol is None else float(aspect_tol)
        self.antialias = bool(antialias)
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(dims)
        self.control_mode = str(control_mode)
        self.exec_horizon = None if exec_horizon is None else int(exec_horizon)
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
        self._boundary_latents: list[torch.Tensor] = []
        self._boundary_poses: list[np.ndarray] = []
        self._cap = self.max_context_chunks + 1

    def reset(self) -> None:
        """Start a new episode (chunk counter back to 0, context dropped)."""
        self.chunk_index = 0
        self.restart()

    def restart(self) -> None:
        """Drop the context. The next call is a cold start at depth 0."""
        self.restart_count += 1
        self._boundary_latents = []
        self._boundary_poses = []
        self.stream.reset()

    def _clip(self, views: dict[str, np.ndarray]) -> torch.Tensor:
        th, tw = self.tile_size
        return canvas_clip(views, th, tw, self.aspect_tol, self.device, self.antialias)

    def prepare(self, obs: dict) -> BimanualChunkInputs:
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
        pose16 = extract_pose16(obs)
        stacks = view_stacks(obs)
        cur_clip = self._clip({v: s[-1:] for v, s in stacks.items()})
        if self.stateless:
            self.restart()
        elif self._boundary_latents:
            need = self.stream.expected_next_rows
            if any(s.shape[0] < need + 1 for s in stacks.values()):
                self.restart()
            else:
                clip = self._clip({v: s[-(need + 1) : -1] for v, s in stacks.items()})
                self.stream.append_executed(clip, pose16_history(obs, need))
        cur = self.encode(cur_clip)
        self._boundary_latents = [*self._boundary_latents, cur.detach()][-self._cap :]
        self._boundary_poses = [*self._boundary_poses, pose16][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        d = len(cleans)
        af = self.stream.anchor_frame()
        anchor = self._boundary_latents[-1] if af is None else self.encode(af)
        bp = self.stream.boundary_poses() if d else None
        if d == 0:
            poses: np.ndarray = pose16[None].astype(np.float32)
        elif bp is not None:
            poses = np.concatenate([bp, pose16[None]], axis=0).astype(np.float32)
        else:
            n = min(d + 1, len(self._boundary_poses))
            poses = np.stack(self._boundary_poses[-n:], axis=0).astype(np.float32)
        proprio, proprio_all = proprio_tensors(poses, self.stats, self.device)
        return BimanualChunkInputs(
            anchor=anchor,
            cleans=cleans,
            boundary_poses=poses,
            proprio=proprio,
            proprio_all=proprio_all,
            depth=d,
            pose_now=pose16,
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
        )

    def decode_actions(
        self, inputs: BimanualChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized rows and the relative actions of the commands."""
        return decode_actions(pred_action, self.stats, inputs.depth, self.dims)

    def finish(
        self, inputs: BimanualChunkInputs, pred_action: torch.Tensor
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Convert the sampled actions of this chunk to the ``[(poses, grips)]`` rows of each arm and advance.

        The rows are cut to ``exec_horizon``. (Upstream returns the command dicts of
        ``rel20_to_wire``; LeRobot's command processor step makes the robot commands.)
        """
        _rows, rel = self.decode_actions(inputs, pred_action)
        arms = rel20_to_pose12(rel, inputs.pose_now)
        self.chunk_index += 1
        if self.exec_horizon is None:
            return arms
        return [(poses[: self.exec_horizon], grips[: self.exec_horizon]) for poses, grips in arms]
