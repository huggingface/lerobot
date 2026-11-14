# Vendored from runwayml/praxis-1-oss at commit 8dad1a4 plus the canvas rule of the released DROID checkpoint,
# original path praxis/single_arm.py, reduced to what the LeRobot policy uses: the flange-frame
# client (no client frame table, forward-kinematics check or control modes; the frame conversion
# stays as the identity with its operations, for bit equality), poses and grippers as arrays
# instead of command dicts, and frame stacks given as prepared uint8 tile tensors. The quaternion
# conversions live in `lerobot.utils.rotation`. Keep the kept functions in sync with upstream.
"""Robot input and output glue for the single-arm rig. No neural networks.

Observation contract (the policy builds it from its frame history):

* ``robot_state.ee_pose``: end-effector pose, 7 values
  ``[x, y, z, qw, qx, qy, qz]`` (quaternion scalar first), metres.
* ``grip``: absolute gripper value, 0 = open.
* ``image.exterior`` and ``image.wrist``: RGB uint8 frames ``(T, H, W, 3)``, or
  the tiles of the LeRobot preprocessor as ``(T, 3, tile_h, tile_w)`` uint8
  tensors on any device. Both views are needed.
* Each value can carry a leading time axis (a frame stack). The current
  frame is the last row.

Model space:

* pose8 = ``[xyz3, quat_wxyz4, grip1]``.
* 10-D = ``[xyz3, rot6d6, grip1]``. rot6d holds the first two columns of the
  rotation matrix.
* Actions are relative to the arm pose at the start of the chunk
  (``xyz - anchor_xyz``, ``R @ R_anchor^T``). The gripper is absolute.
* Actions are q99-normalized on the xyz and grip dims. rot6d stays identity.
  Proprio is mean/std-normalized on all 10 dims.
* The decode clips the normalized xyz and grip dims to [-1, 1], the range of
  the training targets. After the inverse q99 map, it clips the gripper to
  [0, 1].
* The canvas is the tile canvas. Each view gets its own tile (256x416 by
  default, 224x416 with the antialiased fit rule): a centre crop, one
  bilinear resize, a round to uint8. The exterior tile and the wrist tile go
  side by side.
* The model works in the flange frame without the turn: the Franka flange
  position in robot-base coordinates and the flange rotation, which is what
  LeRobot clients report (the DROID convention).

Output: the absolute poses ``[xyz3, R_flat9]`` (row-major) and grippers of a
chunk, one row per action at 15 Hz. Chunk 0 of a context window gives 21
rows. Every later chunk gives 24 rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from lerobot.policies.praxis.praxis_core.robot_io import (
    ACTION_CHUNK,
    ASPECT_TOL,
    FPS,
    FRAMES_PER_TOKEN,
    ContiguousStreamContext,
    NormStats,
    action_rows,
    aspect_crop,
    clip_grip,
    denormalize_rows,
    latent_frames,
    pose8_to_10d,
    proprio_normalize,
    rot6d_to_matrix,
    text_encoder_prompt,
    unit_quat_wxyz_to_matrix,
)
from lerobot.utils.rotation import Rotation

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

VIEWS: tuple[str, ...] = ("exterior", "wrist")
ACTION_DIM = 10
N_CHUNKS = 4
MAX_CONTEXT_CHUNKS = 3

# The tile canvas: one tile (height, width) per view, side by side.
TILE_SIZE: tuple[int, int] = (256, 416)
# The native DROID view (height, width). The fit rule of an antialiased canvas
# crops each view to its aspect before the resize (see :func:`fit_resize_size`).
NATIVE_VIEW: tuple[int, int] = (720, 1280)
# Frames per resize call of the tile canvas (bounds the float32 memory).
TILE_CHUNK_FRAMES = 32

# The turn of the model frame (the flange frame): none.
NO_TURN = np.eye(3)

# Dims of a 10-D row that use the q99 normalization (xyz and grip).
Q99_DIMS = np.array([0, 1, 2, 9])
# The dim of a 10-D row that holds the gripper.
GRIP_DIMS = np.array([9])

POSE_KEYS: tuple[str, ...] = ("ee_pose", "robot_state.ee_pose")
GRIP_KEYS: tuple[str, ...] = ("grip", "robot_state.grip", "gripper")
VIEW_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "exterior": ("exterior_image", "image.exterior", "observation.images.side"),
    "wrist": ("wrist_image", "image.wrist", "observation.images.left"),
}


# ---------------------------------------------------------------------------
# Checkpoint config
# ---------------------------------------------------------------------------


def canvas_size(config: dict) -> tuple[int, int]:
    """Return the canvas ``(height, width)`` of a single-arm checkpoint config.

    The config must declare the tile canvas of the views ``exterior`` and
    ``wrist``. The result is ``(tile_height, 2 * tile_width)``. Another canvas
    raises ``ValueError``.
    """
    video = config.get("video", {})
    views = tuple(video.get("views", VIEWS))
    canvas = video.get("canvas")
    if views != VIEWS or canvas != "tiles":
        raise ValueError(
            f"The single-arm rig supports the tile canvas of {VIEWS}. The config "
            f"has views={views} canvas={canvas!r}. This canvas is not "
            "supported; use the single-arm checkpoint."
        )
    th = int(video.get("tile_height", TILE_SIZE[0]))
    tw = int(video.get("tile_width", TILE_SIZE[1]))
    return th, len(VIEWS) * tw


def q99_dims(config: dict) -> np.ndarray:
    """Return the q99-normalized action dims of a single-arm checkpoint config."""
    norm = config.get("normalization", {})
    return np.asarray(norm.get("q99_normed_dims", Q99_DIMS), dtype=np.int64)


# ---------------------------------------------------------------------------
# Observation parsing
# ---------------------------------------------------------------------------


def _pick(obs: dict, candidates: tuple[str, ...]) -> object:
    for key in candidates:
        if key in obs:
            return obs[key]
    raise KeyError(f"obs missing any of {candidates}; got keys={list(obs)}")


def _stack(value: object) -> np.ndarray | torch.Tensor:
    """Return a frame stack as given: a tensor stays a tensor (tiles), anything else becomes uint8."""
    if isinstance(value, torch.Tensor):
        return value
    return np.asarray(value).astype(np.uint8)


def extract_pose8(obs: dict) -> np.ndarray:
    """Return the current pose8 ``(8,)``: the last row of the pose and grip stacks.

    The grip can be a scalar, ``(1,)``, a stack ``(T,)`` or a stack
    ``(T, 1)``. The current grip is the last value of a stack.
    """
    ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
    if ee.ndim >= 2:
        ee = ee[-1]
    ee = ee.ravel()
    if ee.shape[0] < 7:
        raise ValueError(f"ee_pose must be (>=7,), got {ee.shape}")
    grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    if grip.ndim >= 2:
        grip = grip[-1]
    elif grip.ndim == 1 and grip.shape[0] > 1:
        grip = grip[-1:]
    grip = grip.ravel()
    return np.concatenate([ee[:7], grip[:1]]).astype(np.float32)


def pose8_history(obs: dict, need: int) -> np.ndarray | None:
    """Return the ``(need, 8)`` pose8 rows of the executed frames, or ``None``.

    The rows are stack rows ``[-(need + 1):-1]`` (the current row is not
    included). ``None`` means that the observation has no pose stack or that
    the stack is too short.
    """
    try:
        ee = np.asarray(_pick(obs, POSE_KEYS), dtype=np.float32)
        grip = np.asarray(_pick(obs, GRIP_KEYS), dtype=np.float32)
    except KeyError:
        return None
    if ee.ndim != 2 or ee.shape[0] < need + 1 or ee.shape[1] < 7:
        return None
    g = grip.reshape(grip.shape[0], -1) if grip.ndim >= 1 else None
    if g is None or g.shape[0] < need + 1:
        return None
    rows = np.concatenate([ee[-(need + 1) : -1, :7], g[-(need + 1) : -1, :1]], axis=1)
    return rows.astype(np.float32)


def latest_frame(obs: dict, view: str) -> np.ndarray | torch.Tensor:
    """Return the current frame of one view: ``(H, W, 3)`` uint8, or a ``(3, H, W)`` uint8 tile tensor.

    A missing view raises ``KeyError``.
    """
    value = _pick(obs, VIEW_KEY_CANDIDATES[view])
    if isinstance(value, torch.Tensor):
        tile = value[-1] if value.ndim == 4 else value
        if tile.ndim != 3 or tile.shape[0] != 3 or tile.dtype != torch.uint8:
            raise ValueError(
                f"a tensor frame must be a (3, H, W) uint8 tile, got {tuple(tile.shape)} {tile.dtype}"
            )
        return tile
    img = np.asarray(value)
    if img.ndim == 4:
        img = img[-1]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be (H,W,3), got {img.shape}")
    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


def fit_resize_size(tile_w: int) -> tuple[int, int]:
    """Return the ``(height, width)`` of the fit resize for tiles ``tile_w`` wide.

    The native view scaled down so that its width is ``tile_w``, the height
    rounded down: ``(234, 416)`` for 416-wide tiles. It is also the size the
    client restricts each view to for an antialiased canvas.
    """
    nh, nw = NATIVE_VIEW
    return min(nh, tile_w, nh * tile_w // nw), min(nw, tile_w)


def _tile_view(
    frames: np.ndarray | torch.Tensor, tile_h: int, tile_w: int, tol: float, antialias: bool = False
) -> torch.Tensor:
    """Return the ``(T, 3, tile_h, tile_w)`` uint8 tile of one view.

    ``frames`` is ``(T, H, W, 3)`` or ``(H, W, 3)`` uint8, or a ``(T, 3, tile_h, tile_w)`` uint8
    tensor of tiles prepared already (returned as is, on its device). All resizes are
    bilinear in float32 with ``align_corners=False``, then a round and a clamp
    to uint8. ``antialias`` (the ``video.antialias`` of the checkpoint config)
    selects the canvas rule of the checkpoint's training:

    * off: a centre crop to the tile aspect (:func:`praxis.robot_io.aspect_crop`,
      no crop when the aspects agree within ``tol``) and one resize to the tile.
    * on (the fit rule): a centre crop to the native view aspect, one
      antialiased resize to :func:`fit_resize_size` and a centre trim to the
      tile (234x416 -> 224x416 for `praxis-1-droid`). An input already at the fit size
      is not resized.
    """
    if isinstance(frames, torch.Tensor):
        # Tiles prepared by the LeRobot preprocessor, on any device: the canvas rule ran already.
        tiles = frames if frames.ndim == 4 else frames[None]
        if tiles.ndim != 4 or tiles.shape[1] != 3 or tiles.dtype != torch.uint8:
            raise ValueError(f"tensor frames must be (T, 3, H, W) uint8 tiles, got {tuple(frames.shape)}")
        if tuple(tiles.shape[-2:]) != (tile_h, tile_w):
            raise ValueError(f"tensor frames must be {tile_h}x{tile_w} tiles, got {tuple(tiles.shape[-2:])}")
        return tiles
    a = np.asarray(frames, np.uint8)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"frames must be (T,H,W,3), got {a.shape}")
    fr = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
    if antialias:
        return _fit_view(fr, tile_h, tile_w, tol)
    cropped = aspect_crop(fr, tile_h, tile_w, tol)
    out = torch.cat(
        [
            torch.nn.functional.interpolate(
                chunk.float(),
                size=(tile_h, tile_w),
                mode="bilinear",
                align_corners=False,
                antialias=antialias,
            )
            for chunk in cropped.split(TILE_CHUNK_FRAMES, dim=0)
        ]
    )
    return out.round().clamp(0, 255).to(torch.uint8)


def _fit_view(fr: torch.Tensor, tile_h: int, tile_w: int, tol: float) -> torch.Tensor:
    """Return the fit-rule tile ``(T, 3, tile_h, tile_w)`` of ``fr`` (uint8)."""
    rh, rw = fit_resize_size(tile_w)
    if rh < tile_h or rw < tile_w:
        raise ValueError(f"the fit size {rh}x{rw} is below the tile {tile_h}x{tile_w}")
    cropped = aspect_crop(fr, *NATIVE_VIEW, tol)
    if tuple(cropped.shape[-2:]) != (rh, rw):
        cropped = torch.cat(
            [
                torch.nn.functional.interpolate(
                    chunk.float(),
                    size=(rh, rw),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                for chunk in cropped.split(TILE_CHUNK_FRAMES, dim=0)
            ]
        )
    y0, x0 = (rh - tile_h) // 2, (rw - tile_w) // 2
    return cropped[..., y0 : y0 + tile_h, x0 : x0 + tile_w].contiguous()


def tile_clip(
    exterior: np.ndarray | torch.Tensor,
    wrist: np.ndarray | torch.Tensor | None,
    tile_h: int = TILE_SIZE[0],
    tile_w: int = TILE_SIZE[1],
    tol: float = ASPECT_TOL,
    device: torch.device | str = "cpu",
    antialias: bool | tuple[bool, bool] = False,
) -> torch.Tensor:
    """Convert frames to a ``(1, 3, T, tile_h, 2 * tile_w)`` clip of the tile canvas.

    ``exterior`` and ``wrist`` are ``(T, H, W, 3)`` or ``(H, W, 3)`` uint8
    frames with the same ``T``. Each view becomes one uint8 tile
    (:func:`_tile_view`). The exterior tile goes left and the wrist tile
    right. The clip is float32 with integer values in [0, 255]. The tile
    canvas needs both views: ``wrist=None`` raises ``ValueError``.
    ``antialias`` is the canvas rule of both views, or one flag per view.
    """
    if wrist is None:
        raise ValueError("The tile canvas needs the exterior and the wrist view.")
    rules = (antialias, antialias) if isinstance(antialias, bool) else tuple(antialias)
    tiles = [
        _tile_view(v, tile_h, tile_w, tol, rule) for v, rule in zip((exterior, wrist), rules, strict=True)
    ]
    if tiles[0].shape[0] != tiles[1].shape[0]:
        raise ValueError(f"exterior has {tiles[0].shape[0]} frames, wrist has {tiles[1].shape[0]}")
    canvas = torch.cat(tiles, dim=-1).float()
    return canvas.permute(1, 0, 2, 3).unsqueeze(0).to(device)


def _tile_args(tile_size: tuple[int, int] | None, aspect_tol: float | None) -> tuple[int, int, float]:
    th, tw = TILE_SIZE if tile_size is None else tile_size
    return int(th), int(tw), ASPECT_TOL if aspect_tol is None else float(aspect_tol)


def current_clip(
    obs: dict,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
    antialias: bool = False,
) -> torch.Tensor:
    """Return the tile clip ``(1, 3, 1, H, W)`` of the current frames.

    See :func:`tile_clip`. A missing wrist frame raises ``ValueError``.
    """
    exterior = latest_frame(obs, "exterior")
    try:
        wrist = latest_frame(obs, "wrist")
    except KeyError:
        wrist = None
    th, tw, tol = _tile_args(tile_size, aspect_tol)
    return tile_clip(exterior[None], None if wrist is None else wrist[None], th, tw, tol, device, antialias)


def executed_clip(
    obs: dict,
    need: int,
    device: torch.device | str = "cpu",
    tile_size: tuple[int, int] | None = None,
    aspect_tol: float | None = None,
    antialias: bool = False,
) -> tuple[torch.Tensor, np.ndarray | None] | None:
    """Return the clip and the pose8 rows of the ``need`` executed raws, or ``None``.

    The frames are stack rows ``[-(need + 1):-1]``. ``None`` means that the
    exterior stack or the wrist stack is missing or too short. The pose rows
    are ``None`` when the observation has no pose stack. They are in the frame
    of the rig.
    """
    ext = _stack(_pick(obs, VIEW_KEY_CANDIDATES["exterior"]))
    if ext.ndim != 4 or ext.shape[0] < need + 1:
        return None
    ext_chunk = ext[-(need + 1) : -1]
    wrist_chunk = None
    try:
        wrist = _stack(_pick(obs, VIEW_KEY_CANDIDATES["wrist"]))
        if wrist.ndim == 4 and wrist.shape[0] >= need + 1:
            wrist_chunk = wrist[-(need + 1) : -1]
    except KeyError:
        wrist_chunk = None
    if wrist_chunk is None:
        return None
    th, tw, tol = _tile_args(tile_size, aspect_tol)
    clip = tile_clip(ext_chunk, wrist_chunk, th, tw, tol, device, antialias)
    return clip, pose8_history(obs, need)


# ---------------------------------------------------------------------------
# Pose frame
# ---------------------------------------------------------------------------


def pose8_to_model(pose8: np.ndarray) -> np.ndarray:
    """Normalize client pose8 rows ``(..., 8)`` as the praxis session does for the flange frame.

    LeRobot clients report the Franka flange itself (the DROID convention): no tool offset, no
    turn, no base offset. The praxis session still routes the quaternion through the rotation
    matrix (:meth:`Rotation.quat_xyzw_to_matrix`, then :meth:`Rotation.matrix_to_quat_xyzw` with the sign of
    scipy's largest-component branch) and clips the gripper to [0, 1]. The LeRobot port keeps
    these operations with the zero offsets, so its poses have the bits of the praxis session.
    The result is float32.
    """
    out = np.array(pose8, dtype=np.float64, copy=True)
    rot = Rotation.quat_xyzw_to_matrix(out[..., [4, 5, 6, 3]])
    base: np.ndarray = np.zeros(3, dtype=np.float64)
    out[..., :3] = out[..., :3] - base - 0.0 * rot[..., :, 2]
    turn = NO_TURN.T @ NO_TURN
    out[..., 3:7] = Rotation.matrix_to_quat_xyzw(rot @ turn)[..., [3, 0, 1, 2]]
    out[..., 7] = np.clip(out[..., 7], 0.0, 1.0)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Proprio
# ---------------------------------------------------------------------------


def proprio_rows(pose8: np.ndarray, stats: NormStats) -> np.ndarray:
    """Return the normalized absolute 10-D proprio of pose8 rows (float32)."""
    return proprio_normalize(pose8_to_10d(pose8), stats.proprio_mean, stats.proprio_std)


def proprio_tensors(
    poses: np.ndarray, stats: NormStats, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the expert proprio inputs of the boundary poses (float32).

    ``poses`` holds the ``K`` boundary pose8 rows, current pose last. The
    function returns the current proprio ``(1, 1, 10)`` and the proprio of
    all rows ``(1, K, 10)``.
    """
    p = proprio_rows(poses, stats)
    p = np.asarray(p).reshape(-1, p.shape[-1])
    last = torch.from_numpy(np.ascontiguousarray(p[-1:])).reshape(1, 1, -1).to(device, torch.float32)
    every = torch.from_numpy(np.ascontiguousarray(p)).reshape(1, p.shape[0], -1).to(device, torch.float32)
    return last, every


# ---------------------------------------------------------------------------
# Action tokens -> rows -> poses
# ---------------------------------------------------------------------------


def rel10_to_pose12(rel: np.ndarray, pose8_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compose denormalized ``(T, 10)`` relative actions with the anchor pose.

    ``xyz_abs = xyz_rel + anchor_xyz`` in float64 and ``R_abs = R_rel @ R_anchor``. The anchor is
    the current pose ``pose8_now`` in the flange frame of the model. Returns the poses
    ``(T, 12)`` float32 as ``[xyz3, R_flat9]`` with a row-major rotation, and the absolute
    grippers ``(T,)`` float32. The frame conversion of the praxis session is the identity for the
    flange frame; its operations stay, with zero offsets, so the bits are those of the session.
    """
    rel = np.asarray(rel, dtype=np.float32)
    if rel.ndim != 2 or rel.shape[1] < ACTION_DIM:
        raise ValueError(f"rel actions must be (T, >={ACTION_DIM}), got {rel.shape}")
    anchor = np.asarray(pose8_now)
    anchor_xyz = anchor[:3].astype(np.float64)
    r_anchor = unit_quat_wxyz_to_matrix(anchor[3:7])
    poses = np.empty((rel.shape[0], 12), dtype=np.float32)
    for t, r in enumerate(rel):
        xyz_abs = r[:3].astype(np.float64) + anchor_xyz
        r_abs = rot6d_to_matrix(r[3:9]) @ r_anchor
        xyz_abs = xyz_abs + 0.0 * r_abs[:, 2] + np.zeros(3)
        r_abs = r_abs @ NO_TURN.T @ NO_TURN
        poses[t] = np.concatenate([xyz_abs, r_abs.reshape(-1)]).astype(np.float32)
    return poses, rel[:, 9].astype(np.float32)


def decode_actions(
    pred_action: torch.Tensor,
    stats: NormStats,
    depth: int,
    dims: np.ndarray = Q99_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized rows and the relative actions of the commands of a chunk.

    The function unpacks the ``(1, L, 4 * 10)`` tokens to 21 rows at depth 0
    and 24 rows after (:func:`praxis.robot_io.action_rows`). These normalized
    rows have no clip. The relative actions come from these steps:

    1. Clip ``dims`` to [-1, 1], the range of the training targets.
    2. Apply the inverse q99 map on ``dims``
       (:func:`praxis.robot_io.denormalize_rows`).
    3. Clip the gripper to [0, 1] (:func:`praxis.robot_io.clip_grip`).
    """
    rows = action_rows(pred_action, depth, FRAMES_PER_TOKEN, ACTION_DIM)
    return rows, clip_grip(denormalize_rows(rows, stats, dims), GRIP_DIMS)


# ---------------------------------------------------------------------------
# Serve session state (model free)
# ---------------------------------------------------------------------------


@dataclass
class SingleArmChunkInputs:
    """Model inputs of one served chunk of the single-arm rig.

    ``anchor`` is the 1-latent window anchor and ``cleans`` holds ``depth``
    clean chunk latents (oldest first). ``boundary_poses`` holds the
    ``depth + 1`` pose8 rows at the chunk starts of the window (current pose
    last). ``boundary_poses`` and ``pose8_now`` are in the flange frame of
    the model. ``proprio`` is the normalized current pose ``(1, 1, 10)``
    float32. ``proprio_all`` is the normalized proprio of all boundary poses
    ``(1, depth + 1, 10)`` float32. ``prompt`` is the string for the text
    encoder.
    """

    anchor: torch.Tensor
    cleans: list[torch.Tensor]
    boundary_poses: np.ndarray
    proprio: torch.Tensor
    proprio_all: torch.Tensor
    depth: int
    pose8_now: np.ndarray
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


class SingleArmServeSession:
    """Observed-context state machine of one single-arm rig session.

    Each call of :meth:`prepare` does these steps:

    1. Apply a prompt override (``obs["task"]`` or ``obs["prompt"]``).
    2. Append the executed frames of the previous chunk to the stream.
    3. Encode the current frame.
    4. Build the window ``[anchor | d cleans]`` and the proprio.

    After the sampler ran, :meth:`finish` converts the actions to poses and grippers.
    The sampler uses the seed ``base_seed + chunk_index`` for a chunk.

    The prompt goes to the text encoder verbatim. An override loses its
    leading and trailing white space.

    ``encode`` maps a prepared clip ``(1, 3, T, H, W)`` in [0, 255] to video
    latents ``(1, C, T', H', W')``. The clips are tile clips
    (:func:`tile_clip`, ``tile_size`` and ``aspect_tol`` default to 256x416
    and 0.03, ``antialias`` to off); frames given as uint8 tile tensors are
    tiles already and stay on their device. A missing current wrist frame raises
    ``ValueError``. An exterior or wrist stack that is missing or too short for
    the executed chunk restarts the session (a cold start at depth 0).

    ``exec_horizon`` cuts the commands of a chunk to that many rows. A value
    below 24 moves the next observation off the chunk grid of the context
    window. The session then serves every call as a cold start.
    ``max_context_chunks=0`` has the same effect.

    The client reports the Franka flange pose (the DROID convention); the
    session normalizes it as the praxis session does (:func:`pose8_to_model`).
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
        exec_horizon: int | None = None,
        n_chunks: int = N_CHUNKS,
        tile_size: tuple[int, int] | None = None,
        aspect_tol: float | None = None,
        antialias: bool = False,
    ):
        self.antialias = bool(antialias)
        self.tile_size = None if tile_size is None else (int(tile_size[0]), int(tile_size[1]))
        self.aspect_tol = None if aspect_tol is None else float(aspect_tol)
        self.encode = encode
        self.stats = stats
        self.device = device
        self.prompt = prompt
        self.fps = float(fps)
        self.dims = np.asarray(dims)
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

    def prepare(self, obs: dict) -> SingleArmChunkInputs:
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
        pose8 = pose8_to_model(extract_pose8(obs))
        cur_clip = current_clip(obs, self.device, self.tile_size, self.aspect_tol, self.antialias)
        if self.stateless:
            self.restart()
        elif self._boundary_latents:
            need = self.stream.expected_next_rows
            got = executed_clip(obs, need, self.device, self.tile_size, self.aspect_tol, self.antialias)
            if got is None:
                self.restart()
            else:
                clip, rows = got
                if rows is not None:
                    rows = pose8_to_model(rows)
                self.stream.append_executed(clip, rows)
        cur = self.encode(cur_clip)
        self._boundary_latents = [*self._boundary_latents, cur.detach()][-self._cap :]
        pose = np.asarray(pose8, dtype=np.float32)
        self._boundary_poses = [*self._boundary_poses, pose][-self._cap :]
        cleans = self.stream.cleans(self.encode)
        d = len(cleans)
        af = self.stream.anchor_frame()
        anchor = self._boundary_latents[-1] if af is None else self.encode(af)
        bp = self.stream.boundary_poses() if d else None
        poses: np.ndarray
        if d == 0:
            poses = pose8[None].astype(np.float32)
        elif bp is not None:
            poses = np.concatenate([bp, pose8[None]], axis=0).astype(np.float32)
        else:
            n = min(d + 1, len(self._boundary_poses))
            poses = np.stack(self._boundary_poses[-n:], axis=0).astype(np.float32)
        proprio, proprio_all = proprio_tensors(poses, self.stats, self.device)
        return SingleArmChunkInputs(
            anchor=anchor,
            cleans=cleans,
            boundary_poses=poses,
            proprio=proprio,
            proprio_all=proprio_all,
            depth=d,
            pose8_now=pose8,
            chunk_index=self.chunk_index,
            prompt=text_encoder_prompt(self.prompt),
        )

    def decode_actions(
        self, inputs: SingleArmChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the normalized rows and the relative actions of the commands.

        See :func:`decode_actions`. The normalized rows have no clip.
        """
        return decode_actions(pred_action, self.stats, inputs.depth, self.dims)

    def finish(
        self, inputs: SingleArmChunkInputs, pred_action: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the sampled actions of this chunk to poses and grippers and advance.

        Returns the absolute poses ``(T, 12)`` float32 (``[xyz3, R_flat9]``, row-major) and the
        grippers ``(T,)`` float32 (:func:`rel10_to_pose12`). With ``exec_horizon`` set, only the
        first ``exec_horizon`` rows come back.
        """
        _rows, rel = self.decode_actions(inputs, pred_action)
        poses, grips = rel10_to_pose12(rel, inputs.pose8_now)
        self.chunk_index += 1
        if self.exec_horizon is not None:
            poses, grips = poses[: self.exec_horizon], grips[: self.exec_horizon]
        return poses, grips
