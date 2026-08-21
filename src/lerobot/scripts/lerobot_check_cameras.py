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

"""Pin camera mounts to a reference pose, and put them back after they move.

Camera extrinsics drift. A tripod gets nudged, a wrist mount is re-clamped after a repair, the rig
moves to another table. A policy trained on the old viewpoint then fails silently: there is no
exception and no metric that moves, the robot just does the wrong thing. This saves one reference
frame per camera, and afterwards reports how far each camera has drifted and which way to push it
back.

Run ``--mode=save`` once, when the rig is in a state you are happy to record from. Run
``--mode=check`` before a session, after any hardware work, and after moving the robot.

Because an eye-in-hand camera's view is a function of the joint angles, the reference stores the
arm pose it was captured at and ``check`` refuses to score until the arm is back there. Without
that gate the tool reports arm motion as camera motion.

Requires: pip install 'lerobot[hardware]'

Example:
    ```shell
    lerobot-check-cameras --mode=save \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_awesome_follower_arm \
        --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'

    lerobot-check-cameras --mode=check \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM0 \
        --robot.id=my_awesome_follower_arm \
        --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'
    ```
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat

import cv2
import draccus
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    rebot_b601_follower,
    so_follower,
)
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

CAMERAS = "cameras"
REFERENCE_FILENAME = "reference.json"

# A homography needs a healthy number of agreeing correspondences before its decomposition means
# anything. Below this the fit is dominated by whatever few features happen to survive, and it
# reports large, confident, wrong numbers.
MIN_INLIERS = 25

# The robot region of the live frame is searched over a larger box than the one drawn on the
# reference, so a roughly re-mounted camera still has the robot inside the search area.
ROI_SEARCH_SCALE = 1.6


@dataclass
class MountTolerance:
    """How far a camera may drift before ``check`` calls it out.

    The defaults sit just above the mount-to-mount noise we measured on an SO-101 rig: across eight
    same-session recordings a fixed external camera reproduced to under 3 px, 0.2 deg and 0.4% zoom.
    Tighten them if your policy is more viewpoint-sensitive than that; loosen them if a healthy rig
    keeps failing the check.

    Args:
        shift_px (`float`, *optional*, defaults to `4.0`):
            Maximum translation of the image centre, in pixels of the camera's own resolution.
        roll_deg (`float`, *optional*, defaults to `0.4`):
            Maximum in-plane rotation, in degrees.
        zoom (`float`, *optional*, defaults to `0.01`):
            Maximum scale change, as a fraction. `0.01` is one percent.
        pose_deg (`float`, *optional*, defaults to `2.0`):
            Maximum deviation of any joint from the pose the reference was captured at. Cameras are
            not scored until the arm is within this, because arm motion is indistinguishable from
            camera motion for anything the arm is bolted to or large in the frame of.
    """

    shift_px: float = 4.0
    roll_deg: float = 0.4
    zoom: float = 0.01
    pose_deg: float = 2.0


@dataclass
class MountShift:
    """How the live view differs from the reference, and how well that was measured.

    All four displacement fields describe what the live image *content* must do to land back on the
    reference. The camera has to move the other way; :func:`describe_correction` does that flip.

    **Attributes**:
        - **dx** (`float`) -- Horizontal shift in pixels, positive rightwards.
        - **dy** (`float`) -- Vertical shift in pixels, positive downwards.
        - **roll_deg** (`float`) -- In-plane rotation in degrees. Image coordinates run y-down, so a
          positive value means the content must turn *clockwise* on screen, and the camera therefore
          has to be rolled counter-clockwise to undo it. Do not "correct" this to match the printed
          instruction: they are opposites on purpose.
        - **zoom** (`float`) -- Scale factor. Above 1 the content must grow.
        - **inliers** (`int`) -- Feature correspondences agreeing with the fit.
        - **via** (`str`) -- Which region produced the winning fit, `"frame"` or `"robot"`.
        - **homography** (`np.ndarray`) -- The 3x3 matrix mapping live pixels to reference pixels.
    """

    dx: float
    dy: float
    roll_deg: float
    zoom: float
    inliers: int
    via: str
    homography: NDArray[np.float64] = field(repr=False)

    @property
    def shift_px(self) -> float:
        """`float`: Magnitude of the translation, in pixels."""
        return float(np.hypot(self.dx, self.dy))

    def within(self, tol: MountTolerance) -> bool:
        """Whether every component is inside the given tolerance.

        Args:
            tol (`MountTolerance`):
                The limits to compare against.

        Returns:
            `bool`: `True` if the camera does not need moving.
        """
        return (
            self.shift_px <= tol.shift_px
            and abs(self.roll_deg) <= tol.roll_deg
            and abs(self.zoom - 1.0) <= tol.zoom
        )


@dataclass
class CheckCamerasConfig:
    """Configuration for :func:`check_cameras`.

    Args:
        robot (`RobotConfig`):
            The robot whose cameras are being pinned. Every RGB camera it exposes is checked.
        mode (`str`, *optional*, defaults to `"check"`):
            `"save"` writes a new reference, `"check"` compares the live cameras against it.
        tolerance (`MountTolerance`, *optional*):
            Drift limits. Defaults to `MountTolerance()`.
        select_roi (`bool`, *optional*, defaults to `True`):
            At save time, ask for a box around the robot in each view. That box is the fallback
            matcher when the rest of the scene has changed, which is what happens when the rig moves
            to a different table. Ignored when no GUI is available.
        live (`bool`, *optional*, defaults to `False`):
            At check time, keep measuring and draw an overlay per camera so the mount can be nudged
            until it goes green. Requires a GUI. Without it the check takes one reading and exits.
        drive_to_pose (`bool`, *optional*, defaults to `False`):
            At check time, move the arm to the saved pose instead of refusing when it is elsewhere.
            Off by default because it moves real hardware; make sure the path is clear.
        force (`bool`, *optional*, defaults to `False`):
            Overwrite an existing reference in save mode.
    """

    robot: RobotConfig
    mode: str = "check"
    tolerance: MountTolerance = field(default_factory=MountTolerance)
    select_roi: bool = True
    live: bool = False
    drive_to_pose: bool = False
    force: bool = False

    def __post_init__(self):
        if self.mode not in ("save", "check"):
            raise ValueError(f"mode must be 'save' or 'check', got {self.mode!r}.")


def _gui_available() -> bool:
    """Whether OpenCV can open a window.

    The ``opencv-python-headless`` wheel LeRobot depends on has no GUI on most platforms, so every
    interactive step has to degrade instead of crashing.
    """
    try:
        cv2.namedWindow("__lerobot_gui_probe__")
        cv2.destroyWindow("__lerobot_gui_probe__")
    except cv2.error:
        return False
    return True


def _box_mask(shape: tuple[int, ...], roi: list[int], scale: float) -> NDArray[np.uint8]:
    """A `[x, y, width, height]` box as a mask, grown about its own centre by ``scale``."""
    x, y, w, h = roi
    cx, cy = x + w / 2, y + h / 2
    mask = np.zeros(shape[:2], np.uint8)
    mask[
        max(0, int(cy - h * scale / 2)) : int(cy + h * scale / 2),
        max(0, int(cx - w * scale / 2)) : int(cx + w * scale / 2),
    ] = 255
    return mask


def _fit_homography(
    live: NDArray[np.uint8],
    reference: NDArray[np.uint8],
    live_mask: NDArray[np.uint8] | None = None,
    reference_mask: NDArray[np.uint8] | None = None,
) -> tuple[NDArray[np.float64] | None, int]:
    """Solve for the live-to-reference transform, returning it with its RANSAC inlier count.

    SIFT rather than ORB: the two are comparable over a whole frame, but on the small robot-only
    region that carries the new-table case ORB's translation error was around 5 px against a known
    20 px ground truth, where SIFT's was around 1 px.
    """
    sift = cv2.SIFT_create(4000)
    live_kp, live_desc = sift.detectAndCompute(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), live_mask)
    ref_kp, ref_desc = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), reference_mask)
    if live_desc is None or ref_desc is None or len(live_kp) < 8 or len(ref_kp) < 8:
        return None, 0

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(live_desc, ref_desc, k=2)
    # Lowe's ratio test: keep a match only when the best candidate is clearly better than the second.
    good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return None, len(good)

    src = np.float32([live_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return homography, (0 if inlier_mask is None else int(inlier_mask.sum()))


def decompose_homography(homography: NDArray[np.float64], width: int, height: int) -> dict[str, float]:
    """Turn a homography into the four numbers a human can act on.

    The matrix is applied to the image corners and the displacement is split into a translation, an
    in-plane rotation and a scale. This is a deliberate simplification: a real mount also moves in
    depth and out of plane, and a wide-angle lens is not a pinhole. It holds well over the small
    displacements that matter when re-seating a mount, and degrades for large ones.

    Args:
        homography (`np.ndarray`):
            3x3 matrix mapping live pixels to reference pixels.
        width (`int`):
            Frame width in pixels.
        height (`int`):
            Frame height in pixels.

    Returns:
        `dict[str, float]`: Keys `dx`, `dy`, `roll_deg` and `zoom`, describing what the live image
        content must do to land on the reference.
    """
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    corners = corners.reshape(-1, 2)

    dx, dy = warped.mean(0) - corners.mean(0)

    def polygon_area(quad: NDArray[np.float32]) -> float:
        return 0.5 * abs(
            np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))
        )

    def edge_angle(quad: NDArray[np.float32], i: int, j: int) -> float:
        return float(np.degrees(np.arctan2(quad[j, 1] - quad[i, 1], quad[j, 0] - quad[i, 0])))

    # Average the two horizontal edges so a pure perspective tilt does not read as roll.
    roll_deg = 0.5 * (
        (edge_angle(warped, 0, 1) - edge_angle(corners, 0, 1))
        + (edge_angle(warped, 3, 2) - edge_angle(corners, 3, 2))
    )
    return {
        "dx": float(dx),
        "dy": float(dy),
        "roll_deg": float(roll_deg),
        "zoom": float(np.sqrt(polygon_area(warped) / polygon_area(corners))),
    }


def measure_shift(
    live: NDArray[np.uint8], reference: NDArray[np.uint8], roi: list[int] | None = None
) -> MountShift | None:
    """Measure how far a camera has moved since its reference frame was taken.

    Two fits are attempted and the better-supported one wins. The whole-frame fit is the accurate
    one while the scene is unchanged. The robot-only fit is the one that still works after the rig
    moves to a different table, because the robot is the only rigid thing that comes with it. The
    inlier count is only a proxy for which is right: in a new room that resembles the old one, a
    whole-frame fit on repeated background can out-vote a correct robot fit. The reported `via` says
    which one answered, so an unexpected `"frame"` after a move is worth a second look.

    Args:
        live (`np.ndarray`):
            Current frame, BGR `uint8`.
        reference (`np.ndarray`):
            Saved frame, BGR `uint8`.
        roi (`list[int]`, *optional*):
            Box around the robot in the reference frame, as `[x, y, width, height]`.

    Returns:
        `MountShift | None`: The measurement, or `None` when neither fit found enough support. A
        `None` must be reported as "unknown", never treated as "no drift".
    """
    height, width = reference.shape[:2]
    regions: list[tuple[str, NDArray[np.uint8] | None, NDArray[np.uint8] | None]] = [("frame", None, None)]
    if roi:
        regions.append(
            ("robot", _box_mask(live.shape, roi, ROI_SEARCH_SCALE), _box_mask(reference.shape, roi, 1.0))
        )

    best = None
    for via, live_mask, reference_mask in regions:
        homography, inliers = _fit_homography(live, reference, live_mask, reference_mask)
        if homography is None or inliers < MIN_INLIERS or (best is not None and inliers <= best.inliers):
            continue
        best = MountShift(
            **decompose_homography(homography, width, height),
            inliers=inliers,
            via=via,
            homography=homography,
        )
    return best


def describe_correction(shift: MountShift | None, tol: MountTolerance) -> str:
    """Say which way to physically move the camera.

    Every direction here is two negations deep: the homography reports what the image content must
    do, and the camera has to move the opposite way. A flipped sign walks the mount further off with
    no other symptom, which is why `tests/scripts/test_lerobot_check_cameras.py` pins each one
    against a simulated camera move.

    Args:
        shift (`MountShift | None`):
            A measurement, or `None` when the fit failed.
        tol (`MountTolerance`):
            The limits deciding whether anything needs saying.

    Returns:
        `str`: A human instruction, `"in tolerance"`, or a note that nothing could be measured.
    """
    if shift is None:
        return "no fit - too few matching features (arm at the reference pose? lens covered or dark?)"
    if shift.within(tol):
        return "in tolerance"

    parts = []
    if shift.shift_px > tol.shift_px:
        # Gate on the total, then name only the axes that carry a visible part of it.
        if abs(shift.dx) > 1.0:
            parts.append(f"pan {'LEFT' if shift.dx > 0 else 'RIGHT'} {abs(shift.dx):.0f}px")
        if abs(shift.dy) > 1.0:
            parts.append(f"tilt {'UP' if shift.dy > 0 else 'DOWN'} {abs(shift.dy):.0f}px")
    if abs(shift.roll_deg) > tol.roll_deg:
        parts.append(f"roll {'CCW' if shift.roll_deg > 0 else 'CW'} {abs(shift.roll_deg):.1f}deg")
    if abs(shift.zoom - 1.0) > tol.zoom:
        parts.append(f"move {'CLOSER' if shift.zoom > 1 else 'BACK'} {abs(shift.zoom - 1) * 100:.0f}%")
    return "  ".join(parts) or "in tolerance"


def _reference_dir(robot: Robot) -> Path:
    """Where this robot's camera reference lives, beside its motor calibration."""
    if not robot.id:
        raise ValueError(
            "Pass --robot.id=<name>. A camera reference belongs to one physical rig, and without an "
            "id every rig of this type would share, and overwrite, the same one."
        )
    return HF_LEROBOT_CALIBRATION / CAMERAS / robot.name / robot.id


def _refuse_to_overwrite(path: Path, force: bool) -> None:
    """Stop a save silently replacing a reference. Also called before ``robot.connect()``, so the
    refusal lands before torque comes on."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Pass --force=true to replace it.")


def _support(shift: MountShift | None) -> str:
    """How the reading was arrived at, so a surprising `via` shows up while you are still nudging."""
    return f"[{shift.inliers} pts, {shift.via}]" if shift else "[no fit]"


def _rgb_camera_keys(robot: Robot) -> list[str]:
    """Observation keys whose feature is an `(h, w, 3)` shape, so colour frames but not depth."""
    return [
        key
        for key, ft in robot.observation_features.items()
        if isinstance(ft, tuple) and len(ft) == 3 and ft[2] == 3
    ]


def _grab(robot: Robot, camera_keys: list[str]) -> tuple[dict[str, NDArray[np.uint8]], dict[str, float]]:
    """Read one observation as BGR frames plus a pose keyed for `robot.send_action`.

    Frames are converted to BGR here so everything downstream, including the images on disk, uses
    one convention. Only `.pos` actions count as pose: a mobile base exposes wheel velocities, and a
    velocity says nothing about where the cameras are pointing.
    """
    obs = robot.get_observation()
    frames = {
        key: cv2.cvtColor(np.asarray(obs[key], dtype=np.uint8), cv2.COLOR_RGB2BGR) for key in camera_keys
    }
    pose = {key: float(obs[key]) for key in robot.action_features if key.endswith(".pos") and key in obs}
    return frames, pose


def _pose_error(pose: dict[str, float], reference_pose: dict[str, float]) -> dict[str, float]:
    """Signed current-minus-reference error, for each joint present in both."""
    return {key: pose[key] - value for key, value in reference_pose.items() if key in pose}


def _drive_to_pose(robot: Robot, target: dict[str, float], seconds: float = 3.0, hz: float = 30.0) -> None:
    """Interpolate the arm to ``target``, slowly, because this runs unattended.

    Any velocity axis the robot exposes is commanded to zero throughout, so a mobile base holds
    still while the arm moves and `send_action` still gets the full set of keys it expects.
    """
    start = {key: float(value) for key, value in robot.get_observation().items() if key in target}
    hold_still = {key: 0.0 for key in robot.action_features if key.endswith(".vel")}
    steps = max(1, int(seconds * hz))
    for step in range(1, steps + 1):
        alpha = step / steps
        moving = {k: start.get(k, v) + alpha * (v - start.get(k, v)) for k, v in target.items()}
        robot.send_action({**hold_still, **moving})
        time.sleep(1.0 / hz)
    time.sleep(0.5)


def _overlay(
    live: NDArray[np.uint8], reference: NDArray[np.uint8], shift: MountShift | None
) -> NDArray[np.uint8]:
    """The live view with the reference ghosted over it, its outline, and a correction arrow."""
    height, width = live.shape[:2]
    live_gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    # Reference in magenta, live in green: aligned content turns grey, drift shows as coloured fringes.
    canvas = cv2.merge([ref_gray, live_gray, ref_gray])

    if shift is not None:
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
        outline = cv2.perspectiveTransform(corners, np.linalg.inv(shift.homography))
        cv2.polylines(canvas, [np.int32(outline)], True, (0, 255, 255), 2)
        length = float(np.hypot(shift.dx, shift.dy))
        if length > 1.0:
            scale = min(60.0, length * 3.0) / length
            centre = (width // 2, height // 2)
            tip = (int(centre[0] - shift.dx * scale), int(centre[1] - shift.dy * scale))
            cv2.arrowedLine(canvas, centre, tip, (0, 255, 255), 3, tipLength=0.3)
    return canvas


def save_reference(robot: Robot, cfg: CheckCamerasConfig) -> None:
    """Capture and store the reference frames, arm pose and robot boxes.

    Args:
        robot (`Robot`):
            A connected robot.
        cfg (`CheckCamerasConfig`):
            Parsed configuration.

    Raises:
        FileExistsError: If a reference already exists and `cfg.force` is not set.
    """
    out_dir = _reference_dir(robot)
    ref_path = out_dir / REFERENCE_FILENAME
    _refuse_to_overwrite(ref_path, cfg.force)

    camera_keys = _rgb_camera_keys(robot)
    print(
        "\nPut the arm in a pose you can return to later. It should be clearly visible in every view:\n"
        "the robot is the anchor that survives a move to another table, and something mid-reach works\n"
        "better than the rest pose. Torque is on once the robot is connected, so drive it there with\n"
        "the leader arm rather than pushing against the servos."
    )
    input("Press Enter once the arm is in place... ")
    _, pose = _grab(robot, [])

    # The pose and the frames are captured separately: a hand still on the arm would be baked into
    # the reference as features that are never there again, and every later check would fail on it.
    input("Now move your hands, and anything else temporary, out of every view. Press Enter... ")
    frames, _ = _grab(robot, camera_keys)
    rois: dict[str, list[int] | None] = dict.fromkeys(camera_keys)
    if cfg.select_roi:
        if _gui_available():
            for key in camera_keys:
                print(f"Drag a box around the robot in '{key}', then press Enter. Press c to skip it.")
                box = cv2.selectROI(f"{key}: box the robot", frames[key], showCrosshair=False)
                cv2.destroyAllWindows()
                rois[key] = [int(v) for v in box] if box[2] > 0 and box[3] > 0 else None
        else:
            print(
                "No GUI available, so no robot box was drawn. Checks will still work on this table but "
                "will not survive a move to a different one. Install opencv-python to draw the box."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    reference = {
        "robot_type": robot.name,
        "robot_id": robot.id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "joint_pose": pose,
        "cameras": {},
    }
    for key in camera_keys:
        cv2.imwrite(str(out_dir / f"{key}.png"), frames[key])
        reference["cameras"][key] = {
            "file": f"{key}.png",
            "height": int(frames[key].shape[0]),
            "width": int(frames[key].shape[1]),
            "roi": rois[key],
        }
    ref_path.write_text(json.dumps(reference, indent=2))

    print(f"\nSaved reference for {len(camera_keys)} camera(s) to {out_dir}")
    print("arm pose:", {key: round(value, 1) for key, value in pose.items()})
    print("Check the mounts against it with: lerobot-check-cameras --mode=check ...")


def check_reference(robot: Robot, cfg: CheckCamerasConfig) -> bool:
    """Compare the live cameras against the saved reference.

    Args:
        robot (`Robot`):
            A connected robot.
        cfg (`CheckCamerasConfig`):
            Parsed configuration.

    Returns:
        `bool`: `True` when every camera is inside tolerance.

    Raises:
        FileNotFoundError: If no reference has been saved, or one of its frames is gone.
        RuntimeError: If no camera can be scored, a camera's resolution has changed, the arm is not
            at the reference pose, or `--live` was asked for without a GUI.
    """
    ref_dir = _reference_dir(robot)
    ref_path = ref_dir / REFERENCE_FILENAME
    if not ref_path.exists():
        raise FileNotFoundError(f"No camera reference at {ref_path}. Run with --mode=save first.")
    reference = json.loads(ref_path.read_text())

    camera_keys = [key for key in _rgb_camera_keys(robot) if key in reference["cameras"]]
    if not camera_keys:
        # Silence here would be worse than a crash: an empty check passes, and a recording script
        # gated on the exit code would be waved through with nothing actually verified.
        raise RuntimeError(
            f"No camera in the reference {sorted(reference['cameras'])} matches this robot's "
            f"{sorted(_rgb_camera_keys(robot))}, so nothing can be checked. Are the camera names or "
            "the --robot.id right?"
        )
    missing = sorted(set(reference["cameras"]) - set(camera_keys))
    if missing:
        print(f"WARNING: reference has camera(s) {missing} that this robot does not expose; skipping them.")
    extra = sorted(set(_rgb_camera_keys(robot)) - set(reference["cameras"]))
    if extra:
        print(f"WARNING: camera(s) {extra} are not in the reference, so their mounts go unchecked.")

    ref_frames = {}
    for key in camera_keys:
        spec = reference["cameras"][key]
        frame = cv2.imread(str(ref_dir / spec["file"]))
        if frame is None:
            raise FileNotFoundError(f"Reference frame {ref_dir / spec['file']} is missing or unreadable.")
        height, width, _ = robot.observation_features[key]
        if (height, width) != (spec["height"], spec["width"]):
            raise RuntimeError(
                f"Camera '{key}' is configured at {width}x{height} but its reference is "
                f"{spec['width']}x{spec['height']}. Drift measured in pixels does not carry across a "
                "change of resolution: set the camera back, or save a new reference."
            )
        ref_frames[key] = frame

    if cfg.drive_to_pose:
        print("Moving the arm to the reference pose. Keep the workspace clear.")
        _drive_to_pose(robot, reference["joint_pose"])

    _, pose = _grab(robot, [])
    errors = _pose_error(pose, reference["joint_pose"])
    worst = max(errors.items(), key=lambda kv: abs(kv[1]), default=(None, 0.0))
    if abs(worst[1]) > cfg.tolerance.pose_deg:
        raise RuntimeError(
            f"The arm is not at the pose the reference was captured at ('{worst[0]}' is off by "
            f"{worst[1]:+.1f}). Camera drift cannot be told apart from arm motion until it is.\n"
            f"  reference pose: { ({k: round(v, 1) for k, v in reference['joint_pose'].items()}) }\n"
            f"  current pose:   { ({k: round(v, 1) for k, v in pose.items()}) }\n"
            "Move the arm there, or pass --drive_to_pose=true to have it driven there."
        )

    if cfg.live and not _gui_available():
        raise RuntimeError("--live needs a GUI, but OpenCV cannot open a window. Install opencv-python.")

    print(f"\nreference from {reference['created']} at {ref_dir}")
    if cfg.live:
        print("Nudge each camera until every reading says 'in tolerance'. Press q or Esc to stop.")
    while True:
        frames, _ = _grab(robot, camera_keys)
        shifts = {
            key: measure_shift(frames[key], ref_frames[key], reference["cameras"][key]["roi"])
            for key in camera_keys
        }
        if not cfg.live:
            break

        # One line, overwritten in place. Printing the block every iteration scrolls a dozen lines a
        # second past the person whose hands are on the camera. The support carries here too: a
        # `via` of "frame" after a move to a new room is the sign the wrong fit is answering, and it
        # is no use only appearing once you have stopped nudging.
        summary = " | ".join(
            f"{key}: {describe_correction(s, cfg.tolerance)} {_support(s)}" for key, s in shifts.items()
        )
        print(f"\r  {summary[:200]:<200}", end="", flush=True)
        for key in camera_keys:
            cv2.imshow(
                f"{key}: reference (magenta) vs live (green)",
                _overlay(frames[key], ref_frames[key], shifts[key]),
            )
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            print()
            cv2.destroyAllWindows()
            break

    for key, shift in shifts.items():
        print(f"  {key:<16} {describe_correction(shift, cfg.tolerance):<52} {_support(shift)}")
    return all(shift is not None and shift.within(cfg.tolerance) for shift in shifts.values())


@draccus.wrap()
def check_cameras(cfg: CheckCamerasConfig) -> None:
    """Save or check the camera mount reference for a robot.

    Args:
        cfg (`CheckCamerasConfig`):
            Parsed configuration.

    Raises:
        SystemExit: With status 1 when a check finds a camera out of tolerance.
        ValueError: If the robot exposes no colour cameras.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    if not _rgb_camera_keys(robot):
        raise ValueError(f"{robot} has no colour cameras configured, so there are no mounts to check.")
    # Whatever can be settled before connecting is settled before connecting. Connecting brings
    # torque on, and an uncalibrated robot prompts, so a missing --robot.id or a save that would
    # overwrite an existing reference should not surface with the arm already live.
    reference_path = _reference_dir(robot) / REFERENCE_FILENAME
    if cfg.mode == "save":
        _refuse_to_overwrite(reference_path, cfg.force)

    robot.connect()
    try:
        if cfg.mode == "save":
            save_reference(robot, cfg)
        elif not check_reference(robot, cfg):
            raise SystemExit(1)
    finally:
        robot.disconnect()


def main():
    """Entry point for the ``lerobot-check-cameras`` command."""
    register_third_party_plugins()
    check_cameras()


if __name__ == "__main__":
    sys.exit(main())
