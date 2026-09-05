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

"""Every instruction ``lerobot-check-cameras`` prints is two negations deep: the homography reports
what the image content must do, and the camera has to move the opposite way. A flipped sign walks
the mount further off with no other symptom, because the fit still converges and the numbers still
look reasonable. So these tests apply a known camera move and assert the tool names its inverse.

The scene is synthetic, so there is no artifact dependency. SIFT-over-ORB and the default
tolerances came from real recordings; that reasoning lives in the script's docstrings.
"""

import json

import cv2
import numpy as np
import pytest

import lerobot.scripts.lerobot_check_cameras as check_cameras

ROI = [200, 240, 240, 200]


@pytest.fixture(scope="module")
def reference_frame():
    """A textured frame with a distinct blob inside the ROI, so the masked fit has its own features."""
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), 40, np.uint8)
    for _ in range(400):
        x, y = rng.integers(0, 620), rng.integers(0, 460)
        cv2.rectangle(
            img,
            (int(x), int(y)),
            (int(x) + int(rng.integers(4, 18)), int(y) + int(rng.integers(4, 18))),
            [int(v) for v in rng.integers(60, 255, 3)],
            -1,
        )
    x, y, w, h = ROI
    for _ in range(200):
        cx, cy = rng.integers(x, x + w - 12), rng.integers(y, y + h - 12)
        cv2.circle(
            img, (int(cx), int(cy)), int(rng.integers(2, 7)), [int(v) for v in rng.integers(80, 255, 3)], -1
        )
    return cv2.GaussianBlur(img, (3, 3), 0)


def move_camera(img, pan=0.0, tilt=0.0, roll=0.0, closer=0.0):
    """Render what the camera would see after a known move.

    pan > 0     camera swings right   -> content slides left
    tilt > 0    camera aims up        -> content slides down
    roll > 0    camera body turns CW  -> content turns CCW
    closer > 0  camera approaches     -> content grows
    """
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0 + closer)
    matrix[0, 2] += -pan
    matrix[1, 2] += +tilt
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def replace_background(img, roi):
    """Everything outside the robot box swapped for another scene, as after a move to a new table."""
    rng = np.random.default_rng(1)
    out = np.full_like(img, 90)
    for _ in range(400):
        x, y = rng.integers(0, 620), rng.integers(0, 460)
        cv2.rectangle(
            out,
            (int(x), int(y)),
            (int(x) + int(rng.integers(4, 18)), int(y) + int(rng.integers(4, 18))),
            [int(v) for v in rng.integers(0, 200, 3)],
            -1,
        )
    x, y, w, h = roi
    out[y : y + h, x : x + w] = img[y : y + h, x : x + w]
    return out


# (camera move, the correction the tool must name)
MOVES = [
    ({"pan": 20}, "pan LEFT"),
    ({"pan": -20}, "pan RIGHT"),
    ({"tilt": 15}, "tilt DOWN"),
    ({"tilt": -15}, "tilt UP"),
    ({"roll": 2.0}, "roll CCW"),
    ({"roll": -2.0}, "roll CW"),
    ({"closer": 0.04}, "move BACK"),
    ({"closer": -0.04}, "move CLOSER"),
    ({}, "in tolerance"),
]
MOVE_IDS = [expected.replace(" ", "-") for _, expected in MOVES]


@pytest.mark.parametrize("move,expected", MOVES, ids=MOVE_IDS)
def test_correction_undoes_the_camera_move(reference_frame, move, expected):
    shift = check_cameras.measure_shift(move_camera(reference_frame, **move), reference_frame, ROI)
    assert shift is not None, "homography did not converge on a synthetic warp"
    assert expected in check_cameras.describe_correction(shift, check_cameras.MountTolerance())


@pytest.mark.parametrize("move,expected", MOVES, ids=MOVE_IDS)
def test_correction_survives_a_new_table(reference_frame, move, expected):
    """Everything but the robot is replaced, as after a move to another room. The answer must not
    change. Which of the two fits gets there is not asserted: when the robot fills a good part of
    the frame the whole-frame fit finds it unaided, and that is a fine outcome."""
    live = replace_background(move_camera(reference_frame, **move), ROI)
    shift = check_cameras.measure_shift(live, reference_frame, ROI)
    assert shift is not None, "no fit converged after the scene changed"
    assert expected in check_cameras.describe_correction(shift, check_cameras.MountTolerance())


def test_robot_box_fit_stands_on_its_own(reference_frame):
    """The box fit is the fallback for when the whole-frame fit dies, so exercise it in isolation
    rather than relying on it winning the inlier race."""
    live = replace_background(move_camera(reference_frame, pan=20), ROI)
    homography, inliers = check_cameras._fit_homography(
        live,
        reference_frame,
        check_cameras._box_mask(live.shape, ROI, check_cameras.ROI_SEARCH_SCALE),
        check_cameras._box_mask(reference_frame.shape, ROI, 1.0),
    )
    assert homography is not None
    assert inliers >= check_cameras.MIN_INLIERS
    decomposed = check_cameras.decompose_homography(homography, 640, 480)
    # The camera panned right by 20 px, so the content has to come back the other way.
    assert decomposed["dx"] == pytest.approx(20.0, abs=2.0)


def test_box_mask_grows_about_its_centre():
    """The live frame is searched over a wider box than the reference, so a roughly re-mounted
    camera still has the robot inside the search area."""
    box = [250, 180, 140, 120]  # small enough that doubling it stays inside the frame
    tight = check_cameras._box_mask((480, 640), box, 1.0)
    grown = check_cameras._box_mask((480, 640), box, 2.0)

    assert np.all(grown[tight > 0] > 0), "the grown box must contain the tight one"
    assert grown.sum() == pytest.approx(4 * tight.sum(), rel=0.02)
    for mask in (tight, grown):
        ys, xs = np.nonzero(mask)
        assert xs.mean() == pytest.approx(320.0, abs=1.0)
        assert ys.mean() == pytest.approx(240.0, abs=1.0)


def test_box_mask_is_clipped_to_the_frame():
    """Growing a box that already touches an edge cannot search outside the image."""
    mask = check_cameras._box_mask((480, 640), [200, 240, 240, 200], 2.0)
    assert mask.shape == (480, 640)
    assert mask[:139].sum() == 0  # nothing above the grown top edge
    assert mask[479, 320] > 0  # grown down to the bottom row and stopped there


def test_no_fit_is_reported_not_guessed(reference_frame):
    """A featureless frame must return None, never a confident zero."""
    shift = check_cameras.measure_shift(np.zeros_like(reference_frame), reference_frame, ROI)
    assert shift is None
    assert "no fit" in check_cameras.describe_correction(shift, check_cameras.MountTolerance())


def test_unchanged_view_is_within_tolerance(reference_frame):
    shift = check_cameras.measure_shift(reference_frame, reference_frame, ROI)
    assert shift.within(check_cameras.MountTolerance())
    assert shift.shift_px == pytest.approx(0.0, abs=0.5)


def test_shift_magnitude_matches_the_move(reference_frame):
    shift = check_cameras.measure_shift(move_camera(reference_frame, pan=40), reference_frame, ROI)
    assert shift.shift_px == pytest.approx(40.0, abs=2.0)


def test_tolerance_decides_the_verdict(reference_frame):
    """The same measurement must pass or fail purely on the configured limits."""
    shift = check_cameras.measure_shift(move_camera(reference_frame, pan=10), reference_frame, ROI)
    assert not shift.within(check_cameras.MountTolerance(shift_px=4.0))
    assert shift.within(check_cameras.MountTolerance(shift_px=20.0))


class FakeRobot:
    """The slice of the Robot interface this script touches, with scriptable frames and pose.

    ``velocities`` stands in for a mobile base: axes that appear in `action_features` but are not a
    pose, and that `send_action` refuses to be called without.
    """

    name = "fake_follower"

    def __init__(self, frame, pose, velocities=()):
        self.id = "test_arm"
        self.camera_keys = ["front"]
        self.frame = frame
        self.pose = dict(pose)
        self.velocities = dict.fromkeys(velocities, 0.0)
        self.sent = []

    @property
    def observation_features(self):
        return {**self.action_features, **dict.fromkeys(self.camera_keys, (*self.frame.shape[:2], 3))}

    @property
    def action_features(self):
        return {**dict.fromkeys(self.pose, float), **dict.fromkeys(self.velocities, float)}

    def get_observation(self):
        # The robot hands out RGB; the script is responsible for converting to OpenCV's BGR.
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return {**self.pose, **self.velocities, **dict.fromkeys(self.camera_keys, rgb)}

    def send_action(self, action):
        missing = sorted(set(self.velocities) - set(action))
        if missing:
            raise KeyError(f"send_action indexes {missing} directly, as LeKiwi's does")
        self.sent.append(action)
        self.pose.update({key: value for key, value in action.items() if key in self.pose})
        return action


@pytest.fixture
def saved(tmp_path, monkeypatch, reference_frame):
    """A robot with a reference already on disk, and the config used to write it."""
    monkeypatch.setattr(check_cameras, "HF_LEROBOT_CALIBRATION", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *_: "")
    robot = FakeRobot(reference_frame, {"shoulder_pan.pos": 12.0, "elbow_flex.pos": -30.0})
    cfg = check_cameras.CheckCamerasConfig(robot=None, select_roi=False)
    check_cameras.save_reference(robot, cfg)
    return robot, cfg


def test_save_then_check_round_trips(saved, tmp_path):
    """An untouched rig must pass a check against its own freshly written reference."""
    robot, cfg = saved
    ref_dir = tmp_path / "cameras" / "fake_follower" / "test_arm"
    stored = json.loads((ref_dir / check_cameras.REFERENCE_FILENAME).read_text())

    assert (ref_dir / "front.png").exists()
    assert stored["joint_pose"] == {"shoulder_pan.pos": 12.0, "elbow_flex.pos": -30.0}
    assert stored["cameras"]["front"] == {"file": "front.png", "height": 480, "width": 640, "roi": None}
    assert check_cameras.check_reference(robot, cfg) is True


def test_check_fails_when_the_camera_has_moved(saved):
    robot, cfg = saved
    robot.frame = move_camera(robot.frame, pan=25)
    assert check_cameras.check_reference(robot, cfg) is False


def test_check_refuses_when_the_arm_is_not_at_the_reference_pose(saved):
    """The gate that stops arm motion being reported as camera drift."""
    robot, cfg = saved
    robot.pose["elbow_flex.pos"] += 20.0
    with pytest.raises(RuntimeError, match="not at the pose"):
        check_cameras.check_reference(robot, cfg)


def test_drive_to_pose_returns_the_arm_and_then_the_check_runs(saved):
    robot, cfg = saved
    robot.pose["elbow_flex.pos"] += 20.0
    cfg.drive_to_pose = True

    assert check_cameras.check_reference(robot, cfg) is True
    assert robot.sent, "the arm should have been commanded"
    assert robot.pose["elbow_flex.pos"] == pytest.approx(-30.0, abs=0.1)


def test_save_refuses_to_overwrite_without_force(saved):
    robot, cfg = saved
    with pytest.raises(FileExistsError, match="--force"):
        check_cameras.save_reference(robot, cfg)
    cfg.force = True
    check_cameras.save_reference(robot, cfg)


def test_save_captures_the_pose_before_the_frames(tmp_path, monkeypatch, reference_frame):
    """Two prompts, not one. The operator's hands have to leave the shot between reading the pose
    and grabbing the frames, or they are baked into the reference as features that are never there
    again and every later check fails against them."""
    monkeypatch.setattr(check_cameras, "HF_LEROBOT_CALIBRATION", tmp_path)
    robot = FakeRobot(reference_frame, {"shoulder_pan.pos": 5.0})
    hands_out = move_camera(reference_frame, pan=30)
    prompts = []

    def answer(_):
        prompts.append(len(prompts))
        if len(prompts) == 2:
            robot.pose["shoulder_pan.pos"] = 99.0
            robot.frame = hands_out
        return ""

    monkeypatch.setattr("builtins.input", answer)
    check_cameras.save_reference(robot, check_cameras.CheckCamerasConfig(robot=None, select_roi=False))

    ref_dir = tmp_path / "cameras" / "fake_follower" / "test_arm"
    stored = json.loads((ref_dir / check_cameras.REFERENCE_FILENAME).read_text())
    assert len(prompts) == 2
    assert stored["joint_pose"] == {"shoulder_pan.pos": 5.0}, "pose must be read before the second prompt"
    assert np.array_equal(cv2.imread(str(ref_dir / "front.png")), hands_out), "frames must come after it"


def test_check_without_a_reference_says_so(tmp_path, monkeypatch, reference_frame):
    monkeypatch.setattr(check_cameras, "HF_LEROBOT_CALIBRATION", tmp_path)
    robot = FakeRobot(reference_frame, {"shoulder_pan.pos": 0.0})
    with pytest.raises(FileNotFoundError, match="--mode=save"):
        check_cameras.check_reference(robot, check_cameras.CheckCamerasConfig(robot=None))


def test_check_fails_loudly_when_no_camera_can_be_scored(saved):
    """A check that scores nothing must not pass. `all([])` is True, so a recording script gated on
    the exit code would be waved through with no camera verified at all."""
    robot, cfg = saved
    robot.camera_keys = ["renamed_after_the_reference_was_saved"]
    with pytest.raises(RuntimeError, match="nothing can be checked"):
        check_cameras.check_reference(robot, cfg)


def test_check_says_which_reference_frame_is_gone(saved, tmp_path):
    """cv2.imread returns None for a missing file, which would otherwise surface as a cv2.error
    from inside the matcher."""
    robot, cfg = saved
    (tmp_path / "cameras" / "fake_follower" / "test_arm" / "front.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing or unreadable"):
        check_cameras.check_reference(robot, cfg)


def test_check_warns_about_a_camera_the_reference_never_saw(saved, capsys):
    """Adding a camera after the reference was written is silent otherwise: the check passes, having
    said nothing about the mount you just bolted on."""
    robot, cfg = saved
    robot.camera_keys.append("added_later")  # the reference still only knows about "front"

    assert check_cameras.check_reference(robot, cfg) is True
    assert "['added_later'] are not in the reference" in capsys.readouterr().out


def test_saving_over_a_reference_is_refused_before_the_robot_is_touched(saved, tmp_path):
    """`check_cameras` calls this ahead of `robot.connect()`, so the refusal does not arrive with
    the arm already under torque."""
    path = tmp_path / "cameras" / "fake_follower" / "test_arm" / check_cameras.REFERENCE_FILENAME
    with pytest.raises(FileExistsError, match="--force"):
        check_cameras._refuse_to_overwrite(path, force=False)

    check_cameras._refuse_to_overwrite(path, force=True)
    check_cameras._refuse_to_overwrite(path.with_name("not_there.json"), force=False)


def test_live_keeps_the_support_on_the_status_line(saved, monkeypatch, capsys):
    """`via` is the flag for the wrong fit winning, and it is no use appearing only after you stop
    nudging. One iteration, then quit."""
    robot, cfg = saved
    cfg.live = True
    monkeypatch.setattr(check_cameras, "_gui_available", lambda: True)
    monkeypatch.setattr(cv2, "imshow", lambda *_: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *_: ord("q"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    check_cameras.check_reference(robot, cfg)

    live_line = capsys.readouterr().out.split("\r")[1]
    assert "front: in tolerance" in live_line
    assert "pts, frame]" in live_line, "the inlier count and the winning region belong on this line"


def test_check_refuses_a_changed_resolution(saved):
    """Drift is reported in pixels, so it is not comparable across resolutions."""
    robot, cfg = saved
    robot.frame = cv2.resize(robot.frame, (320, 240))
    with pytest.raises(RuntimeError, match="change of resolution"):
        check_cameras.check_reference(robot, cfg)


def test_reference_dir_needs_a_robot_id(reference_frame):
    """--robot.id is optional upstream, and a None here would land every rig of one type in the
    same directory."""
    robot = FakeRobot(reference_frame, {"shoulder_pan.pos": 0.0})
    robot.id = None
    with pytest.raises(ValueError, match="--robot.id"):
        check_cameras._reference_dir(robot)


def test_velocity_axes_are_not_part_of_the_pose(reference_frame):
    """A mobile base exposes wheel velocities as actions. They say nothing about where the cameras
    point, and gating on them would block a check whenever the base happened to be rolling."""
    robot = FakeRobot(reference_frame, {"shoulder_pan.pos": 3.0}, velocities=["x.vel"])
    robot.velocities["x.vel"] = 0.4
    _, pose = check_cameras._grab(robot, [])
    assert pose == {"shoulder_pan.pos": 3.0}


def test_drive_to_pose_holds_a_mobile_base_still(reference_frame):
    robot = FakeRobot(reference_frame, {"shoulder_pan.pos": 0.0}, velocities=["x.vel", "theta.vel"])
    check_cameras._drive_to_pose(robot, {"shoulder_pan.pos": 10.0}, seconds=0.05, hz=20.0)

    assert robot.sent, "the arm should have been commanded"
    assert all(sent["x.vel"] == 0.0 and sent["theta.vel"] == 0.0 for sent in robot.sent)
    assert robot.pose["shoulder_pan.pos"] == pytest.approx(10.0, abs=0.1)


def test_rgb_camera_keys_skips_depth_and_joints():
    robot = type(
        "FakeRobot",
        (),
        {
            "observation_features": {
                "shoulder_pan.pos": float,
                "front": (480, 640, 3),
                "front_depth": (480, 640, 1),
            }
        },
    )()
    assert check_cameras._rgb_camera_keys(robot) == ["front"]


def test_pose_error_is_signed_and_per_joint():
    errors = check_cameras._pose_error(
        {"a.pos": 10.0, "b.pos": -2.0, "c.pos": 0.0}, {"a.pos": 8.0, "b.pos": 1.0}
    )
    assert errors == {"a.pos": 2.0, "b.pos": -3.0}


def test_mode_must_be_save_or_check():
    with pytest.raises(ValueError, match="mode must be"):
        check_cameras.CheckCamerasConfig(robot=None, mode="inspect")


def test_main_registers_plugins_before_parsing(monkeypatch):
    calls = []
    monkeypatch.setattr(check_cameras, "register_third_party_plugins", lambda: calls.append("register"))
    monkeypatch.setattr(check_cameras, "check_cameras", lambda: calls.append("check"))

    check_cameras.main()

    assert calls == ["register", "check"]
