#!/usr/bin/env python

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

"""Every knob of the generator. Ranges are ``(low, high)`` sampled uniformly once per episode.

Lengths are in meters, angles in degrees, durations in seconds.
"""

from dataclasses import dataclass, field

Range = tuple[float, float]


@dataclass
class ObjectsConfig:
    # Graspable objects on the table, the task's own included.
    count: tuple[int, int] = (2, 4)
    # Probability that an object is a Google Scanned Objects mesh rather than a primitive.
    mesh_prob: float = 0.3
    # Width across the jaws and height of primitives.
    width: Range = (0.022, 0.045)
    height: Range = (0.02, 0.06)
    mass: Range = (0.01, 0.08)
    friction: Range = (0.8, 1.5)
    # Subtle wood-grain or speckle textures; bold patterns don't exist on real objects.
    texture_prob: float = 0.15
    # Jitter of the named palette color, so "red" stays red.
    color_jitter: float = 0.06


@dataclass
class ContainerConfig:
    inner_size: Range = (0.08, 0.13)
    wall_height: Range = (0.012, 0.045)
    wall_thickness: Range = (0.003, 0.006)
    texture_prob: float = 0.1


@dataclass
class LayoutConfig:
    # Placement annulus around the robot base, in front of it. The lower bound keeps the scene from
    # crowding the base, where the SO-101 can only reach awkwardly.
    radius: Range = (0.17, 0.30)
    angle: Range = (-60.0, 60.0)
    min_gap: float = 0.015


@dataclass
class BackgroundConfig:
    # Probability that the room is a photographed indoor panorama (workshop, office, living room...),
    # rather than plain walls and floor in muted colors. Its brightness varies with the exposure.
    hdri_prob: float = 0.9
    hdri_exposure: Range = (0.7, 1.15)
    # Without a panorama, the probability of walls around the table.
    walls_prob: float = 0.7
    # Probability of a photographed table surface (wood, laminate, concrete...) rather than a plain one,
    # and its brightness as a multiplier.
    surface_prob: float = 0.75
    surface_brightness: Range = (0.75, 1.05)
    table_half_size: Range = (0.35, 0.7)


@dataclass
class LightingConfig:
    # Indoor lighting: one or two ceiling lights, and ambient light for what the room reflects. Strong
    # ambient and headlight fill keep shadows as light as indoors: the classic renderer has no soft shadows.
    count: tuple[int, int] = (1, 2)
    diffuse: Range = (0.3, 0.65)
    ambient: Range = (0.2, 0.4)
    headlight: Range = (0.05, 0.2)
    shadow_prob: float = 0.85
    # White light, from warm bulbs to daylight, in kelvin, and the error of the camera's white balance.
    temperature: Range = (2700.0, 6500.0)
    white_balance_error: float = 400.0
    # Lights sit on a hemisphere around the workspace: distance and elevation.
    distance: Range = (0.8, 2.0)
    elevation: Range = (45.0, 85.0)


@dataclass
class FrontCameraConfig:
    # Spherical placement around a look-at point in the workspace. Azimuth 0 faces the robot.
    # Where people mount a camera for an SO-101: close to the table, above it or in front, at most a
    # little to the left or right.
    distance: Range = (0.3, 0.55)
    azimuth: Range = (-35.0, 35.0)
    elevation: Range = (35.0, 80.0)
    lookat: tuple[float, float, float] = (0.2, 0.0, 0.02)
    lookat_jitter: float = 0.05
    roll: float = 3.0
    fovy: Range = (50.0, 70.0)


@dataclass
class WristCameraConfig:
    # Around Menagerie's wrist_cam mount: a mounting tolerance, not a new placement.
    pos_jitter: float = 0.003
    rot_jitter: float = 2.0
    fovy: Range = (57.0, 65.0)


@dataclass
class DynamicsConfig:
    kp_scale: Range = (0.9, 1.1)
    damping_scale: Range = (0.8, 1.2)
    frictionloss_scale: Range = (0.8, 1.2)


@dataclass
class StartPoseConfig:
    # Weights of the start modes: the rest pose, the parked pose real recordings start from (the
    # episode then lifts off to rest), a random collision-free pose, a pose mid-task.
    rest: float = 0.35
    parked: float = 0.35
    random_free: float = 0.15
    mid_task: float = 0.15
    rest_jitter: float = 5.0
    # Per joint, within the spread of real first frames; more fold puts the jaws into the table.
    parked_jitter: tuple[float, ...] = (5.0, 0.0, 4.0, 6.0, 5.0, 0.0)


@dataclass
class MotionConfig:
    # Episode-wide speed factor, then per segment.
    speed: Range = (0.7, 1.3)
    segment_speed: Range = (0.85, 1.15)
    # Average speeds; minimum-jerk peaks are about twice as fast.
    linear_speed: Range = (0.05, 0.12)
    joint_speed: Range = (30.0, 65.0)
    # Peak command speed per joint (deg/s, gripper in %/s): slower segments are stretched to stay under.
    # Careful real SO-101 teleoperation pans under about 60 deg/s and moves the other joints under 110.
    max_joint_speed: tuple[float, ...] = (70.0, 100.0, 100.0, 110.0, 110.0, 150.0)
    # Asymmetry of each move's speed profile: below 1 the hand speeds up quickly and settles slowly.
    time_skew: Range = (0.8, 1.2)
    # Transits bend through random control points, each joint off the straight path by up to this share
    # of its travel, plus up to via_jitter degrees: usually one arc, sometimes an S (s_curve_prob).
    via_spread: float = 0.25
    via_jitter: float = 6.0
    s_curve_prob: float = 0.3
    # Consecutive moves overlap by this share of the shorter one, so the arm flows from one into the
    # next instead of stopping; pausing and the gripper break the flow. A pause before descending onto
    # an object happens with pause_prob.
    blend: Range = (0.4, 0.7)
    pause_prob: float = 0.25
    approach_height: Range = (0.05, 0.11)
    lift_height: Range = (0.06, 0.13)
    # Approach tilt from vertical, outwards from the robot: mostly near vertical, sometimes steeply angled.
    # A 5-DoF arm keeps its approach in the plane of its pan, so tilts are always radial; horizontal
    # grasps would put the gripper's body into the table. Steep ones are only reachable far from the
    # base and, for boxes, from one side: the planner falls back to the nearest reachable tilt.
    tilt: Range = (0.0, 30.0)
    steep_tilt: Range = (30.0, 55.0)
    steep_tilt_prob: float = 0.5
    # Spread of the jaw axis around its preferred direction, for round objects.
    jaw_yaw_jitter: float = 25.0
    # Extra jaw opening around the object before closing.
    open_margin: Range = (0.015, 0.04)
    grasp_offset: float = 0.004
    dwell: Range = (0.05, 0.35)
    # Height of the via point that bends transit moves.
    via_height: Range = (0.0, 0.08)
    # Approaches and placements come in from up to this far off the approach axis, then curve onto it.
    approach_offset: float = 0.035
    return_rest_prob: float = 0.5


@dataclass
class NoiseConfig:
    # Commands: a slow Ornstein-Uhlenbeck drift per joint, and rare short jerks. Both are executed.
    drift_std: float = 0.4
    drift_tau: float = 0.4
    jerk_rate: float = 0.06
    jerk_amplitude: Range = (0.8, 2.0)
    jerk_frames: tuple[int, int] = (4, 9)
    # Gear play per arm joint, in degrees, sampled per episode: on reversing, the joint lags behind the
    # servo until the play is taken up. On the robot side, so it shows between action and state.
    backlash: Range = (1.0, 3.0)


@dataclass
class CalibrationConfig:
    # Spread of a calibration's joint range ends (the user's range sweep), per arm joint: each joint then
    # reads off by about 0.7x this. Real arms disagree most on shoulder_pan.
    range_end_std: tuple[float, ...] = (10.0, 4.0, 4.0, 4.0, 4.0)
    # Spread of the homing pose, which sets the zero of full-turn joints (wrist_roll).
    homing_std: float = 6.0
    # Gripper closed and open ends, around the simulated gripper limits.
    gripper_closed_std: float = 3.0
    gripper_open_std: float = 8.0
    # Sensor noise on state readings, in encoder ticks.
    reading_noise_ticks: float = 1.0


@dataclass
class AnnotationConfig:
    # VQA pairs per second, alternating cameras. VQA frames train only the language head, but ground
    # truth costs one segmentation render (about 3 ms) each.
    vqa_hz: float = 1.0
    question_weights: dict[str, float] = field(
        default_factory=lambda: {
            "bbox": 0.4,
            "keypoint": 0.2,
            "count": 0.1,
            "attribute": 0.15,
            "spatial": 0.15,
        }
    )
    task_aug: int = 10
    memory: bool = True
    min_visible_pixels: int = 40


@dataclass
class TaskMixConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {"place_in": 0.5, "place_near": 0.3, "pick_lift": 0.2}
    )


@dataclass
class GenerationConfig:
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    front_camera: FrontCameraConfig = field(default_factory=FrontCameraConfig)
    wrist_camera: WristCameraConfig = field(default_factory=WristCameraConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    start_pose: StartPoseConfig = field(default_factory=StartPoseConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    tasks: TaskMixConfig = field(default_factory=TaskMixConfig)
    fps: int = 30
    height: int = 480
    width: int = 640
    # Hard cap on an episode's length; longer plans are discarded.
    max_episode_seconds: float = 30.0
