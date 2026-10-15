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

"""How an SO-101 arm, leader or follower, calibrates itself (`lerobot-calibrate ... --auto=true`).

The engine is lerobot.motors.feetech.auto_calibration. This file only says what moves when, so a change of
choreography (another pose for measuring a joint, another grouping) is a change of data here.
"""

from dataclasses import replace

from lerobot.motors.feetech.auto_calibration import AutoCalibrationPlan, JointRange, Move, Sweep, Unfold

SO_ARM_START_POSE = """\
Start pose: the arm folded up as it is packed away: upper arm tilted back, forearm folded down onto it, the
gripper closed and hanging down in front of the base, or a leader's handle hanging down the same way (not
pointing up: the folded arm then leaves the wrist only about 60 degrees of room). Keep the space around the
arm clear: it stretches out forwards, turns its base about 100 degrees each way with the gripper pointing
forwards, and returns to the start pose at the end."""

SO_ARM_PLAN = AutoCalibrationPlan(
    start_pose=SO_ARM_START_POSE,
    # A follower's elbow (STS3215, 1/345) holds the stretched forearm at about 15 % load, well under this limit.
    torque_limit=500,
    # Accepted span of each joint's range: about 10 degrees either side of what an SO-101 follower measures, 194
    # (pan), 213 (lift), 197 (elbow), 207 (wrist flex), 343 (wrist roll, stopper to stopper) and 141 (gripper), so
    # that a joint stopped well before its end stop (by a clamp, the table, a cable) is refused.
    joints={
        "shoulder_pan": JointRange(185, 205),
        "shoulder_lift": JointRange(200, 225),
        "elbow_flex": JointRange(187, 208),
        "wrist_flex": JointRange(195, 218),
        "wrist_roll": JointRange(325, 360, full_turn_ok=True),
        "gripper": JointRange(128, 152),
    },
    steps=(
        # Lift the wrist, the upper arm and the forearm a little, each in its unfold direction (the same on the
        # leader and the follower), then back to the start pose.
        Unfold("wrist_flex", 80, sign=-1),
        Unfold("shoulder_lift", 15, sign=1),
        Unfold("elbow_flex", 30, sign=-1),
        Move({"shoulder_lift": "start", "elbow_flex": "start"}),
        # Upper arm and forearm together: folded back, then stretched out forwards. With the wrist unfolded
        # (gripper forwards, jaws opening sideways), the straight arm reaches the shoulder's own forward stop
        # level with and clear of the table the base is clamped to.
        Sweep(("shoulder_lift", "elbow_flex"), first="fold"),
        Move({"shoulder_lift": "fold", "elbow_flex": "fold"}),
        # Forearm raised: the wrist and the gripper turn in free air.
        Move({"elbow_flex": 80}),
        Sweep(("wrist_roll", "gripper", "wrist_flex")),
        # Still in free air: the wrist back to where it unfolded, the jaws as at the start, the gripper closed.
        Move({"wrist_roll": "start", "gripper": "start", "wrist_flex": "unfolded"}),
        # Folded arm, gripper forwards (the pose after the unfolds): the base turns from stop to stop.
        Move({"shoulder_lift": "start", "elbow_flex": "start"}),
        Sweep(("shoulder_pan",)),
        # Back to the start pose. While the arm is folded the wrist only turns between hanging down and
        # forwards: turned up, the gripper (or a camera on the wrist) meets the forearm.
        Move({"shoulder_pan": "start"}),
        Move({"wrist_flex": "start"}),
    ),
)

# The leader's servos are geared for a hand to move them (1/191 and 1/147, see so101.mdx) and run on 5 V, so they
# need more of their torque: a leader's elbow held the stretched forearm at up to 48 %. Its spans are about 10 degrees
# either side of what a leader measures: 212 (lift), 195 (elbow), 203 (wrist flex), 338 (wrist roll) and 112 (the
# trigger). Its pan turned 239 degrees, 45 more than a follower's, so the pan accepts both.
SO_LEADER_PLAN = replace(
    SO_ARM_PLAN,
    torque_limit=800,
    joints={
        "shoulder_pan": JointRange(185, 250),
        "shoulder_lift": JointRange(200, 223),
        "elbow_flex": JointRange(185, 206),
        "wrist_flex": JointRange(192, 213),
        "wrist_roll": JointRange(325, 360, full_turn_ok=True),
        "gripper": JointRange(100, 122),
    },
)
