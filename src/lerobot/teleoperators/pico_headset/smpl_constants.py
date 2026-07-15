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

"""Single source of truth for the SONIC SMPL whole-body action protocol.

These constants define the wire format shared between the PICO teleoperator
(producer, ``pico_headset.py``), the offline motion->dataset converter
(``smpl_to_dataset.py``), the live stream (``smpl_stream.py``), and the Unitree G1
``SonicWholeBodyController`` (consumer). Keeping them here avoids the producer and
consumer silently drifting apart.
"""

# SMPL encoder window geometry (matches ``smpl_joints_10frame_step1``).
WINDOW = 10  # frames per encoder window
N_JOINTS = 24  # SMPL joints per frame
JOINT_DIM = 3  # xyz per joint
SMPL_OBS_DIM = WINDOW * N_JOINTS * JOINT_DIM  # 720

# Flat action-dict keys carrying the reference through the standard lerobot action
# pipeline as scalar floats: ``smpl.0 .. smpl.719`` and ``root.0 .. root.3``.
SMPL_ACTION_PREFIX = "smpl."
ROOT_ACTION_PREFIX = "root."
ROOT_ACTION_DIM = 4  # per-frame SMPL root orientation (wxyz)

# Full per-frame action vector: 720 joint window + 4 root quaternion = 724.
ACTION_DIM = SMPL_OBS_DIM + ROOT_ACTION_DIM

# ── 3-point VR teleop protocol (SONIC encode_mode 1) ─────────────────────────
# An alternative, sparse upper-body reference: 3 root-relative keypoints
# (left wrist, right wrist, neck), each a position + orientation. The lower body /
# locomotion is driven by the planner (joystick/keyboard), not by these targets.
VR3_N_POINTS = 3  # left wrist, right wrist, neck
VR3_POS_DIM = VR3_N_POINTS * 3  # 9  (3 x xyz)
VR3_ORN_DIM = VR3_N_POINTS * 4  # 12 (3 x wxyz)

# Flat action-dict keys: ``vr3_pos.0 .. vr3_pos.8`` and ``vr3_orn.0 .. vr3_orn.11``.
VR3_POS_PREFIX = "vr3_pos."
VR3_ORN_PREFIX = "vr3_orn."
