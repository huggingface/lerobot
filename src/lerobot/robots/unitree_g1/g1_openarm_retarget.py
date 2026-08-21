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

"""Drive the Unitree G1's arms from OpenArm joint angles, fast enough for a control loop.

A policy trained on the OpenArm emits joint angles for *that* arm, which mean nothing on the
G1's very different geometry. What does transfer is where the hands end up, so one frame of
retargeting is:

    16-D OpenArm state (deg) -> OpenArm FK in a merged scene -> hand TCP poses
      -> EE_OFFSET -> G1 wrist targets -> DLS IK on the 14 G1 arm joints -> ``<joint>.q``

Both robots live in one MuJoCo model so the two sets of frames share a world and the FK
output can be handed straight to the IK. The OpenArm is attached under the ``oa_`` prefix and
placed so its shoulders sit on the G1's.

Costs ~2.4 ms/frame, unlike the ipopt ``G1_29_ArmIK`` in ``g1_kinematics`` at ~114 ms, which
is why offline retargeting scripts use that one and the live path uses this. The two were
checked to aim at the same wrist frames to 0.000 mm.

Pair with ``SonicLowerBodyController``, which publishes legs and waist only, so these arm
targets are the only thing writing motors 15-28.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..openarm_follower.openarm_kinematics import (
    ARM_JOINT_SLICES,
    GRIP_FULL_DEG,
    GRIPPER_IDX,
    POLICY_ORDER,
    ArmIKSolver,
    find_mjcf as find_openarm_mjcf,
)

G1_MJCF_REPO = "lerobot/unitree-g1-mujoco"
G1_MJCF_RELATIVE = "assets/g1_29dof_with_hand.xml"

OA_PREFIX = "oa_"

# Solved in the order the IK reports them: left arm, then right.
G1_ARM_JOINTS = [
    f"{side}_{name}"
    for side in ("left", "right")
    for name in (
        "shoulder_pitch_joint",
        "shoulder_roll_joint",
        "shoulder_yaw_joint",
        "elbow_joint",
        "wrist_roll_joint",
        "wrist_pitch_joint",
        "wrist_yaw_joint",
    )
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

# MJCF joint name -> the motor key `UnitreeG1.send_action` expects.
JOINT_TO_MOTOR = {
    "waist_yaw_joint": "kWaistYaw",
    "waist_roll_joint": "kWaistRoll",
    "waist_pitch_joint": "kWaistPitch",
    "left_shoulder_pitch_joint": "kLeftShoulderPitch",
    "left_shoulder_roll_joint": "kLeftShoulderRoll",
    "left_shoulder_yaw_joint": "kLeftShoulderYaw",
    "left_elbow_joint": "kLeftElbow",
    "left_wrist_roll_joint": "kLeftWristRoll",
    "left_wrist_pitch_joint": "kLeftWristPitch",
    "left_wrist_yaw_joint": "kLeftWristYaw",
    "right_shoulder_pitch_joint": "kRightShoulderPitch",
    "right_shoulder_roll_joint": "kRightShoulderRoll",
    "right_shoulder_yaw_joint": "kRightShoulderYaw",
    "right_elbow_joint": "kRightElbow",
    "right_wrist_roll_joint": "kRightWristRoll",
    "right_wrist_pitch_joint": "kRightWristPitch",
    "right_wrist_yaw_joint": "kRightWristYaw",
}

# The recorded folding data comes off the long build (upper arm +5 cm); the published v1 MJCF
# is the short one. Only the elbow origin differs, so patching it is the whole change.
LONG_ELBOW_POS = [-0.0, 0.0415354, 0.202733]
EE_OFFSET_X = 0.05  # L_ee/R_ee sit 5 cm out along the wrist_yaw joint's local x, as in the IK

# OpenArm hand TCP -> G1 wrist target, tuned in overlay_meshcat.py against the pelvis-at-origin
# frame. The rotation is the hand-frame convention change (the two grippers' axes differ);
# the translation backs off 15 cm along the approach so the G1 palm, not its wrist, lands on
# the OpenArm's grasp point.
EE_OFFSET = np.array(
    [
        [-0.0697564737, 0.0, 0.9975640503, -0.005],
        [0.0, -1.0, 0.0, 0.010],
        [0.9975640503, 0.0, 0.0697564737, -0.150],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Where the OpenArm sits in the pelvis-at-origin frame. Shoulder-on-shoulder would be
# -0.406, but the OpenArm is the longer arm, so matching the shoulders puts its workspace
# lower than the G1's and the recorded wrist poses land under everything the G1 can reach
# comfortably. Raised 16 cm to line up the reachable volumes instead of the mounting points.
#
# Swept against episode 1's reach error: anything from 8 to 16 cm clears the out-of-reach
# threshold entirely, with the flattest error around 12. Past ~20 cm it collapses (a
# quarter of the episode out of reach) as the targets climb out of the top of the G1's
# workspace, so this sits nearer that edge than the numbers alone would pick.
OA_PLACEMENT_XYZ = (0.0, 0.0, -0.246)

# Tuned in render_g1_openarm.py.
IK_ITERS = 60
# The reverse projection is a cheaper problem than the forward one -- 7 DoF per side rather
# than the whole upper body, and warm-started from the previous frame's answer -- so it
# converges in far fewer sweeps. It runs every tick alongside the forward solve, and on the
# Jetson that difference is the budget.
REVERSE_IK_ITERS = 25
ROT_WEIGHT = 0.3
POSTURE_GAIN = 0.25
ELBOW_GAIN = 1.0
WAIST_STIFFNESS = 8.0
WAIST_RETURN = 0.2
WAIST_PITCH_DEG = 12.0


def find_g1_mjcf() -> str:
    """Path to the G1 MJCF, downloading the asset repo if it is not cached."""
    from huggingface_hub import snapshot_download

    override = os.environ.get("LEROBOT_G1_MJCF")
    if override:
        if not Path(override).is_file():
            raise FileNotFoundError(f"LEROBOT_G1_MJCF points at a missing file: {override}")
        return override
    return str(Path(snapshot_download(G1_MJCF_REPO)) / G1_MJCF_RELATIVE)


def absolutize_assets(spec, xml: str) -> None:
    """Rewrite mesh/texture paths to absolute so a merged spec compiles from any cwd."""
    root = os.path.abspath(Path(xml).parent)
    meshdir = spec.meshdir or ""
    texdir = spec.texturedir or spec.meshdir or ""
    # normpath, not resolve(): the HF snapshot's assets are symlinks into blobs/ with no file
    # extension, and MuJoCo picks its mesh decoder from the extension.
    for mesh in spec.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.normpath(os.path.join(root, meshdir, mesh.file))
    for tex in spec.textures:
        if tex.file and not os.path.isabs(tex.file):
            tex.file = os.path.normpath(os.path.join(root, texdir, tex.file))
    spec.meshdir, spec.texturedir = "", ""


def build_scene(
    g1_xml: str | None = None,
    oa_xml: str | None = None,
    oa_xyz=OA_PLACEMENT_XYZ,
    oa_rpy=(0.0, 0.0, 0.0),
    long_arm: bool = True,
):
    """Compile one model holding both robots, plus ``L_ee``/``R_ee`` sites on the G1 wrists."""
    import mujoco

    g1_xml = g1_xml or find_g1_mjcf()
    oa_xml = oa_xml or find_openarm_mjcf()

    g1 = mujoco.MjSpec.from_file(g1_xml)
    oa = mujoco.MjSpec.from_file(oa_xml)
    absolutize_assets(g1, g1_xml)
    absolutize_assets(oa, oa_xml)

    if long_arm:
        # Only the elbow origin moves; the upper-arm mesh stays put, so the joint shows a
        # visible 5 cm gap when rendered. That is the geometry, not a rendering bug.
        for side in ("left", "right"):
            oa.body(f"openarm_{side}_link4").pos = LONG_ELBOW_POS

    # The IK's end-effector frames are not in the MJCF, so add them: the frame EE_OFFSET was
    # tuned against.
    for side in ("left", "right"):
        g1.body(f"{side}_wrist_yaw_link").add_site(
            name=f"{side[0].upper()}_ee", pos=[EE_OFFSET_X, 0.0, 0.0], size=[0.01] * 3
        )

    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.deg2rad(oa_rpy), "xyz")
    frame = g1.worldbody.add_frame(pos=list(oa_xyz), quat=list(quat))
    g1.attach(oa, prefix=OA_PREFIX, frame=frame)
    return g1.compile()


def openarm_qadr(model) -> tuple[dict[str, int], dict[str, list[tuple[int, float, float]]]]:
    """``qpos`` addresses for the attached OpenArm's 14 arm joints and 2 fingers per side."""
    import mujoco

    arms = {}
    for policy_key in POLICY_ORDER:
        if "gripper" in policy_key:
            continue
        side, index = policy_key.split("_joint_")
        name = f"{OA_PREFIX}openarm_{side}_joint{index}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"OpenArm joint {name} missing from the merged model")
        arms[policy_key] = int(model.jnt_qposadr[jid])

    fingers = {}
    for side in ("left", "right"):
        entries = []
        for finger in ("finger_joint1", "finger_joint2"):
            name = f"{OA_PREFIX}openarm_{side}_{finger}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            entries.append((int(model.jnt_qposadr[jid]), *model.jnt_range[jid]))
        fingers[side] = entries
    return arms, fingers


def openarm_arm_dofs(model, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(qpos addresses, dof addresses, ranges)`` for one attached OpenArm's 7 joints."""
    import mujoco

    qadr, dofadr, rng = [], [], []
    for i in range(1, 8):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{OA_PREFIX}openarm_{side}_joint{i}")
        qadr.append(int(model.jnt_qposadr[jid]))
        dofadr.append(int(model.jnt_dofadr[jid]))
        rng.append(model.jnt_range[jid].copy())
    return np.array(qadr), np.array(dofadr), np.array(rng)


def set_openarm(data, arms, fingers, state16: np.ndarray) -> None:
    """Pose the attached OpenArm from a 16-D policy state vector (degrees)."""
    for i, policy_key in enumerate(POLICY_ORDER):
        if policy_key in arms:
            data.qpos[arms[policy_key]] = np.deg2rad(state16[i])
    for side, gripper_index in (("right", 7), ("left", 15)):
        opening = min(1.0, abs(float(state16[gripper_index])) / GRIP_FULL_DEG)
        for adr, lo, hi in fingers[side]:
            data.qpos[adr] = lo + opening * (hi - lo)  # v1 slide joint: lo closed .. hi open


class ArmIK:
    """Damped-least-squares IK driving the G1's arm joints (optionally plus the waist) to a
    pair of wrist poses.

    ``joints`` is solved in the given order. Two mechanisms keep the waist still unless it is
    actually needed: ``stiffness`` prices waist motion above arm motion in the least-squares
    step, and ``rest_gain`` pulls the waist back toward zero through the task nullspace, so it
    unwinds as soon as the arms can reach on their own. Both leave the wrist targets intact.
    """

    def __init__(
        self,
        model,
        data,
        joints: list[str],
        stiffness: np.ndarray | None = None,
        rest_gain: np.ndarray | None = None,
        rot_weight: float = ROT_WEIGHT,
        damping: float = 0.12,
        iters: int = IK_ITERS,
        step_limit: float = 0.25,
        elbow_gain: float = 0.0,
    ):
        import mujoco

        self._mj = mujoco
        self.model, self.data = model, data
        self.rot_weight, self.damping = rot_weight, damping
        self.iters, self.step_limit = iters, step_limit
        self.elbow_gain = elbow_gain

        self.elbows = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_elbow_link") for s in ("left", "right")
        ]
        self.shoulders = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_pitch_link")
            for s in ("left", "right")
        ]
        # The G1's upper arm is its own length, so an OpenArm elbow *position* is not
        # reachable. What transfers is the direction the upper arm points in, which is what
        # "same elbow pose" actually means across two different arms.
        mujoco.mj_kinematics(model, data)
        self._upper_arm = [
            float(np.linalg.norm(data.xpos[e] - data.xpos[s]))
            for e, s in zip(self.elbows, self.shoulders, strict=True)
        ]

        self.joints = list(joints)
        weights = np.ones(len(joints)) if stiffness is None else np.asarray(stiffness, float)
        self._winv = 1.0 / weights
        self._rest_gain = None if rest_gain is None else np.asarray(rest_gain, float)
        self._q_rest = np.zeros(len(joints))

        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints]
        if min(jids) < 0:
            missing = [n for n, j in zip(joints, jids, strict=True) if j < 0]
            raise RuntimeError(f"joints missing from the merged model: {missing}")
        self.qadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dofs = np.array([model.jnt_dofadr[j] for j in jids])
        self.lo = model.jnt_range[jids, 0].copy()
        self.hi = model.jnt_range[jids, 1].copy()
        self.sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_ee") for s in "LR"]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def set_rest(self, q_rest: np.ndarray) -> None:
        """Set the pose the nullspace bias pulls toward (arms: the previous frame's solution,
        so the redundant elbow swivel stops wandering between frames; waist: zero)."""
        self._q_rest = np.asarray(q_rest, float)

    def _error(self, sid: int, target: np.ndarray) -> np.ndarray:
        mj, d = self._mj, self.data
        e_pos = target[:3, 3] - d.site_xpos[sid]
        q_cur, q_tgt, q_neg, q_dif = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
        e_rot = np.zeros(3)
        mj.mju_mat2Quat(q_cur, d.site_xmat[sid])
        mj.mju_mat2Quat(q_tgt, np.ascontiguousarray(target[:3, :3]).reshape(-1))
        mj.mju_negQuat(q_neg, q_cur)
        mj.mju_mulQuat(q_dif, q_tgt, q_neg)
        mj.mju_quat2Vel(e_rot, q_dif, 1.0)
        return np.concatenate([e_pos, self.rot_weight * e_rot])

    def elbow_goals(self, dirs: list[np.ndarray]) -> list[np.ndarray]:
        """Where this robot's elbows should sit to point its upper arms along ``dirs``."""
        return [
            self.data.xpos[s] + self._upper_arm[k] * d / max(np.linalg.norm(d), 1e-9)
            for k, (s, d) in enumerate(zip(self.shoulders, dirs, strict=True))
        ]

    def solve(
        self, targets: list[np.ndarray], elbow_dirs: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine the current pose toward ``targets`` ([left, right] 4x4).

        ``elbow_dirs`` are [left, right] shoulder->elbow directions to copy -- the wrists pin
        only 6 of each arm's 7 DoF, and this is what decides how the leftover swivel is spent.
        Returns (solved joint angles, per-side wrist position error in m).
        """
        mj, model, data = self._mj, self.model, self.data
        eye = np.eye(6 * len(targets))
        for _ in range(self.iters):
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            errs, jac = [], []
            for sid, target in zip(self.sites, targets, strict=True):
                errs.append(self._error(sid, target))
                mj.mj_jacSite(model, data, self._jacp, self._jacr, sid)
                jac.append(np.vstack([self._jacp[:, self.dofs], self.rot_weight * self._jacr[:, self.dofs]]))
            err, jac_task = np.concatenate(errs), np.vstack(jac)
            if np.linalg.norm(err) < 1e-5:
                break
            # Weighted DLS: stiff joints (the waist) are recruited only when the cheap ones
            # cannot do the job.
            jac_pinv = (self._winv[:, None] * jac_task.T) @ np.linalg.inv(
                (jac_task * self._winv) @ jac_task.T + self.damping**2 * eye
            )
            dq = jac_pinv @ err

            # Secondary objectives live in the wrist task's nullspace, so none of them can
            # pull a wrist off target: they only choose among poses that already hit it.
            bias = np.zeros_like(dq)
            if self._rest_gain is not None:
                bias += self._rest_gain * (self._q_rest - data.qpos[self.qadr])
            if elbow_dirs is not None and self.elbow_gain > 0.0:
                for bid, want in zip(self.elbows, self.elbow_goals(elbow_dirs), strict=True):
                    mj.mj_jacBody(model, data, self._jacp, self._jacr, bid)
                    bias += self._jacp[:, self.dofs].T @ (self.elbow_gain * (want - data.xpos[bid]))
            if bias.any():
                dq += bias - jac_pinv @ (jac_task @ bias)
            dq = np.clip(dq, -self.step_limit, self.step_limit)
            data.qpos[self.qadr] = np.clip(data.qpos[self.qadr] + dq, self.lo, self.hi)

        mj.mj_kinematics(model, data)
        pos_err = np.array(
            [np.linalg.norm(t[:3, 3] - data.site_xpos[s]) for s, t in zip(self.sites, targets, strict=True)]
        )
        return data.qpos[self.qadr].copy(), pos_err


class G1OpenArmRetargeter:
    """Stateful per-frame OpenArm -> G1 arm retargeter.

    Warm starts each solve from the previous solution, which is what keeps the redundant
    elbow swivel from wandering between frames.
    """

    def __init__(
        self,
        ee_offset: np.ndarray | None = None,
        use_waist: bool = False,
        iters: int = IK_ITERS,
        reverse_iters: int = REVERSE_IK_ITERS,
        oa_xyz=OA_PLACEMENT_XYZ,
        long_arm: bool = True,
    ) -> None:
        import mujoco

        self._mj = mujoco
        self.reverse_iters = reverse_iters
        self.model = build_scene(oa_xyz=list(oa_xyz), oa_rpy=[0.0, 0.0, 0.0], long_arm=long_arm)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]  # pelvis at the origin: the tuning frame
        self._arms, self._fingers = openarm_qadr(self.model)

        self.joints = list(G1_ARM_JOINTS) + (list(WAIST_JOINTS) if use_waist else [])
        n_arm = len(G1_ARM_JOINTS)
        n_waist = len(WAIST_JOINTS) if use_waist else 0
        self.ik = ArmIK(
            self.model,
            self.data,
            self.joints,
            stiffness=np.concatenate([np.ones(n_arm), np.full(n_waist, WAIST_STIFFNESS)]),
            rest_gain=np.concatenate([np.full(n_arm, POSTURE_GAIN), np.full(n_waist, WAIST_RETURN)]),
            rot_weight=ROT_WEIGHT,
            iters=iters,
            elbow_gain=ELBOW_GAIN,
        )
        # WAIST_JOINTS order is yaw, roll, pitch: only the last one leans the torso forward.
        self._waist_rest = np.array([0.0, 0.0, np.deg2rad(WAIST_PITCH_DEG)]) if n_waist else np.zeros(0)
        self._n_arm = n_arm

        def body(name: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        self._tcp = {s: body(f"{OA_PREFIX}openarm_{s}_hand_tcp") for s in ("left", "right")}
        self._oa_elbows = [body(f"{OA_PREFIX}openarm_{s}_link4") for s in ("left", "right")]
        self._oa_shoulders = [body(f"{OA_PREFIX}openarm_{s}_link1") for s in ("left", "right")]

        self.ee_offset = EE_OFFSET if ee_offset is None else np.asarray(ee_offset, float)
        self._ee_offset_inv = np.linalg.inv(self.ee_offset)
        self._q_prev: np.ndarray | None = None

        self.action_keys = [f"{JOINT_TO_MOTOR[n]}.q" for n in self.joints]

        # Reverse direction (G1 -> OpenArm), for turning the robot's real arm pose back into
        # the state vector the policy was trained on.
        self._oa_dofs = {s: openarm_arm_dofs(self.model, s) for s in ("left", "right")}
        self._oa_solver = ArmIKSolver()
        self._oa_seed: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._g1_arm_qadr = self.ik.qadr[: self._n_arm]

    def reset(self) -> None:
        """Forget the warm start, so the next solve converges from the model's home pose."""
        self._q_prev = None
        self._oa_seed = {"left": None, "right": None}

    def to_openarm(
        self,
        g1_arm_rad: np.ndarray,
        grippers: dict[str, float] | None = None,
        iters: int | None = None,
    ) -> np.ndarray:
        """Project the G1's real arm pose back into a 16-D OpenArm state (degrees).

        The mirror of ``solve``: pose the G1, read where its wrists actually are, undo
        ``EE_OFFSET`` to recover the OpenArm hand pose that would put them there, and solve
        the OpenArm's joints for it. The policy is trained on its own embodiment, so it has
        to be shown one -- and with relative actions the state it is given is the anchor its
        deltas are added to, so this cannot be faked with a constant.

        ``grippers`` is per-side closedness (0 open .. 1 closed), written into the gripper
        slots in degrees. ``g1_arm_rad`` is the 14 arm joints in ``G1_ARM_JOINTS`` order.
        """
        import mujoco

        if iters is None:
            iters = self.reverse_iters
        self.data.qpos[self._g1_arm_qadr] = np.asarray(g1_arm_rad, float)[: self._n_arm]
        mujoco.mj_kinematics(self.model, self.data)

        state16 = np.zeros(16)
        for k, side in enumerate(("left", "right")):
            wrist = np.eye(4)
            wrist[:3, :3] = self.data.site_xmat[self.ik.sites[k]].reshape(3, 3)
            wrist[:3, 3] = self.data.site_xpos[self.ik.sites[k]]
            target = wrist @ self._ee_offset_inv

            qadr, dofadr, rng = self._oa_dofs[side]
            seed = self._oa_seed[side]
            if seed is None:
                seed = self.data.qpos[qadr].copy()
            q7, _ = self._oa_solver.solve(
                self.model,
                self.data,
                self._tcp[side],
                dofadr,
                qadr,
                rng,
                target[:3, 3],
                target[:3, :3],
                seed,
                seed,  # bias toward the previous solution: no better elbow reference exists
                iters,
            )
            self._oa_seed[side] = q7
            state16[ARM_JOINT_SLICES[side]] = np.rad2deg(q7)

        for side, closedness in (grippers or {}).items():
            # The recorded follower encodes both grippers as negative degrees, 0 closed down
            # to -GRIP_FULL_DEG fully open, so the inverse of the opening read in `solve` has
            # to restore both the sign and the sense. A positive value here is off the
            # manifold the policy was trained on entirely.
            opening = 1.0 - min(1.0, max(0.0, float(closedness)))
            state16[GRIPPER_IDX[side]] = -opening * GRIP_FULL_DEG
        return state16

    def solve(self, state16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Retarget one 16-D OpenArm pose (degrees).

        Returns (joint angles in radians, per-hand wrist position error in m).
        """
        mj = self._mj
        set_openarm(self.data, self._arms, self._fingers, np.asarray(state16, np.float64))
        mj.mj_kinematics(self.model, self.data)

        targets = []
        for side in ("left", "right"):
            pose = np.eye(4)
            pose[:3, :3] = self.data.xmat[self._tcp[side]].reshape(3, 3)
            pose[:3, 3] = self.data.xpos[self._tcp[side]]
            targets.append(pose @ self.ee_offset)
        elbow_dirs = [
            self.data.xpos[e] - self.data.xpos[s]
            for e, s in zip(self._oa_elbows, self._oa_shoulders, strict=True)
        ]

        if self._q_prev is None:
            self.ik.solve(targets, elbow_dirs)  # cold start from home: converge before measuring
        else:
            self.ik.set_rest(np.concatenate([self._q_prev[: self._n_arm], self._waist_rest]))
        q, err = self.ik.solve(targets, elbow_dirs)
        self._q_prev = q
        return q, err

    def action(self, state16: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        """Same as ``solve``, returned as a ``send_action`` dict."""
        q, err = self.solve(state16)
        return dict(zip(self.action_keys, (float(v) for v in q), strict=True)), err
