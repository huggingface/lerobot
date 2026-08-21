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

"""OpenArm MuJoCo geometry and end-effector IK, shared by every retargeting path.

Cross-embodiment replay all works the same way: drive one model with the joint angles you
have, read the hand TCP pose out of it (FK), then solve another model's 7 arm joints to put
its TCP in the same place (IK). Only the pair of models changes -- short<->long OpenArm for
policy/hardware mismatch, OpenArm<->G1 for driving the humanoid.

The 7-DOF arm is redundant for a 6-DOF pose target, so the same TCP admits a continuum of
elbow configurations. ``ArmIKSolver`` spends that freedom on staying near a reference pose
rather than letting the solver wander into an arbitrary branch.

Joint angles crossing this module's API are **degrees**, matching the recorded datasets and
the follower's feature keys; everything internal is radians.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SIDES = ("right", "left")

# The 16-vector layout the folding datasets and policy use: each arm's 7 joints followed by
# its gripper, right block first.
POLICY_ORDER = [
    *(f"right_joint_{i}" for i in range(1, 8)),
    "right_gripper",
    *(f"left_joint_{i}" for i in range(1, 8)),
    "left_gripper",
]
ARM_JOINT_SLICES = {"right": slice(0, 7), "left": slice(8, 15)}
GRIPPER_IDX = {"right": 7, "left": 15}

ARM_MAP = {
    **{f"right_joint_{i}": f"openarm_right_joint{i}" for i in range(1, 8)},
    **{f"left_joint_{i}": f"openarm_left_joint{i}" for i in range(1, 8)},
}
TCP = {s: f"openarm_{s}_hand_tcp" for s in SIDES}
FINGER_JOINTS = {s: [f"openarm_{s}_finger_joint1", f"openarm_{s}_finger_joint2"] for s in SIDES}

GRIP_FULL_DEG = 65.0  # follower gripper limit magnitude -> fully open
UPPER_ARM_EXTRA = 0.05  # the "long" build's upper arm is 5 cm longer than the published one

_MJCF_ENV_VAR = "LEROBOT_OPENARM_MJCF"
_MJCF_RELATIVE = Path("third_party/openarm_mujoco/v1/openarm_bimanual.xml")


def find_mjcf() -> str:
    """Locate the bimanual OpenArm MJCF.

    The meshes are a third-party checkout rather than a packaged asset, so there is no path
    that is right for every machine. ``LEROBOT_OPENARM_MJCF`` wins; otherwise walk up from
    the working directory looking for the usual ``third_party/`` layout.
    """
    override = os.environ.get(_MJCF_ENV_VAR)
    if override:
        if not Path(override).is_file():
            raise FileNotFoundError(f"{_MJCF_ENV_VAR} points at a missing file: {override}")
        return override

    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / _MJCF_RELATIVE
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find {_MJCF_RELATIVE} above {Path.cwd()}. "
        f"Set {_MJCF_ENV_VAR} to the bimanual OpenArm MJCF."
    )


def make_long(mjcf: str | None = None, m_short=None):
    """Build the "long" OpenArm: upper arm ``UPPER_ARM_EXTRA`` longer, elbow unmoved at rest.

    The folding data was recorded on a build whose upper arm is 5 cm longer than the
    published MJCF, and recorded joint angles only mean the right thing on the geometry they
    were recorded on. Lengthening the bone alone would push the whole forearm outward, so the
    shoulder mount slides back up the bone by the same amount: at rest the elbow and
    everything below it land in exactly the same world pose, and only the upper arm differs.

    Returns the model and a per-side report of what moved (for parity checks).
    """
    import mujoco

    mjcf = mjcf or find_mjcf()
    if m_short is None:
        m_short = mujoco.MjModel.from_xml_path(mjcf)
    m = mujoco.MjModel.from_xml_path(mjcf)  # fresh copy: the caller keeps the short one

    d = mujoco.MjData(m_short)
    mujoco.mj_resetData(m_short, d)
    mujoco.mj_forward(m_short, d)

    info = {}
    for side in SIDES:
        link3 = mujoco.mj_name2id(m_short, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link3")
        link4 = mujoco.mj_name2id(m_short, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link4")
        link0 = mujoco.mj_name2id(m_short, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_link0")

        elbow_local = m_short.body_pos[link4].copy()  # elbow offset in the link3 frame
        bone = float(np.linalg.norm(elbow_local))
        elbow_new = elbow_local * (bone + UPPER_ARM_EXTRA) / bone
        # Shift the mount back along the bone in world terms, so the elbow stays put.
        mount_shift = -(d.xmat[link3].reshape(3, 3) @ (elbow_new - elbow_local))

        m.body_pos[link4] = elbow_new
        m.body_pos[link0] = m_short.body_pos[link0] + mount_shift
        info[side] = {
            "bone_old": bone,
            "bone_new": bone + UPPER_ARM_EXTRA,
            "mount_shift": mount_shift.copy(),
        }
    return m, info


def joint_adr(m) -> dict[str, int]:
    """``qpos`` address of every arm joint, keyed by its policy-order name."""
    import mujoco

    return {
        policy_key: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)])
        for policy_key, mjcf_name in ARM_MAP.items()
    }


def arm_dofs(m, side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(qpos addresses, dof addresses, joint ranges)`` for one arm's 7 joints, in order."""
    import mujoco

    qadr, dofadr, rng = [], [], []
    for i in range(1, 8):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}")
        qadr.append(int(m.jnt_qposadr[jid]))
        dofadr.append(int(m.jnt_dofadr[jid]))
        rng.append(m.jnt_range[jid].copy())
    return np.array(qadr), np.array(dofadr), np.array(rng)


def finger_adr(m, side: str) -> list[tuple[int, np.ndarray, int]]:
    """``(qpos address, range, joint type)`` for one gripper's finger joints."""
    import mujoco

    out = []
    for name in FINGER_JOINTS[side]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        out.append((int(m.jnt_qposadr[jid]), m.jnt_range[jid].copy(), int(m.jnt_type[jid])))
    return out


def finger_target(gripper_deg: float, rng: np.ndarray, joint_type: int) -> float:
    """Map a gripper command in degrees onto one finger joint's coordinate.

    Hinge fingers take the angle directly (signed by which way the joint opens); sliding
    fingers get the opening as a fraction of travel.
    """
    import mujoco

    if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
        lo, hi = rng
        magnitude = np.deg2rad(min(abs(gripper_deg), GRIP_FULL_DEG))
        return float(np.clip(-magnitude if lo < 0 else magnitude, lo, hi))
    opening = min(1.0, abs(gripper_deg) / GRIP_FULL_DEG)
    return float(rng[0] + opening * (rng[1] - rng[0]))


def set_arms(m, d, qadr: dict[str, int], state_deg: np.ndarray) -> None:
    """Write a 16-vector of policy-order joint angles (degrees) into ``d.qpos``."""
    for i, policy_key in enumerate(POLICY_ORDER):
        if policy_key in qadr:
            d.qpos[qadr[policy_key]] = np.deg2rad(state_deg[i])


def rot_err(rot: np.ndarray, rot_target: np.ndarray) -> np.ndarray:
    """Orientation error between two rotation matrices, as a rotation vector."""
    return 0.5 * (
        np.cross(rot[:, 0], rot_target[:, 0])
        + np.cross(rot[:, 1], rot_target[:, 1])
        + np.cross(rot[:, 2], rot_target[:, 2])
    )


class ArmIKSolver:
    """Damped least-squares IK for a 7-DOF arm reaching a 6-DOF end-effector pose.

    The arm has one redundant DOF, so a pose target alone does not pin the elbow. A secondary
    task pulls the joints toward a reference pose, projected into the task nullspace so it
    cannot move the end-effector. That keeps the elbow near the configuration the motion was
    recorded in instead of letting the solver pick an arbitrary branch, which between frames
    would show up as sudden elbow flips.

    The bias is never allowed to cost end-effector accuracy: it is dropped for the final
    iterations, and ``solve`` falls back to an unbiased solve whenever the bias measurably
    hurts the residual. An infeasible reference degrades elbow matching, never the reach.
    """

    def __init__(
        self,
        null_gain: float = 0.3,
        ee_tol: float = 0.01,
        ee_slack: float = 0.005,
        final_task_iters: int = 3,
        lam: float = 0.06,
        step: float = 0.25,
        null_step: float = 0.08,
    ) -> None:
        self.null_gain = null_gain
        # Only consider the unbiased fallback once the biased residual is worse than this...
        self.ee_tol = ee_tol
        # ...and only take it if the unbiased solve beats the biased one by this much.
        self.ee_slack = ee_slack
        self.final_task_iters = final_task_iters
        self.lam = lam
        self.step = step
        self.null_step = null_step

    def _ik(
        self,
        m,
        d,
        tcp_id: int,
        dofadr: np.ndarray,
        qadr7: np.ndarray,
        rng7: np.ndarray,
        pt: np.ndarray,
        rot_target: np.ndarray,
        seed: np.ndarray,
        q_ref: np.ndarray,
        iters: int,
        use_null: bool = True,
    ) -> tuple[np.ndarray, float]:
        import mujoco

        q = seed.copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        eye6, eye7 = np.eye(6), np.eye(7)
        residual = 1e9

        for it in range(iters):
            d.qpos[qadr7] = q
            # mj_kinematics + mj_comPos is the minimum mj_jac needs; mj_forward would also
            # run the dynamics we never read, and this loop is on the control path.
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

            p = d.xpos[tcp_id].copy()
            rot = d.xmat[tcp_id].reshape(3, 3)
            e = np.concatenate([pt - p, rot_err(rot, rot_target)])
            residual = float(np.linalg.norm(e))

            mujoco.mj_jac(m, d, jacp, jacr, p, tcp_id)
            jac = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])  # 6x7
            # Damped rather than plain pseudo-inverse so the step stays bounded near
            # singularities. Clipped on its own so the task keeps its full authority.
            dq = np.clip(
                jac.T @ np.linalg.solve(jac @ jac.T + (self.lam**2) * eye6, e), -self.step, self.step
            )

            biased = use_null and self.null_gain > 0.0 and (iters - it) > self.final_task_iters
            if biased:
                # True pinv, so jac @ nullproj == 0 exactly and the bias cannot move the TCP.
                nullproj = eye7 - np.linalg.pinv(jac) @ jac
                dq = dq + np.clip(nullproj @ (self.null_gain * (q_ref - q)), -self.null_step, self.null_step)

            q = np.clip(q + dq, rng7[:, 0], rng7[:, 1])
        return q, residual

    def solve(
        self,
        m,
        d,
        tcp_id: int,
        dofadr: np.ndarray,
        qadr7: np.ndarray,
        rng7: np.ndarray,
        pt: np.ndarray,
        rot_target: np.ndarray,
        seed: np.ndarray,
        q_ref: np.ndarray,
        iters: int,
    ) -> tuple[np.ndarray, float]:
        """Solve for the 7 joint angles (radians) putting ``tcp_id`` at ``(pt, rot_target)``.

        ``seed`` starts the descent -- pass the previous solution so consecutive frames stay
        on the same branch. ``q_ref`` is the pose the nullspace bias pulls toward.
        """
        q, residual = self._ik(m, d, tcp_id, dofadr, qadr7, rng7, pt, rot_target, seed, q_ref, iters)
        if self.null_gain > 0.0 and residual > self.ee_tol:
            q_free, residual_free = self._ik(
                m, d, tcp_id, dofadr, qadr7, rng7, pt, rot_target, seed, q_ref, iters, use_null=False
            )
            if residual_free + self.ee_slack < residual:
                return q_free, residual_free
        return q, residual
