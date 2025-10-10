#!/usr/bin/env python3
"""
SO-101 Teleop — 3DoF XYZ with upstream arm, fast best-effort vertical with wrist-flex,
independent wrist-roll + gripper, and basic gravity compensation on the wrist.

Controls (world frame):
  W/S -> +X / -X
  A/D -> +Y / -Y
  E/Q -> +Z / -Z
  [ / ] -> wrist roll -
  , / . -> gripper open/close
  R -> re-home to a safe, above-table pose
  ESC -> quit
"""

import os, time, numpy as np
import mujoco as mj
import glfw

# ----------------- CONFIG -----------------
MODEL_XML    = "./gym-hil/gym_hil/assets/SO101/so101_new_calib.xml"
EE_SITE_NAME = "wrist_site"

# DOF names (order matters)
PAN, LIFT, ELBOW, WRIST_FLEX, WRIST_ROLL = \
    "shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"
GRIPPER = "gripper"

ACT_NAMES = [PAN, LIFT, ELBOW, WRIST_FLEX, WRIST_ROLL, GRIPPER]

# Local “approach” axis of wrist_site (you found -Y is correct)
TOOL_AXIS_SITE = np.array([0.0, -1.0, 0.0])

# Teleop speeds / gains
LIN_SPEED  = 0.04      # slower XYZ so tilt dominates during motion
YAW_SPEED  = 1.20
GRIP_SPEED = 0.7

# Wrist-flex tilt controller (small-angle model)
ORI_GAIN      = 6.0     # aggressive correction toward world -Z
TILT_DEADZONE = 0.03    # rad; ignore tiny errors
TILT_WMAX     = 6.0     # max angular speed command in tilt task (sane cap)
LAMBDA_POS    = 1.0e-2  # DLS for XYZ
LAMBDA_TILT   = 1.0e-4  # DLS for 2x1 wrist tilt solve (very permissive)

# Rate limiting / smoothing
VEL_LIMIT        = 0.5   # big joints
VEL_LIMIT_WRIST  = 8.0   # wrist joints can move fast
SMOOTH_DQ        = 0.30  # big joints smoothing
SMOOTH_DQ_WRIST  = 0.08  # very responsive wrist

# Gravity compensation (wrist-only)
WRIST_GFF_GAIN   = 0.5   # multiplies data.qfrc_bias torque to a dq “assist”; tune 0.2..0.8

DT_CTRL   = 1/200.0
TABLE_Z   = 0.0
CLEARANCE = 0.07         # start >=7 cm above table
# ------------------------------------------

# ---------- helpers ----------
def clamp(x, lo, hi): return np.minimum(np.maximum(x, lo), hi)
def jid(model, name):  return mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
def aid(model, name):  return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
def sid(model, name):  return mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
def rotmat_from_xmat(xmat9): return np.array(xmat9, dtype=float).reshape(3,3)

def joint_limit_margin(q, lo, hi, idx):
    """[0..1]: 1 centered, 0 at limit. Used to fade tilt near limits."""
    mid  = 0.5*(lo[idx] + hi[idx]); half = 0.5*(hi[idx]-lo[idx]) + 1e-9
    return float(np.clip(1.0 - abs(q[idx]-mid)/half, 0.0, 1.0))

def pick_feasible_home(model, data, dofs, site_id):
    """Choose elbow-bent, above-table, down-ish start pose with decent conditioning."""
    j_lo = model.jnt_range[:,0].copy(); j_hi = model.jnt_range[:,1].copy()
    q_best = data.qpos.copy(); best = -1e9; q0 = data.qpos.copy()

    shoulder = np.deg2rad([-10, 0, 15, 30, 40])
    elbow    = np.deg2rad([ 50, 70, 90,110,125])
    wrist    = np.deg2rad([-70,-50,-30,-15,  0])

    for sh in shoulder:
        for el in elbow:
            for wf in wrist:
                q = q0.copy()
                q[dofs[PAN]]        = 0.0
                q[dofs[LIFT]]       = np.clip(sh, j_lo[dofs[LIFT]],  j_hi[dofs[LIFT]])
                q[dofs[ELBOW]]      = np.clip(el, j_lo[dofs[ELBOW]], j_hi[dofs[ELBOW]])
                q[dofs[WRIST_FLEX]] = np.clip(wf, j_lo[dofs[WRIST_FLEX]], j_hi[dofs[WRIST_FLEX]])
                q[dofs[WRIST_ROLL]] = 0.0
                data.qpos[:] = q; data.qvel[:] = 0.0
                mj.mj_forward(model, data)

                pos = data.site_xpos[site_id]
                if pos[2] < TABLE_Z + CLEARANCE:  # ensure above table
                    continue

                R = rotmat_from_xmat(data.site_xmat[site_id])
                a_tool = R @ TOOL_AXIS_SITE
                align  = float(np.dot(a_tool, np.array([0,0,-1.0])))  # 1 = perfect down

                Jp = np.zeros((3,model.nv)); Jr = np.zeros((3,model.nv))
                mj.mj_jacSite(model, data, Jp, Jr, site_id)
                J3 = Jp[:, [dofs[PAN], dofs[LIFT], dofs[ELBOW]]]
                s = np.linalg.svd(J3, compute_uv=False)
                sig = float(np.min(s)) if s.size else 0.0

                score = 3.0*align + 1.0*sig + 0.5*pos[2]
                if score > best:
                    best = score; q_best = q.copy()

    data.qpos[:] = q_best; data.qvel[:] = 0.0
    mj.mj_forward(model, data)
    return q_best.copy()

# ---------- main ----------
def main():
    if not os.path.exists(MODEL_XML): raise FileNotFoundError(MODEL_XML)
    model = mj.MjModel.from_xml_path(MODEL_XML)
    data  = mj.MjData(model)

    site_id = sid(model, EE_SITE_NAME)
    if site_id < 0: raise RuntimeError(f"Site {EE_SITE_NAME} not found")

    # map joint/actuator ids
    dofs = {n: model.jnt_dofadr[jid(model,n)] for n in [PAN,LIFT,ELBOW,WRIST_FLEX,WRIST_ROLL,GRIPPER]}
    acts = {n: aid(model,n) for n in ACT_NAMES}

    j_lo = model.jnt_range[:,0].copy(); j_hi = model.jnt_range[:,1].copy()
    mj.mj_forward(model, data)

    # home above table
    q_home = pick_feasible_home(model, data, dofs, site_id)
    q_des  = q_home.copy()
    for n in ACT_NAMES: data.ctrl[acts[n]] = q_des[dofs[n]]

    # viewer
    if not glfw.init(): raise RuntimeError("GLFW init failed")
    w,h = 1280,720
    window = glfw.create_window(w,h,"SO-101 Teleop (wrist_site EE)",None,None)
    glfw.make_context_current(window); glfw.swap_interval(1)

    class KB:
        def __init__(self): self.down=set(); glfw.set_key_callback(window,self.cb)
        def cb(self,win,k,s,a,m): self.down.add(k) if a==glfw.PRESS else self.down.discard(k) if a==glfw.RELEASE else None
        def held(self,k): return k in self.down
    kb = KB()

    cam = mj.MjvCamera(); opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam); cam.distance=1.3; cam.azimuth=140; cam.elevation=-20
    scene = mj.MjvScene(model, maxgeom=10000); ctx = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150)

    last=time.time(); acc=0.0
    nv = model.nv; dq_filt = np.zeros(nv)

    while not glfw.window_should_close(window):
        now=time.time(); dt=now-last; last=now; acc+=dt

        # world-frame teleop
        vx=vy=vz=0.0
        if kb.held(glfw.KEY_W): vx += LIN_SPEED
        if kb.held(glfw.KEY_S): vx -= LIN_SPEED
        if kb.held(glfw.KEY_A): vy += LIN_SPEED
        if kb.held(glfw.KEY_D): vy -= LIN_SPEED
        if kb.held(glfw.KEY_E): vz += LIN_SPEED
        if kb.held(glfw.KEY_Q): vz -= LIN_SPEED
        moving = (vx!=0.0) or (vy!=0.0) or (vz!=0.0)

        yawrate=0.0
        if kb.held(glfw.KEY_LEFT_BRACKET):  yawrate -= YAW_SPEED
        if kb.held(glfw.KEY_RIGHT_BRACKET): yawrate += YAW_SPEED

        grip_delta=0.0
        if kb.held(glfw.KEY_COMMA):  grip_delta -= GRIP_SPEED
        if kb.held(glfw.KEY_PERIOD): grip_delta += GRIP_SPEED

        if kb.held(glfw.KEY_R):
            q_home = pick_feasible_home(model, data, dofs, site_id)
            q_des[:] = q_home
            mj.mj_forward(model, data)
            for n in ACT_NAMES: data.ctrl[acts[n]] = q_des[dofs[n]]

        if kb.held(glfw.KEY_ESCAPE): break

        while acc >= DT_CTRL:
            acc -= DT_CTRL
            mj.mj_forward(model, data)

            # Jacobians at EE
            Jp = np.zeros((3,nv)); Jr = np.zeros((3,nv))
            mj.mj_jacSite(model, data, Jp, Jr, site_id)

            # --- PRIMARY: XYZ via (PAN, LIFT, ELBOW) only ---
            cols = [dofs[PAN], dofs[LIFT], dofs[ELBOW]]
            J3 = Jp[:, cols]
            v_des = np.array([vx,vy,vz])
            A = J3 @ J3.T + (LAMBDA_POS**2)*np.eye(3)
            dq3 = J3.T @ np.linalg.solve(A, v_des)
            dq  = np.zeros(nv); dq[cols] = dq3

            # --- SECONDARY: best-effort vertical via WRIST_FLEX only ---
            R = rotmat_from_xmat(data.site_xmat[site_id])
            a_tool = R @ TOOL_AXIS_SITE
            e = np.cross(a_tool, np.array([0,0,-1.0]))
            err_xy = e[:2]; err_mag = float(np.linalg.norm(err_xy))

            jcol   = Jr[:2, dofs[WRIST_FLEX]]
            jnorm2 = float(jcol.T @ jcol)

            # fade tilt when near singular XYZ or near WF limit
            s = np.linalg.svd(J3, compute_uv=False)
            smin = float(np.min(s)) if s.size else 0.0
            sing_scale = np.clip(smin/0.10, 0.0, 1.0)

            # --- Directional limit check ---
            q = data.qpos[dofs[WRIST_FLEX]]
            lo = j_lo[dofs[WRIST_FLEX]]
            hi = j_hi[dofs[WRIST_FLEX]]
            mid = 0.5 * (lo + hi)

            # desired delta for wrist flex from correction (dq_wf computed later)
            # but we can compute the sign *before* applying
            # Project error direction onto wrist sensitivity:
            desired_dq_sign = np.sign(dq_wf) if 'dq_wf' in locals() else 0  # handle before calculation

            # distance to limits
            dist_lo = abs(q - lo)
            dist_hi = abs(hi - q)
            limit_threshold = 0.1  # radians, region considered "near limit"

            if dist_lo < limit_threshold and desired_dq_sign < 0:
                # pushing further INTO lower limit → suppress
                lim_scale = dist_lo / limit_threshold  # fades from 0 to 1
            elif dist_hi < limit_threshold and desired_dq_sign > 0:
                # pushing further INTO upper limit → suppress
                lim_scale = dist_hi / limit_threshold
            else:
                # moving away or not near limit → full authority
                lim_scale = 1.0
            tilt_scale = sing_scale * lim_scale

            if (moving or err_mag > TILT_DEADZONE) and tilt_scale > 1e-3 and jnorm2 > 1e-8:
                w_xy = ORI_GAIN * tilt_scale * err_xy
                nrm  = np.linalg.norm(w_xy)
                if nrm > TILT_WMAX: w_xy *= TILT_WMAX/(nrm + 1e-9)
                dq_wf = float(jcol.T @ w_xy) / (jnorm2 + LAMBDA_TILT**2)
                dq[dofs[WRIST_FLEX]] += dq_wf

            # independent wrist-roll spin
            dq[dofs[WRIST_ROLL]] += yawrate

            # --- Gravity compensation for wrist flex (simple feedforward) ---
            # Use MuJoCo bias torques (gravity + Coriolis); mostly gravity here.
            tau_g = data.qfrc_bias[dofs[WRIST_FLEX]]
            dq[dofs[WRIST_FLEX]] += WRIST_GFF_GAIN * tau_g  # heuristic velocity assist

            # ---- Rate limit + smoothing (per joint) ----
            dq_lim = VEL_LIMIT * np.ones(nv)
            dq_lim[dofs[WRIST_FLEX]] = VEL_LIMIT_WRIST
            dq_lim[dofs[WRIST_ROLL]] = VEL_LIMIT_WRIST
            dq = np.clip(dq, -dq_lim, dq_lim)

            alpha = SMOOTH_DQ * np.ones(nv)
            alpha[dofs[WRIST_FLEX]] = SMOOTH_DQ_WRIST
            alpha[dofs[WRIST_ROLL]] = SMOOTH_DQ_WRIST
            dq_filt = (1.0 - alpha) * dq_filt + alpha * dq

            # integrate → target q, clamp
            q_des += dq_filt * DT_CTRL
            q_des = clamp(q_des, j_lo, j_hi)

            # send to position actuators
            for n in ACT_NAMES:
                data.ctrl[acts[n]] = q_des[dofs[n]]

            # gripper rate control
            gidx = acts[GRIPPER]
            data.ctrl[gidx] = clamp(
                data.ctrl[gidx] + grip_delta * DT_CTRL,
                j_lo[dofs[GRIPPER]], j_hi[dofs[GRIPPER]]
            )

            mj.mj_step(model, data)

        # render
        view = mj.MjrRect(0,0,w,h)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL, scene)
        mj.mjr_render(view, scene, ctx)
        glfw.swap_buffers(window); glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
