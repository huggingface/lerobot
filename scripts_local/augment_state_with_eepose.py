#!/usr/bin/env python3
"""Augment a SARM dataset with EE pose via offline mujoco FK.

Reads observation.state (joints, 7-D) → computes ee_pos(3) + ee_quat(4) + grip_qpos(1)
via mujoco forward kinematics → writes new dataset with state expanded to 15-D.

Usage:
    uv run python scripts_local/augment_state_with_eepose.py \\
        --src-repo-id local/sim_3stage_v2_full_v2_nostale \\
        --dst-repo-id local/sim_3stage_v2_full_v2_nostale_ee
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import mujoco

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-repo-id", required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--ee-body-name", default="gripper_base", help="mujoco body name for end effector")
    ap.add_argument("--gripper-finger-joint", default="left_driver_joint", help="finger qpos joint name (auto-detect if None)")
    ap.add_argument("--n-arm-joints", type=int, default=6, help="number of arm joints (UR10=6)")
    args = ap.parse_args()

    src = Path(HF_LEROBOT_HOME) / args.src_repo_id
    dst = Path(HF_LEROBOT_HOME) / args.dst_repo_id

    import simulator_for_il_rl
    pkg_root = Path(simulator_for_il_rl.__file__).parent
    mjcf = pkg_root / "robot" / "scene.xml"
    print(f"[mjcf] using {mjcf}")

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    ee_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.ee_body_name)
    print(f"[fk] ee_body={args.ee_body_name} idx={ee_idx}")

    # State layout in v2 dataset: [arm_joints(6), gripper_width(1)] = 7-D
    n_arm = args.n_arm_joints  # 6 for UR10
    arm_qpos_idx = list(range(n_arm))
    print(f"[fk] arm_qpos_idx={arm_qpos_idx}")

    # Detect finger joint
    finger_qpos_adr = None
    finger_min, finger_max = 0.0, 1.0
    if args.gripper_finger_joint:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, args.gripper_finger_joint)
        finger_qpos_adr = model.jnt_qposadr[jid]
        finger_min = model.jnt_range[jid][0] if model.jnt_limited[jid] else 0.0
        finger_max = model.jnt_range[jid][1] if model.jnt_limited[jid] else 1.0
        print(f"[fk] finger qpos_adr={finger_qpos_adr} range=[{finger_min},{finger_max}]")

    # Copy dataset
    if dst.exists():
        shutil.rmtree(dst)
    print(f"[copy] {src} → {dst}")
    shutil.copytree(src, dst)

    # Walk data parquet files, augment observation.state
    data_root = dst / "data"
    data_files = sorted(data_root.rglob("*.parquet"))
    print(f"[data] {len(data_files)} parquet files")

    for f in data_files:
        t = pq.read_table(f)
        df = t.to_pandas()
        states = df["observation.state"].to_list()
        new_states = []
        for s in states:
            s_arr = np.asarray(s, dtype=np.float64)
            arm = s_arr[:n_arm]
            grip_in = float(s_arr[n_arm]) if len(s_arr) > n_arm else 0.0
            data.qpos[:n_arm] = arm
            mujoco.mj_forward(model, data)
            ee_pos = data.xpos[ee_idx].copy()
            ee_quat = data.xquat[ee_idx].copy()
            new_state = np.concatenate([arm, [grip_in], ee_pos, ee_quat]).astype(np.float32)  # 6+1+3+4=14
            new_states.append(new_state.tolist())
        # Replace column
        df = df.drop(columns=["observation.state"])
        df["observation.state"] = new_states
        # Write back
        pq.write_table(pa.Table.from_pandas(df), f)
        print(f"[wrote] {f.name} ({len(states)} rows)")

    # Update meta info.json shape
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if "observation.state" in info["features"]:
        info["features"]["observation.state"]["shape"] = [14]
    info_path.write_text(json.dumps(info, indent=2))
    print(f"[meta] updated info.json state shape → [14]")

    # Update episode-level stats (per-ep) — set/clear so loader recomputes if needed
    print(f"[done] dst={dst}")


if __name__ == "__main__":
    main()
