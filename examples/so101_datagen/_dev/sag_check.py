"""Gravity sag: does the follower sit below its command more as the arm gets more loaded?

For still frames, regresses the action - state gap of shoulder_lift and elbow_flex on the gravity torque
those joints carry at that pose (from the Menagerie model). Per-episode means are removed first, so
calibration offsets between leader and follower drop out. Real teleop datasets vs our synthetic one.

    python sag_check.py [--repos ...] [--plot review/0_carry/sag.png]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from examples.so101_datagen.so101 import Arm, arm_spec

REAL = [
    "lerobot/svla_so101_pickplace",
    "hungho77/so101_left_arm_pick_red_cube_into_pot",
    "IlyaGusev/so101_pick_gum",
    "legalaspro/so101-pick-and-place-cube-lerobot-30hz",
]
SIM = ["maximellerbach/so101_synthetic_pick_place_4k"]
JOINTS = {1: "shoulder_lift", 2: "elbow_flex"}

parser = argparse.ArgumentParser()
parser.add_argument("--repos", nargs="+", default=REAL + SIM)
parser.add_argument("--max_frames", type=int, default=60000)
parser.add_argument("--plot", default=str(Path(__file__).with_name("review") / "0_carry" / "sag.png"))
args = parser.parse_args()

model = arm_spec().compile()
arm = Arm(model)
data = mujoco.MjData(model)


def gravity(states: np.ndarray) -> np.ndarray:
    """Gravity torque (N m) on every joint at each pose, arm still."""
    out = np.empty((len(states), 6))
    for k, q in enumerate(states):
        data.qpos[arm.qadr] = arm.to_ctrl(q)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        out[k] = data.qfrc_bias[arm.dofadr]
    return out


def load(repo: str) -> pd.DataFrame:
    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["meta/info.json", "data/*"]))
    files = sorted(root.glob("data/**/*.parquet"))
    df = pd.concat(
        [pd.read_parquet(f, columns=["episode_index", "observation.state", "action"]) for f in files]
    )
    return df.iloc[: args.max_frames]


results = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for repo in args.repos:
    df = load(repo)
    state = np.stack(df["observation.state"].to_numpy())[:, :6].astype(np.float64)
    action = np.stack(df["action"].to_numpy())[:, :6].astype(np.float64)
    episode = df["episode_index"].to_numpy()
    # Still: the state barely moves over +-3 frames, so lag and speed don't show as gap.
    speed = np.zeros(len(state))
    for lag in (1, 2, 3):
        same = np.r_[episode[lag:] == episode[:-lag], np.zeros(lag, bool)]
        step = np.r_[np.abs(state[lag:, :5] - state[:-lag, :5]).max(1), np.full(lag, np.inf)]
        speed = np.maximum(speed, np.where(same, step, np.inf))
    still = speed < 0.5
    tau = gravity(state[still])
    gap = (action - state)[still]
    ep = episode[still]
    results[repo] = {"still_frames": int(still.sum())}
    for j, name in JOINTS.items():
        frame = pd.DataFrame({"ep": ep, "gap": gap[:, j], "tau": tau[:, j]})
        demeaned = frame[["gap", "tau"]] - frame.groupby("ep")[["gap", "tau"]].transform("mean")
        slope = float(np.polyfit(demeaned["tau"], demeaned["gap"], 1)[0])
        spread = float(np.quantile(frame["tau"], 0.9) - np.quantile(frame["tau"], 0.1))
        results[repo][name] = {
            "slope_deg_per_Nm": round(slope, 2),
            "tau_p10_p90_Nm": round(spread, 3),
            "sag_over_that_range_deg": round(slope * spread, 2),
        }
        ax = axes[j - 1]
        sample = demeaned.sample(min(len(demeaned), 3000), random_state=0)
        ax.scatter(
            sample["tau"],
            sample["gap"],
            s=2,
            alpha=0.3,
            label=f"{repo.split('/')[1][:28]} ({slope:.1f} deg/Nm)",
        )
        ax.set_title(f"{name}: action - state vs gravity torque, still frames, per-episode demeaned")
        ax.set_xlabel("gravity torque (N m)")
        ax.set_ylabel("gap (deg or units)")
        ax.set_ylim(-8, 8)
for ax in axes:
    ax.legend(fontsize=7, markerscale=5)
    ax.grid(alpha=0.3)
Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.plot, dpi=90, bbox_inches="tight")
print(json.dumps(results, indent=1))
print("plot", args.plot)
