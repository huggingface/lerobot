#!/usr/bin/env python3
"""
绘制主从臂轨迹对比图
用法: python plot_teleop.py teleop_data.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_teleop_data(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} frames from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    # 自动识别关节列
    leader_cols = sorted([c for c in df.columns if c.startswith("leader_")])
    action_cols = sorted([c for c in df.columns if c.startswith("action_")])
    obs_cols = sorted([c for c in df.columns if c.startswith("obs_")])

    # 提取关节编号/名称
    # leader_xxx, action_xxx, obs_xxx 去掉前缀后匹配
    def strip_prefix(cols, prefix):
        return [c.replace(prefix, "") for c in cols]

    leader_names = strip_prefix(leader_cols, "leader_")
    action_names = strip_prefix(action_cols, "action_")
    obs_names = strip_prefix(obs_cols, "obs_")

    # 找到 action 和 obs 的共同关节名（主臂指令 vs 从臂实际）
    common_joints = [n for n in action_names if n in obs_names]
    print(f"\nLeader joints: {leader_names}")
    print(f"Action joints: {action_names}")
    print(f"Obs joints:    {obs_names}")
    print(f"Common (action vs obs): {common_joints}")

    t = df["timestamp"].values

    # ============ 图1: 主臂指令 vs 从臂实际位置 ============
    n_joints = len(common_joints)
    if n_joints == 0:
        print("No common joints found between action_ and obs_ columns!")
        return

    fig1, axes1 = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints), sharex=True)
    if n_joints == 1:
        axes1 = [axes1]

    fig1.suptitle("Action (command) vs Observation (actual) per Joint", fontsize=14, fontweight="bold")

    for i, joint_name in enumerate(common_joints):
        ax = axes1[i]
        action_col = f"action_{joint_name}"
        obs_col = f"obs_{joint_name}"

        ax.plot(t, df[action_col].values, label="Action (指令)", linewidth=1.2, alpha=0.9)
        ax.plot(t, df[obs_col].values, label="Obs (实际)", linewidth=1.2, alpha=0.9)

        # 画误差带
        error = df[action_col].values - df[obs_col].values
        ax.fill_between(t, df[obs_col].values, df[action_col].values, alpha=0.15, color="red")

        ax.set_ylabel(joint_name, fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 标注关节5、6
        if "5" in joint_name or "6" in joint_name:
            ax.set_facecolor("#fff5f5")
            ax.set_ylabel(f"⚠ {joint_name}", fontsize=11, color="red", fontweight="bold")

    axes1[-1].set_xlabel("Time (s)", fontsize=12)
    fig1.tight_layout()

    # ============ 图2: 跟踪误差 ============
    fig2, axes2 = plt.subplots(n_joints, 1, figsize=(14, 2.5 * n_joints), sharex=True)
    if n_joints == 1:
        axes2 = [axes2]

    fig2.suptitle("Tracking Error (Action - Observation) per Joint", fontsize=14, fontweight="bold")

    for i, joint_name in enumerate(common_joints):
        ax = axes2[i]
        action_col = f"action_{joint_name}"
        obs_col = f"obs_{joint_name}"

        error = df[action_col].values - df[obs_col].values

        ax.plot(t, error, linewidth=0.8, color="red", alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.fill_between(t, error, 0, alpha=0.2, color="red")

        rmse = np.sqrt(np.mean(error ** 2))
        max_err = np.max(np.abs(error))
        ax.set_ylabel(joint_name, fontsize=11)
        ax.set_title(f"  RMSE={rmse:.4f}  |  MaxErr={max_err:.4f}", fontsize=9, loc="right")
        ax.grid(True, alpha=0.3)

        if "5" in joint_name or "6" in joint_name:
            ax.set_facecolor("#fff5f5")
            ax.set_ylabel(f"⚠ {joint_name}", fontsize=11, color="red", fontweight="bold")

    axes2[-1].set_xlabel("Time (s)", fontsize=12)
    fig2.tight_layout()

    # ============ 图3: 如果有leader列，画 leader vs action ============
    if leader_cols:
        # leader和action的映射
        leader_action_pairs = []
        for ln in leader_names:
            if ln in action_names:
                leader_action_pairs.append(ln)

        if leader_action_pairs:
            n_pairs = len(leader_action_pairs)
            fig3, axes3 = plt.subplots(n_pairs, 1, figsize=(14, 3 * n_pairs), sharex=True)
            if n_pairs == 1:
                axes3 = [axes3]

            fig3.suptitle("Leader (主臂) vs Action (处理后指令)", fontsize=14, fontweight="bold")

            for i, joint_name in enumerate(leader_action_pairs):
                ax = axes3[i]
                ax.plot(t, df[f"leader_{joint_name}"].values, label="Leader (主臂原始)", linewidth=1.2)
                ax.plot(t, df[f"action_{joint_name}"].values, label="Action (处理后)", linewidth=1.2, linestyle="--")
                ax.set_ylabel(joint_name, fontsize=11)
                ax.legend(loc="upper right", fontsize=9)
                ax.grid(True, alpha=0.3)

            axes3[-1].set_xlabel("Time (s)", fontsize=12)
            fig3.tight_layout()

    # ============ 图4: 频域分析（关节5、6） ============
    suspect_joints = [j for j in common_joints if "5" in j or "6" in j]
    if suspect_joints:
        fig4, axes4 = plt.subplots(len(suspect_joints), 2, figsize=(14, 4 * len(suspect_joints)))
        if len(suspect_joints) == 1:
            axes4 = [axes4]

        fig4.suptitle("Frequency Analysis of Suspect Joints (5, 6)", fontsize=14, fontweight="bold")

        dt = np.median(np.diff(t))
        fs = 1.0 / dt if dt > 0 else 60.0

        for i, joint_name in enumerate(suspect_joints):
            error = df[f"action_{joint_name}"].values - df[f"obs_{joint_name}"].values

            # 时域
            axes4[i][0].plot(t, error, linewidth=0.8, color="red")
            axes4[i][0].set_title(f"{joint_name} - Error (Time Domain)")
            axes4[i][0].set_xlabel("Time (s)")
            axes4[i][0].grid(True, alpha=0.3)

            # 频域
            n = len(error)
            freqs = np.fft.rfftfreq(n, d=dt)
            fft_vals = np.abs(np.fft.rfft(error - np.mean(error)))
            axes4[i][1].plot(freqs, fft_vals, linewidth=0.8, color="blue")
            axes4[i][1].set_title(f"{joint_name} - Error FFT (Freq Domain)")
            axes4[i][1].set_xlabel("Frequency (Hz)")
            axes4[i][1].set_xlim(0, fs / 2)
            axes4[i][1].grid(True, alpha=0.3)

        fig4.tight_layout()

    # 保存
    fig1.savefig("teleop_tracking.png", dpi=150, bbox_inches="tight")
    fig2.savefig("teleop_error.png", dpi=150, bbox_inches="tight")
    print("\nSaved: teleop_tracking.png, teleop_error.png")

    if leader_cols:
        fig3.savefig("teleop_leader_vs_action.png", dpi=150, bbox_inches="tight")
        print("Saved: teleop_leader_vs_action.png")

    if suspect_joints:
        fig4.savefig("teleop_freq_analysis.png", dpi=150, bbox_inches="tight")
        print("Saved: teleop_freq_analysis.png")

    plt.show()


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "teleop_data.csv"
    plot_teleop_data(csv_path)