#!/usr/bin/env python
"""从 rebot LeRobot 数据集的 meta/stats.json 生成 lingbot_vla_v2 的 norm_stats JSON。

对应槽位切片:
  observation.state.arm.position   <- observation.state[0:6]   (6 个臂关节, 度)
  observation.state.effector.position <- observation.state[6:7] (夹爪)
  action.* 同理取 action 向量的对应切片。

两种归一化口径(对应官方 Training_Config.md 真机/仿真两套 recipe):
  - 真机(real):   canonical_norm_type = meanstd,只需要 mean/std(默认)
  - 仿真(robotwin): canonical_norm_type = bounds_99_woclip,需要 q01/q99 分位数
加 --quantiles 即同时输出 q01/q99,一份 JSON 两套 profile 通用(转换器 --profile
robotwin 时 canonical_norm_type 才会真正去读 q01/q99)。

lerobot 数据集的 meta/stats.json 自带真实 q01/q99(RunningQuantileStats 计算),
所以直接切片转存即可,无需重算原始数据。

用法:
  # 真机(meanstd,默认)
  python gen_rebot_norm_stats.py --dataset-root /path/to/rebot_dataset \
      --out rebot_norm_stats.json
  # 仿真(robotwin,带 q01/q99)
  python gen_rebot_norm_stats.py --dataset-root /path/to/rebot_dataset \
      --quantiles --out rebot_norm_stats.robotwin.json
"""

import argparse
import json
from pathlib import Path

SLOT_SLICES = {
    "arm.position": (0, 6),       # shoulder_pan .. wrist_roll
    "effector.position": (6, 7),  # gripper
}

# robotwin profile 的 bounds_99_woclip 读这两个分位键
QUANTILE_KEYS = ("q01", "q99")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="rebot lerobot 数据集根目录(含 meta/stats.json)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "rebot_norm_stats.json"))
    ap.add_argument(
        "--quantiles",
        action="store_true",
        help="同时输出 q01/q99 分位数(供 robotwin profile 的 bounds_99_woclip 用)",
    )
    args = ap.parse_args()

    stats_path = Path(args.dataset_root) / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    out = {"norm_stats": {}}
    for raw_key in ("observation.state", "action"):
        if raw_key not in stats:
            raise KeyError(f"{raw_key} 不在 {stats_path} 里,keys={list(stats.keys())}")
        feat = stats[raw_key]
        mean, std = feat["mean"], feat["std"]
        if len(mean) != 7:
            raise ValueError(f"{raw_key} 应为 7 维,实际 {len(mean)} 维")
        if args.quantiles:
            missing_q = [q for q in QUANTILE_KEYS if q not in feat]
            if missing_q:
                raise KeyError(
                    f"{raw_key} 缺分位数 {missing_q}(数据集较旧?lerobot 新版 stats 才带 q01/q99)。"
                    f"该数据集的统计键: {sorted(feat.keys())}"
                )
        for slot, (s, e) in SLOT_SLICES.items():
            # 防御:接近恒定的维度(如基本不动的夹爪)std 过小会放大噪声
            slot_std = [max(float(v), 1e-3) for v in std[s:e]]
            entry = {
                "mean": [float(v) for v in mean[s:e]],
                "std": slot_std,
            }
            if args.quantiles:
                for q in QUANTILE_KEYS:
                    entry[q] = [float(v) for v in feat[q][s:e]]
            out["norm_stats"][f"{raw_key}.{slot}"] = entry

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"written: {args.out}  (quantiles={'on' if args.quantiles else 'off'})")
    for k, v in out["norm_stats"].items():
        print(f"  {k}: dim={len(v['mean'])} keys={sorted(v.keys())}")


if __name__ == "__main__":
    main()
