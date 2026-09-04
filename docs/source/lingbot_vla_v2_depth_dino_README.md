# LingBot-VLA 2.0 native-depth / DINO-video 蒸馏分支

> **状态（2026-09-01）：全部三个教师（MoGe + MoRGBD + DINO-video）均为 LeRobot 第一方
> 权重兼容实现并完成与上游的数值对照。开发者只需要下载权重文件——不 clone 上游、
> 不设环境变量、不装第三方教师依赖。**

本仓不内置任何上游教师源码，运行时不 import、不 `sys.path` 注入任何第三方仓库。
依赖契约：**LeRobot 第一方实现 + 开发者下载的权重文件**。

## 已完成的接线

| 部分 | 实现 | 状态 |
|---|---|---|
| 对齐头 | `model_core/depth_heads.py`，官方 checkpoint 命名兼容；与上游前向 `atol=0` | 已完成 |
| 配置 | `align_params`、future 采样和 schema 校验 | 已完成 |
| 数据 | current/future 图像路由、状态取当前帧、fps 合成和 padding 防护 | 已完成 |
| 训练钩子 | 训练期 frozen/no-grad target；教师不进 optimizer、DDP 或 checkpoint | 已完成 |
| 转换器 | `--include-depth-heads` 严格加载官方蒸馏头 | 已完成 |
| MoGe 教师 runtime | `teachers/native_depth_models.py`（第一方，纯 torch） | **已完成，已对照** |
| MoRGBD 教师 runtime | `teachers/morgbd_teacher.py`（第一方，纯 torch） | **已完成，已对照** |
| DINO-video 教师 runtime | `teachers/dino_video/`（第一方子包，纯 torch） | **已完成，已对照** |

对照结论（真权重、GPU、与上游 runtime A/B）：

- MoGe：`infer(num_tokens=256)` 的 depth 与上游**逐位一致**（真图 0.0 / 随机图 ≤2e-7 相对误差）；
  337/337 张量严格加载，零缺失。
- MoRGBD：`infer_feat` 的 feat 与 cls 与上游**逐位一致（0.0 差异）**；包括复刻上游
  strict=False 加载怪癖（`depth_mask_patch_embed.*` 被丢弃、`depth_patch_embed` 留在
  构造初始化、`normal_head.*` 因上游从不构造而被丢弃），且我们的初始化带固定种子，
  比 upstream 更可复现。
- DINO-video：`teachers/dino_video/` 以 SDPA 注意力复刻视频 DINOv3-L（3D RoPE、
  storage tokens、frame-block-causal 掩码）。对上游 SDPA 参考**逐位一致（0.0 差异，
  bf16 100% 相同）**，覆盖 T=2/3、warmup `current_index=1`、`fps=None/1.0/30/49`
  （含 per-sample 张量 fps）、`cls_pool=mean/last`、B=1/2。checkpoint 345 个
  backbone 张量严格加载，仅 `dino_head/ibot_head`（14 个，蒸馏训练头，推理不用）
  显式列为未使用；重复调用逐位确定。注意：上游官方配方写的
  `attention_mode=flex_block_causal` 在本机（GB10/torch 2.11）的 flex eager 掩码路径
  数值有损（与上游自己的 SDPA 路径也差 1e-2 量级）；本 runtime 默认走 SDPA 真值，
  `attention_backend="flex"` 只有在通过随机化 SDPA 对照门禁后才可用。
- 三个教师均不需要 `utils3d` / `scipy` / `cv2` / `omegaconf`：全部纯 torch 重新推导
  （config.yaml 由内置极小 YAML 子集解析器读取）。

## 最短路径：完整官方配方（depth + future-depth + DINO-video）

```bash
# 0) 安装（无额外第三方教师依赖）
pip install -e ".[training,lingbot_vla2,lingbot_vla2_depth]"

# 1) 只下权重（HF cache；转换器 --include-depth-heads 会自动解析全部四个路径）
hf download Ruicheng/moge-2-vitb-normal model.pt
hf download robbyant/lingbot-vla-v2-6b --include "depth/*" "dino_video/*"

# 2) 转换官方 Native Depth 6B checkpoint，保留全部蒸馏头并写入完整 align_params
python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-vla-v2-6b-full-lerobot \
  --robot-config-path <robot_config.yaml> \
  --norm-stats-path <norm_stats.json> \
  --include-depth-heads

# 3) 训练（无需任何环境变量 / clone）
lerobot-train \
  --dataset.repo_id=<repo_id> --dataset.root=<root> \
  --policy.path=./lingbot-vla-v2-6b-full-lerobot \
  --policy.dataset_fps=<fps> \
  --batch_size=<batch> --steps=<steps> \
  --output_dir=outputs/train/lingbot_vla_v2_full
```

验收：日志 / wandb 出现 `depth_loss`、`future_depth_loss` 与 `future_video_loss`
三者。缺任一值先停，按排障表查。

DINO 开启时 `dataset_fps` 必填；LeRobot 按
`effective_fps = dataset_fps / max(1, future_frame_offset)` 折算传给教师
（例：30 FPS 数据集、偏移 49 → 教师 fps ≈ 0.612）；缺省时回落教师在
`align_params.video.effective_fps`（官方配方 1.0）。

### 只跑 depth（不启用 DINO-video）

```bash
hf download robbyant/lingbot-vla-v2-6b --include "depth/*"

# depth-only 配方（关掉 video）
python - <<'EOF'
import json
params = json.load(open("src/lerobot/policies/lingbot_vla_v2/scripts/align_params_robotwin.json"))
params.pop("_comment"); params["use_future_video"] = False; params.pop("video")
json.dump(params, open("./align_params_depth_only.json", "w"), indent=2)
EOF

python -m lerobot.policies.lingbot_vla_v2.scripts.convert_upstream_checkpoint \
  --input robbyant/lingbot-vla-v2-6b \
  --output ./lingbot-vla-v2-6b-depth-lerobot \
  --robot-config-path <robot_config.yaml> \
  --norm-stats-path <norm_stats.json> \
  --align-params=@./align_params_depth_only.json

lerobot-train ... --policy.path=./lingbot-vla-v2-6b-depth-lerobot --policy.dataset_fps=<fps>
```

### 自定义 DINO 配方时的关键字段（官方 RoboTwin 值）

```json
{
  "use_future_video": true,
  "video": {
    "ckpt_path": "<HF cache>/dino_video/teacher_step_10000.pth",
    "config_path": "<HF cache>/dino_video/config.yaml",
    "attention_mode": "flex_block_causal",
    "input_size": 256,
    "num_future_frames": 1,
    "use_warmup_frame": true,
    "use_patch_loss": true,
    "use_current_patch_loss": true,
    "use_cls_loss": false,
    "cls_pool": "last",
    "effective_fps": 1.0
  }
}
```

`attention_mode` 是兼容名：第一方 runtime 把 frame-block-causal 语义统一实现为
SDPA 掩码（与上游 SDPA 路径逐位一致）；可选 `"attention_backend": "flex"` 走
flex attention 加速，但必须先通过内置的 SDPA 随机对照门禁，否则报错并回落说明。
任何仓库/checkout 类键（如 `upstream_root`）会被显式拒绝。

## 教师依赖边界

历史上的三种第三方形态均已移除并防回归：①裁剪 vendor 源码树；②
`LINGBOT_VLA_V2_UPSTREAM` + `git clone` 外部 provider；③对上游 runtime 的任何
运行时 import。`teachers/` 只容纳 LeRobot 维护的代码（`test_align_teachers.py`
与 `test_dino_video_scaffold.py` 钉死这些守卫）。

| 权重 | Hub 文件 | 大小 | 状态 |
|---|---|---|---|
| MoGe v2 | `Ruicheng/moge-2-vitb-normal/model.pt` | 419 MB | 可用 |
| LingBot-Depth / MoRGBD | `robbyant/lingbot-vla-v2-6b/depth/model.pt` | 1.32 GB | 可用 |
| DINO-video 教师 | `robbyant/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth` + `config.yaml` | 1.40 GB | 可用 |

`config.yaml` 缺省时按 `ckpt_path` 同目录解析；它只提供结构/超参（ViT-L/16、
storage tokens、RoPE 参数），权重与结构校验以 `teacher_step_10000.pth` 为准。

## 已固定的上游对齐契约

- 目标张量：depth / future-depth / video patch 为 `(B, 256, 1024)` bf16；
  video CLS 为 `(B, 1024)`；
- depth 仅使用第一相机；`pil_images` 为增强后的 `[0,255]` 图像；
- future delta 索引为 `[0, max(1, future_frame_offset)]`；
  `future_video_effective_fps = dataset_fps / max(1, future_frame_offset)`；
- 教师是普通 dataclass 属性，懒构建于策略真实设备，且仅在训练期运行；
- `--include-depth-heads` 必须严格覆盖官方 checkpoint 中全部蒸馏头参数。

## 排障

| 报错 | 含义 / 处置 |
|---|---|
| `DINO video teacher requires video.ckpt_path` / `teacher checkpoint not found` | 权重未下载或路径未填；按上文 `hf download`。绝不会提示 clone 上游。 |
| `repository/checkout keys ... are rejected` | 配方里带了 `upstream_root` 等仓库键；第一方 runtime 不接受。 |
| `unknown align_params.video keys [...]` | 配方多了未知键；对照官方 JSON。 |
| `attention_backend='flex' requires ...` / flex 门禁失败 | flex 在该环境不可用或数值不过门禁；保持默认 SDPA。 |
| `Checkpoint does not match the first-party MoRGBD teacher ...` | 权重文件不是官方发布版。 |
| `batch carries no 'pil_images'` | preprocessor 未开 depth 分支；用带 `align_params` 的 config 重建。 |
| wandb 无 `depth_loss` / `future_video_loss` | `align_params` 为空或对应子开关没开。 |
| 教师吃满显存 | 教师每 rank 一份（ViT-B + ViT-L + DINO-L 合计约 5 GB）；降 `batch_size`。 |

## 验证

```bash
pytest tests/policies/lingbot_vla_v2/test_depth_heads.py
pytest tests/policies/lingbot_vla_v2/test_depth_alignment_support.py
pytest tests/policies/lingbot_vla_v2/test_align_data_pipeline.py
pytest tests/policies/lingbot_vla_v2/test_align_teachers.py      # 含真实权重端到端(有权重时)
pytest tests/policies/lingbot_vla_v2/test_dino_video_scaffold.py # DINO 边界契约
```

`test_align_teachers.py` 与 scaffold 测试同时钉死无第三方仓库回退：失败路径不改变
`sys.path`，代码中不存在 `LINGBOT_VLA_V2_UPSTREAM` resolver，仓库键被显式拒绝。

## 边界

- 教师只在训练期出现；导出的 ckpt / 推理路径零教师依赖；
- DINO runtime 是 inference-only：`dino_head`/`ibot_head`（教师自蒸馏训练头）
  不进入第一方实现；
- `enable_expert_vision`（独立未完成特性）与本分支无关，仍被配置守卫拒绝；
- 消融实验不在本分支范围（开关齐全，开发者自行探索）。
