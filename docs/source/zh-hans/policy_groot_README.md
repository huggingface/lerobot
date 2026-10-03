## 研究论文

GR00T N1 技术报告（涵盖 GR00T N1.x 系列，包括 N1.7）：https://arxiv.org/abs/2503.14734

GR00T N1.7 模型卡片：https://huggingface.co/nvidia/GR00T-N1.7-3B

GR00T N1.5 研究页面（早期版本）：https://research.nvidia.com/labs/gear/gr00t-n1_5/

> GR00T N1.5 支持已从 LeRobot 中移除；最后支持它的版本是 `lerobot==0.5.1`。
> 当前版本仅支持 GR00T N1.7。

## 代码仓库

代码：https://github.com/NVIDIA/Isaac-GR00T

## 引用

```bibtex
@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint     = {2503.14734},
  title      = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author     = {NVIDIA and Johan Bjorck and Fernando Castañeda, Nikita Cherniadev and Xingye Da and Runyu Ding and Linxi "Jim" Fan and Yu Fang and Dieter Fox and Fengyuan Hu and Spencer Huang and Joel Jang and Zhenyu Jiang and Jan Kautz and Kaushil Kundalia and Lawrence Lao and Zhiqi Li and Zongyu Lin and Kevin Lin and Guilin Liu and Edith Llontop and Loic Magne and Ajay Mandlekar and Avnish Narayan and Soroush Nasiriany and Scott Reed and You Liang Tan and Guanzhi Wang and Zu Wang and Jing Wang and Qi Wang and Jiannan Xiang and Yuqi Xie and Yinzhen Xu and Zhenjia Xu and Seonghyeon Ye and Zhiding Yu and Ao Zhang and Hao Zhang and Yizhou Zhao and Ruijie Zheng and Yuke Zhu},
  month      = {March},
  year       = {2025},
  booktitle  = {ArXiv Preprint},
}
```

## 额外资源

博客：https://developer.nvidia.com/isaac/gr00t

Hugging Face 模型：

- GR00T N1.7：https://huggingface.co/nvidia/GR00T-N1.7-3B
- GR00T N1.7 LIBERO 检查点：https://huggingface.co/nvidia/GR00T-N1.7-LIBERO

<details>
<summary><b>原始版本与 LeRobot 实现一致性测试</b></summary>

## 原始版本与 LeRobot 实现一致性测试

`tests/policies/groot/test_groot_vs_original.py` 验证了 LeRobot 对 GR00T N1.7（Qwen3-VL 骨干 + 流匹配动作头）的重现实现与 NVIDIA 原始 `gr00t` 包的一致性。该测试包含两个对比测试，每个测试都针对检查点中的每个 embodiment tag 进行参数化：

1. **模型一致性** — 给定字节级相同的预处理输入和相同的流匹配种子（记录在每个 artifact 中），两个实现必须产生**相同的原始模型输出**（`get_action(...)["action_pred"]`，即归一化的流匹配预测）。输出形状必须完全匹配；任何动作时间范围或动作维度的不匹配都会导致测试失败。
2. **预处理器一致性** — 给定相同的原始观测值（每个摄像头的帧、状态向量、语言指令），LeRobot 自己的预处理器流水线（真实的 Qwen3-VL 聊天模板/分词器/图像打包 + 检查点驱动的状态归一化，无模拟）必须产生与原始包处理器**相同的整理模型输入**（`input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw`、`state`、`embodiment_id`）。

### 为什么需要两个环境

原始 `gr00t` 包固定使用 `transformers==4.57.3`（Python 3.10）；而此集成需要 `transformers>=5.x`（Qwen3-VL）。在 5.x 版本下，`PretrainedConfig` 本身是一个带默认值的数据类，因此原始配置数据类无法导入（`非默认参数跟随默认参数`）。因此两个实现**无法在同一 Python 进程中导入**。

所以测试使用**生产者/消费者**模式跨两个虚拟环境：

1. **生产者** — `tests/policies/groot/utils/dump_original_n1_7.py`，在_原始_ gr00t 虚拟环境中运行。对于每个 embodiment，它根据检查点元数据通用地构建虚拟输入（从 `statistics.json` 获取状态维度；从处理器模态配置获取摄像头/语言键），运行原始模型，并为每个 tag 保存一个 `.npz` 文件：原始观测值（`raw::` 键）、整理后的输入（`in::` 键）、种子和原始 `action_pred`。
2. **消费者** — 上述 pytest，在_LeRobot_ 虚拟环境中运行。它发现每个 `.npz` 文件；模型一致性测试将字节级相同的整理输入与记录的种子一起回放通过 LeRobot 模型，并断言输出匹配；预处理器一致性测试将原始观测值通过 LeRobot 的完整预处理器流水线回放，并断言整理后的张量匹配。

> 由旧版本 dump 脚本生成的 artifacts 不包含 `raw::` 字段；预处理器一致性测试会**跳过**并给出生成提示。重新运行生产者来刷新它们。

### 公平性控制

- **相同的预处理输入（模型一致性）** — 原始处理器的 `input_ids`、`pixel_values`、`image_grid_thw`、`attention_mask`、`state`、`embodiment_id` 被原样输入到 LeRobot 模型（无重新分词/重新归一化），因此模型比较隔离了模型本身。LeRobot 自己的分词/图像打包由预处理器一致性测试单独覆盖，该测试将相同原始观测值的输出与那些整理后的张量进行比较。
- **相同的精度 + 注意力内核** — 双方都运行**fp32 + SDPA**。原始版本默认使用 `use_flash_attention=True`（flash_attention_2 + bf16）；生产者强制使用 SDPA + fp32。（使用默认值时差距约为 3e-2 — 纯粹的内核/舍入噪声，而非实现差异。）
- **相同的流匹配种子** — 在双方采样前固定；生产者在每个 artifact 中记录它（`--seed`，默认 42），消费者回放记录的值。

### 如何运行

```bash
# 解析本地检查点（GR00T-N1.7-LIBERO / libero_10）
CKPT=$(python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(os.path.join(snapshot_download("nvidia/GR00T-N1.7-LIBERO",
      allow_patterns=["libero_10/*"]), "libero_10"))
PY
)

# 1) 为所有 embodiments 生成原始端的 artifacts（原始 gr00t 虚拟环境，CUDA）
CUDA_VISIBLE_DEVICES=0 /path/to/Isaac-GR00T/.venv-original/bin/python \
    tests/policies/groot/utils/dump_original_n1_7.py \
    --ckpt "$CKPT" --out-dir tests/policies/groot/artifacts --device cuda --seed 42

# 2) 运行一致性测试（LeRobot 虚拟环境）— 每个 embodiment 一个参数化测试用例
CUDA_VISIBLE_DEVICES=0 GROOT_PARITY_DEVICE=cuda \
    uv run pytest tests/policies/groot/test_groot_vs_original.py -v -s
```

`.npz` artifacts 是本地文件（git 忽略，每个约 6-10 MB），由生产者生成；它们永远不会被提交。测试在 CI 上或当检查点/artifacts 缺失时**跳过**（不失败）。

#### 环境变量（全部可选）

| 变量                                      | 默认值                           | 用途                                  |
| ----------------------------------------- | -------------------------------- | ------------------------------------- |
| `GROOT_N1_7_PARITY_DIR`                   | `tests/policies/groot/artifacts` | 每个 tag 的 `.npz` artifacts 目录     |
| `GROOT_N1_7_LIBERO_CKPT`                  | auto (HF cache)                  | 覆盖检查点目录                        |
| `GROOT_PARITY_DEVICE`                     | `cuda` if available              | `cpu` 或 `cuda`                       |
| `GROOT_PARITY_ATOL` / `GROOT_PARITY_RTOL` | `1e-3`                           | 比较容差                              |

</details>
