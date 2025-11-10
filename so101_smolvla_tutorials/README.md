# SO101 机械臂 + SmolVLA 完整教程

> **作者**: q442333521
> **环境**: Ubuntu 22.04
> **更新日期**: 2025-11-10

---

## 📋 目录

1. [项目概述](#项目概述)
2. [环境准备](#环境准备)
3. [最近代码更新](#最近代码更新)
4. [预训练模型](#预训练模型)
5. [文件夹结构](#文件夹结构)
6. [快速开始](#快速开始)
7. [常见问题](#常见问题)
8. [参考资源](#参考资源)

---

## 🎯 项目概述

本教程提供 **SO101 机械臂** 与 **SmolVLA 视觉-语言-动作模型** 的完整工作流程，包括：

- ✅ **MuJoCo 仿真环境**：在虚拟环境中快速测试和采集数据
- ✅ **实体机械臂控制**：SO101 主从臂设置、标定、数据采集
- ✅ **SmolVLA 微调**：使用自己的数据集训练模型
- ✅ **模型部署**：在实体机械臂上运行训练好的策略

---

## 🔧 环境准备

### 1. 系统要求

- **操作系统**: Ubuntu 22.04 LTS
- **Python**: 3.10+
- **CUDA**: 11.8+ (推荐用于训练)
- **硬件**:
  - GPU: NVIDIA RTX 3090 / A100 (训练)
  - RAM: 16GB+
  - SO101 主从臂各一套
  - USB 摄像头 1-2 个

### 2. 安装 LeRobot

```bash
# 克隆仓库
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# 安装基础依赖
pip install -e .

# 安装 SmolVLA 依赖
pip install -e ".[smolvla]"

# 安装 Feetech 电机驱动 (SO101 必需)
pip install -e ".[feetech]"
```

### 3. 验证安装

```bash
# 检查 LeRobot 版本
python -c "import lerobot; print(lerobot.__version__)"

# 检查 CUDA 可用性
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📰 最近代码更新

LeRobot 最近的重要更新 (最近 2 个月):

| 日期 | 更新内容 | PR |
|------|----------|-----|
| 2 天前 | 修复数据集访问瓶颈，提升训练速度 | #2408 |
| 5 天前 | 修复数据集工具的关键 bug | #2342 |
| 3 周前 | 修复 SmolVLA 多 GPU 训练设备错误 | #2270 |
| 3 周前 | 新增 NVIDIA GR00T N1.5 模型 | #2292 |
| 2 周前 | 发布 LeRobot 0.4.1 版本 | #2299 |

**重要提示**:
- ✅ SmolVLA 多 GPU 训练问题已修复
- ✅ 数据集访问性能大幅提升
- ⚠️ 建议使用最新版本 (0.4.1+)

---

## 🤖 预训练模型

### 可用的 SmolVLA 模型

| 模型名称 | 参数量 | 用途 | Hugging Face 链接 |
|----------|--------|------|-------------------|
| `lerobot/smolvla_base` | 450M | 基础模型（需要微调） | [🔗 链接](https://huggingface.co/lerobot/smolvla_base) |
| `lerobot/svla_so101_pickplace` | - | SO101 抓放数据集 | [🔗 链接](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) |

### 重要说明

⚠️ **SmolVLA 需要在您自己的数据上微调才能获得最佳性能！**

- ❌ `smolvla_base` **不是**针对 SO101 预训练的即用模型
- ✅ 它是一个**基础模型**，需要用您的数据集微调
- ✅ 推荐采集 **50+ 个演示**作为起点
- ✅ 参考数据集: `lerobot/svla_so101_pickplace` (50 episodes)

### 下载预训练模型

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 下载基础模型
model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

---

## 📁 文件夹结构

```
so101_smolvla_tutorials/
├── README.md                    # 本文件
├── mujoco_sim/                  # MuJoCo 仿真相关
│   └── mujoco_smolvla_tutorial.ipynb
├── real_robot/                  # 实体机械臂相关
│   ├── so101_setup_and_calibration.ipynb
│   ├── so101_data_collection.ipynb
│   └── so101_smolvla_training.ipynb
├── datasets/                    # 数据集存储
│   ├── mujoco_demos/
│   └── real_robot_demos/
├── models/                      # 训练好的模型
│   └── checkpoints/
└── docs/                        # 额外文档
    ├── FAQ.md
    └── troubleshooting.md
```

---

## 🚀 快速开始

### 场景 1: 我有 SO101 实体机械臂

1. **硬件设置和标定** → 使用 `real_robot/so101_setup_and_calibration.ipynb`
2. **数据采集** → 使用 `real_robot/so101_data_collection.ipynb`
3. **训练 SmolVLA** → 使用 `real_robot/so101_smolvla_training.ipynb`

### 场景 2: 我想先在 MuJoCo 中测试

1. **MuJoCo 仿真** → 使用 `mujoco_sim/mujoco_smolvla_tutorial.ipynb`
2. **迁移到实体** → 需要重新采集真实数据 (sim2real gap)

---

## ❓ 常见问题

### Q1: 我的环境（桌面、背景、摄像头）和别人不一样，需要重新采集数据吗？

**答**: ✅ **强烈建议重新采集！**

SmolVLA 是视觉语言模型，对以下因素非常敏感：
- 桌面颜色和材质
- 背景环境
- 光照条件
- 摄像头分辨率和画质

如果环境差异较大，使用别人的数据训练的模型在您的环境中性能会显著下降。

### Q2: MuJoCo 中采集的数据能用于实体机械臂吗？

**答**: ⚠️ **理论上可以，但实际效果通常不理想**

原因：
- **Sim2Real Gap**: 仿真和现实存在差距
- **物理特性差异**: 摩擦力、惯性、延迟等
- **视觉差异**: 仿真渲染 vs 真实摄像头

**推荐做法**:
1. 在 MuJoCo 中快速验证算法和流程
2. 在实体机械臂上重新采集真实数据
3. 可选：使用域随机化 (Domain Randomization) 技术减小差距

### Q3: 有没有别人训练好的 SmolVLA SO101 模型可以直接下载使用？

**答**: ⚠️ **目前没有通用的即用模型**

- `lerobot/smolvla_base`: 基础模型，需要微调
- `lerobot/svla_so101_pickplace`: 数据集，不是模型

即使有别人训练好的模型，由于 Q1 提到的环境差异，也需要在您的环境中重新微调或采集数据。

### Q4: 采集多少数据合适？

**答**:
- **最少**: 25 episodes (可能不够，性能较差)
- **推荐**: 50+ episodes
- **理想**: 100+ episodes，包含多种变化（物体位置、姿态等）

参考 LeRobot 官方数据集 `svla_so101_pickplace`:
- 50 episodes
- 5 个不同的立方体位置
- 每个位置 10 个演示

### Q5: 训练需要多长时间？

**答**:
- **硬件**: 单张 A100 GPU
- **时间**: ~4 小时 (20,000 steps)
- **Batch Size**: 64

根据您的 GPU 和数据集大小调整。

### Q6: 我没有 GPU 怎么办？

**答**:
1. 使用 Google Colab (提供免费 GPU): [🔗 SmolVLA Training Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)
2. 租用云 GPU (AWS, Lambda Labs, etc.)
3. 使用预训练模型进行推理 (CPU 也可以，但较慢)

---

## 📚 参考资源

### 官方文档

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [SO101 组装指南](https://huggingface.co/docs/lerobot/en/so101)
- [SmolVLA 文档](https://huggingface.co/docs/lerobot/en/smolvla)
- [实体机械臂教程](https://huggingface.co/docs/lerobot/en/il_robots)

### 社区资源

- [LeRobot Discord](https://discord.com/invite/s3KuuzsPFb)
- [Hugging Face Forum](https://discuss.huggingface.co/)
- [参考项目: lerobot-mujoco](https://github.com/q442333521/lerobot-mujoco)

### 论文

- [SmolVLA Paper](https://huggingface.co/papers/2506.01844)
- [SmolVLA Blog](https://huggingface.co/blog/smolvla)

---

## 🛠️ 故障排除

### 找不到 USB 端口

```bash
# Linux 需要给予端口权限
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# 查找端口
lerobot-find-port
```

### 电机连接失败

1. 检查电源是否接通
2. 检查 USB 连接
3. 检查 3-pin 电机线缆
4. Waveshare 板需要将跳线设置到 B 通道

### 训练时 CUDA Out of Memory

```bash
# 减小 batch size
lerobot-train --batch_size=32  # 从 64 降到 32

# 使用梯度累积
lerobot-train --gradient_accumulation_steps=2
```

---

## 📞 获取帮助

如有问题，请：

1. 查看本 README 的 [常见问题](#常见问题)
2. 查看 `docs/troubleshooting.md`
3. 在 [LeRobot Discord](https://discord.com/invite/s3KuuzsPFb) 提问
4. 在 [GitHub Issues](https://github.com/huggingface/lerobot/issues) 报告 bug

---

## 📄 许可证

本教程遵循 Apache 2.0 许可证。

---

**祝您使用愉快！🎉**
