# HIL-SERL SurRoL 训练不收敛分析与改进方案

## 问题描述

合作者的项目 [GeorgeAuburn/hilserl-surrol](https://github.com/GeorgeAuburn/hilserl-surrol) 在 Ubuntu + RTX 3060 上训练 3 小时后无明显收敛。

**环境：** MuJoCo 仿真，Franka Panda + PandaPickCubeBase-v0  
**控制：** Touch 触觉设备（7D 动作：xyz位移 + rpy姿态 + 夹爪）  
**算法：** SAC (Soft Actor-Critic)  

---

## 根因分析

经过对 `train_config_gym_hil_touch.json` 和源代码的详细审查，发现以下 **7 个关键问题**（按严重程度排序）：

### 1. Temperature 初始值过低（最严重）

**原值：** `temperature_init: 0.01`  
**问题：** SAC 的核心机制是最大熵 RL，temperature (alpha) 控制探索程度。`0.01` 意味着策略几乎不探索，从一开始就过于贪婪。

**代码层面：**
```python
# modeling_sac.py line 492-493
self.log_alpha = nn.Parameter(torch.tensor([math.log(temp_init)]))
# log(0.01) ≈ -4.6，非常低的初始熵
```

`target_entropy = -continuous_action_dim = -6`（7D动作中6个连续维度）。当 temperature 仅为 0.01 时，策略的熵远低于目标熵，temperature 优化器会试图提升它，但从 0.01 爬到合理范围（~1.0）需要大量梯度步骤。

**修复：** `temperature_init: 1.0`（激进版）或 `0.5`（保守版）

### 2. 夹爪动作范围不对称

**原值：** `action.min[6]: 0.0, action.max[6]: 2.0`  
**问题：** SAC 使用 tanh 输出 [-1, 1]，然后通过 MIN_MAX 归一化映射到 [min, max]。夹爪范围 [0, 2] 使得：
- tanh 输出 -1 → 夹爪 0（全关）
- tanh 输出 0 → 夹爪 1（半开）
- tanh 输出 +1 → 夹爪 2（全开）

但 `ee_orientation_wrapper.py` line 179 做了 `gripper_cmd = action[-1] - 1.0`（[0,2]→[-1,1]），这意味着底层环境期望 [-1, 1] 的夹爪输入。所以 dataset_stats 应该反映 [-1, 1] 而不是 [0, 2]。

**修复（激进版）：** `action.min[6]: -1.0, action.max[6]: 1.0`  
**保守版：** 保持不变（可能不影响离散夹爪控制）

### 3. 折扣因子偏低

**原值：** `discount: 0.97`  
**问题：** 对于需要多步完成的抓取任务，0.97 的有效视界约 33 步（1/(1-0.97)）。在 10 FPS 下只有 3.3 秒。如果一次成功抓取需要 5-10 秒，Q 函数无法有效传播远期奖励。

**修复：** `discount: 0.99`（有效视界 100 步 = 10 秒）

### 4. 视觉编码器冻结 + 特征维度太小

**原值：** `freeze_vision_encoder: true`, `image_encoder_hidden_dim: 32`, `latent_dim: 64`  
**问题：** 
- 冻结的 ResNet10 使用 ImageNet 预训练权重，对 MuJoCo 渲染图像可能不是最优特征
- `image_encoder_hidden_dim: 32` 将 ResNet10 的输出投影到 32 维，信息严重压缩
- `latent_dim: 64` 对 7D 动作空间 + 双摄像头输入偏小

**修复（激进版）：** `freeze_vision_encoder: false`, `image_encoder_hidden_dim: 64`, `latent_dim: 128`  
**保守版：** 保持冻结，但观察是否有改善

### 5. 梯度裁剪过于激进

**原值：** `grad_clip_norm: 10.0`  
**问题：** 虽然 10.0 看起来不小，但 SAC 中 critic 的梯度在训练初期可能很大（Q 值从零开始学习）。结合低 temperature 导致的低方差策略，梯度信号可能不足。

**修复（激进版）：** `grad_clip_norm: 1.0`（更紧的裁剪，防止梯度爆炸但允许快速学习）  
**保守版：** 保持 10.0

### 6. 策略更新过于频繁

**原值：** `policy_update_freq: 1`, `utd_ratio: 2`  
**问题：** 每步都更新策略（频率 1），且 UTD=2 意味着每收集 1 个样本就做 2 次梯度更新。在训练初期，当 critic 还不准确时，频繁更新 actor 会导致 actor 拟合一个错误的 Q 函数。

**修复（激进版）：** `policy_update_freq: 2`, `utd_ratio: 1`  
**保守版：** `utd_ratio: 2` 保持，但提高 `online_step_before_learning: 300`

### 7. Episode 时长偏短

**原值：** `control_time_s: 10.0`  
**问题：** 10 秒 × 10 FPS = 100 步/episode。对于需要接近 + 下降 + 抓取 + 提起的完整序列，可能不够。特别是在探索初期，策略需要更多时间偶然完成任务。

**修复（激进版）：** `control_time_s: 15.0`  
**保守版：** 保持 10.0

---

## 两个改进配置

### 配置 A：激进改进版 (`train_config_improved.json`)

核心改动：最大化收敛速度，接受可能的不稳定性。

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `temperature_init` | 0.01 | **1.0** | 恢复充分探索 |
| `discount` | 0.97 | **0.99** | 扩大奖励传播视界 |
| `freeze_vision_encoder` | true | **false** | 允许适应 MuJoCo 图像 |
| `image_encoder_hidden_dim` | 32 | **64** | 增加视觉特征容量 |
| `latent_dim` | 64 | **128** | 增加表征容量 |
| `actor_lr` | 0.0003 | **0.0001** | Actor 学慢一点，等 Critic 先学好 |
| `utd_ratio` | 2 | **1** | 减少过拟合风险 |
| `policy_update_freq` | 1 | **2** | 减少策略更新频率 |
| `grad_clip_norm` | 10.0 | **1.0** | 更紧裁剪 |
| `online_step_before_learning` | 100 | **500** | 先积累更多数据 |
| `control_time_s` | 10.0 | **15.0** | 更长 episode |
| `gripper_penalty` | -0.02 | **-0.01** | 减少夹爪惩罚 |
| `policy_parameters_push_frequency` | 4 | **2** | 更快同步策略 |
| `action.min[6]/max[6]` | [0, 2] | **[-1, 1]** | 修正夹爪归一化 |
| `eval_freq` / `save_freq` | 20000 | **10000** | 更频繁评估 |
| `wandb.enable` | false | **true** | 开启训练监控 |

### 配置 B：保守改进版 (`train_config_conservative.json`)

核心改动：最小化变更，只修最关键的问题。

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `temperature_init` | 0.01 | **0.5** | 适度增加探索 |
| `discount` | 0.97 | **0.99** | 扩大奖励传播视界 |
| `temperature_lr` | 0.0003 | **0.001** | 加速 temperature 自适应 |
| `online_step_before_learning` | 100 | **300** | 稍多预热 |
| `policy_parameters_push_frequency` | 4 | **2** | 更快同步 |
| `eval_freq` / `save_freq` | 20000 | **10000** | 更频繁评估 |
| `wandb.enable` | false | **true** | 开启训练监控 |

其余参数与原配置相同。

---

## 推荐使用顺序

1. **先试保守版** `configs/surrol/train_config_conservative.json`
   - 风险低，只改了 temperature 和 discount
   - 如果 3 小时内 reward 开始上升 → 说明主要问题是 temperature
   - 如果仍不收敛 → 切换到激进版

2. **再试激进版** `configs/surrol/train_config_improved.json`
   - 修改幅度大，收敛速度可能更快
   - 但解冻视觉编码器会增加显存占用和训练时间
   - 如果 CUDA OOM → 将 `batch_size` 从 256 降到 128

---

## 使用方法

```bash
cd lerobot

# 保守版
python -m lerobot.rl.learner --config_path configs/surrol/train_config_conservative.json
python -m lerobot.rl.actor --config_path configs/surrol/train_config_conservative.json

# 激进版
python -m lerobot.rl.learner --config_path configs/surrol/train_config_improved.json
python -m lerobot.rl.actor --config_path configs/surrol/train_config_improved.json
```

**注意：**
1. `haptic_module_path` 需要改为你的实际路径
2. 确保 `local/franka_sim_touch_demos` 数据集在 `~/.cache/huggingface/lerobot/` 下
3. 如果使用 W&B，先运行 `wandb login`

---

## 训练收敛的判断标准

| 指标 | 不收敛 | 开始收敛 | 良好收敛 |
|------|--------|---------|---------|
| Episodic reward | 持平或波动 | 缓慢上升趋势 | 稳定上升 |
| Critic loss | 持续增大或震荡 | 先升后降 | 逐渐下降 |
| Temperature | 固定不变 | 自适应调整 | 稳定在合理范围 |
| Actor loss | 无规律 | 出现下降趋势 | 稳定下降 |
| Success rate | 0% | >0% | >30% |

---

## 其他观察

### EEOrientationActionWrapper 设计合理
- 使用绝对姿态偏移而非累积，避免漂移
- ActionSafetyWrapper 提供 EMA 平滑和速率限制
- 这部分代码质量好，不需要修改

### 可能的进一步改进（如果上述修改仍不够）
1. **增加网络宽度：** `hidden_dims: [256, 256]` → `[512, 512]`
2. **增加 critic 数量：** `num_critics: 2` → `10`（使用 REDQ 风格的 critic ensemble）
3. **奖励塑形：** 在 gym_hil 环境中添加距离奖励（而非纯稀疏奖励）
4. **数据增强：** 对摄像头图像做随机裁剪/颜色抖动
5. **课程学习：** 从更简单的 reach 任务开始，逐步过渡到 pick
