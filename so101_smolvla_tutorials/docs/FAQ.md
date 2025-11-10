# 常见问题解答 (FAQ)

## 📋 目录

1. [硬件相关](#硬件相关)
2. [软件和安装](#软件和安装)
3. [数据采集](#数据采集)
4. [模型训练](#模型训练)
5. [模型部署](#模型部署)
6. [Sim2Real](#sim2real)
7. [故障排除](#故障排除)

---

## 硬件相关

### Q1: 我需要购买哪些硬件？

**最小配置**:
- SO101 从臂（Follower）x 1
- SO101 主臂（Leader）x 1
- USB 线缆 x 2
- 电源适配器 x 2
- USB 摄像头 x 1（最低）

**推荐配置**:
- USB 摄像头 x 2（前置 + 腕部）
- 好的光照设备
- 稳定的工作台

**成本估算**:
- SO101 套装: ~$1000-1500
- 摄像头: $20-50 每个
- 总计: ~$1100-1600

### Q2: SO101 和 SO100 有什么区别？

| 特性 | SO100 | SO101 |
|------|-------|-------|
| 关节数 | 6 | 6 |
| 主臂齿轮比 | 部分不同 | 优化过的齿轮比 |
| 文档支持 | 较少 | 更完善 |
| 推荐用途 | 早期用户 | 新用户 ✅ |

建议购买 **SO101**，文档更全面。

### Q3: 需要什么样的摄像头？

**要求**:
- USB 接口（UVC 兼容）
- 最低分辨率: 640x480
- FPS: 30
- 支持 Linux (V4L2)

**推荐**:
- Logitech C920 / C922
- Microsoft LifeCam
- 任何支持 OpenCV 的 USB 摄像头

**测试方法**:
```bash
# 查看可用摄像头
v4l2-ctl --list-devices

# 测试摄像头
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

### Q4: 需要什么样的电脑配置？

**训练** (必需 GPU):
- GPU: NVIDIA RTX 3090 / A100
- RAM: 16GB+
- 存储: 100GB+ SSD
- CUDA: 11.8+

**推理** (可用 CPU，但慢):
- GPU: 推荐 (RTX 3060+)
- RAM: 8GB+
- 存储: 50GB+

**替代方案**:
- Google Colab (免费 GPU)
- AWS / Lambda Labs (租用 GPU)

---

## 软件和安装

### Q5: 我应该使用哪个 LeRobot 版本？

**推荐**: 最新版本 (0.4.1+)

```bash
# 从源码安装 (推荐)
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
pip install -e ".[smolvla]"
pip install -e ".[feetech]"
```

**不推荐**: `pip install lerobot` (可能有依赖冲突)

### Q6: 安装遇到错误怎么办？

**常见问题**:

1. **CUDA 版本不匹配**:
```bash
# 检查 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 重新安装对应版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. **Feetech SDK 安装失败**:
```bash
# 确保系统包是最新的
sudo apt-get update
sudo apt-get install build-essential

# 重新安装
pip install -e ".[feetech]" --force-reinstall
```

3. **权限问题**:
```bash
# USB 端口权限
sudo chmod 666 /dev/ttyACM*

# 永久解决
sudo usermod -a -G dialout $USER
# 然后重新登录
```

### Q7: 需要安装 ROS 吗？

**不需要**！LeRobot 是独立的 Python 框架，不依赖 ROS。

---

## 数据采集

### Q8: 需要采集多少数据？

**最少**: 25 episodes
- ⚠️ 可能性能不佳

**推荐**: 50+ episodes
- ✅ 基本可用

**理想**: 100+ episodes
- ✅ 性能良好
- ✅ 泛化能力强

**参考**: LeRobot 官方数据集 `svla_so101_pickplace`
- 50 episodes
- 5 个不同位置
- 每个位置 10 个演示

### Q9: 每个 episode 应该多长？

**推荐**:
- 抓取任务: 10-30 秒
- 复杂任务: 30-60 秒

**关键**:
- 包含完整的任务流程
- 不要太长（避免无关动作）
- 不要太短（任务不完整）

### Q10: 如果我的环境和别人不一样，需要重新采集数据吗？

**是的！强烈建议重新采集。**

SmolVLA 对以下因素非常敏感:
- 桌面颜色和材质
- 背景环境
- 光照条件
- 摄像头型号和参数

**环境差异导致的问题**:
- 模型性能大幅下降
- 泛化能力差
- 可能完全失败

**解决方案**:
- 在您自己的环境中采集数据
- 或者使用域随机化技术（高级）

### Q11: 如何提高数据质量？

**建议**:

1. **多样性**:
   - 不同的物体位置
   - 不同的物体姿态
   - 不同的光照（如果环境光照会变化）

2. **一致性**:
   - 相同的摄像头位置
   - 相同的摄像头参数
   - 稳定的光照（推荐）

3. **质量控制**:
   - 动作流畅，不要卡顿
   - 避免碰撞和失败的尝试
   - 任务完整执行

4. **参考官方数据集**:
   - [svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)
   - 查看可视化，学习数据结构

---

## 模型训练

### Q12: 训练需要多长时间？

**时间估算** (20,000 steps, batch_size=64):

| GPU | 时间 |
|-----|------|
| A100 | ~4 小时 |
| RTX 3090 | ~6-8 小时 |
| RTX 3060 | ~12-16 小时 |
| CPU | ❌ 不推荐 (太慢) |

### Q13: 我的 GPU 显存不够怎么办？

**解决方案**:

1. **减小 Batch Size**:
```bash
# 从 64 降到 32
lerobot-train --batch_size=32 ...
```

2. **使用梯度累积**:
```bash
# 累积 2 步，有效 batch_size = 32 * 2 = 64
lerobot-train --batch_size=32 --gradient_accumulation_steps=2 ...
```

3. **使用更小的图像分辨率** (需要重新采集数据):
```python
# 从 640x480 降到 320x240
camera_config = OpenCVCameraConfig(..., width=320, height=240)
```

4. **使用云 GPU**:
- Google Colab (免费 T4)
- AWS / Lambda Labs (付费)

### Q14: 如何知道模型训练得好不好？

**监控指标** (在 W&B 或日志中):

1. **Loss 曲线**:
   - ✅ 应该持续下降
   - ✅ 趋于稳定（不震荡）
   - ❌ 如果上升或震荡剧烈，检查学习率

2. **Learning Rate**:
   - 正常衰减（如果使用 scheduler）

3. **训练速度**:
   - A100: ~10-15 steps/sec
   - RTX 3090: ~5-10 steps/sec

**定性评估**:
- 在实体机械臂上测试
- 观察动作是否合理
- 任务成功率

### Q15: 有没有已经训练好的模型可以直接用？

**基础模型**:
- `lerobot/smolvla_base`: 450M 参数，**需要微调**

**数据集**:
- `lerobot/svla_so101_pickplace`: SO101 抓放数据集

**重要**:
- ❌ 没有通用的即用模型
- ✅ SmolVLA 需要在您的数据上微调
- ✅ 即使有别人的模型，环境差异也需要重新训练

---

## 模型部署

### Q16: 推理速度是多少？

**控制频率**:
- 推荐: 30 Hz
- 可行: 10-50 Hz

**延迟**:
- GPU 推理: ~20-50 ms
- CPU 推理: ~100-500 ms (慢)

### Q17: 模型不按预期工作怎么办？

**排查步骤**:

1. **确认任务描述一致**:
   - 训练时: "Pick up the cube..."
   - 推理时: **必须相同**！

2. **确认摄像头配置一致**:
   - 分辨率: 640x480
   - 摄像头名称: "front", "wrist"
   - 位置和角度

3. **确认环境相似**:
   - 光照
   - 物体位置
   - 背景

4. **检查数据质量**:
   - 是否有足够的 episodes
   - 是否包含当前场景的变化

5. **尝试更多训练**:
   - 增加 steps 到 40k-50k
   - 采集更多数据

---

## Sim2Real

### Q18: MuJoCo 训练的模型能用于实体机械臂吗？

**简短回答**: ❌ **不能直接用**

**原因**:

1. **物理差异**:
   - 摩擦力、惯性、延迟
   - 仿真是理想化的

2. **视觉差异**:
   - 渲染 vs 真实摄像头
   - 光照、纹理、噪声

3. **环境差异**:
   - 仿真环境简单
   - 真实环境复杂多变

**Sim2Real Gap 有多大？**
- 通常导致 **50-100% 性能下降**
- 在某些任务上可能**完全失效**

### Q19: 那 MuJoCo 有什么用？

**MuJoCo 的价值**:

✅ **适合**:
- 学习 LeRobot 流程
- 快速验证算法
- 教学和演示
- 预训练（再用真实数据微调）

❌ **不适合**:
- 直接部署到实体机械臂
- 需要精确物理交互
- 生产环境

### Q20: 如何做 Sim2Real 迁移？

**高级技术**:

1. **域随机化 (Domain Randomization)**:
   - 随机化光照、颜色、纹理
   - 随机化物理参数
   - 需要修改仿真环境

2. **域适应 (Domain Adaptation)**:
   - 使用 GAN 转换图像
   - 特征对齐
   - 需要额外的模型

3. **混合训练**:
   - 仿真预训练
   - 真实数据微调（推荐）

**最佳方案**:
- ✅ **直接在实体机械臂上采集真实数据**
- 这是最可靠的方法

---

## 故障排除

### Q21: 找不到 USB 端口

```bash
# 1. 检查设备
ls -l /dev/ttyACM* /dev/ttyUSB*

# 2. 给予权限
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# 3. 永久解决
sudo usermod -a -G dialout $USER
# 重新登录

# 4. 使用 LeRobot 工具
lerobot-find-port
```

### Q22: 电机连接失败

**检查清单**:

1. ✅ 电源是否接通
2. ✅ USB 线是否连接
3. ✅ 3-pin 电机线是否插好
4. ✅ 电机是否正确连接（菊花链）
5. ✅ 第一个电机是否连接到控制板
6. ✅ Waveshare 板跳线是否在 B 通道

**重启步骤**:
1. 断开电源
2. 断开 USB
3. 检查所有连接
4. 重新连接电源
5. 重新连接 USB
6. 重试

### Q23: 摄像头无法打开

```bash
# 1. 列出摄像头
v4l2-ctl --list-devices

# 2. 测试摄像头
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.read()[0])"

# 3. 检查权限
ls -l /dev/video*
sudo chmod 666 /dev/video0

# 4. 检查是否被占用
lsof /dev/video0
```

### Q24: 训练时 CUDA Out of Memory

```bash
# 1. 减小 batch size
--batch_size=32  # 从 64 降到 32

# 2. 使用梯度累积
--gradient_accumulation_steps=2

# 3. 清理 GPU 缓存
python -c "import torch; torch.cuda.empty_cache()"

# 4. 检查 GPU 使用情况
nvidia-smi
```

### Q25: 标定失败

**常见原因**:

1. **电机未全部连接**: 确保所有 6 个电机都已连接
2. **电机 ID 未设置**: 先运行 `setup_motors`
3. **运动范围太小**: 确保移动每个关节的完整范围
4. **姿势不对**: 从中间位置开始

**重新标定**:
```bash
# 删除旧的标定文件
rm ~/.cache/lerobot/calibration/*

# 重新运行标定
lerobot-calibrate --robot.type=so101_follower --robot.port=...
```

---

## 获取帮助

### 官方资源

- **Discord**: [LeRobot Discord](https://discord.com/invite/s3KuuzsPFb)
- **GitHub Issues**: [报告 Bug](https://github.com/huggingface/lerobot/issues)
- **论坛**: [Hugging Face Forum](https://discuss.huggingface.co/)

### 社区资源

- **参考项目**: [lerobot-mujoco](https://github.com/q442333521/lerobot-mujoco)
- **文档**: [LeRobot Docs](https://huggingface.co/docs/lerobot)

### 提问技巧

**好的问题**:
- 包含具体的错误信息
- 说明您的环境（OS, GPU, 版本）
- 描述您尝试过的解决方案
- 提供相关代码或日志

**示例**:
```
标题: [SO101] 标定时电机连接失败

环境:
- Ubuntu 22.04
- LeRobot 0.4.1
- SO101 follower

问题:
运行 lerobot-calibrate 时出现错误:
[粘贴错误信息]

已尝试:
1. 检查 USB 连接 - 正常
2. 给予端口权限 - 已执行
3. 重启设备 - 问题依旧

请问可能是什么原因？
```

---

**祝您使用愉快！🎉**

如有更多问题，请参考官方文档或在社区提问。
