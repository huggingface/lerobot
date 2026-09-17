# AGENT_GUIDE.md — LeRobot AI 助手与用户实用指南

本文档是任何 AI 助手（Cursor、Claude、ChatGPT、Codex 等）帮助用户使用 LeRobot 时的实用、可复制粘贴的伴侣指南。它补充了 [`AGENTS.md`](./AGENTS.md)（开发者/贡献者上下文），提供**面向用户的指导**：如何开始、训练什么、训练多久、如何录制数据，以及如何校准 SO-101。

---

## 1. 从这里开始 — 先问用户（强制）

在建议任何命令之前，AI 助手**必须**至少询问用户以下问题并等待回答：

1. **你的目标是什么？**（例如："教我的 SO-101 折叠布料"、"在现有的 HF 数据集上训练策略"、"贡献 PR"、"理解代码库"）
2. **你有什么硬件？**
   - 机器人：无 / SO-100 / SO-101 / Koch / LeKiwi / Reachy / 其他
   - 遥操作：主臂 / 手机 / 键盘 / 游戏手柄 / 无
   - 摄像头：多少个、分辨率、固定还是移动？
3. **你将在什么机器上训练？**
   - GPU 型号 + 显存（例如"笔记本 3060 6GB"、"RTX 4090 24GB"、"A100 80GB"、"仅 CPU"）
   - 操作系统：macOS / Linux / Windows
4. **技能水平和时间预算？** 第一次、有一些 ML 经验、有经验？几小时、几天、一个周末？
5. **你已经有数据集了吗？** 是（HF repo id？）/ 否 / 想要录制一个
6. **我现在能帮你什么？**（选择一个具体的下一步）

只有在得到答案后，才提出具体的路径。如果有模糊之处，再次询问而不是猜测。倾向于为用户的硬件和目标选择**最简单可行的方案**。

---

## 2. 60 秒了解 LeRobot

LeRobot = **数据集 + 策略 + 环境 + 机器人控制**，由一组强大的抽象统一起来。

- **`LeRobotDataset`** — 感知 episode 的数据集（视频或图像 + 动作 + 状态），可从 Hub 或磁盘加载。
- **策略**（`ACT`、`Diffusion`、`SmolVLA`、`π0`、`π0.5`、`Wall-X`、`X-VLA`、`VQ-BeT`、`TD-MPC`、…）— 都继承 `PreTrainedPolicy`，可以推送到 Hub 或从 Hub 拉取。
- **处理器** — 数据集 → 策略 → 机器人之间的小型可组合转换。
- **环境**（仿真）和**机器人**（真实）— 相同的动作/观察合约，代码可以干净地切换。
- **CLI** — `lerobot-record`、`lerobot-train`、`lerobot-eval`、`lerobot-teleoperate`、`lerobot-calibrate`、`lerobot-find-port`、`lerobot-setup-motors`、`lerobot-replay`。

参见 [`AGENTS.md`](./AGENTS.md) 了解仓库架构。

---

## 3. 快速入门路径（选择一个）

### 路径 A — "我有 SO-101，想要训练第一个策略"

前往第 4 节（SO-101 端到端），然后第 5 节（数据技巧），然后第 6 节（选择策略 — 可能是 **ACT**），然后第 7 节（训练多久），然后第 8 节（评估）。

### 路径 B — "没有硬件，我想在现有数据集上训练"

跳过第 4 节。在第 6 节选择策略，在第 7 节选择时长，然后按照第 4.9 节运行 `lerobot-train`，使用 Hub `--dataset.repo_id` 和 `--env.type` 进行评估。最后完成第 8 节。

### 路径 C — "我只想理解代码库"

阅读上面的第 2 节，然后阅读 `AGENTS.md` 的"架构"部分，然后打开 `src/lerobot/policies/act/` 和 `src/lerobot/datasets/lerobot_dataset.py` 作为典型示例。

---

## 4. SO-101 端到端速查表

完整详情见 [`docs/source/so101.mdx`](./docs/source/so101.mdx) 和 [`docs/source/il_robots.mdx`](./docs/source/il_robots.mdx)。以下是最小命令集，按顺序执行。 issuing 前确认机械臂已组装并通电。

**4.1 安装**

```bash
# uv（推荐 — 见 AGENTS.md 和 CLAUDE.md）
uv sync --locked --extra feetech          # SO-100/SO-101 电机堆栈
# uv sync --locked --extra all            # 全部
# uv sync --locked --extra smolvla        # 添加 SmolVLA 依赖

# pip（替代方案，例如不从源码工作时）
# pip install 'lerobot[feetech]'
# pip install 'lerobot[all]'
# pip install 'lerobot[smolvla]'

git lfs install && git lfs pull
hf auth login                             # 推送数据集/策略所需
```

**4.2 查找 USB 端口** — 每个机械臂运行一次，按提示拔掉插头。

```bash
lerobot-find-port
```

macOS: `/dev/tty.usbmodem...`；Linux: `/dev/ttyACM0`（可能需要 `sudo chmod 666 /dev/ttyACM0`）。

**4.3 设置电机 ID 和波特率**（一次性，每个机械臂）

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=<FOLLOWER_PORT>
lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>
```

**4.4 校准** — 将所有关节居中，按 Enter，然后将每个关节扫过其完整范围。`id` 是校准密钥 — 在所有地方重复使用它。

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower
lerobot-calibrate --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>   --teleop.id=my_leader
```

**4.5 遥操作**（健全性检查，不录制）

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

> **SO-100 / SO-101 上的 Feetech 超时/通信错误？** 在调整软件之前，先检查菊花链上的**红色电机 LED**。
>
> - **全部稳定红色，从夹爪→底座链** → 接线正常。
> - **一个或多个电机熄灭/链在中途停止** → 接线问题：重新插入 3 针电缆，检查控制器板电源，确保每个电机完全插入。
> - **LED 闪烁** → 电机处于**错误状态**：通常是过载（强制关节超过其限制）**或电源电压错误**。SO-100 / SO-101 有两种变体 — **5V / 7.4V** 构建和 **12V** 构建 — 它们**不可互换**。在 5V / 7.4V 机械臂上使用 12V 电源（反之亦然）会触发此错误；在上电前确认你的电机组件变体。
>
> 大多数"超时"错误是物理问题，不是代码问题。

**4.6 录制数据集** — 按键：**→** 下一个，**←** 重做，**ESC** 完成并上传。

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')

lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=<LEADER_PORT>  --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/my_task \
  --dataset.single_task="<用一句话描述任务>" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --display_data=true
```

**4.7 可视化** — 训练前**务必**执行此操作。查找缺失的帧、摄像头模糊、无法触及的目标、不一致的物体位置。
上传后：https://huggingface.co/spaces/lerobot/visualize_dataset → 粘贴 `${HF_USER}/my_task`。适用于**任何 LeRobot 格式的 Hub 数据集** — 用它来侦察其他数据集、检查 episode 质量或在重新训练前调试自己的数据。

**4.8 重放 episode**（健全性检查）

```bash
lerobot-replay --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --dataset.repo_id=${HF_USER}/my_task --dataset.episode=0
```

**4.9 训练**（默认：ACT — 最快、最低内存）。Apple silicon：`--policy.device=mps`。没有本地 GPU？添加 `--job.target=<flavor>`（例如 `a10g-small`，用 `hf jobs hardware` 列出它们）在 Hugging Face Jobs 上运行。参见第 6/7 节了解策略和时长。

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/act_my_task \
  --job_name=act_my_task \
  --batch_size=8 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_my_task
```

**4.10 在真实机器人上评估** — 将成功率与遥操作基线进行比较。

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<与训练时相同的任务描述>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/act_my_task
```

---

## 5. 数据收集技巧（初学者 → 可靠策略）

好数据胜过巧妙的模型。采用这些默认值，只有在有证据时才偏离。

### 5.1 设置与人体工程学

- **在接触软件之前固定装置和摄像头。** 如果装置振动或操作员感到沮丧，先解决这个问题 — 更多坏数据没有帮助。
- **照明比分辨率更重要。** 漫射、一致的光线。避免移动的阴影。
- **"你能仅从摄像头视图完成任务吗？"** 如果不能，你的摄像头有问题。在录制前修复。
- 可用时启用**动作插值**以获得更平滑的轨迹。

### 5.2 录制前练习

- 不录制的情况下做 5-10 次演示。建立深思熟虑、可重复的策略。
- 犹豫或不一致的演示会教会模型犹豫。

### 5.3 质量优于速度

深思熟虑、高质量的执行胜过快速 sloppy 的运行。只有在策略调整好后才优化速度 — 永远不要为了速度而牺牲质量。

### 5.4 episode 内和 episode 间的一致性

相同的抓握、接近向量和时间。连贯的策略比 wildly 变化的动作更容易学习。

### 5.5 从小开始，然后扩展（黄金法则）

- **前 50 个 episode = 任务的约束版本**：一个物体、固定位置、固定摄像头设置、一个操作员。
- 快速训练一个 ACT 模型。看看什么失败了。
- **然后一次沿一个轴添加多样性**：更多位置 → 更多照明 → 更多物体 → 更多操作员。
- 不要试图在第一天收集"完美数据集"。迭代。

### 5.6 初学者的策略选择

- **笔记本/第一次/想要快速结果 → ACT。** 效果出奇的好，即使在笔记本 GPU 上也能快速训练。
- **更大的 GPU/语言条件/多任务 → SmolVLA。** 在这里解冻视觉编码器（见第 7 节）是一个大的收获。
- 推迟 π0 / π0.5 / Wall-X / X-VLA，直到你有经过验证的 ACT 基线和 20+ GB GPU。

### 5.7 第一个任务的推荐默认值

| 设置 | 值 |
|------|------|
| Episodes | **50** 开始，第一次训练后扩展到 100-300 |
| Episode 长度 | 20-45 秒（对于抓握/放置，更短也可以） |
| 重置时间 | 10 秒 |
| FPS | 30 |
| 摄像头 | **推荐 2 个摄像头**：1 个固定前置 + 1 个手腕。多视图通常优于单视图。单个固定摄像头也可以保持简单。 |
| 任务描述 | 简短、具体、动作短语的句子 |

### 5.8 故障排除信号

- 策略在一个特定阶段失败 → 录制 10-20 个更多 episode **针对该阶段**。
- 策略抖动/振荡 → 可能是不一致的演示，或需要更多训练；重新录制最差的 episode（使用 **←** 重做）。
- 策略忽略物体 → 摄像头框架或照明问题，不是模型问题。

另见：[什么构成好数据集](https://huggingface.co/blog/lerobot-datasets#what-makes-a-good-dataset)。

---

## 6. 我应该训练哪个策略？

将策略与用户的**GPU 内存**和**时间预算**匹配。以下数字来自内部分析运行（每个策略一次训练更新）。它们**仅供参考** — 见注意事项。

### 6.1 分析快照（参考）

所有策略通常训练 **5-10 个 epoch**（见第 7 节）。

> **面向人类的版本：** [计算硬件指南](./docs/source/hardware_guide.mdx) 重用了下面的表格，并添加了云 GPU 层级指南和 Hugging Face Jobs 指针。

| 策略 | Batch | Update (ms) | 峰值 GPU 内存 (GB) | 最适合 |
|------|------:|------------:|-------------------:|--------|
| `act` | 4 | **83.9** | **0.94** | 首次用户、笔记本、单任务。快速可靠。 |
| `diffusion` | 4 | 168.6 | 4.94 | 多模态动作分布；需要中端 GPU。 |
| `smolvla` | 1 | 357.8 | 3.93 | 语言条件、多任务、小型 VLA。**解冻视觉编码器获得大收益**（见第 7 节）。 |
| `xvla` | 1 | 731.6 | 15.52 | 大型 VLA、多任务。 |
| `wall_x` | 1 | 716.5 | 15.95 | 具有世界模型目标的大型 VLA。 |
| `pi0` | 1 | 940.3 | 15.50 | 强大的大型 VLA 基线（Physical Intelligence）。 |
| `pi05` | 1 | 1055.8 | 16.35 | 更新的 π 策略；与 `pi0` 类似的占用空间。 |

**关键注意事项：**

- **优化器：** 使用 **SGD** 测量。LeRobot 的默认是 **AdamW**，它保留额外的优化器状态 → **使用默认值时峰值内存会明显更高**，尤其是 `pi0`、`pi05`、`wall_x`、`xvla`。
- **Batch 大小：** 大型策略在 batch 1 下分析。实践中使用**更大的 batch** 以获得稳定的训练（见第 7.4 节）。内存与 batch 大致线性扩展。

### 6.2 决策规则

- **< 8 GB VRAM（笔记本、3060、M 系列 Mac）：** → `act`。如果你有 ~6-8 GB 空闲，也许是 `diffusion`。
- **12-16 GB VRAM（4070/4080、A4000）：** → 默认使用 `smolvla`，或使用更大 batch 的 `act`/`diffusion`。`pi0`/`pi05`/`wall_x`/`xvla` 仅在小 batch + 梯度积累下可行。
- **24+ GB VRAM（3090/4090/A5000）：** → 任何策略。多任务优先使用 `smolvla`（解冻）；单任务抓握和放置优先使用 `act`（通常仍然是最佳 ROI）。可以实验 `pi0` 或 `pi05` 或 `xvla`
- **80 GB（A100/H100）：** → 任何，使用健康的 batch。`pi05`、`xvla`、`wall_x` 变得舒适。
- **仅 CPU：** → 不要在这里训练。使用 Google Colab（见 [`docs/source/notebooks.mdx`](./docs/source/notebooks.mdx)）或租用的 GPU。

---

## 7. 我应该训练多久？

机器人模仿学习通常在**数据集上的几个 epoch** 内收敛，而不是数十万原始步骤。首先考虑**epochs**，然后转换为步骤。

### 7.1 经验法则

- **典型总计：5-10 个 epochs。** 从 5 开始，评估，然后决定是否需要更多。
- 非常小的数据集（< 30 episodes）可能需要稍微更多的 epochs — 但首先，**收集更多数据**。
- 具有预训练视觉骨干的 VLA 通常需要比从头训练**更少**的 epochs。

### 7.2 步骤 ↔ epochs 转换

```
total_frames     = 所有 episode 的帧数总和      # 例如 50 eps × 30 fps × 30 s ≈ 45,000
steps_per_epoch  = ceil(total_frames / batch_size)
total_steps      = epochs × steps_per_epoch
```

`--batch_size=8` 的示例：

| 数据集大小 | 帧数 | 每 epoch 步骤 | 5 epochs | 10 epochs |
|------------|------:|-------------:|---------:|----------:|
| 50 eps × 30 s @ 30 fps | 45,000 | ~5,625 | 28k | 56k |
| 100 eps × 30 s @ 30 fps | 90,000 | ~11,250 | 56k | 113k |
| 300 eps × 30 s @ 30 fps | 270,000 | ~33,750 | 169k | 338k |

使用 `--steps=<N>` 传递结果总数；在中间检查点评估（`outputs/train/.../checkpoints/`）。

### 7.3 每策略起点（单任务，~50 episodes）

| 策略 | Batch | 步骤（首次运行） | 注意事项 |
|------|------:|-----------------:|----------|
| `act` | 8-16 | 30k-80k | 单任务通常在 50k 以下收敛。 |
| `diffusion` | 8-16 | 80k-150k | 比 ACT 受益于更长的训练。 |
| `smolvla` | 4-8 | 30k-80k | 预训练 VLM → 快速收敛。 |
| `pi0` / `pi05` | 1-4 | 30k-80k | 内存受限；使用梯度积累使有效 batch ≥ 16！ |

### 7.4 Batch 大小指导

- **更大的 batch 更可取** 以获得遥操作数据的稳定梯度。
- 如果 GPU 内存是瓶颈，使用**梯度积累**来提高*有效*batch 而不提高峰值内存。
- 随 batch 轻轻调整**学习率**；大多数 LeRobot 默认值对于 2-4× batch 变化工作良好。

### 7.5 使用 `--steps` 缩放 LR 计划和检查点

LeRobot 的默认调度器（例如 SmolVLA 的余弦衰减）使用 `scheduler_decay_steps=30_000`，这是为长训练运行调整大小的。当你缩短训练时（例如在小数据集上 5k-10k 步），**缩小调度器以匹配** — 否则 LR 保持在峰值附近并且永远不会衰减。检查点频率也是如此。

```bash
lerobot-train ... \
  --steps=5000 \
  --policy.scheduler_decay_steps=5000 \
  --save_freq=5000
```

经验法则：设置 `scheduler_decay_steps ≈ steps`，`save_freq` 设置为你想要的评估粒度（例如每 1k-5k 步）。如果你的运行非常短，按比例匹配 `scheduler_warmup_steps`。

### 7.6 SmolVLA：解冻视觉编码器获得实际收益

SmolVLA 附带 `freeze_vision_encoder=True`。解冻通常在专门任务上**大幅提高性能**，但代价是更多 VRAM 和更慢的步骤。启用：

```bash
lerobot-train ... --policy.type=smolvla \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false
```

### 7.7 停止/继续的信号

- 训练损失趋于平稳 → 停止，保存 Hub 检查点。
- 训练损失仍在下降且你低于 10 个 epochs → 继续。

---

## 8. 评估与基准

两种评估风格：

### 8.1 真实机器人评估（SO-101 等）

重用 `lerobot-record` 和 `--policy.path` 在机器人上运行训练的策略，并将运行保存为评估数据集。约定：用 `eval_` 前缀命名数据集。

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=<FOLLOWER_PORT> --robot.id=my_follower \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_my_task \
  --dataset.single_task="<与训练时使用的相同任务描述>" \
  --dataset.num_episodes=10 \
  --policy.path=${HF_USER}/act_my_task
```

报告跨 episodes 的成功率。与遥操作基线和早期检查点进行比较以捕捉回归。

### 8.2 仿真基准评估

对于在仿真数据集（PushT、Aloha、LIBERO、MetaWorld、RoboCasa、…）上训练的策略，使用 `lerobot-eval` 对抗匹配的 `env.type`：

```bash
lerobot-eval \
  --policy.path=${HF_USER}/diffusion_pusht \
  --env.type=pusht \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --policy.device=cuda
```

- 使用 `--policy.path=outputs/train/.../checkpoints/<step>/pretrained_model` 获取本地检查点。
- `--eval.n_episodes` 应 ≥ 50 以获得稳定的成功率估计。
- 可用的 envs 位于 `src/lerobot/envs/`。参见 [`docs/source/libero.mdx`](./docs/source/libero.mdx)、[`metaworld.mdx`](./docs/source/metaworld.mdx)、[`robocasa.mdx`](./docs/source/robocasa.mdx)、[`vlabench.mdx`](./docs/source/vlabench.mdx) 了解特定基准。
- 要添加新基准，参见 [`docs/source/adding_benchmarks.mdx`](./docs/source/adding_benchmarks.mdx) 和 [`envhub.mdx`](./docs/source/envhub.mdx)。

### 8.2b 基准评估的 Dockerfiles

基准环境有本地安装痛苦的本地依赖。仓库为每个支持的基准提供**预构建的 Dockerfiles** — 使用这些在可重现的环境中运行 `lerobot-eval`：

| 基准 | Dockerfile |
|------|------------|
| LIBERO | [`docker/Dockerfile.benchmark.libero`](./docker/Dockerfile.benchmark.libero) |
| LIBERO+ | [`docker/Dockerfile.benchmark.libero_plus`](./docker/Dockerfile.benchmark.libero_plus) |
| MetaWorld | [`docker/Dockerfile.benchmark.metaworld`](./docker/Dockerfile.benchmark.metaworld) |
| RoboCasa | [`docker/Dockerfile.benchmark.robocasa`](./docker/Dockerfile.benchmark.robocasa) |
| RoboCerebra | [`docker/Dockerfile.benchmark.robocerebra`](./docker/Dockerfile.benchmark.robocerebra) |
| RoboMME | [`docker/Dockerfile.benchmark.robomme`](./docker/Dockerfile.benchmark.robomme) |
| RoboTwin | [`docker/Dockerfile.benchmark.robotwin`](./docker/Dockerfile.benchmark.robotwin) |
| VLABench | [`docker/Dockerfile.benchmark.vlabench`](./docker/Dockerfile.benchmark.vlabench) |

构建和运行（适应你的基准）：

```bash
docker build -f docker/Dockerfile.benchmark.robomme -t lerobot-bench-robomme .
docker run --gpus all --rm -it \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  lerobot-bench-robomme \
  lerobot-eval --policy.path=<your_policy> --env.type=<env> --eval.n_episodes=50
```

参见 [`docker/README.md`](./docker/README.md) 了解基础镜像详情。

### 8.3 目标成功率

使用 50 个干净 episode 的单任务抓握和放置：ACT 应在训练配置上达到 **> 70% 成功率**。更低 → 数据问题（见第 5 节），不是模型问题。预期泛化到新位置时会下降 — 扩展 episodes 或多样性以恢复。

---

## 9. 进一步阅读与资源

- **入门：** [`installation.mdx`](./docs/source/installation.mdx) · [`il_robots.mdx`](./docs/source/il_robots.mdx) · [什么构成好数据集](https://huggingface.co/blog/lerobot-datasets)
- **每策略文档：** 浏览 [`docs/source/*.mdx`](./docs/source/)（策略、硬件、基准、高级训练）。
- **社区：** [Discord](https://discord.com/invite/s3KuuzsPFb) · [Hub `LeRobot` 标签](https://huggingface.co/datasets?other=LeRobot) · [数据集可视化器](https://huggingface.co/spaces/lerobot/visualize_dataset)

> 保持此文件最新。如果你学到了一条可以防止一类用户错误的规则，将其添加到这里和 [`AGENTS.md`](./AGENTS.md) 中。
