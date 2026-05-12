# LeRobot v0.5.2 — HuggingFace 机器人学习库

**技术栈**: Python 3.12+ / PyTorch / HF Hub / draccus / Gymnasium / uv

## 项目概况

LeRobot 是 HuggingFace 开源的 PyTorch 机器人学习库，提供数据集、预训练策略、训练/评估/数据采集/机器人控制工具，集成 Hugging Face Hub 进行模型和数据集共享。

- **仓库**: https://github.com/huggingface/lerobot
- **文档**: https://huggingface.co/docs/lerobot/index
- **License**: Apache-2.0
- **包管理**: uv
- **Lint**: ruff (target py312, line-length 110)
- **类型检查**: mypy (渐进式，仅部分模块严格)

## 开发环境

```bash
uv sync --locked                            # 基础依赖
uv sync --locked --extra test --extra dev   # 测试+开发工具
uv sync --locked --extra all                # 全部依赖
git lfs install && git lfs pull             # 测试制品
```

## 关键命令

```bash
uv run pytest tests -svv --maxfail=10                 # 全部测试
DEVICE=cuda make test-end-to-end                      # E2E 测试
pre-commit run --all-files                            # Lint + 格式化 (ruff, typos, bandit)
```

## 核心架构 (`src/lerobot/`)

### 模块总览

| 模块 | 职责 | 关键导出 |
|------|------|---------|
| **configs/** | 数据类配置（draccus 解析） | `TrainPipelineConfig`, `EvalConfig`, `PreTrainedConfig`, `DatasetConfig` |
| **policies/** | 策略模型（各子目录含 config+modeling+processor） | `PreTrainedPolicy`, ACT, Diffusion, TDMPC, VQBeT, PI0, PI0.5, GR00T, SmolVLA, WallX, XVLA, SAC, SARM, MultiTaskDiT |
| **processor/** | 数据变换管线（注册制步骤链） | `DataProcessorPipeline`, `PolicyProcessorPipeline`, 标准化/动作/观测/HIL 等处理器 |
| **datasets/** | LeRobotDataset (Parquet+MP4, episode-aware) | `LeRobotDataset`, `LeRobotDatasetMetadata`, `StreamingLeRobotDataset`, 数据集工具 |
| **envs/** | 仿真环境（Gymnasium 封装） | `EnvConfig`, `AlohaEnv`, `PushtEnv`, LIBERO/MetaWorld/RoboCasa 等 |
| **robots/** | 机器人硬件抽象 | `Robot`, `RobotConfig`, SO100/Koch/LeKiwi/HopeJR/Reachy2/UnitreeG1 等 |
| **motors/** | 电机总线驱动 | `Motor`, `MotorCalibration`, Dynamixel/Feetech/Damiao/RobStride |
| **cameras/** | 相机驱动 | `Camera`, OpenCV/RealSense/ZMQ/Reachy2 相机 |
| **teleoperators/** | 遥操作设备 | `Teleoperator`, Keyboard/Gamepad/Phone/SO Leader 等 |
| **scripts/** | 18 个 CLI 入口 | `lerobot-train`, `lerobot-eval`, `lerobot-record`, `lerobot-teleoperate` 等 |
| **optim/** | 优化器与调度器 | Adam/AdamW/SGD 配置, Cosine/Diffuser/VQBeT 调度器 |
| **common/** | 跨模块工具 | `train_utils`, `control_utils`, `wandb_utils` |
| **rl/** | 强化学习 | Actor, Learner, Buffer, HIL-SERL |
| **async_inference/** | 异步推理 | PolicyServer, RobotClient |
| **model/** | 运动学 | `RobotKinematics` |
| **transforms/** | 图像增强 | `ImageTransforms`, `RandomSubsetApply` |
| **transport/** | gRPC 通信 | protobuf 服务定义 |
| **utils/** | 通用工具 | 日志/导入/设备/HUB/旋转/可视化等 |
| **types.py** | 核心类型 | `TransitionKey`, `PolicyAction`, `RobotAction`, `EnvTransition` |

### 策略子模块结构 (每个策略遵循 config+modeling+processor 模式)

| 子模块 | 额外文件 |
|--------|---------|
| `act/` | 标准 3 文件 |
| `diffusion/` | 标准 3 文件 |
| `tdmpc/` | 标准 3 文件 |
| `vqbet/` | + `vqbet_utils.py` |
| `pi0/` | 标准 3 文件 |
| `pi0_fast/` | 标准 3 文件 |
| `pi05/` | 标准 3 文件 |
| `multi_task_dit/` | 标准 3 文件 |
| `sac/` | + `reward_model/` 子目录 (classifier config/modeling/processor) |
| `sarm/` | + `compute_rabc_weights.py`, `sarm_utils.py` |
| `smolvla/` | + `smolvlm_with_expert.py` |
| `groot/` | + `groot_n1.py`, `utils.py`, `action_head/`, `eagle2_hg_model/` |
| `rtc/` | + `action_interpolator.py`, `action_queue.py`, `debug_tracker.py`, `debug_visualizer.py`, `latency_tracker.py` |
| `wall_x/` | + `constant.py`, `utils.py`, `qwen_model/` |
| `xvla/` | + `action_hub.py`, `configuration_florence2.py`, `modeling_florence2.py`, `soft_transformer.py`, `utils.py` |

### 机器人子模块

| 子模块 | 文件 |
|--------|------|
| `koch_follower/` | config + implementation |
| `so_follower/` | config + implementation + `robot_kinematic_processor.py` |
| `bi_so_follower/` | config + implementation |
| `hope_jr/` | config + arm + hand |
| `lekiwi/` | config + lekiwi + client + host |
| `reachy2/` | config + implementation |
| `unitree_g1/` | config + implementation + kinematics + utils + locomotion + server + sdk_socket |
| `omx_follower/` | config + implementation |
| `openarm_follower/` | config + implementation |
| `bi_openarm_follower/` | config + implementation |
| `nero_follower/` | config + implementation + `robot_kinematic_processor.py` |
| `earthrover_mini_plus/` | config + implementation |

### 电机后端子模块 (每个含 driver + register tables)

| 子模块 | 文件 |
|--------|------|
| `dynamixel/` | `dynamixel.py`, `tables.py` |
| `feetech/` | `feetech.py`, `tables.py` |
| `robstride/` | `robstride.py`, `tables.py` |
| `damiao/` | `damiao.py`, `tables.py` |

### 相机后端子模块

| 子模块 | 文件 |
|--------|------|
| `opencv/` | `camera_opencv.py`, `configuration_opencv.py` |
| `realsense/` | `camera_realsense.py`, `configuration_realsense.py` |
| `reachy2_camera/` | `reachy2_camera.py`, `configuration_reachy2_camera.py` |
| `zmq/` | `camera_zmq.py`, `configuration_zmq.py`, `image_server.py` |

### 遥操作子模块

| 子模块 | 文件 |
|--------|------|
| `koch_leader/` | config + implementation |
| `so_leader/` | config + implementation |
| `bi_so_leader/` | config + implementation |
| `openarm_leader/` | config + implementation |
| `bi_openarm_leader/` | config + implementation |
| `openarm_mini/` | config + implementation |
| `omx_leader/` | config + implementation |
| `gamepad/` | config + utils + implementation |
| `keyboard/` | config + implementation |
| `phone/` | config + processor + implementation |
| `homunculus/` | config + arm + glove + joints_translation |
| `reachy2_teleoperator/` | config + implementation |
| `unitree_g1/` | config + implementation + exo_calib + exo_ik + exo_serial |

## CLI 入口 (pyproject.toml [project.scripts])

| 命令 | 入口函数 |
|------|---------|
| `lerobot-calibrate` | `lerobot.scripts.lerobot_calibrate:main` |
| `lerobot-find-cameras` | `lerobot.scripts.lerobot_find_cameras:main` |
| `lerobot-find-port` | `lerobot.scripts.lerobot_find_port:main` |
| `lerobot-record` | `lerobot.scripts.lerobot_record:main` |
| `lerobot-replay` | `lerobot.scripts.lerobot_replay:main` |
| `lerobot-setup-motors` | `lerobot.scripts.lerobot_setup_motors:main` |
| `lerobot-teleoperate` | `lerobot.scripts.lerobot_teleoperate:main` |
| `lerobot-eval` | `lerobot.scripts.lerobot_eval:main` |
| `lerobot-train` | `lerobot.scripts.lerobot_train:main` |
| `lerobot-train-tokenizer` | `lerobot.scripts.lerobot_train_tokenizer:main` |
| `lerobot-dataset-viz` | `lerobot.scripts.lerobot_dataset_viz:main` |
| `lerobot-info` | `lerobot.scripts.lerobot_info:main` |
| `lerobot-find-joint-limits` | `lerobot.scripts.lerobot_find_joint_limits:main` |
| `lerobot-imgtransform-viz` | `lerobot.scripts.lerobot_imgtransform_viz:main` |
| `lerobot-edit-dataset` | `lerobot.scripts.lerobot_edit_dataset:main` |
| `lerobot-setup-can` | `lerobot.scripts.lerobot_setup_can:main` |

## 核心类型 (`types.py`)

- `TransitionKey` (Enum): `OBSERVATION`, `ACTION`, `REWARD`, `DONE`, `TRUNCATED`, `INFO`, `COMPLEMENTARY_DATA`
- `PolicyAction` = `torch.Tensor`
- `RobotAction` = `dict[str, Any]`
- `EnvAction` = `np.ndarray`
- `RobotObservation` = `dict[str, Any]`
- `EnvTransition` (TypedDict)

## 配置类型 (`configs/types.py`)

- `FeatureType` (Enum): `STATE`, `VISUAL`, `ENV`, `ACTION`, `REWARD`, `LANGUAGE`
- `PipelineFeatureType` (Enum): `ACTION`, `OBSERVATION`
- `NormalizationMode` (Enum): `MIN_MAX`, `MEAN_STD`, `IDENTITY`, `QUANTILES`, `QUANTILE10`
- `PolicyFeature` (dataclass): `type: FeatureType`, `shape: tuple[int, ...]`
- `RTCAttentionSchedule` (Enum): `ZEROS`, `ONES`, `LINEAR`, `EXP`

## 可选依赖要点

- 很多策略、环境、机器人在 extras 后面 (如 `lerobot[aloha]`, `lerobot[pi]`)
- 可选包的导入必须有 guard 或 lazy import
- 数据集可存储视频文件，`LeRobotDataset` 处理帧提取，测试需要 ffmpeg
- 优先使用 `uv run` 执行 Python 命令

## Mypy 严格模块

仅以下模块启用严格类型检查:
- `lerobot.envs`
- `lerobot.configs`
- `lerobot.optim`
- `lerobot.model`
- `lerobot.cameras`
- `lerobot.motors`
- `lerobot.transport`

修改这些模块时必须添加类型注解。

## 测试 (`tests/`)

- **~88 个测试文件**, 20 个子目录
- 测试夹具: `conftest.py` + `fixtures/` (数据集工厂、Hub mock、优化器)
- Mock: `mocks/` (Dynamixel/Feetech/Robot/Teleop mock)
- 跳过装饰器: `@require_cuda`, `@require_env`, `@skip_if_package_missing`, `@require_hf_token`
- E2E 测试: 通过 Makefile 运行 ACT/Diffusion/TDMPC/SmolVLA 的训练+评估
- 测试模式: 参数化策略测试、工厂模式数据生成、合约/模式测试、硬件 Mock、回归测试

## 其他目录

| 目录 | 内容 |
|------|------|
| `docker/` | 12 个 Dockerfile (用户镜像 + 多个 benchmark 镜像) |
| `benchmarks/` | 视频性能基准测试 |
| `nero/` | Nero 机器人 URDF/meshes |
| `examples/` | 12 个示例 (数据集/训练/HIL/LeKiwi/Phone 等) |
| `docs/` | HF 文档 (.mdx) |
| `scripts/ci/` | CI 脚本 |
| `build/` | 构建产物 |
