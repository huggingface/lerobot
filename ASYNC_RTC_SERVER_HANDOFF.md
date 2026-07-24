# LeRobot 异步推理 RTC：服务器端移交说明

本文档交给服务器端 Codex Agent 使用。目标是让远程异步推理和本地
`lerobot-rollout` 一样，通过 `--inference.type=rtc` 与
`--inference.rtc.*` 参数启用 Real-Time Chunking（RTC）。

## 1. 版本与结论

- Mac 仓库：`/Users/diwu/lerobot`
- 工作分支：`codex/async-jpeg-transport`
- RTC 功能提交：`32155b5d`
- RTC 修改前的提交：`72636809`
- JPEG 异步传输功能提交：`119cb12d`
- 修改前已有本地快照：`1086b7b1`

修改前的异步推理只会在客户端按时间戳覆盖或加权 action chunk，没有把前一个
chunk 的模型空间剩余动作、推理延迟或 `RTCConfig` 传入服务器模型，因此不是真正
的 RTC。

`32155b5d` 完成了以下闭环：

1. 客户端沿用 rollout 的 `InferenceEngineConfig`、`RTCInferenceConfig` 和
   `RTCConfig`，因此 CLI 参数名和默认值与本地 rollout 一致。
2. 服务器加载模型后注入客户端传来的 `RTCConfig`，再调用
   `init_rtc_processor()`。
3. 服务器给客户端同时返回：
   - 已经过 postprocessor、可直接发送给机械臂的动作；
   - 未经 postprocessor 的模型空间动作，用于下一次 RTC prefix guidance。
4. 客户端在发出下一帧 observation 时，同时携带队列中尚未执行的两类动作和
   端到端延迟步数。
5. 服务器把 prefix 补齐或截断到 `execution_horizon`，然后调用：

   ```python
   policy.predict_action_chunk(
       observation,
       inference_delay=...,
       prev_chunk_left_over=...,
       execution_horizon=...,
   )
   ```

6. 使用相对动作预处理器时，服务器会按照本地 rollout 的逻辑，用当前机器人状态
   对剩余绝对动作重新锚定并归一化。
7. RTC 模式下客户端使用新 chunk 直接替换重叠的旧动作，不再额外执行
   `weighted_average`。RTC 模型本身负责新旧 chunk 的连续过渡。

不需要修改 protobuf，也没有新增第三方依赖。

## 2. 两端版本要求

启用 RTC 前，Mac 客户端与服务器必须都包含 `32155b5d` 的协议修改。只升级一端
不属于受支持的 RTC 组合。

服务器运行时至少需要同步：

- `src/lerobot/async_inference/helpers.py`
- `src/lerobot/async_inference/policy_server.py`

Mac 客户端还需要：

- `src/lerobot/async_inference/configs.py`
- `src/lerobot/async_inference/robot_client.py`

推荐直接应用完整提交，而不是手工只复制两个文件，这样测试和数据类定义保持
一致。服务器必须同时保留此前的 JPEG 解码修改；如果服务器还未应用
`119cb12d`，应先部署 JPEG 提交或一次性应用完整提交序列。

## 3. 服务器端 Git 备份与部署

服务器端 Codex Agent 首先检查当前仓库，不要覆盖未提交修改，也不要把模型、
日志、缓存或数据集加入 Git：

```bash
cd ~/lerobot
git status --short --branch
git log --oneline -5
python -c 'import lerobot; print(lerobot.__file__)'
```

最后一条应指向当前仓库的 `~/lerobot/src/lerobot/`。如果服务器存在相关的未提交
源码，先建立备份分支，并只暂存已经核对过的源码：

```bash
git switch -c codex/server-before-async-rtc
git diff --check
git add <逐个列出需要备份的源码文件>
git commit -m "chore: snapshot server before async RTC"
```

### 3.1 服务器已有 JPEG 改造

Mac 端生成仅包含 RTC 功能的 patch：

```bash
cd /Users/diwu/lerobot
git format-patch -1 32155b5d --stdout > /tmp/lerobot-async-rtc.patch
scp /tmp/lerobot-async-rtc.patch wudi@192.168.18.191:~/
```

服务器端：

```bash
cd ~/lerobot
git switch -c codex/async-rtc-server
git apply --check ~/lerobot-async-rtc.patch
git am ~/lerobot-async-rtc.patch
git log --oneline -5
```

### 3.2 服务器尚未部署 JPEG 改造

Mac 端生成从 JPEG 到 RTC 的完整 patch 序列：

```bash
cd /Users/diwu/lerobot
git format-patch --stdout 119cb12d^..32155b5d > /tmp/lerobot-async-jpeg-rtc.patch
scp /tmp/lerobot-async-jpeg-rtc.patch wudi@192.168.18.191:~/
```

服务器端：

```bash
cd ~/lerobot
git switch -c codex/async-jpeg-rtc-server
git am ~/lerobot-async-jpeg-rtc.patch
git log --oneline -8
```

如果 `git apply --check` 或 `git am` 报冲突，不要使用 `git reset --hard`，也不要
覆盖服务器原有代码。应由服务器端 Codex Agent 对照本文第 1、2 节逐项合并，并
在服务器上创建自己的提交。部署后记录服务器实际提交号：

```bash
git rev-parse HEAD
```

服务器基线不同或经过冲突合并时，最终提交号与 Mac 的 `32155b5d` 不同是正常的。

## 4. 验证代码与参数入口

优先使用项目的 `uv` 环境：

```bash
cd ~/lerobot
uv run ruff check \
  src/lerobot/async_inference/configs.py \
  src/lerobot/async_inference/helpers.py \
  src/lerobot/async_inference/robot_client.py \
  src/lerobot/async_inference/policy_server.py \
  tests/async_inference/test_helpers.py \
  tests/async_inference/test_robot_client.py \
  tests/async_inference/test_policy_server.py

uv run pytest \
  tests/async_inference/test_helpers.py \
  tests/async_inference/test_robot_client.py \
  tests/async_inference/test_policy_server.py -q
```

若服务器使用已经激活的 Conda 环境，则可将 `uv run` 分别替换为
`python -m` 或环境内的 `ruff` 命令。不要为了本功能随意升级 PyTorch、
Transformers、NumPy 或 Hugging Face 依赖。

Mac 客户端应能列出以下入口：

```bash
python -m lerobot.async_inference.robot_client --help |
grep -E 'inference.type|inference.rtc|inference.queue_threshold'
```

预期包含：

```text
--inference.type {sync,rtc}
--inference.rtc.enabled
--inference.rtc.prefix_attention_schedule
--inference.rtc.max_guidance_weight
--inference.rtc.execution_horizon
--inference.rtc.debug
--inference.rtc.debug_maxlen
--inference.queue_threshold
```

## 5. 启动服务器

RTC 参数由客户端在 policy setup 握手中传给服务器，因此服务器启动命令本身不
增加 `--inference.*` 参数。可在 tmux 中运行：

```bash
tmux new -s lerobot-async-rtc
cd ~/lerobot
python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=1
```

按 `Ctrl-b`，再按 `d`，可以退出 tmux 而不停止服务器。重新进入：

```bash
tmux attach -t lerobot-async-rtc
```

SSH 隧道保持原配置：

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -o ServerAliveInterval=20 \
  -o ServerAliveCountMax=3 \
  wudi@192.168.18.191
```

## 6. SmolVLA 客户端示例

先让机械臂周围无障碍，并准备随时断电。首次验证建议把
`max_guidance_weight` 设为 `5.0`，确认运动连续后再逐步提高；rollout 的默认值
仍为 `10.0`。

```bash
cd /Users/diwu/lerobot
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.id=diwu_follower_arm \
  --robot.port=/dev/tty.usbmodem5B7B0096581 \
  --robot.cameras='{ camera1: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}' \
  --task="Grab the red block and place it in the box" \
  --policy_type=smolvla \
  --pretrained_name_or_path=wud24/smolvla_diwu_task_a_20260721_154902 \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --fps=30 \
  --inference.type=rtc \
  --inference.rtc.enabled=true \
  --inference.rtc.prefix_attention_schedule=LINEAR \
  --inference.rtc.max_guidance_weight=5.0 \
  --inference.rtc.execution_horizon=10 \
  --inference.rtc.debug=false \
  --inference.rtc.debug_maxlen=100 \
  --inference.queue_threshold=30 \
  --observation_image_compression=jpeg \
  --jpeg_quality=85 \
  --debug_visualize_queue_size=true
```

当前异步服务器允许列表中，确认具有 RTC processor 接口的策略包括
`smolvla`、`pi0` 和 `pi05`。ACT 不支持该 RTC 生成方式；使用 ACT 时应设置：

```bash
--inference.type=sync
```

## 7. 参数边界与交互

- `execution_horizon` 必须大于 0，并且不能大于 `actions_per_chunk`。
- `queue_threshold` 必须是大于等于 0 的整数。虽然代码不设置上限，但不建议
  大于或等于 `actions_per_chunk`，否则客户端可能刚收到新 chunk 就再次请求推理。
- RTC 模式下：
  - `--inference.queue_threshold` 控制何时请求新 chunk；
  - `--chunk_size_threshold` 保留为兼容参数，但不参与 RTC 门控；
  - `--aggregate_fn_name` 保留为兼容参数，但不再对 RTC 输出做第二次加权。
- `observation_image_compression` 与 RTC 相互独立。远程 1920×1080 图像仍推荐使用
  `jpeg`；相机的 `fourcc=MJPG` 不能替代 gRPC 层 JPEG 压缩。
- RTC 改善的是重叠 action chunk 的过渡连续性，不会消除相机采集、JPEG、
  网络或服务器推理本身的延迟。如果 action queue 仍频繁归零，应先解决吞吐问题。
- RTC 是运行时配置，不会修改或重新上传 Hugging Face 模型的 `config.json`。

## 8. 验收日志

服务器首次握手应出现：

```text
Inference mode: rtc
RTC configured | enabled=True | execution_horizon=10 | ...
```

第一次推理没有旧 chunk，`prefix_steps=0` 和 `inference_delay=0` 属于正常情况。
后续推理应出现类似：

```text
RTC context for observation #... | prefix_steps=30 | inference_delay=5 | execution_horizon=10
```

检查最新日志：

```bash
SERVER_LOG="$(ls -t logs/policy_server_*.log 2>/dev/null | head -1)"
grep -E 'Inference mode|RTC configured|RTC context|Running inference|Action chunk|Error' \
  "$SERVER_LOG" | tail -80
```

验收标准：

- `Inference mode: rtc`，而不是 `sync`。
- `RTC configured` 中的参数与 Mac 命令一致。
- 除第一帧外，`prefix_steps` 通常大于 0。
- `inference_delay` 应能反映当前端到端延迟；例如 30 Hz 下约 200 ms 通常是
  6～7 个控制步。
- action queue 的低谷不应反复降到 0。
- 不应出现“不支持 RTC”、prefix shape、pickle class 或 JPEG 解码错误。

如果后续始终是 `prefix_steps=0`，优先确认两端代码版本是否都包含 `32155b5d`。
如果 `inference_delay` 长期大于 `execution_horizon`，RTC 可利用的平滑过渡窗口很
有限，应先降低端到端延迟，或在安全测试后适度增大 `execution_horizon`。

## 9. 回退

最快、风险最低的功能回退是不改代码，停止当前客户端后改回：

```bash
--inference.type=sync
```

服务器会在下一次客户端握手时按同步模式重新加载策略。

若需要回退 Git，先停止客户端和服务器，再创建反向提交，不改写历史：

Mac：

```bash
cd /Users/diwu/lerobot
git switch codex/async-jpeg-transport
git revert 32155b5d
```

服务器：

```bash
cd ~/lerobot
git log --oneline -8
git revert <服务器上部署 RTC 后的实际提交号>
```

不要使用 `git reset --hard`。如果服务器一次性应用了 JPEG 与 RTC，且只希望关闭
RTC，优先使用 `--inference.type=sync` 或只 revert RTC 对应的服务器提交，保留
已验证的 JPEG 传输功能。
