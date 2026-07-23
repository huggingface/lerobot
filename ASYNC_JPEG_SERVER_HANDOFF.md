# LeRobot 异步推理 JPEG 传输：服务器端移交说明

本文档交给服务器端 Codex Agent 使用。目标是让 Mac 客户端发送真正经过 JPEG
编码的相机图像，并让服务器在进入现有 policy 预处理流程前还原为 RGB
`numpy.ndarray`。客户端还把 JPEG 编码、序列化和 gRPC 上传移到了独立后台线程。

## 1. 已完成版本

- 本地仓库：`/Users/diwu/lerobot`
- 工作分支：`codex/async-jpeg-transport`
- 修改前备份提交：`1086b7b1`
- JPEG 功能提交：`119cb12d`
- 上游基线：`a9879e69`

`1086b7b1` 保存了修改前已有的本地代码状态，包括
`src/lerobot/motors/motors_bus.py` 中的电机重试次数设置。`119cb12d`
只包含 JPEG 异步传输功能及其测试。

不要在服务器尚未更新时启用客户端 JPEG。新服务端兼容旧的未压缩客户端；
旧服务端无法反序列化新的 `EncodedImage`，因此部署顺序必须是：

1. 备份并更新服务器。
2. 验证服务器测试。
3. 启动新服务器。
4. 最后启动启用了 JPEG 的 Mac 客户端。

## 2. 功能设计

传输链路如下：

```text
Mac 相机 RGB uint8(H,W,3)
  -> 后台线程 RGB 转 BGR
  -> cv2.imencode(".jpg", quality=85)
  -> EncodedImage(data, codec, original_shape)
  -> pickle + gRPC
  -> 服务器 pickle.loads
  -> cv2.imdecode
  -> BGR 转 RGB
  -> 原有 LeRobot policy 预处理与推理
```

这与相机参数中的 `fourcc: "MJPG"` 不同。FourCC 只影响相机到 Mac 的采集格式，
不能减少原有 Python `numpy.ndarray` 在 gRPC 中的传输体积。本改造压缩的是实际
跨网络传输的 observation。

客户端同时加入：

- 容量为 1 的 observation 队列，拥塞时保留最新的尚未上传帧。
- 独立 `observation_sender` 线程，负责 JPEG 编码、pickle 和同步 gRPC 上传。
- 单个 action 请求在途控制，避免连续上传造成服务器推理堆积。
- 每个按 action queue 阈值触发的请求设置 `must_go=True`，避免现有的仅关节状态
  相似度过滤把补充 action chunk 的请求丢弃。

## 3. 服务器端必须包含的修改

推荐让 Mac 与服务器使用完整的 `119cb12d`，这样两端协议和测试完全一致。
服务器运行时真正必需的代码是：

### `src/lerobot/async_inference/helpers.py`

- 新增不可变数据类 `EncodedImage`：
  - `data: bytes`
  - `codec: str`
  - `original_shape: tuple[int, ...]`
- 新增 `decode_encoded_image()`：
  - 使用 `cv2.imdecode`。
  - 明确执行 BGR 到 RGB 的转换。
  - 检查解码后 shape 是否等于 `original_shape`。
- 新增 `decode_raw_observation_images()`：
  - 只解码值类型为 `EncodedImage` 的字段。
  - 未压缩的旧 observation 保持不变，因此具备向后兼容性。

### `src/lerobot/async_inference/policy_server.py`

在 `SendObservations()` 中：

1. 接收完整字节流。
2. 执行 `pickle.loads()`。
3. 立即调用 `decode_raw_observation_images()`。
4. 把解码后的 RGB observation 写回 `TimedObservation.observation`。
5. 再执行原有 FPS、相似度检查、队列和推理逻辑。

日志需要分别输出：

- `Payload size`
- `Receive time`
- `Deserialization time`
- `JPEG decode time`
- 解码图像数量

不需要修改 protobuf，也没有新增第三方依赖；当前 LeRobot 已依赖 OpenCV 和
NumPy。

## 4. 推荐部署方法：传输 Git patch

先在 Mac 执行：

```bash
cd /Users/diwu/lerobot
git status --short --branch
git format-patch -1 119cb12d --stdout > /tmp/lerobot-async-jpeg.patch
scp /tmp/lerobot-async-jpeg.patch wudi@192.168.18.191:~/
```

服务器端 Codex Agent 必须先检查当前仓库，保留所有既有修改，不要把模型、
日志、缓存或数据集误加入 Git：

```bash
cd ~/lerobot
git status --short --branch
git log --oneline -3
```

若存在未提交代码，先创建备份分支，并在检查 diff 后只暂存相关源码：

```bash
git switch -c codex/server-before-async-jpeg
git diff --check
git add <逐个列出需要备份的源码文件>
git commit -m "chore: snapshot server before async JPEG transport"
```

然后创建实施分支并应用补丁：

```bash
git switch -c codex/async-jpeg-server
git apply --check ~/lerobot-async-jpeg.patch
git am ~/lerobot-async-jpeg.patch
git log --oneline -3
```

如果 `git apply --check` 或 `git am` 报冲突，不要用 `git reset --hard`，也不要覆盖
服务器既有修改。应由服务器端 Codex Agent 对照第 3 节逐个合并冲突，并重新运行
测试。若没有传输 patch，也可以由服务器端 Agent 按本文档实施相同改造。

应用后记录服务器实际提交号：

```bash
git rev-parse HEAD
```

服务器父提交可能与 Mac 不同，因此通过 patch 生成的服务器提交号不一定仍是
`119cb12d`。

## 5. 环境与测试

先确认服务器实际导入的是当前仓库，而不是另一个 `site-packages` 副本：

```bash
cd ~/lerobot
python -c 'import lerobot; print(lerobot.__file__)'
```

输出应指向 `~/lerobot/src/lerobot/`。本功能没有新增依赖，如果输出不是当前仓库，
可在服务器已有的 LeRobot 环境中执行：

```bash
python -m pip install -e . --no-deps
```

使用 `--no-deps` 是为了避免无关依赖被升级。若服务器原本使用 `uv` 管理环境，
则继续使用项目既有的 `uv` 工作流。

建议验证：

```bash
ruff check \
  src/lerobot/async_inference/configs.py \
  src/lerobot/async_inference/helpers.py \
  src/lerobot/async_inference/robot_client.py \
  src/lerobot/async_inference/policy_server.py \
  tests/async_inference/test_helpers.py \
  tests/async_inference/test_robot_client.py \
  tests/async_inference/test_policy_server.py \
  tests/async_inference/test_e2e.py

pytest \
  tests/async_inference/test_helpers.py \
  tests/async_inference/test_policy_server.py \
  tests/async_inference/test_robot_client.py \
  tests/async_inference/test_e2e.py -q
```

Mac 端已经验证：

- Ruff lint：通过
- Ruff format：通过
- 上述测试：`56 passed`
- 1920×1080 合成测试图：原始约 5.933 MiB，JPEG 约 0.066 MiB；编码约
  10.3 ms，解码约 6.4 ms。真实相机画面的压缩率会随纹理和噪声变化，不能把
  该合成结果当作固定承诺。

## 6. 启动服务器

服务器命令不需要增加 JPEG 参数，解码器会自动识别 `EncodedImage`。可在 tmux
中运行：

```bash
tmux new -s lerobot-async-jpeg
cd ~/lerobot
python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=1
```

按 `Ctrl-b`，再按 `d` 可退出 tmux 而不停止服务。重新进入：

```bash
tmux attach -t lerobot-async-jpeg
```

如果 Mac 通过 SSH 隧道连接，隧道命令保持不变：

```bash
ssh -N \
  -L 8080:127.0.0.1:8080 \
  -o ServerAliveInterval=20 \
  -o ServerAliveCountMax=3 \
  wudi@192.168.18.191
```

## 7. Mac 客户端命令

更新服务器并确认其正在监听后，在 Mac 端运行：

```bash
cd /Users/diwu/lerobot
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.id=diwu_follower_arm \
  --robot.port=/dev/tty.usbmodem5B7B0096581 \
  --robot.cameras='{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}' \
  --task="Grab the red block and place it in the box" \
  --policy_type=act \
  --pretrained_name_or_path=wud24/act_diwu_task_a_20260721_154902 \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --fps=30 \
  --observation_image_compression=jpeg \
  --jpeg_quality=85 \
  --debug_visualize_queue_size=true
```

建议先在机械臂周围无障碍、随时可断电的条件下做短时验证。若 1920×1080 的相机
采集本身仍拖慢控制循环，可以先把相机设为 640×480；JPEG 只能降低网络上传量，
不能消除摄像头驱动读取高分辨率帧所需的时间。

## 8. 验收日志

Mac 客户端 debug 日志应出现：

```text
Observation sender thread starting
JPEG encoded 1 image(s) | Raw: ... MiB | Encoded: ... MiB | Reduction: ...x
```

服务器 debug 日志应出现：

```text
Payload size: ... MiB
JPEG decode time: ...s (1 image(s))
```

可以使用系统自带的 `grep`，不依赖 `rg`：

```bash
SERVER_LOG="$(ls -t logs/policy_server_*.log 2>/dev/null | head -1)"
grep -E 'Payload size|JPEG decode time|Running inference|Action chunk' "$SERVER_LOG" | tail -50
```

验收要点：

- `JPEG decode time` 后是 `(1 image(s))`。
- `Payload size` 明显小于此前 1920×1080 原始帧约 5.933 MiB。
- 服务端持续生成 action chunk，客户端 action queue 不归零或只偶发瞬时归零。
- 日志中的 observation 编号按约 24～25 个环境 step 跳变是
  `actions_per_chunk=50`、`chunk_size_threshold=0.5` 的预期门控结果，不代表丢包。
- 不应出现 `Can't get attribute 'EncodedImage'`、JPEG shape 不匹配或 JPEG 解码失败。

## 9. 回退

最快的无代码回退是先让新客户端停止压缩：

```bash
--observation_image_compression=none
```

该参数会恢复原始 `numpy.ndarray` 传输。若需要回退 Git，推荐生成反向提交，
不要改写历史：

Mac：

```bash
cd /Users/diwu/lerobot
git switch codex/async-jpeg-transport
git revert 119cb12d
```

服务器：

```bash
cd ~/lerobot
git log --oneline -5
git revert <服务器上应用 JPEG 功能后的实际提交号>
```

安全回退顺序是先把客户端设为 `none` 或停止客户端，再回退服务器。修改前本地
快照仍可通过 `1086b7b1` 找到。
