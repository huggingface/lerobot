# 实时音频通话系统

这是一个支持双向实时音频传输的Web应用，包含服务器端录音功能。

## 功能特性

- **双向音频传输**: 客户端和服务器之间可以实时传输音频
- **服务器录音**: 服务器端可以录音并实时传输到web客户端
- **低延迟播放**: 优化的音频缓冲区设置，减少播放延迟
- **实时状态监控**: 显示连接状态和录音状态
- **可配置参数**: 支持动态调整音频配置参数

## 安装依赖

```bash
pip install -r requirements.txt
```

### 系统依赖

在Ubuntu/Debian系统上，可能需要安装以下系统包：

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev python3-pyaudio
```

## 运行服务器

```bash
python app.py
```

服务器将在 `http://localhost:5555` 启动。

## 使用方法

1. 打开浏览器访问 `http://localhost:5555`
2. 点击"开始通话"按钮开始客户端录音
3. 点击"启动服务器录音"按钮开始服务器端录音
4. 服务器录音的音频将实时传输到web客户端播放

## API接口

### 获取音频配置

```
GET /api/audio/config
```

### 更新音频配置

```
POST /api/audio/config
Content-Type: application/json

{
    "buffer_size": 512,
    "queue_size": 3,
    "queue_timeout": 0.001
}
```

### 获取音频统计

```
GET /api/audio/stats
```

### 启动服务器录音

```
POST /api/audio/record/start
```

### 停止服务器录音

```
POST /api/audio/record/stop
```

## WebSocket事件

### 客户端发送事件

- `audio_data`: 发送音频数据到服务器
- `start_recording`: 请求启动服务器录音
- `stop_recording`: 请求停止服务器录音
- `stop_audio`: 请求停止服务器音频播放

### 服务器发送事件

- `server_audio`: 服务器录音数据
- `recording_status`: 录音状态更新
- `recording_error`: 录音错误信息

## 配置参数

可以在 `AUDIO_CONFIG` 中调整以下参数：

- `buffer_size`: pygame缓冲区大小 (越小延迟越低，但可能不稳定)
- `queue_size`: 音频队列大小 (越小延迟越低)
- `queue_timeout`: 队列超时时间 (秒)
- `channels`: 音频通道数
- `frequency`: 采样率
- `enable_direct_play`: 启用直接播放以减少延迟
- `enable_echo_cancellation`: 启用回声消除
- `record_chunk`: 录音块大小
- `record_format`: 录音格式
- `record_channels`: 录音通道数
- `record_rate`: 录音采样率

## 故障排除

### 音频播放问题

- 检查系统音频设备是否正常工作
- 调整 `buffer_size` 参数
- 确保pygame正确初始化

### 录音问题

- 检查麦克风权限
- 确保pyaudio正确安装
- 检查系统音频输入设备

### 网络延迟问题

- 调整 `queue_size` 和 `queue_timeout` 参数
- 启用 `enable_direct_play` 选项
- 检查网络连接质量

## 注意事项

- 服务器录音功能需要系统有可用的音频输入设备
- 在Docker容器中运行时，需要正确配置音频设备
- 高采样率和大缓冲区会增加延迟，但提高稳定性
- 建议在生产环境中使用更专业的音频处理库
