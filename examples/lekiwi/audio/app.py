import base64
import queue
import threading
import time
from collections import deque

import pygame
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 音频配置 - 可调整的延迟设置
AUDIO_CONFIG = {
    "buffer_size": 2056,  # pygame缓冲区大小 (越小延迟越低，但可能不稳定)
    "queue_size": 3,  # 音频队列大小 (越小延迟越低)
    "queue_timeout": 0.001,  # 队列超时时间 (秒)
    "channels": 2,  # 音频通道数
    "frequency": 44100,  # 采样率
    "enable_direct_play": True,  # 启用直接播放以减少延迟
    "enable_echo_cancellation": False,  # 禁用回声消除以减少延迟
}

# 初始化pygame音频 - 优化缓冲区设置以减少延迟
try:
    pygame.mixer.init(
        frequency=AUDIO_CONFIG["frequency"], size=-16, channels=1, buffer=AUDIO_CONFIG["buffer_size"]
    )
    AUDIO_AVAILABLE = True
    print("音频系统初始化成功")

    # 创建声音通道池
    pygame.mixer.set_num_channels(AUDIO_CONFIG["channels"])
except Exception as e:
    AUDIO_AVAILABLE = False
    print(f"音频系统初始化失败: {str(e)}")

# 存储客户端状态
clients = {}

# 音频播放控制 - 优化队列设置
audio_queue = queue.Queue(maxsize=AUDIO_CONFIG["queue_size"])
audio_thread_running = False

# 添加音频缓冲区管理
audio_buffers = deque(maxlen=3)  # 限制缓冲区大小
buffer_lock = threading.Lock()


# 音频播放线程 - 优化处理逻辑
def audio_playback_thread():
    """音频播放线程，持续处理音频队列"""
    global audio_thread_running

    print("音频播放线程启动")

    while audio_thread_running or not audio_queue.empty():
        try:
            # 从队列获取音频数据（使用配置的超时时间）
            audio_data = audio_queue.get(timeout=AUDIO_CONFIG["queue_timeout"])

            # 直接播放PCM数据，避免文件操作
            play_pcm_audio_nonblocking(audio_data)

            # 标记任务完成
            audio_queue.task_done()

        except queue.Empty:
            # 队列为空，继续等待
            pass
        except Exception as e:
            print(f"播放音频时出错: {str(e)}")

    print("音频播放线程停止")


def start_audio_thread():
    """启动音频播放线程"""
    global audio_thread_running

    if not audio_thread_running:
        audio_thread_running = True
        audio_thread = threading.Thread(target=audio_playback_thread)
        audio_thread.daemon = True
        audio_thread.start()
        return True
    return False


def stop_audio_thread():
    """停止音频播放线程"""
    global audio_thread_running
    audio_thread_running = False


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/audio/config", methods=["GET"])
def get_audio_config():
    """获取当前音频配置"""
    return {
        "config": AUDIO_CONFIG,
        "audio_available": AUDIO_AVAILABLE,
        "queue_size": audio_queue.qsize(),
        "queue_maxsize": audio_queue.maxsize,
    }


@app.route("/api/audio/config", methods=["POST"])
def update_audio_config():
    """更新音频配置"""
    try:
        data = request.get_json()
        if data:
            for key, value in data.items():
                if key in AUDIO_CONFIG:
                    AUDIO_CONFIG[key] = value

            # 重新初始化音频系统
            if AUDIO_AVAILABLE:
                pygame.mixer.quit()
                pygame.mixer.init(
                    frequency=AUDIO_CONFIG["frequency"],
                    size=-16,
                    channels=1,
                    buffer=AUDIO_CONFIG["buffer_size"],
                )
                pygame.mixer.set_num_channels(AUDIO_CONFIG["channels"])

            return {"status": "success", "config": AUDIO_CONFIG}
    except Exception as e:
        return {"status": "error", "message": str(e)}, 400


@app.route("/api/audio/stats", methods=["GET"])
def get_audio_stats():
    """获取音频统计信息"""
    return {
        "queue_size": audio_queue.qsize(),
        "queue_maxsize": audio_queue.maxsize,
        "thread_running": audio_thread_running,
        "clients_connected": len(clients),
    }


@socketio.on("connect")
def handle_connect():
    clients[request.sid] = {"active": True}
    print(f"Client connected: {request.sid}")

    # 确保音频线程已启动
    if AUDIO_AVAILABLE:
        start_audio_thread()


@socketio.on("disconnect")
def handle_disconnect():
    clients.pop(request.sid, None)
    print(f"Client disconnected: {request.sid}")


@socketio.on("audio_data")
def handle_audio_data(data):
    """处理从前端收到的音频数据 - 优化处理流程"""
    try:
        # 解码 base64 音频数据
        audio_bytes = base64.b64decode(data["audio"])

        # 根据配置决定是否进行音频处理
        if AUDIO_CONFIG["enable_echo_cancellation"]:
            processed_audio = echo_cancellation(audio_bytes, request.sid)
        else:
            processed_audio = audio_bytes  # 跳过回声消除以降低延迟

        # 将音频数据放入队列
        if AUDIO_AVAILABLE:
            try:
                # 如果队列满了，清空旧数据
                if audio_queue.full():
                    try:
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    except queue.Empty:
                        pass

                # 根据配置决定是否尝试直接播放
                if AUDIO_CONFIG["enable_direct_play"]:
                    if not try_direct_play(processed_audio):
                        audio_queue.put_nowait(processed_audio)
                else:
                    audio_queue.put_nowait(processed_audio)

            except queue.Full:
                print("音频队列已满，丢弃部分数据")

    except Exception as e:
        print(f"Error processing audio: {str(e)}")


def try_direct_play(audio_data):
    """尝试直接播放音频，避免队列延迟"""
    try:
        # 检查是否有可用通道
        channel = pygame.mixer.find_channel()
        if channel and not channel.get_busy():
            # 直接播放
            sound = pygame.mixer.Sound(buffer=bytes(audio_data))
            channel.play(sound)
            return True
    except Exception as e:
        print(f"直接播放失败: {str(e)}")
    return False


def echo_cancellation(audio_data, sid):
    """简单的回声消除示例（实际应用需要更复杂算法）"""
    # 这里只是一个示例 - 实际应该使用专业算法如AEC
    return audio_data


def play_pcm_audio_nonblocking(pcm_data):
    """非阻塞播放PCM音频数据，减少延迟"""
    try:
        # 创建Sound对象
        sound = pygame.mixer.Sound(buffer=bytes(pcm_data))

        # 播放音频（使用下一个可用通道）
        channel = pygame.mixer.find_channel()
        if channel:
            channel.play(sound)
            # 不等待播放完成，立即返回
        else:
            print("没有可用音频通道")

    except Exception as e:
        print(f"播放PCM音频时出错: {str(e)}")


def play_pcm_audio(pcm_data):
    """直接播放PCM音频数据，避免文件操作"""
    try:
        # 创建Sound对象
        sound = pygame.mixer.Sound(buffer=bytes(pcm_data))

        # 播放音频（使用下一个可用通道）
        channel = pygame.mixer.find_channel()
        if channel:
            channel.play(sound)
        else:
            print("没有可用音频通道")

        # 等待播放完成（非阻塞方式）
        while channel and channel.get_busy():
            time.sleep(0.01)

    except Exception as e:
        print(f"播放PCM音频时出错: {str(e)}")


def play_audio_file(file_path):
    """播放音频文件（备用方法）"""
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # 等待播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    except Exception as e:
        print(f"播放音频文件时出错: {str(e)}")


@socketio.on("stop_audio")
def handle_stop_audio():
    """停止音频播放"""
    try:
        if AUDIO_AVAILABLE:
            # 清空队列
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                    audio_queue.task_done()
                except queue.Empty:
                    break

            # 停止所有音频
            pygame.mixer.stop()
            print("音频播放已停止")
    except Exception as e:
        print(f"停止音频播放时出错: {str(e)}")


if __name__ == "__main__":
    # 启动时启动音频线程
    if AUDIO_AVAILABLE:
        start_audio_thread()

    socketio.run(app, host="0.0.0.0", port=5555, debug=True)
