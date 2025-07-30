import struct

import eventlet
import pyaudio
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-very-secret-key-4"
socketio = SocketIO(app, async_mode="eventlet")

# --- 配置区域 (修复环境后，先尝试默认设备) ---
SERVER_INPUT_DEVICE_INDEX = None  # 推荐先设为 None
SERVER_OUTPUT_DEVICE_INDEX = None  # 推荐先设为 None

# --- 音频配置 (保持 48000Hz) ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

p = pyaudio.PyAudio()

# 全局变量
server_audio_task = None
server_output_stream = None
# 诊断标志
server_audio_sent_log_printed = False
client_audio_received_log_printed = False


def send_server_audio():
    global server_audio_sent_log_printed
    print(f"后台任务：准备从设备索引 {SERVER_INPUT_DEVICE_INDEX or '默认'} 采集音频...")
    stream_in = None
    try:
        stream_in = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=SERVER_INPUT_DEVICE_INDEX,
        )
        print("✅ 服务器麦克风流已激活。")
        while True:
            eventlet.sleep(0)
            data = stream_in.read(CHUNK, exception_on_overflow=False)
            socketio.emit("audio_from_server", data)
            if not server_audio_sent_log_printed:
                print(f"✅ [诊断] 服务器正在发送第一批 {len(data)} 字节的音频数据...")
                server_audio_sent_log_printed = True
    except Exception as e:
        print(f"❌ 服务器音频采集任务失败: {e}")
    finally:
        if stream_in and stream_in.is_active():
            stream_in.close()
        print("后台任务：停止服务器音频采集。")


def get_server_output_stream():
    global server_output_stream
    if server_output_stream is None or not server_output_stream.is_active():
        print(f"创建新的服务器输出流，目标设备索引: {SERVER_OUTPUT_DEVICE_INDEX or '默认'}...")
        try:
            server_output_stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK,
                output_device_index=SERVER_OUTPUT_DEVICE_INDEX,
            )
            print("✅ 服务器扬声器流已准备就绪。")
        except Exception as e:
            print(f"❌ 创建服务器输出流失败: {e}")
            server_output_stream = None
    return server_output_stream


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    global server_audio_task
    print("客户端已连接")
    if server_audio_task is None:
        server_audio_task = socketio.start_background_task(target=send_server_audio)


@socketio.on("audio_from_client")
def handle_client_audio(json_data):
    global client_audio_received_log_printed
    float_array = json_data["audio_data"]
    int_array = [int(sample * 32767) for sample in float_array]
    byte_data = struct.pack("%sh" % len(int_array), *int_array)

    if not client_audio_received_log_printed:
        print(f"✅ [诊断] 服务器收到第一批来自客户端的 {len(byte_data)} 字节音频数据，准备播放...")
        client_audio_received_log_printed = True

    try:
        stream_out = get_server_output_stream()
        if stream_out:
            stream_out.write(byte_data)
    except Exception as e:
        print(f"服务器播放音频时出错: {e}")
        global server_output_stream
        if server_output_stream:
            server_output_stream.close()
            server_output_stream = None


# ... (disconnect and main block are the same)
if __name__ == "__main__":
    try:
        print("服务器正在启动，请在浏览器中访问 http://127.0.0.1:5000")
        socketio.run(app, host="0.0.0.0", port=5000)  # 使用 0.0.0.0 更通用
    finally:
        print("服务器正在关闭...")
        if server_output_stream:
            server_output_stream.stop_stream()
            server_output_stream.close()
        p.terminate()
        print("PyAudio 资源已清理。")
