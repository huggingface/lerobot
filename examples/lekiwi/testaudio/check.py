import pyaudio

p = pyaudio.PyAudio()

print("---------- 可用的音频设备列表 ----------")
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    device_info = p.get_device_info_by_index(i)
    is_input = device_info.get('maxInputChannels') > 0
    is_output = device_info.get('maxOutputChannels') > 0
    
    print(f"\n--- 设备索引 (Index): {i} ---")
    print(f"  名称: {device_info.get('name')}")
    print(f"  是否为输入设备 (麦克风): {'是' if is_input else '否'}")
    print(f"  是否为输出设备 (扬声器): {'是' if is_output else '否'}")
    print(f"  默认采样率: {int(device_info.get('defaultSampleRate'))} Hz")

print("\n----------------------------------------")
print("请根据上面的列表，找到您想用的麦克风和扬声器的【设备索引】。")
print("通常，它们的名称会包含 'USB', 'Headset', 'Webcam' 或 'default' 等字样。")

p.terminate()