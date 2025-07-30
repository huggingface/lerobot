#!/usr/bin/env python3

import time

from pynput import keyboard

print("键盘测试脚本启动...")
print("按任意键测试，按 ESC 退出")


def on_press(key):
    try:
        if hasattr(key, "char"):
            print(f"按键按下: {key.char}")
        else:
            print(f"特殊键按下: {key}")
    except AttributeError:
        print(f"特殊键按下: {key}")


def on_release(key):
    if key == keyboard.Key.esc:
        print("ESC 键按下，退出...")
        return False
    try:
        if hasattr(key, "char"):
            print(f"按键释放: {key.char}")
    except AttributeError:
        print(f"特殊键释放: {key}")


# 创建监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)

print("开始监听键盘...")
listener.start()

try:
    while listener.is_alive():
        time.sleep(0.1)
except KeyboardInterrupt:
    print("程序被中断")
finally:
    listener.stop()
    print("键盘监听已停止")
