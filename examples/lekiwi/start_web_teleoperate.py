#!/usr/bin/env python3
"""
LeKiwi 网页遥操作服务启动器

提供两个版本的网页服务：
1. 完整版 (web_teleoperate.py) - 使用Flask，功能更丰富
2. 简化版 (web_teleoperate_simple.py) - 使用标准库，依赖更少
"""

import os
import subprocess
import sys


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import flask
        import flask_cors

        print("✓ Flask 和 Flask-CORS 已安装")
        return True
    except ImportError:
        print("✗ Flask 或 Flask-CORS 未安装")
        return False


def install_dependencies():
    """安装依赖"""
    print("正在安装依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"])
        print("✓ 依赖安装成功")
        return True
    except subprocess.CalledProcessError:
        print("✗ 依赖安装失败")
        return False


def main():
    print("=== LeKiwi 网页遥操作服务启动器 ===")
    print()

    # 检查当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    print("请选择要启动的服务版本：")
    print("1. 完整版 (推荐) - 使用Flask，界面更美观，功能更丰富")
    print("2. 简化版 - 使用标准库，依赖更少，适合快速测试")
    print("3. 安装依赖 (如果选择完整版但依赖未安装)")
    print("4. 退出")
    print()

    while True:
        choice = input("请输入选择 (1-4): ").strip()

        if choice == "1":
            if check_dependencies():
                print("启动完整版网页服务...")
                subprocess.run([sys.executable, "web_teleoperate.py"])
                break
            else:
                print("Flask依赖未安装，请先选择选项3安装依赖")
                continue

        elif choice == "2":
            print("启动简化版网页服务...")
            subprocess.run([sys.executable, "web_teleoperate_simple.py"])
            break

        elif choice == "3":
            if install_dependencies():
                print("依赖安装完成，现在可以选择选项1启动完整版服务")
            else:
                print("依赖安装失败，请手动安装或选择选项2使用简化版")
            continue

        elif choice == "4":
            print("退出")
            break

        else:
            print("无效选择，请输入1-4")


if __name__ == "__main__":
    main()
