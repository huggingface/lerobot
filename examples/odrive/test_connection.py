#!/usr/bin/env python3
"""
测试ODrive连接
"""
import odrive
import sys
import time

print("=" * 60)
print("ODrive 连接测试")
print("=" * 60)

print("\n请确认：")
print("  ✓ Micro USB已连接到电脑")
print("  ✓ ODrive有独立电源供电（18-24V）")
print("  ✓ 电压指示灯亮起")
print()

input("确认后按Enter继续...")

try:
    print("\n正在搜索ODrive设备...")
    print("（首次连接可能需要10-20秒）\n")
    
    odrv = odrive.find_any(timeout=30)
    
    print("✅ 成功连接到ODrive！\n")
    print(f"设备信息:")
    print(f"  序列号: {odrv.serial_number}")
    print(f"  硬件版本: v{odrv.hw_version_major}.{odrv.hw_version_minor}")
    print(f"  固件版本: v{odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
    print(f"  总线电压: {odrv.vbus_voltage:.2f}V")
    
    # 检查轴状态
    print(f"\n轴0 (左轮):")
    print(f"  当前状态: {odrv.axis0.current_state}")
    print(f"  错误码: {hex(odrv.axis0.error)}")
    if odrv.axis0.error != 0:
        print(f"  ⚠️  有错误，需要校准或配置")
    
    print(f"\n轴1 (右轮):")
    print(f"  当前状态: {odrv.axis1.current_state}")
    print(f"  错误码: {hex(odrv.axis1.error)}")
    if odrv.axis1.error != 0:
        print(f"  ⚠️  有错误，需要校准或配置")
    
    print("\n✅ 连接测试成功！")
    print("\n下一步: 运行 python odrive_controller.py 开始控制")
    
except TimeoutError:
    print("❌ 连接超时")
    print("\n故障排查：")
    print("1. 检查USB线连接")
    print("2. 确认ODrive电源已接通")
    print("3. 运行: lsusb | grep 1209")
    print("4. 可能需要重新登录系统使权限生效")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("\n可能需要：")
    print("1. 重新登录系统（使dialout组权限生效）")
    print("2. 或在新终端运行: newgrp dialout")
    sys.exit(1)