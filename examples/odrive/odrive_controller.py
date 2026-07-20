#!/usr/bin/env python3
"""
ODrive USB 控制器
完整版本 - 可直接使用
"""
import odrive
from odrive.enums import *
import time
import sys
import pygame

class ODriveController:
    def __init__(self, wheel_diameter_m=0.165):
        """
        初始化ODrive控制器
        
        Args:
            wheel_diameter_m: 轮子直径(米), 默认165mm
        """
        print("=" * 60)
        print("初始化ODrive控制器")
        print("=" * 60)
        
        # 连接设备
        print("\n[1/5] 正在连接ODrive...")
        try:
            self.odrv = odrive.find_any(timeout=30)
            print(f"      ✓ 已连接: {self.odrv.serial_number}")
            print(f"      ✓ 电压: {self.odrv.vbus_voltage:.1f}V")
        except Exception as e:
            print(f"      ✗ 连接失败: {e}")
            sys.exit(1)
        
        # 轮子参数
        self.wheel_diameter = wheel_diameter_m
        self.wheel_circumference = 3.14159265359 * wheel_diameter_m
        
        # 轴引用
        self.left = self.odrv.axis0
        self.right = self.odrv.axis1
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """完整初始化流程"""
        # 清除错误
        print("\n[2/5] 清除错误...")
        self.left.clear_errors()
        self.right.clear_errors()
        time.sleep(0.2)
        print("      ✓ 完成")
        
        # 设置控制模式
        print("\n[3/5] 配置控制模式...")
        self.left.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.right.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.left.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        self.right.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        print("      ✓ 速度控制模式已设置")
        
        # 禁用看门狗
        print("\n[4/5] 禁用看门狗...")
        self.left.config.enable_watchdog = False
        self.right.config.enable_watchdog = False
        print("      ✓ 看门狗已禁用")
        
        # 启动闭环控制
        print("\n[5/5] 启动闭环控制...")
        self.left.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.right.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        time.sleep(0.5)
        
        # 检查错误
        if self.left.error != 0:
            print(f"      ⚠️  左轮错误: {hex(self.left.error)}")
            print("      提示: 可能需要校准电机，参考ODrive文档")
        if self.right.error != 0:
            print(f"      ⚠️  右轮错误: {hex(self.right.error)}")
            print("      提示: 可能需要校准电机，参考ODrive文档")
        
        if self.left.error == 0 and self.right.error == 0:
            print("      ✓ 闭环控制已启动")
            print("\n" + "=" * 60)
            print("✅ 初始化完成！准备就绪")
            print("=" * 60 + "\n")
    
    # ========== 速度控制 ==========
    
    def set_velocity(self, left_mps, right_mps):
        """
        设置左右轮速度
        
        Args:
            left_mps: 左轮速度(m/s)
            right_mps: 右轮速度(m/s)
        """
        left_rps = left_mps / self.wheel_circumference
        right_rps = right_mps / self.wheel_circumference
        self.left.controller.input_vel = left_rps
        self.right.controller.input_vel = right_rps
    
    def forward(self, speed=0.3):
        """前进"""
        self.set_velocity(speed, speed)
    
    def backward(self, speed=0.3):
        """后退"""
        self.set_velocity(-speed, -speed)
    
    def turn_left(self, speed=0.2):
        """左转（原地）"""
        self.set_velocity(-speed, speed)
    
    def turn_right(self, speed=0.2):
        """右转（原地）"""
        self.set_velocity(speed, -speed)
    
    def stop(self):
        """停止"""
        self.set_velocity(0, 0)
    
    # ========== 状态查询 ==========
    
    def get_velocity_rpm(self):
        """获取当前速度(RPM)"""
        left_rpm = self.left.encoder.vel_estimate * 60
        right_rpm = self.right.encoder.vel_estimate * 60
        return left_rpm, right_rpm
    
    def get_position(self):
        """获取位置(圈数)"""
        return self.left.encoder.pos_estimate, self.right.encoder.pos_estimate
    
    def get_voltage(self):
        """获取总线电压"""
        return self.odrv.vbus_voltage
    
    def get_current(self):
        """获取电流"""
        left_current = self.left.motor.current_control.Iq_measured
        right_current = self.right.motor.current_control.Iq_measured
        return left_current, right_current
    
    def print_status(self):
        """打印当前状态"""
        left_rpm, right_rpm = self.get_velocity_rpm()
        left_pos, right_pos = self.get_position()
        voltage = self.get_voltage()
        left_current, right_current = self.get_current()
        
        print("\n" + "─" * 60)
        print(f"电压: {voltage:5.1f}V")
        print(f"速度: 左={left_rpm:6.1f} RPM  |  右={right_rpm:6.1f} RPM")
        print(f"位置: 左={left_pos:7.2f} 圈  |  右={right_pos:7.2f} 圈")
        print(f"电流: 左={left_current:5.2f} A   |  右={right_current:5.2f} A")
        print("─" * 60)
    
    # ========== 安全关闭 ==========
    
    def shutdown(self):
        """安全关闭"""
        print("\n正在安全关闭...")
        self.stop()
        time.sleep(0.5)
        self.left.requested_state = AXIS_STATE_IDLE
        self.right.requested_state = AXIS_STATE_IDLE
        print("✅ 已安全关闭")


# ========== 测试程序 ==========

def basic_test():
    """基础运动测试"""
    try:
        # 初始化
        robot = ODriveController()
        
        print("开始基础运动测试...\n")
        
        # 测试序列
        tests = [
            ("前进", lambda: robot.forward(0.25), 2),
            ("停止", lambda: robot.stop(), 1),
            ("后退", lambda: robot.backward(0.25), 2),
            ("停止", lambda: robot.stop(), 1),
            ("左转", lambda: robot.turn_left(0.2), 1.5),
            ("停止", lambda: robot.stop(), 1),
            ("右转", lambda: robot.turn_right(0.2), 1.5),
        ]
        
        for i, (name, action, duration) in enumerate(tests, 1):
            print(f"[{i}/{len(tests)}] {name} ({duration}秒)...")
            action()
            time.sleep(duration)
        
        # 最终停止
        robot.stop()
        
        # 显示最终状态
        print("\n✅ 测试完成！")
        robot.print_status()
        
        # 关闭
        robot.shutdown()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        robot.stop()
        robot.shutdown()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def interactive_control():
    """交互式控制"""
    try:
        robot = ODriveController()
        
        print("\n" + "=" * 60)
        print("交互式控制")
        print("=" * 60)
        print("命令:")
        print("  w - 前进")
        print("  s - 后退")
        print("  a - 左转")
        print("  d - 右转")
        print("  x - 停止")
        print("  p - 显示状态")
        print("  q - 退出")
        print("=" * 60 + "\n")
        
        while True:
            cmd = input(">>> ").strip().lower()
            
            if cmd == 'w':
                speed = input("  速度(m/s, 默认0.3): ").strip()
                speed = float(speed) if speed else 0.3
                robot.forward(speed)
                print("  ↑ 前进")
            elif cmd == 's':
                speed = input("  速度(m/s, 默认0.3): ").strip()
                speed = float(speed) if speed else 0.3
                robot.backward(speed)
                print("  ↓ 后退")
            elif cmd == 'a':
                speed = input("  速度(m/s, 默认0.2): ").strip()
                speed = float(speed) if speed else 0.2
                robot.turn_left(speed)
                print("  ← 左转")
            elif cmd == 'd':
                speed = input("  速度(m/s, 默认0.2): ").strip()
                speed = float(speed) if speed else 0.2
                robot.turn_right(speed)
                print("  → 右转")
            elif cmd == 'x':
                robot.stop()
                print("  ■ 停止")
            elif cmd == 'p':
                robot.print_status()
            elif cmd == 'q':
                print("\n退出...")
                break
            else:
                print("  ⚠️  未知命令")
        
        robot.shutdown()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        robot.stop()
        robot.shutdown()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


def pygame_keyboard_control():
    """使用pygame键盘实时控制"""
    robot = None
    try:
        robot = ODriveController()
        
        # 初始化pygame
        pygame.init()
        # 创建一个隐藏窗口来接收键盘事件
        screen = pygame.display.set_mode((1, 1))
        pygame.display.set_caption("ODrive 键盘控制 - 按ESC退出")
        
        # 速度参数
        base_speed = 0.3  # 基础速度 (m/s)
        turn_speed = 0.2  # 转向速度 (m/s)
        speed_step = 0.05  # 速度调节步长
        
        print("\n" + "=" * 60)
        print("Pygame 键盘控制")
        print("=" * 60)
        print("控制说明:")
        print("  W / ↑ - 前进")
        print("  S / ↓ - 后退")
        print("  A / ← - 左转")
        print("  D / → - 右转")
        print("  +/- - 增加/减少速度")
        print("  0 - 重置速度")
        print("  Space - 停止")
        print("  P - 显示状态")
        print("  ESC / Q - 退出")
        print("=" * 60)
        print(f"\n当前速度: {base_speed:.2f} m/s")
        print("按任意键开始控制...\n")
        
        clock = pygame.time.Clock()
        running = True
        last_status_time = 0
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        base_speed = min(4.0, base_speed + speed_step)
                        print(f"速度增加: {base_speed:.2f} m/s")
                    elif event.key == pygame.K_MINUS:
                        base_speed = max(0.05, base_speed - speed_step)
                        print(f"速度减少: {base_speed:.2f} m/s")
                    elif event.key == pygame.K_0:
                        base_speed = 0.3
                        print(f"速度重置: {base_speed:.2f} m/s")
                    elif event.key == pygame.K_p:
                        robot.print_status()
            
            # 获取当前按键状态
            keys = pygame.key.get_pressed()
            
            # 根据按键状态控制机器人
            forward = keys[pygame.K_w] or keys[pygame.K_UP]
            backward = keys[pygame.K_s] or keys[pygame.K_DOWN]
            left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
            stop = keys[pygame.K_SPACE]
            
            # 计算左右轮速度
            left_speed = 0.0
            right_speed = 0.0
            
            if stop:
                left_speed = 0.0
                right_speed = 0.0
            elif forward and backward:
                # 同时按下前进和后退，停止
                left_speed = 0.0
                right_speed = 0.0
            elif forward:
                if left:
                    # 前进+左转 = 左前（左轮更慢）
                    left_speed = -base_speed * 0.5
                    right_speed = base_speed
                elif right:
                    # 前进+右转 = 右前（右轮更慢）
                    left_speed = -base_speed
                    right_speed = base_speed * 0.5
                else:
                    # 纯前进（左轮用负数，因为电机方向接反）
                    left_speed = -base_speed
                    right_speed = base_speed
            elif backward:
                if left:
                    # 后退+左转 = 左后（左轮更慢）
                    left_speed = base_speed * 0.5
                    right_speed = -base_speed
                elif right:
                    # 后退+右转 = 右后（右轮更慢）
                    left_speed = base_speed
                    right_speed = -base_speed * 0.5
                else:
                    # 纯后退（左轮用正数，右轮用负数）
                    left_speed = base_speed
                    right_speed = -base_speed
            elif left:
                # 纯左转（原地）- 左轮速度取负数，右轮不变
                left_speed = turn_speed
                right_speed = turn_speed
            elif right:
                # 纯右转（原地）- 右轮速度取负数，左轮不变
                left_speed = -turn_speed
                right_speed = -turn_speed
            
            # 设置速度
            robot.set_velocity(left_speed, right_speed)
            
            # 定期显示状态（每秒一次）
            current_time = time.time()
            if current_time - last_status_time >= 1.0:
                left_rpm, right_rpm = robot.get_velocity_rpm()
                print(f"\r速度: {base_speed:.2f} m/s | 左轮: {left_rpm:6.1f} RPM | 右轮: {right_rpm:6.1f} RPM | 电压: {robot.get_voltage():.1f}V", end="", flush=True)
                last_status_time = current_time
            
            # 控制帧率
            clock.tick(30)  # 30 FPS
        
        print("\n\n正在退出...")
        if robot:
            robot.shutdown()
        pygame.quit()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        if robot:
            robot.stop()
            robot.shutdown()
        pygame.quit()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        if robot:
            robot.stop()
            robot.shutdown()
        pygame.quit()


# ========== 主程序 ==========

def main():
    print("\nODrive USB 控制程序")
    print("=" * 60)
    print("选择模式:")
    print("  1 - 基础运动测试（自动）")
    print("  2 - 交互式控制（手动输入命令）")
    print("  3 - Pygame 键盘实时控制（推荐）")
    print("=" * 60)
    
    choice = input("\n选择 (1, 2 或 3): ").strip()
    
    if choice == '1':
        basic_test()
    elif choice == '2':
        interactive_control()
    elif choice == '3':
        pygame_keyboard_control()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()