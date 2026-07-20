#!/usr/bin/env python3
import math
import time

import odrive
import pygame
from odrive.enums import (
    AXIS_STATE_CLOSED_LOOP_CONTROL,
    AXIS_STATE_IDLE,
    CONTROL_MODE_VELOCITY_CONTROL,
    INPUT_MODE_PASSTHROUGH,
    INPUT_MODE_VEL_RAMP
)


class ODriveKeyboardTester:
    def __init__(self, wheel_diameter_m=0.165):
        self.wheel_circumference = math.pi * wheel_diameter_m
        self.left_direction = -1.0
        self.right_direction = 1.0
        self.base_speed = 0.3
        self.turn_speed = 0.2
        self.speed_step = 0.05
        self.max_speed = 1.0

        print("=" * 60)
        print("ODrive keyboard test")
        print("=" * 60)
        print("Connecting to ODrive...")

        self.odrv = odrive.find_any(timeout=30)
        self.left = self.odrv.axis0
        self.right = self.odrv.axis1

        print(f"Connected: {self.odrv.serial_number}")
        print(f"Bus voltage: {self.odrv.vbus_voltage:.1f} V")

        self._initialize()

    def _initialize(self):
        self.left.clear_errors()
        self.right.clear_errors()
        time.sleep(0.2)

        self.left.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.right.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        self.left.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.right.controller.config.input_mode = INPUT_MODE_VEL_RAMP
        self.left.controller.config.vel_ramp_rate = 1.0
        self.right.controller.config.vel_ramp_rate = 1.0
        self.left.config.enable_watchdog = False
        self.right.config.enable_watchdog = False

        self.left.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.right.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        time.sleep(0.5)

        print("Closed loop control requested")
        if self.left.error:
            print(f"Axis0 error: {hex(self.left.error)}")
        if self.right.error:
            print(f"Axis1 error: {hex(self.right.error)}")

    def set_velocity(self, left_mps, right_mps):
        self.left.controller.input_vel = left_mps * self.left_direction / self.wheel_circumference
        self.right.controller.input_vel = right_mps * self.right_direction / self.wheel_circumference

    def stop(self):
        self.set_velocity(0.0, 0.0)

    def get_velocity_rpm(self):
        left_rpm = self.left.encoder.vel_estimate * 60 * self.left_direction
        right_rpm = self.right.encoder.vel_estimate * 60 * self.right_direction
        return left_rpm, right_rpm

    def get_position_turns(self):
        left_pos = self.left.encoder.pos_estimate * self.left_direction
        right_pos = self.right.encoder.pos_estimate * self.right_direction
        return left_pos, right_pos

    def get_status(self):
        left_rpm, right_rpm = self.get_velocity_rpm()
        left_pos, right_pos = self.get_position_turns()
        return {
            "voltage": self.odrv.vbus_voltage,
            "left_rpm": left_rpm,
            "right_rpm": right_rpm,
            "left_pos": left_pos,
            "right_pos": right_pos,
        }

    def shutdown(self):
        print("\nStopping ODrive...")
        self.stop()
        time.sleep(0.2)
        self.left.requested_state = AXIS_STATE_IDLE
        self.right.requested_state = AXIS_STATE_IDLE

    def adjust_speed(self, delta):
        self.base_speed = min(self.max_speed, max(0.05, self.base_speed + delta))
        return self.base_speed

    def reset_speed(self):
        self.base_speed = 0.3
        return self.base_speed

    def resolve_command(self, keys):
        forward = keys[pygame.K_w] or keys[pygame.K_UP]
        backward = keys[pygame.K_s] or keys[pygame.K_DOWN]
        left = keys[pygame.K_a] or keys[pygame.K_LEFT]
        right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
        stop = keys[pygame.K_SPACE]

        if stop or (forward and backward):
            return 0.0, 0.0, "stop"

        if forward:
            left_speed = self.base_speed
            right_speed = self.base_speed
            if left and not right:
                left_speed *= 0.5
                return left_speed, right_speed, "forward-left"
            if right and not left:
                right_speed *= 0.5
                return left_speed, right_speed, "forward-right"
            return left_speed, right_speed, "forward"

        if backward:
            left_speed = -self.base_speed
            right_speed = -self.base_speed
            if left and not right:
                left_speed *= 0.5
                return left_speed, right_speed, "backward-left"
            if right and not left:
                right_speed *= 0.5
                return left_speed, right_speed, "backward-right"
            return left_speed, right_speed, "backward"

        if left and not right:
            return -self.turn_speed, self.turn_speed, "turn-left"

        if right and not left:
            return self.turn_speed, -self.turn_speed, "turn-right"

        return 0.0, 0.0, "idle"


def draw_ui(screen, font, tester, action, status):
    screen.fill((18, 20, 24))
    lines = [
        "ODrive Keyboard Test",
        "Focus this window to control the robot",
        "W/S/A/D or arrows: drive   Space: stop   +/-: speed   0: reset   P: print   Esc/Q: quit",
        f"Action: {action}",
        f"Base speed: {tester.base_speed:.2f} m/s   Turn speed: {tester.turn_speed:.2f} m/s",
        f"Left RPM: {status['left_rpm']:7.1f}   Right RPM: {status['right_rpm']:7.1f}   Voltage: {status['voltage']:.1f} V",
        f"Left Pos: {status['left_pos']:7.3f} turns   Right Pos: {status['right_pos']:7.3f} turns",
    ]

    y = 18
    for line in lines:
        text = font.render(line, True, (235, 235, 235))
        screen.blit(text, (20, y))
        y += 28

    pygame.display.flip()


def print_status(status):
    print(
        f"Voltage: {status['voltage']:.1f} V | "
        f"Left RPM: {status['left_rpm']:.1f} | "
        f"Right RPM: {status['right_rpm']:.1f} | "
        f"Left Pos: {status['left_pos']:.3f} turns | "
        f"Right Pos: {status['right_pos']:.3f} turns"
    )


def main():
    tester = ODriveKeyboardTester()

    pygame.init()
    screen = pygame.display.set_mode((980, 220))
    pygame.display.set_caption("ODrive Keyboard Test")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    print()
    print("Controls")
    print("  W/S/A/D or arrow keys: drive")
    print("  Space: stop")
    print("  + / -: adjust base speed")
    print("  0: reset base speed")
    print("  P: print status")
    print("  Esc or Q: quit")

    running = True
    action = "idle"
    last_status_time = 0.0

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
                        speed = tester.adjust_speed(tester.speed_step)
                        print(f"Base speed: {speed:.2f} m/s")
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        speed = tester.adjust_speed(-tester.speed_step)
                        print(f"Base speed: {speed:.2f} m/s")
                    elif event.key in (pygame.K_0, pygame.K_KP0):
                        speed = tester.reset_speed()
                        print(f"Base speed: {speed:.2f} m/s")
                    elif event.key == pygame.K_p:
                        print_status(tester.get_status())

            keys = pygame.key.get_pressed()
            left_speed, right_speed, action = tester.resolve_command(keys)
            tester.set_velocity(left_speed, right_speed)

            status = tester.get_status()
            draw_ui(screen, font, tester, action, status)

            now = time.time()
            if now - last_status_time >= 0.5:
                print(
                    f"\rAction: {action:<14} | "
                    f"Base speed: {tester.base_speed:.2f} m/s | "
                    f"Left RPM: {status['left_rpm']:7.1f} | "
                    f"Right RPM: {status['right_rpm']:7.1f} | "
                    f"Left Pos: {status['left_pos']:7.3f} | "
                    f"Right Pos: {status['right_pos']:7.3f} | "
                    f"Voltage: {status['voltage']:.1f} V",
                    end="",
                    flush=True,
                )
                last_status_time = now

            clock.tick(60)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        tester.shutdown()
        pygame.quit()


if __name__ == "__main__":
    main()
