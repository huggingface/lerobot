#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Touch UI for Bimanual SO-101 Robot Control
Optimized for 7-inch landscape touchscreen (1024x600 or 800x480)
"""

import logging
import os
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame

from lerobot.robots.bi_so101_follower import BiSO101Follower, BiSO101FollowerConfig
from lerobot.teleoperators.bi_so101_leader import BiSO101Leader, BiSO101LeaderConfig

logger = logging.getLogger(__name__)

# Display constants for 7-inch landscape (1024x600)
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 600
FPS = 30

# Colors - Modern flat design
BG_COLOR = (30, 30, 35)
CARD_BG = (45, 45, 50)
BTN_PRIMARY = (70, 130, 230)
BTN_PRIMARY_HOVER = (90, 150, 250)
BTN_SUCCESS = (80, 200, 120)
BTN_SUCCESS_HOVER = (100, 220, 140)
BTN_DANGER = (220, 80, 80)
BTN_DANGER_HOVER = (240, 100, 100)
BTN_WARNING = (230, 180, 50)
BTN_WARNING_HOVER = (250, 200, 70)
BTN_DISABLED = (60, 60, 65)
TEXT_COLOR = (250, 250, 250)
TEXT_MUTED = (150, 150, 155)
ACCENT_COLOR = (70, 130, 230)
ERROR_COLOR = (220, 80, 80)
SUCCESS_COLOR = (80, 200, 120)

# UI dimensions
BUTTON_HEIGHT = 80
BUTTON_MARGIN = 15
HEADER_HEIGHT = 70
FONT_SIZE_TITLE = 36
FONT_SIZE_BUTTON = 28
FONT_SIZE_SMALL = 20


class Screen(Enum):
    MAIN_MENU = "main_menu"
    MOTOR_SETUP = "motor_setup"
    CALIBRATION = "calibration"
    TELEOPERATION = "teleoperation"
    RECORDING = "recording"
    STATUS = "status"


@dataclass
class DeviceStatus:
    left_follower: str = "Not Connected"
    right_follower: str = "Not Connected"
    left_leader: str = "Not Connected"
    right_leader: str = "Not Connected"
    error_message: str = ""


class Button:
    def __init__(
        self, x: int, y: int, width: int, height: int, text: str, color: tuple, hover_color: tuple
    ):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.enabled = True

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos) and self.enabled

        color = self.hover_color if is_hover else self.color
        if not self.enabled:
            color = BTN_DISABLED

        pygame.draw.rect(surface, color, self.rect, border_radius=12)

        text_surface = font.render(self.text, True, TEXT_COLOR if self.enabled else TEXT_MUTED)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, pos: tuple) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class BimanualTouchUI:
    def __init__(
        self,
        follower_left_port: str = "/dev/ttyUSB0",
        follower_right_port: str = "/dev/ttyUSB1",
        leader_left_port: str = "/dev/ttyUSB2",
        leader_right_port: str = "/dev/ttyUSB3",
        calibration_dir: str = "~/.cache/lerobot/calibration",
    ):
        self.follower_left_port = follower_left_port
        self.follower_right_port = follower_right_port
        self.leader_left_port = leader_left_port
        self.leader_right_port = leader_right_port
        self.calibration_dir = calibration_dir

        # Device instances
        self.follower: BiSO101Follower | None = None
        self.leader: BiSO101Leader | None = None

        # UI state
        self.current_screen = Screen.MAIN_MENU
        self.status = DeviceStatus()
        self.running = True

        # Teleoperation state
        self.teleop_active = False
        self.recording_active = False
        self.episode_count = 0

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bimanual SO-101 Control")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.Font(None, FONT_SIZE_TITLE)
        self.font_button = pygame.font.Font(None, FONT_SIZE_BUTTON)
        self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)

        # Initialize buttons for each screen
        self._init_buttons()

    def _init_buttons(self):
        """Initialize all buttons for different screens"""
        button_width = (SCREEN_WIDTH - 4 * BUTTON_MARGIN) // 2
        start_y = HEADER_HEIGHT + BUTTON_MARGIN

        # Main menu buttons
        self.main_menu_buttons = {
            "motor_setup": Button(
                BUTTON_MARGIN,
                start_y,
                button_width,
                BUTTON_HEIGHT,
                "Motor Setup",
                BTN_PRIMARY,
                BTN_PRIMARY_HOVER,
            ),
            "calibration": Button(
                2 * BUTTON_MARGIN + button_width,
                start_y,
                button_width,
                BUTTON_HEIGHT,
                "Calibration",
                BTN_PRIMARY,
                BTN_PRIMARY_HOVER,
            ),
            "teleoperation": Button(
                BUTTON_MARGIN,
                start_y + BUTTON_HEIGHT + BUTTON_MARGIN,
                button_width,
                BUTTON_HEIGHT,
                "Teleoperation",
                BTN_SUCCESS,
                BTN_SUCCESS_HOVER,
            ),
            "recording": Button(
                2 * BUTTON_MARGIN + button_width,
                start_y + BUTTON_HEIGHT + BUTTON_MARGIN,
                button_width,
                BUTTON_HEIGHT,
                "Record Dataset",
                BTN_SUCCESS,
                BTN_SUCCESS_HOVER,
            ),
            "status": Button(
                BUTTON_MARGIN,
                start_y + 2 * (BUTTON_HEIGHT + BUTTON_MARGIN),
                button_width,
                BUTTON_HEIGHT,
                "System Status",
                BTN_PRIMARY,
                BTN_PRIMARY_HOVER,
            ),
            "exit": Button(
                2 * BUTTON_MARGIN + button_width,
                start_y + 2 * (BUTTON_HEIGHT + BUTTON_MARGIN),
                button_width,
                BUTTON_HEIGHT,
                "Exit",
                BTN_DANGER,
                BTN_DANGER_HOVER,
            ),
        }

        # Back button (used on all sub-screens)
        self.back_button = Button(
            BUTTON_MARGIN,
            SCREEN_HEIGHT - BUTTON_HEIGHT - BUTTON_MARGIN,
            200,
            60,
            "Back",
            BTN_DANGER,
            BTN_DANGER_HOVER,
        )

    def draw_header(self, title: str):
        """Draw the screen header"""
        pygame.draw.rect(self.screen, CARD_BG, (0, 0, SCREEN_WIDTH, HEADER_HEIGHT))
        title_surface = self.font_title.render(title, True, TEXT_COLOR)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, HEADER_HEIGHT // 2))
        self.screen.blit(title_surface, title_rect)

    def draw_status_bar(self, y_pos: int, text: str, status: str, is_error: bool = False):
        """Draw a status bar with label and status"""
        label = self.font_small.render(text, True, TEXT_COLOR)
        color = ERROR_COLOR if is_error else SUCCESS_COLOR
        status_text = self.font_small.render(status, True, color)

        self.screen.blit(label, (BUTTON_MARGIN + 10, y_pos))
        self.screen.blit(status_text, (SCREEN_WIDTH // 2, y_pos))

    def draw_main_menu(self):
        """Draw the main menu screen"""
        self.draw_header("Bimanual SO-101 Control")

        for button in self.main_menu_buttons.values():
            button.draw(self.screen, self.font_button)

    def draw_motor_setup_screen(self):
        """Draw the motor setup screen"""
        self.draw_header("Motor Setup")

        y_pos = HEADER_HEIGHT + 30
        info_text = [
            "Motor setup configures motor IDs for each arm.",
            "Follow on-screen prompts to connect motors individually.",
            "",
            "Connect motors in this order:",
            "1. Left Follower (6 motors)",
            "2. Right Follower (6 motors)",
            "3. Left Leader (6 motors)",
            "4. Right Leader (6 motors)",
        ]

        for line in info_text:
            text = self.font_small.render(line, True, TEXT_COLOR if line else TEXT_MUTED)
            self.screen.blit(text, (BUTTON_MARGIN + 20, y_pos))
            y_pos += 35

        # Setup buttons
        button_width = (SCREEN_WIDTH - 4 * BUTTON_MARGIN) // 2
        setup_y = y_pos + 20

        setup_left_follower = Button(
            BUTTON_MARGIN,
            setup_y,
            button_width,
            BUTTON_HEIGHT,
            "Setup Left Follower",
            BTN_WARNING,
            BTN_WARNING_HOVER,
        )
        setup_right_follower = Button(
            2 * BUTTON_MARGIN + button_width,
            setup_y,
            button_width,
            BUTTON_HEIGHT,
            "Setup Right Follower",
            BTN_WARNING,
            BTN_WARNING_HOVER,
        )

        setup_left_follower.draw(self.screen, self.font_button)
        setup_right_follower.draw(self.screen, self.font_button)

        self.back_button.draw(self.screen, self.font_button)

        # Handle clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.back_button.is_clicked(event.pos):
                    self.current_screen = Screen.MAIN_MENU
                elif setup_left_follower.is_clicked(event.pos):
                    self._run_motor_setup("left_follower")
                elif setup_right_follower.is_clicked(event.pos):
                    self._run_motor_setup("right_follower")

    def draw_calibration_screen(self):
        """Draw the calibration screen"""
        self.draw_header("Calibration")

        y_pos = HEADER_HEIGHT + 30
        info_text = [
            "Calibration sets the range of motion for each joint.",
            "You will be prompted to move each arm through its full range.",
            "",
            "Make sure all arms are in a safe position before starting.",
        ]

        for line in info_text:
            text = self.font_small.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (BUTTON_MARGIN + 20, y_pos))
            y_pos += 40

        # Calibration buttons
        button_width = (SCREEN_WIDTH - 4 * BUTTON_MARGIN) // 2
        calib_y = y_pos + 20

        calib_followers = Button(
            BUTTON_MARGIN,
            calib_y,
            button_width,
            BUTTON_HEIGHT,
            "Calibrate Followers",
            BTN_SUCCESS,
            BTN_SUCCESS_HOVER,
        )
        calib_leaders = Button(
            2 * BUTTON_MARGIN + button_width,
            calib_y,
            button_width,
            BUTTON_HEIGHT,
            "Calibrate Leaders",
            BTN_SUCCESS,
            BTN_SUCCESS_HOVER,
        )

        calib_followers.draw(self.screen, self.font_button)
        calib_leaders.draw(self.screen, self.font_button)

        self.back_button.draw(self.screen, self.font_button)

        # Handle clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.back_button.is_clicked(event.pos):
                    self.current_screen = Screen.MAIN_MENU
                elif calib_followers.is_clicked(event.pos):
                    self._run_calibration("followers")
                elif calib_leaders.is_clicked(event.pos):
                    self._run_calibration("leaders")

    def draw_teleoperation_screen(self):
        """Draw the teleoperation screen"""
        self.draw_header("Bimanual Teleoperation")

        y_pos = HEADER_HEIGHT + 30

        if not self.teleop_active:
            info_text = [
                "Teleoperation mode: Leaders control Followers",
                "",
                "Left Leader controls Left Follower",
                "Right Leader controls Right Follower",
                "",
                "Press START to begin teleoperation.",
                "Press STOP to end teleoperation session.",
            ]

            for line in info_text:
                text = self.font_small.render(line, True, TEXT_COLOR)
                self.screen.blit(text, (BUTTON_MARGIN + 20, y_pos))
                y_pos += 40

            # Start button
            start_button = Button(
                SCREEN_WIDTH // 2 - 150,
                y_pos + 20,
                300,
                BUTTON_HEIGHT,
                "START",
                BTN_SUCCESS,
                BTN_SUCCESS_HOVER,
            )
            start_button.draw(self.screen, self.font_button)
        else:
            # Active teleoperation display
            status_text = self.font_title.render("TELEOPERATION ACTIVE", True, SUCCESS_COLOR)
            status_rect = status_text.get_rect(center=(SCREEN_WIDTH // 2, y_pos + 50))
            self.screen.blit(status_text, status_rect)

            # Stop button
            stop_button = Button(
                SCREEN_WIDTH // 2 - 150,
                y_pos + 150,
                300,
                BUTTON_HEIGHT,
                "STOP",
                BTN_DANGER,
                BTN_DANGER_HOVER,
            )
            stop_button.draw(self.screen, self.font_button)

        self.back_button.draw(self.screen, self.font_button)

        # Handle clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.back_button.is_clicked(event.pos):
                    if self.teleop_active:
                        self._stop_teleoperation()
                    self.current_screen = Screen.MAIN_MENU
                elif not self.teleop_active and "start_button" in locals():
                    if start_button.is_clicked(event.pos):
                        self._start_teleoperation()
                elif self.teleop_active and "stop_button" in locals():
                    if stop_button.is_clicked(event.pos):
                        self._stop_teleoperation()

    def draw_recording_screen(self):
        """Draw the dataset recording screen"""
        self.draw_header("Dataset Recording")

        y_pos = HEADER_HEIGHT + 30
        info_text = [
            f"Episodes recorded: {self.episode_count}",
            "",
            "Recording combines teleoperation with data collection.",
            "Each episode is saved to the dataset.",
        ]

        for line in info_text:
            text = self.font_small.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (BUTTON_MARGIN + 20, y_pos))
            y_pos += 40

        # Record button
        record_button = Button(
            SCREEN_WIDTH // 2 - 150,
            y_pos + 40,
            300,
            BUTTON_HEIGHT,
            "Start Recording",
            BTN_DANGER,
            BTN_DANGER_HOVER,
        )
        record_button.draw(self.screen, self.font_button)

        self.back_button.draw(self.screen, self.font_button)

        # Handle clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.back_button.is_clicked(event.pos):
                    self.current_screen = Screen.MAIN_MENU
                elif record_button.is_clicked(event.pos):
                    self._start_recording()

    def draw_status_screen(self):
        """Draw the system status screen"""
        self.draw_header("System Status")

        y_pos = HEADER_HEIGHT + 40

        self.draw_status_bar(y_pos, "Left Follower:", self.status.left_follower)
        y_pos += 40
        self.draw_status_bar(y_pos, "Right Follower:", self.status.right_follower)
        y_pos += 40
        self.draw_status_bar(y_pos, "Left Leader:", self.status.left_leader)
        y_pos += 40
        self.draw_status_bar(y_pos, "Right Leader:", self.status.right_leader)
        y_pos += 60

        if self.status.error_message:
            error_label = self.font_small.render("Last Error:", True, ERROR_COLOR)
            self.screen.blit(error_label, (BUTTON_MARGIN + 10, y_pos))
            y_pos += 30

            # Word wrap error message
            words = self.status.error_message.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if self.font_small.size(test_line)[0] < SCREEN_WIDTH - 2 * BUTTON_MARGIN - 20:
                    line = test_line
                else:
                    error_text = self.font_small.render(line, True, TEXT_MUTED)
                    self.screen.blit(error_text, (BUTTON_MARGIN + 20, y_pos))
                    y_pos += 30
                    line = word + " "

            if line:
                error_text = self.font_small.render(line, True, TEXT_MUTED)
                self.screen.blit(error_text, (BUTTON_MARGIN + 20, y_pos))

        # Connect buttons
        button_width = (SCREEN_WIDTH - 4 * BUTTON_MARGIN) // 2
        connect_y = SCREEN_HEIGHT - 2 * BUTTON_HEIGHT - 2 * BUTTON_MARGIN

        connect_followers = Button(
            BUTTON_MARGIN,
            connect_y,
            button_width,
            BUTTON_HEIGHT,
            "Connect Followers",
            BTN_SUCCESS,
            BTN_SUCCESS_HOVER,
        )
        connect_leaders = Button(
            2 * BUTTON_MARGIN + button_width,
            connect_y,
            button_width,
            BUTTON_HEIGHT,
            "Connect Leaders",
            BTN_SUCCESS,
            BTN_SUCCESS_HOVER,
        )

        connect_followers.draw(self.screen, self.font_button)
        connect_leaders.draw(self.screen, self.font_button)

        self.back_button.draw(self.screen, self.font_button)

        # Handle clicks
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.back_button.is_clicked(event.pos):
                    self.current_screen = Screen.MAIN_MENU
                elif connect_followers.is_clicked(event.pos):
                    self._connect_followers()
                elif connect_leaders.is_clicked(event.pos):
                    self._connect_leaders()

    def _connect_followers(self):
        """Connect to follower arms"""
        try:
            config = BiSO101FollowerConfig(
                left_arm_port=self.follower_left_port,
                right_arm_port=self.follower_right_port,
                calibration_dir=self.calibration_dir,
                id="bimanual_follower",
            )
            self.follower = BiSO101Follower(config)
            self.follower.connect(calibrate=False)
            self.status.left_follower = "Connected"
            self.status.right_follower = "Connected"
            self.status.error_message = ""
            logger.info("Followers connected successfully")
        except Exception as e:
            self.status.error_message = f"Failed to connect followers: {str(e)}"
            logger.error(f"Error connecting followers: {e}")
            logger.error(traceback.format_exc())

    def _connect_leaders(self):
        """Connect to leader arms"""
        try:
            config = BiSO101LeaderConfig(
                left_arm_port=self.leader_left_port,
                right_arm_port=self.leader_right_port,
                calibration_dir=self.calibration_dir,
                id="bimanual_leader",
            )
            self.leader = BiSO101Leader(config)
            self.leader.connect(calibrate=False)
            self.status.left_leader = "Connected"
            self.status.right_leader = "Connected"
            self.status.error_message = ""
            logger.info("Leaders connected successfully")
        except Exception as e:
            self.status.error_message = f"Failed to connect leaders: {str(e)}"
            logger.error(f"Error connecting leaders: {e}")
            logger.error(traceback.format_exc())

    def _run_motor_setup(self, device: str):
        """Run motor setup for specified device"""
        self.status.error_message = f"Motor setup for {device} - Follow console prompts"
        logger.info(f"Starting motor setup for {device}")
        # In a real implementation, this would run the setup in a separate thread
        # or show a dialog with instructions

    def _run_calibration(self, device_type: str):
        """Run calibration for specified device type"""
        try:
            if device_type == "followers":
                if self.follower and self.follower.is_connected:
                    self.follower.calibrate()
                    self.status.error_message = "Follower calibration completed"
                else:
                    self.status.error_message = "Connect followers first"
            elif device_type == "leaders":
                if self.leader and self.leader.is_connected:
                    self.leader.calibrate()
                    self.status.error_message = "Leader calibration completed"
                else:
                    self.status.error_message = "Connect leaders first"
        except Exception as e:
            self.status.error_message = f"Calibration error: {str(e)}"
            logger.error(f"Calibration error: {e}")

    def _start_teleoperation(self):
        """Start teleoperation mode"""
        if not self.follower or not self.follower.is_connected:
            self.status.error_message = "Connect followers first"
            return
        if not self.leader or not self.leader.is_connected:
            self.status.error_message = "Connect leaders first"
            return

        self.teleop_active = True
        logger.info("Teleoperation started")

    def _stop_teleoperation(self):
        """Stop teleoperation mode"""
        self.teleop_active = False
        logger.info("Teleoperation stopped")

    def _start_recording(self):
        """Start dataset recording"""
        if not self.teleop_active:
            self.status.error_message = "Start teleoperation first"
            return

        self.recording_active = True
        self.episode_count += 1
        logger.info(f"Started recording episode {self.episode_count}")

    def _teleoperation_loop(self):
        """Run one iteration of the teleoperation loop"""
        if not self.teleop_active or not self.leader or not self.follower:
            return

        try:
            # Get action from leaders
            action = self.leader.get_action()

            # Send action to followers
            self.follower.send_action(action)

        except Exception as e:
            logger.error(f"Teleoperation error: {e}")
            self.status.error_message = f"Teleoperation error: {str(e)}"
            self.teleop_active = False

    def run(self):
        """Main application loop"""
        while self.running:
            self.screen.fill(BG_COLOR)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.current_screen == Screen.MAIN_MENU:
                        for key, button in self.main_menu_buttons.items():
                            if button.is_clicked(event.pos):
                                if key == "exit":
                                    self.running = False
                                elif key == "motor_setup":
                                    self.current_screen = Screen.MOTOR_SETUP
                                elif key == "calibration":
                                    self.current_screen = Screen.CALIBRATION
                                elif key == "teleoperation":
                                    self.current_screen = Screen.TELEOPERATION
                                elif key == "recording":
                                    self.current_screen = Screen.RECORDING
                                elif key == "status":
                                    self.current_screen = Screen.STATUS

            # Draw current screen
            if self.current_screen == Screen.MAIN_MENU:
                self.draw_main_menu()
            elif self.current_screen == Screen.MOTOR_SETUP:
                self.draw_motor_setup_screen()
            elif self.current_screen == Screen.CALIBRATION:
                self.draw_calibration_screen()
            elif self.current_screen == Screen.TELEOPERATION:
                self.draw_teleoperation_screen()
            elif self.current_screen == Screen.RECORDING:
                self.draw_recording_screen()
            elif self.current_screen == Screen.STATUS:
                self.draw_status_screen()

            # Run teleoperation if active
            if self.teleop_active:
                self._teleoperation_loop()

            pygame.display.flip()
            self.clock.tick(FPS)

        # Cleanup
        self._cleanup()
        pygame.quit()

    def _cleanup(self):
        """Cleanup resources"""
        try:
            if self.follower and self.follower.is_connected:
                self.follower.disconnect()
            if self.leader and self.leader.is_connected:
                self.leader.disconnect()
            logger.info("Resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
