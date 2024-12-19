import pygame
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


# Create an enum for ControlPhase
class ControlPhase:
    TELEOPERATE = "Teleoperate"
    WARMUP = "Warmup"
    RECORD = "Record"
    RESET = "Reset"


@dataclass
class ControlContextConfig:
    display_cameras: bool = False
    play_sounds: bool = False
    assign_rewards: bool = False
    debug_mode: bool = False
    control_phase: ControlPhase = ControlPhase.TELEOPERATE
    num_episodes: int = 0
    # TODO(jackvial): Add robot on this class so we can call robot.disconnect() in cleanup


class ControlContext:
    def __init__(self, config: Optional[ControlContextConfig] = None):
        self.config = config or ControlContextConfig()
        pygame.init()
        if not self.config.display_cameras:
            pygame.display.set_mode((1, 1), pygame.HIDDEN)

        self.screen = None
        self.image_positions = {}
        self.padding = 20
        self.title_height = 30
        self.controls_width = 300  # Width of the controls panel
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "next_reward": 0,
        }

        if config.assign_rewards:
            self.events["next_reward"] = 0

        self.pressed_keys = []
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)  # Smaller font for controls list
        self.current_episode_index = 0

        # Color theme
        self.text_bg_color = (0, 0, 0)  # Black
        self.text_color = (0, 255, 0)  # Green

        # Define the control instructions
        self.controls = [
            ("Right Arrow", "Exit current loop"),
            ("Left Arrow", "Rerecord last episode"),
            ("Escape", "Stop recording"),
            ("Space", "Toggle reward (if enabled)"),
        ]

    def calculate_window_size(self, images: Dict[str, np.ndarray]):
        """Calculate required window size based on images"""
        max_width = 0
        max_height = 0
        n_images = len(images)

        # Calculate grid dimensions
        grid_cols = min(2, n_images)
        grid_rows = (n_images + 1) // 2

        for image in images.values():
            max_width = max(max_width, image.shape[1])
            max_height = max(max_height, image.shape[0])

        # Adjust total width and height calculations to remove extra padding
        total_width = max_width * grid_cols + self.controls_width
        total_height = max_height * grid_rows + self.title_height

        return total_width, total_height, grid_cols

    def render_controls_panel(self, window_width: int, window_height: int):
        """Render the controls panel on the right side"""
        # Draw controls background
        controls_rect = pygame.Rect(
            window_width - self.controls_width,
            self.title_height,
            self.controls_width,
            window_height - self.title_height,
        )
        pygame.draw.rect(self.screen, self.text_bg_color, controls_rect)
        pygame.draw.line(
            self.screen,
            self.text_color,
            (controls_rect.left, self.title_height),
            (controls_rect.left, window_height),
            2,
        )

        # Draw "Controls" header
        header = self.font.render("Controls", True, self.text_color)
        header_rect = header.get_rect(
            centerx=window_width - self.controls_width / 2, top=self.title_height + 10
        )
        self.screen.blit(header, header_rect)

        # Draw control instructions
        y_pos = header_rect.bottom + 20
        for key, action in self.controls:
            # Draw key
            key_surface = self.small_font.render(key, True, self.text_color)
            key_rect = key_surface.get_rect(left=window_width - self.controls_width + 20, top=y_pos)
            self.screen.blit(key_surface, key_rect)

            # Draw action
            action_surface = self.small_font.render(action, True, self.text_color)
            action_rect = action_surface.get_rect(left=key_rect.right + 10, top=y_pos)
            self.screen.blit(action_surface, action_rect)

            y_pos += 30

        # Add status information below controls
        y_pos += 20  # Add some spacing

        # TODO(jackvial): Move control phase to the top bar
        # Control phase
        phase_text = f"Control Phase: {self.config.control_phase}"
        phase_surface = self.small_font.render(phase_text, True, self.text_color)
        phase_rect = phase_surface.get_rect(left=window_width - self.controls_width + 20, top=y_pos)
        self.screen.blit(phase_surface, phase_rect)

        # Pressed keys
        y_pos += 30
        keys_text = f"Pressed: {', '.join(self.pressed_keys)}" if self.pressed_keys else "Pressed: None"
        keys_surface = self.small_font.render(keys_text, True, self.text_color)
        keys_rect = keys_surface.get_rect(left=window_width - self.controls_width + 20, top=y_pos)
        self.screen.blit(keys_surface, keys_rect)

    def handle_events(self):
        """Handle pygame events and update internal state"""
        for event in pygame.event.get():
            if self.config.debug_mode:
                print(event)
            if event.type == pygame.QUIT:
                self.events["stop_recording"] = True
                self.events["exit_early"] = True
            elif event.type == pygame.KEYDOWN:
                key_name = pygame.key.name(event.key)
                self.pressed_keys.append(key_name)

                if event.key == pygame.K_RIGHT:
                    print("Right arrow key pressed. Exiting loop...")
                    self.events["exit_early"] = True
                elif event.key == pygame.K_LEFT:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    self.events["rerecord_episode"] = True
                    self.events["exit_early"] = True
                elif event.key == pygame.K_ESCAPE:
                    print("Escape key pressed. Stopping data recording...")
                    self.events["stop_recording"] = True
                    self.events["exit_early"] = True
                elif self.config.assign_rewards and event.key == pygame.K_SPACE:
                    self.events["next_reward"] = 1 if self.events["next_reward"] == 0 else 0
                    print(f"Space key pressed. New reward: {self.events['next_reward']}")

            elif event.type == pygame.KEYUP:
                key_name = pygame.key.name(event.key)
                if key_name in self.pressed_keys:
                    self.pressed_keys.remove(key_name)

        return self.events

    def render_camera_frames(self, observation: Dict[str, np.ndarray]):
        """Update display with new images from observation dict"""
        image_keys = [key for key in observation if "image" in key]
        images = {k: observation[k].numpy() for k in image_keys}
        if not images:
            return

        # Initialize or resize window if needed
        window_width, window_height, grid_cols = self.calculate_window_size(images)
        if self.screen is None or self.screen.get_size() != (window_width, window_height):
            self.screen = pygame.display.set_mode((window_width, window_height))

            # @TODO - label this window with the camera name
            pygame.display.set_caption("Camera 0")

        self.screen.fill(self.text_bg_color)

        # Update image positions and draw images
        for idx, (key, image) in enumerate(images.items()):
            # Calculate grid position
            col = idx % grid_cols
            row = idx // grid_cols

            # Calculate pixel position - adjust for controls panel
            x = col * (image.shape[1] + self.padding)
            y = row * (image.shape[0] + self.title_height + self.padding)

            # Convert numpy array to pygame surface
            image_surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
            self.screen.blit(image_surface, (x, y + self.title_height))

        pygame.draw.rect(self.screen, self.text_bg_color, (0, 0, window_width, self.title_height))

        # Prepare top bar text
        top_text_str = ""
        if self.config.control_phase == ControlPhase.RECORD:
            top_text_str = f"Episode: {self.current_episode_index}/{self.config.num_episodes}"
            if self.config.assign_rewards:
                next_reward = self.events["next_reward"]
                top_text_str += f" | Reward: {next_reward}"

        top_text = self.font.render(top_text_str, True, self.text_color)
        text_rect = top_text.get_rect(center=(window_width // 2, self.title_height // 2))
        self.screen.blit(top_text, text_rect)

        # Draw controls panel
        self.render_controls_panel(window_width, window_height)

        # TODO(jackvial): Would be nice to show count down timer for warmup phase and reset phase

        pygame.display.flip()

    def update_with_observations(self, observation: Dict[str, np.ndarray]):
        if self.config.display_cameras:
            self.render_camera_frames(observation)
        self.handle_events()
        return self

    def update_current_episode(self, episode_index):
        self.current_episode_index = episode_index
        return self

    def get_events(self):
        """Return current events state"""
        return self.events.copy()
    
    def cleanup(self, robot):
        robot.disconnect()
        pygame.quit()
