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
        self.events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
            "next_reward": 0,
        }

        if config.assign_rewards:
            self.events["next_reward"] = 0

        self.pressed_keys = []
        self.font = pygame.font.SysFont('courier', 24)  # Courier is a monospace font
        self.small_font = pygame.font.SysFont('courier', 18)  # Smaller font for controls list
        self.current_episode_index = 0

        # Color theme
        self.text_bg_color = (0, 0, 0)  # Black
        self.text_color = (0, 255, 0)  # Green

        # Define the control instructions
        self.controls = [
            ("Right Arrow", "Exit Early"),
            ("Left Arrow", "Rerecord"),
            ("Escape", "Stop"),
            ("Space", "Toggle Reward"),
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
        total_width = max_width * grid_cols
        total_height = max_height * grid_rows + self.title_height

        return total_width, total_height, grid_cols

    def draw_top_bar(self, window_width: int):
        top_text_str = f"Mode: {self.config.control_phase}"
        if self.config.control_phase == ControlPhase.RECORD:
            top_text_str += f" | Episode: {self.current_episode_index}/{self.config.num_episodes}"
            if self.config.assign_rewards:
                next_reward = self.events["next_reward"]
                top_text_str += f" | Reward: {next_reward}"

        top_text = self.font.render(top_text_str, True, self.text_color)
        text_rect = top_text.get_rect(center=(window_width // 2, self.title_height // 2))
        self.screen.blit(top_text, text_rect)

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

        self.draw_top_bar(window_width)

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


if __name__ == "__main__":
    import torch
    import cv2
    import time
    import numpy as np

    def read_image_from_camera(cap):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.tensor(frame_rgb).float()

    config = ControlContextConfig(
        display_cameras=True,
        assign_rewards=True,
        debug_mode=True,
        control_phase=ControlPhase.RECORD,
        num_episodes=200,
    )
    context = ControlContext(config)
    context.update_current_episode(199)

    # Initialize cameras with proper naming convention
    cameras = {
        "main": cv2.VideoCapture(0),
        "top": cv2.VideoCapture(4),
        "web": cv2.VideoCapture(8)
    }

    # Check if cameras are opened correctly
    for name, cap in cameras.items():
        if not cap.isOpened():
            raise Exception(f"Error: Could not open {name} camera")

    # Main loop
    while True:
        # Capture images from cameras
        images = {}
        camera_logs = {}
        for name, cap in cameras.items():
            before_camread_t = time.perf_counter()
            images[name] = read_image_from_camera(cap)
            camera_logs[f"read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Create state tensor (simulating follower positions)
        state = torch.tensor([10.0195, 128.9355, 173.0566, -13.2715, -7.2070, 34.4531])

        # Construct observation dictionary with proper naming
        obs_dict = {
            "observation.state": state
        }
        
        # Add camera images to observation dictionary
        for name in cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        # Update context with observations
        context.update_with_observations(obs_dict)
        events = context.get_events()

        if events["exit_early"]:
            break

    # Cleanup
    for cap in cameras.values():
        cap.release()
    cv2.destroyAllWindows()